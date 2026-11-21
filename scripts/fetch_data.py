#!/usr/bin/env python
"""Fetch the pinned third-party datasets into `external/` and verify them.

Two Hugging Face dataset repos are mirrored, each under `<dest>/<repo-name>/` with its own
relative paths (so `router/data/coderouterbench.py` finds `external/CodeRouterBench/...`):

    Lance1573/CodeRouterBench      -> <dest>/CodeRouterBench/
    Lance1573/route-anything-data  -> <dest>/route-anything-data/

Both are pinned to full commit SHAs (REPOS below). `external/CHECKSUMS` (committed) holds the
SHA-256 of every file in `sha256sum` format, so `cd external && shasum -a 256 -c CHECKSUMS`
works without this script.

Source benchmarks (task text for `router/data/prompts.py`) are pinned differently, as extracts
(EXTRACTS below). Each names a repo, a full commit SHA, a per-file allowlist with the Hub's own
hash of each file, and a field allowlist. The upstream files are streamed, never stored: their
bytes are checked against the pinned hash, and only the allowlisted fields of each row are
written to `<dest>/sources/<name>.jsonl`, whose SHA-256 goes into CHECKSUMS like any other
file. Solution, test and expected-output columns therefore never reach the disk, and
LiveCodeBench (4.5 GB, mostly private tests) costs a few MB.

Normal run: read CHECKSUMS, hash the local files, download only the missing or mismatched
ones (rebuilding a missing or mismatched extract from its upstream files), re-hash, exit
non-zero on any mismatch, print `fetched: N, verified: M`. When every file already matches, no
network call is made (works under HF_HUB_OFFLINE=1).

`--write-checksums` is the only mode that creates or overwrites CHECKSUMS. It takes the file
list from the Hub API at the pinned revision and checks every file against the Hub's own hash
(LFS sha256, or the git blob SHA-1 for non-LFS files) before writing anything. For extracts it
checks that the pinned per-file hashes are the Hub's at the pinned revision, then builds them.

Usage:
    pipeline/venv/bin/python scripts/fetch_data.py
    pipeline/venv/bin/python scripts/fetch_data.py --dest /path/to/dir
    pipeline/venv/bin/python scripts/fetch_data.py --verify-only
    pipeline/venv/bin/python scripts/fetch_data.py --write-checksums
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from router.settings import EXTERNAL_DIR  # noqa: E402

# The single pin: local mirror name -> (Hugging Face dataset repo id, full commit SHA).
REPOS: dict[str, tuple[str, str]] = {
    "CodeRouterBench": ("Lance1573/CodeRouterBench", "e567d89bdd569c9c74ffc7c7118e50d15e46b886"),
    "route-anything-data": ("Lance1573/route-anything-data", "bd361209720f27a2383857e314c554476bc81c5c"),
}


@dataclass(frozen=True)
class Extract:
    """Allowlisted fields of pinned upstream files, written as one JSON object per upstream row.

    `files` maps each upstream path to the Hub's own hash of it at `revision`: ("sha256", LFS
    hash) or ("sha1", git blob id). `fields` is the only data read from a row; "a.b" reads the
    nested key b of a. Rows keep upstream order (files in the order listed).
    """

    repo_id: str
    revision: str
    files: dict[str, tuple[str, str]]
    fields: tuple[str, ...]
    # Optional row filter (S1.7 audit): keep only rows whose `keep[0]` field is in `keep[1]`. Every
    # upstream byte is still hashed against the pin.
    keep: tuple[str, tuple[str, ...]] | None = None

    def spec(self) -> str:
        s = f"{self.repo_id}@{self.revision} files={','.join(self.files)} fields={','.join(self.fields)}"
        if self.keep:
            ids = "\n".join(sorted(self.keep[1])).encode()
            s += f" keep={self.keep[0]}:{len(self.keep[1])}:{hashlib.sha256(ids).hexdigest()[:16]}"
        return s


_LCB_FILES = {
    "test.jsonl": ("sha256", "2bd02b38beb48e8c46b5b9987095d999ff38cd8efc255ea5d58974317c48f63f"),
    "test2.jsonl": ("sha256", "095df7c5daf15f882c51a9deb84085cff1e073495a5dbcf95015a564d485f3a3"),
    "test3.jsonl": ("sha256", "28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79"),
    "test4.jsonl": ("sha256", "d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da"),
    "test5.jsonl": ("sha256", "7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214"),
    "test6.jsonl": ("sha256", "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"),
}

# Source-benchmark task text: extract name -> pin. Field allowlists hold task text and ids only.
EXTRACTS: dict[str, Extract] = {
    # DS-1000 at the revision live at the Route Anything snapshot (September 2026). The later
    # a93b81f (2026-11-19) leaves every prompt unchanged.
    "ds1000": Extract(
        "xlangai/DS-1000", "4416080ac5cb80bdf7576aefb8f9a0b4d5426a44",
        {"test.jsonl": ("sha1", "3924161ac7b056443bb7ba724b15e7e128d1108b")},
        ("metadata.problem_id", "prompt"),
    ),
    # The same DS-1000 file, metadata only (S1.6): which problem each perturbed problem was made
    # from. `perturbation_origin_id` counts within a library (it matches `library_problem_id`, not
    # the global `problem_id`), so the library fields are kept with it.
    "ds1000_origin": Extract(
        "xlangai/DS-1000", "4416080ac5cb80bdf7576aefb8f9a0b4d5426a44",
        {"test.jsonl": ("sha1", "3924161ac7b056443bb7ba724b15e7e128d1108b")},
        ("metadata.problem_id", "metadata.library", "metadata.library_problem_id",
         "metadata.perturbation_type", "metadata.perturbation_origin_id"),
    ),
    # BigCodeBench v0.1.4, the release `tasks.jsonl` names.
    "bigcodebench": Extract(
        "bigcode/bigcodebench", "b74c0d0bf70d2c0bc459be537895cca163007f1a",
        {"data/v0.1.4-00000-of-00001.parquet": ("sha256", "d9a4965821c9507ebdfb551c288656b2d5fe553234f5183044333ca8a4018267")},
        ("task_id", "complete_prompt"),
    ),
    # S1.7 audit (checkpoint 2a): BigCodeBench v0.1.4 tests, from the same pinned file. Test code
    # is data here; it is executed only inside the audit sandbox (router/audit_sandbox/). The
    # canonical solution is kept as a harness control: it must pass in the sandbox.
    "bigcodebench_tests": Extract(
        "bigcode/bigcodebench", "b74c0d0bf70d2c0bc459be537895cca163007f1a",
        {"data/v0.1.4-00000-of-00001.parquet": ("sha256", "d9a4965821c9507ebdfb551c288656b2d5fe553234f5183044333ca8a4018267")},
        ("task_id", "code_prompt", "test", "entry_point", "libs", "canonical_solution"),
    ),
    # S1.7 audit (checkpoint 2a), the other Python families. Reference solutions are harness
    # controls only.
    "ds1000_tests": Extract(
        "xlangai/DS-1000", "4416080ac5cb80bdf7576aefb8f9a0b4d5426a44",
        {"test.jsonl": ("sha1", "3924161ac7b056443bb7ba724b15e7e128d1108b")},
        ("metadata.problem_id", "metadata.library", "code_context", "reference_code"),
    ),
    "humanevalplus_tests": Extract(
        "evalplus/humanevalplus", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
        {"data/test-00000-of-00001-5973903632b82d40.parquet": ("sha256", "4436f5c03d77c17e0cbc57543b90665b5c1266f55a43992a5ed7922cd34a7558")},
        ("task_id", "prompt", "entry_point", "test", "canonical_solution"),
    ),
    "humaneval_x_python_tests": Extract(
        "THUDM/humaneval-x", "62c78627f3072a1454fa0cb0184737cafe5e4198",
        {"data/python/data/humaneval.jsonl": ("sha1", "53abae0f4d1340579315a2832f6e721184a41584")},
        ("task_id", "prompt", "declaration", "test", "canonical_solution"),
    ),
    "mbppplus_tests": Extract(
        "evalplus/mbppplus", "b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
        {"data/test-00000-of-00001-d5781c9c51e02795.parquet": ("sha256", "dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b")},
        ("task_id", "prompt", "code", "test_imports", "test_list", "test"),
    ),
    "cruxeval_tests": Extract(
        "cruxeval-org/cruxeval", "b96af0450242eb4da433032b90998f25588a5d0f",
        {"test.jsonl": ("sha1", "4d16f303dce8a83e9805b96f531acee38853c443")},
        ("id", "code", "input", "output"),
    ),
    # LiveCodeBench tests for the sampled lcb tasks only (streams 100 and 122; ids drawn from
    # presence data before any response was read). Private tests are kept encoded as published;
    # they are decoded only inside the sandbox.
    "livecodebench_tests": Extract(
        "livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        _LCB_FILES,
        ("question_id", "platform", "starter_code", "public_test_cases", "private_test_cases",
         "metadata"),
        keep=("question_id", tuple(
            (REPO_ROOT / "router/audit_sandbox/lcb_sample_question_ids.txt").read_text().split())),
    ),
    # S1.7 checkpoint 2b: HumanEval-X and MultiPL-E tests (test code as data, executed only in
    # the audit sandbox). Go keeps its own `import` and `test_setup` fields (CodeGeeX assembly).
    "humaneval_x_tests": Extract(
        "THUDM/humaneval-x", "62c78627f3072a1454fa0cb0184737cafe5e4198",
        {
            "data/cpp/data/humaneval.jsonl": ("sha1", "4c4ce46302c3dcc1e40e517f14b3407fce16053d"),
            "data/java/data/humaneval.jsonl": ("sha1", "4986ea5ba1de8693abbf9627a8376a53d88da55d"),
            "data/js/data/humaneval.jsonl": ("sha1", "b8e32b85b98240656dd362681b100ba8687ad21c"),
        },
        ("task_id", "prompt", "declaration", "canonical_solution", "test"),
    ),
    "humaneval_x_go_tests": Extract(
        "THUDM/humaneval-x", "62c78627f3072a1454fa0cb0184737cafe5e4198",
        {"data/go/data/humaneval.jsonl": ("sha1", "45cc3c3a67bf598fc5bdada98943b34aff2be1f3")},
        ("task_id", "prompt", "import", "declaration", "canonical_solution", "test", "test_setup"),
    ),
    "multipl_e_tests": Extract(
        "nuprl/MultiPL-E", "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2",
        {
            "humaneval-rs/test-00000-of-00001.parquet": ("sha256", "92dba15e4da4deb4dbae5e15ce1712c95ec5a63c3ead69886271470309d9d314"),
            "humaneval-ts/test-00000-of-00001.parquet": ("sha256", "b9090cc4e5a0c06e671811f63d7c35737f8393f1b9d44685472d5a7b1877d136"),
        },
        ("language", "name", "prompt", "tests", "stop_tokens"),
    ),
    # S1.7 Amendment 7: tests of the stream-140 `lcb` kept-subset tasks not already in
    # livecodebench_tests (ids recorded from the draw, which uses no test field).
    "livecodebench_tests_140": Extract(
        "livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        _LCB_FILES,
        ("question_id", "platform", "starter_code", "public_test_cases", "private_test_cases",
         "metadata"),
        keep=("question_id", tuple(
            (REPO_ROOT / "router/audit_sandbox/lcb_kept140_question_ids.txt").read_text().split())),
    ),
    # S1.7 checkpoint 2c: SWE-bench Lite test fields (patch = the gold patch, used only as the
    # harness control; test code is executed only in the audit sandbox).
    "swebench_lite_tests": Extract(
        "princeton-nlp/SWE-bench_Lite", "6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2",
        {"data/test-00000-of-00001.parquet": ("sha256", "7a21f37b8bc179c7db5beeb14e88ac538ba283455c776e6b2535bbfb6e3551b4")},
        ("instance_id", "repo", "base_commit", "version", "environment_setup_commit", "FAIL_TO_PASS",
         "PASS_TO_PASS", "test_patch", "patch"),
    ),
    # HumanEval-X in four languages (Python is fully covered by tasks.jsonl).
    "humaneval_x": Extract(
        "THUDM/humaneval-x", "62c78627f3072a1454fa0cb0184737cafe5e4198",
        {
            "data/cpp/data/humaneval.jsonl": ("sha1", "4c4ce46302c3dcc1e40e517f14b3407fce16053d"),
            "data/go/data/humaneval.jsonl": ("sha1", "45cc3c3a67bf598fc5bdada98943b34aff2be1f3"),
            "data/java/data/humaneval.jsonl": ("sha1", "4986ea5ba1de8693abbf9627a8376a53d88da55d"),
            "data/js/data/humaneval.jsonl": ("sha1", "b8e32b85b98240656dd362681b100ba8687ad21c"),
        },
        ("task_id", "prompt"),
    ),
    # HumanEval+ prompts: the he_completion and he_testgen raw text (HumanEval/115 differs from
    # HumanEval-X and matches this source).
    "humanevalplus": Extract(
        "evalplus/humanevalplus", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
        {"data/test-00000-of-00001-5973903632b82d40.parquet": ("sha256", "4436f5c03d77c17e0cbc57543b90665b5c1266f55a43992a5ed7922cd34a7558")},
        ("task_id", "prompt"),
    ),
    # MultiPL-E HumanEval in Rust and TypeScript.
    "multipl_e": Extract(
        "nuprl/MultiPL-E", "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2",
        {
            "humaneval-rs/test-00000-of-00001.parquet": ("sha256", "92dba15e4da4deb4dbae5e15ce1712c95ec5a63c3ead69886271470309d9d314"),
            "humaneval-ts/test-00000-of-00001.parquet": ("sha256", "b9090cc4e5a0c06e671811f63d7c35737f8393f1b9d44685472d5a7b1877d136"),
        },
        ("language", "name", "prompt"),
    ),
    # LiveCodeBench code_generation_lite, release files v1-v6.
    "livecodebench": Extract(
        "livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        _LCB_FILES,
        ("platform", "question_content", "question_id", "starter_code"),
    ),
}

# Extracts that deliberately hold test, solution or expected-output fields, for the S1.7 artefact
# audit only. They are executed only inside router/audit_sandbox/ and never read by
# router/data/prompts.py. Every other extract keeps the text-only rule.
AUDIT_EXTRACTS: frozenset[str] = frozenset({"bigcodebench_tests", "ds1000_tests", "humanevalplus_tests",
                                          "humaneval_x_python_tests", "mbppplus_tests", "cruxeval_tests",
                                          "livecodebench_tests", "humaneval_x_tests", "humaneval_x_go_tests",
                                          "multipl_e_tests", "livecodebench_tests_140",
                                          "swebench_lite_tests"})

CHECKSUMS_NAME = "CHECKSUMS"
COMMITTED_CHECKSUMS = REPO_ROOT / "external" / CHECKSUMS_NAME
PIN_PREFIX = "# pin: "
EXTRACT_PREFIX = "# extract: "
EXTRACT_DIR = "sources"
WORKERS = 8
CHUNK = 1 << 20


def extract_path(name: str) -> str:
    return f"{EXTRACT_DIR}/{name}.jsonl"


def is_excluded(relpath: str) -> bool:
    """Files that never go into CHECKSUMS: the Hub client's cache dir and .gitattributes."""
    parts = relpath.split("/")
    return ".cache" in parts or parts[-1] == ".gitattributes"


def git_blob_sha1(path: Path) -> str:
    """Git blob object id: sha1(b"blob <size>\\0" + content)."""
    h = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_checksums(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Parse a sha256sum-style file into ({relpath: hex}, {repo_id: sha} from `# pin:` lines)."""
    entries: dict[str, str] = {}
    pins: dict[str, str] = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if line.startswith(PIN_PREFIX):
            repo_id, _, sha = line[len(PIN_PREFIX):].strip().partition("@")
            pins[repo_id] = sha
            continue
        if line.startswith("#"):
            continue
        digest, sep, path = line.partition("  ")
        if not sep or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or not path:
            raise ValueError(f"CHECKSUMS line {n} is not '<sha256>  <path>': {line!r}")
        if path in entries:
            raise ValueError(f"CHECKSUMS line {n} repeats {path}")
        entries[path] = digest
    return entries, pins


def format_checksums(entries: dict[str, str]) -> str:
    header = [
        "# Pinned third-party data for PROBE-CASCADE. Written by scripts/fetch_data.py --write-checksums.",
        "# Verify: cd external && shasum -a 256 -c CHECKSUMS",
    ]
    header += [f"{PIN_PREFIX}{repo_id}@{sha}" for repo_id, sha in REPOS.values()]
    header += [f"{EXTRACT_PREFIX}{name} <- {ex.spec()}" for name, ex in EXTRACTS.items()]
    body = [f"{entries[p]}  {p}" for p in sorted(entries)]
    return "\n".join(header + body) + "\n"


def parse_extract_pins(text: str) -> dict[str, str]:
    """{extract name: spec} from the `# extract:` lines of a CHECKSUMS file."""
    pins: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith(EXTRACT_PREFIX):
            name, sep, spec = line[len(EXTRACT_PREFIX):].strip().partition(" <- ")
            if not sep:
                raise ValueError(f"malformed extract pin: {line!r}")
            pins[name] = spec
    return pins


def extract_name(path: str) -> str | None:
    """'sources/ds1000.jsonl' -> 'ds1000' when it is a pinned extract, else None."""
    head, _, rel = path.partition("/")
    if head == EXTRACT_DIR and rel.endswith(".jsonl") and rel[: -len(".jsonl")] in EXTRACTS:
        return rel[: -len(".jsonl")]
    return None


def split_entry(path: str) -> tuple[str, str]:
    """'CodeRouterBench/a/b.csv' -> ('CodeRouterBench', 'a/b.csv')."""
    repo_name, _, rel = path.partition("/")
    if repo_name not in REPOS or not rel:
        raise ValueError(f"CHECKSUMS path {path!r} is not under one of {sorted(REPOS)}")
    return repo_name, rel


# --- extracts ---------------------------------------------------------------------------------

Opener = Callable[[str, str, str], Iterable[bytes]]


def open_hub(repo_id: str, path: str, revision: str) -> Iterator[bytes]:
    """Stream one dataset file from the Hub at a pinned revision."""
    from huggingface_hub import get_session, hf_hub_url

    url = hf_hub_url(repo_id, path, repo_type="dataset", revision=revision)
    with get_session().stream("GET", url, timeout=120, follow_redirects=True) as resp:
        resp.raise_for_status()
        yield from resp.iter_bytes(CHUNK)


def pick(row: dict, field: str):
    """The allowlisted field of a row; "a.b" reads row["a"]["b"]. A missing field is an error."""
    value = row
    for key in field.split("."):
        value = value[key]
    return value


def _native(value):
    """Plain JSON types, so the extract bytes do not depend on the reader library."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


def _rows_jsonl(chunks: Iterable[bytes]) -> Iterator[dict]:
    buf = b""
    for chunk in chunks:
        buf += chunk
        *lines, buf = buf.split(b"\n")
        for line in lines:
            if line.strip():
                yield json.loads(line)
    if buf.strip():
        yield json.loads(buf)


class _Hashing:
    """Wrap a byte stream; after it is consumed, `hexdigest()` is the Hub-style hash."""

    def __init__(self, chunks: Iterable[bytes], kind: str):
        self.chunks, self.kind = chunks, kind
        self.h = hashlib.sha256() if kind == "sha256" else None
        self.buf: list[bytes] = []

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self.chunks:
            if self.h is not None:
                self.h.update(chunk)
            else:
                self.buf.append(chunk)  # git blob ids need the size first; these files are small
            yield chunk

    def hexdigest(self) -> str:
        if self.h is not None:
            return self.h.hexdigest()
        data = b"".join(self.buf)
        return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def extract_rows(ex: Extract, opener: Opener = open_hub) -> Iterator[dict]:
    """Allowlisted fields of every upstream row, after checking each file's pinned hash.

    Rows of a file are yielded as they stream, and the hash is checked at the end of the file,
    so a caller must not keep any output until the whole generator has finished.
    """
    top = sorted({f.split(".")[0] for f in ex.fields})
    for path, (kind, expected) in ex.files.items():
        stream = _Hashing(opener(ex.repo_id, path, ex.revision), kind)
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq

            data = b"".join(stream)
            rows: Iterable[dict] = pq.read_table(io.BytesIO(data), columns=top).to_pylist()
        elif path.endswith(".jsonl"):
            rows = _rows_jsonl(stream)
        else:
            raise ValueError(f"{ex.repo_id}/{path}: no reader for this file type")
        keep = set(ex.keep[1]) if ex.keep else None
        for row in rows:
            if keep is not None and str(pick(row, ex.keep[0])) not in keep:
                continue
            yield {f: _native(pick(row, f)) for f in ex.fields}
        got = stream.hexdigest()
        if got != expected:
            raise ValueError(f"{ex.repo_id}@{ex.revision}/{path}: {kind} {got}, pinned {expected}")


def build_extract(dest: Path, name: str, opener: Opener = open_hub) -> None:
    """Write `<dest>/sources/<name>.jsonl`; nothing is left behind if any upstream check fails."""
    out = dest / extract_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for rec in extract_rows(EXTRACTS[name], opener):
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)


def download(dest: Path, repo_name: str, rel: str) -> None:
    from huggingface_hub import hf_hub_download

    repo_id, sha = REPOS[repo_name]
    local = dest / repo_name / rel
    if local.exists():
        local.unlink()  # a stale file with a matching Hub cache record would be skipped
    hf_hub_download(repo_id, rel, repo_type="dataset", revision=sha, local_dir=dest / repo_name)


def hash_all(dest: Path, wanted: dict[str, str]) -> dict[str, str | None]:
    """sha256 of each wanted file under dest, or None when missing."""
    def one(path: str) -> str | None:
        f = dest / path
        return sha256_file(f) if f.is_file() else None

    with ThreadPoolExecutor(WORKERS) as ex:
        return dict(zip(wanted, ex.map(one, wanted)))


def run_fetch(dest: Path, verify_only: bool) -> int:
    checksums = dest / CHECKSUMS_NAME if (dest / CHECKSUMS_NAME).is_file() else COMMITTED_CHECKSUMS
    if not checksums.is_file():
        print(f"error: no CHECKSUMS at {dest / CHECKSUMS_NAME} or {COMMITTED_CHECKSUMS}", file=sys.stderr)
        return 2
    text = checksums.read_text()
    wanted, pins = parse_checksums(text)
    for repo_id, sha in REPOS.values():
        if pins.get(repo_id) != sha:
            print(f"error: {checksums} pins {repo_id}@{pins.get(repo_id)}, script pins {sha}", file=sys.stderr)
            return 2
    extract_pins = parse_extract_pins(text)
    for name, ex in EXTRACTS.items():
        if extract_pins.get(name) != ex.spec():
            print(f"error: {checksums} pins extract {name} as {extract_pins.get(name)}, script as {ex.spec()}",
                  file=sys.stderr)
            return 2
    for path in wanted:
        if extract_name(path) is None:
            split_entry(path)

    local = hash_all(dest, wanted)
    todo = [p for p, h in local.items() if h != wanted[p]]
    fetched = 0
    if todo and not verify_only:
        print(f"{len(todo)} of {len(wanted)} files missing or mismatched; downloading")
        mirror = [p for p in todo if extract_name(p) is None]
        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(lambda p: download(dest, *split_entry(p)), mirror))
        for p in todo:
            if (name := extract_name(p)) is not None:
                build_extract(dest, name)
        fetched = len(todo)
        local.update(hash_all(dest, {p: wanted[p] for p in todo}))

    bad = [p for p, h in local.items() if h != wanted[p]]
    for p in bad:
        state = "MISSING" if local[p] is None else "MISMATCH"
        print(f"{state}: {p}", file=sys.stderr)
    print(f"fetched: {fetched}, verified: {len(wanted) - len(bad)}")
    return 1 if bad else 0


def hub_expected(sibling) -> tuple[str, str]:
    """('sha256'|'sha1', hex) the Hub reports for a file: LFS sha256, else git blob SHA-1."""
    if sibling.lfs is not None:
        return "sha256", sibling.lfs["sha256"]
    return "sha1", sibling.blob_id


def run_write_checksums(dest: Path) -> int:
    from huggingface_hub import HfApi

    api = HfApi()
    files: list[tuple[str, str, str, str]] = []  # (path, repo_name, kind, expected)
    for repo_name, (repo_id, sha) in REPOS.items():
        info = api.dataset_info(repo_id, revision=sha, files_metadata=True)
        if info.sha != sha:
            print(f"error: {repo_id}@{sha} resolved to {info.sha}", file=sys.stderr)
            return 2
        for s in info.siblings:
            if is_excluded(s.rfilename):
                continue
            kind, expected = hub_expected(s)
            files.append((f"{repo_name}/{s.rfilename}", repo_name, kind, expected))
    print(f"Hub lists {len(files)} files across {len(REPOS)} repos")

    def check(item) -> tuple[str, str | None, str | None]:
        path, repo_name, kind, expected = item
        f = dest / path
        for attempt in (0, 1):
            if not f.is_file() or attempt == 1:
                download(dest, repo_name, path.partition("/")[2])
            sha256 = sha256_file(f)
            actual = sha256 if kind == "sha256" else git_blob_sha1(f)
            if actual == expected:
                return path, sha256, None
        return path, None, f"{path}: Hub {kind} {expected}, local {actual}"

    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(check, files))
    errors = [e for _, _, e in results if e]
    if errors:
        for e in errors:
            print(f"MISMATCH vs Hub: {e}", file=sys.stderr)
        print("CHECKSUMS not written", file=sys.stderr)
        return 1
    entries = {p: h for p, h, _ in results}

    for name, ex in EXTRACTS.items():
        info = api.dataset_info(ex.repo_id, revision=ex.revision, files_metadata=True)
        hub = {s.rfilename: hub_expected(s) for s in info.siblings}
        if info.sha != ex.revision:
            errors.append(f"{ex.repo_id}@{ex.revision} resolved to {info.sha}")
        errors += [f"{ex.repo_id}/{p}: Hub {hub.get(p)}, pinned {h}" for p, h in ex.files.items() if hub.get(p) != h]
        if errors:
            break
        print(f"building extract {name} from {len(ex.files)} upstream file(s)")
        build_extract(dest, name)
        entries[extract_path(name)] = sha256_file(dest / extract_path(name))
    if errors:
        for e in errors:
            print(f"MISMATCH vs Hub: {e}", file=sys.stderr)
        print("CHECKSUMS not written", file=sys.stderr)
        return 1
    (dest / CHECKSUMS_NAME).write_text(format_checksums(entries))
    print(f"wrote {dest / CHECKSUMS_NAME} with {len(entries)} files, all matching the Hub's hashes")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", type=Path, default=EXTERNAL_DIR, help="mirror root (default: <repo>/external)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--write-checksums", action="store_true",
                      help="download the pinned snapshot, check it against the Hub's hashes, write CHECKSUMS")
    mode.add_argument("--verify-only", action="store_true", help="hash local files only; never download")
    args = ap.parse_args(argv)
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if args.write_checksums:
        return run_write_checksums(dest)
    return run_fetch(dest, args.verify_only)


if __name__ == "__main__":
    sys.exit(main())
