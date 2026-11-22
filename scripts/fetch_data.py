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

LLMRouterBench (S1.8) is pinned as an archive (ARCHIVES below): one tarball at a pinned commit,
with the Hub's own sha256 of it. The tarball is untrusted data. It is downloaded into a fresh
empty temporary directory outside `external/`, checked against the pinned sha256, and only the
allowlisted member prefixes are extracted, through `extract_archive`, which rejects absolute
paths, `..` components and every member that is not a regular file or a directory (symlinks,
hard links, devices), before Python's `data` filter is applied as well. Extraction goes to a
staging directory first and replaces the old files only when it completes. The tarball is then
deleted: the pin allows a re-download. Each extracted file's SHA-256 goes into CHECKSUMS; the
tarball's own hash is in the `# archive:` header line, not the body, since the file is not kept.

Normal run: read CHECKSUMS, hash the local files, download only the missing or mismatched
ones (rebuilding a missing or mismatched extract from its upstream files), re-hash, exit
non-zero on any mismatch, print `fetched: N, verified: M`. When every file already matches, no
network call is made (works under HF_HUB_OFFLINE=1).

`--write-new-checksums` adds the entries of pins that CHECKSUMS does not hold yet (a new
extract or archive), after the same Hub checks; every existing line is kept byte for byte.

`--write-checksums` is the only mode that creates or overwrites CHECKSUMS as a whole. It takes the file
list from the Hub API at the pinned revision and checks every file against the Hub's own hash
(LFS sha256, or the git blob SHA-1 for non-LFS files) before writing anything. For extracts it
checks that the pinned per-file hashes are the Hub's at the pinned revision, then builds them.

Usage:
    pipeline/venv/bin/python scripts/fetch_data.py
    pipeline/venv/bin/python scripts/fetch_data.py --dest /path/to/dir
    pipeline/venv/bin/python scripts/fetch_data.py --verify-only
    pipeline/venv/bin/python scripts/fetch_data.py --write-checksums
    pipeline/venv/bin/python scripts/fetch_data.py --write-new-checksums [--archive-from TARBALL]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
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
    # S1.8: task text with the source ids, to map LLMRouterBench records (which carry only a list
    # position and the task text) to source keys. Full MBPP (974 tasks, all four splits) and the
    # original HumanEval prompts (HumanEval+ edits HumanEval/115's prompt). MBPP keeps `test_list`:
    # LLMRouterBench's MBPP prompt embeds it, and it separates the four pairs of MBPP tasks that
    # share a text. It is data only; nothing executes it.
    "mbpp_full": Extract(
        "google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        {
            "full/prompt-00000-of-00001.parquet": ("sha256", "a053e4bb85ceb77430ae80592addb4ca4dc6ba087592f9e04537800ee88b7431"),
            "full/test-00000-of-00001.parquet": ("sha256", "566fd53060ffba5766dace1d1e2f4c38906781526de222b0dfbdbc325b696c77"),
            "full/train-00000-of-00001.parquet": ("sha256", "09d125ca31edacb7800be8c67c45abff618faf0214ff551291817d06bdb914ae"),
            "full/validation-00000-of-00001.parquet": ("sha256", "3f0ec060987432d99fe8fb409d31e6c67445b208a01741c5583517c80a10fe80"),
        },
        ("task_id", "text", "test_list"),
    ),
    "openai_humaneval": Extract(
        "openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        {"openai_humaneval/test-00000-of-00001.parquet": ("sha256", "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6")},
        ("task_id", "prompt"),
    ),
    # S1.8: the 500 SWE-bench Verified instance ids, to check the ids LLMRouterBench records.
    "swebench_verified_ids": Extract(
        "princeton-nlp/SWE-bench_Verified", "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
        {"data/test-00000-of-00001.parquet": ("sha256", "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd")},
        ("instance_id", "repo"),
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

# S1.8 extracts: source ids with task text, read only by router/data/llmrouterbench.py to map
# LLMRouterBench records to source keys. router/data/prompts.py does not read them.
S1_8_EXTRACTS: frozenset[str] = frozenset({"mbpp_full", "openai_humaneval", "swebench_verified_ids"})
# The one test field outside AUDIT_EXTRACTS: MBPP `test_list`, matched as text against
# LLMRouterBench's MBPP prompt (which embeds it) to tell apart MBPP tasks that share a text.
# Nothing executes it.
S1_8_TEST_FIELDS: frozenset[tuple[str, str]] = frozenset({("mbpp_full", "test_list")})



@dataclass(frozen=True)
class Archive:
    """Allowlisted members of one pinned tarball, extracted under `<dest>/<name>/`.

    `sha256` is the Hub's LFS hash of `path` at `revision`. `members` are path prefixes inside the
    tarball; nothing else is written.
    """

    repo_id: str
    revision: str
    path: str
    sha256: str
    members: tuple[str, ...]

    def spec(self) -> str:
        return (f"{self.repo_id}@{self.revision} file={self.path} sha256={self.sha256} "
                f"members={','.join(self.members)}")


# S1.8: LLMRouterBench (arXiv 2601.07206). The tarball (1,283,503,080 bytes) is unchanged since
# 2025-12-08: the same LFS sha256 at 374c617, 0e5af1b and b95870f (2026-11-02, adds LICENSE).
# Only the four execution-scored code subsets are extracted; arenahard_coding is judge-scored.
ARCHIVES: dict[str, Archive] = {
    "llmrouterbench": Archive(
        "NPULH/LLMRouterBench", "b95870f9e60c589e82f7972c91ffa8de9ca89f85", "bench-release.tar.gz",
        "b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1",
        ("bench-release/humaneval/test/", "bench-release/mbpp/test/",
         "bench-release/livecodebench/test/", "bench-release/swe-bench/verified/"),
    ),
}

CHECKSUMS_NAME = "CHECKSUMS"
COMMITTED_CHECKSUMS = REPO_ROOT / "external" / CHECKSUMS_NAME
PIN_PREFIX = "# pin: "
EXTRACT_PREFIX = "# extract: "
ARCHIVE_PREFIX = "# archive: "
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
    header += [f"{ARCHIVE_PREFIX}{name} <- {ar.spec()}" for name, ar in ARCHIVES.items()]
    body = [f"{entries[p]}  {p}" for p in sorted(entries)]
    return "\n".join(header + body) + "\n"


def parse_extract_pins(text: str, prefix: str = EXTRACT_PREFIX) -> dict[str, str]:
    """{name: spec} from the `# extract:` (or, with `prefix`, `# archive:`) lines of CHECKSUMS."""
    pins: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith(prefix):
            name, sep, spec = line[len(prefix):].strip().partition(" <- ")
            if not sep:
                raise ValueError(f"malformed pin: {line!r}")
            pins[name] = spec
    return pins


def archive_name(path: str) -> str | None:
    """'llmrouterbench/bench-release/...' -> 'llmrouterbench' when it is a pinned archive's file."""
    head, _, rel = path.partition("/")
    return head if head in ARCHIVES and rel else None


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


# --- archives ---------------------------------------------------------------------------------


class UnsafeArchive(ValueError):
    """A tar member that could write outside the target, or is not a plain file or directory."""


def check_member(m: tarfile.TarInfo) -> None:
    """Reject absolute paths, `..` components, and every member type except regular files and
    directories (symlinks, hard links, devices, FIFOs). Applied to every member of the archive."""
    name = m.name
    if name.startswith(("/", "\\")) or (len(name) > 1 and name[1] == ":"):
        raise UnsafeArchive(f"absolute path in archive: {name!r}")
    if ".." in name.replace("\\", "/").split("/"):
        raise UnsafeArchive(f"'..' component in archive: {name!r}")
    if not (m.isfile() or m.isdir()):
        raise UnsafeArchive(f"not a regular file or directory ({m.type!r}): {name!r}")


def extract_archive(tar_path: Path, root: Path, members: tuple[str, ...]) -> list[str]:
    """Extract the members of `tar_path` under the `members` prefixes into `root`.

    Every member is checked (`check_member`), allowlisted or not, so a malformed archive is
    rejected as a whole. Allowlisted members also pass Python's `data` filter. Files are written
    to `root/.part/` first and moved into place only when the whole archive has been read.
    Returns the extracted file paths relative to `root`, sorted.
    """
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    staging = root / ".part"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    written: list[str] = []
    try:
        with tarfile.open(tar_path, "r|*") as tar:
            for m in tar:
                check_member(m)
                if not m.name.startswith(members):
                    continue
                tarfile.data_filter(m, str(staging))
                target = (staging / m.name).resolve()
                if staging not in target.parents:
                    raise UnsafeArchive(f"member resolves outside the target: {m.name!r}")
                if m.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(m)
                with open(target, "wb") as f:
                    shutil.copyfileobj(src, f, CHUNK)
                written.append(m.name)
        tops = sorted({w.split("/")[0] for w in written})
        for top in tops:
            shutil.rmtree(root / top, ignore_errors=True)
            os.replace(staging / top, root / top)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return sorted(written)


def download_to(opener: Opener, repo_id: str, path: str, revision: str, out: Path, sha256: str) -> None:
    """Stream one Hub file to `out` and check its sha256; the file is removed on a mismatch."""
    h = hashlib.sha256()
    with open(out, "wb") as f:
        for chunk in opener(repo_id, path, revision):
            h.update(chunk)
            f.write(chunk)
    if h.hexdigest() != sha256:
        out.unlink()
        raise ValueError(f"{repo_id}@{revision}/{path}: sha256 {h.hexdigest()}, pinned {sha256}")


def fetch_archive(dest: Path, name: str, local: Path | None = None, opener: Opener = open_hub) -> list[str]:
    """Download (or take `local`), verify and extract one pinned archive under `<dest>/<name>/`;
    the tarball never stays on disk unless it was given as `local`. Returns CHECKSUMS paths."""
    ar = ARCHIVES[name]
    with tempfile.TemporaryDirectory(prefix=f"{name}-download-") as tmp:
        if local is not None:
            got = sha256_file(local)
            if got != ar.sha256:
                raise ValueError(f"{local}: sha256 {got}, pinned {ar.sha256}")
            tar_path = local
        else:
            tar_path = Path(tmp) / Path(ar.path).name
            download_to(opener, ar.repo_id, ar.path, ar.revision, tar_path, ar.sha256)
        files = extract_archive(tar_path, dest / name, ar.members)
    return [f"{name}/{f}" for f in files]


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


def run_fetch(dest: Path, verify_only: bool, archive_from: Path | None = None) -> int:
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
    archive_pins = parse_extract_pins(text, ARCHIVE_PREFIX)
    for name, ar in ARCHIVES.items():
        if name in archive_pins and archive_pins[name] != ar.spec():
            print(f"error: {checksums} pins archive {name} as {archive_pins[name]}, script as {ar.spec()}",
                  file=sys.stderr)
            return 2
    for path in wanted:
        if extract_name(path) is None and archive_name(path) is None:
            split_entry(path)

    local = hash_all(dest, wanted)
    todo = [p for p, h in local.items() if h != wanted[p]]
    fetched = 0
    if todo and not verify_only:
        print(f"{len(todo)} of {len(wanted)} files missing or mismatched; downloading")
        mirror = [p for p in todo if extract_name(p) is None and archive_name(p) is None]
        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(lambda p: download(dest, *split_entry(p)), mirror))
        for p in todo:
            if (name := extract_name(p)) is not None:
                build_extract(dest, name)
        for name in sorted({archive_name(p) for p in todo} - {None}):
            fetch_archive(dest, name, local=archive_from)
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
    for name, ar in ([] if errors else ARCHIVES.items()):
        info = api.dataset_info(ar.repo_id, revision=ar.revision, files_metadata=True)
        hub = {s.rfilename: hub_expected(s) for s in info.siblings}
        if info.sha != ar.revision or hub.get(ar.path) != ("sha256", ar.sha256):
            errors.append(f"{ar.repo_id}@{ar.revision}/{ar.path}: Hub {hub.get(ar.path)}, pinned {ar.sha256}")
            break
        print(f"fetching archive {name}")
        for path in fetch_archive(dest, name):
            entries[path] = sha256_file(dest / path)
    if errors:
        for e in errors:
            print(f"MISMATCH vs Hub: {e}", file=sys.stderr)
        print("CHECKSUMS not written", file=sys.stderr)
        return 1
    (dest / CHECKSUMS_NAME).write_text(format_checksums(entries))
    print(f"wrote {dest / CHECKSUMS_NAME} with {len(entries)} files, all matching the Hub's hashes")
    return 0


def _hub_check_extract(api, ex: Extract) -> list[str]:
    info = api.dataset_info(ex.repo_id, revision=ex.revision, files_metadata=True)
    hub = {s.rfilename: hub_expected(s) for s in info.siblings}
    errors = [f"{ex.repo_id}@{ex.revision} resolved to {info.sha}"] if info.sha != ex.revision else []
    return errors + [f"{ex.repo_id}/{p}: Hub {hub.get(p)}, pinned {h}" for p, h in ex.files.items()
                     if hub.get(p) != h]


def run_write_new_checksums(dest: Path, archive_from: Path | None = None) -> int:
    """Add CHECKSUMS entries for extracts and archives the committed file does not pin yet.

    Existing pins must match the script (a changed pin needs --write-checksums); every existing
    line is kept byte for byte, which is checked before writing.
    """
    from huggingface_hub import HfApi

    source = dest / CHECKSUMS_NAME if (dest / CHECKSUMS_NAME).is_file() else COMMITTED_CHECKSUMS
    old = source.read_text()
    entries, pins = parse_checksums(old)
    if any(pins.get(r) != sha for r, sha in REPOS.values()):
        print("error: repo pins differ from the script; use --write-checksums", file=sys.stderr)
        return 2
    ex_pins, ar_pins = parse_extract_pins(old), parse_extract_pins(old, ARCHIVE_PREFIX)
    changed = [n for n, ex in EXTRACTS.items() if n in ex_pins and ex_pins[n] != ex.spec()]
    changed += [n for n, ar in ARCHIVES.items() if n in ar_pins and ar_pins[n] != ar.spec()]
    if changed:
        print(f"error: pins changed for {changed}; use --write-checksums", file=sys.stderr)
        return 2
    api = HfApi()
    for name, ex in EXTRACTS.items():
        if name in ex_pins:
            continue
        errors = _hub_check_extract(api, ex)
        if errors:
            print("\n".join(f"MISMATCH vs Hub: {e}" for e in errors), file=sys.stderr)
            return 1
        print(f"building new extract {name}")
        build_extract(dest, name)
        entries[extract_path(name)] = sha256_file(dest / extract_path(name))
    for name, ar in ARCHIVES.items():
        if name in ar_pins:
            continue
        info = api.dataset_info(ar.repo_id, revision=ar.revision, files_metadata=True)
        hub = {s.rfilename: hub_expected(s) for s in info.siblings}
        if info.sha != ar.revision or hub.get(ar.path) != ("sha256", ar.sha256):
            print(f"MISMATCH vs Hub: {ar.repo_id}@{info.sha}/{ar.path}: {hub.get(ar.path)}, pinned "
                  f"{ar.revision} sha256 {ar.sha256}", file=sys.stderr)
            return 1
        print(f"fetching new archive {name}")
        for path in fetch_archive(dest, name, local=archive_from):
            entries[path] = sha256_file(dest / path)
    new = format_checksums(entries)
    lost = set(old.splitlines()) - set(new.splitlines())
    if lost:
        print(f"error: {len(lost)} existing CHECKSUMS lines would change; nothing written", file=sys.stderr)
        return 1
    (dest / CHECKSUMS_NAME).write_text(new)
    print(f"wrote {dest / CHECKSUMS_NAME}: {len(new.splitlines()) - len(old.splitlines())} lines added")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", type=Path, default=EXTERNAL_DIR, help="mirror root (default: <repo>/external)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--write-checksums", action="store_true",
                      help="download the pinned snapshot, check it against the Hub's hashes, write CHECKSUMS")
    mode.add_argument("--verify-only", action="store_true", help="hash local files only; never download")
    mode.add_argument("--write-new-checksums", action="store_true",
                      help="add entries for pins CHECKSUMS does not hold yet; keep every existing line")
    ap.add_argument("--archive-from", type=Path, default=None,
                    help="use this local copy of a pinned tarball (checked against its sha256)")
    args = ap.parse_args(argv)
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if args.write_checksums:
        return run_write_checksums(dest)
    if args.write_new_checksums:
        return run_write_new_checksums(dest, args.archive_from)
    return run_fetch(dest, args.verify_only, args.archive_from)


if __name__ == "__main__":
    sys.exit(main())
