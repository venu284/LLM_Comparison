#!/usr/bin/env python
"""Fetch the pinned third-party datasets into `external/` and verify them.

Two Hugging Face dataset repos are mirrored, each under `<dest>/<repo-name>/` with its own
relative paths (so `router/data/coderouterbench.py` finds `external/CodeRouterBench/...`):

    Lance1573/CodeRouterBench      -> <dest>/CodeRouterBench/
    Lance1573/route-anything-data  -> <dest>/route-anything-data/

Both are pinned to full commit SHAs (REPOS below). `external/CHECKSUMS` (committed) holds the
SHA-256 of every file in `sha256sum` format, so `cd external && shasum -a 256 -c CHECKSUMS`
works without this script.

Normal run: read CHECKSUMS, hash the local files, download only the missing or mismatched
ones, re-hash, exit non-zero on any mismatch, print `fetched: N, verified: M`. When every file
already matches, no network call is made (works under HF_HUB_OFFLINE=1).

`--write-checksums` is the only mode that creates or overwrites CHECKSUMS. It takes the file
list from the Hub API at the pinned revision and checks every file against the Hub's own hash
(LFS sha256, or the git blob SHA-1 for non-LFS files) before writing anything.

Usage:
    pipeline/venv/bin/python scripts/fetch_data.py
    pipeline/venv/bin/python scripts/fetch_data.py --dest /path/to/dir
    pipeline/venv/bin/python scripts/fetch_data.py --verify-only
    pipeline/venv/bin/python scripts/fetch_data.py --write-checksums
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from router.settings import EXTERNAL_DIR  # noqa: E402

# The single pin: local mirror name -> (Hugging Face dataset repo id, full commit SHA).
REPOS: dict[str, tuple[str, str]] = {
    "CodeRouterBench": ("Lance1573/CodeRouterBench", "e567d89bdd569c9c74ffc7c7118e50d15e46b886"),
    "route-anything-data": ("Lance1573/route-anything-data", "bd361209720f27a2383857e314c554476bc81c5c"),
}

CHECKSUMS_NAME = "CHECKSUMS"
COMMITTED_CHECKSUMS = REPO_ROOT / "external" / CHECKSUMS_NAME
PIN_PREFIX = "# pin: "
WORKERS = 8
CHUNK = 1 << 20


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
    body = [f"{entries[p]}  {p}" for p in sorted(entries)]
    return "\n".join(header + body) + "\n"


def split_entry(path: str) -> tuple[str, str]:
    """'CodeRouterBench/a/b.csv' -> ('CodeRouterBench', 'a/b.csv')."""
    repo_name, _, rel = path.partition("/")
    if repo_name not in REPOS or not rel:
        raise ValueError(f"CHECKSUMS path {path!r} is not under one of {sorted(REPOS)}")
    return repo_name, rel


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
    wanted, pins = parse_checksums(checksums.read_text())
    for repo_id, sha in REPOS.values():
        if pins.get(repo_id) != sha:
            print(f"error: {checksums} pins {repo_id}@{pins.get(repo_id)}, script pins {sha}", file=sys.stderr)
            return 2
    for path in wanted:
        split_entry(path)

    local = hash_all(dest, wanted)
    todo = [p for p, h in local.items() if h != wanted[p]]
    fetched = 0
    if todo and not verify_only:
        print(f"{len(todo)} of {len(wanted)} files missing or mismatched; downloading")
        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(lambda p: download(dest, *split_entry(p)), todo))
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
