"""Pure parts of scripts/fetch_data.py (no network)."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fetch_data", Path(__file__).resolve().parents[2] / "scripts" / "fetch_data.py"
)
fd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fd)

H = "a" * 64


def test_blob_sha1_matches_git(tmp_path):
    f = tmp_path / "x"
    f.write_bytes(b"hello\n")
    # `printf 'hello\n' | git hash-object --stdin`
    assert fd.git_blob_sha1(f) == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert fd.sha256_file(f) == hashlib.sha256(b"hello\n").hexdigest()


def test_parse_checksums_skips_comments_and_reads_pins():
    text = f"# note\n# pin: owner/repo@{'b' * 40}\n\n{H}  CodeRouterBench/a b.csv\n"
    entries, pins = fd.parse_checksums(text)
    assert entries == {"CodeRouterBench/a b.csv": H}
    assert pins == {"owner/repo": "b" * 40}


@pytest.mark.parametrize("line", ["abc  CodeRouterBench/x", f"{H} CodeRouterBench/x", f"{H}  "])
def test_parse_checksums_rejects_malformed(line):
    with pytest.raises(ValueError):
        fd.parse_checksums(line + "\n")


def test_parse_checksums_rejects_duplicates():
    with pytest.raises(ValueError):
        fd.parse_checksums(f"{H}  CodeRouterBench/x\n{H}  CodeRouterBench/x\n")


def test_format_roundtrip_sorted_with_pins():
    entries = {"route-anything-data/z": H, "CodeRouterBench/a": "c" * 64}
    text = fd.format_checksums(entries)
    parsed, pins = fd.parse_checksums(text)
    assert parsed == entries
    assert pins == {rid: sha for rid, sha in fd.REPOS.values()}
    paths = [ln.split("  ", 1)[1] for ln in text.splitlines() if not ln.startswith("#")]
    assert paths == sorted(paths)


def test_exclusions_and_entry_split():
    assert fd.is_excluded(".cache/huggingface/download/x.metadata")
    assert fd.is_excluded("sub/.gitattributes")
    assert not fd.is_excluded("raw_matrices/a.json")
    assert fd.split_entry("CodeRouterBench/a/b.csv") == ("CodeRouterBench", "a/b.csv")
    with pytest.raises(ValueError):
        fd.split_entry("other/a")
