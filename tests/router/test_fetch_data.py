"""Pure parts of scripts/fetch_data.py (no network)."""

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fetch_data", Path(__file__).resolve().parents[2] / "scripts" / "fetch_data.py"
)
fd = importlib.util.module_from_spec(_SPEC)
sys.modules["fetch_data"] = fd  # dataclasses resolve string annotations through sys.modules
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


# --- extract pins ------------------------------------------------------------------------------

# Names of upstream columns that carry solutions, tests or expected outputs in the pinned sources.
FORBIDDEN = {
    "canonical_solution", "reference_code", "code_context", "code", "solution", "solutions",
    "test", "tests", "test_list", "test_imports", "challenge_test_list", "example_test",
    "doctests", "public_test_cases", "private_test_cases", "base_input", "plus_input", "atol",
    "output", "expected_output", "instruct_prompt", "code_prompt",
}


def _sha1_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _toy(files: dict[str, bytes], fields, kind="sha256"):
    def h(b):
        return hashlib.sha256(b).hexdigest() if kind == "sha256" else _sha1_blob(b)
    ex = fd.Extract("owner/toy", "c" * 40, {p: (kind, h(b)) for p, b in files.items()}, tuple(fields))

    def opener(repo_id, path, revision):
        assert (repo_id, revision) == ("owner/toy", "c" * 40)
        data = files[path]
        return [data[i:i + 7] for i in range(0, len(data), 7)]  # odd chunk size splits lines
    return ex, opener


def test_every_pinned_field_allowlist_excludes_solutions_tests_and_outputs():
    for name, ex in fd.EXTRACTS.items():
        assert len(ex.revision) == 40 and all(k in ("sha256", "sha1") for k, _ in ex.files.values())
        if name in fd.AUDIT_EXTRACTS:          # S1.7 audit: tests on purpose, sandbox only
            continue
        for field in ex.fields:
            assert field not in FORBIDDEN and field.split(".")[-1] not in FORBIDDEN, (name, field)


def test_audit_extracts_are_named_and_unused_by_the_prompt_loader():
    from pathlib import Path

    assert fd.AUDIT_EXTRACTS <= set(fd.EXTRACTS)
    prompts_src = (Path(fd.REPO_ROOT) / "router/data/prompts.py").read_text()
    for name in fd.AUDIT_EXTRACTS:
        assert name not in prompts_src, name


@pytest.mark.parametrize("kind", ["sha256", "sha1"])
def test_extract_rows_jsonl_keeps_only_allowlisted_fields_in_order(kind):
    rows = [{"id": i, "prompt": f"p{i}", "canonical_solution": "SECRET", "meta": {"n": i * 10}} for i in range(5)]
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    ex, opener = _toy({"a.jsonl": data}, ["meta.n", "prompt"], kind)
    out = list(fd.extract_rows(ex, opener))
    assert out == [{"meta.n": i * 10, "prompt": f"p{i}"} for i in range(5)]
    assert "SECRET" not in json.dumps(out)


def test_extract_rows_reads_only_allowlisted_parquet_columns():
    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = io.BytesIO()
    pq.write_table(pa.table({"task_id": ["x/0", "x/1"], "prompt": ["a", "b"], "tests": ["T", "T"]}), buf)
    ex, opener = _toy({"d.parquet": buf.getvalue()}, ["task_id", "prompt"])
    assert list(fd.extract_rows(ex, opener)) == [{"task_id": "x/0", "prompt": "a"}, {"task_id": "x/1", "prompt": "b"}]


def test_upstream_hash_mismatch_fails_and_leaves_no_extract(tmp_path, monkeypatch):
    ex, opener = _toy({"a.jsonl": b'{"prompt": "p"}\n'}, ["prompt"])
    ex = fd.Extract(ex.repo_id, ex.revision, {"a.jsonl": ("sha256", "0" * 64)}, ex.fields)
    monkeypatch.setattr(fd, "EXTRACTS", {"toy": ex})
    with pytest.raises(ValueError, match="pinned"):
        fd.build_extract(tmp_path, "toy", opener)
    assert not list((tmp_path / "sources").iterdir())


def test_missing_field_is_an_error():
    ex, opener = _toy({"a.jsonl": b'{"prompt": "p"}\n'}, ["prompt", "entry_point"])
    with pytest.raises(KeyError):
        list(fd.extract_rows(ex, opener))


def test_extract_pins_roundtrip():
    text = fd.format_checksums({})
    assert fd.parse_extract_pins(text) == {n: ex.spec() for n, ex in fd.EXTRACTS.items()}
    assert fd.extract_name("sources/ds1000.jsonl") == "ds1000"
    assert fd.extract_name("sources/other.jsonl") is None
    assert fd.extract_name("CodeRouterBench/sources/ds1000.jsonl") is None


def test_fetch_rebuilds_a_corrupted_extract_and_verify_only_never_fetches(tmp_path, monkeypatch, capsys):
    data = b'{"id": 1, "prompt": "p1"}\n{"id": 2, "prompt": "p2"}\n'
    ex, opener = _toy({"a.jsonl": data}, ["id", "prompt"])
    monkeypatch.setattr(fd, "EXTRACTS", {"toy": ex})
    calls = []

    def fake_open(*a):
        calls.append(a)
        return opener(*a)
    real_build = fd.build_extract
    monkeypatch.setattr(fd, "build_extract", lambda dest, name: real_build(dest, name, fake_open))

    real_build(tmp_path, "toy", opener)
    good = fd.sha256_file(tmp_path / "sources/toy.jsonl")
    (tmp_path / fd.CHECKSUMS_NAME).write_text(fd.format_checksums({"sources/toy.jsonl": good}))
    (tmp_path / "sources/toy.jsonl").write_text("corrupted\n")

    assert fd.run_fetch(tmp_path, verify_only=True) == 1
    assert calls == [] and "MISMATCH: sources/toy.jsonl" in capsys.readouterr().err

    assert fd.run_fetch(tmp_path, verify_only=False) == 0
    assert len(calls) == 1 and fd.sha256_file(tmp_path / "sources/toy.jsonl") == good
    assert "fetched: 1, verified: 1" in capsys.readouterr().out


def test_fetch_refuses_a_checksums_file_with_a_different_extract_pin(tmp_path, monkeypatch):
    ex, _ = _toy({"a.jsonl": b"{}\n"}, ["prompt"])
    monkeypatch.setattr(fd, "EXTRACTS", {"toy": ex})
    text = fd.format_checksums({}).replace("fields=prompt", "fields=prompt,canonical_solution")
    (tmp_path / fd.CHECKSUMS_NAME).write_text(text)
    assert fd.run_fetch(tmp_path, verify_only=True) == 2
