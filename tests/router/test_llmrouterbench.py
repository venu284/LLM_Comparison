"""S1.8: the LLMRouterBench pin, safe extraction, loader and profile.

Synthetic tests need no data. The data tests skip when external/llmrouterbench/ is absent.
"""

import importlib.util
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from router.data import llmrouterbench as L
from router.data.schema import NULL_SCORE

REPO_ROOT = Path(__file__).resolve().parents[2]
HAVE_DATA = L.ROOT.is_dir() and (L.SOURCES / "mbpp_full.jsonl").is_file()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="external/llmrouterbench not fetched")


def _fetch_data():
    if "fetch_data" in sys.modules:               # shared with tests/router/test_fetch_data.py
        return sys.modules["fetch_data"]
    spec = importlib.util.spec_from_file_location("fetch_data", REPO_ROOT / "scripts/fetch_data.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fetch_data"] = mod               # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


fd = _fetch_data()


# --------------------------------------------------------------------------- pin
def test_checksums_pin_the_archive_and_its_files():
    text = fd.COMMITTED_CHECKSUMS.read_text()
    ar = fd.ARCHIVES["llmrouterbench"]
    assert fd.parse_extract_pins(text, fd.ARCHIVE_PREFIX) == {"llmrouterbench": ar.spec()}
    assert len(ar.revision) == 40 and len(ar.sha256) == 64
    entries, _ = fd.parse_checksums(text)
    files = [p for p in entries if fd.archive_name(p) == "llmrouterbench"]
    assert files and all(p.removeprefix("llmrouterbench/").startswith(ar.members) for p in files)
    assert tuple(f"bench-release/{s}/" for s in L.CODE_SUBSETS) == ar.members   # only eligible subsets
    assert f"llmrouterbench/{ar.path}" not in entries       # the tarball itself is not kept


@needs_data
def test_pins_verify():
    entries, _ = fd.parse_checksums(fd.COMMITTED_CHECKSUMS.read_text())
    want = {p: h for p, h in entries.items()
            if fd.archive_name(p) == "llmrouterbench" or fd.extract_name(p) in fd.S1_8_EXTRACTS}
    assert len(want) == 98
    got = fd.hash_all(fd.EXTERNAL_DIR, want)
    assert got == want


# --------------------------------------------------------------------------- safe extraction
def _tar(path: Path, members: list[tarfile.TarInfo], data: dict[str, bytes] | None = None) -> Path:
    with tarfile.open(path, "w:gz") as t:
        for m in members:
            body = (data or {}).get(m.name)
            if body is not None:
                m.size = len(body)
                t.addfile(m, io.BytesIO(body))
            else:
                t.addfile(m)
    return path


def _file(name: str) -> tarfile.TarInfo:
    return tarfile.TarInfo(name)


def _link(name: str, target: str, kind=tarfile.SYMTYPE) -> tarfile.TarInfo:
    m = tarfile.TarInfo(name)
    m.type, m.linkname = kind, target
    return m


ALLOW = ("bench-release/humaneval/test/",)
GOOD = "bench-release/humaneval/test/m/a.json"


def test_extract_keeps_only_allowlisted_members(tmp_path):
    tar = _tar(tmp_path / "a.tar.gz", [_file(GOOD), _file("bench-release/arenahard/m/b.json")],
               {GOOD: b"{}", "bench-release/arenahard/m/b.json": b"{}"})
    out = fd.extract_archive(tar, tmp_path / "out", ALLOW)
    assert out == [GOOD]
    assert (tmp_path / "out" / GOOD).read_bytes() == b"{}"
    assert not (tmp_path / "out/bench-release/arenahard").exists()
    assert not (tmp_path / "out/.part").exists()


@pytest.mark.parametrize("bad", [
    _file("bench-release/humaneval/test/../../../evil.json"),           # '..'
    _file("/tmp/evil.json"),                                              # absolute path
    _link("bench-release/humaneval/test/m/link.json", "a.json"),          # symlink inside the target
    _link("bench-release/humaneval/test/m/link2.json", "/etc/passwd"),    # symlink outside
    _link("bench-release/humaneval/test/m/hard.json", GOOD, tarfile.LNKTYPE),   # hard link
    _link("bench-release/arenahard/x", "/etc/passwd"),                    # unsafe even outside the allowlist
])
def test_extract_rejects_unsafe_members_and_writes_nothing(tmp_path, bad):
    tar = _tar(tmp_path / "a.tar.gz", [_file(GOOD), bad], {GOOD: b"{}", bad.name: b"x"})
    with pytest.raises(fd.UnsafeArchive):
        fd.extract_archive(tar, tmp_path / "out", ALLOW)
    assert not (tmp_path / "out" / "bench-release").exists()    # nothing moved into place
    assert not (tmp_path / "out" / ".part").exists()
    assert not (tmp_path / "evil.json").exists()


def test_device_member_is_rejected():
    m = tarfile.TarInfo("bench-release/humaneval/test/m/dev")
    m.type = tarfile.CHRTYPE
    with pytest.raises(fd.UnsafeArchive):
        fd.check_member(m)


def test_fetch_archive_refuses_a_tarball_with_the_wrong_hash(tmp_path):
    tar = _tar(tmp_path / "a.tar.gz", [_file(GOOD)], {GOOD: b"{}"})
    with pytest.raises(ValueError, match="pinned"):
        fd.fetch_archive(tmp_path, "llmrouterbench", local=tar)
    with pytest.raises(ValueError, match="pinned"):
        fd.download_to(lambda *a: iter([b"abc"]), "r", "p", "rev", tmp_path / "x", "0" * 64)
    assert not (tmp_path / "x").exists()


# --------------------------------------------------------------------------- loader (synthetic)
def _rec(i, score, cost=0.001, raw="```python\npass\n```", tin=10, tout=5, **extra):
    return {"index": i, "origin_query": f"q{i}", "prompt": f"p{i}", "prompt_tokens": tin,
            "completion_tokens": tout, "cost": cost, "score": score, "prediction": "",
            "ground_truth": None, "raw_output": raw, **extra}


def _write(root: Path, subset: str, model: str, records: list[dict]) -> None:
    d = root / subset / model
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{model}.json").write_text(json.dumps({
        "performance": 0.0, "cost": sum(r["cost"] for r in records), "counts": len(records),
        "model_name": model, "records": records}))


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """Two subsets with different model sets: humaneval {a, z (zero cost), p (one zero-cost cell)},
    swe {a, b, openrouter}; one null score."""
    monkeypatch.setattr(L, "source_key", lambda subset, r: f"toy/{subset}/{r['index']}")
    root = tmp_path / "bench-release"
    _write(root, "humaneval/test", "a", [_rec(1, 1.0), _rec(2, 0.0), _rec(3, None, raw="Processing failed: x")])
    _write(root, "humaneval/test", "z", [_rec(i, s, cost=0.0) for i, s in ((1, 1.0), (2, 1.0), (3, 0.0))])
    _write(root, "humaneval/test", "p", [_rec(1, 0.0, cost=0.0, tin=0, tout=0), _rec(2, 1.0), _rec(3, 1.0)])
    for m in ("a", "b", "openrouter"):
        _write(root, "swe-bench/verified", m, [_rec(1, 1.0, instance_id="x__y-1"), _rec(2, 0.0, instance_id="x__y-2")])
    return root


def test_ragged_cells_are_absent_not_zero_filled(toy):
    d = L.load(toy)
    assert len(d.long) == 9 + 4                   # 3 models x 3 tasks + 2 models x 2 tasks
    pairs = set(zip(d.long.family, d.long.model))
    assert ("lrb_swebench_verified", "z") not in pairs and ("lrb_humaneval", "b") not in pairs
    assert "openrouter" not in set(d.long.model)  # router baseline off the model axis
    assert d.files.router_baseline.sum() == 1
    assert {f: m.score.shape for f, m in d.matrices.items()} == {"lrb_humaneval": (3, 3),
                                                                 "lrb_swebench_verified": (2, 2)}
    assert len(L.load(toy, include_router_baselines=True).long) == 15


def test_null_score_is_unavailable_with_a_reason(toy):
    d = L.load(toy)
    bad = d.long[~d.long.available]
    assert len(bad) == 1 and bad.iloc[0].unavailable_reason == NULL_SCORE and np.isnan(bad.iloc[0].score)
    m = d.matrices["lrb_humaneval"]
    assert not m.available.loc["lrb_humaneval_3", "a"]


def test_zero_cost_models_are_excluded_from_cost_but_kept_for_accuracy(toy):
    d = L.load(toy)
    assert d.zero_cost == {"lrb_humaneval": ["z"]}
    z = d.long[d.long.model == "z"]
    assert z.cost_usd.isna().all() and z.available.all() and list(z.score) == [1.0, 1.0, 0.0]
    p = d.cost_flags.set_index(["family", "model"]).loc[("lrb_humaneval", "p")]
    assert bool(p.partial_zero_cost) and not bool(p.zero_cost)
    assert (d.long[d.long.model == "p"].cost_usd == 0).sum() == 1   # recorded, not excluded
    assert d.matrices["lrb_humaneval"].cost["z"].isna().all()


def test_eligibility_does_not_depend_on_scores(toy, tmp_path):
    d1 = L.load(toy)
    rng = np.random.default_rng(0)
    for f in toy.rglob("*.json"):                 # scramble every score
        doc = json.loads(f.read_text())
        for r in doc["records"]:
            r["score"] = float(rng.integers(0, 2))
        f.write_text(json.dumps(doc))
    d2 = L.load(toy)
    cols = ["task_id", "model", "score_type", "eligible"]
    pd.testing.assert_frame_equal(d1.long[cols], d2.long[cols])
    assert all(d1.long.eligible) and set(d1.long.score_type) == {"exec"}


def test_rule_table_is_documented_and_judge_subset_is_ineligible():
    assert L.RULES["arenahard_coding"].score_type == "judge" and not L.RULES["arenahard_coding"].eligible
    for k, r in L.RULES.items():
        assert L.LRB_PAPER in r.lrb_rule and '"' in r.lrb_rule, k
        assert '"' in r.source_rule or not r.eligible, k      # eligible subsets quote the source rule
    assert set(L.CODE_SUBSETS) == set(L.FAMILY) == {k for k, r in L.RULES.items() if r.eligible}
    assert not set(L.FAMILY.values()) & {"humaneval", "mbpp", "lcb", "swebench"}   # no CRB name


def test_index_mismatch_across_files_is_an_error(toy):
    doc_path = next((toy / "humaneval/test/z").glob("*.json"))
    doc = json.loads(doc_path.read_text())
    doc["records"][0]["origin_query"] = "another task"
    doc_path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="different task"):
        L.load(toy)


# --------------------------------------------------------------------------- data
NAMESPACE = {"lrb_humaneval": r"he/\d+", "lrb_mbpp": r"mbpp/\d+", "lrb_livecodebench": r"lcb/\S+",
             "lrb_swebench_verified": r"swe/[\w.-]+__[\w.-]+-\d+"}


@needs_data
def test_source_keys_use_the_splits_namespace():
    from router.data.splits import load_problems
    d = L.load_cached()
    for fam, t in d.tasks.groupby("family"):
        assert t.source_key.notna().all() and t.source_key.is_unique, fam
        assert t.source_key.str.fullmatch(NAMESPACE[fam]).all(), fam
    crb = set(load_problems().source_key)
    for prefix in ("he/", "mbpp/", "lcb/", "swe/"):        # the same strings splits.source_key builds
        assert set(d.tasks.source_key[d.tasks.source_key.str.startswith(prefix)]) & crb, prefix
    he = d.tasks[d.tasks.family == "lrb_humaneval"]
    assert (he.source_key.str.split("/").str[1].astype(int) + 1 == he["index"]).all()


@needs_data
def test_real_data_is_not_zero_filled_and_matches_the_files():
    d = L.load_cached()
    models = d.files[~d.files.router_baseline]
    assert len(d.long) == int(models.records.sum())
    assert (d.files.records == d.files.counts).all()
    for fam, m in d.matrices.items():
        assert m.available.sum().sum() + (~m.available).sum().sum() == len(d.long[d.long.family == fam])
    assert int((~d.long.available).sum()) == 1 and set(d.long.unavailable_reason.dropna()) == {NULL_SCORE}
    assert not d.long[d.long.cost_usd.isna()].merge(
        d.cost_flags[~d.cost_flags.zero_cost], on=["family", "model"]).shape[0]


@needs_data
def test_profile_json_reproduces_byte_for_byte():
    from router.experiments import s1_8_profile as P
    assert P.dumps(P.build()) == P.OUT.read_text(encoding="utf-8")
    profile = json.loads(P.OUT.read_text())
    assert not re.search(r"oracle|dimensionbest|routing_gain", json.dumps(profile).lower())
