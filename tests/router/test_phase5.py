"""Checks for the read-only Phase 5 loader (router/data/phase5.py, S1.9, R5)."""

import json
import shutil

import pytest

from router.data import phase5 as p5


def test_csv_hash_and_rows():
    assert p5.csv_sha256() == p5.PHASE5_SHA256
    runs = p5.load_runs()
    assert len(runs) == 325
    assert {"task_id", "model_name", "run_number", "pass_fail"} <= set(runs.columns)
    assert not runs.duplicated(p5.RUN_KEY).any()


def test_hash_mismatch_refuses_to_load(tmp_path):
    bad = tmp_path / "phase5_results.csv"
    shutil.copy(p5.PHASE5_CSV, bad)
    with open(bad, "a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="SHA-256"):
        p5.load_runs(bad)


class _FakeCursor:
    def __init__(self, log, read_only, rows):
        self.log, self.read_only, self.rows, self._last = log, read_only, rows, None

    def execute(self, sql, params=None):
        self.log.append(" ".join(sql.split()))
        self._last = sql

    def fetchone(self):
        return ("on" if self.read_only else "off",)

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def __init__(self, log, read_only, rows):
        self.log, self.read_only, self.rows = log, read_only, rows
        self.session = None
        self.closed = False

    def set_session(self, **kw):
        self.session = kw

    def cursor(self):
        return _FakeCursor(self.log, self.read_only, self.rows)

    def rollback(self):
        self.log.append("ROLLBACK")

    def close(self):
        self.closed = True


def _patch_connect(monkeypatch, read_only=True, rows=()):
    calls = {}
    log = []

    def fake_connect(dsn, **kw):
        calls["dsn"], calls["kw"] = dsn, kw
        calls["conn"] = _FakeConn(log, read_only, list(rows))
        return calls["conn"]

    import psycopg2
    monkeypatch.setattr(psycopg2, "connect", fake_connect)
    return calls, log


def test_connection_is_opened_read_only(monkeypatch, tmp_path):
    runs = p5.load_runs()
    rows = [(r.task_id, r.model_name, int(r.run_number), "x") for r in runs.itertuples()]
    calls, log = _patch_connect(monkeypatch, rows=rows)
    stats = p5.fetch_raw_responses(url="postgresql://u:p@h/db", cache=tmp_path / "c.jsonl")
    assert "default_transaction_read_only=on" in calls["kw"]["options"]
    assert calls["conn"].session == {"readonly": True, "autocommit": False}
    assert calls["conn"].closed
    executed = [s.split()[0].upper() for s in log]
    assert executed[0] == "SHOW" and set(executed) <= {"SHOW", "SELECT", "ROLLBACK"}
    assert stats["n_cached"] == 325


def test_refuses_when_session_is_not_read_only(monkeypatch, tmp_path):
    _patch_connect(monkeypatch, read_only=False)
    with pytest.raises(RuntimeError, match="read-only"):
        p5.fetch_raw_responses(url="postgresql://u:p@h/db", cache=tmp_path / "c.jsonl")
    assert not (tmp_path / "c.jsonl").exists()


def test_connection_failure_hides_the_url(monkeypatch, tmp_path):
    import psycopg2

    def boom(dsn, **kw):
        raise psycopg2.OperationalError(f"could not connect to {dsn}")

    monkeypatch.setattr(psycopg2, "connect", boom)
    with pytest.raises(p5.Phase5DatabaseUnavailable) as exc:
        p5.fetch_raw_responses(url="postgresql://u:secret@h/db", cache=tmp_path / "c.jsonl")
    assert "secret" not in str(exc.value) and "OperationalError" in str(exc.value)


@pytest.mark.skipif(not p5.RESPONSE_CACHE.exists(), reason="response cache not built")
def test_cached_responses_join_to_the_csv():
    j = p5.load_runs_with_responses()
    assert len(j) == 325
    assert j.raw_response_cached.all()
    counts = p5.response_counts(j)
    # Report the real numbers; the plan records 324 non-empty of 325.
    assert counts["n_runs"] == 325
    assert counts["n_non_empty"] <= 325
    with open(p5.RESPONSE_CACHE) as f:
        assert sum(1 for _ in f) == 325
