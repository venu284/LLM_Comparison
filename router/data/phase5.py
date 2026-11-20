"""Read-only loader for the Phase 5 web-dev runs (S1.9, Rule R5).

- Runs: the frozen snapshot `data/phase5_results.csv` (325 runs; SHA-256 in `data/README.md`),
  loaded through `pipeline/analysis/data.py` with the database switched off. A file whose hash
  differs is refused.
- Raw responses: the CSV has no response text. `fetch_raw_responses()` reads
  `task_id, model_name, run_number, raw_response` from the Phase 5 `runs` table in a read-only
  session (`default_transaction_read_only=on` and `set_session(readonly=True)`, checked with
  `SHOW` before any query) and caches them to a git-ignored JSONL file under `external/phase5/`,
  so later stages work offline. It never writes to the database (R5). Only the rows whose
  (task_id, model_name, run_number) key is in the CSV are cached.
- The connection string comes from `DATABASE_URL` (environment first, then `pipeline/.env`). It is
  never printed, logged or put in an error message.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from router.settings import EXTERNAL_DIR, PHASE5_CSV, REPO_ROOT

PHASE5_SHA256 = "abaceb9a17c65c42333ef98d6762b8685db0ccb2a4d0dec52e3563cdd39f04c5"
RESPONSE_CACHE = EXTERNAL_DIR / "phase5" / "raw_responses.jsonl"
ENV_FILE = REPO_ROOT / "pipeline" / ".env"
RUN_KEY = ["task_id", "model_name", "run_number"]

_QUERY = """
    SELECT t.task_id, m.name AS model_name, r.run_number, r.raw_response
    FROM runs r
    JOIN tasks t ON t.id = r.task_id
    JOIN models m ON m.id = r.model_id
"""


class Phase5DatabaseUnavailable(RuntimeError):
    """The Phase 5 database could not be reached or queried. Carries no connection details."""


def csv_sha256(path: Path = PHASE5_CSV) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_runs(path: Path = PHASE5_CSV) -> pd.DataFrame:
    """The 325 Phase 5 runs from the frozen CSV, after checking its SHA-256."""
    digest = csv_sha256(path)
    if digest != PHASE5_SHA256:
        raise ValueError(f"Phase 5 CSV SHA-256 mismatch: {digest} != {PHASE5_SHA256}; not loading")
    from analysis.data import load_experiment_data   # pipeline module, bare name (section 3)

    return load_experiment_data(csv_path=Path(path), prefer_database=False).runs


def _database_url() -> str | None:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    if ENV_FILE.exists():
        from dotenv import dotenv_values

        return dotenv_values(ENV_FILE).get("DATABASE_URL") or None
    return None


def _connect_read_only(url: str):
    import psycopg2

    try:
        conn = psycopg2.connect(url, connect_timeout=20,
                                options="-c default_transaction_read_only=on")
    except Exception as exc:  # noqa: BLE001 - never echo the DSN
        raise Phase5DatabaseUnavailable(f"connect failed: {type(exc).__name__}") from None
    conn.set_session(readonly=True, autocommit=False)
    return conn


def fetch_raw_responses(url: str | None = None, cache: Path = RESPONSE_CACHE,
                        runs: pd.DataFrame | None = None) -> dict:
    """Read the Phase 5 raw responses once, read-only, and cache the CSV's 325 runs to `cache`.

    One connection attempt; no retries. Returns counts only.
    """
    url = url or _database_url()
    if not url:
        raise Phase5DatabaseUnavailable("DATABASE_URL not set")
    runs = load_runs() if runs is None else runs
    conn = _connect_read_only(url)
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW default_transaction_read_only")
            if cur.fetchone()[0] != "on":
                raise RuntimeError("Phase 5 session is not read-only; refusing to query")
            try:
                cur.execute(_QUERY)
                rows = cur.fetchall()
            except Exception as exc:  # noqa: BLE001
                raise Phase5DatabaseUnavailable(f"query failed: {type(exc).__name__}") from None
        conn.rollback()
    finally:
        conn.close()

    db = pd.DataFrame(rows, columns=RUN_KEY + ["raw_response"])
    db["run_number"] = db.run_number.astype(int)
    keys = runs[RUN_KEY].assign(run_number=runs.run_number.astype(int))
    dup = db.duplicated(RUN_KEY, keep=False)
    j = keys.merge(db[~dup], on=RUN_KEY, how="left", indicator=True)
    cached = j[j._merge == "both"].drop(columns="_merge")

    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        for r in cached.itertuples(index=False):
            f.write(json.dumps({"task_id": r.task_id, "model_name": r.model_name,
                                "run_number": int(r.run_number),
                                "raw_response": r.raw_response}) + "\n")
    return {
        "n_db_rows": len(db),
        "n_db_duplicate_key_rows": int(dup.sum()),
        "n_db_rows_not_in_csv": int(len(db[~dup].merge(keys, on=RUN_KEY, how="left",
                                                       indicator=True)
                                        .query("_merge == 'left_only'"))),
        "n_csv_runs": len(keys),
        "n_cached": len(cached),
        "n_csv_runs_without_response_row": int((j._merge == "left_only").sum()),
    }


def load_responses(cache: Path = RESPONSE_CACHE) -> pd.DataFrame:
    return pd.read_json(cache, lines=True, dtype={"task_id": str, "model_name": str})


def load_runs_with_responses(cache: Path = RESPONSE_CACHE) -> pd.DataFrame:
    """CSV runs left-joined to the cached raw responses."""
    runs = load_runs()
    resp = load_responses(cache)
    j = runs.merge(resp, on=RUN_KEY, how="left", indicator=True, validate="one_to_one")
    j["raw_response_cached"] = j.pop("_merge") == "both"
    return j


def response_counts(j: pd.DataFrame) -> dict:
    """How many responses are NULL, empty, whitespace-only and non-empty."""
    r = j.raw_response
    is_null = r.isna()
    text = r.fillna("").astype(str)
    return {
        "n_runs": len(j),
        "n_joined": int(j.raw_response_cached.sum()),
        "n_null": int(is_null.sum()),
        "n_empty_string": int((~is_null & (text == "")).sum()),
        "n_whitespace_only": int((~is_null & (text != "") & (text.str.strip() == "")).sum()),
        "n_non_empty": int((text.str.strip() != "").sum()),
    }
