"""Single loading path for the Phase 5 experimental results.

Prefers PostgreSQL, falls back to the CSV export. Every Phase 6 consumer goes
through here so that no two analyses disagree about what the data says.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import MODELS, get_database_url, get_paths

logger = logging.getLogger(__name__)

CATEGORIES = ["frontend", "api", "css", "typescript", "bugfix"]
DIFFICULTIES = ["easy", "medium", "hard"]

# Active parameter counts in billions. For mixture-of-experts models this is the
# active count, not the total, because that is what governs inference cost.
MODEL_PARAMS_B = {
    "GPT-OSS-120B": 120.0,
    "Llama-3.3-70B": 70.0,
    "Llama-4-Scout": 17.0,
    "Qwen3-32B": 32.0,
    "Llama-3.1-8B": 8.0,
}

_BOOL_TRUE = {"true", "t", "1", "yes"}

_NUMERIC_COLUMNS = (
    "latency_total_ms",
    "tokens_input",
    "tokens_output",
    "tests_total",
    "tests_passed",
)


@dataclass
class ExperimentData:
    """Tidy run-level results plus the provenance needed to describe them."""

    runs: pd.DataFrame
    source: str
    zero_test_rows: int
    excluded_zero_tests: bool

    @property
    def models(self) -> List[str]:
        return sorted(self.runs["model_name"].unique())

    @property
    def task_ids(self) -> List[str]:
        return sorted(self.runs["task_id"].unique())

    def pass_matrix(self) -> pd.DataFrame:
        """Tasks as rows, models as columns, boolean pass. The paired view.

        A missing task/model cell is filled with False, not coerced. Calling
        `.astype(bool)` directly on the pivot would turn NaN into True and
        silently inflate every downstream paired statistic.

        Cells only go missing when the frame has been filtered (see
        `exclude_zero_tests`). On a filtered frame the pairing is incomplete by
        construction, so paired statistics computed from this matrix -- McNemar,
        the oracle bound, LOOCV -- are not meaningful. Use `missing_mask()` to
        check before relying on them.
        """
        pivot = self.runs.pivot_table(
            index="task_id", columns="model_name", values="pass_fail", aggfunc="first"
        )
        return pivot.fillna(False).astype(bool)

    def missing_mask(self) -> pd.DataFrame:
        """True where a task/model cell has no run. All False on the full data."""
        pivot = self.runs.pivot_table(
            index="task_id", columns="model_name", values="pass_fail", aggfunc="first"
        )
        return pivot.isna()

    def is_fully_paired(self) -> bool:
        """Whether every model attempted every task, as paired tests require."""
        return not bool(self.missing_mask().to_numpy().any())

    def attempt_counts(self) -> pd.Series:
        """Runs per model. Denominators must be reported whenever they differ."""
        return self.runs.groupby("model_name").size()

    def overall_pass_rates(self) -> pd.Series:
        return self.runs.groupby("model_name")["pass_fail"].mean().sort_values(ascending=False)

    def rate_by(self, dimension: str) -> pd.DataFrame:
        """Pass rate as models x levels of `dimension` (category or difficulty)."""
        table = self.runs.pivot_table(
            index="model_name", columns=dimension, values="pass_fail", aggfunc="mean"
        )
        order = CATEGORIES if dimension == "category" else DIFFICULTIES
        return table[[column for column in order if column in table.columns]]

    def task_metadata(self) -> pd.DataFrame:
        """One row per task: its category and difficulty."""
        return (
            self.runs[["task_id", "category", "difficulty"]]
            .drop_duplicates(subset="task_id")
            .set_index("task_id")
            .sort_index()
        )


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(_BOOL_TRUE)


def _from_database(url: str) -> Optional[pd.DataFrame]:
    try:
        import psycopg2
    except ModuleNotFoundError:
        logger.warning("psycopg2 not installed; falling back to CSV")
        return None

    query = """
        SELECT t.task_id, t.category, t.difficulty,
               m.name AS model_name,
               r.run_number, r.pass_fail, r.latency_total_ms,
               r.tokens_input, r.tokens_output, r.extraction_success,
               r.tests_passed, r.tests_total, r.ts_any_count,
               r.compiler_errors, r.failure_mode, r.timestamp
        FROM runs r
        JOIN tasks t ON t.id = r.task_id
        JOIN models m ON m.id = r.model_id
    """
    connection = None
    try:
        connection = psycopg2.connect(url, connect_timeout=20)
        frame = pd.read_sql_query(query, connection)
        logger.info("Loaded %s rows from PostgreSQL", len(frame))
        return frame
    except Exception as exc:  # noqa: BLE001 - any DB failure should fall back, not crash
        logger.warning("Database load failed (%s); falling back to CSV", type(exc).__name__)
        return None
    finally:
        if connection is not None:
            connection.close()


def _from_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No results found at {csv_path}. Run scripts/export_results.py first."
        )
    frame = pd.read_csv(csv_path)
    logger.info("Loaded %s rows from %s", len(frame), csv_path.name)
    return frame


def load_experiment_data(
    csv_path: Optional[Path] = None,
    prefer_database: bool = True,
    exclude_zero_tests: bool = False,
) -> ExperimentData:
    """Load Phase 5 results.

    `exclude_zero_tests` drops runs where the Docker harness reported no test
    count. Those runs were scored as failures in the published Phase 5 numbers,
    so the default keeps them; the flag exists so the report can state whether
    any conclusion depends on that treatment.
    """
    frame: Optional[pd.DataFrame] = None
    source = "csv"

    database_url = get_database_url()
    if prefer_database and database_url:
        frame = _from_database(database_url)
        if frame is not None:
            source = "postgresql"

    if frame is None:
        path = Path(csv_path) if csv_path else (get_paths()["exports_dir"] / "results.csv")
        frame = _from_csv(path)
        source = f"csv:{path.name}"

    frame["pass_fail"] = _coerce_bool(frame["pass_fail"])
    frame["extraction_success"] = _coerce_bool(frame["extraction_success"])
    for column in _NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)

    zero_tests = int((frame["tests_total"] == 0).sum())
    if exclude_zero_tests:
        frame = frame[frame["tests_total"] > 0].copy()

    unexpected = set(frame["model_name"].unique()) - set(MODEL_PARAMS_B)
    if unexpected:
        logger.warning("Results contain models absent from MODEL_PARAMS_B: %s", sorted(unexpected))

    return ExperimentData(
        runs=frame.reset_index(drop=True),
        source=source,
        zero_test_rows=zero_tests,
        excluded_zero_tests=exclude_zero_tests,
    )


def configured_model_names() -> List[str]:
    return [model.name for model in MODELS]
