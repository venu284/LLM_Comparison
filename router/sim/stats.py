"""Statistics for every claim-facing number (S2.4; CLAIMS S0.6, S0.8, R8, R12-R14).

What lives here:

- `rng(seed, stream)`: every random draw in `router/sim/` and `router/baselines/` comes from
  `np.random.SeedSequence([seed, stream_id])`, with the stream id taken from the table below;
- `bootstrap`: the paired bootstrap over tasks and the cluster bootstrap over problems
  (`problem_id`, from `problems.csv`). Both arms of a comparison are columns of one frame, so they
  share every resample. The cluster bootstrap is the one the gates use (S0.8);
- `across_seeds` and `threshold_label`: the five seeds with the gate seed marked (R12), and the
  "at the threshold" reporting label (S0.8, O12);
- `cross_fit`: the nested cross-fitting driver (S0.6, O7).

Stream ids (`SeedSequence([seed, id])`). Ids 0 and 1 belong to S1.6 (development split and
outer folds, `router/data/splits.py`), 100 and up to the S1.7 audit samples. S2 owns 2-99:

| id | stream | used by |
|---:|---|---|
| 2 | `stream_order` | task order of a replay (`sim/stream.py`) |
| 3 | `probe_draw` | calibration-probe draw for a new model (S0.13) |
| 4 | `verifier_noise` | simulated verifier noise, E3 sensitivity only (`sim/verifier.py`) |
| 5 | `random_baseline` | the Random baseline's choices |
| 6 | `bootstrap_task` | paired bootstrap over tasks |
| 7 | `bootstrap_cluster` | cluster bootstrap over problems |
| 8 | `tie_break` | random tie-breaks inside baselines |
| 9-99 | unassigned | add a row here before using one |

A seed is the fold assignment plus the stream order plus the probe draw (S0.8); `GATE_SEED` (0) is
the one gates use, and the other four are reported next to it, never averaged into a gate (R12).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from router.data import splits as splits_mod
from router.settings import (AT_THRESHOLD_MARGIN, CI_LEVEL, GATE_SEED, N_BOOT, N_FOLDS,
                             N_INNER_FOLDS, N_SEEDS, REPO_ROOT)

# ----------------------------------------------------------------------------------- stream ids
STREAM_IDS = {
    "stream_order": 2,
    "probe_draw": 3,
    "verifier_noise": 4,
    "random_baseline": 5,
    "bootstrap_task": 6,
    "bootstrap_cluster": 7,
    "tie_break": 8,
}
STREAM_ID_MIN, STREAM_ID_MAX = 2, 99
SEEDS = tuple(range(N_SEEDS))


def rng(seed: int, stream: str | int) -> np.random.Generator:
    """The generator for one named stream under one seed: `SeedSequence([seed, stream_id])`."""
    sid = STREAM_IDS[stream] if isinstance(stream, str) else int(stream)
    if not STREAM_ID_MIN <= sid <= STREAM_ID_MAX:
        raise ValueError(f"stream id {sid} is outside S2's range {STREAM_ID_MIN}-{STREAM_ID_MAX}")
    return np.random.default_rng(np.random.SeedSequence([int(seed), sid]))


# ------------------------------------------------------------------------------------ bootstrap
Statistic = Callable[[pd.DataFrame, np.ndarray], float]


@dataclass(frozen=True)
class Interval:
    """A point estimate with its percentile bootstrap CI."""
    point: float
    lo: float
    hi: float
    level: float
    n_boot: int
    unit: str            # "task" or "problem"
    n_units: int
    n_rows: int
    seed: int
    draws: np.ndarray = field(repr=False, compare=False)


def bootstrap(frame: pd.DataFrame, statistic: Statistic, *, clusters: pd.Series | None = None,
              n_boot: int = N_BOOT, seed: int = GATE_SEED, level: float = CI_LEVEL) -> Interval:
    """Percentile bootstrap of `statistic(frame, weights)`.

    Each resample draws units with replacement and gives every row the number of times its unit was
    drawn as its weight. With `clusters=None` the unit is the row (the paired bootstrap over tasks,
    stream `bootstrap_task`); otherwise it is the row's cluster, e.g. its `problem_id` (the cluster
    bootstrap over problems, stream `bootstrap_cluster`, the one the gates use). Put both arms of a
    comparison in `frame` so they share every resample (paired). Drop the complete case before
    calling and report the counts dropped (S0.11, O2).
    """
    n = len(frame)
    if n == 0:
        raise ValueError("empty frame")
    if clusters is None:
        codes, unit, stream = np.arange(n), "task", "bootstrap_task"
    else:
        c = clusters.reindex(frame.index)
        if c.isna().any():
            raise ValueError(f"{int(c.isna().sum())} rows have no cluster")
        codes, _ = pd.factorize(c)
        unit, stream = "problem", "bootstrap_cluster"
    k = int(codes.max()) + 1
    g = rng(seed, stream)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        counts = np.bincount(g.integers(0, k, k), minlength=k)
        draws[b] = statistic(frame, counts[codes].astype(float))
    tail = (1 - level) / 2 * 100
    lo, hi = np.nanpercentile(draws, [tail, 100 - tail])
    return Interval(float(statistic(frame, np.ones(n))), float(lo), float(hi), level, n_boot, unit,
                    k, n, seed, draws)


def rate(col: str, scale: float = 100.0) -> Statistic:
    """Weighted mean of one column, in points by default (e.g. the acceptance rate)."""
    return lambda f, w: scale * float(np.sum(w * f[col].to_numpy(float)) / np.sum(w))


def mean_diff(a: str, b: str, scale: float = 100.0) -> Statistic:
    """Paired difference of two weighted means, a - b, in points (e.g. an acceptance gap)."""
    return lambda f, w: scale * float(
        np.sum(w * (f[a].to_numpy(float) - f[b].to_numpy(float))) / np.sum(w))


def saving_pct(cost: str, base_cost: str) -> Statistic:
    """Cost saving of `cost` against `base_cost`, percent, on resampled totals:
    100 * (1 - sum(w * cost) / sum(w * base_cost)). Never a mean of per-task ratios."""
    return lambda f, w: 100.0 * (1.0 - float(np.sum(w * f[cost].to_numpy(float)))
                                 / float(np.sum(w * f[base_cost].to_numpy(float))))


# ---------------------------------------------------------------------------------------- seeds
def across_seeds(fn: Callable[[int], Interval], seeds: Sequence[int] = SEEDS,
                 gate_seed: int = GATE_SEED) -> pd.DataFrame:
    """One row per seed (point, CI, gate flag). The gate reads the `gate` row only (R12); the
    other rows are reported next to it, and their mean is never a gate quantity."""
    rows = []
    for s in seeds:
        iv = fn(s)
        rows.append({"seed": s, "gate": s == gate_seed, "point": iv.point, "lo": iv.lo,
                     "hi": iv.hi, "unit": iv.unit, "n_units": iv.n_units})
    if gate_seed not in seeds:
        raise ValueError("the gate seed must be among the seeds")
    return pd.DataFrame(rows).set_index("seed")


CLEARS, AT_THRESHOLD, DOES_NOT_CLEAR = "clears", "at the threshold", "does not clear"


def threshold_label(points: Mapping[int, float], gate_lower: float, threshold: float, *,
                    strict: bool = False, gate_seed: int = GATE_SEED,
                    margin: float = AT_THRESHOLD_MARGIN) -> str:
    """S0.8 reporting label for a quantity whose larger values are better.

    The gate seed's point estimate clears `threshold` (strictly when `strict`). It is labelled
    "at the threshold" when another seed's point is at or below the threshold, when the gate
    seed's CI lower bound is at or below it, or when the margin is under `margin` points
    (percentage points for savings). A label only: it never changes a verdict (O12).
    """
    p = points[gate_seed]
    if not (p > threshold if strict else p >= threshold):
        return DOES_NOT_CLEAR
    others = [v for s, v in points.items() if s != gate_seed]
    if any(v <= threshold for v in others) or gate_lower <= threshold or p - threshold < margin:
        return AT_THRESHOLD
    return CLEARS


# ------------------------------------------------------------------------- nested cross-fitting
class Fitted(Protocol):
    def score(self, tasks: pd.Index) -> pd.DataFrame:
        """One row per task in `tasks` (index = task id): whatever the selection rule and the
        metrics need. Must not refit."""


class Candidate(Protocol):
    name: str

    def fit(self, train_tasks: pd.Index) -> Fitted:
        """Fit on `train_tasks` only."""


SelectionRule = Callable[[Mapping[str, pd.DataFrame]], str]


@dataclass
class CrossFitResult:
    scores: pd.DataFrame            # one row per scored task: outer_fold, candidate, then score columns
    selected: dict[int, str]        # outer fold -> chosen candidate name
    fits: list[dict]                # every fit: stage, outer_fold, inner_fold, candidate, problem sets
    seed: int
    candidates: tuple[str, ...]
    frozen_tag: str | None


class EvaluationPopulationRefused(PermissionError):
    """Raised when the driver is asked to score the committed evaluation population without the
    frozen candidate list and its git tag (S0.6, S0.7, R14)."""


def committed_eval_problems() -> set[str]:
    """Problem ids in the committed evaluation population (`splits.csv`, split == eval)."""
    s = splits_mod.load_splits()
    return set(s.loc[s.split == splits_mod.EVAL, "problem_id"])


def touches_evaluation_population(task_ids, problems: pd.DataFrame) -> bool:
    """True when any task maps, through the committed `problems.csv` or through `problems`, to a
    problem in the committed evaluation population. Keyed on the committed files, so a substitute
    split table cannot hide a real evaluation task."""
    ev = committed_eval_problems()
    tasks = set(task_ids)
    committed = splits_mod.load_problems()
    via_committed = committed.loc[committed.task_id.isin(tasks), "problem_id"]
    via_given = problems.loc[problems.task_id.isin(tasks), "problem_id"]
    return bool(set(via_committed) & ev or set(via_given) & ev)


def _tag_resolves(tag: str) -> bool:
    r = subprocess.run(["/usr/bin/git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    return r.returncode == 0


def cross_fit(task_ids, candidates: Sequence[Candidate], select: SelectionRule, *,
              seed: int = GATE_SEED, problems: pd.DataFrame | None = None,
              splits: pd.DataFrame | None = None, frozen: bool = False,
              tag: str | None = None) -> CrossFitResult:
    """Nested cross-fitting (S0.6, O7) over the tasks `task_ids`, under fold seed `seed`.

    For each outer fold k: the inner folds are the other `N_FOLDS - 1` outer folds. Each candidate
    is fitted on three of them and scored on the fourth, for each of the four; `select` sees every
    candidate's inner scores and names one; that candidate is refitted on all four and scores fold
    k once. Fold k plays no part in choosing or fitting its configuration.

    `problems` and `splits` default to the committed `problems.csv` and `splits.csv`; tests pass toy
    tables. `splits.r13_violations` runs first and any violation stops the run. The driver refuses
    tasks of the committed evaluation population unless `frozen=True` and `tag` names a git tag
    that exists (the frozen candidate list and selection rule, S0.6). S2 never passes them.
    """
    problems = splits_mod.load_problems() if problems is None else problems
    splits = splits_mod.load_splits() if splits is None else splits
    task_ids = pd.Index(task_ids)
    if task_ids.has_duplicates:
        raise ValueError("duplicate task ids")
    if touches_evaluation_population(task_ids, problems):
        if not frozen or not tag:
            raise EvaluationPopulationRefused(
                "these tasks are in the committed evaluation population; pass frozen=True and the "
                "git tag of the frozen candidate list (S0.6, R14)")
        if not _tag_resolves(tag):
            raise EvaluationPopulationRefused(f"git tag {tag!r} does not exist")
    viol = splits_mod.r13_violations(problems, splits)
    if viol:
        raise ValueError(f"R13 violations in the split tables: {viol[:3]} ...")

    col = splits_mod.fold_column(seed)
    t = splits_mod.task_splits(problems, splits).set_index("task_id")
    missing = task_ids.difference(t.index)
    if len(missing):
        raise KeyError(f"{len(missing)} tasks are not in the problems table, e.g. {list(missing[:3])}")
    t = t.loc[task_ids]
    if (t.split != splits_mod.EVAL).any() or t[col].isna().any():
        raise ValueError("every task must have an outer fold (split == eval) under this seed")
    fold = t[col].astype(int)
    problem = t["problem_id"]
    folds = sorted(fold.unique())
    if folds != list(range(N_FOLDS)):
        raise ValueError(f"expected outer folds 0..{N_FOLDS - 1}, got {folds}")
    names = [c.name for c in candidates]
    if len(set(names)) != len(names) or not names:
        raise ValueError("candidate names must be unique and non-empty")
    by_name = dict(zip(names, candidates))

    fits: list[dict] = []

    def fit_and_score(cand, train_mask, test_mask, stage, k, j):
        train, test = task_ids[train_mask], task_ids[test_mask]
        train_p, test_p = set(problem[train_mask]), set(problem[test_mask])
        if train_p & test_p:
            raise AssertionError("a problem is on both sides of a fit")
        res = cand.fit(train).score(test)
        if not res.index.sort_values().equals(test.sort_values()):
            raise ValueError(f"candidate {cand.name} must score exactly the tasks it is given")
        fits.append({"stage": stage, "outer_fold": k, "inner_fold": j, "candidate": cand.name,
                     "train_problems": frozenset(train_p), "scored_problems": frozenset(test_p)})
        return res.loc[test]

    out, selected = [], {}
    f = fold.to_numpy()
    for k in folds:
        rest = [x for x in folds if x != k]
        if len(rest) != N_INNER_FOLDS:
            raise ValueError("inner folds must be the other outer folds (O7)")
        inner = {}
        for cand in candidates:
            parts = []
            for j in rest:
                train = np.isin(f, [x for x in rest if x != j])
                parts.append(fit_and_score(cand, train, f == j, "inner", k, j))
            inner[cand.name] = pd.concat(parts)
        chosen = select(inner)
        if chosen not in by_name:
            raise ValueError(f"selection rule returned unknown candidate {chosen!r}")
        selected[k] = chosen
        res = fit_and_score(by_name[chosen], np.isin(f, rest), f == k, "outer", k, None)
        out.append(res.assign(outer_fold=k, candidate=chosen))
    scores = pd.concat(out)
    scores = scores[["outer_fold", "candidate"] + [c for c in scores.columns
                                                    if c not in ("outer_fold", "candidate")]]
    return CrossFitResult(scores, selected, fits, seed, tuple(names), tag if frozen else None)
