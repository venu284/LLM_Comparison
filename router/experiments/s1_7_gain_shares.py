"""S1.7 step 13: each family's share of the oracle's and in-sample DimensionBest's gain over the
best single model, in every data view, under the strict success rule and the CRUXEval 0.8
sensitivity. Run only after the labels and artefacts.yaml were committed (blind labelling).
Definitions: docs/research/S1.7_audit.md, "Checkpoint 3: gain shares per family".

    PYTHONPATH=.:pipeline pipeline/venv/bin/python -m router.experiments.s1_7_gain_shares
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.data import audit  # noqa: E402
from router.data.scoring import success  # noqa: E402

OUT = REPO_ROOT / "router/experiments/out/s1_7_gain_shares.csv"
RULES = {"strict": False, "cruxeval_lenient": True}


def success_matrix(m, lenient: bool) -> pd.DataFrame:
    fam = list(m.family.reindex(m.tasks))
    models = sorted(m.models)
    return pd.DataFrame({c: success(m.score[c].to_numpy(), fam, cruxeval_lenient=lenient)
                         for c in models}, index=m.tasks).astype(int)


def _argmax(s: pd.Series) -> str:
    """Most successes; ties broken by sorted model order."""
    return s[s == s.max()].index.sort_values()[0]


def shares(m, lenient: bool) -> tuple[pd.DataFrame, dict]:
    cc = m.subset(m.tasks[m.available.all(axis=1).to_numpy()])
    ok = success_matrix(cc, lenient)
    best = _argmax(ok.sum())
    base = ok[best]
    oracle = ok.max(axis=1)
    dim = cc.dimension.reindex(ok.index)
    table = {d: _argmax(ok[dim == d].sum()) for d in sorted(dim.unique())}
    db = pd.Series([ok.at[t, table[d]] for t, d in dim.items()], index=ok.index)
    fam = cc.family.reindex(ok.index)
    g = pd.DataFrame({"family": fam, "oracle": oracle - base, "dimension_best": db - base})
    per = g.groupby("family").agg(tasks=("oracle", "size"), oracle_gain=("oracle", "sum"),
                                  dimension_best_gain=("dimension_best", "sum"))
    tot_o, tot_d = per.oracle_gain.sum(), per.dimension_best_gain.sum()
    per["oracle_share"] = per.oracle_gain / tot_o if tot_o else float("nan")
    per["dimension_best_share"] = per.dimension_best_gain / tot_d if tot_d else float("nan")
    n = len(ok)
    summary = {"complete_case_tasks": n, "best_single_model": best,
               "best_single_rate": base.mean(), "oracle_rate": oracle.mean(),
               "dimension_best_rate": db.mean(), "oracle_gain_tasks": int(tot_o),
               "dimension_best_gain_tasks": int(tot_d)}
    return per.reset_index(), summary


def main() -> pd.DataFrame:
    m = audit.load_release()
    views = audit.build_views(m)
    rows = []
    for vname, vm in views.items():
        for rule, lenient in RULES.items():
            per, s = shares(vm, lenient)
            rows += [{"view": vname, "rule": rule, **r, **{f"view_{k}": v for k, v in s.items()}}
                     for r in per.to_dict("records")]
    out = pd.DataFrame(rows)
    out.round(4).to_csv(OUT, index=False, lineterminator="\n")
    return out


if __name__ == "__main__":
    t = main()
    pd.set_option("display.width", 250, "display.max_rows", 200)
    s = t.drop_duplicates(["view", "rule"])[["view", "rule"] + [c for c in t if c.startswith("view_")]]
    print(s.to_string(index=False))
    print(t[["view", "rule", "family", "tasks", "oracle_gain", "oracle_share", "dimension_best_gain",
             "dimension_best_share"]].round(3).to_string(index=False))
