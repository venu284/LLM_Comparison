"""S2 checkpoint A: what the development split holds in each data view.

Every S2 number is computed on the development split (S0.6, S0.7, R14). This counts, for each of
the four S1.7 views (all eligible, artefact-filtered, sensitivity, headline), the development
split's tasks, problems, families and complete-case tasks (no unavailable cell, S0.11). It reads
no score and touches no evaluation-fold task beyond the view sizes.

Run from the repo root:  pipeline/venv/bin/python -m router.experiments.s2_dev_composition
Writes router/experiments/out/s2_dev_composition.csv and s2_dev_composition_by_family.csv.
"""

from __future__ import annotations

import pandas as pd

from router.data import splits as S
from router.data.audit import VIEW_NAMES, build_views, load_release
from router.settings import REPO_ROOT

OUT = REPO_ROOT / "router" / "experiments" / "out"


def composition() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = load_release()
    views = build_views(m)
    ts = S.task_splits().set_index("task_id")
    dev = set(ts.index[ts.split == S.DEV])
    rows, fam_rows = [], []
    for name in VIEW_NAMES:
        v = views[name]
        d = [t for t in v.tasks if t in dev]
        sub = v.subset(d)
        cc = sub.available.all(axis=1)
        fam = sub.family
        rows.append({"view": name, "view_tasks": len(v), "dev_tasks": len(d),
                     "dev_problems": ts.loc[d, "problem_id"].nunique(),
                     "dev_families": fam.nunique(), "dev_complete_case": int(cc.sum()),
                     "dev_dropped_unavailable": int((~cc).sum())})
        for f, g in fam.groupby(fam):
            fam_rows.append({"view": name, "family": f, "dev_tasks": len(g),
                             "dev_problems": ts.loc[g.index, "problem_id"].nunique(),
                             "dev_complete_case": int(cc[g.index].sum())})
    return pd.DataFrame(rows), pd.DataFrame(fam_rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    summary, by_family = composition()
    summary.to_csv(OUT / "s2_dev_composition.csv", index=False, lineterminator="\n")
    by_family.to_csv(OUT / "s2_dev_composition_by_family.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print(by_family[by_family.view.isin(["filtered", "headline"])].to_string(index=False))
