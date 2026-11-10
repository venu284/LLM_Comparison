"""M3 analyses: IRT model choice (A4), ability error against k (A6) and the ranking check (A7).

Run from the repo root:  pipeline/venv/bin/python -m router.experiments.m3_irt
Outcomes are binarised at score >= 0.5 (router.data.scoring.ACCEPT_THRESHOLD).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from router.core import irt
from router.core.anchors import onboard
from router.data import coderouterbench as crb
from router.data.scoring import ACCEPT_THRESHOLD, score_types
from router.settings import REPO_ROOT

OUT = REPO_ROOT / "router" / "experiments" / "out"
KS = (5, 10, 25, 50, 100)
SEEDS = range(10)


def binarise(df: pd.DataFrame) -> np.ndarray:
    return (df.T.to_numpy() >= ACCEPT_THRESHOLD).astype(float)     # models x items


def a4(probe: crb.OutcomeMatrix) -> pd.DataFrame:
    types = score_types(probe.score, probe.dimension)
    binary_dims = set(types.index[types.binary_share >= 0.9])
    rows = []
    for subset in ("all dimensions", "binary-like dimensions"):
        cols = probe.tasks if subset == "all dimensions" else probe.tasks[probe.dimension.isin(binary_dims).to_numpy()]
        Y = binarise(probe.score.loc[cols])
        for seed in range(3):
            rng = np.random.default_rng(seed)
            test_mask = rng.random(Y.shape) < 0.1
            Ytr = Y.copy()
            Ytr[test_mask] = np.nan
            for kind in ("1pl", "2pl"):
                p = irt.fit(Ytr, cols, probe.models, kind)
                rows.append({"subset": subset, "model": kind, "seed": seed,
                             "heldout_ll": irt.heldout_loglik(p, Y, test_mask)})
            # per-dimension ability: one 1PL per dimension
            ll, n = 0.0, 0
            for d in probe.dimension.loc[cols].unique():
                sel = (probe.dimension.loc[cols] == d).to_numpy()
                p = irt.fit(Ytr[:, sel], cols[sel], probe.models, "1pl")
                tm = test_mask[:, sel]
                ll += irt.heldout_loglik(p, Y[:, sel], tm) * tm.sum()
                n += tm.sum()
            rows.append({"subset": subset, "model": "1pl per-dimension theta", "seed": seed, "heldout_ll": ll / n})
        # base rate reference: predict each model's training pass rate
        rates = np.nanmean(Ytr, axis=1, keepdims=True)
        q = np.clip(np.broadcast_to(rates, Y.shape)[test_mask], 1e-12, 1 - 1e-12)
        y = Y[test_mask]
        rows.append({"subset": subset, "model": "per-model pass rate (no items)", "seed": 2,
                     "heldout_ll": float((y * np.log(q) + (1 - y) * np.log(1 - q)).mean())})
    df = pd.DataFrame(rows).groupby(["subset", "model"]).heldout_ll.agg(["mean", "std"]).reset_index()
    df.to_csv(OUT / "a4_irt_choice.csv", index=False)
    types.to_csv(OUT / "a4_score_types.csv")
    return df


def a6_a7(probe: crb.OutcomeMatrix, test: crb.OutcomeMatrix, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = probe.tasks.append(test.tasks)
    Yall = binarise(pd.concat([probe.score, test.score]))
    n_probe = len(probe)
    probe_idx, test_idx = np.arange(n_probe), np.arange(n_probe, len(items))
    rows, rank_rows = [], []
    for mi, m in enumerate(probe.models):
        keep = [i for i in range(len(probe.models)) if i != mi]
        params = irt.fit(Yall[keep], items, [probe.models[i] for i in keep], kind)
        y_m = Yall[mi]
        th_full, _ = irt.ability_posterior(params, probe_idx, y_m[probe_idx])
        y_test = y_m[test_idx]

        def score_theta(th):
            p = np.clip(params.p(th, test_idx), 1e-6, 1 - 1e-6)
            return (roc_auc_score(y_test, p),
                    float((y_test * np.log(p) + (1 - y_test) * np.log(1 - p)).mean()))

        auc_full, ll_full = score_theta(th_full)
        auc0, ll0 = score_theta(0.0)
        rows.append({"model": m, "k": 0, "method": "prior only", "seed": 0, "theta": 0.0,
                     "abs_err": abs(th_full), "auc": auc0, "ll": ll0})
        rows.append({"model": m, "k": n_probe, "method": "full probing set", "seed": 0, "theta": th_full,
                     "abs_err": 0.0, "auc": auc_full, "ll": ll_full})
        for k in KS:
            for method in ("fisher", "random"):
                for s in SEEDS:
                    th, sd, _ = onboard(params, k, lambda i: y_m[i], method, seed=s, pool=probe_idx)
                    auc, ll = score_theta(th)
                    rows.append({"model": m, "k": k, "method": method, "seed": s, "theta": th,
                                 "sd": sd, "abs_err": abs(th - th_full), "auc": auc, "ll": ll})
        rank_rows.append({"model": m, "pass_rate_probing": y_m[probe_idx].mean() * 100, "theta_full": th_full})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"a6_onboarding_{kind}.csv", index=False)
    rank = pd.DataFrame(rank_rows).set_index("model")
    k50 = df[(df.k == 50)].groupby(["model", "method"]).theta.mean().unstack()
    rank["theta_k50_fisher"] = k50["fisher"]
    rank["theta_k50_random"] = k50["random"]
    rank.to_csv(OUT / f"a7_ranking_{kind}.csv")
    return df, rank


def ci(x: pd.Series) -> str:
    rng = np.random.default_rng(0)
    v = x.to_numpy()
    d = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    return f"{v.mean():.3f} [{np.percentile(d, 2.5):.3f}, {np.percentile(d, 97.5):.3f}]"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    probe, test = crb.load("probing"), crb.load("id_test")
    print("== A4 held-out log-likelihood per cell (10% of cells held out, 3 seeds)")
    print(a4(probe).round(4).to_string(index=False))
    for kind in ("1pl", "2pl"):
        df, rank = a6_a7(probe, test, kind)
        print(f"\n== A6 ({kind}): |theta_k - theta_full| and test-task AUC, mean over 8 held-out models x 10 seeds")
        on = df[df.method.isin(["fisher", "random"])]
        tab = on.groupby(["k", "method"]).agg(err=("abs_err", ci), auc=("auc", ci)).unstack()
        print(tab.to_string())
        ref = df[df.method.isin(["prior only", "full probing set"])].groupby("method")[["abs_err", "auc"]].mean()
        print(ref.round(3).to_string())
        print(f"\n== A7 ({kind}) ranking")
        print(rank.round(3).to_string())
        for col in ("theta_full", "theta_k50_fisher", "theta_k50_random"):
            print(col, "Spearman vs pass rate:", round(spearmanr(rank.pass_rate_probing, rank[col]).statistic, 3))
