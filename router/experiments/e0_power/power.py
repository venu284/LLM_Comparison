"""Power calculation for the S0.3 performance gate on CodeRouterBench.

Eval split = 20% of problems (S0.6). Acceptance = binary exec score on exec-scored tasks.
Candidate routers are simple stand-ins for what we will build:
  - fixed cascade Qwen3-Max -> Opus with a one-sided noisy verifier (eps false-pass)
  - 3-rung cascade Qwen3-Max -> GPT-5.4 -> Opus
  - DimensionBest (per-dimension best model, fitted on the fit split)
Baseline = best single model by acceptance on the fit split.
"""
import re, sys
import numpy as np, pandas as pd

ROOT = sys.argv[1]
rng = np.random.default_rng(0)
frames = [pd.read_csv(f"{ROOT}/id_probing_results_long.csv"), pd.read_csv(f"{ROOT}/id_test_results_long.csv")]
long = pd.concat(frames)
long["cost_usd"] = long["cost_usd"].fillna(0)
S = long.pivot_table(index="task_id", columns="model", values="score")
C = long.pivot_table(index="task_id", columns="model", values="cost_usd").reindex(S.index)[S.columns]
dim = long.drop_duplicates("task_id").set_index("task_id")["dimension"].reindex(S.index)

binary_dims = ["code_generation", "data_science", "algorithm", "code_completion", "test_generation"]
is_bin = S.isin([0.0, 1.0]).all(axis=1) & dim.isin(binary_dims)
S, C, dim = S[is_bin], C[is_bin], dim[is_bin]
print(f"exec-scored binary tasks (all ID): {len(S)}")
print(dim.value_counts().to_string())

def problem_id(t):
    s = re.sub(r"^dim\d+_", "", t)
    m = re.search(r"HumanEval[_/]?(\d+)", s)
    if m: return f"he{m.group(1)}"
    m = re.search(r"mbpp(?:_completion|_testgen)?_(\d+)", s)
    if m: return f"mbpp{m.group(1)}"
    m = re.search(r"BigCodeBench/(\d+)", s)
    if m: return f"bcb{m.group(1)}"
    m = re.search(r"lcb(?:_testgen)?_(.+)$", s)
    if m: return f"lcb{m.group(1)}"
    return s
pid = pd.Series({t: problem_id(t) for t in S.index})
print(f"distinct problems: {pid.nunique()} (tasks per problem mean {len(pid)/pid.nunique():.2f})")

def policies(fit, ev, eps):
    Sf, Se, Ce = S.loc[fit], S.loc[ev], C.loc[ev]
    best = Sf.mean().idxmax()
    out = {"best": (Se[best].values, Ce[best].values)}
    # DimensionBest fitted on fit split
    db = Sf.groupby(dim.loc[fit]).mean().idxmax(axis=1)
    pick = dim.loc[ev].map(db)
    out["DimensionBest"] = (np.array([Se.at[t, m] for t, m in pick.items()]),
                            np.array([Ce.at[t, m] for t, m in pick.items()]))
    for name, ladder in [("Qwen3-Max>Opus", ["Qwen3-Max", "claude-opus-4-6"]),
                         ("Qwen3-Max>GPT-5.4>Opus", ["Qwen3-Max", "gpt-5.4", "claude-opus-4-6"])]:
        acc = np.zeros(len(ev)); cost = np.zeros(len(ev)); done = np.zeros(len(ev), bool)
        for i, m in enumerate(ladder):
            s = Se[m].values; c = Ce[m].values
            cost += np.where(done, 0, c)
            last = i == len(ladder) - 1
            # one-sided verifier: a wrong answer passes verification with prob eps
            verified = (s == 1) | (rng.random(len(ev)) < eps)
            take = ~done & (verified | last)
            acc = np.where(take, s, acc); done |= take
        out[name] = (acc, cost)
    return best, out

probs = pid.unique()
reps = []
for rep in range(40):
    perm = rng.permutation(probs)
    n = len(perm); ev_p = set(perm[: int(0.2 * n)]); fit_p = set(perm[int(0.4 * n):])
    ev = pid.index[pid.isin(ev_p)]; fit = pid.index[pid.isin(fit_p)]
    best, out = policies(fit, ev, eps=0.1)
    b_acc, b_cost = out["best"]
    g = pid.loc[ev].values
    for name, (a, c) in out.items():
        if name == "best": continue
        d = a - b_acc
        # cluster bootstrap over problems
        df = pd.DataFrame({"g": g, "d": d, "c": c, "bc": b_cost})
        agg = df.groupby("g").agg(d=("d", "sum"), c=("c", "sum"), bc=("bc", "sum"), n=("d", "size"))
        bs_d, bs_r = [], []
        A = agg.values
        for _ in range(400):
            idx = rng.integers(0, len(A), len(A)); x = A[idx]
            bs_d.append(x[:, 0].sum() / x[:, 3].sum()); bs_r.append(1 - x[:, 1].sum() / x[:, 2].sum())
        reps.append(dict(rep=rep, policy=name, best=best, n_eval=len(ev), gap=100 * d.mean(),
                         gap_lo=100 * np.percentile(bs_d, 2.5), gap_hi=100 * np.percentile(bs_d, 97.5),
                         discord=np.mean(d != 0), saving=100 * (1 - c.sum() / b_cost.sum()),
                         saving_lo=100 * np.percentile(bs_r, 2.5)))
R = pd.DataFrame(reps)
print("\nbest single model chosen on fit split:", R.best.value_counts().to_dict())
summ = R.groupby("policy").agg(n_eval=("n_eval", "mean"), gap=("gap", "mean"), gap_sd=("gap", "std"),
                               halfwidth=("gap_hi", lambda x: 0), discord=("discord", "mean"),
                               saving=("saving", "mean"), saving_lo=("saving_lo", "mean"))
summ["halfwidth"] = ((R.gap_hi - R.gap_lo) / 2).groupby(R.policy).mean()
summ["gap_lo_mean"] = R.groupby("policy").gap_lo.mean()
print(summ.round(2).to_string())
print("\nP(gate passes) over 40 random 60/20/20 splits:")
for m in [1, 2, 3, 4, 5]:
    for s in [15, 20, 25, 30]:
        pass
rows = []
for pol, g in R.groupby("policy"):
    for m in [1, 2, 3, 4]:
        for s in [20, 30]:
            rows.append(dict(policy=pol, margin=-m, saving_req=s,
                             p_pass=np.mean((g.gap_lo >= -m) & (g.saving_lo >= s))))
P = pd.DataFrame(rows).pivot_table(index=["policy", "saving_req"], columns="margin", values="p_pass")
print(P.round(2).to_string())

# minimum detectable margin with 80% power for a router whose true gap is 0
for pol, g in R.groupby("policy"):
    hw = ((g.gap_hi - g.gap_lo) / 2).mean()
    print(f"{pol}: CI half-width {hw:.2f} pts; margin needed for 80% power at true gap 0 ~ {hw + 0.84 * hw / 1.96:.2f} pts")
