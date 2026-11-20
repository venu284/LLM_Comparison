import sys, numpy as np, pandas as pd, itertools, re
ROOT=sys.argv[1]; rng=np.random.default_rng(1)
long=pd.concat([pd.read_csv(f"{ROOT}/id_probing_results_long.csv"),pd.read_csv(f"{ROOT}/id_test_results_long.csv")])
long["cost_usd"]=long.cost_usd.fillna(0)
S=long.pivot_table(index="task_id",columns="model",values="score")
C=long.pivot_table(index="task_id",columns="model",values="cost_usd").reindex(S.index)[S.columns]
dim=long.drop_duplicates("task_id").set_index("task_id")["dimension"].reindex(S.index)
b=S.isin([0.0,1.0]).all(axis=1)&dim.isin(["code_generation","data_science","algorithm","code_completion","test_generation"])
S,C,dim=S[b],C[b],dim[b]
t=pd.DataFrame({"accept%":100*S.mean(),"mean_cost_$":C.mean(),"$per1000":1000*C.mean()}).sort_values("accept%",ascending=False)
print(t.round(4).to_string())
best="qwen3.5-plus"; ba=S[best].mean(); bc=C[best].sum()
solved=S.max(axis=1)
print(f"\nunsolved by all: {100*(solved==0).mean():.1f}%  oracle accept {100*solved.mean():.1f}%")
# oracle cheapest-passing cost
cheapest=np.where(solved==1,(C.where(S==1)).min(axis=1),C.min(axis=1))
print(f"oracle cheapest-pass cost vs best single: {100*(1-cheapest.sum()/bc):.1f}% saving")
# all 2- and 3-rung ladders, perfect and eps=0.1 one-sided verifier
def run(ladder,eps):
    acc=np.zeros(len(S));cost=np.zeros(len(S));done=np.zeros(len(S),bool)
    for i,m in enumerate(ladder):
        s=S[m].values;c=C[m].values;cost+=np.where(done,0,c)
        ver=(s==1)|(rng.random(len(S))<eps);take=~done&(ver|(i==len(ladder)-1))
        acc=np.where(take,s,acc);done|=take
    return 100*acc.mean()-100*ba,100*(1-cost.sum()/bc)
rows=[]
ms=list(S.columns)
for k in (2,3):
    for lad in itertools.permutations(ms,k):
        if C[list(lad)].mean().is_monotonic_increasing:
            for eps in (0.0,0.1):
                g,s=run(lad,eps);rows.append(dict(ladder=">".join(lad),eps=eps,gap=g,saving=s))
R=pd.DataFrame(rows)
cand=R[(R.eps==0.1)&(R.gap>=-1)].sort_values("saving",ascending=False).head(8)
print("\nbest ladders at eps=0.1 with gap >= -1 (whole binary set):");print(cand.round(2).to_string(index=False))
print("\nsame ladders at eps=0:");print(R[(R.eps==0)&R.ladder.isin(cand.ladder)].round(2).to_string(index=False))
