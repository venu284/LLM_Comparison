import sys,re,numpy as np,pandas as pd
ROOT=sys.argv[1];rng=np.random.default_rng(2)
long=pd.concat([pd.read_csv(f"{ROOT}/id_probing_results_long.csv"),pd.read_csv(f"{ROOT}/id_test_results_long.csv")])
long["cost_usd"]=long.cost_usd.fillna(0)
S0=long.pivot_table(index="task_id",columns="model",values="score")
C0=long.pivot_table(index="task_id",columns="model",values="cost_usd").reindex(S0.index)[S0.columns]
dim0=long.drop_duplicates("task_id").set_index("task_id")["dimension"].reindex(S0.index)
b=S0.isin([0.0,1.0]).all(axis=1)&dim0.isin(["code_generation","data_science","algorithm","code_completion","test_generation"])
def pidf(t):
    s=re.sub(r"^dim\d+_","",t)
    for pat,p in [(r"HumanEval[_/]?(\d+)","he"),(r"mbpp(?:_completion|_testgen)?_(\d+)","mbpp"),(r"BigCodeBench/(\d+)","bcb"),(r"lcb(?:_testgen)?_(.+)$","lcb")]:
        m=re.search(pat,s)
        if m:return p+m.group(1)
    return s
susp=lambda t: bool(re.search(r"^dim\d+_(lcb_|mbpp_testgen|he_testgen)",t))
def study(mask,label):
    S,C=S0[mask],C0[mask];pid=pd.Series({t:pidf(t) for t in S.index});probs=pid.unique()
    res=[]
    for rep in range(30):
        perm=rng.permutation(probs);n=len(perm)
        ev=pid.index[pid.isin(set(perm[:n//5]))];fit=pid.index[pid.isin(set(perm[2*n//5:]))]
        best=S.loc[fit].mean().idxmax();Se,Ce=S.loc[ev],C.loc[ev]
        for lad in (["Qwen3-Max","kimi-k2.5"],["Qwen3-Max","qwen3.5-plus"],["Qwen3-Max","claude-opus-4-6"]):
            acc=np.zeros(len(ev));cost=np.zeros(len(ev));done=np.zeros(len(ev),bool)
            for i,m in enumerate(lad):
                s=Se[m].values;cost+=np.where(done,0,Ce[m].values)
                take=~done&(((s==1)|(rng.random(len(ev))<0.1))|(i==len(lad)-1));acc=np.where(take,s,acc);done|=take
            df=pd.DataFrame({"g":pid.loc[ev].values,"d":acc-Se[best].values,"c":cost,"bc":Ce[best].values}).groupby("g").agg(d=("d","sum"),c=("c","sum"),bc=("bc","sum"),n=("d","size")).values
            bd=[];br=[]
            for _ in range(300):
                x=df[rng.integers(0,len(df),len(df))];bd.append(100*x[:,0].sum()/x[:,3].sum());br.append(100*(1-x[:,1].sum()/x[:,2].sum()))
            res.append(dict(ladder=">".join(lad),best=best,n=len(ev),gap=np.mean(bd),gap_lo=np.percentile(bd,2.5),hw=(np.percentile(bd,97.5)-np.percentile(bd,2.5))/2,sav=np.mean(br),sav_lo=np.percentile(br,2.5)))
    R=pd.DataFrame(res);print(f"\n== {label}: tasks {mask.sum()}, eval n ~{R.n.mean():.0f}, best single on fit: {R.best.value_counts().to_dict()}")
    print(R.groupby("ladder")[["gap","gap_lo","hw","sav","sav_lo"]].mean().round(2).to_string())
    rows=[]
    for lad,g in R.groupby("ladder"):
        for m in (1,2,3):
            for s in (10,20,30):rows.append(dict(ladder=lad,margin=-m,save=s,p=np.mean((g.gap_lo>=-m)&(g.sav_lo>=s))))
    print(pd.DataFrame(rows).pivot_table(index=["ladder","save"],columns="margin",values="p").round(2).to_string())
study(b,"all exec-binary tasks")
study(b&~pd.Series([susp(t) for t in S0.index],index=S0.index),"suspected-artefact families removed (lcb, lcb_testgen, mbpp/he_testgen)")
