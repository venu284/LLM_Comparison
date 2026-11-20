import sys,re,numpy as np,pandas as pd
exec(open("power3.py").read().split("def study")[0].split("ROOT=sys.argv[1];rng=np.random.default_rng(2)")[1].join(["",""]) if False else "")
ROOT=sys.argv[1];rng=np.random.default_rng(3)
long=pd.concat([pd.read_csv(f"{ROOT}/id_probing_results_long.csv"),pd.read_csv(f"{ROOT}/id_test_results_long.csv")]);long["cost_usd"]=long.cost_usd.fillna(0)
S0=long.pivot_table(index="task_id",columns="model",values="score");C0=long.pivot_table(index="task_id",columns="model",values="cost_usd").reindex(S0.index)[S0.columns]
dim0=long.drop_duplicates("task_id").set_index("task_id")["dimension"].reindex(S0.index)
b=S0.isin([0.0,1.0]).all(axis=1)&dim0.isin(["code_generation","data_science","algorithm","code_completion","test_generation"])
def pidf(t):
    s=re.sub(r"^dim\d+_","",t)
    for pat,p in [(r"HumanEval[_/]?(\d+)","he"),(r"mbpp(?:_completion|_testgen)?_(\d+)","mbpp"),(r"BigCodeBench/(\d+)","bcb"),(r"lcb(?:_testgen)?_(.+)$","lcb")]:
        m=re.search(pat,s)
        if m:return p+m.group(1)
    return s
susp=pd.Series([bool(re.search(r"^dim\d+_(lcb_|mbpp_testgen|he_testgen)",t)) for t in S0.index],index=S0.index)
def cf(mask,label,K=5,reps=20):
    S,C=S0[mask],C0[mask];pid=pd.Series({t:pidf(t) for t in S.index});probs=pid.unique();out=[]
    for r in range(reps):
        fold=dict(zip(rng.permutation(probs),np.arange(len(probs))%K));f=pid.map(fold)
        for lad in (["Qwen3-Max","kimi-k2.5"],["Qwen3-Max","claude-opus-4-6"]):
            d=np.zeros(len(S));c=np.zeros(len(S));bc=np.zeros(len(S))
            for k in range(K):
                tr=(f!=k).values;te=(f==k).values;best=S[tr].mean().idxmax()
                Se,Ce=S[te],C[te];acc=np.zeros(te.sum());cost=np.zeros(te.sum());done=np.zeros(te.sum(),bool)
                for i,m in enumerate(lad):
                    s=Se[m].values;cost+=np.where(done,0,Ce[m].values);take=~done&(((s==1)|(rng.random(te.sum())<0.1))|(i==len(lad)-1));acc=np.where(take,s,acc);done|=take
                d[te]=acc-Se[best].values;c[te]=cost;bc[te]=Ce[best].values
            A=pd.DataFrame({"g":pid.values,"d":d,"c":c,"bc":bc}).groupby("g").agg(d=("d","sum"),c=("c","sum"),bc=("bc","sum"),n=("d","size")).values
            bd=[];br=[]
            for _ in range(300):
                x=A[rng.integers(0,len(A),len(A))];bd.append(100*x[:,0].sum()/x[:,3].sum());br.append(100*(1-x[:,1].sum()/x[:,2].sum()))
            out.append(dict(ladder=">".join(lad),gap=np.mean(bd),gap_lo=np.percentile(bd,2.5),hw=(np.percentile(bd,97.5)-np.percentile(bd,2.5))/2,sav=np.mean(br),sav_lo=np.percentile(br,2.5)))
    R=pd.DataFrame(out);print(f"\n== {label}, 5-fold cross-fitting, n={mask.sum()}");print(R.groupby("ladder").mean().round(2).to_string())
    print("margin for 80% power at true gap 0:", (1.43*R.groupby('ladder').hw.mean()).round(2).to_dict())
cf(b,"all exec-binary");cf(b&~susp,"suspected artefacts removed")
