import csv, random, statistics
rows=list(csv.DictReader(open("/Users/venu/Downloads/Major Project/LLM_Comparison/LLM_Comparison/pipeline/exports/results_full.csv")))
models=sorted({r["model_name"] for r in rows}); tasks=sorted({r["task_id"] for r in rows})
P={};CAT={};DIF={}
for r in rows:
    P[(r["task_id"],r["model_name"])]=r["pass_fail"].strip().lower() in("true","1","pass","passed")
    CAT[r["task_id"]]=r["category"]; DIF[r["task_id"]]=r["difficulty"]
cats=sorted({CAT[t] for t in tasks}); difs=sorted({DIF[t] for t in tasks})
print(f"cells = {len(cats)} categories x {len(difs)} difficulties = {len(cats)*len(difs)}")

def descriptor(m, probe):
    """capability vector from probe tasks: pass rate per (category,difficulty) cell, backed off to marginals"""
    d={}
    for c in cats:
        for df in difs:
            s=[P[(t,m)] for t in probe if CAT[t]==c and DIF[t]==df]
            d[(c,df)]= sum(s)/len(s) if s else None
    # backoff
    gm=sum(P[(t,m)] for t in probe)/len(probe)
    bc={c:(lambda s: sum(s)/len(s) if s else gm)([P[(t,m)] for t in probe if CAT[t]==c]) for c in cats}
    bd={df:(lambda s: sum(s)/len(s) if s else gm)([P[(t,m)] for t in probe if DIF[t]==df]) for df in difs}
    for k in d:
        if d[k] is None: d[k]=(bc[k[0]]+bd[k[1]])/2
    return d

def evaluate(k, trials=200, seed=0):
    """sample k probe tasks; predict pass on the HELD-OUT rest; measure AUC-ish + routing quality"""
    rng=random.Random(seed); accs=[]; picks=[]
    for _ in range(trials):
        probe=rng.sample(tasks,k); held=[t for t in tasks if t not in probe]
        D={m:descriptor(m,probe) for m in models}
        # 1. prediction accuracy: predict pass if descriptor cell > 0.5
        corr=tot=0
        for t in held:
            for m in models:
                pred = D[m][(CAT[t],DIF[t])] > 0.5
                corr += (pred==P[(t,m)]); tot+=1
        accs.append(corr/tot)
        # 2. routing quality: pick argmax-descriptor model per held-out task, measure realised pass rate
        ok=0
        for t in held:
            best=max(models,key=lambda m:D[m][(CAT[t],DIF[t])])
            ok+=P[(t,best)]
        picks.append(ok/len(held))
    return statistics.mean(accs), statistics.mean(picks)

# reference points
oracle=sum(any(P[(t,m)] for m in models) for t in tasks)/len(tasks)
bestsingle=max(sum(P[(t,m)] for t in tasks)/len(tasks) for m in models)
print(f"reference: best-single={bestsingle:.1%}  oracle={oracle:.1%}\n")
print(f"{'k probe':>8} {'pred acc':>9} {'routed pass':>12}   (held-out tasks)")
for k in [3,5,8,10,15,20,25,30,40]:
    a,p=evaluate(k)
    bar="#"*int((a-0.5)*100) if a>0.5 else ""
    print(f"{k:8d} {a:9.1%} {p:12.1%}   {bar}")
