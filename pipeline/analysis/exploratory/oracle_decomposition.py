import csv, itertools
rows=list(csv.DictReader(open("/Users/venu/Downloads/Major Project/LLM_Comparison/LLM_Comparison/pipeline/exports/results_full.csv")))
PRICE={"Llama-3.1-8B":(0.05,0.08),"GPT-OSS-120B":(0.15,0.60),"Llama-4-Scout":(0.11,0.34),"Qwen3-32B":(0.29,0.59),"Llama-3.3-70B":(0.59,0.79)}
models=sorted(PRICE); tasks=sorted({r["task_id"] for r in rows})
P={};C={};CAT={};DIF={}
for r in rows:
    k=(r["task_id"],r["model_name"]); P[k]=r["pass_fail"].strip().lower() in("true","1","pass","passed")
    C[k]=(float(r["tokens_input"] or 0)*PRICE[r["model_name"]][0]+float(r["tokens_output"] or 0)*PRICE[r["model_name"]][1])/1e6
    CAT[r["task_id"]]=r["category"]; DIF[r["task_id"]]=r["difficulty"]
base=sum(C[(t,"GPT-OSS-120B")] for t in tasks)
# classify each task
cls={}
for t in tasks:
    solvers=[m for m in models if P[(t,m)]]
    if not solvers: cls[t]="UNSOLVABLE (no model)"
    elif P[(t,"Llama-3.1-8B")]: cls[t]="cheap-sufficient (8B solves)"
    elif any(P[(t,m)] for m in ["Llama-4-Scout","Llama-3.3-70B"]): cls[t]="mid-tier needed"
    else: cls[t]="top-tier only"
from collections import Counter,defaultdict
print("== task classes (n=65) ==")
for k,v in Counter(cls.values()).most_common(): print(f"  {k:30s} {v:2d}  ({v/65:.0%})")
print(f"\n== where the -78.8% oracle saving comes from (baseline total ${base:.6f}) ==")
sav=defaultdict(float); bcost=defaultdict(float)
for t in tasks:
    solvers=[(C[(t,m)],m) for m in models if P[(t,m)]]
    oc = min(solvers)[0] if solvers else min(C[(t,m)] for m in models)
    sav[cls[t]] += C[(t,"GPT-OSS-120B")]-oc; bcost[cls[t]]+=C[(t,"GPT-OSS-120B")]
tot=sum(sav.values())
for k in sorted(sav,key=lambda x:-sav[x]):
    print(f"  {k:30s} saves ${sav[k]:.6f}  = {sav[k]/tot:5.1%} of all saving  (baseline spend here ${bcost[k]:.6f})")
print(f"  {'TOTAL':30s}       ${tot:.6f}  = {tot/base:.1%} of baseline")
print("\n== is cheap-sufficiency predictable from category/difficulty? ==")
for field,name in [(CAT,"category"),(DIF,"difficulty")]:
    print(f"  by {name}:")
    g=defaultdict(lambda:[0,0])
    for t in tasks:
        g[field[t]][0]+= P[(t,"Llama-3.1-8B")]; g[field[t]][1]+=1
    for k,(s,n) in sorted(g.items(),key=lambda x:-x[1][0]/x[1][1]):
        print(f"    {k:14s} 8B solves {s:2d}/{n:2d} = {s/n:5.1%}")
print("\n== and unsolvable-task rate (pure waste to attempt) ==")
for field,name in [(DIF,"difficulty")]:
    g=defaultdict(lambda:[0,0])
    for t in tasks:
        g[field[t]][0]+= (cls[t]=="UNSOLVABLE (no model)"); g[field[t]][1]+=1
    for k,(s,n) in sorted(g.items(),key=lambda x:-x[1][0]/x[1][1]):
        print(f"    {k:14s} unsolvable {s:2d}/{n:2d} = {s/n:5.1%}")
