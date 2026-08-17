import csv, itertools
rows=list(csv.DictReader(open("/Users/venu/Downloads/Major Project/LLM_Comparison/LLM_Comparison/pipeline/exports/results_full.csv")))
models=sorted({r["model_name"] for r in rows}); tasks=sorted({r["task_id"] for r in rows})
# Groq published $/M tokens (in,out). * = approximate, needs confirming.
PRICE={"Llama-3.1-8B":(0.05,0.08),"GPT-OSS-120B":(0.15,0.60),
       "Llama-4-Scout":(0.11,0.34),"Qwen3-32B":(0.29,0.59),"Llama-3.3-70B":(0.59,0.79)}
P={};TIN={};TOUT={};LAT={}
for r in rows:
    k=(r["task_id"],r["model_name"]); P[k]=r["pass_fail"].strip().lower() in("true","1","pass","passed")
    TIN[k]=float(r["tokens_input"] or 0); TOUT[k]=float(r["tokens_output"] or 0); LAT[k]=float(r["latency_total_ms"] or 0)
def price(k,pr):
    m=k[1]; return (TIN[k]*pr[m][0]+TOUT[k]*pr[m][1])/1e6
def cascade(order,pr):
    np_=0; tot=0.0; calls=0
    for t in tasks:
        c=0.0;k=0;ok=False
        for m in order:
            c+=price((t,m),pr); k+=1
            if P[(t,m)]: ok=True; break
        np_+=ok; tot+=c; calls+=k
    return np_/len(tasks), tot/len(tasks), calls/len(tasks)
print("-- per-task price, single model (Groq list prices) --")
for m in sorted(models,key=lambda m:PRICE[m][1]):
    pr_,c,_=cascade([m],PRICE); print(f"{m:16s} pass={pr_:6.1%}  ${c*1000:7.4f} per 1k tasks-unit  (${c:.8f}/task)")
bp,bc,_=cascade(["GPT-OSS-120B"],PRICE)
union=sum(any(P[(t,m)] for m in models) for t in tasks)/len(tasks)
print(f"\nBASELINE always-GPT-OSS-120B: {bp:.1%} @ ${bc:.8f}/task")
print(f"UNION ceiling: {union:.1%}")
allr=[]
for L in range(1,6):
    for o in itertools.permutations(models,L):
        p_,c,k=cascade(list(o),PRICE); allr.append((p_,c,k,list(o)))
print("\n== chains with pass >= baseline 53.8%, cheapest first ==")
print(f"{'pass':>7} {'$/task':>12} {'vs base':>9} {'calls':>6}  chain")
for p_,c,k,o in sorted([x for x in allr if x[0]>=bp-1e-9],key=lambda x:x[1])[:12]:
    d=(c-bc)/bc*100
    print(f"{p_:7.1%} {c:12.8f} {d:+8.1f}% {k:6.2f}  {' -> '.join(o)}")
print("\n== chains hitting UNION 58.5%, cheapest first ==")
for p_,c,k,o in sorted([x for x in allr if x[0]>=union-1e-9],key=lambda x:x[1])[:6]:
    d=(c-bc)/bc*100
    print(f"{p_:7.1%} {c:12.8f} {d:+8.1f}% {k:6.2f}  {' -> '.join(o)}")
print("\n== ORACLE-style: cheapest single model that solves each task (perfect foresight) ==")
tot=0;np_=0
for t in tasks:
    cands=[(price((t,m),PRICE),m) for m in models if P[(t,m)]]
    if cands: c,m=min(cands); tot+=c; np_+=1
    else: tot+=min(price((t,m),PRICE) for m in models)
print(f"pass={np_/len(tasks):.1%} ${tot/len(tasks):.8f}/task ({(tot/len(tasks)-bc)/bc*100:+.1f}% vs baseline)")
print("\n== SENSITIVITY: how cheap must model-1 be for cascade to beat baseline? ==")
for m in models:
    if m=="GPT-OSS-120B": continue
    fails=sum(1 for t in tasks if not P[(t,m)])
    esc=fails/len(tasks)
    c1=sum(price((t,m),PRICE) for t in tasks)/len(tasks)
    print(f"{m:16s} solo-pass={1-esc:5.1%} escalation-rate={esc:5.1%} -> cascade cost = c1 + {esc:.2f}*base; "
          f"needs c1 < {(1-esc):.3f}*base = ${bc*(1-esc):.8f}; actual c1=${c1:.8f} -> {'WINS' if c1<bc*(1-esc) else 'LOSES'}")
print("\n== latency (ms/task) ==")
def lat(o):
    tot=0
    for t in tasks:
        for m in o:
            tot+=LAT[(t,m)]
            if P[(t,m)]: break
    return tot/len(tasks)
for o in [["GPT-OSS-120B"],["Llama-3.1-8B","GPT-OSS-120B"],["Llama-4-Scout","GPT-OSS-120B"],["GPT-OSS-120B","Llama-4-Scout"]]:
    print(f"{' -> '.join(o):40s} {lat(o):8.0f} ms")
