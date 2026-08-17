import csv
from collections import defaultdict
rows=list(csv.DictReader(open("/Users/venu/Downloads/Major Project/LLM_Comparison/LLM_Comparison/pipeline/exports/results_full.csv")))
cols=["tests_passed","tests_total","eslint_warnings","ts_any_count","compiler_errors","extraction_success","failure_mode","tokens_output","latency_total_ms"]
print("== column fill rate (non-empty, non-zero) over 325 rows ==")
for c in cols:
    ne=sum(1 for r in rows if (r[c] or "").strip() not in ("","0","0.0","None","null"))
    print(f"  {c:20s} populated: {ne:3d}/325 ({ne/325:.0%})")

print("\n== PARTIAL CREDIT: distribution of tests_passed/tests_total ==")
buckets=defaultdict(int)
for r in rows:
    tt=float(r["tests_total"] or 0); tp=float(r["tests_passed"] or 0)
    if tt==0: buckets["no tests ran"]+=1
    else:
        f=tp/tt
        buckets["0% (all fail)" if f==0 else "100% (all pass)" if f>=1 else "PARTIAL 1-99%"]+=1
for k,v in sorted(buckets.items(),key=lambda x:-x[1]): print(f"  {k:18s} {v:3d}  ({v/325:.0%})")

print("\n== KEY TEST: does a failed attempt's signal predict whether ESCALATION succeeds? ==")
# for each (task, cheap model) failure, what fraction of tests passed? does that predict GPT-OSS success?
P={}; frac={}; ce={}; ex={}
for r in rows:
    k=(r["task_id"],r["model_name"]); P[k]=r["pass_fail"].strip().lower() in("true","1","pass","passed")
    tt=float(r["tests_total"] or 0); frac[k]=(float(r["tests_passed"] or 0)/tt) if tt>0 else None
    ce[k]=float(r["compiler_errors"] or 0); ex[k]=(r["extraction_success"] or "").strip().lower() in("true","1")
tasks=sorted({r["task_id"] for r in rows})
for cheap in ["Llama-3.1-8B","Llama-4-Scout"]:
    fails=[t for t in tasks if not P[(t,cheap)]]
    print(f"\n  after {cheap} FAILS ({len(fails)} tasks) -> does GPT-OSS-120B then succeed?")
    grp=defaultdict(lambda:[0,0])
    for t in fails:
        f=frac[(t,cheap)]
        key = "no tests ran" if f is None else "0% tests passed" if f==0 else "partial >0%"
        grp[key][0]+=P[(t,"GPT-OSS-120B")]; grp[key][1]+=1
    for k,(s,n) in sorted(grp.items()):
        print(f"    signal={k:16s} n={n:2d}  escalation succeeds {s}/{n} = {s/n:5.1%}")
    # baseline
    s=sum(P[(t,"GPT-OSS-120B")] for t in fails)
    print(f"    ---- overall: {s}/{len(fails)} = {s/len(fails):.1%}  (this is what a blind 'always escalate' gets)")
