# Runs INSIDE the sandbox (python slim image), never on the host. One cell per container.
# Reads cell.json, extracts code from the response by fixed rules, runs the family's tests in a
# child process with a time limit, prints one "@@CELL@@{json}" line.
#
# Extraction (fixed before any response was read; S1.7 checkpoint 2a):
#   code families: the fenced block that defines `def <entry_point>(`; else the longest fenced
#     block; else the whole text. If the result does not define the entry point and the task
#     ships a prompt (signature + docstring), the prompt is prepended (completion style).
#   cruxeval: the last `assert f(...) == <literal>` in the text; else the last fenced block;
#     else the whole stripped text. The literal is compared as `f(<input>) == <literal>`.
#   ds1000: the last <code>...</code> span; else the longest fenced block; else the whole text;
#     run as code_context's test_execution(solution) (and test_string(solution) if defined).
#   lcb: as code families, without an entry point; LeetCode-style tasks call
#     Solution().<func_name> (or a bare <func_name>), stdin tasks run the program.
import base64, json, pickle, re, subprocess, sys, time, zlib

FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.S)
p = json.load(open("cell.json"))
mode, text = p["mode"], p.get("response") or ""


def extract_code(text, ep=None):
    blocks = FENCE.findall(text)
    if ep:
        hit = [b for b in blocks if re.search(rf"\bdef\s+{re.escape(ep)}\s*\(", b)]
        if hit:
            return hit[-1]
    if blocks:
        return max(blocks, key=len)
    return text


def run_py(src, stdin=None, timeout=None):
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-c", src], input=stdin, capture_output=True, text=True,
                           timeout=timeout or p.get("timeout_s", 30))
        return {"rc": r.returncode, "out": r.stdout, "err": r.stderr[-3000:], "t": time.time() - t0,
                "timeout": False}
    except subprocess.TimeoutExpired:
        return {"rc": None, "out": "", "err": "", "t": time.time() - t0, "timeout": True}


def verdict(r):
    return "timeout" if r["timeout"] else ("pass" if r["rc"] == 0 else "fail")


out = {"mode": mode}
if p["family_kind"] in ("humaneval", "mbpp"):
    ep = p["entry_point"]
    code = p["reference"] if mode == "canonical" else extract_code(text, ep)
    if mode != "canonical" and not re.search(rf"\bdef\s+{re.escape(ep)}\s*\(", code) and p.get("prompt"):
        code = p["prompt"] + "\n" + code
        out["prompt_prepended"] = True
    src = "\n".join([p.get("pre", ""), code, "", p["test"], "", p.get("post", "")])
    r = run_py(src)
    out.update(stat=verdict(r), err=r["err"], elapsed=round(r["t"], 3), code_chars=len(code))
elif p["family_kind"] == "ds1000":
    if mode == "canonical":
        code = p["reference"]
    else:
        spans = re.findall(r"<code>(.*?)</code>", text, re.S)
        code = spans[-1] if spans else extract_code(text)
    ctx = p["code_context"]
    src = ctx + f"\ntest_execution({code!r})\n"
    if re.search(r"\bdef\s+test_string\s*\(", ctx):
        src += f"test_string({code!r})\n"
    r = run_py(src, timeout=p.get("timeout_s", 60))
    out.update(stat=verdict(r), err=r["err"], elapsed=round(r["t"], 3), code_chars=len(code))
elif p["family_kind"] == "cruxeval":
    if mode == "canonical":
        lit = p["output"]
    else:
        m = re.findall(r"assert\s+f\((.*)\)\s*==\s*(.+)", text)
        if m:
            lit = m[-1][1].strip().rstrip("`").strip()
        else:
            b = FENCE.findall(text)
            lit = (b[-1] if b else text).strip()
    out["literal"] = lit[:500]
    src = p["code"] + f"\nassert f({p['input']}) == {lit}\n"
    r = run_py(src)
    out.update(stat=verdict(r), err=r["err"], elapsed=round(r["t"], 3))
elif p["family_kind"] == "lcb":
    tests = json.loads(p["public_test_cases"] or "[]")
    priv = p.get("private_test_cases") or ""
    if priv:
        try:
            tests += json.loads(priv)
        except ValueError:
            tests += json.loads(pickle.loads(zlib.decompress(base64.b64decode(priv.encode()))))
    meta = json.loads(p.get("metadata") or "{}")
    code = p["reference"] if mode == "canonical" else extract_code(text)
    fn = meta.get("func_name")
    fails, t_all, first = 0, 0.0, ""
    for i, t in enumerate(tests):
        if t.get("testtype") == "functional" or fn:
            args = [line for line in t["input"].split("\n") if line.strip()]
            call = (f"import json\n_a=[json.loads(x) for x in {args!r}]\n"
                    f"_f=(Solution().{fn} if 'Solution' in globals() else {fn})\n"
                    f"_r=_f(*_a)\n_e=json.loads({t['output']!r})\n"
                    "assert _r==_e or (isinstance(_r,float) and abs(_r-_e)<1e-6), (_r,_e)\n")
            r = run_py("from typing import *\nimport collections, heapq, math, itertools, bisect\n"
                       + code + "\n" + call, timeout=p.get("per_test_s", 10))
            ok = r["rc"] == 0 and not r["timeout"]
        else:
            r = run_py(code, stdin=t["input"], timeout=p.get("per_test_s", 10))
            ok = (r["rc"] == 0 and not r["timeout"]
                  and r["out"].split() == t["output"].split())
        t_all += r["t"]
        if not ok:
            fails += 1
            if not first:
                first = f"test {i}: " + ("timeout" if r["timeout"] else (r["err"][-800:] or "wrong output"))
            break                                     # pass means every test passes
    out.update(stat="pass" if fails == 0 else ("timeout" if "timeout" in first else "fail"),
               err=first, elapsed=round(t_all, 3), n_tests=len(tests))
print("@@CELL@@" + json.dumps(out))
