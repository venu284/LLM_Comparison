# Runs INSIDE the sandbox (python slim image), never on the host. One cell per container.
# Reads cell.json, extracts code from the response by fixed rules, runs the family's tests in a
# child process with a time limit, prints one "@@CELL@@{json}" line.
#
# Extraction: each benchmark's own post-processing where one exists (revised 2026-11-20 before
# any response of these families was read):
#   humaneval / hex_python / mbpp: EvalPlus sanitize (the same tree-sitter code ships as
#     bigcodebench.sanitize in this image) with the entry point. If the result does not define
#     the entry point and the task ships a prompt (signature + docstring), the prompt is
#     prepended (completion style).
#   cruxeval: the last `assert f(...) == <literal>` in the text; else the last fenced block;
#     else the whole stripped text. The literal is compared as `f(<input>) == <literal>`.
#   ds1000: DS-1000's own postprocess (test_ds1000.py at xlang-ai/DS-1000@cddd245): text before
#     the first </code>, "```python" removed, text before the next ```, before "\nEND SOLUTION",
#     "<code>" removed; run as code_context's test_execution(code) (+ test_string(code)).
#   lcb: the last fenced block (LiveCodeBench's extract_code), else the whole text; LeetCode-style tasks call
#     Solution().<func_name> (or a bare <func_name>), stdin tasks run the program.
import base64, json, pickle, re, subprocess, sys, time, types, zlib



def sanitize(text, ep):
    # bigcodebench.data cannot import in the BigCodeBench image (see harness_bcb.py); sanitize()
    # does not use it. Imported lazily: the DS-1000 image has no bigcodebench package.
    _stub = types.ModuleType("bigcodebench.data")
    for _n in ("get_bigcodebench", "load_solutions", "write_directory", "write_jsonl"):
        setattr(_stub, _n, None)
    sys.modules.setdefault("bigcodebench.data", _stub)
    from bigcodebench.sanitize import sanitize as _s
    return _s(text, ep)

FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.S)
p = json.load(open("cell.json"))
mode, text = p["mode"], p.get("response") or ""
probe = p.get("probe")   # mechanism probes (S1.7 §1.1): model one specific grader fault


def extract_code(text, ep=None):
    blocks = FENCE.findall(text)
    if ep:
        hit = [b for b in blocks if re.search(rf"\bdef\s+{re.escape(ep)}\s*\(", b)]
        if hit:
            return hit[-1]
    if blocks:
        return max(blocks, key=len)
    return text


_N = [0]


def run_py(src, stdin=None, timeout=None):
    _N[0] += 1
    path = f"/work/_run{_N[0]}.py"          # a file, not -c: large tests exceed the argv limit
    with open(path, "w") as f:
        f.write(src)
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, path], input=stdin, capture_output=True, text=True,
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
    code = p["reference"] if mode == "canonical" else sanitize(text, ep)
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
        code = text.split("</code>")[0].replace("```python", "").split("```")[0]
        code = code.split("\nEND SOLUTION")[0].replace("<code>", "")
        if probe == "ds_raw":                      # probe: no extraction at all
            code = text
    ctx = p["code_context"]
    src = ctx + f"\ncode = {code!r}\ntest_execution(code)\n"
    if "test_string(" in ctx:
        src += "test_string(code)\n"
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
    blocks = FENCE.findall(text)
    code = p["reference"] if mode == "canonical" else (blocks[-1] if blocks else text)
    fn = meta.get("func_name")
    fails, t_all, first = 0, 0.0, ""
    for i, t in enumerate(tests):
        if (t.get("testtype") == "functional" or fn) and probe != "lcb_stdin":
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
