# Runs INSIDE a toolchain image (router/audit_sandbox/toolchains/<tc>), never on the host. One
# cell per container. Reads cell.json, extracts and assembles a program by the fixed rules of
# criteria Amendment 9, compiles and runs it with the documented harness's commands, prints one
# "@@CELL@@{json}" line.
#
# Documented harnesses (fetched 2026-11-21):
#   HumanEval-X (cpp, go, java, js): CodeGeeX codegeex/benchmark (THUDM/CodeGeeX@2838420)
#     cpp : g++ -std=c++11 test.cpp [-lcrypto -lssl if "162" in task_id]; ./a.out; pass = exit 0
#     go  : go test main_test.go; pass = exit 0
#     java: javac Main.java; java -cp . Main; pass = exit 0 (CodeGeeX's run line is commented out)
#     js  : node test.js; pass = empty stderr and empty stdout (exit code not read)
#   MultiPL-E (rs, ts): nuprl/MultiPL-E evaluation/src (@3025a53)
#     rs  : rustc <file> -o <bin> (no edition flag); <bin>; pass = exit 0
#     ts  : tsc --target esnext <file>.ts; node <file>.js; pass = exit 0
# Assembly when the extracted code does not define the entry point: the documented
# prompt + code + test. When it does (a full-function chat answer): Amendment 9's rules below.
import json
import os
import re
import shutil
import subprocess
import time

p = json.load(open("cell.json"))
LANG = p["lang"]
MODE = p["mode"]                      # canonical | response
out = {"lang": LANG, "mode": MODE}

TAGS = {"cpp": {"cpp", "c++", "cc", "cxx", "c"}, "go": {"go", "golang"}, "java": {"java"},
        "js": {"js", "javascript", "node", "jsx"}, "rs": {"rust", "rs"},
        "ts": {"ts", "typescript", "tsx"}}
FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
CPP_HELPER = ["#include<stdlib.h>", "#include<algorithm>", "#include<math.h>", "#include<stdio.h>",
              "#include<vector>", "#include<string>", "#include<climits>", "#include<cstring>",
              "#include<iostream>"]                                   # CodeGeeX IMPORT_HELPER["cpp"]
GO_HELPER = ["math", "strings", "fmt", "strconv", "time", "bytes", "regexp", "sort", "math/rand",
             "crypto/md5"]                                            # CodeGeeX IMPORT_HELPER["go"]
COMPILE_S, RUN_S = 120, 30


def defines(code, name):
    n = re.escape(name)
    pat = {"cpp": rf"\b{n}\s*\([^;{{}}]*\)\s*(const\s*)?\{{",
           "java": rf"\b{n}\s*\([^;{{}}]*\)\s*(throws\s+[\w.,\s]+)?\{{",
           "go": rf"\bfunc\s+{n}\s*[\[(]",
           "rs": rf"\bfn\s+{n}\b",
           "js": rf"\bfunction\s*\*?\s+{n}\b|\b(const|let|var)\s+{n}\s*=",
           "ts": rf"\bfunction\s+{n}\b|\b(const|let|var)\s+{n}\s*[:=]"}[LANG]
    return re.search(pat, code) is not None


def extract(text, name):
    """Amendment 9: among fenced blocks tagged with the family's language (else all blocks), the
    last one that defines the entry point, else the last one; no block: the whole text."""
    blocks = FENCE.findall(text)
    tagged = [b for t, b in blocks if t.strip().lower() in TAGS[LANG]]
    cands = tagged or [b for _, b in blocks]
    if not cands:
        return text, "whole_text"
    for b in reversed(cands):
        if defines(b, name):
            return b, "block_defining_entry"
    return cands[-1], "last_block"


def _skip(code, i):
    """Index after a string literal or comment starting at i, else None."""
    c = code[i]
    if code.startswith("//", i):
        j = code.find("\n", i)
        return len(code) if j < 0 else j
    if code.startswith("/*", i):
        j = code.find("*/", i + 2)
        return len(code) if j < 0 else j + 2
    quotes = '"`' if LANG in ("js", "ts", "go") else '"'
    if LANG in ("cpp", "java", "js", "ts"):
        quotes += "'"
    if c in quotes:
        j = i + 1
        while j < len(code) and code[j] != c:
            j += 2 if code[j] == "\\" else 1
        return j + 1
    return None


def remove_blocks(code, start_pat):
    """Remove every top-level construct whose header matches start_pat, through its matching
    closing brace (strings and comments skipped). Returns (code, n_removed)."""
    n = 0
    while True:
        m = re.search(start_pat, code, re.M)
        if not m:
            return code, n
        i = code.find("{", m.end() - 1)
        if i < 0:
            return code, n
        depth, j = 0, i
        while j < len(code):
            s = _skip(code, j)
            if s is not None:
                j = s
                continue
            if code[j] == "{":
                depth += 1
            elif code[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        code = code[:m.start()] + code[j + 1:]
        n += 1


def run(cmd, timeout, cwd="/work"):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {"rc": r.returncode, "out": r.stdout, "err": r.stderr, "t": time.time() - t0, "to": False}
    except subprocess.TimeoutExpired:
        return {"rc": None, "out": "", "err": "timeout", "t": time.time() - t0, "to": True}


def go_imports(src):
    """Import paths declared in src, and src with its package clause and imports removed."""
    paths = re.findall(r'^\s*import\s+(?:\w+\s+)?"([^"]+)"', src, re.M)
    for blk in re.findall(r"^\s*import\s*\((.*?)\)", src, re.M | re.S):
        paths += re.findall(r'"([^"]+)"', blk)
    src = re.sub(r"^\s*import\s*\(.*?\)", "", src, flags=re.M | re.S)
    src = re.sub(r'^\s*import\s+(?:\w+\s+)?"[^"]+"\s*$', "", src, flags=re.M)
    src = re.sub(r"^\s*package\s+\w+\s*$", "", src, flags=re.M)
    return paths, src


def assemble(code, full):
    """Returns {filename: text} and notes."""
    notes = {}
    test = p["test"]
    if LANG == "cpp":
        if full:
            code, notes["removed_main"] = remove_blocks(code, r"^[ \t]*(?:int|void|signed)\s+main\s*\(")
            head = [s for s in CPP_HELPER if s not in code]
            head += [l for l in p["declaration"].splitlines()
                     if l.strip().startswith(("#include", "using namespace")) and l.strip() not in code]
            return {"test.cpp": "\n".join(head) + "\n" + code + "\n" + test}, notes
        setup = "".join(s + "\n" for s in CPP_HELPER if s not in p["prompt"])
        return {"test.cpp": setup + "\n" + p["prompt"] + code + "\n" + test}, notes
    if LANG == "go":
        setup_paths, _ = go_imports(p["test_setup"])
        if full:
            code, notes["removed_main"] = remove_blocks(code, r"^func\s+main\s*\(\s*\)")
            own, body = go_imports(code)
            prompt_paths, _ = go_imports(p["import"])
            cand = list(dict.fromkeys(own + prompt_paths + GO_HELPER))
        else:
            body = p["prompt"].replace(p["import"], "") + code
            cand = list(GO_HELPER)
        extra = [q for q in cand if q not in setup_paths and q.split("/")[-1] + "." in body]
        imports = "import (\n" + "".join(f'    "{q}"\n' for q in setup_paths + extra) + ")\n"
        return {"main_test.go": "package main\n\n" + imports + "\n" + body + "\n" + test}, notes
    if LANG == "java":
        if full:
            code = re.sub(r"^\s*package\s+[\w.]+\s*;", "", code, flags=re.M)
            code, notes["removed_main"] = remove_blocks(code, r"^[ \t]*(?:public\s+)?(?:final\s+)?class\s+Main\b")
            code = re.sub(r"\bpublic\s+(final\s+)?class\s+Solution\b", r"\1class Solution", code)
            imports = re.findall(r"^\s*import\s+[\w.*\s]+;", code, re.M)
            code = re.sub(r"^\s*import\s+[\w.*\s]+;", "", code, flags=re.M)
            decl = re.findall(r"^\s*import\s+[\w.*\s]+;", p["declaration"], re.M)
            imp = list(dict.fromkeys(x.strip() for x in decl + imports))
            return {"Main.java": "\n".join(imp) + "\n" + code + "\n" + test}, notes
        return {"Main.java": p["prompt"] + code + "\n" + test}, notes
    if LANG == "js":
        return {"test.js": (code if full else p["prompt"] + code) + "\n" + test}, notes
    if LANG == "rs":
        if full:
            code, notes["removed_main"] = remove_blocks(code, r"^\s*(?:pub\s+)?fn\s+main\s*\(\s*\)")
            t = test.lstrip()
            t = t[1:] if t.startswith("}") else t          # MultiPL-E tests open by closing the prompt's fn
            return {"main.rs": code + "\n" + t}, notes
        return {"main.rs": p["prompt"] + code + "\n" + test}, notes
    if LANG == "ts":
        return {"main.ts": (code if full else p["prompt"] + code) + "\n" + test}, notes
    raise ValueError(LANG)


def execute(files):
    for k, v in files.items():
        open(os.path.join("/work", k), "w").write(v)
    if LANG == "cpp":
        flags = ["-lcrypto", "-lssl"] if "162" in p["task_id"] else []
        c = run(["g++", "-std=c++11", "test.cpp", *flags], COMPILE_S)
        if c["rc"] != 0:
            return c, None
        return c, run(["./a.out"], RUN_S)
    if LANG == "go":
        shutil.copytree("/opt/goeval", "/work/g")
        shutil.move("/work/main_test.go", "/work/g/main_test.go")
        return None, run(["go", "test", "-mod=vendor", f"-timeout={RUN_S}s", "main_test.go"],
                         COMPILE_S + RUN_S, cwd="/work/g")
    if LANG == "java":
        c = run(["javac", "Main.java"], COMPILE_S)
        if c["rc"] != 0:
            return c, None
        return c, run(["java", "-cp", ".", "Main"], RUN_S)
    if LANG == "js":
        return None, run(["node", "test.js"], RUN_S)
    if LANG == "rs":
        c = run(["rustc", "main.rs", "-o", "main_bin"], COMPILE_S)
        if c["rc"] != 0:
            return c, None
        return c, run(["./main_bin"], RUN_S)
    if LANG == "ts":
        c = run(["tsc", "--target", "esnext", "main.ts"], COMPILE_S)
        if c["rc"] != 0:
            return c, None
        return c, run(["node", "main.js"], RUN_S)


name = p["entry_point"]
if MODE == "canonical":
    code, how, full = p["code"], "canonical", bool(p.get("full"))
else:
    code, how = extract(p["response"], name)
    full = defines(code, name)
out.update(extraction=how, full_function=full, code_chars=len(code))
files, notes = assemble(code, full)
out.update(notes)
comp, ex = execute(files)
if comp is not None and (comp["rc"] != 0):
    out["phase"] = "compile"
    out["stat"] = "timeout" if comp["to"] else "fail"
    out["err"] = (comp["err"] or comp["out"])[-1500:]
else:
    out["phase"] = "run"
    if ex["to"]:
        out["stat"] = "timeout"
    elif LANG == "js":                     # CodeGeeX: any stderr or stdout output is a failure
        out["stat"] = "pass" if not ex["err"].strip() and not ex["out"].strip() else "fail"
        out["exit_code_only"] = "pass" if ex["rc"] == 0 else "fail"
    else:
        out["stat"] = "pass" if ex["rc"] == 0 else "fail"
    out["err"] = (ex["err"] or "")[-1500:]
    out["stdout_tail"] = (ex["out"] or "")[-300:]
    out["rc"] = ex["rc"]
out["elapsed"] = round((comp["t"] if comp else 0) + (ex["t"] if ex else 0), 3)
print("@@CELL@@" + json.dumps(out))
