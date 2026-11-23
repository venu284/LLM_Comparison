"""S1.7 Amendment 11: mechanism `extraction`, sub-type `entry_point_unstated`.

The rule is fixed in `docs/research/S1.7_audit_criteria.md` (Amendment 11), committed before this
code. A sampled cell (graded key) gets the mechanism when all of these hold:

  (i)   the task's harness calls a named entry point the response is expected to define;
  (ii)  the task's `content_text` does not contain that name as a whole word;
  (iii) the response defines one or more functions, none with that name;
  (iv)  the released S0.1 outcome is fail;
  (v)   the cell's run-1 log shows the name is undefined (Python `NameError`, or the
        language's equivalent). A cell meeting (i)-(iv) with no run-1 log is `no_rerun` and does
        not count toward `d`.

For this sub-type only the mechanism may be assigned to a cell where our re-run agrees with the
release. No sandbox run is made: the evidence is the prompt text, the stored responses read as
text, and S1.7's existing re-run logs.

    PYTHONPATH=.:pipeline pipeline/venv/bin/python -m router.experiments.s1_7_entry_point
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.audit_sandbox.reading import read_cell  # noqa: E402

OUT = REPO_ROOT / "router/experiments/out"
MECHANISM = "extraction"
SUBTYPE = "entry_point_unstated"
NO_RERUN = "no_rerun"

# Language of each audited family's responses; every family not listed is Python.
FAMILY_LANG = {"hex_cpp": "cpp", "hex_go": "go", "hex_java": "java", "hex_js": "js",
               "multiple_rs": "rs", "multiple_ts": "ts"}
# (i) fails for these: no name the response must define (Amendment 11).
NO_ENTRY_POINT = ("ds1000", "cruxeval", "swebench")
_IDENT = r"[A-Za-z_]\w*"
_KEYWORDS = r"(?!(?:if|for|while|switch|catch|return|else)\b)"

# Amendment 9's "defines" table (router/audit_sandbox/harness_mx.py `defines`), `{n}` = the name.
MX_DEFINES = {
    "cpp": r"\b{n}\s*\([^;{{}}]*\)\s*(const\s*)?\{{",
    "java": r"\b{n}\s*\([^;{{}}]*\)\s*(throws\s+[\w.,\s]+)?\{{",
    "go": r"\bfunc\s+{n}\s*[\[(]",
    "rs": r"\bfn\s+{n}\b",
    "js": r"\bfunction\s*\*?\s+{n}\b|\b(const|let|var)\s+{n}\s*=",
    "ts": r"\bfunction\s+{n}\b|\b(const|let|var)\s+{n}\s*[:=]",
}
MX_TAGS = {"cpp": {"cpp", "c++", "cc", "cxx", "c"}, "go": {"go", "golang"}, "java": {"java"},
           "js": {"js", "javascript", "node", "jsx"}, "rs": {"rust", "rs"},
           "ts": {"ts", "typescript", "tsx"}}
MX_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)

# (v): the error text that shows the name is undefined, by language. Python is a prefix match
# because Python 3.10 may append "Did you mean ...".
UNDEFINED = {
    "python": lambda n: re.escape(f"NameError: name '{n}' is not defined"),
    "cpp": lambda n: re.escape(f"'{n}' was not declared"),
    "java": lambda n: r"symbol:\s+method " + re.escape(n) + r"\(",
    "go": lambda n: re.escape(f"undefined: {n}"),
    "rs": lambda n: re.escape(f"cannot find function `{n}`"),
    "js": lambda n: re.escape(f"ReferenceError: {n} is not defined"),
    "ts": lambda n: re.escape(f"Cannot find name '{n}'"),
}


# --------------------------------------------------------------------------- the five conditions
def name_stated(name: str, text: str | None) -> bool:
    """(ii) negated: `name` occurs in `text` as a whole word (case-sensitive, ASCII identifier
    boundaries)."""
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text or "") is not None


def _mx_defines(code: str, name_pat: str, lang: str) -> bool:
    return re.search(MX_DEFINES[lang].format(n=name_pat), code) is not None


def mx_extract(text: str, name: str, lang: str) -> str:
    """Amendment 9 extraction: among fenced blocks tagged with the language (else all blocks), the
    last that defines the entry point, else the last; no block: the whole text."""
    blocks = MX_FENCE.findall(text)
    tagged = [b for t, b in blocks if t.strip().lower() in MX_TAGS[lang]]
    cands = tagged or [b for _, b in blocks]
    if not cands:
        return text
    for b in reversed(cands):
        if _mx_defines(b, re.escape(name), lang):
            return b
    return cands[-1]


def defines_other(text: str | None, name: str, lang: str) -> bool:
    """(iii): the response defines one or more functions, none named `name`.
    Python: the §4 reading field `entry_point` is `differs`. Other languages: Amendment 9's
    extracted code defines some identifier and not `name`."""
    if not isinstance(text, str) or not text.strip():
        return False
    if lang == "python":
        return read_cell(text, "python", name)["entry_point"] == "differs"
    code = mx_extract(text, name, lang)
    return (_mx_defines(code, _KEYWORDS + _IDENT, lang)
            and not _mx_defines(code, re.escape(name), lang))


def undefined_in_log(err: str | None, name: str, lang: str) -> bool:
    """(v): the run-1 log's error text shows `name` is undefined."""
    return re.search(UNDEFINED[lang](name), err or "") is not None


def conditions(name: str | None, content: str | None, response: str | None,
               released_pass: bool, err: str | None, has_rerun: bool, lang: str) -> dict:
    """(i)-(v) for one cell, evaluated in order; a later condition is None once an earlier one
    fails. `verdict` is SUBTYPE, NO_RERUN or None."""
    out = {"c_i": name is not None, "c_ii": None, "c_iii": None, "c_iv": None, "c_v": None,
           "verdict": None}
    if not out["c_i"]:
        return out
    out["c_ii"] = not name_stated(name, content)
    if not out["c_ii"]:
        return out
    out["c_iii"] = defines_other(response, name, lang)
    if not out["c_iii"]:
        return out
    out["c_iv"] = not bool(released_pass)
    if not out["c_iv"]:
        return out
    if not has_rerun:
        out["verdict"] = NO_RERUN
        return out
    out["c_v"] = undefined_in_log(err, name, lang)
    out["verdict"] = SUBTYPE if out["c_v"] else None
    return out


# --------------------------------------------------------------------------- data
class Names:
    """(i): the entry-point name the harness calls, per (family, upstream) as in the cell rows."""

    def __init__(self):
        from router.experiments.s1_7_cp2 import Bcb, Mx, Py
        self.py, self.mx, self.bcb = Py(), Mx(), Bcb()

    def __call__(self, family: str, upstream: str) -> str | None:
        if family in NO_ENTRY_POINT:
            return None
        if family in FAMILY_LANG:
            return self.mx.entry(upstream) if self.mx.has_tests(upstream) else None
        if family.startswith("bcb"):
            return self.bcb.entry(upstream) if self.bcb.has_tests(upstream) else None
        if "::" not in upstream:            # reading-only and identical-pair rows: bare upstream id
            upstream = f"{family}::{upstream}"
        if not self.py.has_tests(upstream):
            return None
        if family == "lcb":                  # LeetCode: Solution().<func_name>; stdin: none
            md = self.py.lcb[upstream.split("::", 1)[1]].get("metadata") or {}
            md = json.loads(md) if isinstance(md, str) else md
            return md.get("func_name") or None
        return self.py.entry(upstream)


def _run1_errors(logs: list[str]) -> dict[str, str]:
    """run_id -> run-1 error text (`cell.err`), for the log references in `logs`."""
    want: dict[str, set[str]] = {}
    for ref in logs:
        if isinstance(ref, str) and "#" in ref:
            f, rid = ref.split("#", 1)
            want.setdefault(f, set()).add(rid)
    out = {}
    for f, ids in want.items():
        with open(OUT / f) as fh:
            for line in fh:
                r = json.loads(line)
                if r["run_id"] in ids:
                    out[f"{f}#{r['run_id']}"] = ((r.get("cell") or {}).get("err") or "")
    return out


def apply(c: pd.DataFrame, names: Names | None = None, contents: pd.Series | None = None,
          responses: dict | None = None) -> pd.DataFrame:
    """Columns `ep_name`, `c_i`..`c_v` and `a11` (SUBTYPE, NO_RERUN or empty) for every cell row.
    Responses are loaded only for cells that pass (i) and (ii)."""
    names = names or Names()
    c = c.copy()
    c["ep_name"] = [names(f, u) if isinstance(u, str) else None
                    for f, u in zip(c.family, c.upstream)]
    if contents is None:
        from router.data.prompts import join
        contents = join(sorted(set(c.task_id)))
    ep_names = [n if isinstance(n, str) else None for n in c.ep_name]
    stated = [n is None or name_stated(n, contents.get(t)) for n, t in zip(ep_names, c.task_id)]
    need = c[[not s for s in stated]]
    if responses is None:
        from router.experiments.s1_7_cp2 import load_responses
        responses = load_responses(need) if len(need) else {}
    errs = _run1_errors(list(need.log))
    rows = []
    for r, name in zip(c.itertuples(), ep_names):
        has_rerun = isinstance(r.log, str) and "#" in r.log
        rows.append(conditions(name, contents.get(r.task_id), responses.get((r.task_id, r.key)),
                               bool(r.released_pass), errs.get(r.log), has_rerun,
                               FAMILY_LANG.get(r.family, "python")))
    k = pd.DataFrame(rows, index=c.index)
    for col in ("c_i", "c_ii", "c_iii", "c_iv", "c_v"):
        c[col] = k[col]
    c["a11"] = k["verdict"].fillna("")
    return c


def funnel(c: pd.DataFrame, samples) -> pd.DataFrame:
    """Per sample: drawn graded cells and how many pass (i), (i)-(ii), (i)-(iii), (i)-(iv), then
    (v) or no_rerun. Rows from `apply`."""
    rows = []
    g = c[(c.key_role == "graded") & (c.test_version.fillna("plus") == "plus")]
    for smp in samples:
        s = g[g["sample"] == smp]
        if not len(s):
            continue
        t = lambda col: s[col].fillna(False).astype(bool)
        i = t("c_i")
        ii = i & t("c_ii")
        iii = ii & t("c_iii")
        iv = iii & t("c_iv")
        rows.append({"sample": smp, "drawn": len(s), "i_named_entry_point": int(i.sum()),
                     "ii_name_not_stated": int(ii.sum()), "iii_defines_other": int(iii.sum()),
                     "iv_released_fail": int(iv.sum()),
                     "v_undefined_in_log": int((s.a11 == SUBTYPE).sum()),
                     "no_rerun": int((s.a11 == NO_RERUN).sum())})
    return pd.DataFrame(rows)
