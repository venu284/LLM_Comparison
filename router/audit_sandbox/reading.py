"""S1.7 reading protocol (criteria §4): fixed categories per response, filled by deterministic
rules. Written and committed before it is run on any response.

`read_cell(text, language, entry_point)` returns the §4 fields. It never sees the released score;
the reading table is joined to scores only after every sampled cell is categorised.
"""

from __future__ import annotations

import re

FENCE = re.compile(r"^[ \t]*```([^\n`]*)$", re.M)
LANG_ALIASES = {
    "python": {"python", "py", "python3"},
    "cpp": {"cpp", "c++", "cc", "cxx"},
    "go": {"go", "golang"},
    "java": {"java"},
    "javascript": {"javascript", "js", "node"},
    "rust": {"rust", "rs"},
    "typescript": {"typescript", "ts"},
}
CODE_LINE = re.compile(
    r"^\s*(def |class |import |from \S+ import |#include|func |package |public |private |"
    r"function |const |let |fn |use |return\b|for .*:|if .*:|while .*:)", re.M)
REFUSAL = re.compile(
    r"\b(I can(?:'|no)t (?:help|assist|provide|complete)|I'm sorry|I am sorry|I am unable|"
    r"I'm unable|could you (?:please )?clarify|need more (?:information|details))", re.I)
BRACKETS = {"(": ")", "[": "]", "{": "}"}


def blocks(text: str) -> tuple[list[tuple[str, str]], bool]:
    """Fenced blocks as (tag, body), and whether a fence is left open."""
    marks = list(FENCE.finditer(text))
    out = []
    for i in range(0, len(marks) - 1, 2):
        open_, close = marks[i], marks[i + 1]
        out.append((open_.group(1).strip().lower(), text[open_.end():close.start()]))
    return out, len(marks) % 2 == 1


def _unbalanced(code: str) -> bool:
    code = re.sub(r"(\"\"\"|''')[\s\S]*?\1", "", code)
    code = re.sub(r"\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", "", code)
    code = re.sub(r"#[^\n]*|//[^\n]*", "", code)
    stack = []
    for ch in code:
        if ch in BRACKETS:
            stack.append(BRACKETS[ch])
        elif ch in BRACKETS.values():
            if not stack or stack.pop() != ch:
                return True
    return bool(stack)


def read_cell(text, language: str, entry_point: str | None = None) -> dict:
    text = text if isinstance(text, str) else ""
    empty = not text.strip()
    bl, open_fence = blocks(text)
    if open_fence:                        # an unclosed final fence: take the rest as its body
        last = list(FENCE.finditer(text))[-1]
        bl.append((last.group(1).strip().lower(), text[last.end():]))
    code_like = bool(CODE_LINE.search(text))
    if empty:
        fmt = "no_code"
    elif len(bl) == 1:
        fmt = "fenced_one"
    elif len(bl) > 1:
        fmt = "fenced_many"
    elif code_like:
        fmt = "unfenced_code"
    else:
        fmt = "no_code"
    code = "\n".join(b for _, b in bl) if bl else (text if fmt == "unfenced_code" else "")

    if empty or fmt == "no_code":
        truncated = "no"
    elif open_fence:
        truncated = "yes"
    else:
        truncated = "unclear" if _unbalanced(bl[-1][1] if bl else code) else "no"

    refusal = "yes" if (not empty and REFUSAL.search(text[:400]) and fmt == "no_code") else "no"

    tags = {t.split()[0] for t, _ in bl if t}
    want = LANG_ALIASES.get(language, {language})
    if fmt == "no_code":
        lang = "n_a"
    elif tags and not (tags & want):
        lang = "other"
    else:
        lang = "expected"

    if not entry_point:
        ep = "n_a"
    else:
        name = re.escape(entry_point)
        pat = {"python": rf"\bdef\s+{name}\s*\(", "javascript": rf"\bfunction\s+{name}\b|\b{name}\s*=",
               "typescript": rf"\bfunction\s+{name}\b|\b{name}\s*=", "go": rf"\bfunc\s+{name}\s*\(",
               "rust": rf"\bfn\s+{name}\s*[(<]"}.get(language, rf"\b{name}\s*\(")
        anydef = {"python": r"\bdef\s+\w+\s*\(", "go": r"\bfunc\s+\w+", "rust": r"\bfn\s+\w+",
                  "javascript": r"\bfunction\s+\w+", "typescript": r"\bfunction\s+\w+"}.get(
            language, r"\w+\s*\([^)]*\)\s*\{")
        hay = code or text
        ep = "matches" if re.search(pat, hay) else ("differs" if re.search(anydef, hay) else "absent")

    think_open = "<think>" in text
    after = text.split("</think>", 1)[1] if "</think>" in text else ("" if think_open else text)
    reasoning_only = "yes" if (think_open and not after.strip()) else "no"

    return {"empty": "yes" if empty else "no", "code_format": fmt, "truncated": truncated,
            "refusal": refusal, "language": lang, "entry_point": ep,
            "reasoning_only": reasoning_only, "n_chars": len(text), "n_blocks": len(bl)}
