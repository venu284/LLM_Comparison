"""S1.7 artefact audit.

Checkpoint 1 (this module so far) uses released scores and response **presence and length
only**. No function here returns or prints response text: `response_stats` reduces every response
to `is_empty`, `n_chars` and a SHA-256, and the text is dropped. The rules are pre-registered in
`docs/research/S1.7_audit_criteria.md` (§5); section numbers below refer to that file.

Checkpoint 3 adds the frozen labels (`LABELS`), `build_artefacts()` / `load_artefacts()` for
`artefacts.yaml`, and `build_views()`, which builds the data views from an `OutcomeMatrix`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from router.data import coderouterbench as crb
from router.data.prices import CRB_PRICES
from router.data.scoring import RULES
from router.data.splits import PROBLEMS_CSV
from router.settings import SUCCESS_SCORE

RESPONSE_MATRIX = crb.CRB_DIR / "raw_matrices/phase1_acrouter_v2/response_matrix.json"
RELEASED_MODELS: tuple[str, ...] = tuple(sorted(CRB_PRICES))
SPLITS = ("probing", "id_test")

FLAGSHIP_K = 3                 # §5.1
ALIAS_SHARE = 0.99             # §5.5
CORR_TIE = 0.02                # §5.5
SCREEN_FRACTION = 0.99         # §5.3
PILEUP_MIN_CELLS = 10          # §5.3
BCB_PREFIX = "bcb"             # BigCodeBench families: bcb, bcb_algo2, bcb_completion, bcb_ds, bcb_v014


# --------------------------------------------------------------------------- released scores
def released_long(root: Path = crb.CRB_DIR) -> pd.DataFrame:
    """Both ID splits in the S1.2 long format (`complete` view: release CSV scores)."""
    return pd.concat([crb.long_table(s, "complete", root) for s in SPLITS], ignore_index=True)


# --------------------------------------------------------------------------- §5.1 flagship
def flagship_models(prices=CRB_PRICES, k: int = FLAGSHIP_K) -> list[str]:
    """The k released models with the highest output price; a tie at place k includes every tied
    model."""
    out = sorted(((p.output_per_m, m) for m, p in prices.items()), reverse=True)
    cut = out[k - 1][0]
    return sorted(m for price, m in out if price >= cut)


# --------------------------------------------------------------------------- responses (length only)
def _is_empty(v) -> bool:
    return not isinstance(v, str) or not v.strip()


def response_stats(path: Path = RESPONSE_MATRIX) -> pd.DataFrame:
    """One row per (task_id, key) present in the response matrix: `empty`, `n_chars`, `sha256`.
    The response text itself is not kept."""
    with open(path) as fh:
        matrix = json.load(fh)
    rows = []
    for task_id, by_key in matrix.items():
        for key, v in by_key.items():
            empty = _is_empty(v)
            text = v if isinstance(v, str) else ""
            rows.append((task_id, key, empty, len(text),
                         None if empty else hashlib.sha256(text.encode("utf-8")).hexdigest()))
    del matrix
    return pd.DataFrame(rows, columns=["task_id", "key", "is_empty", "n_chars", "sha256"])


# --------------------------------------------------------------------------- §5.5 model mapping
def alias_shares(stats: pd.DataFrame) -> pd.DataFrame:
    """For each pair of keys: tasks where both responses are non-empty, and the share of those
    that are byte-identical."""
    h = stats[~stats["is_empty"]].pivot(index="task_id", columns="key", values="sha256")
    keys = sorted(h.columns)
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            both = h[a].notna() & h[b].notna()
            n = int(both.sum())
            same = int((h.loc[both, a] == h.loc[both, b]).sum())
            rows.append({"key_a": a, "key_b": b, "both_nonempty": n, "identical": same,
                         "share": same / n if n else float("nan")})
    return pd.DataFrame(rows)


def length_correlation(stats: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of response character length (key) against released `output_tokens`
    (model), over tasks where the response is non-empty and the cell is available. Keys × models."""
    lengths = stats[~stats["is_empty"]].pivot(index="task_id", columns="key", values="n_chars")
    tokens = long[long.available].pivot(index="task_id", columns="model", values="output_tokens")
    out = pd.DataFrame(index=sorted(lengths.columns), columns=sorted(tokens.columns), dtype=float)
    for k in out.index:
        for m in out.columns:
            pair = pd.concat([lengths[k], tokens[m]], axis=1, join="inner").dropna()
            out.loc[k, m] = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
    return out


def chars_per_token(stats: pd.DataFrame, long: pd.DataFrame, ratio: float = 8.0) -> pd.DataFrame:
    """NOT a §5.5 rule; a diagnostic reported with the mapping. For each name-equal key: visible
    characters per released output token, and the share of cells whose visible text is longer than
    `ratio` characters per token (text the released token count cannot hold)."""
    st = stats[~stats["is_empty"] & stats.key.isin(RELEASED_MODELS)]
    m = st.merge(long[long.available][["task_id", "model", "output_tokens"]],
                 left_on=["task_id", "key"], right_on=["task_id", "model"])
    m["cpt"] = m.n_chars / m.output_tokens.clip(lower=1)
    g = m.groupby("key")
    return pd.DataFrame({"cells": g.size(), "median_chars_per_token": g.cpt.median(),
                         "share_over_ratio": g.cpt.apply(lambda s: float((s > ratio).mean())),
                         "median_output_tokens": g.output_tokens.median(),
                         "median_chars": g.n_chars.median()}).reset_index()


def over_ratio_matrix(stats: pd.DataFrame, long: pd.DataFrame, ratio: float = 8.0) -> pd.DataFrame:
    """Control for `chars_per_token`: the share of cells over `ratio` characters per token for
    every key's text against every released model's token counts (keys × models). A check that
    also passes wrong keys cannot confirm a mapping."""
    st = stats[~stats["is_empty"] & stats.key.isin(RELEASED_MODELS)][["task_id", "key", "n_chars"]]
    tok = long[long.available][["task_id", "model", "output_tokens"]]
    m = st.merge(tok, on="task_id")
    m["over"] = m.n_chars / m.output_tokens.clip(lower=1) > ratio
    return m.groupby(["key", "model"])["over"].mean().unstack()


def constant_token_records(long: pd.DataFrame) -> pd.DataFrame:
    """Score-free: model × family blocks (available cells) where released `output_tokens` takes a
    single value, i.e. token counts that cannot be per-response records."""
    av = long[long.available]
    g = av.groupby(["model", "family"])["output_tokens"]
    t = pd.DataFrame({"cells": g.size(), "distinct": g.nunique(), "value": g.first()}).reset_index()
    return t[(t.distinct == 1) & (t.cells > 1)].drop(columns="distinct").reset_index(drop=True)


def map_keys(aliases: pd.DataFrame, corr: pd.DataFrame) -> dict:
    """Apply §5.5. Returns {"key_to_model": {key: model or None}, "graded_key": {model: key or
    [keys] if unsettled}, "aliases": [[a, b], ...], "notes": [...]}."""
    alias_pairs = [sorted((r.key_a, r.key_b)) for r in aliases.itertuples() if r.share >= ALIAS_SHARE]
    key_to_model: dict[str, str | None] = {}
    for k in corr.index:
        top = corr.loc[k].astype(float).idxmax()
        key_to_model[k] = k if (k in corr.columns and top == k) else None
    changed = True
    while changed:                      # an alias of a mapped key maps to the same model
        changed = False
        for a, b in alias_pairs:
            for x, y in ((a, b), (b, a)):
                if key_to_model.get(x) and not key_to_model.get(y):
                    key_to_model[y] = key_to_model[x]
                    changed = True
    notes, graded = [], {}
    for m in corr.columns:
        cands = sorted(k for k, v in key_to_model.items() if v == m)
        if not cands:
            graded[m] = None
            notes.append(f"{m}: no key maps")
            continue
        # aliases of one another are one response set; pick the name-equal key to represent it
        groups = []
        for k in cands:
            g = next((g for g in groups if any(sorted((k, o)) in alias_pairs for o in g)), None)
            (g.append(k) if g is not None else groups.append([k]))
        if len(groups) == 1:
            graded[m] = m if m in cands else cands[0]
            continue
        reps = [m if m in g else g[0] for g in groups]
        rs = sorted(((float(corr.loc[r, m]), r) for r in reps), reverse=True)
        if rs[0][0] - rs[1][0] < CORR_TIE:
            graded[m] = [r for _, r in rs]
            notes.append(f"{m}: unsettled between {graded[m]}")
        else:
            graded[m] = rs[0][1]
    return {"key_to_model": key_to_model, "graded_key": graded,
            "aliases": alias_pairs, "notes": notes}


# --------------------------------------------------------------------------- §5.2 family × model table
def family_model_table(long: pd.DataFrame) -> pd.DataFrame:
    """Per family × released model, over available cells."""
    g = long.groupby(["family", "model"])
    av = long[long.available]
    ga = av.groupby(["family", "model"])["score"]
    t = pd.DataFrame({
        "cells": g.size(),
        "unavailable": g["available"].apply(lambda s: int((~s.astype(bool)).sum())),
        "available": ga.size(),
        "mean": ga.mean(),
        "share_one": ga.apply(lambda s: float((s >= SUCCESS_SCORE).mean())),
        "share_zero": ga.apply(lambda s: float((s == 0).mean())),
        "distinct": ga.nunique(),
        "off_binary": ga.apply(lambda s: int((~s.isin([0.0, 1.0])).sum())),
    })
    t[["available", "distinct", "off_binary"]] = t[["available", "distinct", "off_binary"]].fillna(0).astype(int)
    return t.reset_index()


def family_flags(table: pd.DataFrame, flagship: list[str]) -> pd.DataFrame:
    """§5.2 flags per family."""
    rows = []
    for fam, t in table.groupby("family"):
        r = RULES[fam]
        t = t.set_index("model")
        med = float(np.median(t["share_one"]))
        below = sorted(m for m in flagship if m in t.index and t.at[m, "share_one"] < med)
        off_cells = int(t["off_binary"].sum())
        off_rule = r.score_type == "exec" and r.partial_credit in ("unstated", "no") and off_cells > 0
        all_zero = bool((t["share_zero"] == 1).all())
        rows.append({"family": fam, "eligible": r.eligible, "score_type": r.score_type,
                     "partial_credit": r.partial_credit, "pool_median_share_one": med,
                     "all_zero": all_zero, "flagship_below_median": below,
                     "off_rule": off_rule, "off_rule_cells": off_cells if r.score_type == "exec" else 0,
                     "flagged": bool(all_zero or below or off_rule)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- §5.3 truncation screen
def truncation_screen(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per model: M, cells exactly at M, cells screened (>= 0.99 M), pile-up. Also the screened
    cells (task_id, model, family, output_tokens)."""
    av = long[long.available]
    rows, cells = [], []
    for m, t in av.groupby("model"):
        top = int(t["output_tokens"].max())
        at = int((t["output_tokens"] == top).sum())
        hit = t[t["output_tokens"] >= SCREEN_FRACTION * top]
        rows.append({"model": m, "max_output_tokens": top, "cells_at_max": at,
                     "cells_screened": int(len(hit)), "pileup": at >= PILEUP_MIN_CELLS})
        cells.append(hit[["task_id", "model", "family", "output_tokens"]])
    return pd.DataFrame(rows), pd.concat(cells, ignore_index=True).sort_values(["model", "task_id"])


# --------------------------------------------------------------------------- §5.4 coverage
def coverage(stats: pd.DataFrame, long: pd.DataFrame, key_to_model: dict) -> pd.DataFrame:
    """Per family × response key: non-empty; empty with released output_tokens > 0; empty with
    output_tokens == 0; empty with released score 1.0; empty on an unavailable cell. Keys with no
    released model get only the presence counts."""
    fam = long.drop_duplicates("task_id").set_index("task_id")["family"]
    rel = long.set_index(["task_id", "model"])
    s = stats.copy()
    s["family"] = s["task_id"].map(fam)
    s["model"] = s["key"].map(key_to_model)
    idx = pd.MultiIndex.from_arrays([s["task_id"], s["model"].fillna("")])
    s["output_tokens"] = rel["output_tokens"].reindex(idx).to_numpy()
    s["score"] = rel["score"].reindex(idx).to_numpy()
    s["available"] = rel["available"].reindex(idx).to_numpy()
    e = s["is_empty"]
    mapped = s["model"].notna()
    s["nonempty"] = ~e
    s["empty_tokens_pos"] = e & mapped & (s["output_tokens"] > 0)
    s["empty_tokens_zero"] = e & mapped & (s["output_tokens"] == 0)
    s["empty_score_one"] = e & mapped & (s["score"] >= SUCCESS_SCORE)
    s["empty_unavailable"] = e & mapped & (s["available"] == False)  # noqa: E712
    cols = ["nonempty", "empty_tokens_pos", "empty_tokens_zero", "empty_score_one", "empty_unavailable"]
    out = s.groupby(["family", "key"]).agg(cells=("is_empty", "size"), empty=("is_empty", "sum"),
                                           **{c: (c, "sum") for c in cols})
    return out.astype(int).reset_index()


# --------------------------------------------------------------------------- identical prompts (§8.2)
def content_hashes(problems_csv: Path = PROBLEMS_CSV) -> pd.DataFrame:
    """task_id -> S1.5 `content_sha256` (the criteria's definition, §8.2)."""
    return pd.read_csv(problems_csv, usecols=["task_id", "content_sha256"]).dropna().rename(
        columns={"content_sha256": "sha"})


def wrapped_prompt_hashes() -> pd.DataFrame:
    """task_id -> S1.5 `prompt_sha256` (the wrapped prompt). Reported next to the criteria's
    definition because the plan's wording is "prompt identical to a task in another dimension"."""
    from router.data.prompts import prompt_table
    return prompt_table()[["task_id", "prompt_sha256"]].dropna().rename(columns={"prompt_sha256": "sha"})


def identical_prompt_bcb(long: pd.DataFrame, hashes: pd.DataFrame, across: str = "family") -> dict:
    """BigCodeBench-family tasks whose hash equals a task's in a different `across` group (family
    or dimension). For each pair and model with both cells available, compare the released S0.1
    outcomes."""
    meta = long.drop_duplicates("task_id").set_index("task_id")[["family", "dimension"]]
    p = hashes[hashes.task_id.isin(meta.index)].join(meta, on="task_id")
    pairs = []
    for _, g in p.groupby("sha"):
        if g[across].nunique() < 2:
            continue
        for a in g[g.family.str.startswith(BCB_PREFIX)].itertuples():
            for b in g[g[across] != getattr(a, across)].itertuples():
                pairs.append((a.task_id, a.family, b.task_id, b.family))
    pairs = sorted(set(pairs))
    ok = long[long.available].set_index(["task_id", "model"])["score"] >= SUCCESS_SCORE
    cells, differ = 0, []
    for a, fa, b, fb in pairs:
        for m in RELEASED_MODELS:
            if (a, m) in ok.index and (b, m) in ok.index:
                cells += 1
                if bool(ok[(a, m)]) != bool(ok[(b, m)]):
                    differ.append({"bcb_task": a, "other_task": b, "other_family": fb, "model": m,
                                   "bcb_pass": bool(ok[(a, m)]), "other_pass": bool(ok[(b, m)])})
    by_fam = pd.Series([f"{fa}|{fb}" for _, fa, _, fb in pairs], dtype=object).value_counts()
    return {"across": across, "bcb_tasks": len({a for a, *_ in pairs}), "pairs": len(pairs),
            "pairs_by_family": {k: int(v) for k, v in sorted(by_fam.items())},
            "cells_compared": cells, "cells_outcome_differs": len(differ),
            "differing_cells": differ,
            "pairs_list": [{"bcb_task": a, "bcb_family": fa, "other_task": b, "other_family": fb}
                           for a, fa, b, fb in pairs]}


def name_match_keys(keys, aliases: pd.DataFrame) -> dict:
    """NOT the §5.5 rule: map each key whose name equals a released model to it, plus byte aliases
    of such keys. Used only to give the empty-response decision full counts while §5.5's outcome
    is reviewed."""
    out = {k: (k if k in RELEASED_MODELS else None) for k in keys}
    for r in aliases.itertuples():
        if r.share >= ALIAS_SHARE:
            for x, y in ((r.key_a, r.key_b), (r.key_b, r.key_a)):
                if out.get(x) and not out.get(y):
                    out[y] = out[x]
    return out


# --------------------------------------------------------------------------- checkpoint 2: samples (§8, Amendments 1, 3)
UNCONFIRMED = ("MiniMax-M2.7", "glm-5", "qwen3.5-plus")          # Amendment 1
MAIN_STREAMS = {"lcb": 100, "mbpp": 101, "hex_cpp": 102, "hex_go": 103, "hex_java": 104,
                "hex_js": 105, "multiple_rs": 106, "multiple_ts": 107, "swebench": 108,
                "cruxeval": 109, "bcb": 110, "bcb_v014": 111, "bcb_algo2": 112,
                "bcb_completion": 113, "bcb_ds": 114, "ds1000": 115, "humaneval": 116,
                "hex_python": 130, "mbpp_completion": 131}
SAMPLE_N = 97
N_TASKS_ALL = 12
QWEN_CODEGEN = ("bcb", "bcb_v014", "humaneval", "mbpp")          # Amendment 1 (decision B)


def response_key(model: str) -> str:
    """Amendment 1: every released model reads the key of the same name (Qwen3-Max also has
    `qwen3.6-plus` on QWEN_CODEGEN, read alongside)."""
    return model


def frame(long: pd.DataFrame, stats: pd.DataFrame, tasks) -> pd.DataFrame:
    """§8.1 frame: available cells of `tasks` whose mapped response is non-empty, sorted by
    (task_id, model)."""
    ne = stats.loc[~stats["is_empty"], ["task_id", "key"]]
    c = long[long.task_id.isin(set(tasks)) & long.available][["task_id", "model", "family"]]
    c = c.assign(key=c.model.map(response_key)).merge(ne, on=["task_id", "key"])
    return c.sort_values(["task_id", "model"]).reset_index(drop=True)


def draw(fr: pd.DataFrame, stream: int, n: int = SAMPLE_N, n_tasks_all: int = N_TASKS_ALL) -> pd.DataFrame:
    """§8.1: every cell when the frame has n_tasks_all tasks or fewer, else n cells without
    replacement from SeedSequence([0, stream])."""
    few_tasks = "task_id" in fr and fr.task_id.nunique() <= n_tasks_all
    if few_tasks or len(fr) <= n:
        out = fr.copy()
    else:
        rng = np.random.default_rng(np.random.SeedSequence([0, stream]))
        idx = np.sort(rng.choice(len(fr), size=n, replace=False))
        out = fr.iloc[idx].copy()
    return out.assign(stream=stream).reset_index(drop=True)


def all_zero_tasks(long: pd.DataFrame, families) -> list[str]:
    """X12b stratum (§8.2, streams 120-124): tasks of `families` whose every available released
    cell is 0. Score-defined, so evidence only (§3.5)."""
    t = long[long.family.isin(set(families)) & long.available]
    z = t.groupby("task_id")["score"].apply(lambda s: bool((s == 0).all()))
    return sorted(z[z].index)


def identical_frame(differing_cells: list[dict]) -> pd.DataFrame:
    """Amendment 3 frame, sorted by (bcb_task, other_task, model)."""
    return (pd.DataFrame(differing_cells).sort_values(["bcb_task", "other_task", "model"])
            .reset_index(drop=True))


# --------------------------------------------------------------------------- checkpoint 3: final labels (step 10)
# Labels written after 2c and the mapping test (Amendment 2), from the §3.2 rules applied to the
# main samples plus decision D, and the user's decisions at the checkpoint 2 reviews
# (2026-11-21). Evidence: docs/research/S1.7_audit.md ("Checkpoint 3: final labels") and the
# per-sample rows in router/experiments/out/s1_7_interim_*.csv and s1_7_mechanisms_*.csv.
CRITERIA_COMMIT = "feadbed"      # last criteria amendment (Amendment 10) before any label
CONFIRMED, CLEARED, UNDECIDED = "confirmed defect", "cleared", "undecided"
NOT_SUSPECTED = ("he_completion",)       # eligible, never flagged (§5.2): stays in every view
LCB_SUBSET_PROPERTY = "platform"         # Amendment 7: LiveCodeBench `platform`, non-score
LCB_EXCLUDED_PLATFORM = "leetcode"

_EV = "docs/research/S1.7_audit.md"
LABELS: dict[str, dict] = {
    # unit: family, or a part of one fixed by an approved subset rule (§7)
    **{f: {"family": f, "label": CONFIRMED, "mechanisms": ["toolchain"], "sample": f,
           "interim": "s1_7_interim_2a.csv", "mechanism_rows": "s1_7_mechanisms_2a.csv",
           "evidence": f"{_EV}: 2a §3, decision D (D-BCB-LIB)"}
       for f in ("bcb", "bcb_v014", "bcb_algo2", "bcb_completion", "bcb_ds")},
    "ds1000": {"family": "ds1000", "label": CONFIRMED, "mechanisms": ["extraction", "toolchain"],
               "sample": "ds1000", "interim": "s1_7_interim_2a.csv",
               "mechanism_rows": "s1_7_mechanisms_2a.csv",
               "evidence": f"{_EV}: 2a §3, decision D (D-DS-LIB)"},
    "lcb": {"family": "lcb", "part": "LeetCode-platform tasks (subset rule, Amendment 7)",
            "label": CONFIRMED, "mechanisms": ["harness_timeout", "test_version"], "sample": "lcb",
            "interim": "s1_7_interim_2a.csv", "mechanism_rows": "s1_7_mechanisms_2a.csv",
            "evidence": f"{_EV}: 2a §3 (R-LCB-CALL), decision D (D-LCB-PUBLIC); subset rule "
                        "accepted by the user 2026-11-21"},
    "lcb_kept": {"family": "lcb", "part": "AtCoder and Codeforces tasks (subset rule, Amendment 7)",
                 "label": UNDECIDED, "reason": "capped share (Amendment 6)",
                 "mechanisms": ["test_version"], "sample": "lcb_kept",
                 "interim": "s1_7_interim_lcbk.csv", "mechanism_rows": "s1_7_mechanisms_lcbk.csv",
                 "evidence": f"{_EV}: 2a remainder §1"},
    "mbpp": {"family": "mbpp", "label": CLEARED, "mechanisms": [], "sample": "mbpp",
             "interim": "s1_7_interim_2a.csv", "mechanism_rows": "s1_7_mechanisms_2a.csv",
             "evidence": f"{_EV}: 2a §2"},
    "hex_python": {"family": "hex_python", "label": CLEARED, "mechanisms": ["release_correct"],
                   "sample": "hex_python", "interim": "s1_7_interim_2a.csv",
                   "mechanism_rows": "s1_7_mechanisms_2a.csv",
                   "evidence": f"{_EV}: decision D (D-EXTRACT, Python/50; reviewer read the "
                               "response, 2026-11-21)"},
    "cruxeval": {"family": "cruxeval", "label": CLEARED, "mechanisms": ["release_correct"],
                 "sample": "cruxeval", "interim": "s1_7_interim_2a.csv",
                 "mechanism_rows": "s1_7_mechanisms_2a.csv",
                 "evidence": f"{_EV}: decision D (D-EXTRACT, sample_505; reviewer read the "
                             "response, 2026-11-21); strict success rule (CLAIMS A3)",
                 "note": "cleared under the strict rule (released score 1.0, CLAIMS A3). Against "
                         "the 0.8 sensitivity reading, main-sample agreement is 62/97 under "
                         "Amendment 8 (s1_7_cruxeval_a2.csv), which would not clear"},
    "humaneval": {"family": "humaneval", "label": UNDECIDED,
                  "reason": "a = 0.948 < 0.95; base vs HumanEval+ unresolved (user 2026-11-21)",
                  "mechanisms": ["test_version"], "sample": "humaneval",
                  "interim": "s1_7_interim_2a.csv", "mechanism_rows": "s1_7_mechanisms_2a.csv",
                  "evidence": f"{_EV}: 2a §3, decision D (D-HE-BASE)"},
    "mbpp_completion": {"family": "mbpp_completion", "label": UNDECIDED,
                        "reason": "1 unexplained cell (u > 0)", "mechanisms": ["release_correct"],
                        "sample": "mbpp_completion", "interim": "s1_7_interim_2a.csv",
                        "mechanism_rows": "s1_7_mechanisms_2a.csv",
                        "evidence": f"{_EV}: decision D (task 599 unexplained)"},
    **{f: {"family": f, "label": CONFIRMED, "mechanisms": ["wrong_field"], "sample": f,
           "interim": "s1_7_interim_2b.csv", "mechanism_rows": "s1_7_mechanisms_2b.csv",
           "evidence": f"{_EV}: 2b §3 (user decision F for multiple_*)"}
       for f in ("hex_cpp", "hex_go", "hex_java", "hex_js", "multiple_rs", "multiple_ts")},
    "swebench": {"family": "swebench", "label": UNDECIDED, "reason": "capped share (Amendment 6)",
                 "mechanisms": [], "sample": "swebench", "interim": "s1_7_interim_2c.csv",
                 "mechanism_rows": "s1_7_mechanisms_2c.csv",
                 "evidence": f"{_EV}: 2c §3 (85 of 97 answers carry no patch)"},
    **{f: {"family": f, "label": UNDECIDED, "eligible": False,
           "reason": "reading only (§3.3); ineligible since S1.4; affects no view",
           "mechanisms": [], "sample": None, "interim": None, "mechanism_rows": None,
           "evidence": f"{_EV}: 2a §7 (streams 126-128)"}
       for f in ("lcb_testgen", "he_testgen", "mbpp_testgen")},
}


def lcb_platforms(sources_dir: Path | None = None) -> dict[str, str]:
    """LiveCodeBench `question_id` -> `platform`, from the pinned text extract (Amendment 7)."""
    from router.settings import EXTERNAL_DIR
    path = (sources_dir or EXTERNAL_DIR / "sources") / "livecodebench.jsonl"
    rows = (json.loads(l) for l in path.read_text().splitlines() if l.strip())
    return {r["question_id"]: r[LCB_SUBSET_PROPERTY] for r in rows}


def label_unit(task_id: str, family: str, leetcode: set[str]) -> str | None:
    """The LABELS key that governs a task: its family, `lcb` split by the subset rule, or None
    for a family that was never suspected."""
    if family == "lcb":
        return "lcb" if task_id in leetcode else "lcb_kept"
    return family if family in LABELS else None


# --------------------------------------------------------------------------- checkpoint 3: artefacts.yaml and the views
# Steps 11 and 12. `artefacts.yaml` is generated by `build_artefacts()` and committed before any
# router result (R10); `load_artefacts()` reads it back and `build_views()` builds the three data
# views (plus the sensitivity view) from an OutcomeMatrix. Membership depends only on the labels,
# eligibility (S1.4) and X13 coverage (S0.11), never on a score.
ARTEFACTS_YAML = Path(__file__).with_name("artefacts.yaml")
LABELS_COMMIT = "a865dfb"        # the commit that wrote LABELS (checkpoint 3, step 10)
VIEW_NAMES = ("all_eligible", "filtered", "sensitivity", "headline")
STORAGE_CAP_CHARS = 2000         # Amendment 6: a stored response of exactly this length is capped
EXCLUDED_LABELS = (CONFIRMED, UNDECIDED)      # leave the filtered view
SENSITIVITY_ADDS = (UNDECIDED,)               # come back in the sensitivity view


def load_release(root: Path = crb.CRB_DIR) -> "OutcomeMatrix":
    """Both CodeRouterBench ID splits (complete view) as one OutcomeMatrix, with the empty X13
    cells marked unavailable (EMPTY_RESPONSE, S0.11) next to the 148 missing token records."""
    from router.data import verdicts
    from router.data.schema import EMPTY_RESPONSE, OutcomeMatrix
    long = pd.concat([crb.long_table(s, "complete", root) for s in SPLITS], ignore_index=True)
    m = OutcomeMatrix.from_long(long, name="crb/id/complete")
    v = verdicts.load_verdicts(root=root)
    empty = v.loc[v.unavailable, ["task_id", "model"]].itertuples(index=False, name=None)
    return m.mark_unavailable(list(empty), EMPTY_RESPONSE)


def x13_tasks() -> set[str]:
    from router.data import verdicts
    return set(verdicts.load_raw().task_id)


def task_labels(families: pd.Series, leetcode: set[str]) -> pd.Series:
    """task -> label for one task index: a LABELS label, or "not suspected"."""
    out = {}
    for t, f in families.items():
        unit = label_unit(t, f, leetcode)
        out[t] = "not suspected" if unit is None else LABELS[unit]["label"]
    return pd.Series(out, dtype=object).reindex(families.index)


def view_task_ids(families: pd.Series, leetcode: set[str], x13: set[str]) -> dict[str, list[str]]:
    """The four views' sorted task ids (S0.10, S0.11, R10) for a task -> family series."""
    from router.data.scoring import eligible_families
    fam = families[families.isin(set(eligible_families()))]
    lab = task_labels(fam, leetcode)
    filtered = lab[~lab.isin(EXCLUDED_LABELS)].index
    sens = lab[~lab.isin(EXCLUDED_LABELS) | lab.isin(SENSITIVITY_ADDS)].index
    return {"all_eligible": sorted(fam.index), "filtered": sorted(filtered),
            "sensitivity": sorted(sens), "headline": sorted(t for t in filtered if t in x13)}


def _sha(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def build_views(matrix, artefacts: dict | None = None, x13: set[str] | None = None) -> dict:
    """view name -> OutcomeMatrix subset of `matrix`, by the frozen labels in `artefacts`
    (default: the committed artefacts.yaml). `x13` is the set of tasks with X13 verdicts
    (default: read from the X13 file)."""
    art = load_artefacts() if artefacts is None else artefacts
    leetcode = set(art["subset_rules"]["lcb"]["excluded_task_ids"])
    ids = view_task_ids(matrix.family, leetcode, x13_tasks() if x13 is None else x13)
    return {name: matrix.subset(ids[name]) for name in VIEW_NAMES}


def _py(x):
    """Plain Python types for yaml.safe_dump."""
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return round(float(x), 4)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _success_counts(m, lenient: bool) -> dict:
    """Success-dependent counts on a view (S0.1; CRUXEval 0.8 only with lenient=True)."""
    from router.data.scoring import success
    fam = m.family.reindex(m.score.index)
    ok = pd.DataFrame({c: success(m.score[c].to_numpy(), list(fam), cruxeval_lenient=lenient)
                       for c in m.models}, index=m.tasks) & m.available
    complete = m.available.all(axis=1)
    rate = ok[complete].mean()
    best = rate.sort_values(ascending=False, kind="stable")
    return {"successes_per_model": {k: int(ok[k].sum()) for k in sorted(m.models)},
            "tasks_solved_by_no_model": int((~ok.any(axis=1)).sum()),
            "complete_case_tasks_solved_by_no_model": int((~ok[complete].any(axis=1)).sum()),
            "best_single_model_complete_case": {"model": best.index[0], "success_rate": best.iloc[0]}}


def view_counts(m, task_split: pd.Series, stats: pd.DataFrame) -> dict:
    """Every audit-dependent count for one view (step 11)."""
    from router.data.schema import UNAVAILABLE_REASONS
    tasks = list(m.tasks)
    long = m.long
    fam = m.family.value_counts()
    split = task_split.reindex(tasks)
    reason = m.unavailable_reason.stack()
    complete = m.available.all(axis=1)
    st = stats[stats.task_id.isin(set(tasks)) & stats.key.isin(set(m.models))]
    empty = st[st.is_empty].set_index(["task_id", "key"]).index
    sc = m.score.stack()
    empty_unavail = m.available.copy()
    for t, k in empty:
        empty_unavail.loc[t, k] = False
    capped = st[(st.n_chars == STORAGE_CAP_CHARS) & ~st.is_empty]
    return {
        "tasks": len(tasks), "problems": int(long.drop_duplicates("task_id").problem_id.nunique()),
        "families": int(len(fam)), "tasks_per_family": {k: int(fam[k]) for k in sorted(fam.index)},
        "development_tasks": int((split == "dev").sum()), "evaluation_tasks": int((split == "eval").sum()),
        "cells": int(m.available.size),
        "unavailable_cells": {r: int((reason == r).sum()) for r in UNAVAILABLE_REASONS},
        "complete_case_tasks": int(complete.sum()),
        "x13_tasks": None,     # filled by build_artefacts
        "response_matrix_empty_cells": int(len(empty)),
        "response_matrix_empty_cells_score_1": int(sum(sc.get((t, k), 0.0) >= SUCCESS_SCORE for t, k in empty)),
        "complete_case_tasks_if_empty_responses_unavailable": int(empty_unavail.all(axis=1).sum()),
        "capped_cells": int(len(capped)),
        "capped_cell_share": len(capped) / max(1, len(st[~st.is_empty])),
        "strict": _success_counts(m, lenient=False),
        "cruxeval_lenient_sensitivity": _success_counts(m, lenient=True),
    }


def _interim(unit: dict) -> dict | None:
    if not unit.get("interim"):
        return None
    out_dir = Path(__file__).resolve().parents[1] / "experiments/out"
    r = pd.read_csv(out_dir / unit["interim"]).set_index("family").loc[unit["sample"]]
    keys = ("drawn", "n_exec", "agree", "a", "d", "r_wilson_lo", "u", "our_harness_or_error",
            "capped", "capped_wilson_lo")
    return {k: _py(r[k].item() if hasattr(r[k], "item") else r[k]) for k in keys}


def build_artefacts(root: Path = crb.CRB_DIR) -> dict:
    """The full artefacts.yaml content (deterministic: no dates, no own commit)."""
    from router.data import splits, verdicts
    from router.data.prompts import upstream_id
    from router.data.scoring import eligible_families
    m = load_release(root)
    plat = lcb_platforms()
    lcb_ids = [t for t in m.tasks if m.family[t] == "lcb"]
    leetcode = sorted(t for t in lcb_ids if plat.get(upstream_id(t, "lcb")) == LCB_EXCLUDED_PLATFORM)
    x13 = x13_tasks()
    ts = splits.task_splits().set_index("task_id")["split"]
    stats = response_stats()
    ids = view_task_ids(m.family, set(leetcode), x13)
    el = set(ids["all_eligible"])
    bad = sorted(set(ts.reindex(sorted(el)).dropna().unique()) - {"dev", "eval"})
    if bad or ts.reindex(sorted(el)).isna().any():
        raise ValueError(f"eligible tasks outside dev/eval: {bad}")

    units = {}
    for key, u in LABELS.items():
        tasks = [t for t in m.tasks if label_unit(t, m.family[t], set(leetcode)) == key]
        units[key] = {k: u[k] for k in ("family", "part", "label", "reason", "mechanisms",
                                         "sample", "evidence", "note") if u.get(k) is not None}
        units[key].update(tasks=len(tasks), eligible=u.get("eligible", True),
                          main_sample=_interim(u),
                          mechanism_rows=(f"router/experiments/out/{u['mechanism_rows']}"
                                          if u.get("mechanism_rows") else None))
    for f in NOT_SUSPECTED:
        units[f] = {"family": f, "label": "not suspected", "tasks": int((m.family == f).sum()),
                    "eligible": f in set(eligible_families()), "evidence": "§5.2: never flagged"}

    views = {}
    for name in VIEW_NAMES:
        sub = m.subset(ids[name])
        c = view_counts(sub, ts, stats)
        c["x13_tasks"] = int(sum(t in x13 for t in ids[name]))
        c["task_ids_sha256"] = _sha(ids[name])
        views[name] = c

    v = verdicts.load_verdicts(root=root)
    rel = m.score.stack()
    x13_empty = v[v.unavailable]
    labels_count = pd.Series([u["label"] for u in units.values() for _ in range(u["tasks"])
                              if u["eligible"]]).value_counts()
    return _py({
        "about": ("S1.7 artefact list (plan S1.7, CLAIMS S0.10, S0.11, R10). Generated by "
                  "router.data.audit.build_artefacts(); committed before any router result."),
        "criteria": {"file": "docs/research/S1.7_audit_criteria.md", "commit": CRITERIA_COMMIT},
        "labels_commit": LABELS_COMMIT,
        "report": "docs/research/S1.7_audit.md",
        "success_rule": ("released score 1.0 for every family (S0.1, CLAIMS A3). "
                         "cruxeval_lenient_sensitivity counts CRUXEval >= 0.8 as success "
                         "(scoring.success(..., cruxeval_lenient=True)); a sensitivity check only"),
        "views": {
            "definitions": {
                "all_eligible": "every task of an eligible family (S1.4)",
                "filtered": "all_eligible minus confirmed-defect and undecided units (S0.10 headline view)",
                "sensitivity": "filtered plus undecided units (S0.10)",
                "headline": "filtered tasks with X13 verdicts (S0.11 headline set for cascade claims)",
                "paper": ("obs_matrix.json scores with missing cells zero-filled: reproduction of "
                          "the paper only, never used for claims"),
            },
            **views,
        },
        "label_task_counts_eligible": {k: int(labels_count[k]) for k in sorted(labels_count.index)},
        "units": units,
        "subset_rules": {"lcb": {
            "property": LCB_SUBSET_PROPERTY, "excluded_value": LCB_EXCLUDED_PLATFORM,
            "source": "external/sources/livecodebench.jsonl (livecodebench/code_generation_lite@0fe84c3)",
            "decision": "user, 2026-11-21 (criteria Amendment 7)",
            "excluded_unit": "lcb", "kept_unit": "lcb_kept",
            "excluded_tasks": len(leetcode), "kept_tasks": len(lcb_ids) - len(leetcode),
            "excluded_task_ids": leetcode}},
        "unavailable": {
            "missing_token_record_cells": int((m.unavailable_reason == "missing_token_record").sum().sum()),
            "x13_empty_cells": int(len(x13_empty)),
            "x13_empty_cells_released_score_1": int(sum(rel.get((t, mm), 0.0) >= SUCCESS_SCORE
                                                        for t, mm in zip(x13_empty.task_id, x13_empty.model))),
            "note": ("empty X13 cells and the Qwen3-Max missing token records are unavailable "
                     "(S0.11, user 2026-11-20); response_matrix.json empty cells keep their "
                     "released scores and are counted per view for S3's sensitivity check"),
        },
        "response_matrix": {
            "empty_cells_8_models": int(stats[stats.key.isin(set(m.models))].is_empty.sum()),
            "capped_cells_8_models": int(((stats.n_chars == STORAGE_CAP_CHARS) & ~stats.is_empty
                                          & stats.key.isin(set(m.models))).sum()),
        },
        "x13": {"tasks": len(x13), "eligible_tasks": int(sum(t in x13 for t in el))},
        "paper_view": {"zero_filled_cells": {s: crb.missing_cells(s, root) for s in SPLITS},
                       "use": "reproduction only"},
    })


def write_artefacts(path: Path = ARTEFACTS_YAML) -> dict:
    import yaml
    art = build_artefacts()
    path.write_text(yaml.safe_dump(art, sort_keys=False, allow_unicode=True, width=100))
    return art


def load_artefacts(path: Path = ARTEFACTS_YAML) -> dict:
    import yaml
    return yaml.safe_load(path.read_text())
