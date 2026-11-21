"""S1.7 artefact audit.

Checkpoint 1 (this module so far) uses released scores and response **presence and length
only**. No function here returns or prints response text: `response_stats` reduces every response
to `is_empty`, `n_chars` and a SHA-256, and the text is dropped. The rules are pre-registered in
`docs/research/S1.7_audit_criteria.md` (§5); section numbers below refer to that file.

The view builder and `load_artefacts()` come in checkpoint 3.
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
