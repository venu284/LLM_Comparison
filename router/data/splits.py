"""Problem ids for every task (S1.6, checkpoint 1).

A *problem* groups the tasks that are variants of one source problem, so that S0.6's
development split and outer folds can keep them on one side (R13). Problem ids are built from
three kinds of link, and nothing else:

1. **Source key.** The source benchmark's own id, read from the task id (`source_key`). Tasks of
   different families that share a key are one problem:
   - HumanEval `n`: `humaneval`, `he_completion`, `he_testgen` (`HumanEval_<n>`), `hex_<lang>`
     (`<Lang>/<n>`, HumanEval-X) and `multiple_<lang>` (`HumanEval_<n>_<fn>`, MultiPL-E);
   - MBPP `n`: `mbpp`, `mbpp_completion`, `mbpp_testgen`;
   - BigCodeBench `n`: the five `bcb*` families;
   - LiveCodeBench question id: `lcb` and `lcb_testgen`.
   Every other task is its own key (`<family>/<id>`, with `ood/` in front for OOD tasks).
2. **DS-1000 origin.** DS-1000 builds perturbed problems from StackOverflow originals. The
   pinned release records, per problem, `perturbation_type` and `perturbation_origin_id`; the
   origin id counts within a library (it equals `library_problem_id` for every "Origin" row, and
   resolves within the library for every perturbed row), so a perturbed problem is linked to the
   row with the same `library` and `library_problem_id == perturbation_origin_id`.
3. **Identical content.** Tasks whose `content_sha256` (S1.5's router field) is equal. A task
   with no content links nothing this way.

`problem_id` is the connected component (union-find) of all three. The merge is **for splits
only**: tasks are never dropped or deduplicated for scoring (user, 2026-11-20).

Each problem is named after its smallest member key: source namespaces in the order of
`NAMESPACE_ORDER`, then a natural sort of the id, so the names never depend on iteration order.

`link` says which links a task has, in the order `source_key`, `ds1000_origin`, `content`,
joined by "+"; "none" when the task is a problem of its own:
- `source_key`: another task shares its source key;
- `ds1000_origin`: it is a perturbation of, or the origin of, another ds1000 task;
- `content`: it shares content with a task that source keys and DS-1000 origins do not already
  put in its problem (so this marks only what the content rule adds).

No score, cost, verdict or audit label is read here (S1.6 rule). `problems.csv` is written from
`derive()` by `router/experiments/s1_6_splits.py`; `annotate()` reads only that file.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROBLEMS_CSV = Path(__file__).with_name("problems.csv")
PROBLEM_COLUMNS = ["task_id", "problem_id", "link", "source_key", "content_sha256"]
LINKS = ("source_key", "ds1000_origin", "content")
NO_LINK = "none"
NAMESPACE_ORDER = ("he", "mbpp", "bcb", "lcb", "ds1000", "crux", "swe")
DS1000_ORIGIN_EXTRACT = "ds1000_origin"
_HE_EXACT = ("humaneval", "he_completion", "he_testgen")


# --------------------------------------------------------------------------- source keys
def _body(task_id: str, family: str) -> str:
    """The task id without its `dim<k>_<family>_` prefix (ID tasks) or `...::` prefix (OOD)."""
    if "::" in task_id:
        return task_id.rsplit("::", 1)[1]
    body = re.sub(r"^dim\d+_", "", task_id)
    if not body.startswith(family + "_"):
        raise ValueError(f"task id {task_id!r} does not carry its family {family!r}")
    return body[len(family) + 1:]


def source_key(task_id: str, family: str, split_kind: str = "id") -> str:
    """The source benchmark's own problem key for a task, namespaced by source.

    Bare upstream ids collide across sources (`ds1000_138` and `mbpp_138` are both "138"), so
    every key carries its source: `he/47`, `mbpp/71`, `bcb/10`, `lcb/1873_A`, `ds1000/138`,
    `crux/7`, `swe/<instance>`; a task with no shared source id is `<family>/<id>`.
    """
    from router.data.prompts import upstream_id

    if split_kind == "ood":
        return f"ood/{family}/{_body(task_id, family)}"
    up = upstream_id(task_id, family)
    if family in _HE_EXACT or family.startswith("hex_"):
        m = re.fullmatch(r"(?:HumanEval|[A-Za-z+]+)/(\d+)", up or "")
        if not m:
            raise ValueError(f"{task_id}: no HumanEval number in {up!r}")
        return f"he/{int(m.group(1))}"
    if family.startswith("multiple_"):
        m = re.match(r"HumanEval_(\d+)_", up or "")   # the function name may hold underscores
        if not m:
            raise ValueError(f"{task_id}: no HumanEval number in {up!r}")
        return f"he/{int(m.group(1))}"
    if family in ("mbpp", "mbpp_completion", "mbpp_testgen"):
        return f"mbpp/{int(up)}"
    if family.startswith("bcb"):
        m = re.fullmatch(r"BigCodeBench/(\d+)", up or "")
        if not m:
            raise ValueError(f"{task_id}: no BigCodeBench number in {up!r}")
        return f"bcb/{int(m.group(1))}"
    if family in ("lcb", "lcb_testgen"):
        return f"lcb/{up}"
    if family == "ds1000":
        return f"ds1000/{int(up)}"
    if family == "cruxeval":
        return f"crux/{int(up.removeprefix('sample_'))}"
    if family == "swebench":
        return f"swe/{_body(task_id, family)}"
    return f"{family}/{_body(task_id, family)}"


def key_order(key: str) -> tuple:
    """Sort key for problem names: namespace order, then a natural sort of the id."""
    ns, _, rest = key.partition("/")
    rank = NAMESPACE_ORDER.index(ns) if ns in NAMESPACE_ORDER else len(NAMESPACE_ORDER)
    parts = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in re.split(r"(\d+)", rest) if p)
    return (rank, ns, parts)


# --------------------------------------------------------------------------- DS-1000 origins
def read_ds1000_origin() -> tuple[dict, ...]:
    """Rows of the pinned `ds1000_origin` extract; empty when it is not fetched."""
    from router.data.prompts import read_extract

    return read_extract(DS1000_ORIGIN_EXTRACT)


def ds1000_origin_pairs(rows) -> list[tuple[int, int]]:
    """(problem_id, origin problem_id) for every perturbed DS-1000 problem, both global ids.

    The origin id is resolved within the problem's library. Raises if an origin does not resolve,
    or if an "Origin" row does not point to itself (the id space would then be wrong).
    """
    by_lib = {}
    for r in rows:
        k = (r["metadata.library"], r["metadata.library_problem_id"])
        if k in by_lib:
            raise ValueError(f"DS-1000: {k} is not unique")
        by_lib[k] = r
    pairs = []
    for r in rows:
        pid = r["metadata.problem_id"]
        target = by_lib.get((r["metadata.library"], r["metadata.perturbation_origin_id"]))
        if target is None:
            raise ValueError(f"DS-1000 problem {pid}: origin {r['metadata.perturbation_origin_id']} "
                             f"not found in {r['metadata.library']}")
        if r["metadata.perturbation_type"] == "Origin":
            if target is not r:
                raise ValueError(f"DS-1000 problem {pid}: an Origin row that does not point to itself")
            continue
        pairs.append((pid, target["metadata.problem_id"]))
    return sorted(pairs)


# --------------------------------------------------------------------------- union-find
class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[max(ra, rb)] = min(ra, rb)
        return True


def components(task_ids, *edge_lists) -> dict[str, str]:
    """task id -> a representative task id of its connected component."""
    uf = _UnionFind(task_ids)
    for edges in edge_lists:
        for a, b in edges:
            uf.union(a, b)
    return {t: uf.find(t) for t in task_ids}


def group_edges(df: pd.DataFrame, column: str) -> list[tuple[str, str]]:
    """Edges joining tasks with equal non-null `column` (a star per group)."""
    out = []
    for _, g in df[df[column].notna()].groupby(column, sort=True):
        ids = sorted(g.task_id)
        out += [(ids[0], t) for t in ids[1:]]
    return out


# --------------------------------------------------------------------------- derivation
def key_table() -> pd.DataFrame:
    """Every task in the family table with its source key and S1.5 hashes (needs `external/`)."""
    from router.data.prompts import prompt_table
    from router.data.sources import load_table

    table = load_table()
    pt = prompt_table().set_index("task_id")
    out = table.copy()
    out["source_key"] = [source_key(t, f, k) for t, f, k in zip(table.task_id, table.family, table.split_kind)]
    for c in ("content_sha256", "raw_sha256", "prompt_sha256", "joined", "upstream_id", "prompt_source"):
        out[c] = pt.loc[table.task_id, c].to_numpy()
    return out


def origin_edges(kt: pd.DataFrame, rows) -> list[tuple[str, str]]:
    """DS-1000 origin links as task-id edges."""
    by_pid = {int(k.split("/", 1)[1]): t for t, k in zip(kt.task_id, kt.source_key) if k.startswith("ds1000/")}
    pairs = ds1000_origin_pairs(rows)
    missing = sorted({p for pair in pairs for p in pair} - set(by_pid))
    if missing:
        raise ValueError(f"DS-1000 problems with no ds1000 task: {missing[:10]}")
    return [(by_pid[a], by_pid[b]) for a, b in pairs]


def derive(kt: pd.DataFrame | None = None, ds1000_rows=None) -> pd.DataFrame:
    """`problems.csv` rebuilt from the pinned data: one row per task, `PROBLEM_COLUMNS`.

    Raises if any component joins an ID task to an OOD task (S1.6 stop rule: the problem would
    need both the `ood` and an ID label).
    """
    kt = key_table() if kt is None else kt
    rows = read_ds1000_origin() if ds1000_rows is None else ds1000_rows
    if not rows:
        raise FileNotFoundError("ds1000_origin extract missing (run scripts/fetch_data.py)")
    ids = list(kt.task_id)
    key_e = group_edges(kt, "source_key")
    orig_e = origin_edges(kt, rows)
    cont_e = group_edges(kt, "content_sha256")
    before = components(ids, key_e, orig_e)
    final = components(ids, key_e, orig_e, cont_e)

    kind = dict(zip(kt.task_id, kt.split_kind))
    key = dict(zip(kt.task_id, kt.source_key))
    members: dict[str, list[str]] = {}
    for t in ids:
        members.setdefault(final[t], []).append(t)
    name = {}
    for root, ts in members.items():
        if len({kind[t] for t in ts}) > 1:
            raise ValueError(f"problem joins ID and OOD tasks: {sorted(ts)[:10]}")
        name[root] = min((key[t] for t in ts), key=key_order)

    key_size = kt.source_key.value_counts()
    has = {lk: set() for lk in LINKS}
    has["source_key"] = {t for t in ids if key_size[key[t]] > 1}
    has["ds1000_origin"] = {t for e in orig_e for t in e}
    # Every member of a content group that spans more than one pre-content component.
    for _, g in kt[kt.content_sha256.notna()].groupby("content_sha256"):
        if len({before[t] for t in g.task_id}) > 1:
            has["content"] |= set(g.task_id)
    link = ["+".join(lk for lk in LINKS if t in has[lk]) or NO_LINK for t in ids]

    return pd.DataFrame({
        "task_id": ids,
        "problem_id": [name[final[t]] for t in ids],
        "link": link,
        "source_key": kt.source_key.to_numpy(),
        "content_sha256": kt.content_sha256.to_numpy(),
    }, columns=PROBLEM_COLUMNS)


def write_problems(df: pd.DataFrame, path: Path = PROBLEMS_CSV) -> None:
    df.to_csv(path, index=False, lineterminator="\n")


# --------------------------------------------------------------------------- long table hook
@lru_cache(maxsize=1)
def load_problems(path: Path = PROBLEMS_CSV) -> pd.DataFrame:
    """The committed problem table (no external data needed)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values={"content_sha256": [""]})


def annotate(long: pd.DataFrame) -> pd.DataFrame:
    """Fill `problem_id` from the committed `problems.csv`. Reads only `task_id`."""
    pid = load_problems().set_index("task_id").problem_id
    unknown = set(long["task_id"]) - set(pid.index)
    if unknown:
        raise KeyError(f"no problem id for tasks {sorted(unknown)[:10]}")
    out = long.copy()
    out["problem_id"] = out["task_id"].map(pid).astype(object)
    return out
