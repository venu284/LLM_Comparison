"""S2: the legacy threshold is retired from router/sim/ and router/baselines/, and claim code
cannot reach the paper metric (O10). Both checks walk the import graph of the router package."""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEGACY_NAMES = {"LEGACY_ACCEPT_THRESHOLD", "legacy_accepted"}
LEGACY_MODULE = "router.experiments.m2_quickcheck_sim"
PAPER_MODULE = "router.sim.paper"
# Checkpoint B moved the 2026-11-10 baselines to experiments/m2_quickcheck_baselines/ and rebuilt
# router/baselines/ on router.sim; no baseline may import the quick-check simulator any more.
PAPER_IMPORTERS = ("router/experiments/s2_reproduce",)     # prefix: s2_reproduce*.py


def _module(path: Path) -> str:
    rel = path.relative_to(REPO).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


def _imports(path: Path) -> tuple[set[str], set[str]]:
    """(modules imported, names imported from modules) by one file."""
    mods, names = set(), set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
            for a in node.names:
                names.add(a.name)
                mods.add(f"{node.module}.{a.name}")
    return mods, names


def _router_files():
    return {_module(p): p for p in (REPO / "router").rglob("*.py")}


def _closure(start: str, files: dict) -> set[str]:
    """Every router module reachable from `start` through imports."""
    seen, todo = set(), [start]
    while todo:
        m = todo.pop()
        if m in seen or m not in files:
            continue
        seen.add(m)
        todo += [x for x in _imports(files[m])[0] if x.startswith("router")]
    return seen


def test_sim_never_reaches_the_legacy_threshold():
    files = _router_files()
    sim = [m for m in files if m.startswith("router.sim")]
    assert {"router.sim.stream", "router.sim.verifier", "router.sim.metrics",
            "router.sim.stats"} <= set(sim)
    for m in sim:
        for reached in _closure(m, files):
            mods, names = _imports(files[reached])
            if reached != "router.data.scoring":     # it defines the names; importing it is fine
                assert not (names & LEGACY_NAMES), (m, reached)
            assert LEGACY_MODULE not in mods and reached != LEGACY_MODULE, (m, reached)


def test_baselines_do_not_import_the_legacy_threshold():
    for p in (REPO / "router" / "baselines").glob("*.py"):
        mods, names = _imports(p)
        rel = str(p.relative_to(REPO))
        assert not (names & LEGACY_NAMES), rel
        assert not any(x.startswith(LEGACY_MODULE) or x.startswith("router.experiments")
                       for x in mods), rel


def test_only_reproductions_import_the_paper_metric():
    for p in list((REPO / "router").rglob("*.py")) + list((REPO / "scripts").rglob("*.py")):
        rel = str(p.relative_to(REPO))
        if rel == "router/sim/paper.py":
            continue
        mods, _ = _imports(p)
        if any(x == PAPER_MODULE or x.startswith(PAPER_MODULE + ".") for x in mods) or \
                ("router.sim" in mods and "paper" in _imports(p)[1]):
            assert rel.startswith(PAPER_IMPORTERS), rel


def test_claim_modules_cannot_reach_the_paper_metric():
    files = _router_files()
    for m in ("router.sim.metrics", "router.sim.stats", "router.sim.stream", "router.sim.verifier"):
        assert PAPER_MODULE not in _closure(m, files), m
