"""FROZEN RECORD: the 2026-11-10 quick-check baselines (router/baselines/ before S2 checkpoint B).

Moved here verbatim so `m2_reproduce.py` and `tests/router/test_m1_m3.py` keep their 2026-11-10
numbers while `router/baselines/` is rebuilt on `router/sim/`. Do not use them in new code: they
run on the quick-check simulator (`m2_quickcheck_sim`), which accepts at the legacy threshold and
passes the released score and cost to the router.
"""
