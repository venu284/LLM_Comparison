# Runs INSIDE the BigCodeBench evaluation image (bigcodebench 0.2.4), never on the host.
# Mirrors bigcodebench.evaluate for chat-model output with its defaults:
#   solution = sanitize(response, entry_point)          (bigcodebench.sanitize, codegen step)
#   code     = code_prompt + "\n    pass\n" + solution   (calibrated=True)
#   untrusted_check(code, test, entry_point, max_as_limit=30*1024, max_data_limit=30*1024,
#                   max_stack_limit=10, min_time_limit=1, gt_time_limit=<canonical time> or 20)
# Canonical mode runs complete_prompt + "\n" + canonical_solution (bigcodebench.get_groundtruth).
import json
import sys
import time
import types

# The v0.2.1 image's `datasets` import fails (its pyarrow needs numpy 2; numpy is 1.21.2).
# bigcodebench.sanitize imports four bigcodebench.data helpers it does not use in sanitize();
# a stub module stands in for bigcodebench.data so the official sanitize code still runs.
_stub = types.ModuleType("bigcodebench.data")
for _n in ("get_bigcodebench", "load_solutions", "write_directory", "write_jsonl"):
    setattr(_stub, _n, None)
sys.modules["bigcodebench.data"] = _stub

from bigcodebench.eval import untrusted_check
from bigcodebench.sanitize import sanitize

p = json.load(open("cell.json"))
out = {}
if p.get("probe") == "block_sci":
    # Mechanism probe (S1.7 §1.1): the task's own code cannot import these libraries. The block
    # is installed after bigcodebench's reliability_guard (which itself imports matplotlib).
    import importlib.abc
    import bigcodebench.eval as _ev

    class _Block(importlib.abc.MetaPathFinder):
        BLOCKED = {"matplotlib", "sklearn", "seaborn", "scipy", "wordcloud"}

        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in self.BLOCKED:
                raise ModuleNotFoundError(f"probe: {name} blocked")
            return None

    _guard = _ev.reliability_guard

    def _guard_then_block(*a, **k):
        _guard(*a, **k)
        for m in [m for m in sys.modules if m.split(".")[0] in _Block.BLOCKED]:
            del sys.modules[m]
        sys.meta_path.insert(0, _Block())

    _ev.reliability_guard = _guard_then_block
if p["mode"] == "canonical":
    code = p["complete_prompt"] + "\n" + p["canonical_solution"]
else:
    sol = sanitize(p["response"], p["entry_point"])
    out["sanitized_chars"] = len(sol)
    code = p["code_prompt"] + "\n    pass\n" + sol
t0 = time.time()
stat, details = untrusted_check(code, p["test"], p["entry_point"], 30 * 1024, 30 * 1024, 10, 1,
                                p.get("gt_time") or 20)
out["elapsed"] = round(time.time() - t0, 3)
out["stat"] = stat
out["details"] = {k: str(v)[-1500:] for k, v in list(dict(details).items())[:8]}
print("@@CELL@@" + json.dumps(out))
