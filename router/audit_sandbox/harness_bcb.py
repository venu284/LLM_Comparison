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
