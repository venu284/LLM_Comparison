"""Count missing task-model cells in the uncleaned ACRouter matrix on the ID test split.

Usage: python ragged.py <CodeRouterBench snapshot dir>
"""

import collections
import json
import sys

import pandas as pd

D = sys.argv[1]
obs = json.load(open(f"{D}/obs_matrix.json"))
test_ids = set(pd.read_csv(f"{D}/id_test_results_long.csv").task_id)
sizes = [len(v) for k, v in obs.items() if k in test_ids]
print("test tasks", len(sizes), "cells", sum(sizes), "of", 8 * len(sizes),
      "tasks with <8 models", sum(n < 8 for n in sizes))
print(collections.Counter(m for k, v in obs.items() if k in test_ids for m in v))
