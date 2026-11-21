# Superseded S1.7 sandbox runs (kept for the record)

- `s1_7_runs_bcb.gate1.jsonl`: the first BigCodeBench harness-gate run (reference solutions only).
  Every run errored: the v0.2.1 image cannot import `datasets` (pyarrow built for numpy 2). Fixed
  by stubbing `bigcodebench.data` (commit "harness gate passes").
- `s1_7_runs_py.controls1.jsonl`: the first reference-solution controls for the other Python
  families, before Amendment 5 (DS-1000 on the BigCodeBench image, programs passed with
  `python -c`).
- `s1_7_probe_block_sci_v1_runs.jsonl`: the first `block_sci` probe. It blocked the libraries
  before BigCodeBench's `reliability_guard`, which itself imports matplotlib, so every cell timed
  out. Fixed by installing the block after the guard.

None of these runs fed a summary or a mechanism.
