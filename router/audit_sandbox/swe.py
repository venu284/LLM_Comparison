"""S1.7 checkpoint 2c: SWE-bench Lite in the audit sandbox (criteria Amendment 10).

SWE-bench's own harness (swebench 4.0.4) runs its evaluation containers as root with the network
on; a model patch is untrusted code, so it is not used to run cells. It is used only to build
the instance image (trusted: base OS, conda environment, repository at `base_commit`). Each cell
then runs in a fresh container of that image through `runner.run`: `--network none`, user 65534,
read-only root, tmpfs /work and /tmp, CPU/memory/PID/wall limits. Inside:

  1. /testbed is copied to /work/testbed (the root file system is read-only);
  2. the extracted patch is applied with SWE-bench's own cascade (GIT_APPLY_CMDS):
     `git apply --verbose`, `git apply --verbose --reject`, `patch --batch --fuzz=5 -p1 -i`;
  3. SWE-bench's eval script for the instance runs with /testbed rewritten to /work/testbed
     (test files reset, test_patch applied, the test command run between its output markers),
     with PYTHONPATH=/work/testbed so the patched copy shadows the image's editable install.

The log is graded on the host by swebench's own `get_eval_report` (resolved = every FAIL_TO_PASS
and PASS_TO_PASS test passes). Patch extraction from a chat response (Amendment 10):
  1. fenced blocks tagged `diff` or `patch`, concatenated in order;
  2. else fenced blocks whose body starts with `diff --git` or `--- `;
  3. else the unfenced text from the first line starting `diff --git` or `--- a/`;
  4. else no patch: the cell fails without running ("no_patch"), as SWE-bench's harness fails a
     prediction whose patch cannot be applied.
"""

from __future__ import annotations

import re

FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
GIT_APPLY_CMDS = ("git apply --verbose", "git apply --verbose --reject", "patch --batch --fuzz=5 -p1 -i")
APPLY_FAIL, APPLY_PASS = ">>>>> Patch Apply Failed", ">>>>> Applied Patch"


def extract_patch(text: str) -> tuple[str | None, str]:
    blocks = FENCE.findall(text or "")
    tagged = [b for t, b in blocks if t.strip().lower() in ("diff", "patch")]
    if tagged:
        return "".join(b if b.endswith("\n") else b + "\n" for b in tagged), "fenced_diff"
    starts = [b for _, b in blocks if b.lstrip().startswith(("diff --git", "--- "))]
    if starts:
        return "".join(b if b.endswith("\n") else b + "\n" for b in starts), "fenced_untagged"
    m = re.search(r"^(diff --git|--- a/)", text or "", re.M)
    if m:
        body = text[m.start():]
        return (body if body.endswith("\n") else body + "\n"), "unfenced"
    return None, "no_patch"


def run_script(eval_script: str) -> tuple[str, str]:
    """(run.sh, eval.sh) for the container; the patch is written to /work/patch.diff."""
    ev = eval_script.replace("/testbed", "/work/testbed")
    apply = "\n".join(
        f'if [ $applied = 0 ] && {c} /work/patch.diff; then applied=1; fi' for c in GIT_APPLY_CMDS)
    run = f"""#!/bin/bash
set -u
export HOME=/work
cp -r /testbed /work/testbed
cd /work/testbed
git config --global --add safe.directory '*'
git config --global user.email audit@example.invalid
git config --global user.name audit
applied=0
{apply}
if [ $applied = 0 ]; then echo "{APPLY_FAIL}"; exit 0; fi
echo "{APPLY_PASS}"
export PYTHONPATH=/work/testbed
bash /work/eval.sh
"""
    return run, ev
