"""S1.7 checkpoint 2: reading protocol (criteria §4) and sample draws (§8.1), on synthetic text."""

import numpy as np
import pandas as pd

from router.audit_sandbox.reading import read_cell
from router.data import audit


def test_reading_categories():
    r = read_cell("Here:\n```python\ndef task_func(x):\n    return x\n```\n", "python", "task_func")
    assert (r["code_format"], r["truncated"], r["language"], r["entry_point"]) == (
        "fenced_one", "no", "expected", "matches")
    r = read_cell("```python\ndef f(x):\n    return (x\n", "python", "task_func")
    assert (r["truncated"], r["entry_point"]) == ("yes", "differs")
    r = read_cell("```js\nfunction f(){}\n```\n```js\nf()\n```", "python")
    assert (r["code_format"], r["language"], r["entry_point"]) == ("fenced_many", "other", "n_a")
    r = read_cell("   \n", "python", "f")
    assert (r["empty"], r["code_format"], r["entry_point"]) == ("yes", "no_code", "absent")
    r = read_cell("I'm sorry, I can't help with that.", "python")
    assert (r["refusal"], r["code_format"]) == ("yes", "no_code")
    r = read_cell("<think>step one", "python")
    assert r["reasoning_only"] == "yes"
    r = read_cell("def f(a):\n    return a\n", "python", "f")
    assert (r["code_format"], r["entry_point"]) == ("unfenced_code", "matches")


def test_draw_is_seeded_and_without_replacement():
    fr = pd.DataFrame({"task_id": [f"t{i // 8}" for i in range(400)],
                       "model": [f"m{i % 8}" for i in range(400)]})
    a, b = audit.draw(fr, 110), audit.draw(fr, 110)
    assert len(a) == 97 and a.equals(b) and not a.duplicated().any()
    rng = np.random.default_rng(np.random.SeedSequence([0, 110]))
    assert list(a.index) == list(range(97))
    assert a.task_id.tolist() == fr.iloc[np.sort(rng.choice(400, 97, replace=False))].task_id.tolist()
    assert not audit.draw(fr, 111).equals(a)


def test_small_frame_takes_every_cell():
    fr = pd.DataFrame({"task_id": [f"t{i // 8}" for i in range(96)], "model": ["m"] * 96})
    assert len(audit.draw(fr, 120)) == 96
