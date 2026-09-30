"""Run the unmodified official scorers on one (gold, prediction) pair — the reference the port is checked against.

The scripts hard-code their input paths ('gold/', 'test/', 'test/baseline_ilo/') and print
their results, so we execute their source with `open` redirected and `print` captured.
They are slow (nested loops over edges), so this is for verification only.
"""
import builtins
import json
import tempfile
from pathlib import Path

from . import ROOT

SCRIPTS = {"ari": ROOT / "reference/eval-scripts/eval_argument.py",
           "ilo": ROOT / "reference/eval-scripts/eval_illocutions.py"}


def run_official(task: str, gold: dict, pred: dict) -> tuple:
    """(general, focused) as (P, R, F1) tuples, like score.score_map; focused is None if the script raises on it."""
    with tempfile.TemporaryDirectory() as tmp:
        gold_path, pred_path = Path(tmp, "gold.json"), Path(tmp, "pred.json")
        gold_path.write_text(json.dumps({"AIF": gold}), encoding="utf-8")   # the scripts read gold['AIF']
        pred_path.write_text(json.dumps(pred), encoding="utf-8")

        def fake_open(path, *a, **k):
            path = {"gold/": gold_path, "test/": pred_path, "test/baseline_ilo/": pred_path}.get(path, path)
            return builtins.open(path, *a, **k)

        results = {}

        def capture(*args, **_):
            if args and args[0] in ("General", "Focused"):
                results[args[0]] = tuple(float(x) for x in args[1][:3])

        env = {"__name__": "__main__", "open": fake_open, "print": capture}
        try:
            exec(compile(SCRIPTS[task].read_text(encoding="utf-8"), str(SCRIPTS[task]), "exec"), env)
        except ValueError:          # sklearn on an empty focused list
            pass
    return results.get("General"), results.get("Focused")
