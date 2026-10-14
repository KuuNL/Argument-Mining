"""The labeller's decision layer on Jev's probabilities (D099): ilo_llm.tune's ScoredDev and tuned_report, unchanged.

No temperature (as for Claude): the link threshold and per-type-group thresholds are fitted on all of dev, and the
honest score is cross-fitted by leaving one dev episode out (the same 5 episodes as ilo/tune.py).

    python -m ilo_jev.tune --run=dev     -> results/ilo_jev/<run>/thresholds.json
"""
import numpy as np

from common.aif import load_map
from common.split import load_split
from common.timing import timed
from ilo.data import LABELS
from ilo_jev.parse import rows_probs
from ilo_jev.runner import OUT, answers, arg, builds, config
from ilo_llm.tune import ScoredDev, tuned_report, write_report


def load_run(run: str) -> tuple[ScoredDev, list[str]]:
    """The run's maps as a ScoredDev (maps without a complete answer predict nothing), and those maps."""
    cfg = config(run)
    paths = {p.stem: p for p in load_split(cfg["split"])}
    gold = {n: load_map(paths[n]) for n in cfg["maps"]}
    bs, ans = builds(run), answers(run)
    per_map = {n: rows_probs(bs[n], ans[n]) if n in ans else ([], np.zeros((0, len(LABELS)))) for n in cfg["maps"]}
    return ScoredDev(per_map, gold), [n for n in cfg["maps"] if n not in ans]


def main():
    run = arg("run", "dev")
    with timed("ilo_jev", f"tune:{run}") as row:
        dev, missing = load_run(run)
        report = tuned_report(dev, run, missing)
        row.update(report["dev_ilo_f1"]["cross_fitted (honest)"])
    write_report(report, OUT / run)


if __name__ == "__main__":
    main()
