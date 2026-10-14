"""The labeller's decision layer on Claude's votes, scored honestly by leave-one-dev-episode-out cross-fitting.

ClaudeDev has the interface of ilo.tune.Dev (rows, raw, idx, names, gold, n_pairs, gold_aif, links(),
scores()), with gold built exactly as Dev builds it and rows / raw from ilo_llm.parse over a run's
answers. ilo.tune.tune() then fits the link threshold on vote share and the per-type-group thresholds
(GROUPINGS["shared"]), with one-to-one matching (ilo.decode.decode_map). There is no temperature:
vote shares are used as they are. The cross-fitting loop is ilo.tune.main()'s, over the same 5 episodes.

    python -m ilo_llm.tune --run=dev     -> results/ilo_llm/<run>/thresholds.json
"""
import gzip
import json
from collections import defaultdict

import numpy as np

from common.aif import load_map, strip_to_input
from common.score import ilo_labels, score_map
from common.serialise import serialise
from common.split import episodes, load_split, map_speakers
from common.timing import timed
from ilo.data import LABELS
from ilo.decode import predicted_map
from ilo.tune import GROUPINGS, Dev, ilo_f1, mean_f1, tune
from ilo_llm.parse import rows_probs, votes
from ilo_llm.runner import OUT, RUNS, arg, evaluate, resolve


class ScoredDev(Dev):
    """ilo.tune.Dev's interface over any system's candidate rows and (n, 11) probabilities (ilo.data.LABELS order)."""

    def __init__(self, per_map: dict, gold_aif: dict):
        """per_map: name -> (rows, probs); gold_aif: name -> gold map."""
        self.names = sorted(per_map)
        self.gold_aif = {n: gold_aif[n] for n in self.names}
        self.rows, parts = [], []
        for n in self.names:
            rows, probs = per_map[n]
            self.rows += rows
            parts.append(probs)
        self.raw = np.concatenate(parts)
        self.idx = defaultdict(list)
        for k, r in enumerate(self.rows):
            self.idx[r["map"]].append(k)
        self.gold, self.n_pairs = {}, {}
        for name, g in self.gold_aif.items():                       # as ilo.tune.Dev.__init__
            props = {n["nodeID"] for n in g["nodes"] if n["type"] == "I"}
            locs = {n["nodeID"] for n in g["nodes"] if n["type"] == "L"}
            self.gold[name] = {k: y for k, y in ilo_labels(g, props | locs).items() if k[0] in locs and k[1] in props}
            self.n_pairs[name] = len(locs) * len(props)
        self.labels = None


class ClaudeDev(ScoredDev):
    def __init__(self, maps: dict, gold_aif: dict):
        """maps: name -> (Serialised, [answers]); gold_aif: name -> gold map. Maps with no answer predict nothing."""
        super().__init__({n: rows_probs(s, votes(s, answers)) if answers else ([], np.zeros((0, len(LABELS))))
                          for n, (s, answers) in maps.items()}, gold_aif)


def run_answers(run: str) -> tuple[dict, dict, list[str]]:
    """From results/ilo_llm/<run>/responses.jsonl.gz: (name -> (Serialised, accepted answers), gold maps,
    maps with no accepted answer). Each record is re-judged with runner.evaluate."""
    cfg = resolve(RUNS[run])
    paths = {p.stem: p for p in load_split(cfg["split"])}
    gold = {n: load_map(paths[n]) for n in cfg["maps"]}
    maps = {n: (serialise(strip_to_input(gold[n]), n, []), []) for n in cfg["maps"]}
    with gzip.open(OUT / run / "responses.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            s, answers = maps[rec["map"]]
            ok = [a for a in rec["attempts"] if evaluate(a["result"], s) is None]
            if ok:
                answers.append(ok[0]["result"]["structured_output"])
    return maps, gold, [n for n, (_, a) in maps.items() if not a]


def load_run(run: str) -> ClaudeDev:
    maps, gold, _ = run_answers(run)
    return ClaudeDev(maps, gold)


def cross_fit(dev: Dev, probs: np.ndarray, grouping: dict) -> tuple[dict, dict]:
    """ilo.tune.main()'s leave-one-episode-out loop: thresholds never see the maps they are scored on."""
    ep_of, *_ = episodes(map_speakers())
    by_ep = defaultdict(list)
    for name in dev.names:
        by_ep[ep_of[int(name.removeprefix("nodeset"))]].append(name)
    held, folds = {}, {}
    for ep, names in sorted(by_ep.items()):
        fit = tune(dev, probs, grouping, [n for n in dev.names if n not in names])
        held.update(dev.scores(probs, fit["params"]["link_min"], fit["type_min"], names))
        folds[str(ep)] = {"maps": len(names), **fit["params"]}
    return held, folds


def tuned_report(dev: ScoredDev, run: str, missing: list) -> dict:
    """The labeller's decision layer on any ScoredDev: thresholds on all of dev, cross-fitted honest score, argmax."""
    grouping = GROUPINGS["shared"]
    # The fast metric must equal the official scorer: check every map under the argmax decode.
    for name in dev.names:
        links = dev.links(dev.raw, name, 0.5, None)
        official = score_map(dev.gold_aif[name], predicted_map(dev.gold_aif[name], links))["ilo"]
        fast = ilo_f1(dev.gold[name], {(t, p): y for t, p, y in links}, dev.n_pairs[name])
        for o, f in zip(official, fast):
            assert (o is None and f is None) or all(abs(a - b) < 1e-9 for a, b in zip(o, f)), (name, o, f)
    final = tune(dev, dev.raw, grouping, dev.names)
    held, folds = cross_fit(dev, dev.raw, grouping)
    argmax = dev.scores(dev.raw, 0.5, None)
    return {
        "run": run, "groups": "shared", "temperature": None, "maps": len(dev.names), "maps_without_answers": missing,
        "link_min": final["params"]["link_min"], "type_min": final["type_min"],
        "dev_ilo_f1": {
            "argmax_decode": {"general": round(mean_f1(argmax, 0), 2), "focused": round(mean_f1(argmax, 1), 2)},
            "tuned_on_all_dev (optimistic)": {"general": final["objective"],
                                             "focused": round(mean_f1(dev.scores(dev.raw, final["params"]["link_min"], final["type_min"]), 1), 2)},
            "cross_fitted (honest)": {"general": round(mean_f1(held, 0), 2), "focused": round(mean_f1(held, 1), 2)},
        },
        "folds": folds, "fast_metric_matches_official": True,
        "per_map_cross_fitted": {n: held[n] for n in dev.names},     # [general (P, R, F1), focused or None]
    }


def write_report(report: dict, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "thresholds.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "per_map_cross_fitted"}, indent=1))


def main():
    run = arg("run", "dev")
    with timed("ilo_llm", f"tune:{run}") as row:
        maps, gold, missing = run_answers(run)
        report = tuned_report(ClaudeDev(maps, gold), run, missing)
        row.update(report["dev_ilo_f1"]["cross_fitted (honest)"])
    write_report(report, OUT / run)


if __name__ == "__main__":
    main()
