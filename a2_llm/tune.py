"""A2.3: the decision layer on dev, the honest score, and diagnostics next to A1.

Every unordered pair of every dev map gets A2's five scores (a2_llm/parse.py: the mean over the answers
of the stated confidence). `Scores` gives them the interface A1's code expects, so A1's decision layer
applies unchanged:
1. thresholds for Inference / Conflict / Rephrase by coordinate ascent on mean per-map ARI general F1
   (a1_cross_encoder/decide.tune), fitted on all of dev and kept for the test run;
2. the honest estimate: leave-one-dev-episode-out cross-fitting, exactly as for A1 (D044);
3. diagnostics: A1's (per type, Inference direction, invented types), plus relations per map, fan-out,
   the run-to-run spread (each answer scored alone), and the overlap with A1's tuned dev predictions.

    python -m a2_llm.tune --run=dev-v1      -> results/a2/<run>/thresholds.json and dev_report.json
"""
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np

from common import RESULTS, ROOT
from common.aif import load_map, view
from common.score import ari_labels
from common.split import episodes, load_split, map_speakers
from a1_cross_encoder.decide import Dev as A1Dev, ari_f1, decode, mean_f1, tune
from a1_cross_encoder.tune import diagnostics
from a2_llm.parse import pair_scores
from a2_llm.runner import RUNS, jobs, load

OUT = RESULTS / "a2"
ALL = {"Inference": 0.01, "Conflict": 0.01, "Rephrase": 0.01}       # every listed relation
MEDIUM = {"Inference": 0.5, "Conflict": 0.5, "Rephrase": 0.5}       # medium or high, for a single answer


@lru_cache
def _answered(run: str) -> list:
    """(job, record) for every job of a run; built once (it builds the example pool)."""
    return [(job, load(job["key"], job["s"])) for job in jobs(RUNS[run])]


class Scores:
    """A2's pair scores over the dev maps; `use` picks which answers (sample numbers) are averaged."""

    def __init__(self, run: str, use=None, flat: bool = False):
        by_map = defaultdict(list)
        for job, rec in _answered(run):
            if use is None or job["sample"] in use:
                by_map[job["map"]].append((job, rec))
        paths = {p.stem: p for p in load_split("dev")}
        self.names, self.rows, self.gold, self.n_pairs, self.answers = sorted(by_map), [], {}, {}, {}
        self.idx, blocks, self.s, self.answer_lists = defaultdict(list), [], {}, {}
        for name in self.names:
            items = sorted(by_map[name], key=lambda x: x[0]["sample"])
            s = items[0][0]["s"]
            answers = [rec["attempts"][rec["answer"]]["result"]["structured_output"] for _, rec in items if rec and rec["ok"]]
            self.answers[name], self.s[name], self.answer_lists[name] = len(answers), s, answers
            rows, sc = pair_scores(s, answers, flat)
            self.idx[name] = list(range(len(self.rows), len(self.rows) + len(rows)))
            self.rows += rows
            blocks.append(sc)
            props = set(view(load_map(paths[name]), name).props)
            g = load_map(paths[name])
            self.gold[name] = {k: c for k, c in ari_labels(g, props).items() if k[1] in props and k[0] != k[1]}
            self.n_pairs[name] = len(props) * (len(props) - 1)
        self.probs = np.concatenate(blocks)

    def predicted(self, probs, name, th, cap=None):
        idx = self.idx[name]
        return decode([self.rows[k] for k in idx], probs[idx], th, cap)

    def scores(self, probs, th, names=None, cap=None):
        return {n: ari_f1(self.gold[n], self.predicted(probs, n, th, cap), self.n_pairs[n]) for n in (names or self.names)}


def both(scores) -> dict:
    return {"general": round(mean_f1(scores, 0), 2), "focused": round(mean_f1(scores, 1), 2)}


def cross_fitted(dev: Scores) -> tuple[dict, dict]:
    ep_of, *_ = episodes(map_speakers())
    by_ep = defaultdict(list)
    for n in dev.names:
        by_ep[ep_of[int(n.removeprefix("nodeset"))]].append(n)
    held, folds = {}, {}
    for ep, names in sorted(by_ep.items()):
        fit = tune(dev, dev.probs, [n for n in dev.names if n not in names])
        held.update(dev.scores(dev.probs, fit["th"], names))
        folds[str(ep)] = {"maps": len(names), **fit["th"]}
    return held, folds


def structure(dev: Scores, th) -> dict:
    """Relations per map and fan-out (propositions starting 2+ relations), predicted vs gold."""
    out = {}
    for who, rel in (("gold", lambda n: dev.gold[n]), ("predicted", lambda n: dev.predicted(dev.probs, n, th))):
        per_map, fan, props = [], 0, 0
        for n in dev.names:
            r = rel(n)
            per_map.append(len(r))
            fan += sum(1 for k in Counter(a for a, _ in r).values() if k >= 2)
            props += int((1 + (1 + 4 * dev.n_pairs[n]) ** 0.5) / 2)
        out[who] = {"relations_per_map": round(st.mean(per_map), 2), "props_starting_2plus": round(fan / props, 3)}
    return out


def overlap_with_a1(dev: Scores, th) -> dict:
    """Pairs each system gets right (exact label and direction), and an oracle that takes A2's label where it is
    right and A1's otherwise: the ceiling of combining them. Both use thresholds tuned on all of dev."""
    a1 = A1Dev(ROOT / "checkpoints" / "a1" / "v1" / "dev_probs.npy")
    a1_th = json.loads((RESULTS / "a1" / "v1" / "thresholds.json").read_text())
    a1_probs = a1.calibrated(a1_th["temperature"])
    count, oracle, a1_scores = Counter(), {}, {}
    for n in dev.names:
        g, p1, p2 = dev.gold[n], a1.predicted(a1_probs, n, a1_th["thresholds"]), dev.predicted(dev.probs, n, th)
        a1_scores[n] = ari_f1(g, p1, dev.n_pairs[n])
        label = lambda d, pair: next(((k, c) for k, c in d.items() if frozenset(k) == pair), None)
        combo = {}
        for pair in {frozenset(k) for d in (g, p1, p2) for k in d}:
            gl, l1, l2 = label(g, pair), label(p1, pair), label(p2, pair)
            r1, r2 = l1 == gl, l2 == gl
            count[{(True, True): "both right", (True, False): "only A1 right",
                   (False, True): "only A2 right", (False, False): "both wrong"}[(r1, r2)]] += 1
            pick = l2 if r2 else l1
            if pick:
                combo[pick[0]] = pick[1]
        oracle[n] = ari_f1(g, combo, dev.n_pairs[n])
    return {"pairs_with_any_relation": dict(count), "a1_tuned_on_all_dev": both(a1_scores), "oracle_per_pair": both(oracle)}


def main():
    run = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--run=")), "dev-v1")
    dev = Scores(run)
    final = tune(dev, dev.probs, dev.names)
    held, folds = cross_fitted(dev)
    spread = defaultdict(list)
    for i in range(RUNS[run]["samples"]):
        one = Scores(run, use={i})
        for name, th in (("all listed", ALL), ("medium+", MEDIUM)):
            spread[name].append(both(one.scores(one.probs, th)))
    report = {
        "run": run, "maps": len(dev.names), "answers_per_map": dict(Counter(dev.answers.values())),
        "thresholds": final["th"],
        "dev_ari_f1": {"all listed": both(dev.scores(dev.probs, ALL)),
                       "tuned on all dev (optimistic)": both(dev.scores(dev.probs, final["th"])),
                       "cross-fitted (honest)": both(held)},
        "a1_reference_honest": {"general": 50.50, "focused": 32.61},
        "folds": folds,
        "diagnostics_tuned": diagnostics(dev, dev.probs, final["th"]),
        "structure_tuned": structure(dev, final["th"]),
        "single_answer_spread": {name: {"general": [x["general"] for x in v], "focused": [x["focused"] for x in v]}
                                 for name, v in spread.items()},
        "overlap_with_a1": overlap_with_a1(dev, final["th"]),
    }
    (OUT / run).mkdir(parents=True, exist_ok=True)
    with open(OUT / run / "thresholds.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"run": run, "thresholds": final["th"], "objective": final["objective"]}, indent=1))
    with open(OUT / run / "dev_report.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
