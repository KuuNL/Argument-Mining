"""A1 decision layer: pair probabilities -> relations, scored with an exact, fast ARI metric.

Decoding: each pair's best non-None label c is kept if P(c) >= the threshold of its relation type
(Inference, Conflict, Rephrase); with no thresholds the plain argmax over all five labels is used.
Thresholds can sit below 0.5, trading precision for recall where macro-F1 rewards it.

Used in A1.2/A1.3 to pick checkpoints (tuned_score: temperature + thresholds on all of dev);
A1.4 adds leave-one-episode-out cross-fitting, consistency repair and the report.
"""
import json
from collections import defaultdict

import numpy as np

from common import ROOT
from common.aif import load_map, view
from common.score import ari_labels
from common.split import load_split
from a1_cross_encoder.assemble import LABELS, relation

CACHE = ROOT / "cache" / "a1"
KIND = {1: "Inference", 2: "Inference", 3: "Conflict", 4: "Rephrase"}
CODE = {"Inference": 0, "Conflict": 1, "Rephrase": 2}       # official label ids; None = 3
GROUPS = ["Inference", "Conflict", "Rephrase"]
TH_GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)] + [1.01]   # 1.01 = never predict
T_GRID = [round(x, 2) for x in np.arange(0.5, 3.01, 0.05)]


def read(split):
    with open(CACHE / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def ari_f1(gold: dict, pred: dict, n_pairs: int):
    """(general, focused) macro (P, R, F1) over ordered proposition pairs, exactly as the official
    scorer computes it. gold / pred: {(source, target): code} without None; n_pairs = n(n-1)."""
    stats = defaultdict(lambda: [0, 0, 0])
    keys = gold.keys() | pred.keys()
    for k in keys:
        g, p = gold.get(k, 3), pred.get(k, 3)
        if g == p:
            stats[g][0] += 1
        else:
            stats[p][1] += 1
            stats[g][2] += 1
    stats[3][0] += n_pairs - len(keys)
    labels = set(gold.values()) | set(pred.values())
    if n_pairs > len(gold) or n_pairs > len(pred):
        labels.add(3)
    if not n_pairs:
        return None, None
    general = tuple(np.mean([_prf(*stats[c])[i] for c in labels]) for i in range(3))
    if not gold:
        return general, None
    fs = defaultdict(lambda: [0, 0, 0])
    for k, g in gold.items():
        p = pred.get(k, 3)
        if g == p:
            fs[g][0] += 1
        else:
            fs[p][1] += 1
            fs[g][2] += 1
    flabels = set(gold.values()) | {pred.get(k, 3) for k in gold}
    focused = tuple(np.mean([_prf(*fs[c])[i] for c in flabels]) for i in range(3))
    return general, focused


def decode(rows, probs, th: dict | None) -> dict:
    """{(source, target): code} for the pairs of one map."""
    pred = {}
    for r, p in zip(rows, probs):
        if th is None:
            c = int(np.argmax(p))
            if c == 0:
                continue
        else:
            c = 1 + int(np.argmax(p[1:]))
            if p[c] < th[KIND[c]]:
                continue
        kind, src, tgt = relation(r["a"], r["b"], LABELS[c])
        pred[(src, tgt)] = CODE[kind]
    return pred


class Dev:
    def __init__(self, probs_path=None, load=True):
        self.rows = read("dev")
        self.names = sorted({r["map"] for r in self.rows})
        self.idx = defaultdict(list)
        for k, r in enumerate(self.rows):
            self.idx[r["map"]].append(k)
        self.gold_aif = {p.stem: load_map(p) for p in load_split("dev") if p.stem in self.idx}
        self.gold, self.n_pairs = {}, {}
        for name, g in self.gold_aif.items():
            props = set(view(g, name).props)
            # Only pairs the official scorer sees: ordered pairs of two *different* propositions
            # (gold contains a few self-loops, which it never scores).
            self.gold[name] = {k: c for k, c in ari_labels(g, props).items() if k[1] in props and k[0] != k[1]}
            self.n_pairs[name] = len(props) * (len(props) - 1)
        self.labels = np.array([r["label_id"] for r in self.rows])
        self.raw = np.load(probs_path) if load else None

    def calibrated(self, t: float, probs=None) -> np.ndarray:
        z = np.log(np.clip(self.raw if probs is None else probs, 1e-12, 1.0)) / t
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predicted(self, probs, name, th):
        idx = self.idx[name]
        return decode([self.rows[k] for k in idx], probs[idx], th)

    def scores(self, probs, th, names=None):
        return {n: ari_f1(self.gold[n], self.predicted(probs, n, th), self.n_pairs[n]) for n in (names or self.names)}


def mean_f1(scores: dict, which: int = 0) -> float:
    return 100 * float(np.mean([s[which][2] for s in scores.values() if s[which] is not None]))


def fit_temperature(dev: Dev, probs) -> float:
    nll = []
    for t in T_GRID:
        p = dev.calibrated(t, probs)
        nll.append(-np.mean(np.log(np.clip(p[np.arange(len(p)), dev.labels], 1e-12, 1.0))))
    return T_GRID[int(np.argmin(nll))]


def tune(dev: Dev, probs, names) -> dict:
    """Coordinate ascent (2 passes) on mean ARI general F1 over `names`."""
    th = {g: 0.5 for g in GROUPS}
    best = mean_f1(dev.scores(probs, th, names))
    for _ in range(2):
        for g in GROUPS:
            for v in TH_GRID:
                f = mean_f1(dev.scores(probs, {**th, g: v}, names))
                if f > best + 1e-9:
                    best, th = f, {**th, g: v}
    return {"th": th, "objective": round(best, 2)}


def tuned_score(dev: Dev, probs) -> dict:
    """Temperature and thresholds fitted on all of dev: the score used to pick checkpoints."""
    t = fit_temperature(dev, probs)
    cal = dev.calibrated(t, probs)
    fit = tune(dev, cal, dev.names)
    return {"ari_tuned_general": fit["objective"], "ari_tuned_focused": round(mean_f1(dev.scores(cal, fit["th"]), 1), 2),
            "tuned_temperature": t, **{f"th_{g.lower()}": v for g, v in fit["th"].items()}}


def simple_score(dev: Dev, probs) -> dict:
    s = dev.scores(probs, None)
    return {"ari_general_f1": round(mean_f1(s, 0), 2), "ari_focused_f1": round(mean_f1(s, 1), 2)}
