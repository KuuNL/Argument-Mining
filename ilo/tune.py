"""Step 3.4: calibrate the ILO labeller and tune its decision layer on dev.

1. Temperature T: one scalar, fitted on the dev candidate pairs (lowest negative log-likelihood).
2. Thresholds: link_min on 1 - P(None), and a minimum P(type | link) for each type group below
   which a link falls back to Asserting. Chosen by coordinate ascent on the official metric:
   mean per-map ILO general F1, computed exactly (ilo_f1 equals the official scorer) but fast.
3. Honest estimate: leave-one-dev-episode-out cross-fitting (tune on 4 episodes, score the 5th).
   The final thresholds are tuned on all of dev.

    python -m ilo.tune [--run=v2]                    # groups: 3 question types each, 6 rare types shared
    python -m ilo.tune [--run=v2] --groups=per-type  # every type its own threshold
Without --run, the selected run (results/ilo/selected.json) is used. Writes results/ilo/<run>/.
"""
import json
import sys
from collections import defaultdict

import numpy as np

from common.aif import load_map
from common.score import ilo_labels, score_map
from common.split import episodes, load_split, map_speakers
from ilo.data import LABELS
from ilo.decode import NONE, decode_map, predicted_map
from ilo.train import arg, dirs, read, selected_run

QUESTIONS = ["Pure Questioning", "Assertive Questioning", "Rhetorical Questioning"]
RARE = ["Default Illocuting", "Agreeing", "Challenging", "Disagreeing", "Arguing", "Restating"]
GROUPINGS = {
    "shared": {**{q: q for q in QUESTIONS}, **{r: "rare" for r in RARE}},
    "per-type": {t: t for t in QUESTIONS + RARE},
}
# Widened twice because the optimum sat on the floor (0.20 in v1's first tuning, 0.05 during v2).
LINK_GRID = [0.01, 0.02, 0.03] + [round(x, 2) for x in np.arange(0.05, 0.86, 0.05)]
TYPE_GRID = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]   # 1.01 = never predict this type
T_GRID = [round(x, 2) for x in np.arange(0.5, 3.01, 0.05)]


# ---------------------------------------------------------------- exact, fast ILO metric

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def ilo_f1(gold: dict, pred: dict, n_pairs: int):
    """(general, focused) macro (P, R, F1) over (L, I) pairs, as sklearn computes it in the official
    scorer. gold / pred: {(L, I): label} without None entries; n_pairs = |L| x |I|."""
    stats = defaultdict(lambda: [0, 0, 0])
    keys = gold.keys() | pred.keys()
    for k in keys:
        g, p = gold.get(k, "None"), pred.get(k, "None")
        if g == p:
            stats[g][0] += 1
        else:
            stats[p][1] += 1
            stats[g][2] += 1
    stats["None"][0] += n_pairs - len(keys)
    labels = set(gold.values()) | set(pred.values())
    if n_pairs > len(gold) or n_pairs > len(pred):
        labels.add("None")
    general = tuple(np.mean([_prf(*stats[c])[i] for c in labels]) for i in range(3))
    if not gold:
        return general, None
    fs = defaultdict(lambda: [0, 0, 0])
    for k, g in gold.items():
        p = pred.get(k, "None")
        if g == p:
            fs[g][0] += 1
        else:
            fs[p][1] += 1
            fs[g][2] += 1
    flabels = set(gold.values()) | {pred.get(k, "None") for k in gold}
    focused = tuple(np.mean([_prf(*fs[c])[i] for c in flabels]) for i in range(3))
    return general, focused


# ---------------------------------------------------------------- dev data

class Dev:
    def __init__(self, probs_path=None, load=True):
        self.rows = read("dev")
        self.raw = np.load(probs_path or dirs(selected_run())[0] / "dev_probs.npy") if load else None
        names = sorted({r["map"] for r in self.rows})
        self.gold_aif = {p.stem: load_map(p) for p in load_split("dev") if p.stem in names}
        self.idx = defaultdict(list)
        for k, r in enumerate(self.rows):
            self.idx[r["map"]].append(k)
        self.gold, self.n_pairs = {}, {}
        for name, g in self.gold_aif.items():
            props = {n["nodeID"] for n in g["nodes"] if n["type"] == "I"}
            locs = {n["nodeID"] for n in g["nodes"] if n["type"] == "L"}
            self.gold[name] = {k: y for k, y in ilo_labels(g, props | locs).items() if k[0] in locs and k[1] in props}
            self.n_pairs[name] = len(locs) * len(props)
        self.names = names
        self.labels = np.array([r["label_id"] for r in self.rows])

    def calibrated(self, t: float) -> np.ndarray:
        z = np.log(np.clip(self.raw, 1e-12, 1.0)) / t
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def links(self, probs, name, link_min, type_min):
        idx = self.idx[name]
        return decode_map([self.rows[k] for k in idx], probs[idx], link_min, type_min)

    def scores(self, probs, link_min, type_min, names=None):
        out = {}
        for name in names or self.names:
            pred = {(t, p): y for t, p, y in self.links(probs, name, link_min, type_min)}
            out[name] = ilo_f1(self.gold[name], pred, self.n_pairs[name])
        return out


def mean_f1(scores: dict, which: int = 0) -> float:
    vals = [s[which][2] for s in scores.values() if s[which] is not None]
    return 100 * float(np.mean(vals))


# ---------------------------------------------------------------- fitting

def fit_temperature(dev: Dev) -> float:
    nll = []
    for t in T_GRID:
        p = dev.calibrated(t)
        nll.append(-np.mean(np.log(np.clip(p[np.arange(len(p)), dev.labels], 1e-12, 1.0))))
    return T_GRID[int(np.argmin(nll))]


def tune(dev: Dev, probs, grouping: dict, names) -> dict:
    """Coordinate ascent (2 passes) on mean ILO general F1 over `names`."""
    groups = sorted(set(grouping.values()))
    params = {"link_min": 0.5, **{g: 0.0 for g in groups}}

    def objective(p):
        type_min = {t: p[g] for t, g in grouping.items()}
        return mean_f1(dev.scores(probs, p["link_min"], type_min, names))

    best = objective(params)
    for _ in range(2):
        for key in ["link_min"] + groups:
            for v in (LINK_GRID if key == "link_min" else TYPE_GRID):
                trial = {**params, key: v}
                f = objective(trial)
                if f > best + 1e-9:
                    best, params = f, trial
    return {"params": params, "type_min": {t: params[g] for t, g in grouping.items()}, "objective": round(best, 2)}


def tuned_score(dev: Dev, probs: np.ndarray, grouping=GROUPINGS["shared"]) -> dict:
    """Fit temperature and thresholds on all of dev and return the tuned score (used to pick checkpoints)."""
    dev.raw = probs
    t = fit_temperature(dev)
    cal = dev.calibrated(t)
    fit = tune(dev, cal, grouping, dev.names)
    return {"ilo_tuned_general": fit["objective"],
            "ilo_tuned_focused": round(mean_f1(dev.scores(cal, fit["params"]["link_min"], fit["type_min"]), 1), 2),
            "tuned_temperature": t, "tuned_link_min": fit["params"]["link_min"]}


def main():
    mode = arg("groups", "shared")
    run = arg("run", selected_run())
    ckpt, out = dirs(run)
    grouping = GROUPINGS[mode]
    dev = Dev(ckpt / "dev_probs.npy")

    # The fast metric must equal the official scorer: check every dev map under the argmax decode.
    for name in dev.names:
        links = dev.links(dev.raw, name, 0.5, None)
        official = score_map(dev.gold_aif[name], predicted_map(dev.gold_aif[name], links))["ilo"]
        fast = ilo_f1(dev.gold[name], {(t, p): y for t, p, y in links}, dev.n_pairs[name])
        for o, f in zip(official, fast):
            assert (o is None and f is None) or all(abs(a - b) < 1e-9 for a, b in zip(o, f)), (name, o, f)

    t = fit_temperature(dev)
    probs = dev.calibrated(t)
    final = tune(dev, probs, grouping, dev.names)

    # Leave-one-episode-out: thresholds never see the maps they are scored on.
    ep_of, *_ = episodes(map_speakers())
    by_ep = defaultdict(list)
    for name in dev.names:
        by_ep[ep_of[int(name.removeprefix("nodeset"))]].append(name)
    held = {}
    folds = {}
    for ep, names in sorted(by_ep.items()):
        fit = tune(dev, probs, grouping, [n for n in dev.names if n not in names])
        held.update(dev.scores(probs, fit["params"]["link_min"], fit["type_min"], names))
        folds[str(ep)] = {"maps": len(names), **fit["params"]}
    argmax = dev.scores(dev.raw, 0.5, None)
    report = {
        "run": run, "groups": mode, "temperature": t, "link_min": final["params"]["link_min"], "type_min": final["type_min"],
        "dev_ilo_f1": {
            "argmax_decode (3.3)": {"general": round(mean_f1(argmax, 0), 2), "focused": round(mean_f1(argmax, 1), 2)},
            "tuned_on_all_dev (optimistic)": {"general": final["objective"],
                                             "focused": round(mean_f1(dev.scores(probs, final["params"]["link_min"], final["type_min"]), 1), 2)},
            "cross_fitted (honest)": {"general": round(mean_f1(held, 0), 2), "focused": round(mean_f1(held, 1), 2)},
        },
        "folds": folds, "fast_metric_matches_official": True,
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"thresholds_{mode}.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
