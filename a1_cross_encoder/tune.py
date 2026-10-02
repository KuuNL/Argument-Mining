"""A1.4: calibrate and tune A1's decision layer on dev, and report it honestly.

1. Temperature T fitted on the dev pairs; thresholds for Inference / Conflict / Rephrase by
   coordinate ascent on mean per-map ARI general F1 (decide.py; the fast metric equals the official one).
2. Honest estimate: leave-one-dev-episode-out cross-fitting (tune on 4 episodes, score the 5th).
3. Diagnostics: per relation type (pooled P/R/F1 over ordered pairs), Inference direction accuracy,
   and how often each type is predicted in maps that have none (the rare-label trap).

    python -m a1_cross_encoder.tune --run=v1             -> results/a1/<run>/thresholds.json
    python -m a1_cross_encoder.tune --run=v1 --cap=1     -> results/a1/<run>/thresholds_cap1.json (D046)
"""
import json
from collections import Counter, defaultdict

import numpy as np

from common.split import episodes, map_speakers
from a1_cross_encoder.decide import CODE, Dev, ece, fit_temperature, mean_f1, tune
from a1_cross_encoder.train import RUNS, arg, dirs

NAMES = {v: k for k, v in CODE.items()}


def diagnostics(dev: Dev, probs, th, cap=None) -> dict:
    tp, fp, fn = Counter(), Counter(), Counter()
    direction_right = direction_total = 0
    spurious = Counter()                     # predicted type in a map whose gold has none of it
    for n in dev.names:
        gold, pred = dev.gold[n], dev.predicted(probs, n, th, cap)
        for k in gold.keys() | pred.keys():
            g, p = gold.get(k), pred.get(k)
            if g == p:
                tp[g] += 1
            else:
                if p is not None: fp[p] += 1
                if g is not None: fn[g] += 1
        for (s, t), c in gold.items():       # direction: gold Inference whose pair we predicted as Inference
            if c == 0:
                if pred.get((s, t)) == 0:
                    direction_right += 1; direction_total += 1
                elif pred.get((t, s)) == 0:
                    direction_total += 1
        gold_types, pred_types = set(gold.values()), Counter(pred.values())
        for c, k in pred_types.items():
            if c not in gold_types:
                spurious[NAMES[c]] += 1
    per_type = {}
    for c in (0, 1, 2):
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        per_type[NAMES[c]] = {"gold": tp[c] + fn[c], "predicted": tp[c] + fp[c], "precision": round(p, 3),
                              "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}
    return {"per_type": per_type,
            "inference_direction_correct": round(direction_right / max(direction_total, 1), 4),
            "inference_pairs_found": direction_total,
            "maps_with_spurious_type": dict(spurious)}


def main():
    run = arg("run", "v1")
    cap = int(arg("cap", 0)) or None
    ckpt, out = dirs(run)
    dev = Dev(ckpt / "dev_probs.npy", data=RUNS[run].get("data"))
    t = fit_temperature(dev, dev.raw)
    probs = dev.calibrated(t)
    final = tune(dev, probs, dev.names, cap)

    ep_of, *_ = episodes(map_speakers())
    by_ep = defaultdict(list)
    for n in dev.names:
        by_ep[ep_of[int(n.removeprefix("nodeset"))]].append(n)
    held, folds = {}, {}
    for ep, names in sorted(by_ep.items()):
        fit = tune(dev, probs, [n for n in dev.names if n not in names], cap)
        held.update(dev.scores(probs, fit["th"], names, cap))
        folds[str(ep)] = {"maps": len(names), **fit["th"]}

    simple = dev.scores(dev.raw, None)
    tuned_all = dev.scores(probs, final["th"], cap=cap)
    report = {
        "run": run, "cap": cap, "temperature": t, "thresholds": final["th"],
        "ece_top_label": {"before_temperature": ece(dev.raw, dev.labels), "after_temperature": ece(probs, dev.labels)},
        "dev_ari_f1": {
            "argmax_decode": {"general": round(mean_f1(simple, 0), 2), "focused": round(mean_f1(simple, 1), 2)},
            "tuned_on_all_dev (optimistic)": {"general": round(mean_f1(tuned_all, 0), 2), "focused": round(mean_f1(tuned_all, 1), 2)},
            "cross_fitted (honest)": {"general": round(mean_f1(held, 0), 2), "focused": round(mean_f1(held, 1), 2)},
        },
        "folds": folds,
        "diagnostics_tuned": diagnostics(dev, probs, final["th"], cap),
        "diagnostics_argmax": diagnostics(dev, dev.raw, None),
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / ("thresholds.json" if cap is None else f"thresholds_cap{cap}.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
