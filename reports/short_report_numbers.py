"""Numbers derived for reports/06-short-report.md, recomputed from committed result files.

    python reports/short_report_numbers.py      # from the repo root; writes results/short_report/derived.json

No model runs, no scoring, and data/evaluation/ is not read: only the per-map scores and diagnostics
that the final test runs already committed under results/.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MAPS = [f"map{i}" for i in range(1, 12)]


def load(rel):
    with open(RES / rel, encoding="utf-8") as f:
        return json.load(f)


def main():
    a1, a2 = load("a1/v1/test_global.json"), load("a2/test_global.json")
    a3, a4 = load("a3/test_global.json")["votes"], load("a4/test_global.json")
    comb = load("combination/test_global.json")
    per_map = {
        "A1": [a1["per_map_f1"][m]["ari"][0] for m in MAPS],
        "A2": [a2["per_map_f1"][m]["ari"][0] for m in MAPS],
        "A3": [a3["per_map_ari_f1"][m][0] for m in MAPS],
        "A4": [a4["per_map_ari_f1"][m][0] for m in MAPS],
        "A1+A2+A4": [comb["A1+A2+A4"]["per_map_ari_f1"][m][0] for m in MAPS],
    }
    # 95% bootstrap interval of each system's mean test ARI over the 11 maps (20,000 resamples, seed 0).
    rng = np.random.default_rng(0)
    boot = {}
    for k, v in per_map.items():
        v = np.array(v)
        means = rng.choice(v, (20000, len(v))).mean(1)
        boot[k] = [round(float(x), 1) for x in (v.mean(), *np.percentile(means, [2.5, 97.5]))]

    # Pooled test relations for the combination: correct = recall x gold, per relation type.
    pt = comb["A1+A2+A4"]["diagnostics"]["per_type"]
    gold = sum(t["gold"] for t in pt.values())
    pred = sum(t["predicted"] for t in pt.values())
    correct = sum(round(t["recall"] * t["gold"]) for t in pt.values())

    # Share of the dev gap between "predict no relations" (30.93) and the pipeline ceiling (98.64) that is closed.
    def closed(ari):
        return round((ari - 30.93) / (98.64 - 30.93), 3)

    dev_combo = load("combination/dev_report.json")["A1+A2+A4"]["honest_ari"]["general"]
    dev_a1 = load("a4/robust.json")["with_dev_thresholds (honest)"]["A1"]["general"]

    # Split of the Global margin over DFKI-MLST (overview paper: ARI 55.33 / 30.40, ILO 78.78 / 66.10).
    def g(t, s):
        return comb["A1+A2+A4"][t][s][2]

    ilo_half = {s: (g("ilo", s) - d) / 2 for s, d in (("general", 78.78), ("focused", 66.10))}
    ari_half = {s: (g("ari", s) - d) / 2 for s, d in (("general", 55.33), ("focused", 30.40))}

    out = {
        "headroom_share_combo": closed(dev_combo),
        "headroom_share_A1": closed(dev_a1),
        "combo_test_pooled": {"gold": gold, "correct": correct, "predicted": pred,
                              "precision": round(correct / pred, 3), "recall": round(correct / gold, 3)},
        "test_ari_boot95": boot,
        "ilo_test_per_map": {m: a1["per_map_f1"][m]["ilo"][0] for m in MAPS},
        "ilo_share_of_global_gain_general": round(ilo_half["general"] / (ilo_half["general"] + ari_half["general"]), 3),
        "ilo_share_of_global_gain_focused": round(ilo_half["focused"] / (ilo_half["focused"] + ari_half["focused"]), 3),
    }
    dst = RES / "short_report" / "derived.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
