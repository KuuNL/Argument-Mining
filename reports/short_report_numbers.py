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

    # Why relations barely moved (section "Why relation mining barely moved since 2024" in both reports).
    disc = load("a4/discussion.json")
    systems = {"A1": "A1", "A2": "A2", "A4": "A4 both"}
    errors = {}
    for name, key in systems.items():
        wg, inv = disc["what_gold_becomes"][key], disc["invented_by_type"][key]
        right = sum(wg[t].get("right", 0) for t in wg)
        # Same relation type, other direction ("Inference, reversed"; for Rephrase / Conflict the same label, not right).
        reversed_ = wg["Inference"].get("Inference, reversed", 0) + wg["Rephrase"].get("Rephrase", 0) + wg["Conflict"].get("Conflict", 0)
        missed = sum(wg[t].get("None", 0) for t in wg)
        n_rel = sum(sum(v.values()) for v in wg.values())
        wrong_type = n_rel - right - reversed_ - missed
        invented = sum(inv.values())
        errors[name] = {"gold": n_rel, "right": right, "reversed": reversed_, "wrong_type": wrong_type,
                        "missed": missed, "invented": invented,
                        "share_of_errors_on_whether_related": round((missed + invented) / (missed + invented + wrong_type + reversed_), 3)}
    recall_by_distance = {}
    for name, key in systems.items():
        rec = disc["recall"][key]
        row = {}
        for dist in ("1 turn", "2-3 turns", "4-10 turns"):
            cells = [rec[f"{t}, {dist}"] for t in ("Inference", "Rephrase", "Conflict") if f"{t}, {dist}" in rec]
            n_gold = sum(c["gold"] for c in cells)
            row[dist] = {"gold": n_gold, "recall": round(sum(c["recall"] * c["gold"] for c in cells) / n_gold, 3)}
        recall_by_distance[name] = row
    # The honest protocol's cost for A1: thresholds tuned on all of dev vs cross-fitted by episode.
    a1_dev = load("a1/v1/thresholds.json")["dev_ari_f1"]
    protocol_cost_a1 = round(a1_dev["tuned_on_all_dev (optimistic)"]["general"] - a1_dev["cross_fitted (honest)"]["general"], 2)
    # How the per-map metric dilutes relation gains: test Inference F1 and ARI, A1 against the combination.
    a1_pt = a1["diagnostics"]["per_type"]
    dilution = {"A1": {"inference_f1": a1_pt["Inference"]["f1"], "ari_general": round(a1["ari"]["general"][2], 2)},
                "A1+A2+A4": {"inference_f1": pt["Inference"]["f1"], "ari_general": round(g("ari", "general"), 2)}}
    # Dev -> test change in ARI general for each system (honest dev against the single test run).
    dev_honest = load("a4/robust.json")["with_dev_thresholds (honest)"]
    test_ari = {"A1": a1["ari"]["general"][2], "A2": load("a2/test_global.json")["ari"]["general"][2],
                "A3": a3["ari"]["general"][2], "A4": a4["ari"]["general"][2]}
    dev_to_test = {k: round(test_ari[k] - dev_honest[k]["general"], 2) for k in test_ari}
    dev_to_test["A1+A2+A4"] = round(g("ari", "general") - dev_combo, 2)

    # README figures that use single systems only (the README does not present the combination).
    a2_pt = a2["diagnostics"]["per_type"]
    a2_gold = sum(t["gold"] for t in a2_pt.values())
    a2_pred = sum(t["predicted"] for t in a2_pt.values())
    a2_correct = sum(round(t["recall"] * t["gold"]) for t in a2_pt.values())
    singles = {"A1": a1, "A2": a2, "A3": a3, "A4": a4}
    readme = {
        "a2_test_pooled": {"gold": a2_gold, "correct": a2_correct, "predicted": a2_pred,
                           "precision": round(a2_correct / a2_pred, 3), "recall": round(a2_correct / a2_gold, 3),
                           "conflict_found": round(a2_pt["Conflict"]["recall"] * a2_pt["Conflict"]["gold"]),
                           "conflict_gold": a2_pt["Conflict"]["gold"]},
        "headroom_share_A2": closed(dev_honest["A2"]["general"]),
        "metric_dilution_A1_to_A2_test": {"inference_f1": [a1_pt["Inference"]["f1"], a2_pt["Inference"]["f1"]],
                                          "ari_general": [round(a1["ari"]["general"][2], 2), round(a2["ari"]["general"][2], 2)]},
        "test_ranges_single_systems": {
            t: [round(min(v[t]["general"][2] for v in singles.values()), 2), round(max(v[t]["general"][2] for v in singles.values()), 2)]
            for t in ("ari", "ilo", "global")},
    }

    out = {
        "readme_single_systems": readme,
        "relation_errors_dev_matched_density": errors,
        "relation_recall_by_distance_dev": recall_by_distance,
        "protocol_cost_a1_ari_general": protocol_cost_a1,
        "metric_dilution_test": dilution,
        "dev_to_test_ari_general": dev_to_test,
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
