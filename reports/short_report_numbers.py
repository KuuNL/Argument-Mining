"""Numbers derived for the README and the short / final reports, recomputed from committed result files.

    python reports/short_report_numbers.py      # from the repo root; writes results/short_report/derived.json

Single systems only (A1-A4). No model runs, no scoring, and data/evaluation/ is not read: only the per-map
scores and diagnostics that the final test runs already committed under results/.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MAPS = [f"map{i}" for i in range(1, 12)]
# 2024 winner (DFKI-MLST) on the 11 test maps, general / focused F1: overview paper, Tables 1-3.
DFKI = {"ari": (55.33, 30.40), "ilo": (78.78, 66.10), "global": (67.05, 48.25)}
# Dev references (D044): no relations predicted, and every candidate pair labelled right.
DEV_MAJORITY, DEV_CEILING = 30.93, 98.64


def load(rel):
    with open(RES / rel, encoding="utf-8") as f:
        return json.load(f)


def main():
    S = {"A1": load("a1/v1/test_global.json"), "A2": load("a2/test_global.json"),
         "A3": load("a3/test_global.json")["votes"], "A4": load("a4/test_global.json")}
    per_map = {"A1": [S["A1"]["per_map_f1"][m]["ari"][0] for m in MAPS],
               "A2": [S["A2"]["per_map_f1"][m]["ari"][0] for m in MAPS],
               "A3": [S["A3"]["per_map_ari_f1"][m][0] for m in MAPS],
               "A4": [S["A4"]["per_map_ari_f1"][m][0] for m in MAPS]}

    # 95% bootstrap interval of each system's mean test ARI over the 11 maps (20,000 resamples, seed 0).
    rng = np.random.default_rng(0)
    boot = {}
    for k, v in per_map.items():
        v = np.array(v)
        means = rng.choice(v, (20000, len(v))).mean(1)
        boot[k] = [round(float(x), 1) for x in (v.mean(), *np.percentile(means, [2.5, 97.5]))]

    # Each system's Global margin over DFKI, split into the illocution half and the relation half
    # (Global is the mean of ARI and ILO, so each task contributes half of its own gain).
    split = {}
    for k, d in S.items():
        split[k] = {}
        for si, s in enumerate(("general", "focused")):
            ilo = (d["ilo"][s][2] - DFKI["ilo"][si]) / 2
            ari = (d["ari"][s][2] - DFKI["ari"][si]) / 2
            split[k][s] = {"from_ilo": round(ilo, 2), "from_ari": round(ari, 2),
                           "global_gain": round(d["global"][s][2] - DFKI["global"][si], 2)}

    # Pooled test relations per system: correct = recall x gold, per relation type.
    pooled = {}
    for k, d in S.items():
        pt = d["diagnostics"]["per_type"]
        gold = sum(t["gold"] for t in pt.values())
        pred = sum(t["predicted"] for t in pt.values())
        correct = sum(round(t["recall"] * t["gold"]) for t in pt.values())
        pooled[k] = {"gold": gold, "correct": correct, "predicted": pred,
                     "precision": round(correct / pred, 3), "recall": round(correct / gold, 3),
                     "inference_f1": pt["Inference"]["f1"],
                     "conflict_found": round(pt["Conflict"]["recall"] * pt["Conflict"]["gold"]),
                     "conflict_gold": pt["Conflict"]["gold"]}

    # Share of the dev gap between "predict no relations" and the pipeline ceiling that each system closes.
    dev_honest = load("a4/robust.json")["with_dev_thresholds (honest)"]
    headroom = {k: round((dev_honest[k]["general"] - DEV_MAJORITY) / (DEV_CEILING - DEV_MAJORITY), 3) for k in S}
    dev_to_test = {k: round(S[k]["ari"]["general"][2] - dev_honest[k]["general"], 2) for k in S}

    # Why relations barely moved (final report section 10, short report section 6).
    disc = load("a4/discussion.json")
    views = {"A1": "A1", "A2": "A2", "A4": "A4 both"}
    errors, recall_by_distance = {}, {}
    for name, key in views.items():
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

    out = {
        "test_ranges": {t: [round(min(d[t]["general"][2] for d in S.values()), 2),
                            round(max(d[t]["general"][2] for d in S.values()), 2)] for t in ("ari", "ilo", "global")},
        "gain_split_vs_dfki": split,
        "test_pooled_relations": pooled,
        "test_ari_boot95": boot,
        "metric_dilution_A1_to_A2_test": {"inference_f1": [pooled["A1"]["inference_f1"], pooled["A2"]["inference_f1"]],
                                          "ari_general": [round(S["A1"]["ari"]["general"][2], 2), round(S["A2"]["ari"]["general"][2], 2)]},
        "headroom_share_dev": headroom,
        "dev_to_test_ari_general": dev_to_test,
        "relation_errors_dev_matched_density": errors,
        "relation_recall_by_distance_dev": recall_by_distance,
        "protocol_cost_a1_ari_general": protocol_cost_a1,
        "ilo_test_per_map": {m: S["A1"]["per_map_f1"][m]["ilo"][0] for m in MAPS},
    }
    dst = RES / "short_report" / "derived.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
