"""A3.6 post-hoc, read-only: is A3's dev -> test jump (+6.1 ARI, against A1 +0.9 and A2 +1.1) chance or a warning?

Nothing here changes a submitted map or a threshold; all three systems' test maps were scored once before this ran.
1. Per-map test ARI (general) for A1, A2, A3 (votes and greedy), from their saved test predictions.
2. How often an 11-map subset of dev reproduces the test gaps: A3 - A1 and A3 - A2 over 20,000 random subsets of
   the 225 dev maps (honest per-map scores), against the observed test gaps.

    python -m a3_finetune.posthoc_test     -> results/a3/test_posthoc.json
"""
import json
import random

from common import RESULTS
from common.aif import load_map, test_paths
from common.score import score_map
from a3_finetune.tune import Base, a1_scores, a2_scores, a3_scores, honest
from a3_finetune.runner import load_export

PRED = {"A1": RESULTS / "a1" / "v1" / "test_predictions", "A2": RESULTS / "a2" / "test_predictions",
        "A3 votes": RESULTS / "a3" / "test_predictions", "A3 greedy": RESULTS / "a3" / "test_predictions_greedy"}


def main():
    golds = {name: load_map(g) for name, _, g in test_paths()}
    per_map = {k: {n: round(100 * score_map(golds[n], load_map(d / f"{n}.json"))["ari"][0][2], 2) for n in golds}
               for k, d in PRED.items()}
    mean = {k: round(sum(v.values()) / len(v), 2) for k, v in per_map.items()}
    test_gap = {"A3-A1": round(mean["A3 votes"] - mean["A1"], 2), "A3-A2": round(mean["A3 votes"] - mean["A2"], 2)}

    base = Base()
    held = {"A1": honest(a1_scores(base))[1], "A2": honest(a2_scores(base))[1],
            "A3": honest(a3_scores(base, load_export("v1", "ep2.0_dev_samples8")))[1]}
    names = base.names
    rng = random.Random(13)
    draws = {"A3-A1": [], "A3-A2": [], "A3 mean": []}
    for _ in range(20000):
        sub = rng.sample(names, 11)
        m = {k: 100 * sum(held[k][n][0][2] for n in sub) / 11 for k in held}
        draws["A3-A1"].append(m["A3"] - m["A1"])
        draws["A3-A2"].append(m["A3"] - m["A2"])
        draws["A3 mean"].append(m["A3"])
    frac = lambda xs, t: round(sum(x >= t for x in xs) / len(xs), 4)
    report = {
        "test_ari_general_per_map": per_map, "test_mean": mean, "test_gaps": test_gap,
        "dev_11_map_subsets": {
            "P(A3 - A1 >= test gap)": frac(draws["A3-A1"], test_gap["A3-A1"]),
            "P(A3 - A2 >= test gap)": frac(draws["A3-A2"], test_gap["A3-A2"]),
            "P(A3 mean >= test mean)": frac(draws["A3 mean"], mean["A3 votes"]),
            "A3-A1 range (2.5-97.5%)": [round(sorted(draws["A3-A1"])[k], 2) for k in (500, 19499)],
            "A3 mean range (2.5-97.5%)": [round(sorted(draws["A3 mean"])[k], 2) for k in (500, 19499)]},
        "maps_A3_beats_A1": sum(per_map["A3 votes"][n] > per_map["A1"][n] for n in golds),
        "maps_A3_beats_A2": sum(per_map["A3 votes"][n] > per_map["A2"][n] for n in golds),
    }
    (RESULTS / "a3" / "test_posthoc.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
