"""A4.2: the pilot table and the pre-set choices (D076, D077), from the saved answers of pilot-k0 and pilot-k1.

On A2's 50 pilot maps (training split, gold links), every arm x view is scored at matched density: one threshold for
every relation type, set so the arm predicts as many relations as gold (pooled over the maps). Threshold-free, so
the arms compare as models, not as tuning.
Rules, fixed before the pilot ran:
1. examples (D077): k = 1 is kept only if its best view beats k = 0's best view by >= 1.0 ARI general;
2. wording (D078): the source-question fix (pilot-k0-w2) is adopted if its `both` ARI general beats pilot-k0's;
3. view (D076): on the chosen arm, the view with the best ARI general; if `pair` or `source` alone is within 1.0 of
   `both`, that single view (it halves later requests; the better of the two if both qualify).
The confusion table (gold label vs predicted, per pair, at matched density) is for reading errors: at most one
wording fix, and only for a clear failure.

    python -m a4_jev.pilot      -> results/a4/pilot_table.json
"""
import json
from collections import Counter

from common import RESULTS
from common.aif import load_map
from common.score import ari_labels
from common.split import load_split
from a1_cross_encoder.decide import decode
from a2_llm.runner import targets
from a4_jev.parse import VIEWS, pair_scores
from a4_jev.probe import matched
from a4_jev.runner import answers, builds, config

OUT = RESULTS / "a4"
ARMS = ("pilot-k0", "pilot-k1", "pilot-k0-w2")
NAMES = {0: "Inference", 1: "Conflict", 2: "Rephrase"}


def confusion(names, bs, ans, maps, view, th) -> dict:
    """Unordered pairs: gold kind (with direction for Inference) -> predicted kind."""
    c = Counter()
    for n in names:
        b = bs[n]
        gold = {k: v for k, v in ari_labels(maps[n][0], set(b.s.prop.values())).items() if k[0] != k[1]}
        pred = decode(*pair_scores(b, ans[n], view), {t: th for t in ("Inference", "Conflict", "Rephrase")})
        pairs = {frozenset(k) for k in gold} | {frozenset(k) for k in pred}
        for pr in pairs:
            g = next(((k, v) for k, v in gold.items() if frozenset(k) == pr), None)
            p = next(((k, v) for k, v in pred.items() if frozenset(k) == pr), None)
            gl = NAMES[g[1]] if g else "None"
            pl = NAMES[p[1]] if p else "None"
            if g and p and gl == pl == "Inference" and g[0] != p[0]:
                pl = "Inference reversed"
            c[f"{gl} -> {pl}"] += 1
    return dict(sorted(c.items(), key=lambda x: -x[1]))


def main():
    table, keep = {}, {}
    paths = {p.stem: p for p in load_split("train")}
    for arm in ARMS:
        bs, ans = builds(arm), answers(arm)
        maps = {n: (load_map(paths[n]), inp, links) for n, inp, links in targets(config(arm))}
        names = list(bs)
        flat = {n: ans[n][0] for n in names}
        table[arm] = {v: matched(names, bs, flat, maps, v, ("f", "r")) for v in VIEWS}
        keep[arm] = (bs, flat, maps, names)
    best = {arm: max(VIEWS, key=lambda v: table[arm][v]["ari"][0]) for arm in ARMS}
    gain = table["pilot-k1"][best["pilot-k1"]]["ari"][0] - table["pilot-k0"][best["pilot-k0"]]["ari"][0]
    arm = "pilot-k1" if gain >= 1.0 else "pilot-k0"
    w2_gain = table["pilot-k0-w2"]["both"]["ari"][0] - table["pilot-k0"]["both"]["ari"][0]
    if arm == "pilot-k0" and w2_gain > 0:
        arm = "pilot-k0-w2"
    t = table[arm]
    view = best[arm]
    if view == "both":
        singles = [v for v in ("pair", "source") if t["both"]["ari"][0] - t[v]["ari"][0] <= 1.0]
        if singles:
            view = max(singles, key=lambda v: t[v]["ari"][0])
    bs, flat, maps, names = keep[arm]
    rep = {"maps": len(names), "table_matched_density": table, "best_view_per_arm": best,
           "k1_gain_general": round(gain, 2), "w2_gain_general_both": round(w2_gain, 2),
           "choice": {"arm": arm, "k": 1 if arm == "pilot-k1" else 0, "wording": "v2" if arm.endswith("w2") else "v1", "view": view,
                      "rules": ["k = 1 only if its best view gains >= 1.0 general over k = 0's best view",
                                "best view; a single view within 1.0 of both is taken instead"]},
           "confusion_at_matched_density": {}, "inference_direction_right": {}}
    for a in ARMS:
        bs_a, flat_a, maps_a, names_a = keep[a]
        conf = {v: confusion(names_a, bs_a, flat_a, maps_a, v, table[a][v]["threshold"]) for v in VIEWS}
        rep["confusion_at_matched_density"][a] = conf
        rep["inference_direction_right"][a] = {
            v: round(c.get("Inference -> Inference", 0) / max(1, c.get("Inference -> Inference", 0) + c.get("Inference -> Inference reversed", 0)), 3)
            for v, c in conf.items()}
    with open(OUT / "pilot_table.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
