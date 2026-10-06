"""How solid A4's dev result is (D082), from saved answers; nothing new is tuned.

1. Run to run: the dev requests sent twice (run dev-repeat; pass 1 is the `dev` run). Honest ARI of each pass and of
   their mean; per-map |difference| and the overlap of decided relations at the dev thresholds; A4 - A1 per pass.
2. No labels: every system decoded without dev-tuned thresholds, to separate the model from the labelled decision
   layer: argmax for the probability systems (A1, A4); every listed relation for the generative ones (A2; A3's
   greedy answer from D069).

    python -m a4_jev.robust      -> results/a4/robust.json
"""
import json
import statistics as st

import numpy as np

from common import RESULTS
from a2_llm.tune import both
from a3_finetune.tune import ALL, Base, Scores, a2_scores, bootstrap, honest
from a4_jev.parse import pair_scores
from a4_jev.runner import answers, builds
from a4_jev.tune import VIEW, a1_tempered

OUT = RESULTS / "a4"


def scores(base: Base, bs, per_map) -> Scores:
    """per_map: name -> list of answer dicts (passes); pair scores averaged over them."""
    lookup = {}
    for n in base.names:
        vecs = [pair_scores(bs[n], a, VIEW) for a in per_map[n]]
        rows = vecs[0][0]
        mean = np.mean([v[1] for v in vecs], axis=0)
        lookup.update({(n, r["a"], r["b"]): v for r, v in zip(rows, mean)})
    return Scores(base, base.aligned(lookup))


def main():
    base = Base()
    bs, ans = builds("dev-repeat"), answers("dev-repeat")
    assert all(len(ans[n]) == 2 for n in base.names)
    passes = {"pass 1": scores(base, bs, {n: [ans[n][0]] for n in base.names}),
              "pass 2": scores(base, bs, {n: [ans[n][1]] for n in base.names}),
              "mean of both passes": scores(base, bs, {n: [ans[n][0], ans[n][1]] for n in base.names})}
    a1 = a1_tempered(base)
    _, a1_held, _ = honest(a1)
    th = json.loads((OUT / "dev" / "thresholds.json").read_text())["thresholds"]
    run = {}
    held = {}
    for k, s in passes.items():
        final, h, _ = honest(s)
        held[k] = h
        run[k] = {"honest": both(h), "thresholds": final["th"], "A4_minus_A1_general": bootstrap(h, a1_held, 0)}
    s1, s2 = passes["pass 1"], passes["pass 2"]
    d = [abs(100 * (s1.scores(s1.probs, th, [n])[n][0][2] - s2.scores(s2.probs, th, [n])[n][0][2])) for n in base.names]
    same, union = 0, 0
    for n in base.names:
        p1, p2 = set(s1.predicted(s1.probs, n, th).items()), set(s2.predicted(s2.probs, n, th).items())
        same += len(p1 & p2)
        union += len(p1 | p2)
    run["pass 1 vs pass 2 at the dev thresholds"] = {
        "mean_abs_map_ari_difference": round(st.mean(d), 2), "max_abs_map_ari_difference": round(max(d), 2),
        "relations_identical_share": round(same / union, 3)}

    a2 = a2_scores(base)
    no_labels = {"A1 argmax": both(a1.scores(a1.probs, None)),
                 "A2 every listed relation (3 answers)": both(a2.scores(a2.probs, ALL)),
                 "A3 greedy, every listed relation (D069)": {"general": 49.46, "focused": 31.19},
                 "A4 argmax": both(s1.scores(s1.probs, None)),
                 "A4 argmax, mean of both passes": both(passes["mean of both passes"].scores(passes["mean of both passes"].probs, None))}
    a2_held = honest(a2)[1]
    tuned = {"A1": both(a1_held), "A2": both(a2_held), "A3": {"general": 47.65, "focused": 27.23}, "A4": run["pass 1"]["honest"]}
    pairs = {"A2 - A1": bootstrap(a2_held, a1_held, 0), "A4 (pass 1) - A2": bootstrap(held["pass 1"], a2_held, 0)}
    rep = {"run_to_run": run, "no_labels_decode": no_labels, "with_dev_thresholds (honest)": tuned,
           "paired_bootstrap_general": pairs}
    with open(OUT / "robust.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
