"""A3.4: the decision layer on dev, the honest score, the success tests and the hypothesis measures (D065).

Every unordered pair of every dev map gets A3's five scores: the share of the 8 sampled answers that list each
relation (a3_finetune/parse.py; vote shares are the main score, D069). `Scores` gives any system's pair scores
A1's interface, so A1's decision layer and A2's helpers apply unchanged, and A1 / A2 are re-scored on the same
rows by the same code (their pruned or unlisted pairs score None).
1. Thresholds per relation type by coordinate ascent on mean per-map ARI general F1 (A1's `tune`), fitted on all
   of dev (kept for the test run) and cross-fitted by dev episode (the honest number).
2. Success test 1: a paired per-map bootstrap of honest ARI general, A3 against A2 and against A1 (95% interval).
3. Success test 2: cross-fitted averages of the systems' pair scores; A1 + A2 + A3 must beat A1 + A2 by >= 1.0.
4. Hypotheses H1-H4 at matched density: one threshold for every type, set so that the system predicts as many
   relations as gold. A3's vote shares come in steps of 1/8, so for this measure only its ties are broken by token
   probability (votes + 0.001 x token-probability score); otherwise its density can't be matched. The same
   measures at each system's own tuned thresholds are given for reference.
5. A2.3's diagnostics at the tuned thresholds, and each single sample's score.

    python -m a3_finetune.tune --run=v1 --adapter=ep2.0     -> results/a3/<run>/dev_report.json, thresholds.json
"""
import json
import random
import statistics as st
from collections import Counter

import numpy as np

from common import RESULTS, ROOT
from common.aif import view
from common.score import ari_labels
from a1_cross_encoder.decide import Dev as A1Dev, ari_f1, decode, mean_f1, tune
from a1_cross_encoder.tune import diagnostics
from a2_llm.check import dev_maps
from a2_llm.tune import Scores as A2Scores, both, cross_fitted, structure
from a3_finetune.data import dev_examples
from a3_finetune.model import tokenizer
from a3_finetune.parse import entry_probs, pair_scores
from a3_finetune.runner import arg, load_export
from a3_finetune.train import OUT

ALL = {"Inference": 0.01, "Conflict": 0.01, "Rephrase": 0.01}       # every listed relation
EMPTY = {"illocutions": [], "relations": []}


class Scores:
    """Pair scores of one system on the dev maps, with A1's interface (names, rows, idx, probs, gold, n_pairs)."""

    def __init__(self, base: "Base", probs_by_map: dict):
        self.names, self.gold, self.n_pairs = base.names, base.gold, base.n_pairs
        self.rows, self.idx, blocks = [], {}, []
        for n in self.names:
            self.idx[n] = list(range(len(self.rows), len(self.rows) + len(base.rows[n])))
            self.rows += base.rows[n]
            blocks.append(probs_by_map[n])
        self.probs = np.concatenate(blocks)

    def predicted(self, probs, name, th, cap=None):
        idx = self.idx[name]
        return decode([self.rows[k] for k in idx], probs[idx], th, cap)

    def scores(self, probs, th, names=None, cap=None):
        return {n: ari_f1(self.gold[n], self.predicted(probs, n, th, cap), self.n_pairs[n]) for n in (names or self.names)}


class Base:
    """The 225 dev maps: A3's rows (every unordered pair), gold relations, positions and speakers."""

    def __init__(self):
        exs = {e.name: e for e in dev_examples()}
        self.names = sorted(exs)
        self.s = {n: exs[n].s for n in self.names}
        self.rows, self.gold, self.n_pairs, self.speaker = {}, {}, {}, {}
        for name, g, _, _ in dev_maps():
            props = set(view(g, name).props)
            self.gold[name] = {k: c for k, c in ari_labels(g, props).items() if k[1] in props and k[0] != k[1]}
            self.n_pairs[name] = len(props) * (len(props) - 1)
            v = view(g, name)
            self.speaker[name] = {i: v.locs[v.dialogue[k]].speaker for i, k in self.s[name].pos.items()}
        from a2_llm.parse import pair_rows
        for n in self.names:
            self.rows[n] = pair_rows(self.s[n])

    def aligned(self, lookup) -> dict:
        """probs per map from lookup[(map, a, b)] -> 5 scores; pairs it lacks score None. Every lookup key must be
        one of A3's rows (the same pair order), which `self.coverage` records."""
        keys = {(n, r["a"], r["b"]) for n in self.names for r in self.rows[n]}
        self.coverage = {"rows_given": len(lookup), "rows_matched": len(keys & lookup.keys())}
        out = {}
        for n in self.names:
            m = np.zeros((len(self.rows[n]), 5))
            m[:, 0] = 1.0
            for k, r in enumerate(self.rows[n]):
                v = lookup.get((n, r["a"], r["b"]))
                if v is not None:
                    m[k] = v
            out[n] = m
        return out


def a3_scores(base: Base, recs: dict, use=None, weights_tok=None) -> Scores:
    probs = {}
    for n in base.names:
        outs = recs[n]["outs"] if use is None else [recs[n]["outs"][i] for i in use]
        answers = [o["answer"] if o["valid"] else EMPTY for o in outs]
        weights = None
        if weights_tok is not None:
            weights = [entry_probs(weights_tok, o["tokens"], o["logps"]) if o["valid"] else {} for o in outs]
        probs[n] = pair_scores(base.s[n], answers, weights)[1]
    return Scores(base, probs)


def a1_scores(base: Base) -> Scores:
    a1 = A1Dev(ROOT / "checkpoints" / "a1" / "v1" / "dev_probs.npy")
    t = json.loads((RESULTS / "a1" / "v1" / "thresholds.json").read_text())["temperature"]
    p = a1.calibrated(t)
    return Scores(base, base.aligned({(r["map"], r["a"], r["b"]): p[k] for k, r in enumerate(a1.rows)}))


def a2_scores(base: Base) -> Scores:
    a2 = A2Scores("dev-v1")
    return Scores(base, base.aligned({(n, a2.rows[k]["a"], a2.rows[k]["b"]): a2.probs[k]
                                      for n in a2.names for k in a2.idx[n]}))


def average(base: Base, systems: list[Scores]) -> Scores:
    return Scores(base, {n: np.mean([s.probs[s.idx[n]] for s in systems], axis=0) for n in base.names})


def honest(dev: Scores) -> tuple[dict, dict, dict]:
    """Thresholds fitted on all dev, and per-map cross-fitted (honest) scores with each fold's thresholds."""
    final = tune(dev, dev.probs, dev.names)
    held, folds = cross_fitted(dev)
    return final, held, folds


def bootstrap(a: dict, b: dict, which: int, n: int = 10000, seed: int = 13) -> dict:
    """Mean per-map difference a - b with a 95% percentile interval, resampling maps (paired)."""
    names = [m for m in a if a[m][which] is not None and b[m][which] is not None]
    d = np.array([100 * (a[m][which][2] - b[m][which][2]) for m in names])
    rng = random.Random(seed)
    means = sorted(float(np.mean([d[rng.randrange(len(d))] for _ in d])) for _ in range(n))
    return {"maps": len(names), "mean_difference": round(float(d.mean()), 2),
            "ci95": [round(means[int(0.025 * n)], 2), round(means[int(0.975 * n) - 1], 2)],
            "a_better_maps": int((d > 0).sum()), "b_better_maps": int((d < 0).sum())}


def matched(dev: Scores) -> dict:
    """One threshold for every type, set so the system predicts as many relations as gold (in total)."""
    target = sum(len(dev.gold[n]) for n in dev.names)
    best = dev.probs[:, 1:].max(1)                         # one relation per pair: its best label, kept if >= t
    desc = np.sort(best[best > 0])[::-1]
    cand = np.unique(desc)[::-1]
    counts = np.searchsorted(-desc, -cand, side="right")   # pairs scoring >= each candidate
    return dict.fromkeys(ALL, float(cand[int(np.argmin(np.abs(counts - target)))]))


def hypotheses(base: Base, dev: Scores, th: dict) -> dict:
    """H1-H4 measures for one system at thresholds th. Pairs are unordered; right = label and direction match."""
    c = Counter()
    for n in base.names:
        g, p = base.gold[n], dev.predicted(dev.probs, n, th)
        pos, spk = base.s[n].pos, base.speaker[n]
        c["relations"] += len(p)
        undirected_gold = {frozenset(k) for k in g}
        c["unrelated_pairs_given_a_relation"] += sum(1 for k in p if frozenset(k) not in undirected_gold)
        for (a, b), code in g.items():
            hit = p.get((a, b)) == code
            if code == 0:                                          # Inference
                if a in pos and b in pos and abs(pos[a] - pos[b]) == 1:
                    c["neighbour_inference_gold"] += 1
                    c["neighbour_inference_found"] += hit
                if p.get((a, b)) == 0 or p.get((b, a)) == 0:
                    c["inference_pairs_found"] += 1
                    c["inference_direction_right"] += hit
            if code == 1 and a in spk and b in spk and spk[a] == spk[b]:   # Conflict within one speaker
                c["within_speaker_conflict_gold"] += 1
                c["within_speaker_conflict_found"] += hit
    r = lambda x, y: round(c[x] / c[y], 3) if c[y] else None
    return {"relations_per_map": round(c["relations"] / len(base.names), 2),
            "H1_neighbour_inference_recall": r("neighbour_inference_found", "neighbour_inference_gold"),
            "H2_unrelated_pairs_given_a_relation": c["unrelated_pairs_given_a_relation"],
            "H3_inference_direction_right": r("inference_direction_right", "inference_pairs_found"),
            "H4_within_speaker_conflict_recall": r("within_speaker_conflict_found", "within_speaker_conflict_gold"),
            "counts": dict(c)}


def main():
    run, adapter = arg("run", "v1"), arg("adapter", "ep2.0")
    base = Base()
    recs = load_export(run, f"{adapter}_dev_samples8")
    greedy = load_export(run, f"{adapter}_dev_greedy")
    a3 = a3_scores(base, recs)
    final, held, folds = honest(a3)
    systems = {"A1": a1_scores(base)}
    coverage = {"A1": base.coverage}
    systems["A2"] = a2_scores(base)
    coverage["A2"] = base.coverage
    systems["A3"] = a3
    print("coverage", coverage, flush=True)
    tok = tokenizer(json.loads((OUT / run / "config.json").read_text())["model"])
    a3_tok = a3_scores(base, recs, weights_tok=tok)
    a3_tie = Scores(base, {n: a3.probs[a3.idx[n]] + 1e-3 * a3_tok.probs[a3_tok.idx[n]] for n in base.names})
    fits = {"A3": (final, held, folds)}
    for k in ("A1", "A2"):
        fits[k] = honest(systems[k])
    print({k: both(v[1]) for k, v in fits.items()}, flush=True)

    combos = {}
    for name, members in (("A1+A2", ["A1", "A2"]), ("A1+A2+A3", ["A1", "A2", "A3"]), ("A2+A3", ["A2", "A3"]),
                          ("A1+A3", ["A1", "A3"])):
        combos[name] = both(honest(average(base, [systems[m] for m in members]))[1])
    gain = round(combos["A1+A2+A3"]["general"] - combos["A1+A2"]["general"], 2)

    test1 = {"vs_A2": {"general": bootstrap(held, fits["A2"][1], 0), "focused": bootstrap(held, fits["A2"][1], 1)},
             "vs_A1": {"general": bootstrap(held, fits["A1"][1], 0), "focused": bootstrap(held, fits["A1"][1], 1)}}
    a3_general = both(held)["general"]
    passed1 = a3_general > 54.63 and test1["vs_A2"]["general"]["ci95"][0] > 0

    hyp = {}
    for k, dev in systems.items():
        th_m = matched(a3_tie if k == "A3" else dev)
        hyp[k] = {"matched_threshold": th_m["Inference"],
                  "at_matched_density": hypotheses(base, a3_tie if k == "A3" else dev, th_m),
                  "at_own_tuned_thresholds": hypotheses(base, dev, fits[k][0]["th"])}
    m3, m1, m2 = (hyp[k]["at_matched_density"] for k in ("A3", "A1", "A2"))
    verdicts = {
        "H1 (>= A2 + 0.05)": m3["H1_neighbour_inference_recall"] >= m2["H1_neighbour_inference_recall"] + 0.05,
        "H2 (<= 90% of A1)": m3["H2_unrelated_pairs_given_a_relation"] <= 0.9 * m1["H2_unrelated_pairs_given_a_relation"],
        "H3 (>= 85%)": m3["H3_inference_direction_right"] >= 0.85,
        "H4 (>= 15%)": m3["H4_within_speaker_conflict_recall"] >= 0.15,
    }

    one = [both(s.scores(s.probs, ALL)) for s in (a3_scores(base, recs, use=[i]) for i in range(8))]
    g_scores = a3_scores(base, greedy)
    report = {
        "run": run, "adapter": adapter, "maps": len(base.names),
        "valid_answers": sum(o["valid"] for r in recs.values() for o in r["outs"]),
        "answers": sum(len(r["outs"]) for r in recs.values()),
        "thresholds": final["th"],
        "dev_ari_f1": {"all listed (no thresholds)": both(a3.scores(a3.probs, ALL)),
                       "tuned on all dev (optimistic)": both(a3.scores(a3.probs, final["th"])),
                       "cross-fitted (honest)": both(held)},
        "folds": folds,
        "references_recomputed_here (honest)": {k: both(fits[k][1]) for k in ("A1", "A2")},
        "row_coverage": coverage,
        "success_test_1": {**test1, "passed": passed1,
                           "rule": "A3 honest ARI general > 54.63 and the 95% interval of A3 - A2 above 0"},
        "success_test_2": {"honest_ari": combos, "gain_from_adding_A3": gain, "passed": gain >= 1.0,
                           "rule": "A1+A2+A3 beats A1+A2 by >= 1.0 general (cross-fitted averages of pair scores)"},
        "hypotheses": {"verdicts_at_matched_density": verdicts, **hyp},
        "diagnostics_tuned": diagnostics(a3, a3.probs, final["th"]),
        "structure_tuned": structure(a3, final["th"]),
        "single_sample_all_listed": one,
        "greedy_all_listed": both(g_scores.scores(g_scores.probs, ALL)),
        "token_probability_score (honest)": both(honest(a3_tok)[1]),
    }
    (OUT / run / "thresholds.json").write_text(json.dumps({"run": run, "adapter": adapter, "thresholds": final["th"],
                                                           "objective": final["objective"]}, indent=1) + "\n",
                                               encoding="utf-8", newline="\n")
    (OUT / run / "dev_report.json").write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8",
                                               newline="\n")
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
