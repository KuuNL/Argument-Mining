"""A4.3-A4.4: the decision layer on dev, the honest score, the success tests, H1-H5 and complete dev maps (D073, D076).

Every unordered pair of every dev map gets A4's five scores (a4_jev/parse.py, the frozen view `both`, D078), aligned
to A3's rows with A3's comparison code, so A1 and A2 are re-scored on the same rows by the same code.
1. Thresholds per relation type (A1's coordinate ascent on mean per-map ARI general), fitted on all dev (kept for
   the test run) and cross-fitted by dev episode (the honest number). No temperature step: Jev's outputs are
   probabilities already, and H1 measures their calibration as they come.
2. Success test 1: A4 honest ARI general > A1's 50.50 with the paired bootstrap 95% interval of A4 - A1 above 0.
   Success test 2: cross-fitted averages of pair scores, A1 + A2 + A4 against A1 + A2, gain >= 1.0 general.
3. Hypotheses (D073), at matched density unless stated:
   H1 class-wise ECE (Inference, Conflict, Rephrase) <= A1's on the same candidate pairs, both uncalibrated;
   H2 source view >= pair view + 1.0 general, with <= 90% of its invented relations (unrelated pairs given one);
   H3 within-speaker Conflict recall >= 0.15;  H4 Inference direction >= 0.85;  H5 median <= 1 s per map.
4. Diagnostics at the tuned thresholds; overlap with A1 and A2 (who gets which pairs right).
5. Complete dev maps: A4's relations at each episode's cross-fitted thresholds + the ILO labeller's links and types
   (A1's build_map), validated and scored: honest dev Global.

    python -m a4_jev.tune      -> results/a4/dev/{dev_report,thresholds,dev_global}.json
"""
import json
import statistics as st
from collections import Counter

import numpy as np

from common import RESULTS, ROOT
from common.score import aggregate, format_table, score_map
from common.split import episodes, map_speakers
from common.timing import timed
from common.validate import validate
from a1_cross_encoder.decide import Dev as A1Dev
from a1_cross_encoder.predict import build_map, relation_diagnostics
from a1_cross_encoder.tune import diagnostics
from a2_llm.check import dev_maps
from a2_llm.runner import targets
from a2_llm.tune import both, structure
from a3_finetune.tune import Base, Scores, a2_scores, average, bootstrap, honest, hypotheses, matched
from a4_jev.parse import VIEWS, pair_scores
from a4_jev.runner import answers, builds, config, jobs, load

OUT = RESULTS / "a4" / "dev"
RUN, VIEW = "dev", "both"
KINDS = {"Inference": (1, 2), "Conflict": (3,), "Rephrase": (4,)}


def a4_scores(base: Base, bs, ans, view: str = VIEW) -> Scores:
    lookup = {}
    for n in base.names:
        rows, sc = pair_scores(bs[n], ans[n][0], view)
        lookup.update({(n, r["a"], r["b"]): v for r, v in zip(rows, sc)})
    return Scores(base, base.aligned(lookup))


def a1_raw(base: Base) -> Scores:
    """A1's dev probabilities before its temperature (for H1)."""
    a1 = A1Dev(ROOT / "checkpoints" / "a1" / "v1" / "dev_probs.npy")
    p = a1.calibrated(1.0)
    return Scores(base, base.aligned({(r["map"], r["a"], r["b"]): p[k] for k, r in enumerate(a1.rows)}))


def a1_tempered(base: Base) -> Scores:
    a1 = A1Dev(ROOT / "checkpoints" / "a1" / "v1" / "dev_probs.npy")
    t = json.loads((RESULTS / "a1" / "v1" / "thresholds.json").read_text())["temperature"]
    p = a1.calibrated(t)
    return Scores(base, base.aligned({(r["map"], r["a"], r["b"]): p[k] for k, r in enumerate(a1.rows)}))


def gold_kind(base: Base, sc: Scores) -> np.ndarray:
    """Per row: 0 None, 1 Inference (either direction), 2 Conflict, 3 Rephrase."""
    out = np.zeros(len(sc.rows), dtype=int)
    for k, r in enumerate(sc.rows):
        g = base.gold[r["map"]]
        c = g.get((r["a"], r["b"]), g.get((r["b"], r["a"])))
        out[k] = 0 if c is None else c + 1
    return out


def ece(base: Base, sc: Scores, mask: np.ndarray, bins: int = 15) -> dict:
    """Class-wise expected calibration error over the masked rows, with each class's reliability bins."""
    kind = gold_kind(base, sc)[mask]
    probs = sc.probs[mask]
    out = {}
    for c, (name, cols) in enumerate(KINDS.items(), start=1):
        p = probs[:, list(cols)].sum(1)
        y = (kind == c).astype(float)
        edges = np.minimum((p * bins).astype(int), bins - 1)
        e, rel = 0.0, []
        for b in range(bins):
            sel = edges == b
            if sel.any():
                e += sel.mean() * abs(p[sel].mean() - y[sel].mean())
                rel.append([round(b / bins, 3), int(sel.sum()), round(float(p[sel].mean()), 4), round(float(y[sel].mean()), 4)])
        out[name] = {"ece": round(float(e), 4), "reliability [bin start, n, mean p, observed]": rel}
    out["mean"] = round(st.mean(v["ece"] for v in out.values()), 4)
    return out


def overlap(base: Base, systems: dict, ths: dict) -> dict:
    """Pairs where gold or any system has a relation: which systems get each right (label and direction)."""
    c = Counter()
    for n in base.names:
        g = base.gold[n]
        preds = {k: s.predicted(s.probs, n, ths[k]) for k, s in systems.items()}
        pairs = {frozenset(x) for x in g} | {frozenset(x) for p in preds.values() for x in p}
        for pr in pairs:
            gk = {x: v for x, v in g.items() if frozenset(x) == pr}
            right = tuple(k for k, p in preds.items() if {x: v for x, v in p.items() if frozenset(x) == pr} == gk)
            c["+".join(right) or "none"] += 1
    return dict(sorted(c.items(), key=lambda x: -x[1]))


def speed() -> dict:
    per_seq, per_par, usd, toks = {}, {}, 0.0, 0
    for j in jobs(RUN):
        rec = load(j["key"])
        s = rec["result"]["seconds"]
        per_seq[j["map"]] = per_seq.get(j["map"], 0) + s
        per_par[j["map"]] = max(per_par.get(j["map"], 0), s)
        resp = rec["result"]["response"]
        usd += float(((resp.get("providerMetadata") or {}).get("gateway") or {}).get("cost") or 0)
        toks += int((resp.get("usage") or {}).get("inputTokens", 0))
    return {"maps": len(per_seq), "seconds_per_map_sequential_p50": round(st.median(per_seq.values()), 2),
            "seconds_per_map_parallel_p50": round(st.median(per_par.values()), 2),
            "seconds_per_map_sequential_max": round(max(per_seq.values()), 2),
            "dev_pass_usd": round(usd, 4), "dev_pass_input_tokens": toks}


def summary(golds: dict, preds: dict) -> dict:
    names = sorted(preds)
    per_map = [score_map(golds[n], preds[n]) for n in names]
    agg = aggregate(per_map)
    return {"table": format_table(agg), **{k: agg[k] for k in ("ari", "ilo", "global", "skipped_maps")},
            "diagnostics": relation_diagnostics(golds, preds)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with timed("a4", "A4.3_tune") as row:
        base = Base()
        bs, ans = builds(RUN), answers(RUN)
        assert len(ans) == len(base.names), (len(ans), len(base.names))
        a4 = a4_scores(base, bs, ans)
        coverage = base.coverage
        final, held, folds = honest(a4)
        systems = {"A1": a1_tempered(base), "A2": a2_scores(base), "A4": a4}
        fits = {"A4": (final, held, folds), "A1": honest(systems["A1"]), "A2": honest(systems["A2"])}

        test1 = {"vs_A1": {"general": bootstrap(held, fits["A1"][1], 0), "focused": bootstrap(held, fits["A1"][1], 1)},
                 "vs_A2": {"general": bootstrap(held, fits["A2"][1], 0), "focused": bootstrap(held, fits["A2"][1], 1)}}
        general = both(held)["general"]
        passed1 = general > 50.50 and test1["vs_A1"]["general"]["ci95"][0] > 0
        combos = {name: both(honest(average(base, [systems[m] for m in members]))[1])
                  for name, members in (("A1+A2", ["A1", "A2"]), ("A1+A2+A4", ["A1", "A2", "A4"]),
                                        ("A2+A4", ["A2", "A4"]), ("A1+A4", ["A1", "A4"]))}
        gain = round(combos["A1+A2+A4"]["general"] - combos["A1+A2"]["general"], 2)

        # H1: calibration on A4's candidate pairs (the same 10-turn window as A1's), both systems uncalibrated.
        cand = {(n, b.s.prop[a], b.s.prop[c]) for n, b in bs.items() for a, c in b.candidates}
        mask = np.array([(r["map"], r["a"], r["b"]) in cand for r in a4.rows])
        cal = {"A4": ece(base, a4, mask), "A1 (before temperature)": ece(base, a1_raw(base), mask),
               "candidate_pairs": int(mask.sum())}

        # H2-H4 at matched density, per view and for A1 / A2.
        views = {v: (a4 if v == VIEW else a4_scores(base, bs, ans, v)) for v in VIEWS}
        md = {}
        for k, s in [*((f"A4 {v}", s) for v, s in views.items()), ("A1", systems["A1"]), ("A2", systems["A2"])]:
            th = matched(s)
            md[k] = {"threshold": round(th["Inference"], 4), "ari": both(s.scores(s.probs, th)), **hypotheses(base, s, th)}
        sp = speed()
        a4m, pm, sm = md["A4 both"], md["A4 pair"], md["A4 source"]
        verdicts = {
            "H1 class-wise ECE <= A1's": cal["A4"]["mean"] <= cal["A1 (before temperature)"]["mean"],
            "H2 source >= pair + 1.0 general, invented <= 90%": (sm["ari"]["general"] >= pm["ari"]["general"] + 1.0
                and sm["H2_unrelated_pairs_given_a_relation"] <= 0.9 * pm["H2_unrelated_pairs_given_a_relation"]),
            "H3 within-speaker Conflict recall >= 0.15": (a4m["H4_within_speaker_conflict_recall"] or 0) >= 0.15,
            "H4 Inference direction >= 0.85": (a4m["H3_inference_direction_right"] or 0) >= 0.85,
            "H5 median <= 1 s per map (requests in parallel)": sp["seconds_per_map_parallel_p50"] <= 1.0,
        }
        ths = {"A1": fits["A1"][0]["th"], "A2": fits["A2"][0]["th"], "A4": final["th"]}
        report = {
            "run": RUN, "view": VIEW, "maps": len(base.names), "row_coverage": coverage, "thresholds": final["th"],
            "dev_ari_f1": {"tuned on all dev (optimistic)": both(a4.scores(a4.probs, final["th"])),
                           "cross-fitted (honest)": both(held),
                           **{f"{v} view, cross-fitted": both(honest(s)[1]) for v, s in views.items() if v != VIEW}},
            "folds": folds,
            "references_recomputed_here (honest)": {k: both(fits[k][1]) for k in ("A1", "A2")},
            "success_test_1": {**test1, "passed": passed1,
                               "rule": "A4 honest ARI general > 50.50 and the 95% interval of A4 - A1 above 0"},
            "success_test_2": {"honest_ari": combos, "gain_from_adding_A4": gain, "passed": gain >= 1.0,
                               "rule": "A1+A2+A4 beats A1+A2 by >= 1.0 general (cross-fitted averages of pair scores)"},
            "hypotheses": {"verdicts": verdicts, "calibration": cal, "matched_density": md, "speed_cost": sp},
            "diagnostics_tuned": diagnostics(a4, a4.probs, final["th"]),
            "structure_tuned": structure(a4, final["th"]),
            "overlap_tuned (who is right, per pair with a gold or predicted relation)": overlap(base, systems, ths),
        }
        (OUT / "thresholds.json").write_text(json.dumps({"run": RUN, "view": VIEW, "thresholds": final["th"],
                                                         "objective": final["objective"]}, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
        (OUT / "dev_report.json").write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")

        # A4.4: complete dev maps at each episode's cross-fitted thresholds, the labeller's links and types.
        ep_of, *_ = episodes(map_speakers())
        golds = {name: g for name, g, _, _ in dev_maps()}
        preds, invalid = {}, []
        for name, inp, links in targets(config(RUN)):
            fold = folds[str(ep_of[int(name.removeprefix("nodeset"))])]
            th = {k: fold[k] for k in ("Inference", "Conflict", "Rephrase")}
            b = bs[name]
            m = build_map(inp, name, a4.predicted(a4.probs, name, th), links, b.s.pos)
            if validate(m):
                invalid.append(name)
            preds[name] = m
        glob = {"maps": len(preds), "invalid": invalid, **summary(golds, preds)}
        (OUT / "dev_global.json").write_text(json.dumps(glob, indent=1) + "\n", encoding="utf-8", newline="\n")
        row.update(maps=len(base.names), honest=both(held)["general"])
    print(json.dumps({k: v for k, v in report.items() if k not in ("diagnostics_tuned",)}, indent=1, default=float)[:12000])
    print(glob["table"], "invalid:", len(invalid))


if __name__ == "__main__":
    main()
