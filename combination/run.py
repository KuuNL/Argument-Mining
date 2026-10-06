"""Combination (PLAN: Combination; D083): average the pair scores of A1, A2 and A4, then A1's decision layer.

The simplest form of the planned combination. Every pair's five scores are the mean of the systems' scores: A1
with its temperature, A2's confidence-weighted votes (3 answers) and A4's two-view probabilities. Thresholds are
fitted on the averaged dev scores (A1's coordinate ascent); the honest number is cross-fitted by dev episode.
Pre-registered before the test gold is read (D083): main = A1 + A2 + A4, the best on dev; A1 + A2 is reported
beside it, to show what A4 adds on test. Turn types are the ILO labeller's. Every test answer is already cached,
so the test run costs nothing. The confidence-routing variant (A2 only on uncertain pairs) is not run.

    python -m combination.run --dev     -> results/combination/dev_report.json, thresholds.json
    python -m combination.run --test    -> results/combination/test_predictions/<combo>/, test_global.json
"""
import json
import sys

import numpy as np

from common import RESULTS, ROOT
from common.aif import load_map, test_paths
from common.score import score_map
from common.split import episodes, map_speakers
from common.validate import validate
from a1_cross_encoder.decide import decode, read
from a1_cross_encoder.predict import build_map
from a2_llm.check import dev_maps
from a2_llm.parse import pair_scores as a2_pair_scores
from a2_llm.runner import RUNS as A2_RUNS, jobs as a2_jobs, load as a2_load, targets
from a2_llm.tune import both
from a3_finetune.tune import Base, a2_scores, average, honest
from a4_jev.parse import pair_scores as a4_pair_scores
from a4_jev.runner import answers, builds, config
from a4_jev.tune import a1_tempered, a4_scores, summary

OUT = RESULTS / "combination"
COMBOS = {"A1+A2+A4": ("A1", "A2", "A4"), "A1+A2": ("A1", "A2")}
MAIN = "A1+A2+A4"
NONE = np.array([1.0, 0, 0, 0, 0])


def dev():
    OUT.mkdir(parents=True, exist_ok=True)
    base = Base()
    bs, ans = builds("dev"), answers("dev")
    systems = {"A1": a1_tempered(base), "A2": a2_scores(base), "A4": a4_scores(base, bs, ans)}
    rep, ths = {}, {}
    ep_of, *_ = episodes(map_speakers())
    golds = {name: g for name, g, _, _ in dev_maps()}
    for combo, members in COMBOS.items():
        sc = average(base, [systems[m] for m in members])
        final, held, folds = honest(sc)
        ths[combo] = final["th"]
        preds = {}
        for name, inp, links in targets(config("dev")):
            fold = folds[str(ep_of[int(name.removeprefix("nodeset"))])]
            th = {k: fold[k] for k in ("Inference", "Conflict", "Rephrase")}
            preds[name] = build_map(inp, name, sc.predicted(sc.probs, name, th), links, bs[name].s.pos)
        invalid = [n for n, m in preds.items() if validate(m)]
        glob = summary(golds, preds)
        rep[combo] = {"honest_ari": both(held), "thresholds": final["th"], "folds": folds, "invalid": invalid,
                      "dev_global_honest": {k: glob[k] for k in ("ari", "ilo", "global")}, "table": glob["table"]}
    (OUT / "thresholds.json").write_text(json.dumps({"main": MAIN, "thresholds": ths}, indent=1) + "\n", encoding="utf-8", newline="\n")
    (OUT / "dev_report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    for combo, r in rep.items():
        print(combo, r["honest_ari"], "invalid", len(r["invalid"]))
        print(r["table"])


def test_scores() -> tuple[dict, dict, dict]:
    """Per test map: rows (A2 / A4 pair order) and each system's (n, 5) scores, plus inputs and links."""
    a1_rows = read("test")
    t = json.loads((RESULTS / "a1" / "v1" / "thresholds.json").read_text())["temperature"]
    raw = np.load(ROOT / "checkpoints" / "a1" / "v1" / "test_probs.npy")
    z = np.log(np.clip(raw, 1e-12, 1.0)) / t
    z -= z.max(1, keepdims=True)
    p1 = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    a1 = {(r["map"], r["a"], r["b"]): p1[k] for k, r in enumerate(a1_rows)}
    by_map = {}
    for job in a2_jobs(A2_RUNS["test-v1"]):
        by_map.setdefault(job["map"], []).append((job, a2_load(job["key"], job["s"])))
    a2 = {}
    for name, items in by_map.items():
        ans = [rec["attempts"][rec["answer"]]["result"]["structured_output"] for _, rec in items if rec and rec["ok"]]
        rows, sc = a2_pair_scores(items[0][0]["s"], ans)
        a2.update({(name, r["a"], r["b"]): v for r, v in zip(rows, sc)})
    bs, ans = builds("test"), answers("test")
    out, ctx = {}, {}
    for name, inp, links in targets(config("test")):
        rows, s4 = a4_pair_scores(bs[name], ans[name][0], "both")
        keys = [(name, r["a"], r["b"]) for r in rows]
        assert all(k in a2 for k in keys), name                    # A2 and A4 share the row order
        out[name] = {"rows": rows, "A1": np.array([a1.get(k, NONE) for k in keys]),
                     "A2": np.array([a2[k] for k in keys]), "A4": s4}
        ctx[name] = (inp, links, bs[name].s.pos)
    return out, ctx, {"a1_rows_matched": sum(k in a1 for m in out.values() for k in
                                              [(r["map"], r["a"], r["b"]) for r in m["rows"]]), "a1_rows": len(a1)}


def test():
    ths = json.loads((OUT / "thresholds.json").read_text())["thresholds"]
    sc, ctx, coverage = test_scores()
    assert len(sc) == 11
    preds = {c: {} for c in COMBOS}
    for combo, members in COMBOS.items():
        d = OUT / "test_predictions" / combo.replace("+", "_")
        d.mkdir(parents=True, exist_ok=True)
        for name, s in sc.items():
            probs = np.mean([s[m] for m in members], axis=0)
            inp, links, pos = ctx[name]
            m = build_map(inp, name, decode(s["rows"], probs, ths[combo]), links, pos)
            assert not validate(m), (combo, name, validate(m)[:3])
            (d / f"{name}.json").write_text(json.dumps(m, indent=1), encoding="utf-8", newline="\n")
            preds[combo][name] = m
    print("22 maps (11 per combination) assembled, validated and saved before reading the gold", coverage)

    gold_paths = {name: g for name, _, g in test_paths()}             # the gold is read here, once
    golds = {name: load_map(gold_paths[name]) for name in sc}
    res = {"main": MAIN, "thresholds": ths, "a1_coverage": coverage}
    for combo, p in preds.items():
        r = summary(golds, p)
        r["per_map_ari_f1"] = {n: [round(100 * x[0][2], 2), round(100 * x[1][2], 2) if x[1] else None]
                               for n, x in ((n, score_map(golds[n], p[n])["ari"]) for n in sorted(p))}
        res[combo] = r
        print(combo)
        print(r["table"])
    (OUT / "test_global.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    if "--dev" in sys.argv:
        dev()
    if "--test" in sys.argv:
        test()
