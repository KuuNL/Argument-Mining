"""A3.6: complete maps on dev (honest Global) and the single test run (D071).

Each map = the input map + A3's relations + the ILO labeller's links and types (cache/ilo/links_*.json; the LLM's
turn types are not used, D071) + unscored transition anchors (A1's build_map). Two configurations, both frozen
before the test gold is read:
- votes (the pre-registered main score): 8 relation samples -> vote shares -> per-type thresholds;
- greedy (reported beside it): the greedy answer, every relation it lists kept.

--dev: the 225 dev maps, with each episode's thresholds fitted on the other four (cross-fitted, as in A3.4).
--test: answers for the 11 test inputs (cached, exported), thresholds fitted on all of dev; every map must pass the
validator and is saved to results/a3/test_predictions[_greedy]/ before data/evaluation/ is read, once.

    python -m a3_finetune.predict --dev      -> results/a3/v1/dev_global.json
    python -m a3_finetune.predict --test     -> results/a3/test_predictions*/, results/a3/test_global.json
"""
import json
import sys
from collections import defaultdict

from common import RESULTS
from common.aif import load_map, test_paths
from common.score import aggregate, format_table, score_map
from common.split import episodes, map_speakers
from common.validate import validate
from a1_cross_encoder.decide import decode
from a1_cross_encoder.predict import build_map, relation_diagnostics
from a2_llm.check import dev_maps
from a3_finetune.data import dev_examples, test_examples
from a3_finetune.parse import pair_scores
from a3_finetune.runner import Runner, export, load_export
from a3_finetune.tune import ALL, EMPTY

OUT = RESULTS / "a3"
RUN, ADAPTER = "v1", "ep2.0"


def answers(rec) -> list[dict]:
    return [o["answer"] if o["valid"] else EMPTY for o in rec["outs"]]


def assemble(ex, samples, greedy, th) -> dict[str, dict]:
    """Both configurations' complete maps for one example."""
    out = {}
    for conf, ans, t in (("votes", answers(samples), th), ("greedy", answers(greedy), ALL)):
        rows, scores = pair_scores(ex.s, ans)
        out[conf] = build_map(ex.inp, ex.name, decode(rows, scores, t), ex.links, ex.s.pos)
    return out


def summary(golds: dict, preds: dict) -> dict:
    names = sorted(preds)
    per_map = [score_map(golds[n], preds[n]) for n in names]
    agg = aggregate(per_map)
    return {"table": format_table(agg), **{k: agg[k] for k in ("ari", "ilo", "global", "skipped_maps")},
            "per_map_ari_f1": {n: [round(100 * s["ari"][0][2], 2), round(100 * s["ari"][1][2], 2) if s["ari"][1] else None]
                               for n, s in zip(names, per_map)},
            "diagnostics": relation_diagnostics(golds, preds)}


def dev():
    report = json.loads((OUT / RUN / "dev_report.json").read_text())
    folds = report["folds"]
    ep_of, *_ = episodes(map_speakers())
    samples, greedy = load_export(RUN, f"{ADAPTER}_dev_samples8"), load_export(RUN, f"{ADAPTER}_dev_greedy")
    golds = {name: g for name, g, _, _ in dev_maps()}
    preds, invalid = defaultdict(dict), []
    for ex in dev_examples():
        fold = folds[str(ep_of[int(ex.name.removeprefix("nodeset"))])]
        th = {k: fold[k] for k in ("Inference", "Conflict", "Rephrase")}
        for conf, m in assemble(ex, samples[ex.name], greedy[ex.name], th).items():
            if validate(m):
                invalid.append((conf, ex.name))
            preds[conf][ex.name] = m
    res = {"maps": len(golds), "invalid": invalid, **{conf: summary(golds, p) for conf, p in preds.items()}}
    (OUT / RUN / "dev_global.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    for conf in preds:
        print(conf, f"(honest, cross-fitted), {len(golds)} maps, invalid: {len(invalid)}")
        print(res[conf]["table"])


def test():
    th = json.loads((OUT / RUN / "thresholds.json").read_text())["thresholds"]
    exs = test_examples()
    r = Runner(RUN, ADAPTER, json.loads((OUT / RUN / "config.json").read_text())["model"])
    samples = r.samples(exs, 8, seed=0)
    greedy = r.greedy(exs)
    export(RUN, f"{ADAPTER}_test_samples8", samples)
    export(RUN, f"{ADAPTER}_test_greedy", greedy)
    r.close()
    preds, problems = defaultdict(dict), {}
    for ex in exs:
        for conf, m in assemble(ex, samples[ex.name], greedy[ex.name], th).items():
            bad = validate(m)
            if bad:
                problems[(conf, ex.name)] = bad[:3]
            d = OUT / ("test_predictions" if conf == "votes" else "test_predictions_greedy")
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{ex.name}.json").write_text(json.dumps(m, indent=1), encoding="utf-8", newline="\n")
            preds[conf][ex.name] = m
    assert all(len(p) == 11 for p in preds.values()) and not problems, problems
    print("22 maps (11 per configuration) assembled, validated and saved before reading the gold")

    gold_paths = {name: g for name, _, g in test_paths()}             # the gold is read here, once
    golds = {name: load_map(gold_paths[name]) for name in preds["votes"]}
    res = {"run": RUN, "adapter": ADAPTER, "relation_thresholds": th,
           "valid_answers": sum(o["valid"] for rec in samples.values() for o in rec["outs"]),
           **{conf: summary(golds, p) for conf, p in preds.items()}}
    (OUT / "test_global.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    for conf in preds:
        print(conf)
        print(res[conf]["table"])
        print(json.dumps(res[conf]["diagnostics"]))


if __name__ == "__main__":
    if "--dev" in sys.argv:
        dev()
    if "--test" in sys.argv:
        test()
