"""A2.5: the single test run. Complete maps for the 11 test inputs, saved before the gold is read (D049's rule).

Each map = the test input
  + A2's relations: the 3 answers of run test-v1, pair scores, thresholds fitted on all of dev (A2.3);
  + the ILO labeller's links (cache/ilo/links_test.json), with the LLM's type where it clears the
    per-type threshold fitted on all of dev (A2.4, adopted);
  + unscored transition anchors (A1's build_map).
Every map must pass the validator and is saved to results/a2/test_predictions/ first. Only then is
data/evaluation/ read, once, to score; nothing is tuned afterwards.

    python -m a2_llm.predict     -> results/a2/test_predictions/*.json, results/a2/test_global.json
"""
import json

from common import RESULTS, ROOT
from common.aif import load_map, test_paths
from common.score import aggregate, format_table, score_map
from common.validate import validate
from a1_cross_encoder.decide import decode
from a1_cross_encoder.predict import build_map, relation_diagnostics
from a2_llm.parse import pair_scores, type_votes
from a2_llm.runner import RUNS, jobs, load

OUT = RESULTS / "a2"
RUN, DEV = "test-v1", "dev-v1"
NEVER = 1.01


def typed(links, votes, th):
    """The labeller's links, with the LLM's most-voted type where it clears its threshold (as in evaluate.Types)."""
    out = []
    for t, p, y in links:
        v = votes.get((t, p), {})
        best = max(v, key=lambda k: (v[k], k == y)) if v else y
        out.append([t, p, best if best != y and v[best] >= th.get(best, NEVER) else y])
    return out


def main():
    rel_th = json.loads((OUT / DEV / "thresholds.json").read_text())["thresholds"]
    a24 = json.loads((OUT / DEV / "a24_report.json").read_text())["turn_types"]
    type_th = a24["thresholds"] if a24["adopted"] else {}
    links = json.loads((ROOT / "cache" / "ilo" / "links_test.json").read_text())
    inputs = {name: load_map(inp) for name, inp, _ in test_paths()}

    by_map = {}
    for job in jobs(RUNS[RUN]):
        by_map.setdefault(job["map"], []).append((job, load(job["key"], job["s"])))
    preds, problems, answers = {}, {}, {}
    pred_dir = OUT / "test_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name, items in sorted(by_map.items()):
        s = items[0][0]["s"]
        ans = [rec["attempts"][rec["answer"]]["result"]["structured_output"] for _, rec in items if rec and rec["ok"]]
        answers[name] = len(ans)
        rows, scores = pair_scores(s, ans)
        m = build_map(inputs[name], name, decode(rows, scores, rel_th),
                      typed(links.get(name, []), type_votes(s, ans), type_th), s.pos)
        bad = validate(m)
        if bad:
            problems[name] = bad[:3]
        with open(pred_dir / f"{name}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(m, f, indent=1)
        preds[name] = m
    assert len(preds) == 11 and not problems, (len(preds), problems)
    print(f"11 maps assembled, validated and saved to {pred_dir} before reading the gold; answers per map {answers}")

    gold_paths = {name: g for name, _, g in test_paths()}             # the gold is read here, once
    golds = {name: load_map(gold_paths[name]) for name in preds}
    per_map = [score_map(golds[n], preds[n]) for n in sorted(preds)]
    agg = aggregate(per_map)
    report = {"run": RUN, "relation_thresholds": rel_th, "type_thresholds": type_th, "answers_per_map": answers,
              "table": format_table(agg), **{k: agg[k] for k in ("ari", "ilo", "global", "skipped_maps")},
              "per_map_f1": {n: {"ari": [round(100 * s["ari"][0][2], 2), round(100 * s["ari"][1][2], 2) if s["ari"][1] else None],
                                 "ilo": [round(100 * s["ilo"][0][2], 2), round(100 * s["ilo"][1][2], 2) if s["ilo"][1] else None]}
                             for n, s in zip(sorted(preds), per_map)},
              "diagnostics": relation_diagnostics(golds, preds)}
    with open(OUT / "test_global.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1))
    print(report["table"])
    print(json.dumps(report["diagnostics"], indent=1))


if __name__ == "__main__":
    main()
