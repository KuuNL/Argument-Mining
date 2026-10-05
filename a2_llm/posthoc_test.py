"""Post-hoc look at the A2 test run (read-only; changes nothing that was submitted).

- Which turn types the LLM changed on test, and whether each change was right.
- The same 11 maps with the labeller's types restored: a diagnostic only, *not* the submitted A2 system
  (the turn-type rule was fixed on dev before test, D063).
- ARI and ILO per map, A2 against A1.

    python -m a2_llm.posthoc_test    -> results/a2/test_posthoc.json
"""
import json
from collections import Counter

from common import RESULTS, ROOT
from common.aif import load_map, test_paths
from common.score import aggregate, ilo_labels, score_map


def main():
    links = json.loads((ROOT / "cache" / "ilo" / "links_test.json").read_text())
    a1 = json.loads((RESULTS / "a1" / "v1" / "test_global.json").read_text())["per_map_f1"]
    a2 = json.loads((RESULTS / "a2" / "test_global.json").read_text())["per_map_f1"]
    changes, restored = Counter(), []
    for name, _, gold_path in test_paths():
        g = load_map(gold_path)
        m = json.loads((RESULTS / "a2" / "test_predictions" / f"{name}.json").read_text())
        props = {n["nodeID"] for n in g["nodes"] if n["type"] == "I"}
        locs = {n["nodeID"] for n in g["nodes"] if n["type"] == "L"}
        gold, pred = ilo_labels(g, props | locs), ilo_labels(m, props | locs)
        labeller = {(t, p): y for t, p, y in links.get(name, [])}
        for (t, p), y in labeller.items():
            new = pred.get((t, p))
            if new != y:
                verdict = "right" if gold.get((t, p)) == new else "was right" if gold.get((t, p)) == y else "both wrong"
                changes[f"{y} -> {new} ({verdict})"] += 1
        m2 = json.loads(json.dumps(m))
        into = {e["toID"]: e["fromID"] for e in m2["edges"]}
        out = {e["fromID"]: e["toID"] for e in m2["edges"]}
        for n in m2["nodes"]:
            key = (into.get(n["nodeID"]), out.get(n["nodeID"]))
            if n["type"] == "YA" and n["nodeID"].startswith("a1_ya_") and key in labeller:
                n["text"] = labeller[key]
        restored.append(score_map(g, m2))
    agg = aggregate(restored)
    wins = sum(a2[n]["ari"][0] > a1[n]["ari"][0] for n in a1)
    report = {
        "turn_type_changes_on_test": dict(changes.most_common()),
        "labeller_types_restored (diagnostic, not submitted)": {
            task: {s: round(agg[task][s][2], 2) for s in ("general", "focused")} for task in ("ari", "ilo", "global")},
        "ari_general_per_map": {n: {"a1": a1[n]["ari"][0], "a2": a2[n]["ari"][0]} for n in sorted(a1, key=lambda x: int(x[3:]))},
        "a2_better_ari_maps": f"{wins} of {len(a1)}",
        "ilo_general_per_map": {n: {"a1": a1[n]["ilo"][0], "a2": a2[n]["ilo"][0]} for n in sorted(a1, key=lambda x: int(x[3:]))},
    }
    with open(RESULTS / "a2" / "test_posthoc.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
