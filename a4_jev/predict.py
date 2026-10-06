"""A4.5: the single test run (D073, D077, D078).

Frozen configuration (D078): pair + source questions, both option orders, scores = the mean of the two views,
relation thresholds fitted on all of dev (results/a4/dev/thresholds.json), the ILO labeller's links and types.
The answers for the 11 test inputs come from the cached runner (`python -m a4_jev.runner --run=test --export`).
Every map must pass the validator and is saved to results/a4/test_predictions/ before data/evaluation/ is read,
once; nothing is changed afterwards.

    python -m a4_jev.predict      -> results/a4/test_predictions/, results/a4/test_global.json
"""
import json

from common import RESULTS
from common.aif import load_map, test_paths
from common.score import score_map
from common.timing import timed
from common.validate import validate
from a1_cross_encoder.decide import decode
from a1_cross_encoder.predict import build_map
from a2_llm.runner import targets
from a4_jev.parse import pair_scores, problems
from a4_jev.runner import answers, builds, config
from a4_jev.tune import VIEW, summary

OUT = RESULTS / "a4"
RUN = "test"


def main():
    th = json.loads((OUT / "dev" / "thresholds.json").read_text())["thresholds"]
    with timed("a4", "A4.5_test") as row:
        bs, ans = builds(RUN), answers(RUN)
        maps = targets(config(RUN))
        assert len(ans) == len(maps) == 11, (len(ans), len(maps))
        preds, bad = {}, {}
        (OUT / "test_predictions").mkdir(parents=True, exist_ok=True)
        for name, inp, links in maps:
            b, a = bs[name], ans[name][0]
            assert not problems(b, a), problems(b, a)[:3]
            m = build_map(inp, name, decode(*pair_scores(b, a, VIEW), th), links, b.s.pos)
            if validate(m):
                bad[name] = validate(m)[:3]
            (OUT / "test_predictions" / f"{name}.json").write_text(json.dumps(m, indent=1), encoding="utf-8", newline="\n")
            preds[name] = m
        assert not bad, bad
        print("11 maps assembled, validated and saved before reading the gold")

        gold_paths = {name: g for name, _, g in test_paths()}             # the gold is read here, once
        golds = {name: load_map(gold_paths[name]) for name in preds}
        res = {"run": RUN, "view": VIEW, "relation_thresholds": th, **summary(golds, preds)}
        per = {n: score_map(golds[n], preds[n])["ari"] for n in sorted(preds)}
        res["per_map_ari_f1"] = {n: [round(100 * s[0][2], 2), round(100 * s[1][2], 2) if s[1] else None] for n, s in per.items()}
        (OUT / "test_global.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
        row.update(maps=len(preds))
    print(res["table"])
    print(json.dumps(res["diagnostics"]))


if __name__ == "__main__":
    main()
