"""How Jev performs: read-only error analysis on dev for reports/A4-discussion.md (no calls, nothing tuned).

Systems: A1 (trained encoder), A2 (Claude whole-map), and A4 (Jev) with its pair view, source view and both, every
one at matched density (one threshold, as many relations as gold, as in A3's discussion), so the comparisons are
between models, not thresholds. Pairs are unordered; a prediction is right when label and direction match gold.
1. what each gold relation becomes (right, reversed, another type, missed), per type;
2. recall by turn distance and, for Rephrase, by content-word overlap; Conflict within vs across speakers;
3. invented relations by predicted type;
4. pair vs source questions: which gold relations each view finds (complementarity);
5. honest per-map ARI, A4 against A1 and A2, by map size;
6. test per-map ARI general of all four systems (read from the committed test results).

    python -m a4_jev.discuss      -> results/a4/discussion.json
"""
import json
from collections import Counter, defaultdict

from common import RESULTS
from common.aif import view
from a2_llm.check import dev_maps
from a2_llm.discuss import bucket, label, outcome, rate, words
from a2_llm.tune import both
from a3_finetune.tune import Base, a2_scores, honest, matched
from a4_jev.parse import VIEWS
from a4_jev.runner import answers, builds
from a4_jev.tune import RUN, a1_tempered, a4_scores

OUT = RESULTS / "a4"
DIST = ([1, 3, 10], ["1 turn", "2-3 turns", "4-10 turns"])


def main():
    base = Base()
    bs, ans = builds(RUN), answers(RUN)
    systems = {"A1": a1_tempered(base), "A2": a2_scores(base), **{f"A4 {v}": a4_scores(base, bs, ans, v) for v in VIEWS}}
    preds = {}
    for k, s in systems.items():
        th = matched(s)
        preds[k] = {n: s.predicted(s.probs, n, th) for n in base.names}
    texts, spk = {}, base.speaker
    for name, g, _, _ in dev_maps():
        texts[name] = view(g, name).props

    becomes = {k: defaultdict(Counter) for k in systems}
    recall = {k: defaultdict(Counter) for k in systems}
    invented = {k: Counter() for k in systems}
    found_by = defaultdict(Counter)                    # gold relation -> which A4 views find it
    for n in base.names:
        g, pos = base.gold[n], base.s[n].pos
        pairs = {frozenset(x) for x in g} | {frozenset(x) for k in systems for x in preds[k][n]}
        for pr in pairs:
            a, b = tuple(pr)
            gl, _ = label(g, a, b)
            d = abs(pos[a] - pos[b]) if a in pos and b in pos else None
            hits = {}
            for k in systems:
                gl2, pl, right = outcome(g, preds[k][n], a, b)
                hits[k] = right
                if gl == "None":
                    if pl != "None":
                        invented[k][pl] += 1
                    continue
                becomes[k][gl]["right" if right else pl] += 1
                band = bucket(d, *DIST) if d is None or d <= 10 else "11+ turns"
                recall[k][f"{gl}, {band}"]["gold"] += 1
                recall[k][f"{gl}, {band}"]["found"] += right
                if gl == "Rephrase":
                    wa, wb = words(texts[n][a]), words(texts[n][b])
                    j = len(wa & wb) / len(wa | wb) if wa | wb else 0
                    o = "no shared content words" if j == 0 else "overlap >= 0.5" if j >= 0.5 else "some overlap"
                    recall[k][f"Rephrase, {o}"]["gold"] += 1
                    recall[k][f"Rephrase, {o}"]["found"] += right
                if gl == "Conflict" and a in spk[n] and b in spk[n]:
                    who = "same speaker" if spk[n][a] == spk[n][b] else "different speakers"
                    recall[k][f"Conflict, {who}"]["gold"] += 1
                    recall[k][f"Conflict, {who}"]["found"] += right
            if gl != "None":
                key = "+".join(v for v in ("pair", "source") if hits[f"A4 {v}"]) or "neither"
                found_by[gl][key] += 1

    # Honest per-map ARI by map size (cross-fitted thresholds, as in the main results).
    held = {k: honest(systems[k])[1] for k in ("A1", "A2", "A4 both")}
    size = {n: len(base.s[n].prop) for n in base.names}
    bands = [(0, 10, "<= 10 propositions"), (11, 16, "11-16"), (17, 25, "17-25"), (26, 99, "26+")]
    by_size = {}
    for lo, hi, nm in bands:
        ns = [n for n in base.names if lo <= size[n] <= hi]
        if ns:
            m = lambda k: sum(100 * held[k][x][0][2] for x in ns) / len(ns)
            by_size[nm] = {"maps": len(ns), "A4 - A1": round(m("A4 both") - m("A1"), 2), "A4 - A2": round(m("A4 both") - m("A2"), 2)}

    test = {"A1": json.loads((RESULTS / "a1" / "v1" / "test_global.json").read_text()),
            "A2": json.loads((RESULTS / "a2" / "test_global.json").read_text()),
            "A3": json.loads((RESULTS / "a3" / "test_global.json").read_text())["votes"],
            "A4": json.loads((OUT / "test_global.json").read_text())}
    per_map = {}
    for k, d in test.items():
        pm = d.get("per_map_ari_f1") or {n: v["ari"] for n, v in d.get("per_map_f1", {}).items()}
        per_map[k] = {n: v[0] for n, v in pm.items()}
    maps = sorted(per_map["A4"], key=lambda x: int(x[3:]))
    wins = {k: sum(per_map["A4"][m] > per_map[k][m] for m in maps) for k in ("A1", "A2", "A3") if per_map.get(k)}

    rep = {
        "matched_density_ari": {k: both(s.scores(s.probs, matched(s))) for k, s in systems.items()},
        "what_gold_becomes": {k: {t: dict(c) for t, c in v.items()} for k, v in becomes.items()},
        "recall": {k: {x: {"gold": c["gold"], "recall": rate(c["found"], c["gold"])} for x, c in sorted(v.items())}
                   for k, v in recall.items()},
        "invented_by_type": {k: dict(v) for k, v in invented.items()},
        "a4_views_finding_each_gold_relation": {t: dict(c) for t, c in found_by.items()},
        "honest_per_map_by_size": by_size,
        "test_per_map_ari_general": {"maps": maps, **{k: [per_map[k].get(m) for m in maps] for k in per_map},
                                     "A4_better_than": wins},
    }
    with open(OUT / "discussion.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
