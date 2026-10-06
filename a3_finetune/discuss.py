"""Why A3 trails A1 and A2: read-only error analysis on dev for reports/A3-discussion.md (no model runs, nothing tuned).

All three systems are compared at matched density (8.32 relations per map, as in A3.4's hypotheses), so recall
differences aren't threshold differences. A gold relation is found when its label and direction match.
1. Recall by turn distance, word overlap, speaker, map size, and the source's position in the map (A3 writes its
   relations in P order, so late propositions come late in the answer).
2. What each system does with the gold Inferences it misses (none, reversed, Rephrase, Conflict).
3. Is the signal in A3's samples at all? Gold relations listed by at least one of the 8 samples, and the vote shares
   of gold vs non-gold listings; A2's 3 answers for comparison.
4. How far the samples drift from the greedy answer.
5. Per-map honest ARI gap to A2 by map size.

    python -m a3_finetune.discuss     -> results/a3/v1/discussion.json
"""
import json
import statistics as st
from collections import Counter, defaultdict

from a2_llm.discuss import bucket, words
from a3_finetune.runner import load_export
from a3_finetune.train import OUT
from a3_finetune.tune import Base, a1_scores, a2_scores, a3_scores, honest, matched
from a2_llm.check import dev_maps
from common.aif import view
from a3_finetune.model import tokenizer

RUN, ADAPTER = "v1", "ep2.0"
NAME = {0: "Inference", 1: "Conflict", 2: "Rephrase"}


def outcome(gold_code, a, b, pred):
    if pred.get((a, b)) == gold_code:
        return "right"
    if gold_code == 0 and pred.get((b, a)) == 0:
        return "reversed"
    other = pred.get((a, b), pred.get((b, a)))
    return "missed" if other is None else f"called {NAME[other]}"


def main():
    base = Base()
    text = {name: view(g, name).props for name, g, _, _ in dev_maps()}
    recs = load_export(RUN, f"{ADAPTER}_dev_samples8")
    greedy = load_export(RUN, f"{ADAPTER}_dev_greedy")
    a3 = a3_scores(base, recs)
    tok = tokenizer(json.loads((OUT / RUN / "config.json").read_text())["model"])
    a3_tok = a3_scores(base, recs, weights_tok=tok)
    a3_tie = type(a3)(base, {n: a3.probs[a3.idx[n]] + 1e-3 * a3_tok.probs[a3_tok.idx[n]] for n in base.names})
    systems = {"A1": a1_scores(base), "A2": a2_scores(base), "A3": a3_tie}
    preds = {}
    for k, dev in systems.items():
        th = matched(dev)
        preds[k] = {n: dev.predicted(dev.probs, n, th) for n in base.names}

    feats = {"distance": lambda d: bucket(d, [0, 1, 3, 7, 10 ** 6], ["same turn", "1", "2-3", "4-7", "8+"]),
             "overlap": lambda o: bucket(o, [0.0, 0.15, 0.3, 1.0], ["0", "0-0.15", "0.15-0.3", ">0.3"])}
    table = {f: defaultdict(Counter) for f in ("distance", "overlap", "speakers", "map size", "source position")}
    errors = {k: Counter() for k in systems}
    for n in base.names:
        s, g = base.s[n], base.gold[n]
        pos, spk, props = s.pos, base.speaker[n], s.prop
        order = {node: int(p[1:]) for p, node in props.items()}
        size = len(props)
        for (a, b), code in g.items():
            dist = abs(pos[a] - pos[b]) if a in pos and b in pos else None
            wa, wb = words(text[n][a]), words(text[n][b])
            ov = len(wa & wb) / len(wa | wb) if wa | wb else 0.0
            same = None if a not in spk or b not in spk else spk[a] == spk[b]
            keys = {"distance": feats["distance"](dist), "overlap": feats["overlap"](ov),
                    "speakers": {True: "same speaker", False: "different speakers", None: "unplaced"}[same],
                    "map size": bucket(size, [10, 16, 25, 10 ** 6], ["<=10", "11-16", "17-25", "26+"]),
                    "source position": bucket(order[a] / size, [1 / 3, 2 / 3, 1.0], ["first third", "middle third", "last third"])}
            for f, v in keys.items():
                table[f][v][f"{NAME[code]} gold"] += 1
            for k in systems:
                out = outcome(code, a, b, preds[k][n])
                if code == 0:
                    errors[k][out] += 1
                for f, v in keys.items():
                    table[f][v][f"{k} {NAME[code]} found"] += out == "right"

    def recall_table(t):
        res = {}
        for v, c in t.items():
            row = {}
            for rel in ("Inference", "Conflict", "Rephrase"):
                if c[f"{rel} gold"]:
                    row[rel] = {"gold": c[f"{rel} gold"],
                                **{k: round(c[f"{k} {rel} found"] / c[f"{rel} gold"], 3) for k in systems}}
            res[v] = row
        return res

    # 3. Signal in the samples; 4. drift from greedy
    listed_any, gold_total, gold_votes, other_votes, a2_any = 0, 0, [], [], 0
    jacc, greedy_kept = [], Counter()
    for n in base.names:
        s, g = base.s[n], base.gold[n]
        pid = s.pid
        gold_keys = {(NAME[c], pid[a], pid[b]) for (a, b), c in g.items() if a in pid and b in pid}
        votes = Counter(rel for o in recs[n]["outs"] if o["valid"]
                        for rel in {(r["type"], r["from"], r["to"]) for r in o["answer"]["relations"]})
        gold_total += len(gold_keys)
        listed_any += sum(1 for k in gold_keys if votes[k] > 0)
        gold_votes += [votes[k] / 8 for k in gold_keys]
        other_votes += [v / 8 for k, v in votes.items() if k not in gold_keys]
        gset = {(r["type"], r["from"], r["to"]) for r in (greedy[n]["outs"][0]["answer"] or {}).get("relations", [])}
        for o in recs[n]["outs"]:
            sset = {(r["type"], r["from"], r["to"]) for r in o["answer"]["relations"]} if o["valid"] else set()
            if gset | sset:
                jacc.append(len(gset & sset) / len(gset | sset))
        for k in gset:
            greedy_kept[votes[k]] += 1
    a2 = systems["A2"]
    for n in base.names:
        for (a, b), c in base.gold[n].items():
            row = next((k for k in a2.idx[n] if {a2.rows[k]["a"], a2.rows[k]["b"]} == {a, b}), None)
            if row is not None:
                fwd = a2.rows[row]["a"] == a
                col = (1 if fwd else 2) if c == 0 else (3 if c == 1 else 4)
                a2_any += a2.probs[row, col] > 0

    # 5. Per-map honest gap to A2 by map size
    held3 = honest(a3)[1]
    held2 = honest(systems["A2"])[1]
    gap = defaultdict(list)
    for n in base.names:
        size = len(base.s[n].prop)
        gap[bucket(size, [10, 16, 25, 10 ** 6], ["<=10", "11-16", "17-25", "26+"])].append(
            100 * (held3[n][0][2] - held2[n][0][2]))

    report = {
        "density": {k: round(sum(len(p) for p in preds[k].values()) / len(base.names), 2) for k in systems},
        "recall_at_matched_density": {f: recall_table(t) for f, t in table.items()},
        "gold_inference_outcomes": {k: dict(c) for k, c in errors.items()},
        "signal_in_samples": {
            "gold_relations": gold_total,
            "A3_listed_by_at_least_1_of_8": round(listed_any / gold_total, 3),
            "A2_listed_by_at_least_1_of_3": round(a2_any / gold_total, 3),
            "A3_mean_vote_share_gold_listed": round(st.mean([v for v in gold_votes if v > 0]), 3),
            "A3_mean_vote_share_non_gold_listed": round(st.mean(other_votes), 3),
            "A3_non_gold_listings": len(other_votes)},
        "samples_vs_greedy": {"mean_jaccard_sample_vs_greedy": round(st.mean(jacc), 3),
                              "greedy_relations_by_samples_listing_them": {f"{k}/8": greedy_kept[k] for k in range(9)}},
        "honest_ari_gap_A3_minus_A2_by_map_size": {k: {"maps": len(v), "mean": round(st.mean(v), 1),
                                                        "A3_better": sum(x > 0 for x in v)} for k, v in sorted(gap.items())},
    }
    (OUT / RUN / "discussion.json").write_text(json.dumps(report, indent=1, default=int) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=1, default=int))



if __name__ == "__main__":
    main()
