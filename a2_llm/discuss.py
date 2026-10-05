"""Why A2 and A1 differ: read-only error analysis on dev for reports/A2-discussion.md.

Both systems use thresholds fitted on all of dev (as in tune.overlap_with_a1). No model calls, nothing tuned.
Pairs are unordered; a prediction is right when its label and direction match gold.
1. where each system's errors go (gold label -> predicted label);
2. recall and precision by turn distance, by word overlap and by speaker;
3. Conflict: is the signal there but cut by the threshold, or absent?
4. Rephrase of a question (the IAT convention: an answer rephrases its question);
5. the A2 - A1 gap by map size, and the shape of the predicted graphs;
6. turn types: the LLM's vote against gold on dev, and the 3 test changes (post-hoc, read-only).

    python -m a2_llm.discuss     -> results/a2/discussion.json
"""
import json
import re
import statistics as st
from collections import Counter, defaultdict

from common import RESULTS, ROOT
from common.aif import load_map, test_paths, view
from common.score import ilo_labels
from a1_cross_encoder.decide import Dev as A1Dev, ari_f1
from a2_llm.evaluate import Types
from a2_llm.tune import Scores

NAME = {0: "Inference", 1: "Conflict", 2: "Rephrase", None: "None"}
QUESTION = {"Pure Questioning", "Assertive Questioning", "Rhetorical Questioning"}
STOP = set("""a an the and or but if of to in on at for with by from as is are was were be been being it its this that
these those there here i you he she we they them his her our their my your me us do does did not no so than then
just very can could would should will shall may might must have has had what which who whom how why when where
all any some more most about into over out up down also too only s t""".split())


def words(text):
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOP}


def label(d, a, b):
    """The relation between a and b in either direction: (class, direction-aware key) or ("None", None)."""
    if (a, b) in d:
        return NAME[d[(a, b)]], (a, b)
    if (b, a) in d:
        return NAME[d[(b, a)]], (b, a)
    return "None", None


def outcome(g, p, a, b):
    (gl, gk), (pl, pk) = label(g, a, b), label(p, a, b)
    return gl, ("Inference, reversed" if gl == pl == "Inference" and gk != pk else pl), (gl, gk) == (pl, pk)


def bucket(x, edges, names):
    for e, n in zip(edges, names):
        if x is None or x <= e:
            return n if x is not None else "unplaced"
    return names[-1]


def rate(hit, total):
    return round(hit / total, 3) if total else None


def main():
    a2 = Scores("dev-v1")
    a2_th = json.loads((RESULTS / "a2" / "dev-v1" / "thresholds.json").read_text())["thresholds"]
    a1 = A1Dev(ROOT / "checkpoints" / "a1" / "v1" / "dev_probs.npy")
    a1_cfg = json.loads((RESULTS / "a1" / "v1" / "thresholds.json").read_text())
    a1_probs = a1.calibrated(a1_cfg["temperature"])
    a1_conf = {(a1.rows[k]["a"], a1.rows[k]["b"]): float(a1_probs[k, 3]) for k in range(len(a1.rows))}
    links = json.loads((ROOT / "cache" / "ilo" / "links_dev.json").read_text())

    confusion = {"A1": Counter(), "A2": Counter()}
    by = {k: defaultdict(lambda: Counter()) for k in ("distance", "overlap", "speakers")}
    only_right = {"only A1 right": Counter(), "only A2 right": Counter()}
    conflict = Counter()
    q_reph = defaultdict(Counter)
    per_map, shape = [], {w: Counter() for w in ("gold", "A1", "A2")}
    for n in a2.names:
        g, v = a2.gold[n], view(load_map(ROOT / "data" / "training" / f"{n}.json"), n)
        p1, p2 = a1.predicted(a1_probs, n, a1_cfg["thresholds"]), a2.predicted(a2.probs, n, a2_th)
        pos = a2.s[n].pos
        speaker = {i: v.locs[v.dialogue[k]].speaker for i, k in pos.items()}
        gold_types = defaultdict(set)
        for (l, i), y in ilo_labels(v.aif, set(v.props) | set(v.locs)).items():
            if l in v.locs and i in v.props:
                gold_types[i].add(y)
        s1, s2 = ari_f1(g, p1, a2.n_pairs[n]), ari_f1(g, p2, a2.n_pairs[n])
        per_map.append((len(v.props), 100 * (s2[0][2] - s1[0][2])))
        for who, d in (("gold", g), ("A1", p1), ("A2", p2)):
            shape[who]["maps"] += 1
            shape[who]["relations"] += len(d)
            shape[who]["props"] += len(v.props)
            shape[who]["start 2+"] += sum(1 for c in Counter(a for a, _ in d).values() if c >= 2)
            shape[who]["receive 2+"] += sum(1 for c in Counter(b for _, b in d).values() if c >= 2)
        pairs = {frozenset(k) for d in (g, p1, p2) for k in d}
        for pair in pairs:
            a, b = sorted(pair)
            dist = abs(pos[a] - pos[b]) if a in pos and b in pos else None
            wa, wb = words(v.props[a]), words(v.props[b])
            ov = len(wa & wb) / len(wa | wb) if wa | wb else 0.0
            same = None if a not in speaker or b not in speaker else speaker[a] == speaker[b]
            feats = {"distance": bucket(dist, [0, 1, 3, 7, 10 ** 6], ["same turn", "1", "2-3", "4-7", "8+"]),
                     "overlap": bucket(ov, [0.0, 0.15, 0.3, 1.0], ["0", "0-0.15", "0.15-0.3", ">0.3"]),
                     "speakers": {True: "same speaker", False: "different speakers", None: "unplaced"}[same]}
            right = {}
            for who, p in (("A1", p1), ("A2", p2)):
                gl, pl, ok = outcome(g, p, a, b)
                right[who] = ok
                confusion[who][f"{gl} -> {pl}"] += 1
                for k, f in feats.items():
                    c = by[k][f]
                    if gl != "None":
                        c[f"{gl} gold"] += who == "A1"
                        c[f"{who} {gl} found"] += ok
                    if pl != "None":
                        c[f"{who} predicted"] += 1
                        c[f"{who} predicted right"] += ok
            if right["A1"] != right["A2"]:
                only_right["only A1 right" if right["A1"] else "only A2 right"][label(g, a, b)[0]] += 1
            gl, gk = label(g, a, b)
            if gl == "Conflict":
                row = next(k for k in a2.idx[n] if {a2.rows[k]["a"], a2.rows[k]["b"]} == {a, b})
                sc = a2.probs[row, 3]
                conflict["gold"] += 1
                conflict["different speakers"] += same is False
                conflict["A2 listed by >= 1 answer"] += sc > 0
                conflict["A2 above its threshold"] += sc >= a2_th["Conflict"]
                pa1 = a1_conf.get((a, b), a1_conf.get((b, a), 0.0))     # pairs A1 pruned count as 0
                conflict["A1 P(Conflict) >= 0.2"] += pa1 >= 0.2
                conflict["A1 above its threshold"] += pa1 >= a1_cfg["thresholds"]["Conflict"]
                conflict[f"A1 called it {outcome(g, p1, a, b)[1]}"] += 1
                conflict[f"A2 called it {outcome(g, p2, a, b)[1]}"] += 1
            if gl == "Rephrase":
                kind = "of a question" if gold_types.get(gk[1], set()) & QUESTION else "other"
                q_reph[kind]["gold"] += 1
                q_reph[kind]["A1 found"] += outcome(g, p1, a, b)[2]
                q_reph[kind]["A2 found"] += outcome(g, p2, a, b)[2]
        rows = [a2.rows[k] for k in a2.idx[n]]
        listed = a2.probs[a2.idx[n], 3] > 0
        conflict["A2 pairs listed as Conflict by >= 1 answer"] += int(listed.sum())
        conflict["of which gold Conflict"] += sum(label(g, r["a"], r["b"])[0] == "Conflict" for r, l in zip(rows, listed) if l)

    def summarise(table):
        out = {}
        for f, c in table.items():
            row = {"gold relations": sum(v for k, v in c.items() if k.endswith(" gold"))}
            for who in ("A1", "A2"):
                found = sum(v for k, v in c.items() if k.startswith(who) and k.endswith(" found"))
                row[f"{who} recall"] = rate(found, row["gold relations"])
                row[f"{who} precision"] = rate(c[f"{who} predicted right"], c[f"{who} predicted"])
                row[f"{who} predicted"] = c[f"{who} predicted"]
            for rel in ("Inference", "Conflict", "Rephrase"):
                if c[f"{rel} gold"]:
                    row[rel] = {"gold": c[f"{rel} gold"], "A1 recall": rate(c[f"A1 {rel} found"], c[f"{rel} gold"]),
                                "A2 recall": rate(c[f"A2 {rel} found"], c[f"{rel} gold"])}
            out[f] = row
        return out

    sizes = [("<=10", 10), ("11-16", 16), ("17-25", 25), ("26+", 10 ** 6)]
    size_gap = {}
    for name, hi in sizes:
        lo = {"<=10": 0, "11-16": 10, "17-25": 16, "26+": 25}[name]
        d = [x for n_, x in per_map if lo < n_ <= hi]
        size_gap[name] = {"maps": len(d), "mean A2 - A1 (general)": round(st.mean(d), 1) if d else None,
                          "A2 better": sum(x > 0 for x in d)}

    a2.gold_aif = {n: load_map(ROOT / "data" / "training" / f"{n}.json") for n in a2.names}
    ty = Types(a2)
    vote_vs_gold = Counter()
    for n in ty.names:
        for (t, p), v in ty.votes[n].items():
            top = max(v, key=v.get)
            gold = ty.gold[n].get((t, p), "no link")
            if top != "Asserting" or gold != "Asserting":
                vote_vs_gold[f"LLM {top} | gold {gold}"] += 1

    test_changes = []
    pred_dir = RESULTS / "a2" / "test_predictions"
    test_links = json.loads((ROOT / "cache" / "ilo" / "links_test.json").read_text())
    for name, inp, gold_path in test_paths():
        m, g = json.loads((pred_dir / f"{name}.json").read_text()), load_map(gold_path)
        vg = view(g, name)
        pred = ilo_labels(m, set(vg.props) | set(vg.locs))
        gold = ilo_labels(g, set(vg.props) | set(vg.locs))
        for t, p, y in test_links.get(name, []):
            if pred.get((t, p)) != y:
                test_changes.append({"map": name, "turn": vg.locs[t].text, "labeller": y,
                                     "llm": pred.get((t, p)), "gold": gold.get((t, p))})

    report = {
        "errors_by_gold_label": {who: dict(sorted(c.items())) for who, c in confusion.items()},
        "pairs_only_one_system_gets_right": {k: dict(c) for k, c in only_right.items()},
        "by_turn_distance": summarise(by["distance"]),
        "by_word_overlap (Jaccard of content words)": summarise(by["overlap"]),
        "by_speaker": summarise(by["speakers"]),
        "conflict": dict(conflict),
        "rephrase_by_target": {k: {**c, "A1 recall": rate(c["A1 found"], c["gold"]), "A2 recall": rate(c["A2 found"], c["gold"])}
                               for k, c in q_reph.items()},
        "a2_minus_a1_by_map_size": size_gap,
        "graph_shape": {w: {"relations per map": round(c["relations"] / c["maps"], 2),
                            "props starting 2+": rate(c["start 2+"], c["props"]),
                            "props receiving 2+": rate(c["receive 2+"], c["props"])} for w, c in shape.items()},
        "turn_types_llm_vote_vs_gold_dev (non-Asserting only)": dict(vote_vs_gold.most_common(25)),
        "turn_type_changes_on_test (post-hoc)": test_changes,
    }
    with open(RESULTS / "a2" / "discussion.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1, ensure_ascii=False, default=lambda o: o.item()))
    print(json.dumps(report, indent=1, ensure_ascii=False, default=lambda o: o.item()))


if __name__ == "__main__":
    main()
