"""Phase 2 exploratory data analysis (PLAN Phase 1, A1 slice) -> eda/findings.json.

Training split only (dev stays clean). Test *inputs* are read for sizes and text overlap;
evaluation gold is never read. The findings are explained in eda/README.md.

    python -m eda.run_eda
"""
import copy
import json
import random
import re
import statistics as st
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from common.aif import load_map, test_paths, view
from common.score import score_map
from common.split import HOST, load_split
from common.timing import timed

OUT = Path(__file__).resolve().parent / "findings.json"
REL = {"RA": "Inference", "CA": "Conflict", "MA": "Rephrase"}
QUESTIONING = {"Pure Questioning", "Assertive Questioning", "Rhetorical Questioning"}
MAX_K = 12
TEST_LIKE = 40      # test maps have at most 35 propositions; a few training maps have up to 287
SIMILAR = 0.8       # text similarity for placing an unanchored proposition at its most similar turn
TOKENIZER = "answerdotai/ModernBERT-base"
WINDOWS = (0, 1, 2, 3, 5)


def pct(xs, qs=(50, 90, 99)):
    xs = sorted(xs)
    return {f"p{q}": xs[min(len(xs) - 1, int(q / 100 * len(xs)))] for q in qs} | {"max": xs[-1], "mean": round(st.mean(xs), 2)}


def share(a, b):
    return round(a / b, 4) if b else None


# ---------------------------------------------------------------- per-map facts

def norm(s):
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


def positions(v):
    """Dialogue turn of each proposition: its anchoring locution, else the transition it hangs off,
    else its most similar turn (>= SIMILAR). The rest are 'off-dialogue'."""
    turn = v.turn
    pos, how = {}, {}
    for s, t, _ in v.anchors():
        if s in turn and (t not in pos or turn[s] < pos[t]):
            pos[t], how[t] = turn[s], "locution"
    types = {n["nodeID"]: n["type"] for n in v.aif["nodes"]}
    ta_dest = defaultdict(list)
    for e in v.aif["edges"]:
        if types.get(e["fromID"]) == "TA" and e["toID"] in turn:
            ta_dest[e["fromID"]].append(turn[e["toID"]])
    for s, t, _ in v.illocutions:
        if t in v.props and t not in pos and ta_dest.get(s):
            pos[t], how[t] = min(ta_dest[s]), "transition"
    for i, text in v.props.items():
        if i not in pos and v.dialogue:
            sim, k = max((SequenceMatcher(None, norm(text), norm(v.locs[l].body)).ratio(), k)
                         for k, l in enumerate(v.dialogue))
            if sim >= SIMILAR:
                pos[i], how[i] = k, "similar-text"
    return pos, how


def prop_speaker(v, pos):
    by_turn = {k: v.locs[l].speaker for k, l in enumerate(v.dialogue)}
    return {i: by_turn[p] for i, p in pos.items()}


def relation_pairs(v):
    """(source I, target I, type) as the scorer sees them; one row per premise."""
    return [(p, r.target, r.type) for r in v.relations if r.target in v.props for p in r.premises]


# ---------------------------------------------------------------- analyses

def anatomy(views, tests):
    def sizes(vs):
        return {"dialogue_turns": pct([len(v.dialogue) for v in vs]),
                "propositions": pct([len(v.props) for v in vs]),
                "speakers": pct([len({v.locs[i].speaker for i in v.dialogue}) for v in vs])}
    host_first = sum(bool(v.dialogue) and v.locs[v.dialogue[0]].speaker == HOST for v in views)
    big = [v for v in views if len(v.props) > TEST_LIKE]
    n_pairs = lambda vs: sum(len(v.props) * (len(v.props) - 1) for v in vs)
    n_rel = lambda vs: sum(len(relation_pairs(v)) for v in vs)
    return {"train": sizes(views), "test_inputs": sizes(tests),
            "maps_over_test_like": len(big), "share_of_pairs_in_them": share(n_pairs(big), n_pairs(views)),
            "share_of_relations_in_them": share(n_rel(big), n_rel(views)),
            "test_map_sizes": [(v.name, len(v.props), len(v.dialogue)) for v in tests],
            "maps": len(views), "host_opens": share(host_first, len(views)),
            "analyst_copy_share": share(sum(l.is_copy for v in views for l in v.locs.values()),
                                        sum(len(v.locs) for v in views))}


def anchoring(views, pos_how):
    per_prop, per_turn, how = [], Counter(), Counter()
    off_in_rel = 0
    for v in views:
        pos, h = pos_how[v.name]
        n_anch = Counter(t for s, t, _ in v.anchors())
        per_prop += [n_anch.get(i, 0) for i in v.props]
        per_turn.update(Counter(min(c, 3) for c in Counter(s for s, _, _ in v.anchors()).values()))
        per_turn[0] += sum(1 for l in v.dialogue if l not in {s for s, _, _ in v.anchors()})
        how.update(h.values())
        in_rel = {x for p, t, _ in relation_pairs(v) for x in (p, t)}
        for i in v.props:
            if i not in pos:
                how["off-dialogue"] += 1
            if h.get(i) in (None, "similar-text"):
                off_in_rel += i in in_rel
    n = len(per_prop)
    return {"propositions": n, "positioned_by": dict(how),
            "positioned_share": share(n - how["off-dialogue"], n),
            "locutions_per_proposition": dict(Counter(min(c, 3) for c in per_prop)),
            "propositions_per_turn": dict(per_turn),
            "unanchored_in_a_relation": share(off_in_rel, how["off-dialogue"] + how["similar-text"])}


def illocutions(views):
    labels, q_mark, xxx, wh = Counter(), Counter(), Counter(), Counter()
    wh_re = re.compile(r"^(what|why|how|who|whom|which|when|where|do|does|did|is|are|was|were|should|"
                       r"can|could|will|would|shall|have|has)\b", re.I)
    for v in views:
        for s, t, y in v.anchors():
            if s not in v.turn:
                continue
            labels[y] += 1
            q_mark[y] += "?" in v.locs[s].body
            xxx[y] += "xxx" in v.props[t].lower()
            wh[y] += bool(wh_re.match(v.locs[s].body))
    # Alignment by text similarity alone: does each turn's most similar proposition match gold?
    hit = tot = hit_pos = 0
    for v in views:
        gold = defaultdict(set)
        for s, t, _ in v.anchors():
            gold[s].add(t)
        for s, targets in gold.items():
            if s not in v.turn:
                continue
            sims = {i: SequenceMatcher(None, norm(v.locs[s].body), norm(txt)).ratio() for i, txt in v.props.items()}
            tot += 1
            hit += max(sims, key=sims.get) in targets
    return {"L_to_I_labels": dict(labels.most_common()),
            "has_question_mark": {y: share(q_mark[y], labels[y]) for y in labels},
            "proposition_has_xxx": {y: share(xxx[y], labels[y]) for y in labels},
            "turn_starts_with_question_word": {y: share(wh[y], labels[y]) for y in labels},
            "similarity_alignment_top1": share(hit, tot), "alignment_turns": tot}


def relations(views, pos_how):
    counts, snodes_premises, maps_with = Counter(), Counter(), Counter()
    direction = defaultdict(Counter)
    dist = defaultdict(list)
    speakers = defaultdict(Counter)
    multi, q_ma, q_total, q_with_ma = 0, 0, 0, 0
    ta_ya = defaultdict(Counter)
    unpositioned_rel = Counter()
    for v in views:
        pos, _ = pos_how[v.name]
        spk = prop_speaker(v, pos)
        pairs = relation_pairs(v)
        for r in v.relations:
            snodes_premises[r.type, min(len(r.premises), 4)] += 1
        kinds = {t for _, _, t in pairs}
        for k in REL:
            maps_with[k] += k in kinds
        maps_with["none"] += not kinds
        seen = Counter(frozenset((a, b)) for a, b, _ in pairs)
        multi += sum(c > 1 for c in seen.values())
        q_props = {t for s, t, y in v.anchors() if y in QUESTIONING}
        q_total += len(q_props)
        q_with_ma += sum(1 for q in q_props if any(t == q and ty == "MA" for _, t, ty in pairs))
        for a, b, ty in pairs:
            counts[ty] += 1
            if ty == "MA":
                q_ma += b in q_props
            if a not in pos or b not in pos:
                unpositioned_rel[ty] += 1
                continue
            d = pos[a] - pos[b]
            direction[ty]["later_to_earlier" if d > 0 else "same_turn" if d == 0 else "earlier_to_later"] += 1
            dist[ty].append(abs(d))
            speakers[ty]["same" if spk[a] == spk[b] else "different"] += 1
        # which unscored YA the annotators put on each relation's transition
        yas = {n["nodeID"]: n["text"] for n in v.aif["nodes"] if n["type"] == "YA"}
        stype = {r.id: r.type for r in v.relations}
        for e in v.aif["edges"]:
            if e["fromID"] in yas and e["toID"] in stype:
                ta_ya[stype[e["toID"]]][yas[e["fromID"]]] += 1

    all_d = [d for ty in dist for d in dist[ty]]
    coverage = {ty: [share(sum(d <= k for d in ds), len(ds)) for k in range(MAX_K + 1)]
                for ty, ds in list(dist.items()) + [("all", all_d)]}
    k_for = {f"{int(q * 100)}%": {ty: next((k for k, c in enumerate(cov) if c >= q), None) for ty, cov in coverage.items()}
             for q in (0.95, 0.98, 0.99)}
    hist = {ty: [sum(d == k for d in ds) for k in range(MAX_K + 1)] + [sum(d > MAX_K for d in ds)]
            for ty, ds in dist.items()}
    return {"relation_pairs": dict(counts), "s_nodes_by_premises": {f"{t}:{n}": c for (t, n), c in sorted(snodes_premises.items())},
            "maps_with": {k: share(c, len(views)) for k, c in maps_with.items()},
            "unordered_pairs_with_2plus_relations": multi,
            "direction": {ty: {k: share(c, sum(d.values())) for k, c in d.items()} for ty, d in direction.items()},
            "distance": {ty: pct(ds) for ty, ds in dist.items()}, "distance_hist": hist,
            "coverage_by_k": coverage, "k_for_coverage": k_for,
            "relations_with_off_dialogue_end": dict(unpositioned_rel),
            "speakers": {ty: {k: share(c, sum(d.values())) for k, c in d.items()} for ty, d in speakers.items()},
            "rephrase_targets_a_question": share(q_ma, counts["MA"]),
            "questions_answered_by_rephrase": share(q_with_ma, q_total),
            "transition_illocution_per_relation": {ty: dict(c.most_common(4)) for ty, c in ta_ya.items()}}


def pruning(views, pos_how):
    """Candidate pairs if only propositions within k turns are compared (off-dialogue ones always kept)."""
    rows = []
    for k in range(MAX_K + 1):
        cand = pos_rel = total_pairs = total_rel = both_pos = same_spk = 0
        for v in views:
            pos, _ = pos_how[v.name]
            spk = prop_speaker(v, pos)
            props = list(v.props)
            related = {frozenset((a, b)) for a, b, _ in relation_pairs(v)}
            for x in range(len(props)):
                for y in range(x + 1, len(props)):
                    a, b = props[x], props[y]
                    keep = a not in pos or b not in pos or abs(pos[a] - pos[b]) <= k
                    total_pairs += 1
                    total_rel += frozenset((a, b)) in related
                    cand += keep
                    pos_rel += keep and frozenset((a, b)) in related
                    if keep and a in pos and b in pos:
                        both_pos += 1
                        same_spk += spk[a] == spk[b]
        rows.append({"k": k, "candidates_share": share(cand, total_pairs), "relations_kept": share(pos_rel, total_rel),
                     "positive_rate": share(pos_rel, cand), "same_speaker_share": share(same_spk, both_pos)})
    return {"unordered_pairs": total_pairs, "related_unordered_pairs": total_rel, "by_k": rows}


def rare_label_trap(views, seed=0):
    """Score a perfect prediction plus ONE spurious label: what does a single mistake cost?"""
    rng = random.Random(seed)
    out = defaultdict(list)
    for v in views:
        gold = v.aif
        pairs = relation_pairs(v)
        related = {(a, b) for a, b, _ in pairs} | {(b, a) for a, b, _ in pairs}
        free = [(a, b) for a in v.props for b in v.props if a != b and (a, b) not in related]
        has_ca = any(t == "CA" for _, _, t in pairs)
        if free and pairs:
            a, b = rng.choice(free)
            pred = copy.deepcopy(gold)
            pred["nodes"].append({"nodeID": "spurious", "type": "CA", "text": "Default Conflict"})
            pred["edges"] += [{"fromID": a, "toID": "spurious"}, {"fromID": "spurious", "toID": b}]
            s = score_map(gold, pred)["ari"]
            out["ari_general_with_CA" if has_ca else "ari_general_no_CA"].append(s[0][2])
        anchored = {(s, t) for s, t, _ in v.anchors()}
        free_ilo = [(l, i) for l in v.dialogue for i in v.props if (l, i) not in anchored]
        labels = {y for _, _, y in v.anchors()}
        if free_ilo and "Challenging" not in labels:
            l, i = rng.choice(free_ilo)
            pred = copy.deepcopy(gold)
            pred["nodes"].append({"nodeID": "spurious", "type": "YA", "text": "Challenging"})
            pred["edges"] += [{"fromID": l, "toID": "spurious"}, {"fromID": "spurious", "toID": i}]
            out["ilo_general_new_label"].append(score_map(gold, pred)["ilo"][0][2])
    return {k: {"maps": len(xs), "mean_f1_after_one_mistake": round(100 * st.mean(xs), 2)} for k, xs in out.items()}


def lengths(views, tests, pos_how, k):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def n(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    prop_len, turn_len, map_len, win = [], [], [], defaultdict(list)
    for v in views:
        pos, _ = pos_how[v.name]
        pt = {i: n(t) for i, t in v.props.items()}
        tt = [n(f"{v.locs[l].speaker}: {v.locs[l].body}") + 1 for l in v.dialogue]   # +1 separator
        prop_len += pt.values()
        turn_len += tt
        map_len.append(sum(tt) + sum(pt.values()))
        props = list(v.props)
        for x in range(len(props)):
            for y in range(x + 1, len(props)):
                a, b = props[x], props[y]
                if a in pos and b in pos and abs(pos[a] - pos[b]) <= k:
                    lo, hi = min(pos[a], pos[b]), max(pos[a], pos[b])
                    for w in WINDOWS:
                        span = sum(tt[max(0, lo - w):hi + w + 1])
                        win[w].append(3 + pt[a] + pt[b] + span + 4)   # CLS/SEP/SEP + A/B markers
                    win["whole_map"].append(3 + pt[a] + pt[b] + sum(tt) + 4)
    test_map_len = []
    for v in tests:
        test_map_len.append(sum(n(f"{v.locs[l].speaker}: {v.locs[l].body}") + 1 for l in v.dialogue)
                            + sum(n(t) for t in v.props.values()))
    return {"tokenizer": TOKENIZER, "k_used": k, "proposition": pct(prop_len), "turn": pct(turn_len),
            "whole_map_text": pct(map_len), "test_whole_map_text": pct(test_map_len),
            "pair_input_by_window": {str(w): pct(xs) for w, xs in win.items()},
            "note": "approximate: sum of separately tokenized parts plus special/marker tokens"}


def leakage(views, tests):
    train_props = {norm(t) for v in views for t in v.props.values()}
    train_turns = {norm(v.locs[l].body) for v in views for l in v.dialogue}
    index = defaultdict(set)
    train_sets = []
    for t in train_props:
        s = set(t.split())
        if len(s) >= 5:
            train_sets.append(s)
            for w in s:
                index[w].add(len(train_sets) - 1)
    rows = []
    for v in tests:
        props = [norm(t) for t in v.props.values()]
        exact = sum(p in train_props for p in props)
        near = 0
        for p in props:
            s = set(p.split())
            if len(s) < 5:
                continue
            cands = Counter(j for w in s for j in index.get(w, ()))
            near += any(c / len(s | train_sets[j]) >= 0.8 for j, c in cands.items() if c >= 0.8 * len(s))
        turns = [norm(v.locs[l].body) for l in v.dialogue]
        rows.append({"map": v.name, "propositions": len(props), "exact_prop": exact, "near_prop_jaccard_0.8": near,
                     "exact_turn": sum(t in train_turns for t in turns if len(t.split()) >= 5)})
    dates = {v.name: sorted({l["start"][:10] for l in v.aif.get("locutions", []) if l.get("start")}) for v in tests}
    return {"per_test_map": rows, "test_dates": dates}


def main():
    with timed("common", "phase2_eda") as row:
        views = [view(load_map(p), p.stem) for p in load_split("train")]
        tests = [view(load_map(i), name) for name, i, _ in test_paths()]   # inputs only
        pos_how = {v.name: positions(v) for v in views}
        like = [v for v in views if len(v.props) <= TEST_LIKE]
        all_rel = relations(views, pos_how)
        findings = {"data": {"split": "train", "maps": len(views), "test_like_maps": len(like),
                             "test_like_max_propositions": TEST_LIKE},
                    "anatomy": anatomy(views, tests),
                    "anchoring": anchoring(views, pos_how),
                    "illocutions": illocutions(views),
                    "relations": relations(like, pos_how),
                    "relations_all_maps": {k: all_rel[k] for k in ("k_for_coverage", "direction", "speakers")},
                    "pruning": pruning(like, pos_how)}
        k = findings["relations"]["k_for_coverage"]["98%"]["all"]
        findings["rare_label_trap"] = rare_label_trap(views)
        findings["lengths"] = lengths(like, tests, pos_how, k)
        findings["leakage"] = leakage(views, tests)
        row.update(maps=len(views))
    OUT.write_text(json.dumps(findings, indent=1))
    print(json.dumps(findings, indent=1)[:20000])


if __name__ == "__main__":
    main()
