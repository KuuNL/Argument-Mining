"""Phase 1 gates: scorer, strip, validator, graph view, split.

    python -m common.check_foundation      (from the repo root, in the WSL venv)

Reads the 11 evaluation gold maps only to check the scorer (majority baseline, as the
plan specifies). Nothing is trained or tuned here.
"""
import copy
import json
import random
from collections import Counter

from . import RESULTS
from .aif import load_map, strip_to_input, test_paths, training_paths, view
from .official import run_official
from .score import format_table, score_map, score_maps
from .split import make_split, matches_stored
from .timing import timed
from .validate import YA_LABELS, problem_kind, validate

PAPER_MAJORITY_GENERAL = {"ari": (28.79, 30.28, 29.52), "ilo": (34.71, 35.90, 35.29), "global": (31.75, 33.09, 32.40)}
PLAN_MAJORITY_F1 = {"ari": 29.53, "ilo": 35.30, "global": 32.41}   # the gate in PLAN.md / CLAUDE.md
PAPER_TOL = 0.015   # the paper's own figures are not exact per-map means (D015)
N_PERTURBED = 150


def perturb(gold: dict, rng: random.Random) -> dict:
    """A plausible-but-wrong prediction that exercises every scorer quirk."""
    m = copy.deepcopy(gold)
    nodes, edges = m["nodes"], m["edges"]
    props = [n["nodeID"] for n in nodes if n["type"] == "I"]
    locs = [n["nodeID"] for n in nodes if n["type"] == "L"]
    drop = set()
    for n in nodes:
        r = rng.random()
        if n["type"] in ("RA", "CA", "MA"):
            if r < 0.25:
                drop.add(n["nodeID"])
            elif r < 0.45:
                n["type"] = rng.choice(["RA", "CA", "MA"])
            elif r < 0.6:   # reverse direction
                for e in edges:
                    if e["toID"] == n["nodeID"] or e["fromID"] == n["nodeID"]:
                        e["fromID"], e["toID"] = e["toID"], e["fromID"]
        elif n["type"] == "YA":
            if r < 0.2:
                drop.add(n["nodeID"])
            elif r < 0.4:
                n["text"] = rng.choice(sorted(YA_LABELS))
    m["nodes"] = nodes = [n for n in nodes if n["nodeID"] not in drop]
    m["edges"] = edges = [e for e in edges if e["fromID"] not in drop and e["toID"] not in drop]

    def add(ntype, text, src, dst, first=False):
        nid = f"x{len(nodes)}_{rng.randrange(10**6)}"
        (nodes.insert(0, {"nodeID": nid, "type": ntype, "text": text}) if first
         else nodes.append({"nodeID": nid, "type": ntype, "text": text}))
        for s in src:
            edges.append({"fromID": s, "toID": nid})
        for d in dst:
            edges.append({"fromID": nid, "toID": d})

    if len(props) >= 2:
        for _ in range(rng.randrange(4)):                      # spurious relations, some linked
            a, b = rng.sample(props, 2)
            extra = [rng.choice(props)] if rng.random() < 0.3 else []
            add(rng.choice(["RA", "CA", "MA"]), "x", [a] + extra, [b])
        a, b = rng.sample(props, 2)
        add("RA", "x", [a], [b, rng.choice(props)])               # two outgoing edges: first one counts
        if locs:
            add("CA", "x", [a], [rng.choice(locs)])                # non-I target
    if locs and props:
        for _ in range(2):                                         # duplicate YA on one (L, I) pair
            l, i = rng.choice(locs), rng.choice(props)
            add("YA", rng.choice(sorted(YA_LABELS)), [l], [i], first=rng.random() < 0.5)
    return m


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return all(abs(x - y) < 1e-9 for x, y in zip(a, b))


def main():
    report, ok = {}, {}
    with timed("common", "phase1_checks") as row:
        tests = [(name, load_map(i), load_map(g)) for name, i, g in test_paths()]

        # 1. Scorer reproduces the paper's majority baseline (predict nothing = the input map).
        agg = score_maps([(gold, inp) for _, inp, gold in tests])
        report["majority_table"] = format_table(agg)
        # F1 and precision within rounding. Recall is reported only: the paper's ARI recall (30.28)
        # is not reachable as a mean of per-map recalls (each is 1/k), so it isn't a per-map mean (D015).
        report["majority_vs_paper"] = {t: {"ours": [round(x, 4) for x in agg[t]["general"]],
                                           "paper": PAPER_MAJORITY_GENERAL[t]} for t in ("ari", "ilo", "global")}
        ok["majority_f1_equals_plan"] = all(round(agg[t]["general"][2], 2) == PLAN_MAJORITY_F1[t] for t in PLAN_MAJORITY_F1)
        ok["majority_p_f1_near_paper"] = all(abs(agg[t]["general"][k] - PAPER_MAJORITY_GENERAL[t][k]) <= PAPER_TOL
                                             for t in ("ari", "ilo", "global") for k in (0, 2))
        ok["majority_focused_zero"] = all(v == 0 for t in ("ari", "ilo", "global") for v in agg[t]["focused"])

        # 2. Port == official scripts, map by map.
        rng = random.Random(0)
        train = training_paths()
        cases = [(f"{n}:majority", g, i) for n, i, g in tests] + [(f"{n}:gold", g, g) for n, _, g in tests]
        for p in rng.sample(train, N_PERTURBED):
            g = load_map(p)
            cases.append((f"{p.stem}:perturbed", g, perturb(g, rng)))
        mismatches = []
        for name, gold, pred in cases:
            port = score_map(gold, pred)
            for task in ("ari", "ilo"):
                ref = run_official(task, gold, pred)
                if not (same(port[task][0], ref[0]) and same(port[task][1], ref[1])):
                    mismatches.append((name, task, port[task], ref))
        report["parity"] = {"cases": len(cases), "comparisons": 4 * len(cases), "mismatches": mismatches[:5],
                            "n_mismatches": len(mismatches)}
        ok["parity_with_official"] = not mismatches
        ok["gold_vs_gold_is_100"] = all(score_map(g, g)[t][s][2] == 1.0 for _, _, g in tests
                                        for t in ("ari", "ilo") for s in (0, 1))

        # 3. strip_to_input(gold) reproduces each test input exactly (D004).
        def canon(m):
            return (sorted((n["nodeID"], n["type"], n["text"]) for n in m["nodes"]),
                    sorted((e["fromID"], e["toID"]) for e in m["edges"]), m.get("locutions"))
        ok["strip_equals_test_input"] = all(canon(strip_to_input(g)) == canon(i) for _, i, g in tests)

        # 4. Validator: our stripped maps are clean; gold irregularities are counted.
        gold_problems, maps_with_problems = Counter(), 0
        views = []
        for p in train:
            m = load_map(p)
            probs = validate(m)
            maps_with_problems += bool(probs)
            gold_problems.update(problem_kind(x) for x in probs)
            views.append(view(m, p.stem))
        for _, _, g in tests:
            gold_problems.update(problem_kind(x) for x in validate(g))
        report["gold_validator"] = {"maps_with_problems": maps_with_problems, "by_kind": dict(gold_problems.most_common())}
        ok["stripped_maps_validate"] = all(not validate(strip_to_input(load_map(p))) for p in train[:200])

        # 5. Graph view sanity on all training maps.
        n_l = sum(len(v.locs) for v in views)
        n_copy = sum(1 for v in views for l in v.locs.values() if l.is_copy)
        n_exact = sum(1 for v in views for l in v.locs.values() if l.copy_of)
        anchors = [(v, s, t) for v in views for s, t, _ in v.anchors()]
        anchor_from_copy = sum(1 for v, s, _ in anchors if v.locs[s].is_copy)
        anchor_off_dialogue = sum(1 for v, s, _ in anchors if s not in set(v.dialogue))
        unanchored = sum(1 for v in views for i in v.props if i not in {t for _, t, _ in v.anchors()})
        report["graph_view"] = {
            "L_nodes": n_l, "analyst_copies": n_copy, "exact_copies": n_exact, "copy_share": round(n_copy / n_l, 3),
            "dialogue_L_nodes": sum(len(v.dialogue) for v in views),
            "L_to_I_anchors": len(anchors), "anchors_from_copies": anchor_from_copy,
            "anchors_from_non_dialogue_L": anchor_off_dialogue,
            "I_nodes": sum(len(v.props) for v in views), "I_nodes_without_anchor": unanchored,
            "maps_lt2_I": sum(len(v.props) < 2 for v in views),
            "maps_without_relations": sum(not v.relations for v in views),
            "speakers_empty": sum(1 for v in views for i in v.dialogue if not v.locs[i].speaker),
        }
        ok["no_anchor_from_a_copy"] = anchor_from_copy == 0
        ok["non_dialogue_anchors_under_1pct"] = anchor_off_dialogue < 0.01 * len(anchors)

        # 6. Episode-grouped split with a leakage audit. Recomputed and compared, never rewritten (D027).
        split, speakers, tr, dev, episode_of = make_split(write=False)
        ok["split_matches_tracked_files"] = matches_stored(tr, dev)
        report["split"] = {**split["counts"], "leaky_pairs": split["leaky_pairs"][:10],
                           "dev_episodes": [e for e in split["episodes"] if e["split"] == "dev"]}
        ok["split_disjoint_and_complete"] = not (tr & dev) and len(tr | dev) == len(train)
        ok["split_dev_share_14_to_17pct"] = 0.14 <= split["counts"]["dev_share"] <= 0.17
        ok["split_no_leaky_pairs"] = split["counts"]["leaky_dev_train_pairs"] == 0
        row.update(checks=len(ok), passed=sum(ok.values()))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase1_foundation.json").write_text(json.dumps({"ok": ok, **report}, indent=1, default=str))
    print(report["majority_table"])
    print(json.dumps({k: v for k, v in report.items() if k != "majority_table"}, indent=1, default=str)[:6000])
    print("\nGATES:", json.dumps(ok, indent=1))


if __name__ == "__main__":
    main()
