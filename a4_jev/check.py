"""A4.1 gate checks, no API calls.

1. Lossless round trip: each dev map's gold labels, written as Jev answers (one-hot pair answers; source answers
   splitting their mass evenly over a proposition's gold options), go through the answer check -> parse -> A1's
   decoder -> build_map -> validator -> official scorer. The pair view must give A1's pruning ceiling,
   98.64 / 94.44 (D041). The source and both views are reported (a proposition starting 3+ relations can't be
   represented above the round-trip threshold by one choice).
2. No node IDs anywhere in a request (D029).
3. Sizes: questions per map by kind, options per source question (Jev allows 255), estimated tokens and requests
   per map, for the pilot arms (defs short / full, k = 0 / 1), on the dev, test and pilot maps.
4. Cost projection for A4.2-A4.5 under the two billing hypotheses (state billed once per request, or once per
   question), at the CHARS_PER_TOKEN estimate; A4.1's probes replace the estimate with the measured ratio.
5. A readable sample request for the median dev map: results/a4/samples/.

    python -m a4_jev.check      -> results/a4/check.json
"""
import json
import re
import statistics as st
from collections import Counter

from common import RESULTS
from common.aif import load_map, test_paths, view
from common.score import ari_labels, score_map
from common.timing import timed
from common.validate import validate
from a1_cross_encoder.decide import decode
from a1_cross_encoder.predict import build_map
from a2_llm.check import dev_maps, links_of
from a2_llm.runner import resolve, targets
from a2_llm.schema import TURN_TYPES
from a4_jev import questions as Q
from a4_jev.parse import VIEWS, pair_scores, problems

OUT = RESULTS / "a4"
TH = {"Inference": 0.3, "Conflict": 0.3, "Rephrase": 0.3}
PRICE = 0.042 / 1e6                      # $ per input token (gateway listing, D074)
ARMS = {"short_k0": ("short", 0), "full_k0": ("full", 0), "short_k1": ("short", 1)}


def gold_answers(b, gold: dict) -> dict:
    """Jev-format answers that state the gold annotation exactly."""
    s = b.s
    rel = ari_labels(gold, set(s.prop.values()))
    label, wanted = {}, {}
    for a, c in b.candidates:
        A, C = s.prop[a], s.prop[c]
        ab, ba = rel.get((A, C)), rel.get((C, A))
        lab = 1 if ab == 0 else 2 if ba == 0 else 3 if 1 in (ab, ba) else 4 if 2 in (ab, ba) else 0
        label[(a, c)] = lab
        src = {1: (a, ("Inference", c)), 2: (c, ("Inference", a)), 3: (c, ("Conflict", a)), 4: (c, ("Rephrase", a))}.get(lab)
        if src:
            wanted.setdefault(src[0], []).append(src[1])
    types = {}
    for t, p, y in view(gold, s.name).anchors():
        types.setdefault((t, p), y)
    out = {}
    for qid, (kind, _, item, meaning) in b.meta.items():
        if kind == "pair":
            hit = [o for o, m in meaning.items() if m == label[item]]
        elif kind == "source":
            hit = [o for o, m in meaning.items() if m in wanted.get(item, [])] or [o for o, m in meaning.items() if m is None]
        else:
            y = types.get((s.turn[item[0]], s.prop[item[1]]))
            hit = [y if y in TURN_TYPES else "Asserting"]
        probs = {o: (1 / len(hit) if o in hit else 0.0) for o in b.questions[qid]["criteria"]}
        out[qid] = {"type": "choice", "choice": hit[0], "probabilities": probs}
    return out


def leaks(b, inp: dict) -> list[str]:
    ids = {n["nodeID"] for n in inp["nodes"] if len(str(n["nodeID"])) >= 4}
    text = json.dumps(b.requests, ensure_ascii=False)
    return sorted(i for i in ids if re.search(rf"(?<![\w.]){re.escape(str(i))}(?![\w.])", text))


def spread(xs):
    xs = sorted(xs)
    return {"p50": xs[len(xs) // 2], "p90": xs[int(0.9 * (len(xs) - 1))], "max": xs[-1], "sum": sum(xs)}


def size(b) -> dict:
    kinds = Counter(k for k, *_ in b.meta.values())
    state = Q.tokens(b.state)
    qtok = [Q.tokens(q) + Q.tokens(qid) for qid, q in b.questions.items()]
    src_opts = [len(q["criteria"]) for qid, q in b.questions.items() if b.meta[qid][0] == "source"]
    return {"pair_q": kinds["pair"], "source_q": kinds["source"], "type_q": kinds["type"], "max_options": max(src_opts, default=0),
            "state": state, "requests": len(b.requests),
            "per_request_billing": state * len(b.requests) + sum(qtok),     # state billed once per request
            "per_question_billing": state * len(qtok) + sum(qtok)}          # state billed once per question


def main():
    report = {}
    with timed("a4", "A4.1_check") as row:
        dev = list(dev_maps())
        res = {v: {"gen": [], "foc": []} for v in VIEWS}
        invalid, bad_answers, leaked = Counter(), [], {}
        sizes = {arm: {} for arm in ARMS}
        for name, gold, inp, links in dev:
            b = Q.build(inp, name, links)
            ans = gold_answers(b, gold)
            if problems(b, ans):
                bad_answers.append(name)
            if leaks(b, inp):
                leaked[name] = leaks(b, inp)[:5]
            for v in VIEWS:
                rows, scores = pair_scores(b, ans, v)
                m = build_map(inp, name, decode(rows, scores, TH), links, b.s.pos)
                if validate(m):
                    invalid[v] += 1
                ari = score_map(gold, m)["ari"]
                res[v]["gen"].append(100 * ari[0][2])
                if ari[1] is not None:
                    res[v]["foc"].append(100 * ari[1][2])
            sizes["short_k0"][("dev", name)] = size(b)
        report["round_trip"] = {"maps": len(dev), "expected_pair_view": [98.64, 94.44], "threshold": 0.3,
                                **{v: [round(st.mean(r["gen"]), 2), round(st.mean(r["foc"]), 2)] for v, r in res.items()},
                                "invalid_maps": dict(invalid), "answers_failing_check": bad_answers}
        report["node_id_leaks"] = leaked

        # Sizes for every arm: dev with the labeller's links, test likewise, the 50 pilot maps with gold links.
        from a2_llm.prompt import Pool
        pilot_cfg = resolve({"split": "train", "maps": "pilot50", "exclude": "pilot", "k": 0})
        pools = {"dev": Pool(), "pilot": Pool(set(pilot_cfg["exclude"]))}
        test_links = links_of("test")
        sets = {"dev": [(n, i, l) for n, _, i, l in dev],
                "test": [(n, load_map(p), test_links.get(n, [])) for n, p, _ in test_paths()],
                "pilot": targets(pilot_cfg)}
        for arm, (defs, k) in ARMS.items():
            for split, maps in sets.items():
                for name, inp, links in maps:
                    if (split, name) in sizes[arm]:
                        continue
                    b = Q.build(inp, name, links, defs=defs, k=k, pool=pools["dev" if split == "test" else split])
                    sizes[arm][(split, name)] = size(b)
        report["sizes"] = {arm: {split: {f: spread([d[f] for (sp, _), d in per.items() if sp == split])
                                         for f in next(iter(per.values()))}
                                 for split in sets} for arm, per in sizes.items()}

        # Projection: the plan's passes (report §8), at list price.
        tot = lambda arm, split, f: sum(d[f] for (sp, _), d in sizes[arm].items() if sp == split)
        proj = {}
        for f in ("per_request_billing", "per_question_billing"):
            passes = {"A4.2 pilot: 3 arms + 2 revisions (short, k0)": tot("short_k0", "pilot", f) * 3
                      + tot("full_k0", "pilot", f) + tot("short_k1", "pilot", f),
                      "A4.3 dev pass": tot("short_k0", "dev", f),
                      "A4.4 ablations (~3 dev passes)": 3 * tot("short_k0", "dev", f),
                      "A4.5 test": tot("short_k0", "test", f)}
            proj[f] = {**{k: {"tokens": v, "usd": round(v * PRICE, 3)} for k, v in passes.items()},
                       "total_usd": round(sum(passes.values()) * PRICE, 2)}
        report["projection"] = {"chars_per_token_assumed": Q.CHARS_PER_TOKEN, **proj}

        # A readable sample: the median dev map, the state and the first two questions of each kind.
        name, gold, inp, links = sorted(dev, key=lambda d: len(view(d[2], d[0]).props))[len(dev) // 2]
        b = Q.build(inp, name, links)
        picked, seen = {}, Counter()
        for qid, q in b.questions.items():
            kind = b.meta[qid][0]
            if seen[kind] < 2:
                picked[qid] = q
                seen[kind] += 1
        (OUT / "samples").mkdir(parents=True, exist_ok=True)
        with open(OUT / "samples" / f"request_{name}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump({"map": name, "requests": len(b.requests), "questions": len(b.questions),
                       "state": b.state, "questions_sample": picked}, f, ensure_ascii=False, indent=1)
        report["sample"] = name
        row.update(maps=len(dev), round_trip=report["round_trip"]["pair"])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "check.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
