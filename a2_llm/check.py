"""A2.1 gate checks, no model calls.

1. Lossless round trip: each dev map's gold relations, written as if the model had answered (all
   "high"), go through the local schema check -> parse -> A1's decision code -> assembly -> official
   scorer. It must give the no-pruning pipeline ceiling, 98.92 / 96.51 ARI (D041), with every map valid.
2. Readable inputs: three serialised dev maps with their gold answers in results/a2/samples/.
3. Prompt sizes: characters of system prompt + user message for every dev and test map at k = 0, 4, 8.

    python -m a2_llm.check      -> results/a2/check.json
"""
import json
import statistics as st

from common import RESULTS, ROOT
from common.aif import load_map, strip_to_input, test_paths, view
from common.score import score_map
from common.serialise import serialise
from common.split import load_split
from common.timing import timed
from common.validate import validate
from a1_cross_encoder.decide import decode
from a1_cross_encoder.predict import build_map
from a2_llm.parse import pair_scores
from a2_llm.prompt import GIANT, Pool, gold_answer, render, system_file, user_message
from a2_llm.schema import problems

OUT = RESULTS / "a2"
HALF = {"Inference": 0.5, "Conflict": 0.5, "Rephrase": 0.5}


def links_of(split: str) -> dict:
    return json.loads((ROOT / "cache" / "ilo" / f"links_{split}.json").read_text())


def dev_maps():
    """(name, gold, test-format input, labeller links) for the 225 dev maps A1 also uses."""
    links = links_of("dev")
    for p in load_split("dev"):
        gold = load_map(p)
        inp = strip_to_input(gold)
        if len(view(inp, p.stem).props) <= GIANT:
            yield p.stem, gold, inp, links.get(p.stem, [])


def spread(xs):
    xs = sorted(xs)
    return {"p50": xs[len(xs) // 2], "p90": xs[int(0.9 * (len(xs) - 1))], "max": xs[-1]}


def main():
    report = {}
    with timed("a2", "A2.1_check") as row:
        dev = list(dev_maps())
        gen, foc, invalid, bad, picks = [], [], [], [], {}
        for name, gold, inp, links in dev:
            s = serialise(inp, name, links)
            answer = gold_answer(gold, s)
            if problems(answer, s):
                bad.append(name)
            rows, scores = pair_scores(s, [answer])
            m = build_map(inp, name, decode(rows, scores, HALF), links, s.pos)
            if validate(m):
                invalid.append(name)
            ari = score_map(gold, m)["ari"]
            gen.append(100 * ari[0][2])
            if ari[1] is not None:
                foc.append(100 * ari[1][2])
            unplaced = len(s.prop) > len(s.pos)
            shared = len(s.pos.values()) > len(set(s.pos.values()))
            if unplaced and "unplaced" not in picks:
                picks["unplaced"] = (name, s, answer)
            elif shared and not unplaced and "two_per_turn" not in picks:
                picks["two_per_turn"] = (name, s, answer)
        sizes = sorted(dev, key=lambda d: len(view(d[2], d[0]).props))
        name, gold, inp, links = sizes[len(sizes) // 2]
        s = serialise(inp, name, links)
        picks["median"] = (name, s, gold_answer(gold, s))
        report["round_trip"] = {"maps": len(dev), "ari_general": round(st.mean(gen), 2), "ari_focused": round(st.mean(foc), 2),
                                "expected": [98.92, 96.51], "invalid_maps": invalid, "answers_failing_schema": bad}

        (OUT / "samples").mkdir(parents=True, exist_ok=True)
        for kind, (name, s, answer) in picks.items():
            with open(OUT / "samples" / f"{kind}_{name}.txt", "w", encoding="utf-8", newline="\n") as f:
                f.write(f"{render(s)}\n\nGold answer:\n{json.dumps(answer, indent=1, ensure_ascii=False)}\n")
        report["samples"] = {kind: v[0] for kind, v in picks.items()}

        pool = Pool()
        examples = {}
        system = len(system_file("v0").read_text(encoding="utf-8"))
        test_links = links_of("test")
        targets = [(n, i) for n, _, i, _ in dev] + [(n, load_map(p)) for n, p, _ in test_paths()]
        sizes = {}
        for k in (0, 4, 8):
            chars = {"dev": [], "test": []}
            for name, inp in targets:
                split = "test" if name.startswith("map") else "dev"
                links = test_links.get(name, []) if split == "test" else dict((d[0], d[3]) for d in dev)[name]
                s = serialise(inp, name, links)
                ex = [examples.setdefault(e, pool.example(e)) for e in pool.nearest(inp, name, k)]
                chars[split].append(system + len(user_message(s, ex)))
            sizes[f"k{k}"] = {split: spread(v) for split, v in chars.items()}
        report["prompt_chars"] = sizes
        report["pool_maps"] = len(pool.names)
        row.update(maps=len(dev), round_trip=report["round_trip"]["ari_general"])
    with open(OUT / "check.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
