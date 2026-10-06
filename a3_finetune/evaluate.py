"""A3 scoring of generated answers.

    python -m a3_finetune.evaluate --overfit --run=overfit --adapter=ep10.0
        A3.2 gate: greedy answers on the run's own training maps against their gold answers
        (relations exact: type, from, to; turn types; whole answers identical).
    python -m a3_finetune.evaluate --greedy-dev --run=v1 [--adapters=ep0.5,ep1.0]
        A3.3: one greedy answer per dev map for every saved adapter, scored like A2.1's round trip
        (every listed relation kept -> A1's decision code -> build_map -> official scorer); the checkpoint rule
        picks the best ARI general, ties to the earlier adapter (D065).
    python -m a3_finetune.evaluate --diversity --run=v1 [--adapter=ep2.0] [--maps=20]
        A3.3 diversity rule (D065): 8 samples on the first 20 dev maps with the chosen adapter. If fewer than
        10% of the listed relations get a vote share strictly between 0 and 1, the votes are all-or-nothing
        and token probability becomes the main score for A3.4.
"""
from collections import Counter
import json
import statistics as st
import sys

from common import RESULTS
from common.score import score_map
from common.validate import validate
from a1_cross_encoder.decide import decode
from a1_cross_encoder.predict import build_map
from a2_llm.check import HALF, dev_maps
from a3_finetune.parse import pair_scores
from a3_finetune.runner import Runner, arg, examples, export
from a3_finetune.train import CKPT, OUT


def rel_set(answer: dict | None) -> set:
    return {(r["type"], r["from"], r["to"]) for r in (answer or {}).get("relations", [])}


def ilo_map(answer: dict | None) -> dict:
    return {i["link"]: i["type"] for i in (answer or {}).get("illocutions", [])}


def overfit(run: str, adapter: str) -> dict:
    cfg = json.loads((OUT / run / "config.json").read_text())
    r = Runner(run, adapter, cfg["model"])
    exs = examples(run, "train")
    recs = r.greedy(exs)
    tp = fp = fn = same = ilo_ok = ilo_n = valid = 0
    for ex in exs:
        pred = recs[ex.name]["outs"][0]["answer"]
        gold = json.loads(ex.answer)
        valid += recs[ex.name]["outs"][0]["valid"]
        p, g = rel_set(pred), rel_set(gold)
        tp, fp, fn = tp + len(p & g), fp + len(p - g), fn + len(g - p)
        same += pred == gold
        gi, pi = ilo_map(gold), ilo_map(pred)
        ilo_n += len(gi)
        ilo_ok += sum(pi.get(k) == v for k, v in gi.items())
    res = {"run": run, "adapter": adapter, "maps": len(exs), "valid": valid, "identical_answers": same,
           "relations_gold": tp + fn, "relation_recall": round(tp / (tp + fn), 4),
           "relation_precision": round(tp / (tp + fp), 4) if tp + fp else None,
           "turn_types_right": round(ilo_ok / ilo_n, 4), "gate_recall_0.95": tp / (tp + fn) >= 0.95}
    export(run, f"{adapter}_train_greedy", recs)
    return res


def greedy_ari(run: str, adapter: str, model_key: str, gold: dict) -> dict:
    r = Runner(run, adapter, model_key)
    exs = examples(run, "dev")
    recs = r.greedy(exs)
    gen, foc, invalid, n_rel, unfinished = [], [], 0, 0, 0
    for ex in exs:
        out = recs[ex.name]["outs"][0]
        answer = out["answer"] if out["valid"] else {"illocutions": [], "relations": []}
        unfinished += not out["finished"]
        n_rel += len(answer["relations"])
        rows, scores = pair_scores(ex.s, [answer])
        m = build_map(ex.inp, ex.name, decode(rows, scores, HALF), ex.links, ex.s.pos)
        invalid += bool(validate(m))
        ari = score_map(gold[ex.name], m)["ari"]
        gen.append(100 * ari[0][2])
        if ari[1] is not None:
            foc.append(100 * ari[1][2])
    export(run, f"{adapter}_dev_greedy", recs)
    r.close()
    return {"adapter": adapter, "ari_general": round(st.mean(gen), 2), "ari_focused": round(st.mean(foc), 2),
            "relations_per_map": round(n_rel / len(exs), 2), "invalid_maps": invalid, "unfinished": unfinished}


def diversity(run: str, adapter: str, model_key: str, maps: int, n: int = 8) -> dict:
    r = Runner(run, adapter, model_key)
    exs = examples(run, "dev")[:maps]
    recs = r.samples(exs, n, seed=0)
    shares, valid, total = [], 0, 0
    for ex in exs:
        outs = recs[ex.name]["outs"]
        valid += sum(o["valid"] for o in outs)
        total += len(outs)
        votes = Counter(rel for o in outs for rel in rel_set(o["answer"]))
        shares += [c / len(outs) for c in votes.values()]
    export(run, f"{adapter}_dev{maps}_samples{n}", recs)
    r.close()
    mid = sum(0 < s < 1 for s in shares) / len(shares) if shares else 0.0
    hist = Counter(round(s * n) for s in shares)
    return {"adapter": adapter, "maps": maps, "samples": n, "valid": f"{valid}/{total}",
            "listed_relations": len(shares), "share_strictly_between_0_and_1": round(mid, 4),
            "votes_histogram": {f"{k}/{n}": hist[k] for k in range(1, n + 1)},
            "main_score": "votes" if mid >= 0.10 else "token_probability"}


def main():
    run = arg("run")
    if "--overfit" in sys.argv:
        res = overfit(run, arg("adapter"))
        (OUT / run / "overfit_report.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(res, indent=1))
    elif "--greedy-dev" in sys.argv:
        cfg = json.loads((OUT / run / "config.json").read_text())
        adapters = arg("adapters")
        adapters = adapters.split(",") if adapters else sorted((p.name for p in (CKPT / run).iterdir() if p.is_dir()),
                                                               key=lambda a: float(a[2:]))
        gold = {name: g for name, g, _, _ in dev_maps()}
        path = OUT / run / "greedy_dev.json"
        rows = json.loads(path.read_text())["adapters"] if path.exists() else []
        done = {x["adapter"] for x in rows}
        for a in adapters:
            if a not in done:
                rows.append(greedy_ari(run, a, cfg["model"], gold))
                print(rows[-1], flush=True)
        rows.sort(key=lambda x: float(x["adapter"][2:]))
        best = max(rows, key=lambda x: (x["ari_general"], -float(x["adapter"][2:])))
        path.write_text(json.dumps({"adapters": rows, "chosen": best["adapter"]}, indent=1) + "\n",
                        encoding="utf-8", newline="\n")
        print("chosen:", best)
    elif "--diversity" in sys.argv:
        cfg = json.loads((OUT / run / "config.json").read_text())
        adapter = arg("adapter") or json.loads((OUT / run / "greedy_dev.json").read_text())["chosen"]
        res = diversity(run, adapter, cfg["model"], int(arg("maps", 20)))
        (OUT / run / "diversity.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
