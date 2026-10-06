"""A4.1 live measurements, from the saved answers of runs `probe` (5 training maps) and `probe-repeat` (no new calls).

- billing: input tokens per request against the request's characters, and against the state billed per question;
- speed: seconds per request and per map;
- option-order bias: forward vs reversed answers to the same question (choice agreement, total-variation distance,
  the probability the first-listed option gains). Rule fixed before the probe (D076): later runs ask one order
  (forward) if choices agree on >= 95% of questions and the mean distance is <= 0.05; otherwise both;
- repeatability: the same request sent twice;
- a sanity score on the 5 maps (gold links): ARI at matched density (one threshold, so as many relations as gold)
  per view and order. Five maps only show whether the answers carry signal; nothing is chosen from it;
- the cost projection for the remaining passes (pilot 50, dev 225, test 11) from the measured characters per token.

    python -m a4_jev.probe      -> results/a4/probe.json
"""
import json
import statistics as st

import numpy as np

from common import RESULTS
from common.aif import load_map, test_paths
from common.score import ari_labels, score_map
from common.split import load_split
from a1_cross_encoder.decide import decode
from a1_cross_encoder.predict import build_map
from a2_llm.check import dev_maps, links_of
from a2_llm.runner import resolve, targets
from a4_jev import client
from a4_jev import questions as Q
from a4_jev.parse import VIEWS, pair_scores
from a4_jev.runner import answers, builds, config, jobs, load

OUT = RESULTS / "a4"
PRICE = 0.042 / 1e6


def order_bias(b, ans: dict) -> list[dict]:
    """Per question stem: forward vs reversed probabilities, by meaning."""
    out = []
    for qid, (kind, o, item, meaning) in b.meta.items():
        if o != "f" or qid[:-2] + "_r" not in ans or qid not in ans:
            continue
        rq = qid[:-2] + "_r"
        pf = {meaning[k]: v for k, v in ans[qid]["probabilities"].items()}
        pr = {b.meta[rq][3][k]: v for k, v in ans[rq]["probabilities"].items()}
        first_f = meaning[next(iter(b.questions[qid]["criteria"]))]       # listed first going forward, last reversed
        first_r = b.meta[rq][3][next(iter(b.questions[rq]["criteria"]))]
        out.append({"kind": kind, "agree": max(pf, key=pf.get) == max(pr, key=pr.get),
                    "tvd": 0.5 * sum(abs(pf[m] - pr.get(m, 0)) for m in pf),
                    "first_gain": ((pf[first_f] - pr[first_f]) + (pr[first_r] - pf[first_r])) / 2})
    return out


def matched(names, bs, ans, maps, view, orders):
    """Mean per-map ARI (general, focused) at one threshold that predicts as many relations as gold, pooled.
    maps: name -> (gold, input, links)."""
    per = {n: pair_scores(bs[n], ans[n], view, orders) for n in names}
    gold_count = sum(len(ari_labels(maps[n][0], set(bs[n].s.prop.values()))) for n in names)
    best = None
    for t in np.arange(0.02, 0.98, 0.01):
        th = {"Inference": t, "Conflict": t, "Rephrase": t}
        count = sum(len(decode(*per[n], th)) for n in names)
        if best is None or abs(count - gold_count) < abs(best[1] - gold_count):
            best = (th, count)
    th = best[0]
    gen, foc = [], []
    for n in names:
        gold, inp, links = maps[n]
        m = build_map(inp, n, decode(*per[n], th), links, bs[n].s.pos)
        ari = score_map(gold, m)["ari"]
        gen.append(100 * ari[0][2])
        if ari[1] is not None:
            foc.append(100 * ari[1][2])
    return {"threshold": round(float(th["Inference"]), 2), "relations": best[1], "gold": gold_count,
            "ari": [round(st.mean(gen), 2), round(st.mean(foc), 2) if foc else None]}


def main():
    rep = {}
    recs = [load(j["key"]) for j in jobs("probe")]
    reqs = [j["request"] for j in jobs("probe")]
    toks = [client.input_tokens(r["result"]["response"]) for r in recs]
    state_tok = [len(json.dumps(q["state"], ensure_ascii=False)) for q in reqs]
    ratio = sum(r["chars"] for r in recs) / sum(toks)
    per_question = [s / ratio * len(q["questions"]) for s, q in zip(state_tok, reqs)]
    rep["billing"] = {"requests": len(recs), "input_tokens": sum(toks), "chars_per_token": round(ratio, 3),
                      "tokens_if_state_billed_per_question": int(sum(per_question)),
                      "verdict": "state billed once per request" if sum(toks) < 0.5 * sum(per_question) else "state billed per question",
                      "usd": round(sum(client.cost(r["result"]["response"]) for r in recs), 6),
                      "models": sorted({r["result"]["response"].get("model") for r in recs})}
    secs = [r["result"]["seconds"] for r in recs]
    by_map = {}
    for r in recs:
        by_map[r["map"]] = by_map.get(r["map"], 0) + r["result"]["seconds"]
    rep["speed"] = {"request_seconds": {"p50": st.median(secs), "max": max(secs)},
                    "map_seconds_sequential": {m: round(v, 2) for m, v in by_map.items()},
                    "map_seconds_p50": round(st.median(by_map.values()), 2)}

    bs, ans = builds("probe"), answers("probe")
    rows = [x for n in bs for x in order_bias(bs[n], ans[n][0])]
    agree, tvd = st.mean(x["agree"] for x in rows), st.mean(x["tvd"] for x in rows)
    one = agree >= 0.95 and tvd <= 0.05
    rep["order_bias"] = {"questions": len(rows), "choice_agreement": round(agree, 4), "mean_tvd": round(tvd, 4),
                         "first_listed_gain": round(st.mean(x["first_gain"] for x in rows), 4),
                         **{f"{k}_agreement": round(st.mean(x["agree"] for x in rows if x["kind"] == k), 4) for k in ("pair", "source")},
                         "rule": "one order if agreement >= 0.95 and mean TVD <= 0.05",
                         "decision": "one order (forward)" if one else "both orders"}

    rb, ra = builds("probe-repeat"), answers("probe-repeat")
    name = next(iter(ra))
    a0, a1 = ra[name][0], ra[name][1]
    diffs = [abs(a0[q]["probabilities"][o] - a1[q]["probabilities"][o]) for q in a0 for o in a0[q]["probabilities"]]
    rep["repeatability"] = {"map": name, "questions": len(a0), "max_abs_difference": max(diffs),
                            "identical_choices": sum(a0[q]["choice"] == a1[q]["choice"] for q in a0) / len(a0)}

    cfg = config("probe")
    paths = {p.stem: p for p in load_split("train")}
    maps = {n: (load_map(paths[n]), inp, links) for n, inp, links in targets(cfg)}
    names = list(bs)
    flat = {n: ans[n][0] for n in names}
    rep["sanity_matched_density_5_maps"] = {f"{v}_{''.join(o)}": matched(names, bs, flat, maps, v, o)
                                            for v in VIEWS for o in (("f",), ("r",), ("f", "r"))}

    # Projection for the remaining passes, at the measured characters per token.
    pilot = resolve({"split": "train", "maps": "pilot50", "exclude": "pilot", "k": 0})
    test_links = links_of("test")
    sets = {"pilot": targets(pilot), "dev": [(n, i, l) for n, _, i, l in dev_maps()],
            "test": [(n, load_map(p), test_links.get(n, [])) for n, p, _ in test_paths()]}
    proj = {}
    for label, orders in (("both_orders", ("f", "r")), ("one_order", ("f",))):
        tot = {}
        for split, maps in sets.items():
            chars = sum(len(json.dumps(r, ensure_ascii=False)) for n, i, l in maps
                        for r in Q.build(i, n, l, orders=orders).requests)
            tot[split] = {"tokens": int(chars / ratio), "usd": round(chars / ratio * PRICE, 3)}
        proj[label] = {**tot, "total_usd": round(sum(v["usd"] for v in tot.values()), 3)}
    rep["projection"] = proj
    with open(OUT / "probe.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
