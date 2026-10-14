"""Step 4: score the pilot runs (20 training maps, 1 answer each) and choose the effort (D090, D094). No calls.

Argmax decoding only: ilo.decode.decode_map with link_min = 0.5 and no type thresholds. Not comparable
with the labeller, which was trained on these maps. The effort rule (D094): `xhigh` only if it beats
`medium` by at least 5 points on ILO general or focused; otherwise `medium`.

    python -m ilo_llm.pilot                                   -> results/ilo_llm/pilot.json
    python -m ilo_llm.pilot --runs=pilot-medium,pilot-xhigh,pilot-medium-v1
    python -m ilo_llm.pilot --errors=pilot-medium             -> results/ilo_llm/pilot_errors.md
"""
import json
from collections import Counter

from common.aif import view
from common.score import score_map
from common.timing import timed
from ilo.decode import predicted_map
from ilo.evaluate import per_type
from ilo.tune import ilo_f1, mean_f1
from ilo_llm.runner import OUT, arg
from ilo_llm.schema import NONE
from ilo_llm.tune import ClaudeDev, run_answers

TIE = 5.0


def gold_turn_links(dev: ClaudeDev, name: str) -> set:
    v = view(dev.gold_aif[name], name)
    return {(t, p) for t, p, _ in v.anchors() if t in v.turn}


def score_run(run: str) -> dict:
    maps, gold, missing = run_answers(run)
    dev = ClaudeDev(maps, gold)
    links = {n: dev.links(dev.raw, n, 0.5, None) for n in dev.names}
    scores = {}
    for n in dev.names:
        fast = ilo_f1(dev.gold[n], {(t, p): y for t, p, y in links[n]}, dev.n_pairs[n])
        official = score_map(dev.gold_aif[n], predicted_map(dev.gold_aif[n], links[n]))["ilo"]
        for o, f in zip(official, fast):
            assert (o is None and f is None) or all(abs(a - b) < 1e-9 for a, b in zip(o, f)), (n, o, f)
        scores[n] = fast
    hit = n_gold = n_pred = 0
    trap = 0
    for n in dev.names:
        g, p = gold_turn_links(dev, n), {(t, q) for t, q, _ in links[n]}
        hit, n_gold, n_pred = hit + len(g & p), n_gold + len(g), n_pred + len(p)
        trap += bool({y for *_, y in links[n]} - set(dev.gold[n].values()))
    summ = json.loads((OUT / run / "summary.json").read_text())
    pa = summ["per_answer"]
    raw_types = Counter(it["type"] for _, answers in maps.values() for a in answers
                        for it in a["illocutions"] if it["proposition"] != NONE)
    return {"maps": len(dev.names), "maps_without_answers": missing,
            "ilo_argmax": {"general": round(mean_f1(scores, 0), 2), "focused": round(mean_f1(scores, 1), 2)},
            "linking": {"recall": round(hit / n_gold, 4), "precision": round(hit / max(n_pred, 1), 4),
                        "gold": n_gold, "predicted": n_pred},
            "per_type": per_type(dev, links), "types_as_answered": dict(raw_types.most_common()),
            "maps_with_a_type_absent_from_gold": trap,
            "per_call": {"thinking_tokens_p50": pa["thinking_tokens"]["p50"], "thinking_tokens_max": pa["thinking_tokens"]["max"],
                         "seconds_p50": pa["seconds"]["p50"], "seconds_max": pa["seconds"]["max"],
                         "api_equivalent_usd": round(summ["api_equivalent_usd"] / max(summ["calls_made"], 1), 4)},
            "calls_made": summ["calls_made"], "api_equivalent_usd_total": summ["api_equivalent_usd"],
            "per_map": {n: [round(100 * scores[n][0][2], 1), round(100 * scores[n][1][2], 1) if scores[n][1] else None]
                        for n in dev.names}}


def errors(run: str) -> str:
    """Every disagreement with gold on the pilot, turn by turn: missed, wrong or extra links, and wrong types."""
    maps, gold, _ = run_answers(run)
    dev = ClaudeDev(maps, gold)
    kinds = Counter()
    out = []
    for n in dev.names:
        s = maps[n][0]
        v = view(dev.gold_aif[n], n)
        g = {}
        for t, p, y in v.anchors():
            if t in v.turn:
                g.setdefault(t, (p, y))
        pred = {t: (p, y) for t, p, y in dev.links(dev.raw, n, 0.5, None)}
        turn_id = {lid: tid for tid, lid in s.turn.items()}
        lines = []
        for lid in v.dialogue:
            gp, pp = g.get(lid), pred.get(lid)
            if gp == pp:
                continue
            if gp and pp and gp[0] == pp[0]:
                kind = f"type {gp[1]} -> {pp[1]}"
            elif gp and pp:
                kind = "wrong proposition"
            elif gp:
                kind = "missed link"
            else:
                kind = "extra link"
            kinds[kind.split(" ->")[0] if kind.startswith("type") else kind] += 1
            say = lambda x: f"{s.pid[x[0]]} [{x[1]}] {v.props[x[0]]}" if x else "none"
            lines.append(f"- **{turn_id[lid]}** {v.locs[lid].speaker}: {v.locs[lid].body}\n"
                         f"  - {kind}. Gold: {say(gp)}\n  - Claude: {say(pp)}")
        if lines:
            out.append(f"### {n}\n" + "\n".join(lines))
    head = [f"# Pilot errors: {run}", "",
            "Argmax decoding on the 20 pilot training maps (D090 step 4); one entry per turn where Claude and gold differ "
            "(gold = the turn's first link). Generated by `python -m ilo_llm.pilot --errors=" + run + "`.", "",
            "| Kind | Turns |", "|---|---|"] + [f"| {k} | {c} |" for k, c in kinds.most_common()] + [""]
    return "\n".join(head + out) + "\n"


def main():
    err = arg("errors")
    if err:
        text = errors(err)
        (OUT / "pilot_errors.md").write_text(text, encoding="utf-8", newline="\n")
        print(text[:3000])
        return
    runs = arg("runs", "pilot-medium,pilot-xhigh").split(",")
    with timed("ilo_llm", "pilot_score", runs=runs) as row:
        rep = {"decode": "argmax (link_min 0.5, no type thresholds)", "runs": {r: score_run(r) for r in runs}}
        m, x = rep["runs"].get("pilot-medium"), rep["runs"].get("pilot-xhigh")
        if m and x:
            gain = {k: round(x["ilo_argmax"][k] - m["ilo_argmax"][k], 2) for k in ("general", "focused")}
            rep["effort_choice"] = {"rule": "xhigh only if it beats medium by >= 5 points on general or focused (D094)",
                                    "xhigh_minus_medium": gain,
                                    "chosen": "xhigh" if max(gain.values()) >= TIE else "medium"}
        rep["sanity_gate"] = {"rule": "pilot ILO general >= 80",
                              "passed": {r: v["ilo_argmax"]["general"] >= 80 for r, v in rep["runs"].items()}}
        row.update({r: v["ilo_argmax"] for r, v in rep["runs"].items()})
    with open(OUT / "pilot.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rep, indent=1))
    print(json.dumps({k: v if k != "runs" else {r: {kk: vv for kk, vv in s.items() if kk != "per_map"} for r, s in v.items()}
                      for k, v in rep.items()}, indent=1))


if __name__ == "__main__":
    main()
