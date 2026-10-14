"""Pilot (D099): k = 0 vs k = 4 worked examples on the 20 pilot training maps, argmax decoding. No calls.

Argmax: ilo.decode.decode_map with link_min = 0.5 and no type thresholds, as Claude's pilot (ilo_llm/pilot.py).
Rule (fixed before the runs): keep k = 4 only if it gains >= 1.0 ILO general; otherwise k = 0. Sanity gate: >= 80.
Diagnostic added after the gate failed (D100): the same argmax with the six rare types switched off.

    python -m ilo_jev.pilot      -> results/ilo_jev/pilot.json
"""
import json

from common.timing import timed
from ilo.evaluate import per_type
from ilo.tune import RARE, mean_f1
from ilo_jev.runner import OUT
from ilo_jev.tune import load_run
from ilo_llm.evaluate import link_stats, trap

GAIN = 1.0


def score_run(run: str) -> dict:
    dev, missing = load_run(run)
    links = {n: dev.links(dev.raw, n, 0.5, None) for n in dev.names}
    scores = dev.scores(dev.raw, 0.5, None)
    off = dev.scores(dev.raw, 0.5, dict.fromkeys(RARE, 1.01))     # diagnostic (D100): the six rare types switched off
    s = json.loads((OUT / run / "summary.json").read_text())
    return {"maps": len(dev.names), "maps_without_answers": missing,
            "ilo_argmax": {"general": round(mean_f1(scores, 0), 2), "focused": round(mean_f1(scores, 1), 2)},
            "ilo_argmax_rare_types_off (diagnostic, D100)": {"general": round(mean_f1(off, 0), 2), "focused": round(mean_f1(off, 1), 2)},
            "linking": link_stats(dev, links), "per_type": per_type(dev, links), "rare_label_trap": trap(dev, links),
            "usd": s["usd"], "requests": s["requests"], "seconds_per_request_p50": s["per_request"]["seconds_p50"],
            "per_map": {n: [round(100 * v[0][2], 1), round(100 * v[1][2], 1) if v[1] else None] for n, v in scores.items()}}


def main():
    with timed("ilo_jev", "pilot_score") as row:
        rep = {"decode": "argmax (link_min 0.5, no type thresholds)", "runs": {r: score_run(r) for r in ("pilot-k0", "pilot-k4")}}
        g0, g4 = (rep["runs"][r]["ilo_argmax"]["general"] for r in ("pilot-k0", "pilot-k4"))
        rep["k_choice"] = {"rule": "k = 4 only if it gains >= 1.0 ILO general on the pilot (D099)",
                           "k4_minus_k0_general": round(g4 - g0, 2), "chosen_k": 4 if g4 - g0 >= GAIN else 0}
        rep["sanity_gate"] = {"rule": "pilot ILO general >= 80",
                              "passed": {r: v["ilo_argmax"]["general"] >= 80 for r, v in rep["runs"].items()},
                              "diagnostic (D100): passed with the six rare types off":
                                  {r: v["ilo_argmax_rare_types_off (diagnostic, D100)"]["general"] >= 80 for r, v in rep["runs"].items()}}
        row.update({r: v["ilo_argmax"] for r, v in rep["runs"].items()})
    with open(OUT / "pilot.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rep, indent=1))
    print(json.dumps({k: v if k != "runs" else {r: {kk: vv for kk, vv in x.items() if kk not in ("per_map", "per_type")}
                                                for r, x in v.items()} for k, v in rep.items()}, indent=1))


if __name__ == "__main__":
    main()
