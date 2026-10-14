"""Gates before any Jev call (D099), and the cost projection.

1. Gold round trip: each dev map's gold answer, written as one-hot Jev answers to every question (both orders) ->
   problems() empty -> rows_probs -> decode_map(link_min=0.5) -> official scorer. It must equal Claude's round
   trip (98.70 / 96.28, results/ilo_llm/check.json): the same one-to-one decoder, so the same ceiling.
2. Requests build: every dev and pilot map at k = 0 and k = 4 packs into requests within A4's budgets, and no node
   ID appears in any request (D029; a4_jev.check.leaks).
3. Cost projection at A4's measured dev rate (dollars per billed token) and characters per token: smoke + both pilot
   arms + dev at the costlier k. Gate: at most $2 (D099).

    python -m ilo_jev.check      -> results/ilo_jev/check.json
"""
import json

import numpy as np

from common import RESULTS
from common.aif import load_map, strip_to_input
from common.score import score_map
from common.split import load_split
from common.timing import timed
from a4_jev.check import leaks
from ilo.decode import decode_map, predicted_map
from ilo_jev.parse import problems, rows_probs
from ilo_jev.questions import build
from ilo_jev.runner import OUT, RUNS, builds, config
from ilo_llm.prompt import gold_answer
from ilo_llm.runner import targets
from ilo_llm.schema import NONE

CAP = 2.0


def one_hot(b, gold_aif: dict) -> dict:
    """Gold, as Jev answers: every question's gold option with probability 1."""
    ans = {it["turn"]: it for it in gold_answer(gold_aif, b.s)["illocutions"]}
    out = {}
    for qid, (kind, _, t, meaning) in b.meta.items():
        want = (ans[t]["proposition"] if ans[t]["proposition"] != NONE else None) if kind == "link" else ans[t]["type"]
        opts = list(b.questions[qid]["criteria"])
        pick = next(o for o in opts if meaning[o] == want)
        out[qid] = {"type": "choice", "choice": pick, "probabilities": {o: float(o == pick) for o in opts}}
    return out


def round_trip() -> dict:
    claude = json.loads((RESULTS / "ilo_llm" / "check.json").read_text())["1_gold_round_trip"]
    paths = {p.stem: p for p in load_split("dev")}
    gen, foc, bad = [], [], []
    for name in config("dev")["maps"]:
        gold = load_map(paths[name])
        b = build(strip_to_input(gold), name, k=0)
        ans = one_hot(b, gold)
        if problems(b, ans):
            bad.append(name)
        rows, probs = rows_probs(b, ans)
        g, f = score_map(gold, predicted_map(gold, decode_map(rows, probs, 0.5)))["ilo"]
        gen.append(g[2])
        if f is not None:
            foc.append(f[2])
    got = (round(100 * float(np.mean(gen)), 2), round(100 * float(np.mean(foc)), 2))
    return {"maps": len(gen), "ilo_general": got[0], "ilo_focused": got[1], "claude_round_trip": [claude["ilo_general"], claude["ilo_focused"]],
            "answers_failing_check": bad, "passed": not bad and list(got) == [claude["ilo_general"], claude["ilo_focused"]]}


def requests() -> tuple[dict, dict]:
    a4 = json.loads((RESULTS / "a4" / "dev" / "summary.json").read_text())
    usd_per_token = a4["usd"] / a4["input_tokens"]
    chars_per_token = a4["per_request"]["chars_per_token"]
    sizes, leaked, chars = {}, {}, {}
    for run in ("smoke", "pilot-k0", "pilot-k4", "dev"):
        for k in ((0, 4) if run == "dev" else (RUNS[run]["k"],)):
            bs = builds(run, k)
            inputs = dict(targets(config(run)))
            for name, b in bs.items():
                if leaks(b, inputs[name]):
                    leaked.setdefault(f"{run}/k{k}", []).append(name)
            reqs = [r for b in bs.values() for r in b.requests]
            c = [len(json.dumps(r, ensure_ascii=False)) for r in reqs]
            chars[f"{run}/k{k}"] = sum(c)
            sizes[f"{run}/k{k}"] = {"maps": len(bs), "requests": len(reqs), "questions": sum(len(r["questions"]) for r in reqs),
                                    "max_requests_per_map": max(len(b.requests) for b in bs.values()),
                                    "chars_max_request": max(c)}
    usd = {key: round(n / chars_per_token * usd_per_token, 4) for key, n in chars.items()}
    plan = usd["smoke/k0"] + usd["pilot-k0/k0"] + usd["pilot-k4/k4"] + max(usd["dev/k0"], usd["dev/k4"])
    check2 = {"runs": sizes, "maps_with_node_ids": leaked, "passed": not leaked}
    check3 = {"a4_usd_per_million_tokens": round(1e6 * usd_per_token, 4), "a4_chars_per_token": chars_per_token,
              "projected_usd": usd, "plan_usd (smoke + pilot arms + dev at the costlier k)": round(plan, 4), "cap_usd": CAP,
              "passed": plan <= CAP}
    return check2, check3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with timed("ilo_jev", "check") as row:
        report = {"1_gold_round_trip": round_trip()}
        report["2_requests_build"], report["3_cost_projection"] = requests()
        report["all_passed"] = all(v["passed"] for v in report.values() if isinstance(v, dict))
        row["all_passed"] = report["all_passed"]
    with open(OUT / "check.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
