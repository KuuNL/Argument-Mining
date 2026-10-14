"""Step 2 gates of the brief (plans/claude-ilo-dev-test/README.md), no model calls.

1. Gold round trip: each dev map's gold answer -> problems() empty -> votes -> rows_probs ->
   decode_map(link_min=0.5) -> official scorer. Mean ILO must reach the one-to-one decoder ceiling of the
   03 report (98.63 / 94.73, gate 98.6 / 94.7), and every map must be valid (answer and predicted map).
2. Prompts build: every dev and pilot map gives a non-empty user message, and a schema whose enums cover
   exactly its turns and propositions.
3. Prompt sizes: system prompt + user message, next to A2's dev prompts at k = 4 (results/a2/check.json);
   none may exceed 1.2x A2's largest.
4. Reference reproduced: the labeller's honest per-map dev scores (run v1, shared groups) recomputed with
   ilo.tune; they must average to 90.08 / 80.64. Saved for step 6's bootstrap.

    python -m ilo_llm.check      -> results/ilo_llm/check.json, results/ilo_llm/labeller_per_map.json
"""
import json
import statistics as st

import numpy as np

from common.aif import load_map, strip_to_input
from common.score import score_map
from common.serialise import serialise
from common.split import load_split
from common.timing import timed
from common.validate import validate
from ilo.decode import decode_map, predicted_map
from ilo.train import dirs
from ilo.tune import GROUPINGS, Dev, fit_temperature, mean_f1
from ilo_llm.parse import rows_probs, votes
from ilo_llm.prompt import gold_answer
from ilo_llm.runner import OUT, RESULTS, RUNS, jobs, resolve
from ilo_llm.schema import NONE, problems
from ilo_llm.tune import cross_fit

CEILING = (98.6, 94.7)
LABELLER = {"general": 90.08, "focused": 80.64}
SIZE_FACTOR = 1.2


def round_trip() -> dict:
    gen, foc, invalid, bad = [], [], [], []
    paths = {p.stem: p for p in load_split("dev")}
    for name in resolve(RUNS["dev"])["maps"]:
        gold = load_map(paths[name])
        s = serialise(strip_to_input(gold), name, [])
        ans = gold_answer(gold, s)
        if problems(ans, s):
            bad.append(name)
        rows, probs = rows_probs(s, votes(s, [ans]))
        pred = predicted_map(gold, decode_map(rows, probs, 0.5))
        if validate(pred):
            invalid.append(name)
        g, f = score_map(gold, pred)["ilo"]
        gen.append(g[2])
        if f is not None:
            foc.append(f[2])
    out = {"maps": len(gen), "ilo_general": round(100 * float(np.mean(gen)), 2),
           "ilo_focused": round(100 * float(np.mean(foc)), 2), "gate": list(CEILING),
           "answers_failing_schema": bad, "invalid_maps": invalid}
    out["passed"] = out["ilo_general"] >= CEILING[0] and out["ilo_focused"] >= CEILING[1] and not bad and not invalid
    return out


def prompts() -> tuple[dict, dict]:
    """Checks 2 and 3 over the dev run (with v0 and a placeholder effort: no key is used) and the pilot."""
    builds, sizes, failures = {}, {}, []
    cfgs = {"dev": {**RUNS["dev"], "samples": 1, "effort": "medium", "prompt": "v0"}, "pilot": {**RUNS["pilot-medium"]}}
    for split, cfg in cfgs.items():
        chars = []
        for j in jobs(cfg):
            s, sch = j["s"], j["schema"]["properties"]["illocutions"]["items"]["properties"]
            ok = (j["user"].strip() and set(sch["turn"]["enum"]) == set(s.turn)
                  and set(sch["proposition"]["enum"]) == set(s.prop) | {NONE} and len(j["examples"]) == cfg["k"])
            if not ok:
                failures.append(j["map"])
            chars.append(j["chars"])
        builds[split] = len(chars)
        sizes[split] = {"min": min(chars), "p50": int(st.median(chars)), "max": max(chars)}
    a2 = json.loads((RESULTS / "a2" / "check.json").read_text())["prompt_chars"]["k4"]["dev"]
    limit = int(SIZE_FACTOR * a2["max"])
    check2 = {"maps": builds, "failures": failures, "passed": not failures and builds["dev"] == 225 and builds["pilot"] == 20}
    check3 = {"chars": sizes, "a2_dev_k4": a2, "limit": limit,
              "passed": all(v["max"] <= limit for v in sizes.values())}
    return check2, check3


def reference() -> dict:
    """The labeller's honest dev score, recomputed as ilo.tune.main() does (temperature on all dev, then
    thresholds cross-fitted); per-map scores saved for the paired bootstrap of step 6."""
    ckpt, _ = dirs("v1")
    dev = Dev(ckpt / "dev_probs.npy")
    t = fit_temperature(dev)
    held, _ = cross_fit(dev, dev.calibrated(t), GROUPINGS["shared"])
    got = {"general": round(mean_f1(held, 0), 2), "focused": round(mean_f1(held, 1), 2)}
    with open(OUT / "labeller_per_map.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"run": "v1", "groups": "shared", "temperature": t, "mean": got,
                            "per_map_cross_fitted": {n: held[n] for n in dev.names}}, indent=1))
    return {"run": "v1", "maps": len(held), "temperature": t, "got": got, "expected": LABELLER, "passed": got == LABELLER}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with timed("ilo_llm", "check") as row:
        report = {"1_gold_round_trip": round_trip()}
        report["2_prompts_build"], report["3_prompt_sizes"] = prompts()
        report["4_labeller_reference"] = reference()
        report["all_passed"] = all(v["passed"] for v in report.values() if isinstance(v, dict))
        row["all_passed"] = report["all_passed"]
    with open(OUT / "check.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
