"""Step 6: Claude vs the trained labeller on dev, Task B end to end (D090). No calls.

Reads results/ilo_llm/<run>/thresholds.json (ilo_llm.tune), results/ilo_llm/labeller_per_map.json (check.py)
and results/ilo/v1/. Links for the analyses (linking, per type, rare-label trap) are each system's honest
links: every map decoded with the thresholds of the fold that held its episode out, as in the honest score.

    python -m ilo_llm.evaluate --run=dev      -> results/ilo_llm/<run>/dev_report.json
Runs in ~/.venvs/a3: a3_finetune.tune (bootstrap) imports xgrammar.
"""
import csv
import json
from collections import Counter

from common import RESULTS
from common.aif import view
from common.split import episodes, map_speakers
from common.timing import timed
from a3_finetune.tune import bootstrap
from ilo.evaluate import per_type
from ilo.train import dirs
from ilo.tune import GROUPINGS, Dev, ilo_f1, mean_f1
from ilo_llm.runner import OUT, arg
from ilo_llm.schema import NONE
from ilo_llm.tune import ClaudeDev, cross_fit, run_answers

MARGIN = 0.5
RARE6 = ["Default Illocuting", "Agreeing", "Challenging", "Disagreeing", "Arguing", "Restating"]


def fold_links(dev, probs, folds: dict, grouping: dict) -> dict:
    """Each map decoded with its held-out fold's thresholds (the links behind the honest score)."""
    ep_of, *_ = episodes(map_speakers())
    out = {}
    for n in dev.names:
        p = folds[str(ep_of[int(n.removeprefix("nodeset"))])]
        out[n] = dev.links(probs, n, p["link_min"], {t: p[g] for t, g in grouping.items()})
    return out


def as_tuple(per_map: dict) -> dict:
    return {n: tuple(None if x is None else tuple(x) for x in v) for n, v in per_map.items()}


def turn_links(links: list) -> dict:
    return {t: (p, y) for t, p, y in links}


def disagreements(cdev, claude: dict, labeller: dict, name: str = "claude") -> dict:
    """Turn by turn: links only the system makes (`name`; Claude by default), links only the labeller makes, and
    turns both link differently."""
    kinds = {f"{name}_only": [], "labeller_only": [], "different_proposition": []}
    for n in cdev.names:
        v = view(cdev.gold_aif[n], n)
        gold = {}
        for t, p, y in v.anchors():
            if t in v.turn:
                gold.setdefault(t, set()).add(p)
        c, l = turn_links(claude[n]), turn_links(labeller[n])
        gold_turn = {}                                           # proposition -> the turns gold links it from
        for t, ps in gold.items():
            for p in ps:
                gold_turn.setdefault(p, []).append(t)
        say = lambda t: f"{v.locs[t].speaker}: {v.locs[t].body}"
        for t in v.dialogue:
            def ex(who):
                p = who[t][0]
                elsewhere = [u for u in gold_turn.get(p, []) if u != t]
                return {"map": n, "turn": say(t), "proposition": v.props[p], "type": who[t][1],
                        "gold_links_this_proposition_from": say(elsewhere[0]) if elsewhere else None,
                        f"that_turn_linked_by_{name}": bool(elsewhere) and elsewhere[0] in c,
                        "that_turn_linked_by_labeller": bool(elsewhere) and elsewhere[0] in l}
            g = [v.props[p] for p in sorted(gold.get(t, ()))]
            if t in c and t not in l:
                kinds[f"{name}_only"].append({**ex(c), "gold": g, "right": c[t][0] in gold.get(t, ())})
            elif t in l and t not in c:
                kinds["labeller_only"].append({**ex(l), "gold": g, "right": l[t][0] in gold.get(t, ())})
            elif t in c and t in l and c[t][0] != l[t][0]:
                kinds["different_proposition"].append({
                    "map": n, "turn": f"{v.locs[t].speaker}: {v.locs[t].body}", name: v.props[c[t][0]],
                    "labeller": v.props[l[t][0]], "gold": g,
                    "right": name if c[t][0] in gold.get(t, ()) else "labeller" if l[t][0] in gold.get(t, ()) else "neither"})
    out = {}
    for k, items in kinds.items():
        out[k] = {"turns": len(items), "examples": items[:5]}
        if k == "different_proposition":
            out[k]["right"] = dict(Counter(i["right"] for i in items))
        else:
            out[k]["right"] = sum(i["right"] for i in items)
            wrong = [i for i in items if not i["right"]]
            # A wrong link whose proposition gold credits to another turn: the proposition was moved, not invented.
            out[k]["wrong_with_proposition_gold_linked_elsewhere"] = sum(bool(i["gold_links_this_proposition_from"]) for i in wrong)
            out[k]["examples_right"] = [i for i in items if i["right"]][:5]
            out[k]["examples_wrong"] = wrong[:5]
            del out[k]["examples"]
    return out


def link_stats(dev, links: dict) -> dict:
    hit = n_gold = n_pred = 0
    for n in dev.names:
        v = view(dev.gold_aif[n], n)
        g = {(t, p) for t, p, _ in v.anchors() if t in v.turn}
        p = {(t, q) for t, q, _ in links[n]}
        hit, n_gold, n_pred = hit + len(g & p), n_gold + len(g), n_pred + len(p)
    return {"recall": round(hit / n_gold, 4), "precision": round(hit / max(n_pred, 1), 4), "gold": n_gold, "predicted": n_pred}


def trap(dev, links: dict) -> dict:
    """Maps where the system predicts a type absent from that map's gold (the rare-label trap)."""
    maps, types = 0, Counter()
    for n in dev.names:
        extra = {y for *_, y in links[n]} - set(dev.gold[n].values())
        maps += bool(extra)
        types.update(extra)
    return {"maps": maps, "by_type": dict(types.most_common())}


def cost(run: str) -> dict:
    s = json.loads((OUT / run / "summary.json").read_text())
    rows = [r for r in csv.DictReader(open(RESULTS / "timing.csv", encoding="utf-8"))
            if r["approach"] == "ilo_llm" and r["stage"] == f"run:{run}"]
    hits = sum(1 for r in rows if (i := json.loads(r["info"]))["to_send"] and i["ok"] < i["jobs"])   # a pass cut short by the limit
    return {"calls_made": s["calls_made"], "accepted_answers": s["ok"], "failed": s["failed"],
            "seconds_per_call_p50": s["per_answer"]["seconds"]["p50"], "thinking_tokens_p50": s["per_answer"]["thinking_tokens"]["p50"],
            "api_equivalent_usd": s["api_equivalent_usd"],
            "runner_passes": len(rows), "usage_limit_hits": hits, "pro_sessions_used": 1 + hits,
            "seconds_in_runner": round(sum(float(r["seconds"]) for r in rows), 1)}


def main():
    run = arg("run", "dev")
    grouping = GROUPINGS["shared"]
    with timed("ilo_llm", f"evaluate:{run}") as row:
        th = json.loads((OUT / run / "thresholds.json").read_text())
        lab = json.loads((OUT / "labeller_per_map.json").read_text())
        lab_report = json.loads((RESULTS / "ilo" / "v1" / "dev_report.json").read_text())
        maps, gold, _ = run_answers(run)
        cdev = ClaudeDev(maps, gold)
        raw_types = Counter(it["type"] for _, answers in maps.values() for a in answers
                            for it in a["illocutions"] if it["proposition"] != NONE)
        c_links = fold_links(cdev, cdev.raw, th["folds"], grouping)

        ckpt, _ = dirs("v1")
        ldev = Dev(ckpt / "dev_probs.npy")
        lprobs = ldev.calibrated(lab["temperature"])
        _, lfolds = cross_fit(ldev, lprobs, grouping)
        l_links = fold_links(ldev, lprobs, lfolds, grouping)

        c_map, l_map = as_tuple(th["per_map_cross_fitted"]), as_tuple(lab["per_map_cross_fitted"])
        assert set(c_map) == set(l_map) == set(cdev.names), "the two systems must be scored on the same 225 maps"
        c_honest = {"general": round(mean_f1(c_map, 0), 2), "focused": round(mean_f1(c_map, 1), 2)}
        l_honest = {"general": round(mean_f1(l_map, 0), 2), "focused": round(mean_f1(l_map, 1), 2)}
        assert l_honest == lab["mean"]
        # The per-map links must reproduce each system's honest score.
        for dev, links, want in ((cdev, c_links, c_map), (ldev, l_links, l_map)):
            got = {n: ilo_f1(dev.gold[n], {(t, p): y for t, p, y in links[n]}, dev.n_pairs[n]) for n in dev.names}
            assert all(abs(got[n][0][2] - want[n][0][2]) < 1e-9 for n in dev.names)

        boot = {k: bootstrap(c_map, l_map, i) for i, k in enumerate(("general", "focused"))}
        better = c_honest["general"] > l_honest["general"] + MARGIN and c_honest["focused"] > l_honest["focused"] + MARGIN
        report = {
            "run": run, "maps": len(cdev.names),
            "1_main": {
                "claude": {"cross_fitted (honest)": c_honest, **{k: v for k, v in th["dev_ilo_f1"].items() if k != "cross_fitted (honest)"}},
                "labeller": {"cross_fitted (honest)": l_honest,
                             "argmax_decode": lab_report["ilo_f1"]["model, argmax decode"],
                             "tuned_on_all_dev (optimistic)": lab_report["ilo_f1"]["model, tuned decision layer (on all dev; optimistic)"]},
                "claude_thresholds_on_all_dev": {"link_min": th["link_min"], "type_min": th["type_min"]},
            },
            "2_bootstrap_claude_minus_labeller": {**boot, "resamples": 10000, "seed": 13},
            "3_verdict": {
                "rule": f"better only if honest general > {l_honest['general'] + MARGIN:.2f} and focused > {l_honest['focused'] + MARGIN:.2f} (D090)",
                "better": better,
                "line": (f"Claude {'beats' if better else 'does not beat'} the labeller on dev: honest ILO "
                         f"{c_honest['general']:.2f} / {c_honest['focused']:.2f} vs {l_honest['general']:.2f} / {l_honest['focused']:.2f}.")},
            "4_linking": {"claude": link_stats(cdev, c_links), "labeller": link_stats(cdev, l_links),
                          "labeller_reported (results/ilo/v1, tuned on all dev)": lab_report["linking"]["model_tuned"],
                          "disagreements": disagreements(cdev, c_links, l_links)},
            "5_per_type": {"claude": per_type(cdev, c_links), "labeller": per_type(ldev, l_links), "rare_types": RARE6,
                           "claude_types_as_answered (all answers, before voting and thresholds)": dict(raw_types.most_common())},
            "6_rare_label_trap": {"claude": trap(cdev, c_links), "labeller": trap(ldev, l_links)},
            "7_cost": cost(run),
        }
        row.update({"claude": c_honest, "better": better})
    with open(OUT / run / "dev_report.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k not in ("4_linking",)}, indent=1, ensure_ascii=False)[:6000])
    print(json.dumps(report["4_linking"]["claude"]), json.dumps(report["4_linking"]["labeller"]))
    print({k: (v["turns"], v["right"]) for k, v in report["4_linking"]["disagreements"].items()})


if __name__ == "__main__":
    main()
