"""Jev vs the trained labeller (and vs Claude) on dev, Task B end to end (D099). No calls.

The same analyses as ilo_llm/evaluate.py, from its functions: each system's honest links (every map decoded with
its held-out fold's thresholds), the paired bootstrap (a3_finetune.tune.bootstrap, 10,000 resamples, seed 13),
linking, per type, the rare-label trap and cost. Claude's per-map honest scores come from results/ilo_llm/dev/.

    ~/.venvs/a3/bin/python -m ilo_jev.evaluate --run=dev      -> results/ilo_jev/<run>/dev_report.json
"""
import csv
import json
from collections import Counter

from common import RESULTS
from common.timing import timed
from a3_finetune.tune import bootstrap
from ilo.evaluate import per_type
from ilo.train import dirs
from ilo.tune import GROUPINGS, Dev, ilo_f1, mean_f1
from ilo_jev.parse import averaged
from ilo_jev.runner import OUT, answers, arg, builds
from ilo_jev.tune import load_run
from ilo_llm.evaluate import MARGIN, RARE6, as_tuple, disagreements, fold_links, link_stats, trap
from ilo_llm.tune import cross_fit


def cost(run: str) -> dict:
    s = json.loads((OUT / run / "summary.json").read_text())
    rows = [r for r in csv.DictReader(open(RESULTS / "timing.csv", encoding="utf-8"))
            if r["approach"] == "ilo_jev" and r["stage"] == f"run:{run}"]
    return {"requests": s["requests"], "answered": s["answered"], "maps": s["maps"], "usd": s["usd"],
            "usd_per_map": round(s["usd"] / s["maps"], 5), "input_tokens": s["input_tokens"],
            "seconds_per_request_p50": s["per_request"]["seconds_p50"], "retries": s["per_request"]["retries"],
            "seconds_in_runner": round(sum(float(r["seconds"]) for r in rows), 1), "models": s["models"]}


def type_argmax_counts(run: str) -> dict:
    """How often each type is Jev's most probable type for a turn it links (argmax link != none), before thresholds."""
    bs, ans = builds(run), answers(run)
    c = Counter()
    for name, b in bs.items():
        link, typ = averaged(b, ans[name], "link"), averaged(b, ans[name], "type")
        for t, dist in link.items():
            if max(dist, key=dist.get) is not None:
                c[max(typ[t], key=typ[t].get)] += 1
    return dict(c.most_common())


def main():
    run = arg("run", "dev")
    grouping = GROUPINGS["shared"]
    with timed("ilo_jev", f"evaluate:{run}") as row:
        th = json.loads((OUT / run / "thresholds.json").read_text())
        lab = json.loads((RESULTS / "ilo_llm" / "labeller_per_map.json").read_text())
        claude = json.loads((RESULTS / "ilo_llm" / "dev" / "thresholds.json").read_text())
        lab_report = json.loads((RESULTS / "ilo" / "v1" / "dev_report.json").read_text())
        jdev, _ = load_run(run)
        j_links = fold_links(jdev, jdev.raw, th["folds"], grouping)

        ckpt, _ = dirs("v1")
        ldev = Dev(ckpt / "dev_probs.npy")
        lprobs = ldev.calibrated(lab["temperature"])
        _, lfolds = cross_fit(ldev, lprobs, grouping)
        l_links = fold_links(ldev, lprobs, lfolds, grouping)

        j_map, l_map, c_map = (as_tuple(x["per_map_cross_fitted"]) for x in (th, lab, claude))
        assert set(j_map) == set(l_map) == set(c_map) == set(jdev.names), "all three systems on the same 225 maps"
        for dev, links, want in ((jdev, j_links, j_map), (ldev, l_links, l_map)):
            got = {n: ilo_f1(dev.gold[n], {(t, p): y for t, p, y in links[n]}, dev.n_pairs[n]) for n in dev.names}
            assert all(abs(got[n][0][2] - want[n][0][2]) < 1e-9 for n in dev.names)
        mean = lambda m: {"general": round(mean_f1(m, 0), 2), "focused": round(mean_f1(m, 1), 2)}
        j_honest, l_honest, c_honest = mean(j_map), mean(l_map), mean(c_map)
        assert l_honest == lab["mean"] and c_honest == claude["dev_ilo_f1"]["cross_fitted (honest)"]

        better = j_honest["general"] > l_honest["general"] + MARGIN and j_honest["focused"] > l_honest["focused"] + MARGIN
        report = {
            "run": run, "maps": len(jdev.names),
            "1_main": {
                "jev": {"cross_fitted (honest)": j_honest, **{k: v for k, v in th["dev_ilo_f1"].items() if k != "cross_fitted (honest)"}},
                "claude": claude["dev_ilo_f1"],
                "labeller": {"cross_fitted (honest)": l_honest,
                             "argmax_decode": lab_report["ilo_f1"]["model, argmax decode"],
                             "tuned_on_all_dev (optimistic)": lab_report["ilo_f1"]["model, tuned decision layer (on all dev; optimistic)"]},
                "jev_thresholds_on_all_dev": {"link_min": th["link_min"], "type_min": th["type_min"]},
            },
            "2_bootstrap": {"jev_minus_labeller": {k: bootstrap(j_map, l_map, i) for i, k in enumerate(("general", "focused"))},
                            "jev_minus_claude": {k: bootstrap(j_map, c_map, i) for i, k in enumerate(("general", "focused"))},
                            "resamples": 10000, "seed": 13},
            "3_verdict": {
                "rule": f"better only if honest general > {l_honest['general'] + MARGIN:.2f} and focused > {l_honest['focused'] + MARGIN:.2f} (D099)",
                "better": better,
                "line": (f"Jev {'beats' if better else 'does not beat'} the labeller on dev: honest ILO "
                         f"{j_honest['general']:.2f} / {j_honest['focused']:.2f} vs {l_honest['general']:.2f} / {l_honest['focused']:.2f}"
                         f" (Claude {c_honest['general']:.2f} / {c_honest['focused']:.2f}).")},
            "4_linking": {"jev": link_stats(jdev, j_links), "labeller": link_stats(jdev, l_links),
                          "disagreements_with_labeller": disagreements(jdev, j_links, l_links, name="jev")},
            "5_per_type": {"jev": per_type(jdev, j_links), "labeller": per_type(ldev, l_links), "rare_types": RARE6,
                           "jev_most_probable_type_on_linked_turns (before thresholds)": type_argmax_counts(run)},
            "6_rare_label_trap": {"jev": trap(jdev, j_links), "labeller": trap(ldev, l_links)},
            "7_cost": cost(run),
        }
        row.update({"jev": j_honest, "better": better})
    with open(OUT / run / "dev_report.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1, ensure_ascii=False))
    out = {k: v for k, v in report.items() if k not in ("4_linking", "5_per_type")}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(report["4_linking"]["jev"]), json.dumps(report["4_linking"]["labeller"]))
    print({k: (v["turns"], v["right"]) for k, v in report["4_linking"]["disagreements_with_labeller"].items()})
    print({t: (report["5_per_type"]["jev"][t]["predicted"], report["5_per_type"]["jev"][t]["f1"]) for t in report["5_per_type"]["jev"]})
    print(report["5_per_type"]["jev_most_probable_type_on_linked_turns (before thresholds)"])


if __name__ == "__main__":
    main()
