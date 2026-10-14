"""Draw reports/figures/ilo-llms-vs-labeller.png (D090, D099: can a general model do Task B?) from committed results.

    python reports/ilo_llm_figure.py      # from the repo root

Needs numpy and matplotlib only (as reports/short_report_figures.py, whose style and colours it reuses: the labeller in
A1's grey, Claude in A2's blue, Jev in A4's purple). Reads only results/ilo_llm/dev/dev_report.json (labeller, Claude)
and results/ilo_jev/dev/dev_report.json (Jev).
a: honest (cross-fitted) dev ILO, general and focused, with each LLM's paired-bootstrap difference from the labeller.
b: per-type F1 pooled over dev (honest links), each type annotated with its number of gold instances.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from short_report_figures import COL, letter, style

ROOT = Path(__file__).resolve().parents[1]
CLAUDE = ROOT / "results" / "ilo_llm" / "dev" / "dev_report.json"
JEV = ROOT / "results" / "ilo_jev" / "dev" / "dev_report.json"
OUT = ROOT / "reports" / "figures" / "ilo-llms-vs-labeller.png"
SYSTEMS = [("labeller", "Trained labeller", COL["A1"]), ("claude", "Claude Sonnet 5.5", COL["A2"]), ("jev", "Jev", COL["A4"])]


def minus(s: str) -> str:
    return s.replace("-", "−")


def main():
    c = json.loads(CLAUDE.read_text(encoding="utf-8"))
    j = json.loads(JEV.read_text(encoding="utf-8"))
    honest = {"labeller": c["1_main"]["labeller"]["cross_fitted (honest)"], "claude": c["1_main"]["claude"]["cross_fitted (honest)"],
              "jev": j["1_main"]["jev"]["cross_fitted (honest)"]}
    assert honest["labeller"] == j["1_main"]["labeller"]["cross_fitted (honest)"]
    boot = {"claude": c["2_bootstrap_claude_minus_labeller"], "jev": j["2_bootstrap"]["jev_minus_labeller"]}
    types = {"labeller": c["5_per_type"]["labeller"], "claude": c["5_per_type"]["claude"], "jev": j["5_per_type"]["jev"]}

    style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.25, 1.5]})
    x = np.arange(2)
    w = 0.26
    for i, (key, name, col) in enumerate(SYSTEMS):
        h = [honest[key]["general"], honest[key]["focused"]]
        xs = x + (i - 1) * w
        a.bar(xs, h, w, color=col, label=name)
        for xi, hi in zip(xs, h):
            a.text(xi, hi + 1, f"{hi:.1f}", ha="center", va="bottom", fontsize=5.5)
        if key in boot:
            for xi, k in zip(xs, ("general", "focused")):
                d, (lo, hi) = boot[key][k]["mean_difference"], boot[key][k]["ci95"]
                a.text(xi, 7, minus(f"{d:+.1f}\n[{lo:+.1f},\n{hi:+.1f}]"), ha="center", va="bottom", fontsize=4.8, color="white")
    a.set_xticks(x, ["General", "Focused"])
    a.set_ylim(0, 108)
    a.set_yticks(range(0, 101, 20))
    a.set_ylabel("Dev ILO F1, cross-fitted")
    a.set_title("Honest dev score (225 maps)", loc="left")
    a.text(0.5, -0.2, "White: difference from the labeller, 95% CI (paired bootstrap)", transform=a.transAxes,
           ha="center", va="top", fontsize=5.5, style="italic")
    a.legend(loc="upper center", frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.27), handlelength=1.2, columnspacing=1.0)
    letter(a, "a")

    shown = [t for t in types["labeller"] if any(types[k][t]["f1"] for k in types)]
    rare = [t for t in types["labeller"] if t not in shown]
    y = np.arange(len(shown))[::-1]
    h = 0.26
    for i, (key, _, col) in enumerate(SYSTEMS):
        b.barh(y + (1 - i) * h, [types[key][t]["f1"] for t in shown], h, color=col)
    b.set_yticks(y, [f"{t}  (n={types['labeller'][t]['gold']})" for t in shown])
    b.set_xlim(0, 1)
    b.set_xlabel("F1, pooled over dev (honest links)")
    b.set_title("Per illocution type", loc="left")
    n_rare = sum(types["labeller"][t]["gold"] for t in rare)
    b.text(0.0, -0.3, f"Not shown: {len(rare)} rare types ({n_rare} gold links), F1 = 0 for all three systems:\n"
           "each system's tuned thresholds switch them off.", transform=b.transAxes, ha="left", va="top", fontsize=5.5, color="#6b7280")
    letter(b, "b")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
