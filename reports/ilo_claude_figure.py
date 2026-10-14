"""Draw reports/figures/ilo-claude-vs-labeller.png (D090: can Claude do Task B?) from committed results.

    python reports/ilo_claude_figure.py      # from the repo root

Needs numpy and matplotlib only (as reports/short_report_figures.py, whose style and colours it reuses:
the labeller in A1's grey, Claude in A2's blue). Reads only results/ilo_llm/dev/dev_report.json.
a: honest (cross-fitted) dev ILO, general and focused, with the paired-bootstrap mean difference and its 95% interval.
b: per-type F1 pooled over dev for both systems, each type annotated with its number of gold instances.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from short_report_figures import COL, letter, style

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "ilo_llm" / "dev" / "dev_report.json"
OUT = ROOT / "reports" / "figures" / "ilo-claude-vs-labeller.png"
C_LAB, C_CLA = COL["A1"], COL["A2"]


def main():
    r = json.loads(REPORT.read_text(encoding="utf-8"))
    style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.15, 1.6]})

    main_ = r["1_main"]
    lab, cla = main_["labeller"]["cross_fitted (honest)"], main_["claude"]["cross_fitted (honest)"]
    x = np.arange(2)
    w = 0.36
    for off, vals, c, name in ((-w / 2, lab, C_LAB, "Trained labeller"), (w / 2, cla, C_CLA, "Claude Sonnet 5.5")):
        h = [vals["general"], vals["focused"]]
        a.bar(x + off, h, w, color=c, label=name)
        for xi, hi in zip(x + off, h):
            a.text(xi, hi + 1, f"{hi:.2f}", ha="center", va="bottom", fontsize=6)
    boot = r["2_bootstrap_claude_minus_labeller"]
    for xi, k in zip(x, ("general", "focused")):
        d, (lo, hi) = boot[k]["mean_difference"], boot[k]["ci95"]
        a.text(xi, 103, f"{d:+.2f}\n[{lo:+.2f}, {hi:+.2f}]".replace("-", "−"),
               ha="center", va="bottom", fontsize=6)
    a.text(0.5, 121, "Claude − labeller, 95% CI (paired bootstrap)", ha="center", va="bottom", fontsize=6, style="italic")
    a.set_xticks(x, ["General", "Focused"])
    a.set_ylim(0, 128)
    a.set_yticks(range(0, 101, 20))
    a.set_ylabel("Dev ILO F1, cross-fitted")
    a.set_title("Honest dev score (225 maps)", loc="left")
    a.legend(loc="lower center", frameon=False, ncol=1, bbox_to_anchor=(0.5, -0.42))
    letter(a, "a")

    pt = r["5_per_type"]
    types = [t for t in pt["claude"] if pt["claude"][t]["gold"] or pt["claude"][t]["predicted"] or pt["labeller"][t]["predicted"]]
    y = np.arange(len(types))[::-1]
    h = 0.38
    b.barh(y + h / 2, [pt["labeller"][t]["f1"] for t in types], h, color=C_LAB)
    b.barh(y - h / 2, [pt["claude"][t]["f1"] for t in types], h, color=C_CLA)
    b.set_yticks(y, [f"{t}  (n={pt['claude'][t]['gold']})" for t in types])
    zero = [yi for yi, t in zip(y, types) if not pt["labeller"][t]["f1"] and not pt["claude"][t]["f1"]]
    if zero:
        b.text(0.03, float(np.mean(zero)), f"F1 = 0 for both systems on these {len(zero)} types:\n"
               "neither predicts them once its tuned thresholds apply", va="center", fontsize=6, color="#6b7280")
    b.set_xlim(0, 1)
    b.set_xlabel("F1, pooled over dev (honest links)")
    b.set_title("Per illocution type", loc="left")
    letter(b, "b")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
