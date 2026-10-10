"""Redraw the four figures of reports/06-short-report.md from committed files.

    python reports/short_report_figures.py      # from the repo root; writes reports/figures/short-*.png

Needs numpy and matplotlib only (matplotlib is not in the pinned ML venv; any Python with both works).
Reads committed results under results/ and, for the bootstrap intervals and pooled counts, the output of
reports/short_report_numbers.py (results/short_report/derived.json; run that first if it is missing).
data/evaluation/ is not read. The few values that are not in results/ are quoted constants, each with its source.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "reports" / "figures"
MAPS = [f"map{i}" for i in range(1, 12)]

# ---- quoted constants (not in results/) ---------------------------------------------------------------
# 2024 shared-task results on the 11 test maps: overview paper (reference/2024.argmining-1.8.pdf), Tables 1-3.
DFKI = {"ari": (55.33, 30.40), "ilo": (78.78, 66.10), "global": (67.05, 48.25)}
FIELD_2024 = [("DFKI (2024 1st)", 55.33, 30.40), ("Pungene", 46.22, 20.51), ("KnowComp", 32.75, 9.06),
              ("Pokemon", 30.64, 35.89), ("Turiya", 30.75, 6.65), ("RoBERTa baseline", 26.46, 22.80),
              ("Predict nothing", 29.52, 0.0)]
# Dev references (A1 report section 5, D044): no relations predicted; every candidate pair labelled right.
DEV_MAJORITY, DEV_CEILING = 30.93, 98.64
# Time per map, seconds (final report section 7.3; A3 is a 22-40 s range, plotted at its midpoint).
TIME = {"A1": 0.94, "A2": 25.0, "A3": 31.0, "A4": 0.8}
A3_TIME_RANGE = (22.0, 40.0)

COL = {"A1": "#6b7280", "A2": "#2a78d6", "A3": "#b45309", "A4": "#7c3aed"}
C_ILO, C_ARI = "#2a78d6", "#e08a1e"


def load(rel):
    with open(RES / rel, encoding="utf-8") as f:
        return json.load(f)


def style():
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
    })


def letter(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")


def systems():
    a3 = load("a3/test_global.json")["votes"]
    return {"A1": load("a1/v1/test_global.json"), "A2": load("a2/test_global.json"), "A3": a3,
            "A4": load("a4/test_global.json")}


def fig_gain(S):
    names = {"A1": "A1 · encoder", "A2": "A2 · Claude", "A3": "A3 · fine-tuned Qwen", "A4": "A4 · Jev"}
    keys = list(names)[::-1]
    y = np.arange(len(keys))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, (si, s) in zip(axs, enumerate(("general", "focused"))):
        d_ilo = np.array([(S[k]["ilo"][s][2] - DFKI["ilo"][si]) / 2 for k in keys])
        d_ari = np.array([(S[k]["ari"][s][2] - DFKI["ari"][si]) / 2 for k in keys])
        d_glob = [S[k]["global"][s][2] - DFKI["global"][si] for k in keys]
        ax.barh(y, d_ilo, color=C_ILO, height=0.6)
        ax.barh(y, d_ari.clip(min=0), left=d_ilo, color=C_ARI, height=0.6)
        ax.barh(y, d_ari.clip(max=0), color=C_ARI, alpha=0.55, height=0.6)
        for yi, di, da, dg in zip(y, d_ilo, d_ari, d_glob):
            ax.text(di + max(da, 0) + 0.3, yi, f"+{dg:.1f}", va="center", fontsize=6.5)
        ax.axvline(0, color="#374151", lw=0.7)
        ax.set_xlim(-2.6, 13.5)
        ax.set_title(f"{s.capitalize()} F1", loc="left")
        ax.set_xlabel("Global F1 above the 2024 winner (points)")
    axs[0].set_yticks(y, [names[k] for k in keys])
    fig.legend(handles=[Patch(color=C_ILO, label="from illocutions (Task B)"),
                        Patch(color=C_ARI, label="from relations (Task A)"),
                        Patch(color=C_ARI, alpha=0.55, label="relations below 2024 (subtracts)")],
               frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.55, 0.0))
    for a, l in zip(axs, "ab"):
        letter(a, l)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "short-gain-vs-2024.png")
    plt.close(fig)


def per_map_ari(S):
    return {"A1": [S["A1"]["per_map_f1"][m]["ari"][0] for m in MAPS],
            "A2": [S["A2"]["per_map_f1"][m]["ari"][0] for m in MAPS],
            "A3": [S["A3"]["per_map_ari_f1"][m][0] for m in MAPS],
            "A4": [S["A4"]["per_map_ari_f1"][m][0] for m in MAPS]}


def fig_dev_vs_test(S, derived):
    rob = load("a4/robust.json")
    a3_vs_a1 = load("a3/v1/dev_report.json")["success_test_1"]["vs_A1"]["general"]
    diffs = [("A2 Claude", "A2", rob["paired_bootstrap_general"]["A2 - A1"]),
             ("A4 Jev", "A4", rob["run_to_run"]["pass 1"]["A4_minus_A1_general"]),
             ("A3 Qwen", "A3", a3_vs_a1)]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.25]})
    yy = np.arange(len(diffs))[::-1]
    for yi, (name, k, d) in zip(yy, diffs):
        lo, hi = d["ci95"]
        axa.plot([lo, hi], [yi, yi], color=COL[k], lw=1.6)
        axa.plot(d["mean_difference"], yi, "o", color=COL[k], ms=5)
        axa.text(hi + 0.3, yi, f"{d['mean_difference']:+.1f}", va="center", fontsize=6.5, color=COL[k])
    axa.axvline(0, color="#374151", lw=0.7, ls=":")
    axa.set_yticks(yy, [d[0] for d in diffs])
    axa.set_xlim(-6, 8.5)
    axa.set_ylim(-0.6, 2.6)
    axa.set_xlabel("Dev relation F1 minus A1 encoder (points, 95% CI)")
    axa.set_title("Dev (225 maps) separates the systems", loc="left")

    pm = per_map_ari(S)
    order = ["A1", "A2", "A3", "A4"]
    labels = {"A1": "A1 encoder", "A2": "A2 Claude", "A3": "A3 Qwen", "A4": "A4 Jev"}
    jit = np.random.default_rng(1)
    for i, k in enumerate(order):
        v = np.array(pm[k])
        axb.scatter(i + jit.uniform(-0.17, 0.17, len(v)), v, s=9, color=COL[k], alpha=0.45, lw=0)
        m, lo, hi = derived["test_ari_boot95"][k]
        axb.plot([i, i], [lo, hi], color=COL[k], lw=1.6)
        axb.plot([i - 0.22, i + 0.22], [m, m], color=COL[k], lw=2)
    axb.axhline(DFKI["ari"][0], color="#374151", lw=0.9, ls="--")
    axb.text(-0.45, 81.5, f"dashed line: 2024 winner, {DFKI['ari'][0]:.1f}", fontsize=6.5, color="#374151", va="center")
    axb.set_xticks(range(len(order)), [labels[k] for k in order])
    axb.set_xlim(-0.5, 3.5)
    axb.set_ylim(15, 85)
    axb.set_ylabel("Test relation F1 per map (general)")
    axb.set_title("Eleven test maps cannot separate them", loc="left")
    letter(axa, "a")
    letter(axb, "b")
    fig.tight_layout()
    fig.savefig(OUT / "short-dev-vs-test.png")
    plt.close(fig)


def fig_strengths():
    d4, d3 = load("a4/discussion.json"), load("a3/v1/discussion.json")
    ra = d3["recall_at_matched_density"]
    rows = [("Inference, adjacent turns", "Inference, 1 turn", ("distance", "1", "Inference")),
            ("Inference, 2–3 turns apart", "Inference, 2-3 turns", ("distance", "2-3", "Inference")),
            ("Rephrase, adjacent turns", "Rephrase, 1 turn", ("distance", "1", "Rephrase")),
            ("Rephrase, no shared content words", "Rephrase, no shared content words", ("overlap", "0", "Rephrase")),
            ("Conflict, between speakers", "Conflict, different speakers", ("speakers", "different speakers", "Conflict")),
            ("Conflict, within one speaker", "Conflict, same speaker", ("speakers", "same speaker", "Conflict"))]
    M = np.array([[d4["recall"]["A1"][k]["recall"], d4["recall"]["A2"][k]["recall"], ra[g][b][t]["A3"],
                   d4["recall"]["A4 both"][k]["recall"]] for _, k, (g, b, t) in rows])
    n = [d4["recall"]["A1"][k]["gold"] for _, k, _ in rows]
    fate = {k: [o["right"], o["reversed"], o["called Rephrase"], o["called Conflict"], o["missed"]]
            for k, o in d3["gold_inference_outcomes"].items()}
    o = d4["what_gold_becomes"]["A4 both"]["Inference"]
    fate["A4"] = [o.get("right", 0), o.get("Inference, reversed", 0), o.get("Rephrase", 0), o.get("Conflict", 0), o.get("None", 0)]
    total = sum(fate["A1"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
    im = ax1.imshow(M, cmap="Blues", vmin=0, vmax=0.8, aspect="auto")
    ax1.set_xticks(range(4), ["A1\nencoder", "A2\nClaude", "A3\nQwen", "A4\nJev"])
    ax1.set_yticks(range(len(rows)), [f"{r[0]} (n={k})" for r, k in zip(rows, n)])
    for i in range(M.shape[0]):
        best = M[i].argmax()
        for j in range(4):
            ax1.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                     color="white" if M[i, j] > 0.45 else "#111827", fontweight="bold" if j == best else "normal")
    ax1.tick_params(length=0)
    for s in ax1.spines.values():
        s.set_visible(False)
    ax1.set_title("Each model leads on a different kind of relation", loc="left")
    fig.colorbar(im, ax=ax1, fraction=0.04, pad=0.02).set_label("Recall at matched density (dev)")

    colors = ["#2a78d6", "#9ec1ee", "#e08a1e", "#f1c488", "#d1d5db"]
    names = ["found, right direction", "found, reversed", "called Rephrase", "called Conflict", "missed (None)"]
    ks = ["A1", "A2", "A3", "A4"]
    yb = np.arange(4)[::-1]
    left = np.zeros(4)
    for j in range(5):
        w = np.array([fate[k][j] for k in ks])
        ax2.barh(yb, w, left=left, color=colors[j], height=0.62, label=names[j], edgecolor="white", lw=0.4)
        if j in (0, 2):
            for yi, wi, l0 in zip(yb, w, left):
                narrow = wi < 80
                ax2.text(l0 + (wi + 8 if narrow else wi / 2), yi, str(wi), ha="left" if narrow else "center",
                         va="center", fontsize=6.5, color="white" if (j == 0 and not narrow) else "#1f2937")
        left += w
    ax2.set_yticks(yb, ["A1 encoder", "A2 Claude", "A3 Qwen", "A4 Jev"])
    ax2.set_xlim(0, total)
    ax2.set_xlabel(f"Gold Inference relations on dev (n = {total:,})")
    ax2.set_title("Where gold support relations go", loc="left")
    ax2.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3)
    letter(ax1, "a")
    letter(ax2, "b")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT / "short-model-strengths.png")
    plt.close(fig)


def fig_cost_and_scoring(S):
    rob = load("a4/robust.json")["with_dev_thresholds (honest)"]
    dev = {k: rob[k]["general"] for k in ("A1", "A2", "A3", "A4")}
    labels = {"A1": "A1 encoder\n$0 · 94 min training", "A2": "A2 Claude\n$0.18/map (API-equivalent)",
              "A3": "A3 Qwen\n$0 · 110 min training", "A4": "A4 Jev\n$0.003/map · no training"}
    offsets = {"A1": (6, -24), "A2": (-6, 12), "A3": (8, -22), "A4": (6, 10)}

    fig, (axc, axs) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axc.plot(A3_TIME_RANGE, [dev["A3"]] * 2, color=COL["A3"], lw=1.2)
    for k in labels:
        axc.plot(TIME[k], dev[k], "o", color=COL[k], ms=5, zorder=4)
        axc.annotate(labels[k], (TIME[k], dev[k]), xytext=offsets[k], textcoords="offset points", fontsize=6, color=COL[k])
    for yref in (DEV_MAJORITY, DEV_CEILING):
        axc.axhline(yref, color="#9aa5b1", lw=0.8, ls=":")
    axc.text(0.55, DEV_MAJORITY + 1.6, f"predict no relations: {DEV_MAJORITY:.1f}", fontsize=6, color="#4b5563")
    axc.text(0.55, DEV_CEILING - 3.6, f"every candidate pair labelled right: {DEV_CEILING:.1f}", fontsize=6,
             color="#4b5563", va="top")
    axc.set_xscale("log")
    axc.set_xlim(0.5, 200)
    axc.set_xticks([1, 10, 100], ["1 s", "10 s", "100 s"])
    axc.set_ylim(25, 102)
    axc.set_xlabel("Time per map (log scale)")
    axc.set_ylabel("Dev relation F1 (general, honest)")
    axc.set_title("All methods sit in a narrow band, far below the ceiling", loc="left")

    off24 = {"Pokemon": (4, -2), "DFKI (2024 1st)": (-62, -10), "Predict nothing": (4, 1), "Turiya": (-26, -3),
             "KnowComp": (4, -1), "RoBERTa baseline": (4, -2), "Pungene": (4, -3)}
    for name, x, y in FIELD_2024:
        axs.plot(x, y, "s", ms=4.5, mfc="white", mec="#6b7280", mew=1)
        axs.annotate(name, (x, y), xytext=off24[name], textcoords="offset points", fontsize=6, color="#4b5563")
    for k in ("A1", "A2", "A3", "A4"):
        axs.plot(S[k]["ari"]["general"][2], S[k]["ari"]["focused"][2], "o", ms=4.5, color=COL[k], zorder=4)
    c = S["A2"]["ari"]
    axs.annotate("ours (A1–A4)", (c["general"][2], c["focused"][2]), xytext=(-40, 20),
                 textcoords="offset points", fontsize=6, color="#111827",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color="#6b7280"))
    axs.set_xlim(20, 62)
    axs.set_ylim(-3, 42)
    axs.set_xlabel("Test relation F1, general (rewards abstaining)")
    axs.set_ylabel("Test relation F1, focused (rewards guessing)")
    axs.set_title("The two settings reward opposite strategies", loc="left")
    letter(axc, "a")
    letter(axs, "b")
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "short-cost-and-scoring.png")
    plt.close(fig)


def fig_why_relations(derived):
    err = derived["relation_errors_dev_matched_density"]
    rec = derived["relation_recall_by_distance_dev"]
    names = {"A1": "A1 encoder", "A2": "A2 Claude", "A4": "A4 Jev"}
    ks = ["A1", "A2", "A4"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
    parts = [("wrong_type", "related, wrong type", "#e08a1e", 1.0), ("reversed", "related, wrong direction", "#f1c488", 1.0),
             ("missed", "missed (gold relation called unrelated)", "#6b7280", 1.0),
             ("invented", "invented (unrelated pair given a relation)", "#c4c9d1", 1.0)]
    y = np.arange(len(ks))[::-1]
    left = np.zeros(len(ks))
    for key, lab, col, _ in parts:
        w = np.array([err[k][key] for k in ks])
        axa.barh(y, w, left=left, color=col, height=0.6, label=lab, edgecolor="white", lw=0.4)
        if key in ("missed", "invented"):
            for yi, wi, l0 in zip(y, w, left):
                axa.text(l0 + wi / 2, yi, str(wi), ha="center", va="center", fontsize=6.5,
                         color="white" if key == "missed" else "#1f2937")
        left += w
    for yi, k, tot in zip(y, ks, left):
        share = err[k]["share_of_errors_on_whether_related"]
        axa.text(tot + 40, yi, f"{share:.0%}", va="center", fontsize=6.5)
    axa.set_yticks(y, [f"{names[k]}\n({err[k]['right']} of {err[k]['gold']:,} right)" for k in ks])
    axa.set_xlim(0, max(left) * 1.12)
    axa.set_xlabel("Errors on dev at matched density (count)\n% at right = share that are missed or invented")
    axa.set_title("Most errors: is this pair related at all?", loc="left")
    axa.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2)

    dists = ["1 turn", "2-3 turns", "4-10 turns"]
    xl = [f"adjacent\n(n={rec['A1']['1 turn']['gold']:,})", f"2–3 turns\n(n={rec['A1']['2-3 turns']['gold']})",
          f"4–10 turns\n(n={rec['A1']['4-10 turns']['gold']})"]
    for k in ks:
        v = [rec[k][d]["recall"] for d in dists]
        axb.plot(range(3), v, "-o", color=COL[k], ms=4, lw=1.4)
        axb.text(2.08, v[-1], names[k], color=COL[k], fontsize=6.5, va="center")
    axb.set_xticks(range(3), xl)
    axb.set_xlim(-0.2, 2.75)
    axb.set_ylim(0, 0.55)
    axb.set_ylabel("Recall (dev, matched density)")
    axb.set_title("The pairwise encoder loses distant relations", loc="left")
    letter(axa, "a")
    letter(axb, "b")
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "short-why-relations.png")
    plt.close(fig)


def main():
    derived_path = RES / "short_report" / "derived.json"
    if not derived_path.exists():
        raise SystemExit("run `python reports/short_report_numbers.py` first")
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    S = systems()
    fig_gain(S)
    fig_dev_vs_test(S, derived)
    fig_strengths()
    fig_cost_and_scoring(S)
    fig_why_relations(derived)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("short-*.png"))))


if __name__ == "__main__":
    main()
