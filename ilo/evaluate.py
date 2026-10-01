"""Step 3.5: evaluate the ILO labeller on dev (maps in test-input format) against simple baselines.

    python -m ilo.evaluate [--run=v2] [--groups=shared|per-type]   -> results/ilo/<run>/dev_report.json
    python -m ilo.evaluate [--run=v2] --headroom                   -> results/ilo/<run>/headroom.json

Rows compared on the official ILO metric (mean per-map macro F1, general / focused):
  majority (no illocutions) · similarity + Asserting (link by text similarity, one-to-one, threshold
  tuned on dev) · trained model with argmax decode · trained model with tuned decision layer.
Plus linking recall / precision, per-type F1 (pooled over dev), and inference time per map.
"""
import json
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.finetune import load_model, predict, tokenize, Config
from ilo.data import LABELS
from ilo.train import arg, dirs, selected_run
from ilo.tune import Dev, ilo_f1, mean_f1



def similarity_links(dev: Dev, name: str, sim_min: float):
    rows = [dev.rows[k] for k in dev.idx[name]]
    turns, props = sorted({r["turn"] for r in rows}), sorted({r["prop"] for r in rows})
    ti, pi = {t: i for i, t in enumerate(turns)}, {p: j for j, p in enumerate(props)}
    m = np.zeros((len(turns), len(props)))
    for r in rows:
        if r["sim"] >= sim_min:
            m[ti[r["turn"]], pi[r["prop"]]] = r["sim"] - sim_min + 1e-6
    rr, cc = linear_sum_assignment(-m)
    return [(turns[i], props[j], "Asserting") for i, j in zip(rr, cc) if m[i, j] > 0]


def link_stats(dev: Dev, links_by_map: dict) -> dict:
    hit = n_gold = n_pred = 0
    for name, links in links_by_map.items():
        gold = {k for k in dev.gold[name] if k[0] in {dev.rows[i]["turn"] for i in dev.idx[name]}}
        pred = {(t, p) for t, p, _ in links}
        hit, n_gold, n_pred = hit + len(gold & pred), n_gold + len(gold), n_pred + len(pred)
    return {"recall": round(hit / n_gold, 4), "precision": round(hit / max(n_pred, 1), 4)}


def per_type(dev: Dev, links_by_map: dict) -> dict:
    tp, fp, fn = Counter(), Counter(), Counter()
    for name, links in links_by_map.items():
        gold, pred = dev.gold[name], {(t, p): y for t, p, y in links}
        for k in gold.keys() | pred.keys():
            g, p = gold.get(k), pred.get(k)
            if g == p:
                tp[g] += 1
            else:
                if p: fp[p] += 1
                if g: fn[g] += 1
    out = {}
    for y in LABELS[1:]:
        n = tp[y] + fn[y]
        p = tp[y] / (tp[y] + fp[y]) if tp[y] + fp[y] else 0.0
        r = tp[y] / n if n else 0.0
        out[y] = {"gold": n, "predicted": tp[y] + fp[y], "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}
    return out


def headroom(dev: Dev, th: dict) -> dict:
    """What would fixing each weakness be worth? Take the tuned predictions, correct one group of
    types to gold (or all types on correctly linked pairs), and re-score. No retraining involved."""
    from ilo.tune import QUESTIONS, RARE
    probs = dev.calibrated(th["temperature"])
    base = {n: {(t, p): y for t, p, y in dev.links(probs, n, th["link_min"], th["type_min"])} for n in dev.names}

    def fixed(types=None, all_types=False):
        out = {}
        for n in dev.names:
            pred = dict(base[n])
            for k, y in dev.gold[n].items():
                if (types and y in types) or (all_types and k in pred):
                    pred[k] = y
            out[n] = ilo_f1(dev.gold[n], pred, dev.n_pairs[n])
        return out

    variants = {"as is (tuned)": fixed(),
                "rare types correct": fixed(set(RARE)),
                "assertive + rhetorical questions correct": fixed({"Assertive Questioning", "Rhetorical Questioning"}),
                "both of the above": fixed(set(RARE) | {"Assertive Questioning", "Rhetorical Questioning"}),
                "every linked pair typed correctly": fixed(all_types=True)}
    maps_with = lambda types: sum(any(y in types for y in dev.gold[n].values()) for n in dev.names)
    return {"dev_maps": len(dev.names),
            "maps_with_a_rare_type": maps_with(set(RARE)),
            "maps_with_assertive_or_rhetorical": maps_with({"Assertive Questioning", "Rhetorical Questioning"}),
            "ilo_f1": {k: {"general": round(mean_f1(v, 0), 2), "focused": round(mean_f1(v, 1), 2)} for k, v in variants.items()}}


def main():
    mode = arg("groups", "shared")
    run = arg("run", selected_run())
    CKPT, OUT = dirs(run)
    th = json.loads((OUT / f"thresholds_{mode}.json").read_text())
    dev = Dev(CKPT / "dev_probs.npy")
    if "--headroom" in sys.argv:
        rep = headroom(dev, th)
        with open(OUT / "headroom.json", "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1))
        return

    # Inference timing (and a check that the saved dev probabilities come from this checkpoint).
    cfg = Config(num_labels=len(LABELS), out_dir=CKPT, log_path=OUT / "unused.jsonl")
    from transformers import AutoTokenizer
    pad = AutoTokenizer.from_pretrained(cfg.model).pad_token_id
    ids = tokenize(cfg.model, [r["text_a"] for r in dev.rows], [r["text_b"] for r in dev.rows], cfg.max_len)
    model = load_model(cfg, str(CKPT))
    t0 = time.perf_counter()
    probs_now = predict(model, ids, pad, cfg.eval_max_tokens)
    secs = time.perf_counter() - t0
    same = float(np.abs(probs_now - dev.raw).max())

    calibrated = dev.calibrated(th["temperature"])
    rows = {}
    rows["majority (no illocutions)"] = {n: ilo_f1(dev.gold[n], {}, dev.n_pairs[n]) for n in dev.names}
    best_sim, sim_scores = None, None
    for s in [x / 20 for x in range(4, 18)]:
        sc = {n: ilo_f1(dev.gold[n], {(t, p): y for t, p, y in similarity_links(dev, n, s)}, dev.n_pairs[n]) for n in dev.names}
        if best_sim is None or mean_f1(sc) > mean_f1(sim_scores):
            best_sim, sim_scores = s, sc
    rows[f"similarity + Asserting (threshold {best_sim}, tuned on dev)"] = sim_scores
    rows["model, argmax decode"] = dev.scores(dev.raw, 0.5, None)
    tuned_links = {n: dev.links(calibrated, n, th["link_min"], th["type_min"]) for n in dev.names}
    rows["model, tuned decision layer (on all dev; optimistic)"] = {
        n: ilo_f1(dev.gold[n], {(t, p): y for t, p, y in tuned_links[n]}, dev.n_pairs[n]) for n in dev.names}

    report = {
        "run": run, "dev_maps": len(dev.names), "groups": mode,
        "ilo_f1": {k: {"general": round(mean_f1(v, 0), 2), "focused": round(mean_f1(v, 1), 2)} for k, v in rows.items()},
        "ilo_f1_cross_fitted (honest)": th["dev_ilo_f1"]["cross_fitted (honest)"],
        "linking": {
            "similarity": link_stats(dev, {n: similarity_links(dev, n, best_sim) for n in dev.names}),
            "model_argmax": link_stats(dev, {n: dev.links(dev.raw, n, 0.5, None) for n in dev.names}),
            "model_tuned": link_stats(dev, tuned_links),
        },
        "per_type_tuned": per_type(dev, tuned_links),
        "inference": {"seconds_all_dev": round(secs, 1), "seconds_per_map": round(secs / len(dev.names), 3),
                      "max_abs_diff_vs_saved_probs": same},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "dev_report.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
