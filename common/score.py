"""Port of the official DialAM-2024 scorers (reference/eval-scripts/), with their quirks.

Per map: every ordered pair of gold I-nodes (ARI) and every (L-node, I-node) pair (ILO)
gets a gold and a predicted label; macro P/R/F1 is taken over the labels present in
either ("general"), and over gold-related pairs only ("focused"). Maps are averaged,
and Global is the mean of ARI and ILO (matches the paper's Tables 1-3).

Quirks kept from the official scripts:
- a relation/illocution node's target is its FIRST outgoing edge; sources are all incoming edges;
- ARI sources must be gold I-nodes; the target is not checked (non-I targets just never match a pair);
- one pair with several relations: Inference > Conflict > Rephrase;
- one (L, I) pair with several YAs: the first in node order, then edge order, wins;
- node IDs in the prediction are matched against the GOLD node lists.
Deviation: a map whose focused list is empty (no gold relations) makes sklearn raise;
it is left out of the focused mean and counted instead (D013).
"""
from collections import defaultdict
from itertools import permutations, product

from sklearn.metrics import precision_recall_fscore_support

ARI_CODES = {"RA": 0, "CA": 1, "MA": 2}   # official label ids; 3 = None
ARI_NONE, ILO_NONE = 3, "None"


def _edges(aif):
    out_e, in_e = defaultdict(list), defaultdict(list)
    for e in aif["edges"]:
        out_e[e["fromID"]].append(e["toID"])
        in_e[e["toID"]].append(e["fromID"])
    return out_e, in_e


def ari_labels(aif: dict, props: set) -> dict:
    """(premise, target) -> label code, highest-priority relation kept."""
    out_e, in_e = _edges(aif)
    rel = {}
    for n in aif["nodes"]:
        code = ARI_CODES.get(n["type"])
        if code is None or not out_e[n["nodeID"]]:
            continue
        tgt = out_e[n["nodeID"]][0]
        for src in in_e[n["nodeID"]]:
            if src in props and code < rel.get((src, tgt), ARI_NONE):
                rel[(src, tgt)] = code
    return rel


def ilo_labels(aif: dict, proploc: set) -> dict:
    """(source, target) -> YA label, first occurrence kept."""
    out_e, in_e = _edges(aif)
    ill = {}
    for n in aif["nodes"]:
        if n["type"] != "YA" or not out_e[n["nodeID"]]:
            continue
        tgt = out_e[n["nodeID"]][0]
        if tgt not in proploc:
            continue
        for src in in_e[n["nodeID"]]:
            if src in proploc:
                ill.setdefault((src, tgt), n["text"])
    return ill


def _prf(y_true, y_pred):
    if not y_true:
        return None
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return float(p), float(r), float(f)


def _general_focused(y_true, y_pred, none):
    foc = [(t, p) for t, p in zip(y_true, y_pred) if t != none]
    return _prf(y_true, y_pred), _prf([t for t, _ in foc], [p for _, p in foc])


def score_map(gold: dict, pred: dict) -> dict:
    """{'ari': (general, focused), 'ilo': (general, focused)}; each a (P, R, F1) tuple or None."""
    props = [n["nodeID"] for n in gold["nodes"] if n["type"] == "I"]
    locs = [n["nodeID"] for n in gold["nodes"] if n["type"] == "L"]
    prop_set, proploc = set(props), set(props) | set(locs)

    t_ari, p_ari = ari_labels(gold, prop_set), ari_labels(pred, prop_set)
    pairs = list(permutations(props, 2))
    ari = _general_focused([t_ari.get(q, ARI_NONE) for q in pairs],
                           [p_ari.get(q, ARI_NONE) for q in pairs], ARI_NONE)

    t_ilo, p_ilo = ilo_labels(gold, proploc), ilo_labels(pred, proploc)
    pairs = list(product(locs, props))
    ilo = _general_focused([t_ilo.get(q, ILO_NONE) for q in pairs],
                           [p_ilo.get(q, ILO_NONE) for q in pairs], ILO_NONE)
    return {"ari": ari, "ilo": ilo}


def aggregate(per_map: list[dict]) -> dict:
    """Mean over maps, as percentages: {task: {setting: (P, R, F1)}} for ari, ilo, global.

    Also reports how many maps were left out of each mean (empty focused lists)."""
    out, skipped = {}, {}
    for task in ("ari", "ilo"):
        out[task] = {}
        for k, setting in enumerate(("general", "focused")):
            vals = [m[task][k] for m in per_map if m[task][k] is not None]
            skipped[f"{task}_{setting}"] = len(per_map) - len(vals)
            out[task][setting] = tuple(100 * sum(v[i] for v in vals) / len(vals) for i in range(3)) if vals else None
    out["global"] = {s: tuple((a + b) / 2 for a, b in zip(out["ari"][s], out["ilo"][s]))
                     for s in ("general", "focused") if out["ari"][s] and out["ilo"][s]}
    out["skipped_maps"] = skipped
    return out


def score_maps(pairs) -> dict:
    """pairs: iterable of (gold aif, predicted aif). Returns aggregate() plus per-map scores."""
    per_map = [score_map(g, p) for g, p in pairs]
    return {**aggregate(per_map), "per_map": per_map}


def format_table(agg: dict) -> str:
    rows = ["| | General P / R / F1 | Focused P / R / F1 |", "|---|---|---|"]
    for task in ("ari", "ilo", "global"):
        cells = [" / ".join(f"{x:.2f}" for x in agg[task][s]) if agg[task].get(s) else "—"
                 for s in ("general", "focused")]
        rows.append(f"| {task.upper()} | {cells[0]} | {cells[1]} |")
    return "\n".join(rows)
