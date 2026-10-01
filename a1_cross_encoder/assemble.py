"""Turn A1 pair labels into AIF relation nodes (used by the A1.1 round trip; extended in A1.5).

A pair (A, B) is ordered with A the earlier proposition. Inference keeps its predicted direction;
Conflict and Rephrase are written later -> earlier, i.e. B -> A (D020).
"""
import copy

LABELS = ["None", "Inference A->B", "Inference B->A", "Conflict", "Rephrase"]
S_NODE = {"Inference": ("RA", "Default Inference"), "Conflict": ("CA", "Default Conflict"),
          "Rephrase": ("MA", "Default Rephrase")}


def relation(a: str, b: str, label: str):
    """(relation type, source, target) for one labelled pair, or None."""
    if label == "Inference A->B":
        return "Inference", a, b
    if label == "Inference B->A":
        return "Inference", b, a
    if label in ("Conflict", "Rephrase"):
        return label, b, a
    return None


def with_relations(aif: dict, rels) -> dict:
    """A copy of `aif` with one S-node (and its two edges) per (type, source, target)."""
    m = copy.deepcopy(aif)
    for k, (kind, src, tgt) in enumerate(rels):
        t, text = S_NODE[kind]
        nid = f"a1_{t}_{k}"
        m["nodes"].append({"nodeID": nid, "type": t, "text": text})
        m["edges"] += [{"edgeID": f"{nid}_in", "fromID": src, "toID": nid, "formEdgeID": None},
                       {"edgeID": f"{nid}_out", "fromID": nid, "toID": tgt, "formEdgeID": None}]
    return m
