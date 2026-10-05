"""A2 answers -> relations, pair scores in A1's format, and turn-type votes.

Pair rows are every unordered pair of propositions (no pruning), A = the earlier one in P order,
which is A1's order (D020, D029). A row's five scores follow A1's labels
[None, Inference A->B, Inference B->A, Conflict, Rephrase]. Each relation score is the mean over
samples of the stated confidence (high 1, medium 2/3, low 1/3; 0 when a sample doesn't list it),
so A1's decision layer (a1_cross_encoder/decide.py) applies unchanged. The direction the model
gives a Conflict or Rephrase is kept in `relations` but not in the scores: the direction rule decides it.
"""
from collections import defaultdict

import numpy as np

from a2_llm.schema import CONFIDENCE, link_id

LABEL = {"Conflict": 3, "Rephrase": 4}


def relations(answer: dict, s) -> list[tuple[str, str, str, float]]:
    """(type, source I-node, target I-node, weight); self-loops dropped, a repeated pair keeps its highest weight."""
    best = {}
    for r in answer.get("relations", []):
        a, b = s.prop.get(r["from"]), s.prop.get(r["to"])
        if a and b and a != b:
            key = (r["type"], a, b)
            best[key] = max(best.get(key, 0.0), CONFIDENCE[r["confidence"]])
    return [(t, a, b, w) for (t, a, b), w in best.items()]


def pair_rows(s) -> list[dict]:
    props = list(s.prop.values())                     # P order
    return [{"map": s.name, "a": props[i], "b": props[j], "a_pos": s.pos.get(props[i]), "b_pos": s.pos.get(props[j])}
            for i in range(len(props)) for j in range(i + 1, len(props))]


def pair_scores(s, answers: list[dict], flat: bool = False) -> tuple[list[dict], np.ndarray]:
    """flat: plain votes, so every listed relation weighs 1 whatever its stated confidence (an A2.4 ablation)."""
    rows = pair_rows(s)
    index = {(r["a"], r["b"]): k for k, r in enumerate(rows)}
    scores = np.zeros((len(rows), 5))
    for answer in answers:
        one = defaultdict(float)                      # (row, label) -> weight, within this sample
        for t, src, tgt, w in relations(answer, s):
            k, forward = (index[(src, tgt)], True) if (src, tgt) in index else (index[(tgt, src)], False)
            label = (1 if forward else 2) if t == "Inference" else LABEL[t]
            one[(k, label)] = max(one[(k, label)], 1.0 if flat else w)
        for (k, label), w in one.items():
            scores[k, label] += w
    if answers:
        scores /= len(answers)
    scores[:, 0] = 1.0 - scores[:, 1:].max(1)
    return rows, scores


def type_votes(s, answers: list[dict]) -> dict[tuple[str, str], dict[str, float]]:
    """(L-node, I-node) -> {turn type: share of samples}."""
    ids = {link_id(t, p): (s.turn[t], s.prop[p]) for t, p in s.links}
    votes = defaultdict(lambda: defaultdict(float))
    for answer in answers:
        seen = {}
        for it in answer.get("illocutions", []):
            if it["link"] in ids:
                seen.setdefault(it["link"], it["type"])   # first entry per link in a sample
        for link, y in seen.items():
            votes[ids[link]][y] += 1 / len(answers)
    return {k: dict(v) for k, v in votes.items()}
