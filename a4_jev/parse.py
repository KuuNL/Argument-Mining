"""A4 answers -> pair scores in A1's format, turn-type probabilities, and a strict answer check.

Rows are A2's: every unordered pair of propositions in P order (a2_llm/parse.pair_rows), so A1, A2, A3 and A4
scores line up. A row's five scores follow A1's labels [None, Inference A->B, Inference B->A, Conflict, Rephrase].
Each question's probabilities are averaged over the option orders asked (forward / reversed). Then:
- pair view: the pair question's probabilities, label by label;
- source view: A->B Inference from A's question (supports_B), B->A Inference from B's (supports_A), Conflict and
  Rephrase from the later proposition's (attacks_A / restates_A); None = 1 - their sum (clipped at 0);
- both: the mean of the two.
Pairs outside the candidates score None = 1. `answers` maps question id -> Jev's answer {type, choice, probabilities}.
"""
from collections import defaultdict

import numpy as np

from a2_llm.parse import pair_rows

VIEWS = ("pair", "source", "both")


def problems(b, answers: dict) -> list[str]:
    """Everything wrong with a map's answers: missing questions, unknown options, probabilities that don't sum to 1."""
    out = []
    for qid, q in b.questions.items():
        a = answers.get(qid)
        if a is None:
            out.append(f"{qid}: no answer")
            continue
        opts = set(q["criteria"])
        probs = a.get("probabilities") or {}
        if a.get("type") != "choice" or a.get("choice") not in opts:
            out.append(f"{qid}: bad choice {a.get('choice')!r}")
        if set(probs) != opts:
            out.append(f"{qid}: options {sorted(set(probs) ^ opts)[:4]} differ")
        elif abs(sum(probs.values()) - 1) > 0.02:
            out.append(f"{qid}: probabilities sum to {sum(probs.values()):.3f}")
    return out


def averaged(b, answers: dict, kind: str, orders=None) -> dict:
    """item -> {meaning: probability}, averaged over the orders asked (or only `orders`)."""
    acc = defaultdict(lambda: defaultdict(float))
    n = defaultdict(int)
    for qid, (k, o, item, meaning) in b.meta.items():
        if k != kind or (orders and o not in orders) or qid not in answers:
            continue
        n[item] += 1
        for opt, p in answers[qid]["probabilities"].items():
            acc[item][meaning[opt]] += p
    return {item: {m: p / n[item] for m, p in d.items()} for item, d in acc.items()}


def pair_scores(b, answers: dict, view: str = "both", orders=None) -> tuple[list[dict], np.ndarray]:
    s = b.s
    rows = pair_rows(s)
    index = {(r["a"], r["b"]): k for k, r in enumerate(rows)}
    scores = np.zeros((len(rows), 5))
    scores[:, 0] = 1.0
    pair = averaged(b, answers, "pair", orders) if view in ("pair", "both") else {}
    src = averaged(b, answers, "source", orders) if view in ("source", "both") else {}
    for a, c in b.candidates:
        k = index[(s.prop[a], s.prop[c])]
        vecs = []
        if view in ("pair", "both"):
            d = pair[(a, c)]
            vecs.append(np.array([d.get(i, 0.0) for i in range(5)]))
        if view in ("source", "both"):
            da, dc = src.get(a, {}), src.get(c, {})
            rel = [da.get(("Inference", c), 0.0), dc.get(("Inference", a), 0.0),
                   dc.get(("Conflict", a), 0.0), dc.get(("Rephrase", a), 0.0)]
            vecs.append(np.array([max(0.0, 1.0 - sum(rel))] + rel))
        scores[k] = np.mean(vecs, axis=0)
    return rows, scores


def type_probs(b, answers: dict, orders=None) -> dict[tuple[str, str], dict[str, float]]:
    """(L-node, I-node) -> {turn type: probability}."""
    s = b.s
    return {(s.turn[t], s.prop[p]): d for (t, p), d in averaged(b, answers, "type", orders).items()}
