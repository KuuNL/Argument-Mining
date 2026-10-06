"""A3 answers -> pair scores in A1's format, and turn-type votes.

Rows are A2's: every unordered pair of propositions, A = the earlier one in P order (a2_llm/parse.pair_rows),
with A1's five scores [None, Inference A->B, Inference B->A, Conflict, Rephrase]. A relation's score is the
mean over the answers of its weight in each answer: 1 when listed (vote share, the main score, D069), or the
probability the model gave the entry's tokens (the token-probability ablation, D065); 0 when not listed.
None = 1 - the best relation score, so A1's decision layer applies unchanged.
"""
import re
from collections import defaultdict

import numpy as np

from a2_llm import parse as a2
from a2_llm.parse import pair_rows

LABEL = {"Conflict": 3, "Rephrase": 4}
ENTRY = re.compile(r'\{"type":"(Inference|Conflict|Rephrase)","from":"(P\d+)","to":"(P\d+)"\}')


def entry_probs(tok, tokens: list[int], logps: list[float]) -> dict:
    """(type, from, to) -> probability of the tokens that wrote that relation entry, for one sampled answer.
    The relations part is ASCII, so each token's decoded length gives exact character offsets."""
    pieces = [tok.decode([t]) for t in tokens]
    ends = np.cumsum([len(p) for p in pieces])
    starts = ends - np.array([len(p) for p in pieces])
    text = "".join(pieces)
    out = {}
    for m in ENTRY.finditer(text):
        span = [k for k in range(min(len(tokens), len(logps))) if starts[k] < m.end() and ends[k] > m.start()]
        p = float(np.exp(sum(logps[k] for k in span)))
        out[m.groups()] = max(out.get(m.groups(), 0.0), p)
    return out


def pair_scores(s, answers: list[dict], weights: list[dict] | None = None):
    """answers: parsed A3 answers; weights[i]: {(type, from, to): weight} for answer i (default: 1 per listed relation)."""
    rows = pair_rows(s)
    index = {(r["a"], r["b"]): k for k, r in enumerate(rows)}
    scores = np.zeros((len(rows), 5))
    for i, answer in enumerate(answers):
        one = defaultdict(float)
        for r in answer.get("relations", []):
            a, b = s.prop.get(r["from"]), s.prop.get(r["to"])
            if not a or not b or a == b:
                continue
            w = 1.0 if weights is None else weights[i].get((r["type"], r["from"], r["to"]), 0.0)
            k, forward = (index[(a, b)], True) if (a, b) in index else (index[(b, a)], False)
            label = (1 if forward else 2) if r["type"] == "Inference" else LABEL[r["type"]]
            one[(k, label)] = max(one[(k, label)], w)
        for (k, label), w in one.items():
            scores[k, label] += w
    if answers:
        scores /= len(answers)
    scores[:, 0] = 1.0 - scores[:, 1:].max(1)
    return rows, scores


def relations(answer: dict, s):
    return a2.relations({"relations": [dict(r, confidence="high") for r in answer.get("relations", [])]}, s)


def type_votes(s, answers: list[dict]):
    return a2.type_votes(s, answers)
