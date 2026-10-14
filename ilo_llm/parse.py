"""Answers -> votes per (turn, proposition) -> candidate rows and probabilities in ilo.decode's format.

For every (L-node, I-node) pair that any answer links: the link share (answers linking it / number of
answers) and the type shares (among the answers linking it). As probabilities over ilo.data.LABELS:
P(None) = 1 - link share and P(type) = link share x type share, so ilo.decode.decode_map and the
labeller's decision layer (ilo/tune.py) apply unchanged. Pairs no answer links are not rows (None).
"""
from collections import defaultdict

import numpy as np

from ilo.data import LABELS
from ilo_llm.schema import NONE


def votes(s, answers: list[dict]) -> dict[tuple[str, str], dict]:
    """(L-node, I-node) -> {"link": share, "types": {type: share among linking answers}}."""
    counts = defaultdict(lambda: defaultdict(int))
    for answer in answers:
        seen = {}
        for it in answer.get("illocutions", []):
            if it["proposition"] != NONE and it["turn"] in s.turn and it["proposition"] in s.prop:
                seen.setdefault((s.turn[it["turn"]], s.prop[it["proposition"]]), it["type"])   # first entry per pair
        for key, y in seen.items():
            counts[key][y] += 1
    out = {}
    for key, c in counts.items():
        n = sum(c.values())
        out[key] = {"link": n / len(answers), "types": {y: k / n for y, k in c.items()}}
    return out


def rows_probs(s, v: dict) -> tuple[list[dict], np.ndarray]:
    """Rows {"map", "turn", "prop"} in dialogue then P order, and an (n, 11) array in LABELS order."""
    tpos = {lid: k for k, lid in enumerate(s.turn.values())}
    ppos = {p: k for k, p in enumerate(s.prop.values())}
    keys = sorted(v, key=lambda k: (tpos[k[0]], ppos[k[1]]))
    probs = np.zeros((len(keys), len(LABELS)))
    for i, k in enumerate(keys):
        probs[i, 0] = 1.0 - v[k]["link"]
        for y, share in v[k]["types"].items():
            probs[i, LABELS.index(y)] = v[k]["link"] * share
    return [{"map": s.name, "turn": t, "prop": p} for t, p in keys], probs
