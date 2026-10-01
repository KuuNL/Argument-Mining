"""Turn candidate probabilities into L->I illocutions, and score them on dev.

decode_map: one proposition per turn and one turn per proposition (Hungarian matching on the
link score 1 - P(None)), keeping only links whose score clears `link_min`; each kept link gets
its most probable type. Step 3.4 adds temperature scaling and per-type thresholds on top.
"""
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.aif import strip_to_input, view
from common.score import score_map
from ilo.data import LABELS

NONE = LABELS.index("None")


ASSERTING = LABELS.index("Asserting")


def decode_map(rows: list[dict], probs: np.ndarray, link_min: float = 0.5,
               type_min: dict[str, float] | None = None) -> list[tuple[str, str, str]]:
    """rows / probs: the candidate pairs of ONE map. Returns [(turn, proposition, label)].

    type_min: per-type minimum of P(type | link); a link whose best type falls short becomes
    Asserting, so rare types are only predicted when the model is confident (3.4)."""
    turns = sorted({r["turn"] for r in rows})
    props = sorted({r["prop"] for r in rows})
    ti, pi = {t: i for i, t in enumerate(turns)}, {p: j for j, p in enumerate(props)}
    margin = np.zeros((len(turns), len(props)))
    label = {}
    for r, pr in zip(rows, probs):
        i, j = ti[r["turn"]], pi[r["prop"]]
        s = 1.0 - pr[NONE]
        if s >= link_min:
            margin[i, j] = s - link_min + 1e-6        # below-threshold pairs can't win the matching
        best = 1 + int(np.argmax(np.delete(pr, NONE)))
        if type_min and best != ASSERTING and pr[best] / max(s, 1e-9) < type_min.get(LABELS[best], 0.0):
            best = ASSERTING
        label[i, j] = LABELS[best]
    rr, cc = linear_sum_assignment(-margin)
    return [(turns[i], props[j], label[i, j]) for i, j in zip(rr, cc) if margin[i, j] > 0]


def predicted_map(gold_aif: dict, links) -> dict:
    """The test-format map plus one YA node per predicted link."""
    m = strip_to_input(gold_aif)
    for k, (t, p, y) in enumerate(links):
        ya = f"pred_ya_{k}"
        m["nodes"].append({"nodeID": ya, "type": "YA", "text": y})
        m["edges"] += [{"edgeID": f"{ya}_in", "fromID": t, "toID": ya, "formEdgeID": None},
                       {"edgeID": f"{ya}_out", "fromID": ya, "toID": p, "formEdgeID": None}]
    return m


class DevScorer:
    """Scores dev probabilities with the official ILO metric plus linking accuracy."""

    def __init__(self, rows: list[dict], gold_maps: dict[str, dict]):
        self.rows, self.gold = rows, gold_maps
        self.by_map = defaultdict(list)
        for k, r in enumerate(rows):
            self.by_map[r["map"]].append(k)
        self.gold_links = {}
        for name in self.by_map:
            v = view(gold_maps[name], name)
            self.gold_links[name] = {(s, t) for s, t, _ in v.anchors() if s in v.turn}

    def __call__(self, probs: np.ndarray, link_min: float = 0.5) -> dict:
        gen, foc, hit, n_gold, n_pred = [], [], 0, 0, 0
        for name, idx in self.by_map.items():
            links = decode_map([self.rows[k] for k in idx], probs[idx], link_min)
            s = score_map(self.gold[name], predicted_map(self.gold[name], links))["ilo"]
            gen.append(s[0][2])
            if s[1] is not None:
                foc.append(s[1][2])
            pairs = {(t, p) for t, p, _ in links}
            hit += len(pairs & self.gold_links[name])
            n_gold += len(self.gold_links[name])
            n_pred += len(pairs)
        return {"ilo_general_f1": round(100 * float(np.mean(gen)), 2),
                "ilo_focused_f1": round(100 * float(np.mean(foc)), 2),
                "link_recall": round(hit / n_gold, 4), "link_precision": round(hit / max(n_pred, 1), 4)}
