"""Jev answers -> candidate rows and probabilities in ilo.decode's format (D099).

Each question's probabilities are averaged over the option orders asked (a4_jev.parse.averaged). For turn T,
L(P) = P(T expresses P) from link_T and Y(y) = P(type y) from type_T. Every (T, P) with L(P) > 0 is a row with
P(None) = 1 - L(P) and P(y) = L(P) x Y(y), so ilo.decode.decode_map and the labeller's decision layer apply unchanged.
"""
import numpy as np

from a4_jev.parse import averaged, problems  # noqa: F401  (problems: the strict answer check, reused as is)
from ilo.data import LABELS


def rows_probs(b, answers: dict) -> tuple[list[dict], np.ndarray]:
    s = b.s
    link, typ = averaged(b, answers, "link"), averaged(b, answers, "type")
    rows, probs = [], []
    for t, lid in s.turn.items():                        # dialogue order, then P order
        y = typ.get(t, {"Asserting": 1.0})
        for p, pid in s.prop.items():
            share = link.get(t, {}).get(p, 0.0)
            if share <= 0:
                continue
            row = np.zeros(len(LABELS))
            row[0] = 1.0 - share
            for label, q in y.items():
                row[LABELS.index(label)] = share * q
            rows.append({"map": s.name, "turn": lid, "prop": pid})
            probs.append(row)
    return rows, np.array(probs).reshape(-1, len(LABELS))
