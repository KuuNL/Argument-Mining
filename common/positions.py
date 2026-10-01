"""Where each proposition sits in the dialogue, given turn -> proposition links (D040).

Chain: a link (gold on training maps, the ILO labeller's predictions on dev/test) -> the most
similar dialogue turn if similarity >= SIMILAR -> unplaced. Node IDs and node order are never used
(D029). A proposition's speaker is the speaker of the turn it is placed at.
"""
from difflib import SequenceMatcher

from common.aif import MapView
from common.text import norm

SIMILAR = 0.8


def place(v: MapView, links) -> tuple[dict[str, int], dict[str, str]]:
    """links: iterable of (turn, proposition, ...). Returns (position by proposition, how it was placed)."""
    turn = v.turn
    pos, how = {}, {}
    for t, p, *_ in links:
        if t in turn and p in v.props and (p not in pos or turn[t] < pos[p]):
            pos[p], how[p] = turn[t], "link"
    bodies = [norm(v.locs[l].body) for l in v.dialogue]
    for p, text in v.props.items():
        if p in pos or not bodies:
            continue
        t = norm(text)
        sim, k = max((SequenceMatcher(None, t, b, autojunk=False).ratio(), k) for k, b in enumerate(bodies))
        if sim >= SIMILAR:
            pos[p], how[p] = k, "similar"
    return pos, how


def speakers(v: MapView, pos: dict[str, int]) -> dict[str, str]:
    return {p: v.locs[v.dialogue[k]].speaker for p, k in pos.items()}
