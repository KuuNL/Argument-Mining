"""Map -> text for the whole-map approaches (A2; reused by A3 and A4).

Turns are T1... in dialogue order, written "Speaker: words". Under each turn come the propositions
placed at it (common/positions.py: links, then text similarity), renumbered P1... in dialogue order
(ties by text, unplaced last). The order is A1's pair order, so node IDs and node order never reach
the model (D029). Links are gold on training maps and the ILO labeller's predictions on dev and test (D040).

    s = serialise(aif, name, links)   # s.text; s.prop: P -> I-node; s.links: (T, P) pairs whose type is asked
"""
from collections import defaultdict
from dataclasses import dataclass

from common.aif import view
from common.positions import place
from common.text import norm


@dataclass
class Serialised:
    name: str
    text: str
    prop: dict[str, str]              # "P3" -> I-node id, in P order
    turn: dict[str, str]              # "T2" -> L-node id, in dialogue order
    pid: dict[str, str]               # I-node id -> "P3"
    links: list[tuple[str, str]]      # (T, P) from the given links, in dialogue order: the turns to type
    pos: dict[str, int]               # I-node id -> dialogue position (the direction rule and assembly need it)


def serialise(aif: dict, name: str, links) -> Serialised:
    v = view(aif, name)
    links = [tuple(x[:2]) for x in links]
    pos, _ = place(v, links)
    order = sorted(v.props, key=lambda p: (pos.get(p, float("inf")), norm(v.props[p])))
    pid = {p: f"P{k + 1}" for k, p in enumerate(order)}
    tid = {lid: f"T{k + 1}" for k, lid in enumerate(v.dialogue)}
    under = defaultdict(list)
    for p in order:
        if p in pos:
            under[pos[p]].append(p)
    lines = []
    for k, lid in enumerate(v.dialogue):
        loc = v.locs[lid]
        lines.append(f"{tid[lid]}  {loc.speaker + ': ' if loc.speaker else ''}{loc.body}")
        lines += [f"    [{pid[p]}] {v.props[p]}" for p in under[k]]
    loose = [p for p in order if p not in pos]
    if loose:
        lines.append("Propositions not placed at any turn:")
        lines += [f"    [{pid[p]}] {v.props[p]}" for p in loose]
    shown = sorted({(tid[t], pid[p]) for t, p in links if t in tid and p in pid},
                   key=lambda x: (int(x[0][1:]), int(x[1][1:])))
    return Serialised(name, "\n".join(lines), {i: p for p, i in pid.items()}, {i: l for l, i in tid.items()},
                      pid, shown, pos)
