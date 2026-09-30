"""Load, view, strip and write AIF argument maps.

An AIF map is {"nodes", "edges", "locutions"}; evaluation gold is wrapped as {"AIF": ...}.
`MapView` is the structured view later phases work from: propositions, locutions with
speakers, dialogue order along the TA chain, relations and illocutions.
"""
import copy
import heapq
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import DATA

RELATION_TYPES = {"RA": "Inference", "CA": "Conflict", "MA": "Rephrase"}
INPUT_TYPES = {"I", "L", "TA"}  # all a test input map contains


# ---------------------------------------------------------------- files

def load_map(path) -> dict:
    """Read an AIF map, unwrapping evaluation gold's {"AIF": ...}."""
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    return m.get("AIF", m)


def save_map(aif: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(aif, f, ensure_ascii=False, indent=1)


def _num(s: str) -> int:
    return int(re.search(r"(\d+)", s).group(1))


def training_paths() -> list[Path]:
    return sorted((DATA / "training").glob("*.json"), key=lambda p: _num(p.stem))


def test_paths() -> list[tuple[str, Path, Path]]:
    """(name, input map, gold map) for the 11 test maps. Gold is for the final run only."""
    out = []
    for p in sorted((DATA / "test").glob("test_map*.json"), key=lambda p: _num(p.stem)):
        name = p.stem.removeprefix("test_")
        out.append((name, p, DATA / "evaluation" / f"{name}.json"))
    return out


def strip_to_input(aif: dict) -> dict:
    """Reduce a gold map to what a test input contains: I, L, TA nodes and the edges among them (D004)."""
    keep = [n for n in aif["nodes"] if n["type"] in INPUT_TYPES]
    ids = {n["nodeID"] for n in keep}
    return {
        "nodes": copy.deepcopy(keep),
        "edges": [dict(e) for e in aif["edges"] if e["fromID"] in ids and e["toID"] in ids],
        "locutions": copy.deepcopy(aif.get("locutions", [])),
    }


# ---------------------------------------------------------------- view

@dataclass
class Locution:
    id: str
    text: str             # full node text, "Speaker : words"
    speaker: str
    body: str
    is_copy: bool         # analyst copy ("Analyst: Speaker : words"): bookkeeping, never in the TA chain or anchored
    copy_of: str | None   # the original L-node, when the copy is exact


@dataclass
class Relation:
    id: str
    type: str             # RA / CA / MA
    premises: list[str]   # incoming I-nodes
    target: str | None    # node on the first outgoing edge (what the official scorer reads)


@dataclass
class MapView:
    name: str
    aif: dict
    props: dict[str, str]                     # I-node id -> text, in node order
    locs: dict[str, Locution]                 # every L-node, originals and analyst copies
    dialogue: list[str]                       # dialogue L-node ids (no copies) in dialogue order
    relations: list[Relation]
    illocutions: list[tuple[str, str, str]]   # (source, target, YA label), in scorer order

    @property
    def turn(self) -> dict[str, int]:
        """Dialogue position of each original L-node."""
        return {lid: k for k, lid in enumerate(self.dialogue)}

    def anchors(self) -> list[tuple[str, str, str]]:
        """Scored illocutions: (L-node, I-node, label)."""
        return [(s, t, y) for s, t, y in self.illocutions if s in self.locs and t in self.props]


def split_speaker(text: str) -> tuple[str, str]:
    """'Fiona Bruce : words' or 'Fiona Bruce: words' -> ('Fiona Bruce', 'words')."""
    head, sep, body = text.partition(":")
    return (head.strip(), body.strip()) if sep else ("", text.strip())


def _norm(s: str) -> str:
    return " ".join(s.split())


def _sort_key(node_id: str):
    return (int(node_id), node_id) if node_id.isdigit() else (float("inf"), node_id)


def view(aif: dict, name: str = "") -> MapView:
    nodes = aif["nodes"]
    out_e, in_e = defaultdict(list), defaultdict(list)   # edge order preserved
    for e in aif["edges"]:
        out_e[e["fromID"]].append(e["toID"])
        in_e[e["toID"]].append(e["fromID"])

    props = {n["nodeID"]: n["text"] for n in nodes if n["type"] == "I"}
    l_nodes = [n for n in nodes if n["type"] == "L"]

    # Analyst copies put an analyst's name before "Speaker : words". A prefix counts as an
    # analyst's when it yields an exact copy of another L-node in this map; every L-node with
    # that prefix (exact or partial copy) is then a copy (D014).
    by_text = defaultdict(list)
    for n in l_nodes:
        by_text[_norm(n["text"])].append(n["nodeID"])
    exact, analysts = {}, set()
    for n in l_nodes:
        head, _, rest = n["text"].partition(":")
        originals = [i for i in by_text.get(_norm(rest), []) if i != n["nodeID"]] if ":" in rest else []
        if originals:
            exact[n["nodeID"]] = originals[0]
            analysts.add(head.strip())
    locs = {}
    for n in l_nodes:
        head, _, rest = n["text"].partition(":")
        is_copy = n["nodeID"] in exact or (head.strip() in analysts and ":" in rest)
        speaker, body = split_speaker(rest if is_copy else n["text"])
        locs[n["nodeID"]] = Locution(n["nodeID"], n["text"], speaker, body, is_copy, exact.get(n["nodeID"]))

    # Dialogue order: topological order of L -> TA -> L, ties broken by node id.
    succ, indeg, in_ta = defaultdict(set), {i: 0 for i in locs}, set()
    for ta in (n["nodeID"] for n in nodes if n["type"] == "TA"):
        for a in in_e[ta]:
            for b in out_e[ta]:
                if a in locs and b in locs and b not in succ[a]:
                    succ[a].add(b)
                    indeg[b] += 1
                    in_ta |= {a, b}
    heap = [(_sort_key(i), i) for i, d in indeg.items() if d == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, i = heapq.heappop(heap)
        order.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(heap, (_sort_key(j), j))
    order += sorted((i for i in locs if i not in set(order)), key=_sort_key)  # cycles, if any
    # Dialogue = non-copies with a speaker or a place in the TA chain; speakerless fragments
    # outside the chain are bookkeeping too (2% anchored in training).
    dialogue = [i for i in order if not locs[i].is_copy and (locs[i].speaker or i in in_ta)]

    relations, illocutions = [], []
    for n in nodes:
        nid, t = n["nodeID"], n["type"]
        if t in RELATION_TYPES:
            relations.append(Relation(nid, t, [s for s in in_e[nid] if s in props],
                                      out_e[nid][0] if out_e[nid] else None))
        elif t == "YA" and out_e[nid]:
            illocutions += [(s, out_e[nid][0], n["text"]) for s in in_e[nid]]

    return MapView(name, aif, props, locs, dialogue, relations, illocutions)
