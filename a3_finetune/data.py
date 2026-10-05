"""A3 examples: (fixed instruction, serialised map) -> compact gold answer; giant maps as chunks.

Maps are written exactly as A2 writes them (common/serialise.py, a2_llm/prompt.render). Training maps use
their gold turn links (D040); dev and test maps the ILO labeller's (cache/ilo/links_*.json), in test-input
format. Test-like maps have at most GIANT (40) propositions (D022). The 19 giant training maps have no turn
links (D030): they are cut into windows of consecutive turns holding at most CHUNK propositions (the largest
test map has 35), placed by text similarity, and a relation is kept when both its ends fall in the window.
They are used only by the giant-map retrain ablation (D065).

    ex = train_examples()          # 1,231 Examples: ex.user (map text), ex.answer (compact JSON)
    ids, labels = encode(tok, ex)  # chat-formatted prompt + answer; labels -100 on the prompt
"""
import json
from dataclasses import dataclass, field

from common.aif import load_map, strip_to_input, test_paths, view
from common.positions import place
from common.score import ari_labels
from common.serialise import Serialised, serialise
from common.split import load_split
from a2_llm.check import dev_maps, links_of
from a2_llm.prompt import GIANT, gold_answer, render
from a3_finetune.schema import compact, dumps

CHUNK = 35
END = "<|im_end|>"
INSTRUCTION = """You annotate debate maps from BBC Question Time in the style of the QT30 corpus (Inference Anchoring Theory).
The map lists the turns T1, T2, ... in dialogue order as "Speaker: words"; under each turn are the propositions [P1], [P2], ... it expresses.
Answer with JSON only:
- "illocutions": for each link in "Links to type", how the turn expresses the proposition.
- "relations": every relation between propositions that an annotator would draw: Inference (from a premise to the conclusion it supports), Conflict (from a proposition to one it attacks) or Rephrase (from a proposition to one it restates or answers)."""


@dataclass
class Example:
    name: str
    s: Serialised
    user: str                        # the map as text, with the links to type
    answer: str | None               # compact gold answer (None for test inputs)
    inp: dict | None = None          # test-format input map (dev / test): what build_map assembles on
    links: list = field(default_factory=list)


def _example(gold: dict | None, inp: dict, name: str, links) -> Example:
    s = serialise(inp, name, links)
    answer = dumps(compact(gold_answer(gold, s))) if gold is not None else None
    return Example(name, s, render(s), answer, inp, list(links))


def train_examples() -> list[Example]:
    out = []
    for p in load_split("train"):
        g = load_map(p)
        v = view(g, p.stem)
        if len(v.props) <= GIANT:
            out.append(_example(g, g, p.stem, v.anchors()))
    return out


def dev_examples() -> list[Example]:
    """The 225 test-like dev maps in test-input format, with the labeller's links and the gold answer."""
    return [_example(gold, inp, name, links) for name, gold, inp, links in dev_maps()]


def test_examples() -> list[Example]:
    links = links_of("test")
    return [_example(None, load_map(ip), name, links.get(name, [])) for name, ip, _ in test_paths()]


def _window(aif: dict, turns: set, props: set) -> dict:
    """The sub-map of a window: its turns, propositions, the transitions between its turns, and the
    relations whose target (first outgoing edge, as the scorer reads it) and at least one premise are inside."""
    types = {n["nodeID"]: n["type"] for n in aif["nodes"]}
    out_e, in_e = {}, {}
    for e in aif["edges"]:
        out_e.setdefault(e["fromID"], []).append(e["toID"])
        in_e.setdefault(e["toID"], []).append(e["fromID"])
    keep = set(turns) | set(props)
    for nid, t in types.items():
        if t == "TA":
            ends = in_e.get(nid, []) + out_e.get(nid, [])
            if in_e.get(nid) and out_e.get(nid) and all(x in turns for x in ends if types.get(x) == "L"):
                keep.add(nid)
        elif t in ("RA", "CA", "MA") and out_e.get(nid):
            if out_e[nid][0] in props and any(x in props for x in in_e.get(nid, [])):
                keep.add(nid)
    return {"nodes": [n for n in aif["nodes"] if n["nodeID"] in keep],
            "edges": [e for e in aif["edges"] if e["fromID"] in keep and e["toID"] in keep],
            "locutions": [l for l in aif.get("locutions", []) if str(l.get("nodeID")) in turns]}


def giant_chunks() -> tuple[list[Example], dict]:
    """Windows of the giant training maps, and what they lose: relations with an unplaced end, or across a window edge."""
    out, stats = [], {"maps": 0, "props": 0, "unplaced_props": 0, "relations": 0, "kept": 0,
                      "lost_unplaced": 0, "lost_edge": 0}
    for p in load_split("train"):
        g = load_map(p)
        v = view(g, p.stem)
        if len(v.props) <= GIANT:
            continue
        pos, _ = place(v, [])
        gold = {k: c for k, c in ari_labels(g, set(v.props)).items() if k[0] != k[1]}
        stats["maps"] += 1
        stats["props"] += len(v.props)
        stats["unplaced_props"] += len(v.props) - len(pos)
        stats["relations"] += len(gold)
        at = {}
        for prop, k in pos.items():
            at.setdefault(k, []).append(prop)
        windows, cur, n = [], [], 0
        for k in range(len(v.dialogue)):
            m = len(at.get(k, []))
            if cur and n + m > CHUNK:
                windows.append(cur)
                cur, n = [], 0
            cur.append(k)
            n += m
        if cur:
            windows.append(cur)
        which = {prop: w for w, ks in enumerate(windows) for k in ks for prop in at.get(k, [])}
        for a, b in gold:
            if a not in which or b not in which:
                stats["lost_unplaced"] += 1
            elif which[a] != which[b]:
                stats["lost_edge"] += 1
        for w, ks in enumerate(windows):
            props = {prop for k in ks for prop in at.get(k, [])}
            if not props:
                continue
            sub = _window(g, {v.dialogue[k] for k in ks}, props)
            ex = _example(sub, sub, f"{p.stem}#c{w}", [])
            stats["kept"] += len(json.loads(ex.answer)["relations"])
            out.append(ex)
    stats["chunks"] = len(out)
    return out, stats


def messages(user: str) -> list[dict]:
    return [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": user}]


def prompt(tok, user: str) -> str:
    """The chat-formatted prompt, ending where the answer starts. Thinking is off (Qwen3's template)."""
    return tok.apply_chat_template(messages(user), tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode(tok, ex: Example) -> tuple[list[int], list[int]]:
    """Token ids of prompt + answer + end-of-turn, and labels that score the answer only."""
    p = tok(prompt(tok, ex.user), add_special_tokens=False)["input_ids"]
    a = tok(ex.answer + END, add_special_tokens=False)["input_ids"]
    return p + a, [-100] * len(p) + a
