"""Prompts: the fixed system prompt, worked examples, and the per-map user message.

Every map, target or worked example, is rendered in test format: serialised with no links, so
propositions are placed only by text similarity (common/positions.py) and the rest are listed as
unplaced. Worked examples are retrieved exactly as in A2 (a2_llm.prompt.Pool: TF-IDF, most similar
shown last) and followed by their gold answer: one entry per dialogue turn.
"""
import json
from functools import cache
from pathlib import Path

from common.aif import load_map, strip_to_input, view
from common.serialise import Serialised, serialise
from common.split import load_split
from ilo_llm.schema import NONE, TYPES

PROMPTS = Path(__file__).parent / "prompts"


def system_file(version: str) -> Path:
    return PROMPTS / f"system_{version}.md"


def render(s: Serialised) -> str:
    return s.text


def gold_answer(gold_aif: dict, s: Serialised) -> dict:
    """One entry per dialogue turn: its first gold link (view(gold).anchors(), dialogue turns only),
    else "none" typed Asserting. Used for worked examples and the check.py round trip."""
    v = view(gold_aif, s.name)
    first = {}
    for t, p, y in v.anchors():
        if t in v.turn:
            first.setdefault(t, (p, y))
    out = []
    for tid, lid in s.turn.items():                  # dialogue order
        p, y = first.get(lid, (None, "Asserting"))
        assert y in TYPES, (s.name, lid, y)
        out.append({"turn": tid, "proposition": s.pid[p] if p else NONE, "type": y})
    return {"illocutions": out}


@cache
def _train_paths() -> dict:
    return {p.stem: p for p in load_split("train")}


def example(name: str, gold_aif: dict | None = None) -> str:
    """A training map rendered as the target is (no links), then its gold answer."""
    g = gold_aif if gold_aif is not None else load_map(_train_paths()[name])
    s = serialise(strip_to_input(g), name, [])
    return f"{render(s)}\n\nAnswer:\n{json.dumps(gold_answer(g, s), ensure_ascii=False)}"


def user_message(s: Serialised, examples: list[str]) -> str:
    """examples: rendered worked examples, most similar first; they are shown most similar last."""
    parts = []
    if examples:
        parts.append(f"Below are {len(examples)} maps from the same corpus with the annotators' answers, then the map to annotate.")
        parts += [f"# Example {k + 1}\n{e}" for k, e in enumerate(reversed(examples))]
    parts.append(f"# Map to annotate\n{render(s)}")
    return "\n\n".join(parts)
