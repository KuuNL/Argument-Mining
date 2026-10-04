"""A2 prompts: the fixed system prompt, worked examples, and the per-map user message.

Worked examples are training maps with their gold answers, chosen by TF-IDF cosine similarity of
the maps' turn and proposition text (deterministic; ties by name). The pool holds test-like training
maps (at most GIANT propositions, D022) that have an illocution layer (D030), minus any excluded
episodes. The most similar example is placed last, nearest the target.
"""
import json
from pathlib import Path

from common.aif import load_map, view
from common.score import ari_labels
from common.serialise import Serialised, serialise
from common.split import load_split
from a2_llm.schema import RELATIONS, TURN_TYPES, link_id

PROMPTS = Path(__file__).parent / "prompts"
GIANT = 40


def system_file(version: str) -> Path:
    return PROMPTS / f"system_{version}.md"


def render(s: Serialised) -> str:
    links = ", ".join(link_id(t, p) for t, p in s.links) or "none"
    return f"{s.text}\n\nLinks to type: {links}"


def gold_answer(gold_aif: dict, s: Serialised) -> dict:
    """A map's gold annotation in the answer format: worked examples, and the A2.1 round trip."""
    v = view(gold_aif, s.name)
    rel = ari_labels(gold_aif, set(v.props))
    pairs = [(a, b, c) for (a, b), c in rel.items() if a != b and a in s.pid and b in s.pid]
    pnum = lambda node: int(s.pid[node][1:])
    relations = [{"type": RELATIONS[c], "from": s.pid[a], "to": s.pid[b], "confidence": "high"}
                 for a, b, c in sorted(pairs, key=lambda x: (pnum(x[0]), pnum(x[1])))]
    types = {}
    for t, p, y in v.anchors():
        types.setdefault((t, p), y)                      # first occurrence wins, as in the scorer
    illocutions = [{"link": link_id(t, p), "type": types[(s.turn[t], s.prop[p])]}
                   for t, p in s.links if types.get((s.turn[t], s.prop[p])) in TURN_TYPES]
    return {"illocutions": illocutions, "relations": relations}


def map_text(aif: dict, name: str) -> str:
    v = view(aif, name)
    return " ".join([v.locs[l].body for l in v.dialogue] + list(v.props.values()))


class Pool:
    """Training maps available as worked examples, with a TF-IDF index over their text."""

    def __init__(self, exclude: set[str] = frozenset()):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.maps = {}
        for p in load_split("train"):
            if p.stem in exclude:
                continue
            g = load_map(p)
            v = view(g, p.stem)
            if len(v.props) <= GIANT and v.anchors():
                self.maps[p.stem] = g
        self.names = sorted(self.maps)
        self.vec = TfidfVectorizer(sublinear_tf=True, min_df=2)
        self.X = self.vec.fit_transform([map_text(self.maps[n], n) for n in self.names])

    def nearest(self, aif: dict, name: str, k: int) -> list[str]:
        """The k most similar pool maps, most similar first (never the map itself)."""
        if k == 0:
            return []
        sims = (self.X @ self.vec.transform([map_text(aif, name)]).T).toarray().ravel()
        ranked = sorted(range(len(self.names)), key=lambda i: (-sims[i], self.names[i]))
        return [self.names[i] for i in ranked if self.names[i] != name][:k]

    def example(self, name: str) -> str:
        g = self.maps[name]
        s = serialise(g, name, view(g, name).anchors())
        return f"{render(s)}\n\nAnswer:\n{json.dumps(gold_answer(g, s), ensure_ascii=False)}"


def user_message(s: Serialised, examples: list[str]) -> str:
    """examples: rendered worked examples, most similar first; they are shown most similar last."""
    parts = []
    if examples:
        parts.append(f"Below are {len(examples)} maps from the same corpus with the annotators' answers, then the map to annotate.")
        parts += [f"# Example {k + 1}\n{e}" for k, e in enumerate(reversed(examples))]
    parts.append(f"# Map to annotate\n{render(s)}")
    return "\n\n".join(parts)
