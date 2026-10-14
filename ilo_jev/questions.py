"""Jev state and questions for Task B on one map (D099).

State: {"guide": prompts/guide_<v>.md, "examples": k worked training maps (optional), "dialogue": the map
serialised with no links (common/serialise.py; no node IDs, D029)}.

Questions (all `choice`), each asked in two option orders, forward ("_f") and reversed ("_r"), as A4 (D076):
- link_<T>: which proposition does turn T express? One option per proposition (its text), plus "none".
- type_<T>: how does turn T put its proposition forward? The labeller's 10 types (ilo.data.LABELS[1:]).
Requests are packed by a4_jev.questions.split under A4's token budgets.

    b = build(inp, name, k=0, pool=None)   # b.requests: [{"state", "questions"}]; b.meta: qid -> (kind, order, T, {option: meaning})
"""
import re
from pathlib import Path

from common.aif import strip_to_input, view
from common.serialise import serialise
from a4_jev.questions import ORDERS, REQUEST_BUDGET, Built, split
from ilo_llm.prompt import gold_answer
from ilo_llm.schema import NONE, TYPES

PROMPTS = Path(__file__).parent / "prompts"
TYPE_DEF = {   # a4_jev's seven, plus the three added types in D091's spec-only terms
    "Asserting": "states it as true",
    "Pure Questioning": "a genuine question, with no commitment to an answer",
    "Assertive Questioning": "question form, but the speaker also conveys their own position",
    "Rhetorical Questioning": "a question asked to make a point, not to get an answer",
    "Default Illocuting": "fits none of the other types",
    "Agreeing": "agrees with an earlier claim; the proposition is the claim agreed with",
    "Challenging": "challenges an earlier claim and demands a justification",
    "Disagreeing": "disagrees with an earlier claim; the proposition is the claim disagreed with",
    "Arguing": "very rare on a turn: normally marks the step that creates an inference",
    "Restating": "very rare on a turn: normally marks the step that creates a rephrase",
}
assert list(TYPE_DEF) == TYPES


def guide(version: str = "v1") -> str:
    return (PROMPTS / f"guide_{version}.md").read_text(encoding="utf-8").strip()


def example(pool, name: str, n: int) -> str:
    """A worked training map rendered as the target (no links) with its gold answer; IDs prefixed E<n>."""
    g = pool.maps[name]
    s = serialise(strip_to_input(g), name, [])
    text = re.sub(r"(?m)^T(\d+)", rf"E{n}.T\1", s.text)
    text = re.sub(r"\[P(\d+)\]", rf"[E{n}.P\1]", text)
    links = [f"E{n}.{it['turn']} expresses " + (f"E{n}.{it['proposition']} ({it['type']})" if it["proposition"] != NONE else "none")
             for it in gold_answer(g, s)["illocutions"]]
    return f"{text}\nAnnotated: {'; '.join(links)}"


def build(inp: dict, name: str, k: int = 0, pool=None, version: str = "v1", orders=ORDERS,
          budget: int = REQUEST_BUDGET) -> Built:
    v = view(inp, name)
    s = serialise(inp, name, [])
    texts = {p: v.props[i] for p, i in s.prop.items()}
    state = {"guide": guide(version)}
    if k:
        state["examples"] = [example(pool, e, n + 1) for n, e in enumerate(pool.nearest(inp, name, k))]
    state["dialogue"] = s.text
    items = []                                     # (qid stem, kind, T, instructions, [(option, description, meaning)])
    for t, lid in s.turn.items():
        loc = v.locs[lid]
        said = f"{loc.speaker}: {loc.body}" if loc.speaker else loc.body
        items.append((f"link_{t}", "link", t, f'Which proposition does turn {t} ("{said}") express? Choose none if it expresses no proposition.',
                      [(p, texts[p], p) for p in s.prop] + [("none", f"{t} expresses no proposition", None)]))
        items.append((f"type_{t}", "type", t, f'How does turn {t} ("{said}") put forward the proposition it expresses?',
                      [(y, TYPE_DEF[y], y) for y in TYPES]))
    questions, meta = {}, {}
    for stem, kind, item, instr, opts in items:
        for o in orders:
            seq = opts if o == "f" else opts[::-1]
            qid = f"{stem}_{o}"
            questions[qid] = {"type": "choice", "instructions": instr, "criteria": {n: d for n, d, _ in seq}}
            meta[qid] = (kind, o, item, {n: m for n, _, m in opts})
    b = Built(s, texts, [], state, questions, meta)
    b.requests = split(state, questions, budget)
    return b
