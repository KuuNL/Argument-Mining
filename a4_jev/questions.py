"""A4 state and questions for one map (D073).

State: {"guide": the IAT definitions and the annotators' habits (prompts/guide_<v>.md), "examples": k worked
training maps (optional), "dialogue": the serialised map (common/serialise.py: T1... turns, P1... propositions,
no node IDs, D029)}.

Questions (all `choice`), each asked in two option orders, forward ("_f") and reversed ("_r"), against
Jev's first-option bias:
- pair_<A>_<B>: one per candidate pair (A before B in P order, which is A1's pair order), with A1's five labels:
  A supports B, B supports A, conflict, rephrase, none. Conflict and Rephrase run later -> earlier (D020).
- from_<P>: one per proposition with candidates: "none", or one (target, relation) option per candidate target:
  supports_<Q> for every candidate Q, plus restates_<Q> and attacks_<Q> when Q comes before P. The targets
  compete inside one distribution (A2 discussion, Q1).
- type_<T>_<P>: one per link from the labeller (gold on training maps): the seven scored turn types. Not asked
  since D076 (the labeller keeps Task B); kept for kinds=(..., "type").
Candidates are A1's: pairs at most K turns apart, or with an unplaced proposition (D018, D041).

Option descriptions are short (`defs="short"`: the definitions live once in the guide) or carry the full
definition in every option (`defs="full"`), an A4.2 pilot arm. Requests are split so that each stays under
REQUEST_BUDGET estimated tokens (Jev allows 64k per request; 32k for the state plus the longest question).

    b = build(inp, name, links, defs="short", k=0)   # b.requests: [{"state", "questions"}]; b.meta: qid -> meaning
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from common.aif import view
from common.serialise import Serialised, serialise
from a2_llm.schema import TURN_TYPES

PROMPTS = Path(__file__).parent / "prompts"
K = 10                         # candidate window in turns, as A1 (D041)
CHARS_PER_TOKEN = 3.0          # A4.1 measured 3.07 characters per Jev token (D076); 3.0 keeps a margin
REQUEST_BUDGET = 50_000        # estimated tokens per request: <= ~58k real at the densest ratio seen (2.69), under 64k
STATE_BUDGET = 30_000          # state + longest question, under the 32k limit
ORDERS = ("f", "r")
VERB = {"Inference": "supports", "Conflict": "attacks", "Rephrase": "restates"}

TYPE_DEF = {
    "Asserting": "states it as true",
    "Pure Questioning": "a genuine question, with no commitment to an answer",
    "Assertive Questioning": "question form, but the speaker also conveys their own position, or steers the discussion",
    "Rhetorical Questioning": "a question asked to make a point, not to get an answer",
    "Default Illocuting": "fits none of the other types",
    "Agreeing": "agrees with an earlier claim; the proposition is the claim agreed with",
    "Challenging": "challenges an earlier claim and demands a justification",
}
FULL = {   # the guide's definitions, repeated in each option when defs="full"
    "supports": "it is a reason, evidence or explanation for it, and a different point from it",
    "restates": "it makes the same point again: restates, reformulates, narrows or sharpens it, or answers the question",
    "attacks": "it contradicts, rebuts or undermines it, or both can't be held; this includes a speaker's own concession",
    "none": "no direct relation; being on the same topic is not enough",
}


def guide(version: str = "v1") -> str:
    return (PROMPTS / f"guide_{version}.md").read_text(encoding="utf-8").strip()


def tokens(obj) -> int:
    """Estimated Jev tokens of a JSON value."""
    return int(len(json.dumps(obj, ensure_ascii=False)) / CHARS_PER_TOKEN) + 1


@dataclass
class Built:
    s: Serialised
    texts: dict[str, str]                  # P -> proposition text
    candidates: list[tuple[str, str]]      # (A, B) P ids, A first in P order
    state: dict
    questions: dict[str, dict]             # qid -> question, in request order
    meta: dict[str, tuple]                 # qid -> (kind, order, item, {option: meaning})
    requests: list[dict] = field(default_factory=list)


def candidates(s: Serialised) -> list[tuple[str, str]]:
    pids = list(s.prop)
    out = []
    for i, a in enumerate(pids):
        for b in pids[i + 1:]:
            pa, pb = s.pos.get(s.prop[a]), s.pos.get(s.prop[b])
            if pa is not None and pb is not None and abs(pa - pb) > K:
                continue
            out.append((a, b))
    return out


def _desc(kind: str, src: str, tgt: str, text: str, defs: str) -> str:
    base = {"supports": f"{src} is a reason or evidence for {tgt}", "restates": f"{src} makes the same point as {tgt} again",
            "attacks": f"{src} attacks {tgt}"}[kind]
    return (f"{base} ({FULL[kind]})" if defs == "full" else base) + (f': "{text}"' if text else "")


def _pair_options(a: str, b: str, defs: str) -> list[tuple[str, str, int]]:
    """(name, description, A1 label index) in forward order."""
    full = defs == "full"
    return [(f"{a}_supports_{b}", _desc("supports", a, b, "", defs), 1),
            (f"{b}_supports_{a}", _desc("supports", b, a, "", defs), 2),
            ("conflict", _desc("attacks", b, a, "", defs), 3),
            ("rephrase", _desc("restates", b, a, "", defs) + (", or answers it" if not full else ""), 4),
            ("none", "no direct relation" + (f" ({FULL['none']})" if full else ""), 0)]


def _source_options(p: str, targets: list[str], earlier: set[str], texts: dict, defs: str) -> list[tuple[str, str, tuple]]:
    """(name, description, (relation, target)) in forward order: targets in P order, "none" last."""
    out = []
    for q in targets:
        out.append((f"supports_{q}", _desc("supports", p, q, texts[q], defs), ("Inference", q)))
        if q in earlier:
            out.append((f"restates_{q}", _desc("restates", p, q, texts[q], defs), ("Rephrase", q)))
            out.append((f"attacks_{q}", _desc("attacks", p, q, texts[q], defs), ("Conflict", q)))
    out.append(("none", f"{p} starts no relation" + (f" ({FULL['none']})" if defs == "full" else ""), None))
    return out


def _example(pool, name: str, n: int) -> str:
    """A worked training map with its gold relations; its IDs are prefixed E<n>. so they can't be confused with the target's."""
    from a2_llm.prompt import gold_answer
    g = pool.maps[name]
    s = serialise(g, name, view(g, name).anchors())
    text = re.sub(r"(?m)^T(\d+)", rf"E{n}.T\1", s.text)
    text = re.sub(r"\[P(\d+)\]", rf"[E{n}.P\1]", text)
    rel = [f"E{n}.{r['from']} {VERB[r['type']]} E{n}.{r['to']}" for r in gold_answer(g, s)["relations"]]
    return f"{text}\nAnnotated relations: {'; '.join(rel) if rel else 'none'}"


SOURCE_ASK = {   # the source question's instruction; v2 is A4.2's one wording fix (D078)
    "v1": lambda p, text: f'Which one proposition does {p} ("{text}") most directly bear on, and how? Choose none if it starts no relation.',
    "v2": lambda p, text: (f'Which one proposition does {p} ("{text}") support (as a reason or evidence for it), restate, '
                           f'or attack? Choose none if {p} does none of these.'),
}


def build(inp: dict, name: str, links, defs: str = "short", k: int = 0, pool=None, version: str = "v1", wording: str = "v1",
          kinds=("pair", "source"), orders=ORDERS, budget: int = REQUEST_BUDGET) -> Built:
    v = view(inp, name)
    s = serialise(inp, name, links)
    texts = {p: v.props[i] for p, i in s.prop.items()}
    state = {"guide": guide(version)}
    if k:
        state["examples"] = [_example(pool, e, n + 1) for n, e in enumerate(pool.nearest(inp, name, k))]
    state["dialogue"] = s.text
    cand = candidates(s)
    order = {p: n for n, p in enumerate(s.prop)}
    items = []                                     # (qid stem, kind, item, instructions, [(option, desc, meaning)])
    if "pair" in kinds:
        for a, b in cand:
            items.append((f"pair_{a}_{b}", "pair", (a, b),
                          f'How is {a} ("{texts[a]}") related to {b} ("{texts[b]}")?', _pair_options(a, b, defs)))
    if "source" in kinds:
        nbrs = {p: [] for p in s.prop}
        for a, b in cand:
            nbrs[a].append(b)
            nbrs[b].append(a)
        for p, qs in nbrs.items():
            if qs:
                qs = sorted(qs, key=order.get)
                items.append((f"from_{p}", "source", p, SOURCE_ASK[wording](p, texts[p]),
                              _source_options(p, qs, {q for q in qs if order[q] < order[p]}, texts, defs)))
    if "type" in kinds:
        for t, p in s.links:
            loc = v.locs[s.turn[t]]
            said = f"{loc.speaker}: {loc.body}" if loc.speaker else loc.body
            items.append((f"type_{t}_{p}", "type", (t, p), f'How does turn {t} ("{said}") put forward {p} ("{texts[p]}")?',
                          [(y, TYPE_DEF[y], y) for y in TURN_TYPES]))
    questions, meta = {}, {}
    for stem, kind, item, instr, opts in items:
        for o in orders:
            seq = opts if o == "f" else opts[::-1]
            qid = f"{stem}_{o}"
            questions[qid] = {"type": "choice", "instructions": instr, "criteria": {n: d for n, d, _ in seq}}
            meta[qid] = (kind, o, item, {n: m for n, _, m in opts})
    b = Built(s, texts, cand, state, questions, meta)
    b.requests = split(state, questions, budget)
    return b


def split(state: dict, questions: dict[str, dict], budget: int = REQUEST_BUDGET) -> list[dict]:
    """Questions in order, packed greedily into requests that share the state and stay under the budgets."""
    base = tokens(state)
    longest = max((tokens(q) for q in questions.values()), default=0)
    if base + longest > STATE_BUDGET:
        raise ValueError(f"state + longest question ≈ {base + longest} tokens > {STATE_BUDGET}")
    out, cur, size = [], {}, base
    for qid, q in questions.items():
        n = tokens(q) + tokens(qid)
        if cur and size + n > budget:
            out.append({"state": state, "questions": cur})
            cur, size = {}, base
        cur[qid] = q
        size += n
    if cur:
        out.append({"state": state, "questions": cur})
    return out
