"""Well-formedness checks for AIF maps (PLAN: Foundation).

`validate` returns a list of problems. Our own predicted maps must have none. Gold maps
contain a few irregular relation nodes (about 1% of S-nodes), which are reported, not fixed.
"""
from collections import defaultdict

NODE_TYPES = {"I", "L", "TA", "YA", "RA", "CA", "MA"}
YA_LABELS = {
    "Asserting", "Pure Questioning", "Assertive Questioning", "Rhetorical Questioning",
    "Agreeing", "Disagreeing", "Challenging", "Arguing", "Restating", "Default Illocuting", "Analysing",
}
S_TEXT = {"RA": "Default Inference", "CA": "Default Conflict", "MA": "Default Rephrase", "TA": "Default Transition"}


def validate(aif: dict) -> list[str]:
    problems = []
    types = {}
    for n in aif["nodes"]:
        if n["nodeID"] in types:
            problems.append(f"duplicate node {n['nodeID']}")
        types[n["nodeID"]] = n["type"]
        if n["type"] not in NODE_TYPES:
            problems.append(f"unknown node type {n['type']} ({n['nodeID']})")
        elif n["type"] == "YA" and n["text"] not in YA_LABELS:
            problems.append(f"unknown YA label {n['text']!r} ({n['nodeID']})")
        elif n["type"] in S_TEXT and n["text"] != S_TEXT[n["type"]]:
            problems.append(f"{n['type']} text {n['text']!r} ({n['nodeID']})")

    out_e, in_e, edge_ids = defaultdict(list), defaultdict(list), set()
    for e in aif["edges"]:
        if e.get("edgeID") is not None:
            if e["edgeID"] in edge_ids:
                problems.append(f"duplicate edge {e['edgeID']}")
            edge_ids.add(e["edgeID"])
        if e["fromID"] not in types or e["toID"] not in types:
            problems.append(f"dangling edge {e['fromID']} -> {e['toID']}")
            continue
        out_e[e["fromID"]].append(types[e["toID"]])
        in_e[e["toID"]].append(types[e["fromID"]])

    for nid, t in types.items():
        if t in ("RA", "CA", "MA"):
            if "I" not in in_e[nid]:
                problems.append(f"{t} without an incoming I-node ({nid})")
            if out_e[nid].count("I") != 1 or len(out_e[nid]) != 1:
                problems.append(f"{t} needs exactly one outgoing edge, to an I-node; has {out_e[nid]} ({nid})")
        elif t == "YA":
            if len(out_e[nid]) != 1:
                problems.append(f"YA with {len(out_e[nid])} outgoing edges ({nid})")
            if not in_e[nid]:
                problems.append(f"YA without a source ({nid})")
    return problems


def problem_kind(problem: str) -> str:
    """Strip ids so problems can be counted by kind."""
    return problem.rsplit(" (", 1)[0] if problem.endswith(")") else problem.split(" ")[0] + " …"
