"""A3's answer format: A2's per-map schema without the confidence field, written as compact JSON.

The model is trained on, and constrained to, exactly this text: keys in a fixed order, separators "," and ":",
no whitespace (D065). Allowed links and P-ids are this map's own enums, as in A2 (a2_llm/schema.py).
"""
import json

from a2_llm.schema import RELATIONS, TURN_TYPES, _obj, link_id

SEPARATORS = (",", ":")


def schema(s) -> dict:
    """s: a common.serialise.Serialised map."""
    pids = list(s.prop)
    links = [link_id(t, p) for t, p in s.links] or ["none"]      # a map with no links still needs a valid enum
    return _obj({
        "illocutions": {"type": "array", "items": _obj({
            "link": {"type": "string", "enum": links},
            "type": {"type": "string", "enum": TURN_TYPES}})},
        "relations": {"type": "array", "items": _obj({
            "type": {"type": "string", "enum": RELATIONS},
            "from": {"type": "string", "enum": pids},
            "to": {"type": "string", "enum": pids}})},
    })


def compact(answer: dict) -> dict:
    """An A2-format answer (with confidence) -> A3's (without), keys in schema order."""
    return {"illocutions": [{"link": i["link"], "type": i["type"]} for i in answer["illocutions"]],
            "relations": [{"type": r["type"], "from": r["from"], "to": r["to"]} for r in answer["relations"]]}


def dumps(answer: dict) -> str:
    return json.dumps(answer, ensure_ascii=False, separators=SEPARATORS)


def problems(answer, s) -> list[str]:
    """Everything wrong with an answer, checked against the schema by hand."""
    sch = schema(s)
    if not isinstance(answer, dict) or list(answer) != sch["required"]:
        return [f"top-level keys {list(answer) if isinstance(answer, dict) else type(answer).__name__}"]
    out = []
    for key in sch["required"]:
        spec = sch["properties"][key]["items"]
        if not isinstance(answer[key], list):
            out.append(f"{key} is not a list")
            continue
        for k, it in enumerate(answer[key]):
            if not isinstance(it, dict) or list(it) != spec["required"]:
                out.append(f"{key}[{k}] keys")
                continue
            out += [f"{key}[{k}].{f}={it[f]!r}" for f, rule in spec["properties"].items() if it[f] not in rule["enum"]]
    return out
