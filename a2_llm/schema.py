"""The per-map answer schema (passed to `claude -p --json-schema`) and a strict local check of an answer.

Allowed proposition IDs and turn links are this map's own enums, so an invalid ID can't be written.
Illocutions come first in the answer: typing the turns before relating propositions mirrors the
2024 finding that illocutions-first helped.
"""
RELATIONS = ["Inference", "Conflict", "Rephrase"]
TURN_TYPES = ["Asserting", "Pure Questioning", "Assertive Questioning", "Rhetorical Questioning",
              "Default Illocuting", "Agreeing", "Challenging"]
CONFIDENCE = {"low": 1 / 3, "medium": 2 / 3, "high": 1.0}


def link_id(t: str, p: str) -> str:
    return f"{t}-{p}"


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


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
            "to": {"type": "string", "enum": pids},
            "confidence": {"type": "string", "enum": list(CONFIDENCE)}})},
    })


def problems(answer, s) -> list[str]:
    """Everything wrong with an answer, checked against the schema by hand (no jsonschema dependency)."""
    sch = schema(s)
    if not isinstance(answer, dict) or set(answer) != set(sch["required"]):
        return [f"top-level keys {sorted(answer) if isinstance(answer, dict) else type(answer).__name__}"]
    out = []
    for key in sch["required"]:
        items = answer[key]
        spec = sch["properties"][key]["items"]
        if not isinstance(items, list):
            out.append(f"{key} is not a list")
            continue
        for k, it in enumerate(items):
            if not isinstance(it, dict) or set(it) != set(spec["required"]):
                out.append(f"{key}[{k}] keys")
                continue
            out += [f"{key}[{k}].{f}={it[f]!r}" for f, rule in spec["properties"].items() if it[f] not in rule["enum"]]
    return out
