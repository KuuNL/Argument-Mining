"""The per-map answer schema (passed to `claude -p --json-schema`) and a strict local check of an answer.

One entry per turn: the proposition it expresses ("none" if none) and its type, one of the labeller's
10 types (ilo.data.LABELS without None). Turn and proposition IDs are this map's own enums, so an
invalid ID can't be written.
"""
from ilo.data import LABELS

TYPES = LABELS[1:]
NONE = "none"


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def schema(s) -> dict:
    """s: a common.serialise.Serialised map."""
    return _obj({
        "illocutions": {"type": "array", "items": _obj({
            "turn": {"type": "string", "enum": list(s.turn)},
            "proposition": {"type": "string", "enum": list(s.prop) + [NONE]},
            "type": {"type": "string", "enum": TYPES}})},
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
