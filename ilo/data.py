"""Step 3.2: model inputs and labels for the ILO labeller, cached as JSON lines in cache/ilo/.

Each candidate pair (ilo.candidates) becomes one example:

    text_a = "Speaker: turn"
    text_b = "proposition [SEP] before: S: turn / S: turn [SEP] after: S: turn / S: turn"

with up to two dialogue turns on each side as context. The label is the gold illocution on that
(turn, proposition) pair, or None. Dev examples are built from dev maps stripped to test-input
format (D004); their labels come from gold and are used only for scoring and tuning.
Maps with no L->I links at all (the late re-annotation batches) are skipped (D030).

    python -m ilo.data          # builds cache/ilo/{train,dev}.jsonl, writes results/ilo/dataset.json
"""
import json
import statistics as st
from collections import Counter

from common import RESULTS, ROOT
from common.aif import load_map, strip_to_input, view
from common.split import load_split
from common.timing import timed
from ilo.candidates import NEXT, OWN, PREV, candidates

CACHE = ROOT / "cache" / "ilo"
LABELS = ["None", "Asserting", "Pure Questioning", "Assertive Questioning", "Rhetorical Questioning",
          "Default Illocuting", "Agreeing", "Challenging", "Disagreeing", "Arguing", "Restating"]
CONTEXT = 2          # turns on each side
TOKENIZER = "answerdotai/ModernBERT-base"


def turn_text(v, lid):
    l = v.locs[lid]
    return f"{l.speaker}: {l.body}" if l.speaker else l.body


def inputs(v, name: str, sep: str) -> list[dict]:
    """Model inputs for every candidate pair of one map (no labels: works on test inputs too)."""
    pos = {lid: k for k, lid in enumerate(v.dialogue)}
    out = []
    for c in candidates(v):
        k = pos[c["turn"]]
        before = " / ".join(turn_text(v, l) for l in v.dialogue[max(0, k - CONTEXT):k]) or "-"
        after = " / ".join(turn_text(v, l) for l in v.dialogue[k + 1:k + 1 + CONTEXT]) or "-"
        out.append({"map": name, **c, "text_a": turn_text(v, c["turn"]),
                    "text_b": f"{v.props[c['prop']]} {sep} before: {before} {sep} after: {after}"})
    return out


def examples(gold_aif: dict, name: str, sep: str, test_format: bool) -> list[dict] | None:
    """All candidate examples of one map, with gold labels. With test_format, inputs come from the
    stripped map. None if the map has no L->I links at all: its illocution layer was never annotated (D030)."""
    gold = view(gold_aif, name)
    if not gold.anchors():
        return None
    v = view(strip_to_input(gold_aif), name) if test_format else gold
    assert v.dialogue == gold.dialogue and list(v.props) == list(gold.props), name
    label = {}
    for s, t, y in gold.anchors():
        label.setdefault((s, t), y)                       # first occurrence wins, as in the scorer
    out = []
    for r in inputs(v, name, sep):
        y = label.get((r["turn"], r["prop"]), "None")
        out.append({**r, "label": y, "label_id": LABELS.index(y) if y in LABELS else -1})
    return out


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    CACHE.mkdir(parents=True, exist_ok=True)
    report = {"candidates": {"own": OWN, "prev": PREV, "next": NEXT}, "context_turns": CONTEXT, "labels": LABELS}
    with timed("ilo", "3.2_dataset") as row:
        for split, test_format in (("train", False), ("dev", True)):
            rows, skipped = [], []
            for p in load_split(split):
                ex = examples(load_map(p), p.stem, tok.sep_token, test_format)
                if ex is None:
                    skipped.append(p.stem)
                else:
                    rows += ex
            with open(CACHE / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            enc = tok([r["text_a"] for r in rows], [r["text_b"] for r in rows])
            lengths = sorted(len(ids) for ids in enc["input_ids"])
            q = lambda x: lengths[min(len(lengths) - 1, int(x * len(lengths)))]
            labels = Counter(r["label"] for r in rows)
            report[split] = {
                "maps": len({r["map"] for r in rows}), "skipped_maps_without_ilo": skipped, "examples": len(rows),
                "positive_share": round(1 - labels["None"] / len(rows), 4),
                "labels": dict(labels.most_common()),
                "unknown_labels": sum(r["label_id"] < 0 for r in rows),
                "tokens": {"p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "p999": q(0.999), "max": lengths[-1],
                           "mean": round(st.mean(lengths), 1), "over_256": sum(x > 256 for x in lengths),
                           "over_384": sum(x > 384 for x in lengths)},
            }
            row[split] = len(rows)
    out = RESULTS / "ilo"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "dataset.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
