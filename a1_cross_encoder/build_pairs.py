"""A1.1: candidate proposition pairs, labels and model inputs (D018-D022, D029, D040).

For every map (at most GIANT propositions, D022):
1. Place each proposition in the dialogue: gold links on training maps, the ILO labeller's predicted
   links on dev and test (D040), then the most similar turn, else unplaced (common/positions.py).
2. Keep unordered pairs at most K turns apart; a pair with an unplaced proposition is always kept (D018).
3. Order each pair: A = the earlier proposition (unplaced last; ties by proposition text, D029).
4. One example per pair (D019, D021):
       text_a = proposition A
       text_b = proposition B [SEP] distance: d turns; same speaker [SEP] the dialogue, with
                [[A]] ... [[/A]] and [[B]] ... [[/B]] around their turns
   The whole dialogue is used when it fits MAX_LEN; otherwise a window around the pair.
5. Label (5 classes, D020): None, Inference A->B, Inference B->A, Conflict, Rephrase.

Gate checks: relations surviving pruning; gold labels -> pairs -> AIF scores at the ceiling and
validates; token lengths.

    python -m a1_cross_encoder.build_pairs   -> cache/a1/{train,dev,test}.jsonl, results/a1/pairs.json
"""
import json
import statistics as st
from collections import Counter

from common import RESULTS, ROOT
from common.aif import load_map, strip_to_input, test_paths, view
from common.positions import place, speakers
from common.score import ari_labels, score_map
from common.split import load_split
from common.text import norm
from common.timing import timed
from common.validate import validate
from a1_cross_encoder.assemble import LABELS, relation, with_relations

K, MAX_LEN, GIANT, MARGIN = 10, 1024, 40, 16    # K = 10 since D041 (was 8, D018)
CACHE = ROOT / "cache" / "a1"
OUT = RESULTS / "a1"
TOKENIZER = "answerdotai/ModernBERT-base"


def turn_text(v, lid):
    l = v.locs[lid]
    return f"{l.speaker}: {l.body}" if l.speaker else l.body


class Builder:
    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(TOKENIZER)
        self.sep = self.tok.sep_token

    def ntok(self, texts):
        return [len(x) for x in self.tok(list(texts), add_special_tokens=False)["input_ids"]] if texts else []

    def map_pairs(self, aif: dict, name: str, links, gold: dict | None):
        """Examples for one map. aif: test-format input; links: (turn, prop, ...); gold: full gold AIF or None."""
        v = view(aif, name)
        pos, how = place(v, links)
        spk = speakers(v, pos)
        props = list(v.props)
        key = {p: (pos.get(p, float("inf")), norm(v.props[p])) for p in props}
        turns = [turn_text(v, l) for l in v.dialogue]
        turn_tok = [n + 2 for n in self.ntok(turns)]              # + " | " separator
        prop_tok = dict(zip(props, self.ntok([v.props[p] for p in props])))
        rel = ari_labels(gold, set(props)) if gold else {}
        out = []
        for x in range(len(props)):
            for y in range(x + 1, len(props)):
                a, b = sorted((props[x], props[y]), key=lambda p: key[p])
                placed = a in pos and b in pos
                if placed and abs(pos[a] - pos[b]) > K:
                    continue
                d = abs(pos[a] - pos[b]) if placed else None
                same = (spk[a] == spk[b]) if placed else None
                header = (f"distance: {d if d is not None else 'unknown'} turns; "
                          + {True: "same speaker", False: "different speakers", None: "speaker unknown"}[same])
                budget = MAX_LEN - prop_tok[a] - prop_tok[b] - 40 - MARGIN   # header, markers, specials
                lo, hi = self.window(pos.get(a), pos.get(b), turn_tok, budget)
                marked = []
                for k in range(lo, hi):
                    tags = "+".join(t for t, p in (("A", a), ("B", b)) if pos.get(p) == k)
                    marked.append(f"[[{tags}]] {turns[k]} [[/{tags}]]" if tags else turns[k])
                dialogue = ("… | " if lo > 0 else "") + " | ".join(marked) + (" | …" if hi < len(turns) else "")
                label = None
                if gold is not None:
                    ab, ba = rel.get((a, b)), rel.get((b, a))
                    label = ("Inference A->B" if ab == 0 else "Inference B->A" if ba == 0 else
                             "Conflict" if 1 in (ab, ba) else "Rephrase" if 2 in (ab, ba) else "None")
                out.append({"map": name, "a": a, "b": b, "a_pos": pos.get(a), "b_pos": pos.get(b),
                            "a_how": how.get(a, "unplaced"), "b_how": how.get(b, "unplaced"),
                            "distance": d, "same_speaker": same, "windowed": (lo, hi) != (0, len(turns)),
                            "label": label, "label_id": LABELS.index(label) if label else None,
                            "text_a": v.props[a], "text_b": f"{v.props[b]} {self.sep} {header} {self.sep} {dialogue}"})
        return out, pos, how

    @staticmethod
    def window(pa, pb, turn_tok, budget):
        """Whole dialogue if it fits, else the widest window around the pair that does."""
        n = len(turn_tok)
        if sum(turn_tok) <= budget:
            return 0, n
        placed = [p for p in (pa, pb) if p is not None]
        lo, hi = (min(placed), max(placed) + 1) if placed else (n // 2, n // 2 + 1)
        while True:
            grown = False
            for nlo, nhi in ((lo - 1, hi), (lo, hi + 1)):
                if 0 <= nlo and nhi <= n and sum(turn_tok[nlo:nhi]) <= budget:
                    lo, hi, grown = nlo, nhi, True
            if not grown:
                return lo, hi


def round_trip(gold: dict, inp: dict, pairs: list[dict]):
    """Gold labels of the candidate pairs -> AIF -> official ARI score (the ceiling A1 can reach)."""
    rels = [r for r in (relation(p["a"], p["b"], p["label"]) for p in pairs) if r]
    pred = with_relations(inp, rels)
    return score_map(gold, pred)["ari"], validate(pred)


def main():
    b = Builder()
    CACHE.mkdir(parents=True, exist_ok=True)
    report = {"k": K, "max_len": MAX_LEN, "giant_over": GIANT, "labels": LABELS}
    links_dev = json.loads((ROOT / "cache" / "ilo" / "links_dev.json").read_text())
    links_test = json.loads((ROOT / "cache" / "ilo" / "links_test.json").read_text())
    with timed("a1", "A1.1_build_pairs") as row:
        for split in ("train", "dev", "test"):
            if split == "test":
                maps = [(name, load_map(inp), None) for name, inp, _ in test_paths()]   # inputs only
            else:
                maps = [(p.stem, None, load_map(p)) for p in load_split(split)]
            rows, skipped, how_count = [], [], Counter()
            kept_rel = total_rel = 0
            rt_gen, rt_foc, invalid, gate_maps = [], [], 0, 0
            for name, inp, gold in maps:
                if gold is not None:
                    inp = strip_to_input(gold)
                if len(view(inp, name).props) > GIANT:
                    skipped.append(name)
                    continue
                links = (view(gold, name).anchors() if split == "train" else
                         links_dev.get(name, []) if split == "dev" else links_test.get(name, []))
                pairs, pos, how = b.map_pairs(inp, name, links, gold)
                rows += pairs
                how_count.update(how.values())
                how_count["unplaced"] += len(view(inp, name).props) - len(pos)
                if gold is not None:
                    related = {frozenset(k) for k in ari_labels(gold, set(view(inp, name).props))}
                    cand = {frozenset((p["a"], p["b"])) for p in pairs}
                    kept_rel += len(related & cand)
                    total_rel += len(related)
                    if split == "dev" or gate_maps < 300:             # round trip on all dev + 300 train maps
                        (gen, foc), problems = round_trip(gold, inp, pairs)
                        gate_maps += 1
                        rt_gen.append(gen[2])
                        if foc is not None:
                            rt_foc.append(foc[2])
                        invalid += bool(problems)
            with open(CACHE / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            lengths = sorted(len(x) for x in b.tok([r["text_a"] for r in rows], [r["text_b"] for r in rows])["input_ids"])
            q = lambda x: lengths[min(len(lengths) - 1, int(x * len(lengths)))]
            rep = {"maps": len({r["map"] for r in rows}), "skipped_giant_maps": skipped, "pairs": len(rows),
                   "placed_by": dict(how_count),
                   "tokens": {"p50": q(0.5), "p99": q(0.99), "max": lengths[-1], "mean": round(st.mean(lengths), 1),
                              "over_max_len": sum(x > MAX_LEN for x in lengths)},
                   "windowed_pairs": sum(r["windowed"] for r in rows)}
            if split != "test":
                rep["labels"] = dict(Counter(r["label"] for r in rows).most_common())
                rep["related_pairs_kept"] = f"{kept_rel}/{total_rel} = {kept_rel / total_rel:.4f}"
                rep["round_trip_ari"] = {"maps": gate_maps, "general": round(100 * st.mean(rt_gen), 2),
                                         "focused": round(100 * st.mean(rt_foc), 2), "invalid_maps": invalid}
            report[split] = rep
            row[split] = len(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "pairs.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
