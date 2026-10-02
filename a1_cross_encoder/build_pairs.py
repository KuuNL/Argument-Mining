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

    python -m a1_cross_encoder.build_pairs                  -> cache/a1/{train,dev,test}.jsonl, results/a1/pairs.json
    python -m a1_cross_encoder.build_pairs --variant=ctx2   -> cache/a1/ctx2/{train,dev}.jsonl (A1.6 ablation)

Ablation variants (A1.6, D047), each changing one lever:
    ctx2    dialogue limited to 2 turns either side of the pair (instead of the whole map)
    nomark  no written distance / same-speaker hints
    k99     no pruning: every pair of propositions
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
VARIANTS = {"ctx2": {"context": 2}, "nomark": {"header": False}, "k99": {"k": 99}}
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

    def map_pairs(self, aif: dict, name: str, links, gold: dict | None, k: int | None = None,
                  context: int | None = None, header: bool = True):
        window_k = k or K            # own name: `k` below is also a turn index (that clash broke A1.6 builds)
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
                if placed and abs(pos[a] - pos[b]) > window_k:
                    continue
                d = abs(pos[a] - pos[b]) if placed else None
                same = (spk[a] == spk[b]) if placed else None
                head = (f"distance: {d if d is not None else 'unknown'} turns; "
                          + {True: "same speaker", False: "different speakers", None: "speaker unknown"}[same])
                budget = MAX_LEN - prop_tok[a] - prop_tok[b] - 40 - MARGIN   # header, markers, specials
                if context is None:
                    lo, hi = self.window(pos.get(a), pos.get(b), turn_tok, budget)
                else:                                          # ablation: a fixed window around the pair
                    placed_at = [q for q in (pos.get(a), pos.get(b)) if q is not None]
                    lo, hi = ((max(0, min(placed_at) - context), min(len(turns), max(placed_at) + context + 1))
                              if placed_at else (0, 0))
                marked = []
                for i in range(lo, hi):
                    tags = "+".join(t for t, p in (("A", a), ("B", b)) if pos.get(p) == i)
                    marked.append(f"[[{tags}]] {turns[i]} [[/{tags}]]" if tags else turns[i])
                dialogue = (("… | " if lo > 0 else "") + " | ".join(marked) + (" | …" if hi < len(turns) else "")) or "-"
                label = None
                if gold is not None:
                    ab, ba = rel.get((a, b)), rel.get((b, a))
                    label = ("Inference A->B" if ab == 0 else "Inference B->A" if ba == 0 else
                             "Conflict" if 1 in (ab, ba) else "Rephrase" if 2 in (ab, ba) else "None")
                out.append({"map": name, "a": a, "b": b, "a_pos": pos.get(a), "b_pos": pos.get(b),
                            "a_how": how.get(a, "unplaced"), "b_how": how.get(b, "unplaced"),
                            "distance": d, "same_speaker": same, "windowed": (lo, hi) != (0, len(turns)),
                            "label": label, "label_id": LABELS.index(label) if label else None,
                            "text_a": v.props[a],
                            "text_b": (f"{v.props[b]} {self.sep} {head} {self.sep} {dialogue}" if header
                                       else f"{v.props[b]} {self.sep} {dialogue}")})
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
    import sys
    variant = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--variant=")), None)
    opts = VARIANTS.get(variant, {})
    cache = CACHE / variant if variant else CACHE
    b = Builder()
    cache.mkdir(parents=True, exist_ok=True)
    report = {"variant": variant, "k": opts.get("k", K), "max_len": MAX_LEN, "giant_over": GIANT, "labels": LABELS, **opts}
    links_dev = json.loads((ROOT / "cache" / "ilo" / "links_dev.json").read_text())
    links_test = json.loads((ROOT / "cache" / "ilo" / "links_test.json").read_text())
    with timed("a1", "A1.1_build_pairs") as row:
        for split in (("train", "dev") if variant else ("train", "dev", "test")):
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
                pairs, pos, how = b.map_pairs(inp, name, links, gold, **opts)
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
            with open(cache / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            lengths = []                                   # in chunks: all-at-once exhausted WSL RAM on k99
            for s in range(0, len(rows), 8192):
                part = rows[s:s + 8192]
                lengths += [len(x) for x in b.tok([r["text_a"] for r in part], [r["text_b"] for r in part],
                                                  return_attention_mask=False)["input_ids"]]
            lengths.sort()
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
    with open(OUT / (f"pairs_{variant}.json" if variant else "pairs.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
