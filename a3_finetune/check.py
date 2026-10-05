"""A3.1 gate checks, no model (the tokenizer only).

1. Lossless round trip: each dev map's gold answer, written as compact A3 text, goes through json -> the local
   schema check -> A3's parser -> A1's decision code -> build_map -> official scorer. It must give the
   no-pruning pipeline ceiling, 98.92 / 96.51 ARI (as A2.1), with every map valid.
2. Grammar: every training, dev and giant-chunk gold answer, tokenised as in training, is accepted by its map's
   grammar and ends it (D065).
3. Leakage: no dev or test map (by name or by identical text) among the training examples.
4. Sizes: tokens of prompt + answer per split; giant-chunk losses.
5. Samples: two training examples and two giant chunks in results/a3/samples/, to read by eye.

    python -m a3_finetune.check [--tokenizer=Qwen/Qwen3-8B]     -> results/a3/check.json
"""
import json
import statistics as st
import sys

from transformers import AutoConfig, AutoTokenizer

from common import RESULTS
from common.score import score_map
from common.serialise import serialise
from common.timing import timed
from common.validate import validate
from a1_cross_encoder.decide import decode
from a1_cross_encoder.predict import build_map
from a2_llm.check import HALF, dev_maps
from a2_llm.prompt import gold_answer
from a3_finetune.data import encode, giant_chunks, prompt, test_examples, train_examples, dev_examples
from a3_finetune.grammar import Grammars
from a3_finetune.parse import pair_scores
from a3_finetune.schema import compact, dumps, problems

OUT = RESULTS / "a3"


def spread(xs):
    xs = sorted(xs)
    return {"n": len(xs), "p50": xs[len(xs) // 2], "p90": xs[int(0.9 * (len(xs) - 1))],
            "p99": xs[int(0.99 * (len(xs) - 1))], "max": xs[-1]}


def round_trip() -> dict:
    gen, foc, invalid, bad = [], [], [], []
    for name, gold, inp, links in dev_maps():
        s = serialise(inp, name, links)
        answer = json.loads(dumps(compact(gold_answer(gold, s))))
        if problems(answer, s):
            bad.append(name)
        rows, scores = pair_scores(s, [answer])
        m = build_map(inp, name, decode(rows, scores, HALF), links, s.pos)
        if validate(m):
            invalid.append(name)
        ari = score_map(gold, m)["ari"]
        gen.append(100 * ari[0][2])
        if ari[1] is not None:
            foc.append(100 * ari[1][2])
    return {"maps": len(gen), "ari_general": round(st.mean(gen), 2), "ari_focused": round(st.mean(foc), 2),
            "expected": [98.92, 96.51], "invalid_maps": invalid, "answers_failing_schema": bad}


def main():
    model = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--tokenizer=")), "Qwen/Qwen3-8B")
    report = {"tokenizer": model}
    with timed("a3", "A3.1_check") as row:
        report["round_trip"] = round_trip()
        print("round trip", report["round_trip"], flush=True)

        train, dev, test = train_examples(), dev_examples(), test_examples()
        chunks, giant = giant_chunks()
        report["giant_chunks"] = giant
        names = {e.name for e in train}
        texts = {e.user for e in train}
        report["leakage"] = {"dev_names_in_train": sorted(e.name for e in dev if e.name in names),
                             "dev_texts_in_train": sorted(e.name for e in dev if e.user in texts),
                             "test_texts_in_train": sorted(e.name for e in test if e.user in texts)}

        tok = AutoTokenizer.from_pretrained(model)
        grammars = Grammars(tok, AutoConfig.from_pretrained(model).vocab_size)
        sizes, rejected = {}, {}
        for split, exs in (("train", train), ("dev", dev), ("giant_chunks", chunks)):
            total, answer, bad = [], [], []
            for e in exs:
                ids, labels = encode(tok, e)
                ans = [t for t, y in zip(ids, labels) if y != -100]
                total.append(len(ids))
                answer.append(len(ans))
                if not grammars.accepts(e.s, ans):
                    bad.append(e.name)
            sizes[split] = {"total_tokens": spread(total), "answer_tokens": spread(answer)}
            rejected[split] = {"checked": len(exs), "rejected": bad[:20], "n_rejected": len(bad)}
            print(split, sizes[split], rejected[split]["n_rejected"], "rejected", flush=True)
        sizes["test_prompt_tokens"] = spread([len(tok(prompt(tok, e.user), add_special_tokens=False)["input_ids"])
                                              for e in test])
        report["tokens"], report["grammar"] = sizes, rejected

        samples = OUT / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        by_size = sorted(train, key=lambda e: len(e.answer))
        for tag, e in (("train_median", by_size[len(by_size) // 2]), ("train_largest", by_size[-1]),
                       ("giant_chunk_first", chunks[0]), ("giant_chunk_mid", chunks[len(chunks) // 2])):
            (samples / f"{tag}.txt").write_text(f"# {e.name}\n\n{prompt(tok, e.user)}{e.answer}\n", encoding="utf-8", newline="\n")
        row["note"] = f"round trip {report['round_trip']['ari_general']}/{report['round_trip']['ari_focused']}"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("round_trip", "leakage", "giant_chunks")}, indent=1))


if __name__ == "__main__":
    main()
