"""A1.2 / A1.3: train the A1 relation model (ModernBERT-base, 5 pair labels) on cache/a1/.

Every run has a fixed recipe in RUNS. Lessons from Phase 3 are built in: checkpoints are judged by
the *tuned* dev score (a1_cross_encoder.decide.tuned_score), every evaluation's dev probabilities
are kept, batches are capped by tokens, long runs are launched detached.

    python -m a1_cross_encoder.train --run=v1 --overfit   # sanity: memorise 72 pairs
    python -m a1_cross_encoder.train --run=v1 --bench     # 300 optimizer steps + a dev pass: speed, VRAM
    python -m a1_cross_encoder.train --run=v1             # full run (A1.3)

Outputs: checkpoints/a1/<run>/ (git-ignored) and results/a1/<run>/ (train_log.jsonl, train_summary.json).
"""
import json
import sys
import time
from collections import Counter

import numpy as np

from common import RESULTS, ROOT
from common.finetune import Config, predict, tokenize, train
from common.timing import timed
from a1_cross_encoder.assemble import LABELS
from a1_cross_encoder.decide import Dev, read, simple_score, tuned_score

WEIGHT_CAP = 30.0
RUNS = {
    "v1": {"lr": 3e-5, "accum_steps": 3, "max_tokens": 4096, "epochs": 3, "patience": 99,
           "note": "first A1 run: whole-dialogue inputs (k = 10), tuned-score checkpoint selection"},
}
# A1.6 ablations (D047): the v1 recipe with one lever off, via the input variant they train on.
for _name, _data, _note in (("abl-ctx2", "ctx2", "ablation: 2 turns of context either side instead of the whole map"),
                            ("abl-nomark", "nomark", "ablation: no written distance / same-speaker hints"),
                            ("abl-k99", "k99", "ablation: no pruning, every pair")):
    RUNS[_name] = {**RUNS["v1"], "data": _data, "note": _note}


def arg(name: str, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")), default)


def dirs(run: str):
    return ROOT / "checkpoints" / "a1" / run, RESULTS / "a1" / run


def class_weights(labels: list[int]) -> list[float]:
    """(most frequent / count) ** 0.5, capped (same rule as the ILO labeller)."""
    c = Counter(labels)
    top = max(c.values())
    return [min(WEIGHT_CAP, (top / c[k]) ** 0.5) if c[k] else WEIGHT_CAP for k in range(len(LABELS))]


def main():
    run = arg("run", "v1")
    recipe = RUNS[run]
    ckpt, out = dirs(run)
    cfg = Config(num_labels=len(LABELS), out_dir=ckpt, log_path=out / "train_log.jsonl", max_len=1024,
                 lr=recipe["lr"], accum_steps=recipe["accum_steps"], max_tokens=recipe["max_tokens"],
                 epochs=recipe["epochs"], patience=recipe["patience"], keep_eval_probs=True,
                 metric="ari_tuned_general", extra={"run": run, "note": recipe["note"]})
    from transformers import AutoTokenizer
    pad_id = AutoTokenizer.from_pretrained(cfg.model).pad_token_id
    train_rows = read("train", recipe.get("data"))
    y = [r["label_id"] for r in train_rows]
    weights = class_weights(y)

    if "--overfit" in sys.argv:
        rng = np.random.default_rng(0)
        pick = [int(i) for i in rng.choice(len(train_rows), 64, replace=False)]
        pick += [i for i, r in enumerate(train_rows) if r["label"] == "Conflict"][:8]
        sub = [train_rows[i] for i in pick]
        ids = tokenize(cfg.model, [r["text_a"] for r in sub], [r["text_b"] for r in sub], cfg.max_len)
        labels = [r["label_id"] for r in sub]
        cfg.log_path, cfg.out_dir = out / "overfit_log.jsonl", ckpt / "overfit"
        cfg.lr, cfg.warmup, cfg.accum_steps = 1e-4, 0.0, 1
        with timed("a1", "A1.2_overfit", examples=len(sub)) as row:
            steps = int(arg("steps", 100))
            res = train(cfg, ids, labels, [], [1.0] * len(LABELS), None, pad_id, max_steps=steps)
            hit = predict(res["model"], ids, pad_id).argmax(1) == np.array(labels)
            acc = float(np.mean(hit))
            row.update(train_accuracy=acc, steps=steps)
        wrong = Counter(r["label"] for r, h in zip(sub, hit) if not h)
        print(json.dumps({"overfit_examples": len(sub), "steps": steps, "train_accuracy": acc,
                          "labels": dict(Counter(r["label"] for r in sub)), "wrong_by_label": dict(wrong)}))
        return

    if "--bench" in sys.argv:
        import torch
        dev_rows = read("dev")[:3000]
        tr_ids = tokenize(cfg.model, [r["text_a"] for r in train_rows], [r["text_b"] for r in train_rows], cfg.max_len)
        train_rows.clear()                                  # free the texts; ids are compact arrays (D042)
        dv_ids = tokenize(cfg.model, [r["text_a"] for r in dev_rows], [r["text_b"] for r in dev_rows], cfg.max_len)
        cfg.log_path = out / "bench_log.jsonl"
        with timed("a1", "A1.2_bench", max_tokens=cfg.max_tokens, accum=cfg.accum_steps) as row:
            res = train(cfg, tr_ids, y, [], weights, None, pad_id, max_steps=300)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            predict(res["model"], dv_ids, pad_id, cfg.eval_max_tokens)
            secs = time.perf_counter() - t0
            train_tokens = sum(len(x) for x in tr_ids)
            row.update(train_tokens_per_s=res["tokens_per_s"], train_peak_vram=res["peak_vram_gib"],
                       dev_pairs_per_s=round(len(dv_ids) / secs), dev_peak_vram=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       train_tokens_per_epoch=train_tokens,
                       est_minutes_per_epoch=round(train_tokens / res["tokens_per_s"] / 60, 1))
        print(json.dumps(row))
        return

    dev = Dev(load=False, data=recipe.get("data"))
    scorer = lambda probs: {**simple_score(dev, probs), **tuned_score(dev, probs)}
    tr_ids = tokenize(cfg.model, [r["text_a"] for r in train_rows], [r["text_b"] for r in train_rows], cfg.max_len)
    train_rows.clear()                                      # free the texts; ids are compact arrays (D042)
    dv_ids = tokenize(cfg.model, [r["text_a"] for r in dev.rows], [r["text_b"] for r in dev.rows], cfg.max_len)
    with timed("a1", f"A1.3_train_{run}", lr=cfg.lr, train=len(tr_ids), dev=len(dv_ids)) as row:
        res = train(cfg, tr_ids, y, dv_ids, weights, scorer, pad_id)
        row.update({k: v for k, v in res.items() if k != "model"})
    summary = ({"run": run, **recipe} | {k: v for k, v in res.items() if k != "model"}
               | {"class_weights": dict(zip(LABELS, [round(w, 2) for w in weights]))})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train_summary.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
