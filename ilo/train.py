"""Step 3.3 and its iterations: train the ILO labeller.

Every training run has a name and a fixed recipe in RUNS, so each result traces back to its settings.

    python -m ilo.train --run=v2              # full training run
    python -m ilo.train --run=v1 --overfit    # sanity: can it memorise 72 examples?
    python -m ilo.train --run=v1 --bench      # 400 real steps: speed and peak VRAM
    python -m ilo.train --run=v1 --probe      # sdpa vs flex_attention on real batches

Outputs per run: checkpoints/ilo/<run>/ (best model, dev_probs.npy, evals/*.npy; git-ignored) and
results/ilo/<run>/ (train_log.jsonl, train_summary.json). results/ilo/selected.json names the run
the rest of the project uses.
"""
import json
import sys
import time
from collections import Counter

import numpy as np

from common import RESULTS, ROOT
from common.aif import load_map
from common.finetune import Config, collate, load_model, predict, tokenize, train
from common.split import load_split
from common.timing import timed
from ilo.data import CACHE, LABELS
from ilo.decode import DevScorer

WEIGHT_CAP = 30.0
RUNS = {
    "v1": {"lr": 5e-5, "select": "simple", "patience": 2, "keep_eval_probs": False,
           "note": "first run (D032-D034): early stopping on the simple decode; best checkpoint at 0.5 epoch"},
    "v2": {"lr": 3e-5, "select": "tuned", "patience": 99, "keep_eval_probs": True,
           "note": "rare-type iteration (D038): all 3 epochs, checkpoint chosen by the tuned score"},
}


def arg(name: str, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")), default)


def dirs(run: str):
    """(checkpoint dir, results dir) of a run."""
    return ROOT / "checkpoints" / "ilo" / run, RESULTS / "ilo" / run


def selected_run() -> str:
    return json.loads((RESULTS / "ilo" / "selected.json").read_text())["run"]


def read(split):
    with open(CACHE / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def class_weights(labels: list[int]) -> list[float]:
    """(most frequent / count) ** 0.5, capped: rare types count more without blowing up."""
    c = Counter(labels)
    top = max(c.values())
    return [min(WEIGHT_CAP, (top / c[k]) ** 0.5) if c[k] else WEIGHT_CAP for k in range(len(LABELS))]


def make_scorer(select: str, dev_rows):
    """Dev evaluation used during training: the simple decode, plus the tuned score for 'tuned' runs."""
    gold = {p.stem: load_map(p) for p in load_split("dev") if p.stem in {r["map"] for r in dev_rows}}
    simple = DevScorer(dev_rows, gold)
    if select == "simple":
        return simple
    from ilo.tune import Dev, tuned_score
    dev = Dev(load=False)
    return lambda probs: {**simple(probs), **tuned_score(dev, probs)}


def main():
    run = arg("run", "v2")
    recipe = RUNS[run]
    ckpt, out = dirs(run)
    cfg = Config(num_labels=len(LABELS), out_dir=ckpt, log_path=out / "train_log.jsonl", lr=recipe["lr"],
                 patience=recipe["patience"], keep_eval_probs=recipe["keep_eval_probs"],
                 metric="ilo_tuned_general" if recipe["select"] == "tuned" else "ilo_general_f1",
                 extra={"run": run, "note": recipe["note"]})
    from transformers import AutoTokenizer
    pad_id = AutoTokenizer.from_pretrained(cfg.model).pad_token_id
    train_rows = read("train")
    y = [r["label_id"] for r in train_rows]
    weights = class_weights(y)

    if "--overfit" in sys.argv:
        rng = np.random.default_rng(0)
        pick = [int(i) for i in rng.choice(len(train_rows), 64, replace=False)]
        pick += [i for i, r in enumerate(train_rows) if r["label"] in ("Agreeing", "Challenging")][:8]
        sub = [train_rows[i] for i in pick]
        ids = tokenize(cfg.model, [r["text_a"] for r in sub], [r["text_b"] for r in sub], cfg.max_len)
        labels = [r["label_id"] for r in sub]
        cfg.log_path, cfg.out_dir = out / "overfit_log.jsonl", ckpt / "overfit"
        cfg.lr, cfg.warmup, cfg.batch_size = 1e-4, 0.0, 16
        with timed("ilo", "3.3_overfit", examples=len(sub)) as row:
            res = train(cfg, ids, labels, [], [1.0] * len(LABELS), None, pad_id, max_steps=100)
            acc = float(np.mean(predict(res["model"], ids, pad_id).argmax(1) == np.array(labels)))
            row.update(train_accuracy=acc)
        print(json.dumps({"overfit_examples": len(sub), "train_accuracy": acc, "labels": dict(Counter(r["label"] for r in sub))}))
        return

    if "--bench" in sys.argv:                          # 400 real steps + a dev pass: speed and peak VRAM
        import torch
        dev_rows = read("dev")[:5000]
        tr_ids = tokenize(cfg.model, [r["text_a"] for r in train_rows], [r["text_b"] for r in train_rows], cfg.max_len)
        dv_ids = tokenize(cfg.model, [r["text_a"] for r in dev_rows], [r["text_b"] for r in dev_rows], cfg.max_len)
        cfg.log_path = out / "bench_log.jsonl"
        with timed("ilo", "3.3_bench", max_tokens=cfg.max_tokens) as row:
            res = train(cfg, tr_ids, y, [], weights, None, pad_id, max_steps=400)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            predict(res["model"], dv_ids, pad_id, cfg.eval_max_tokens)
            row.update(train_tokens_per_s=res["tokens_per_s"], train_peak_vram=res["peak_vram_gib"],
                       dev_examples_per_s=round(len(dv_ids) / (time.perf_counter() - t0)),
                       dev_peak_vram=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        print(json.dumps(row))
        return

    if "--probe" in sys.argv:
        import torch
        sample = train_rows[:6400]
        ids = tokenize(cfg.model, [r["text_a"] for r in sample], [r["text_b"] for r in sample], cfg.max_len)
        speeds = {}
        for attn in ("sdpa", "flex_attention"):
            cfg.attn = attn
            model = load_model(cfg)
            model.train()
            optim = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
            order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
            batches = [order[i:i + 32] for i in range(0, len(order), 32)]
            np.random.default_rng(0).shuffle(batches)
            tok, t0 = 0, None
            for k, b in enumerate(batches[:120]):
                if k == 20:                                   # skip warm-up / compilation
                    torch.cuda.synchronize(); t0, tok = time.perf_counter(), 0
                x, m = collate(ids, b, pad_id, "cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=x, attention_mask=m, labels=torch.zeros(len(b), dtype=torch.long, device="cuda")).loss
                loss.backward(); optim.step(); optim.zero_grad(set_to_none=True)
                tok += int(m.sum())
            torch.cuda.synchronize()
            speeds[attn] = round(tok / (time.perf_counter() - t0))
            del model, optim
            torch.cuda.empty_cache()
        print(json.dumps({"tokens_per_s": speeds}))
        return

    dev_rows = read("dev")
    scorer = make_scorer(recipe["select"], dev_rows)
    tr_ids = tokenize(cfg.model, [r["text_a"] for r in train_rows], [r["text_b"] for r in train_rows], cfg.max_len)
    dv_ids = tokenize(cfg.model, [r["text_a"] for r in dev_rows], [r["text_b"] for r in dev_rows], cfg.max_len)
    with timed("ilo", f"3.3_train_{run}", attn=cfg.attn, lr=cfg.lr, train=len(tr_ids), dev=len(dv_ids)) as row:
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
