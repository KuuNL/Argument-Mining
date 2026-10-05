"""A3 training: QLoRA on the training maps with an answer-only loss, saving an adapter every half epoch (D065).

One map per forward pass; gradients are accumulated over `accum` maps, and each update weighs every answer
token equally. Paged 8-bit AdamW, linear warm-up (3%) and decay, gradient clipping at 1. At every save
point the adapter goes to checkpoints/a3/<run>/ep<E>/ and the dev answer loss (225 dev maps, teacher-forced)
is logged. Greedy dev ARI per adapter is scored afterwards (a3_finetune/evaluate.py), so training and
generation never share the card.

Before training, the loss mask is checked on every example: the labelled tokens decode to exactly the gold
answer plus the end-of-turn token.

    python -m a3_finetune.train --run=v1                                    # 1,231 maps, 3 epochs
    python -m a3_finetune.train --run=overfit --maps=32 --epochs=10 --accum=1 --no-dev
"""
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from common import RESULTS, ROOT
from common.finetune import log
from common.timing import timed
from a3_finetune.data import END, dev_examples, encode, giant_chunks, train_examples
from a3_finetune.model import answer_loss, load, tokenizer

CKPT = ROOT / "checkpoints" / "a3"
OUT = RESULTS / "a3"


@dataclass
class Config:
    run: str
    model: str = "qwen3-4b"
    epochs: float = 3
    lr: float = 2e-4
    accum: int = 16
    maps: int = 0                  # 0 = every test-like training map; else a seeded sample of this many
    giants: bool = False           # add the giant-map chunks (the retrain ablation)
    dev: bool = True               # dev answer loss at every save point
    saves_per_epoch: int = 2
    warmup: float = 0.03
    grad_clip: float = 1.0
    seed: int = 13


def args() -> Config:
    kw = {}
    for a in sys.argv[1:]:
        k, _, v = a.lstrip("-").partition("=")
        k = k.replace("-", "_")
        if k == "no_dev":
            kw["dev"] = False
        elif k == "giants":
            kw["giants"] = True
        elif k in Config.__dataclass_fields__:
            kw[k] = Config.__dataclass_fields__[k].type(v)
    return Config(**kw)


def check_masking(tok, examples, encoded):
    """Every example's labelled tokens are exactly its answer plus the end-of-turn token."""
    for e, (ids, labels) in zip(examples, encoded):
        kept = [t for t, y in zip(ids, labels) if y != -100]
        first = next(k for k, y in enumerate(labels) if y != -100)
        assert all(y == -100 for y in labels[:first]) and all(y != -100 for y in labels[first:]), e.name
        assert tok.decode(kept) == e.answer + END, e.name
    return {"examples": len(examples), "answer_tokens": sum(sum(y != -100 for y in l) for _, l in encoded),
            "prompt_tokens_masked": sum(sum(y == -100 for y in l) for _, l in encoded)}


@torch.no_grad()
def dev_loss(model, dev) -> float:
    model.eval()
    total, n = 0.0, 0
    for ids, labels in dev:
        loss, k = answer_loss(model, torch.tensor([ids], device="cuda"), torch.tensor([labels], device="cuda"))
        total += loss.item() * k
        n += k
    model.train()
    return total / n


def main():
    import bitsandbytes as bnb
    cfg = args()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out, ckpt = OUT / cfg.run, CKPT / cfg.run
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    tok = tokenizer(cfg.model)
    examples = sorted(train_examples(), key=lambda e: e.name)
    if cfg.maps:
        examples = sorted(random.Random(cfg.seed).sample(examples, cfg.maps), key=lambda e: e.name)
    if cfg.giants:
        examples += giant_chunks()[0]
    data = [encode(tok, e) for e in examples]
    masking = check_masking(tok, examples, data)
    dev = [encode(tok, e) for e in dev_examples()] if cfg.dev else []
    (out / "config.json").write_text(json.dumps({**asdict(cfg), "train_maps": [e.name for e in examples],
                                                 "masking": masking}, indent=1) + "\n", encoding="utf-8", newline="\n")

    model = load(cfg.model, train=True)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.PagedAdamW8bit(params, lr=cfg.lr, weight_decay=0.0)
    per_epoch = math.ceil(len(data) / cfg.accum)
    total = math.ceil(per_epoch * cfg.epochs)
    sched = get_linear_schedule_with_warmup(opt, max(1, int(cfg.warmup * total)), total)
    save_at = {round(per_epoch * (k + 1) / cfg.saves_per_epoch) for k in range(cfg.saves_per_epoch)}
    rng = random.Random(cfg.seed)
    log(log_path, {"event": "start", "config": asdict(cfg), "train": len(data), "dev": len(dev),
                   "updates": total, "updates_per_epoch": per_epoch, "masking": masking})

    torch.cuda.reset_peak_memory_stats()
    step, tokens, t0 = 0, 0, time.perf_counter()
    window = {"loss": 0.0, "tok": 0, "t": time.perf_counter()}
    with timed("a3", f"train:{cfg.run}", model=cfg.model, maps=len(data)) as row:
        while step < total:
            order = list(range(len(data)))
            rng.shuffle(order)
            for g in range(0, len(order), cfg.accum):
                group = order[g:g + cfg.accum]
                n_ans = sum(sum(y != -100 for y in data[i][1]) for i in group)
                group_loss = 0.0
                for i in group:
                    ids, labels = data[i]
                    loss, k = answer_loss(model, torch.tensor([ids], device="cuda"),
                                          torch.tensor([labels], device="cuda"))
                    (loss * k / n_ans).backward()
                    group_loss += loss.item() * k / n_ans
                    tokens += len(ids)
                    window["tok"] += len(ids)
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                window["loss"] += group_loss
                in_epoch = (step - 1) % per_epoch + 1
                if step % 10 == 0 or in_epoch in save_at:
                    dt = time.perf_counter() - window["t"]
                    n = step % 10 or 10
                    log(log_path, {"event": "train", "step": step, "epoch": round(step / per_epoch, 3),
                                   "loss": round(window["loss"] / n, 4), "lr": sched.get_last_lr()[0],
                                   "tokens_per_s": round(window["tok"] / dt),
                                   "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)})
                    window = {"loss": 0.0, "tok": 0, "t": time.perf_counter()}
                if in_epoch in save_at:
                    tag = f"ep{step / per_epoch:.1f}"
                    model.save_pretrained(ckpt / tag)
                    rec = {"event": "save", "step": step, "epoch": round(step / per_epoch, 3), "adapter": tag}
                    if dev:
                        rec["dev_loss"] = round(dev_loss(model, dev), 4)
                    log(log_path, rec)
                    window["t"] = time.perf_counter()
                if step >= total:
                    break
        elapsed = time.perf_counter() - t0
        summary = {"event": "end", "updates": step, "seconds": round(elapsed, 1),
                   "minutes_per_epoch": round(elapsed / (step / per_epoch) / 60, 1),
                   "tokens_per_s": round(tokens / elapsed),
                   "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
        log(log_path, summary)
        row.update(summary)


if __name__ == "__main__":
    main()
