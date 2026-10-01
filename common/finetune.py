"""Reusable fine-tuning loop for pair classifiers (ILO labeller in Phase 3, A1 in A1.2).

The caller supplies tokenized examples, labels, class weights and an `evaluate(probs) -> dict`
function that scores dev predictions with the real metric. The loop handles the rest:
length-grouped batches, bf16 autocast with fp32 weights, weighted cross-entropy, AdamW with
linear warm-up and decay, periodic dev evaluation, early stopping on `metric`, saving the best
checkpoint and its dev probabilities, and logging progress, timing and peak VRAM.
"""
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # variable shapes fragment memory

import numpy as np
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

transformers.logging.set_verbosity_error()   # the freshly initialised classifier head is expected


@dataclass
class Config:
    num_labels: int
    out_dir: Path                      # checkpoint directory (git-ignored)
    log_path: Path                     # JSON-lines training log (tracked)
    model: str = "answerdotai/ModernBERT-base"
    attn: str = "sdpa"
    max_len: int = 320
    max_tokens: int = 4096             # padded tokens per training batch (8 GB card; D033)
    batch_size: int = 64               # at most this many sequences per batch
    eval_max_tokens: int = 16384
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup: float = 0.06
    epochs: int = 3
    evals_per_epoch: int = 2
    patience: int = 2                  # evaluations without improvement before stopping
    grad_clip: float = 1.0
    seed: int = 13
    metric: str = "score"              # key of evaluate()'s result to maximise
    extra: dict = field(default_factory=dict)


def tokenize(model_name: str, a: list[str], b: list[str], max_len: int) -> list[list[int]]:
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = tok(a, b, truncation="only_second", max_length=max_len)
    return enc["input_ids"]


def _pack(order: list[int], lengths: list[int], max_tokens: int, max_seqs: int) -> list[list[int]]:
    """Greedy batches over length-sorted indices: padded size (rows x longest) stays within max_tokens."""
    out, cur, width = [], [], 0
    for i in order:
        w = max(width, lengths[i])
        if cur and (w * (len(cur) + 1) > max_tokens or len(cur) >= max_seqs):
            out.append(cur)
            cur, w = [], lengths[i]
        cur.append(i)
        width = w
    if cur:
        out.append(cur)
    return out


def batches(lengths: list[int], max_tokens: int, max_seqs: int, rng: random.Random | None):
    """Index batches grouped by length (less padding) under a token budget; shuffled when rng is given."""
    idx = list(range(len(lengths)))
    if rng is None:
        return _pack(sorted(idx, key=lambda i: lengths[i]), lengths, max_tokens, max_seqs)
    rng.shuffle(idx)
    chunk = max_seqs * 50
    out = []
    for s in range(0, len(idx), chunk):
        out += _pack(sorted(idx[s:s + chunk], key=lambda i: lengths[i]), lengths, max_tokens, max_seqs)
    rng.shuffle(out)
    return out


def collate(ids: list[list[int]], batch: list[int], pad_id: int, device):
    width = max(len(ids[i]) for i in batch)
    x = torch.full((len(batch), width), pad_id, dtype=torch.long)
    m = torch.zeros((len(batch), width), dtype=torch.long)
    for r, i in enumerate(batch):
        x[r, :len(ids[i])] = torch.tensor(ids[i])
        m[r, :len(ids[i])] = 1
    return x.to(device, non_blocking=True), m.to(device, non_blocking=True)


def load_model(cfg: Config, path: str | None = None):
    model = AutoModelForSequenceClassification.from_pretrained(
        path or cfg.model, num_labels=cfg.num_labels, attn_implementation=cfg.attn)
    return model.cuda()


@torch.no_grad()
def predict(model, ids: list[list[int]], pad_id: int, max_tokens: int = 16384) -> np.ndarray:
    model.eval()
    probs = np.zeros((len(ids), model.config.num_labels), dtype=np.float32)
    for b in batches([len(x) for x in ids], max_tokens, 256, None):
        x, m = collate(ids, b, pad_id, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x, attention_mask=m).logits
        probs[b] = torch.softmax(logits.float(), -1).cpu().numpy()
    return probs


def log(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def train(cfg: Config, train_ids, train_labels, dev_ids, class_weights, evaluate, pad_id: int,
          max_steps: int | None = None) -> dict:
    """Train with early stopping on evaluate(dev_probs)[cfg.metric]. Returns a summary dict."""
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    model = load_model(cfg)
    model.train()
    no_decay = ("bias", "norm")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n.lower() for k in no_decay)],
               "weight_decay": cfg.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(k in n.lower() for k in no_decay)],
               "weight_decay": 0.0}]
    optim = torch.optim.AdamW(groups, lr=cfg.lr, fused=True)
    lengths = [len(x) for x in train_ids]
    steps_per_epoch = len(batches(lengths, cfg.max_tokens, cfg.batch_size, random.Random(0)))
    total = max_steps or steps_per_epoch * cfg.epochs
    sched = get_linear_schedule_with_warmup(optim, int(cfg.warmup * total), total)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float, device="cuda"))
    labels = torch.tensor(train_labels)
    eval_every = max(1, steps_per_epoch // cfg.evals_per_epoch)

    torch.cuda.reset_peak_memory_stats()
    best, best_step, bad_evals, step, tokens, t0 = -1.0, 0, 0, 0, 0, time.perf_counter()
    window_loss, window_t, window_tok = 0.0, time.perf_counter(), 0
    log(cfg.log_path, {"event": "start", "config": {k: str(v) for k, v in asdict(cfg).items()},
                       "train": len(train_ids), "dev": len(dev_ids), "steps": total})
    done = False
    for epoch in range(cfg.epochs if max_steps is None else 10**6):
        for b in batches(lengths, cfg.max_tokens, cfg.batch_size, rng):
            x, m = collate(train_ids, b, pad_id, "cuda")
            y = labels[b].to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x, attention_mask=m).logits
            loss = loss_fn(logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            n_tok = int(m.sum())
            tokens += n_tok
            window_loss += loss.item()
            window_tok += n_tok
            if step % 100 == 0:
                dt = time.perf_counter() - window_t
                log(cfg.log_path, {"event": "train", "step": step, "epoch": round(step / steps_per_epoch, 2),
                                   "loss": round(window_loss / 100, 4), "lr": sched.get_last_lr()[0],
                                   "tokens_per_s": round(window_tok / dt), "peak_vram_gib":
                                   round(torch.cuda.max_memory_allocated() / 2**30, 2)})
                window_loss, window_t, window_tok = 0.0, time.perf_counter(), 0
            if max_steps is None and step % eval_every == 0:
                probs = predict(model, dev_ids, pad_id, cfg.eval_max_tokens)
                res = evaluate(probs)
                improved = res[cfg.metric] > best
                log(cfg.log_path, {"event": "eval", "step": step, "epoch": round(step / steps_per_epoch, 2),
                                   **res, "best": improved})
                if improved:
                    best, best_step, bad_evals = res[cfg.metric], step, 0
                    model.save_pretrained(cfg.out_dir)
                    np.save(cfg.out_dir / "dev_probs.npy", probs)
                else:
                    bad_evals += 1
                model.train()
                if bad_evals >= cfg.patience:
                    done = True
            if step >= total or done:
                done = True
                break
        if done:
            break
    elapsed = time.perf_counter() - t0
    summary = {"event": "end", "steps": step, "epochs_run": round(step / steps_per_epoch, 2),
               "best_" + cfg.metric: best, "best_step": best_step, "seconds": round(elapsed, 1),
               "minutes_per_epoch": round(elapsed / (step / steps_per_epoch) / 60, 1),
               "tokens_per_s": round(tokens / elapsed), "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    log(cfg.log_path, summary)
    return {"model": model, **summary}
