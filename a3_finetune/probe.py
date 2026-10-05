"""A3.1 memory and speed probes, which apply D065's model rule.

Training: `steps` optimizer steps (batch 1) on the longest training examples, with the full memory path
(a3_finetune/model.py) and paged 8-bit AdamW. Measures peak GPU memory (PyTorch's and the driver's view)
and training tokens per second after 3 warm-up steps.
Generation (--gen): for `maps` dev maps, the greedy answer, then 8 relation samples at the largest batch of
8 / 4 / 2 that runs; answers per minute and validity.

Rule (D065): Qwen3-8B if its training peak is <= 7.5 GiB, it trains at >= 200 tokens/s, and generation runs
a batch of >= 2; otherwise Qwen3-4B-Instruct-2507.

    python -m a3_finetune.probe --model=qwen3-8b [--steps=20] [--gen] [--maps=5] [--sweep]
"""
import json
import sys
import time

import torch

from common import RESULTS
from common.timing import timed
from a3_finetune.data import dev_examples, encode, train_examples
from a3_finetune.generate import Generator
from a3_finetune.grammar import Grammars
from a3_finetune.model import load, tokenizer, answer_loss
from a3_finetune.schema import problems

OUT = RESULTS / "a3"
GIB = 2 ** 30


def arg(name, default):
    return next((type(default)(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith(f"--{name}=")), default)


def driver_used() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / GIB


def train_probe(key: str, steps: int) -> dict:
    import bitsandbytes as bnb
    tok = tokenizer(key)
    examples = sorted((encode(tok, e) for e in train_examples()), key=lambda x: -len(x[0]))[:steps]
    before = driver_used()
    model = load(key, train=True)
    loaded = {"torch_gib": torch.cuda.memory_allocated() / GIB, "driver_gib": driver_used(), "driver_before_gib": before}
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    times, toks, peak_driver, losses = [], [], 0.0, []
    for k, (ids, labels) in enumerate(examples):
        x = torch.tensor([ids], device="cuda")
        y = torch.tensor([labels], device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss, _ = answer_loss(model, x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        toks.append(len(ids))
        losses.append(round(loss.item(), 4))
        peak_driver = max(peak_driver, driver_used())
        print(f"step {k}: {len(ids)} tokens, {times[-1]:.2f} s, loss {losses[-1]}, "
              f"peak {torch.cuda.max_memory_reserved() / GIB:.2f} GiB", flush=True)
    warm = slice(3, None)
    res = {"model": key, "trainable_params": trainable, "loaded": loaded, "steps": len(examples),
           "longest_tokens": toks[0], "tokens_per_s": sum(toks[warm]) / sum(times[warm]),
           "seconds_per_step": sum(times[warm]) / len(times[warm]),
           "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
           "peak_driver_used_gib": peak_driver, "losses": losses}
    del model, opt
    torch.cuda.empty_cache()
    return res


def length_sweep(key: str, lengths: list[int]) -> dict:
    """One forward + backward (train mode, no optimizer) at training examples nearest each length: where a model
    stops fitting. A step that spills past the card into system memory shows up as a collapse in tokens/s."""
    tok = tokenizer(key)
    exs = [encode(tok, e) for e in train_examples()]
    model = load(key, train=True)
    model.train()
    res = {"model": key, "loaded_torch_gib": torch.cuda.memory_allocated() / GIB, "loaded_driver_gib": driver_used(),
           "points": []}
    for target in lengths:
        ids, labels = min(exs, key=lambda x: abs(len(x[0]) - target))
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            loss, _ = answer_loss(model, torch.tensor([ids], device="cuda"), torch.tensor([labels], device="cuda"))
            loss.backward()
            torch.cuda.synchronize()
        except (torch.OutOfMemoryError, RuntimeError) as e:
            res["points"].append({"tokens": len(ids), "error": str(e)[:100]})
            break
        sec = time.perf_counter() - t0
        res["points"].append({"tokens": len(ids), "seconds": round(sec, 2), "tokens_per_s": round(len(ids) / sec, 1),
                              "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 2),
                              "driver_used_gib": round(driver_used(), 2)})
        print(res["points"][-1], flush=True)
        model.zero_grad(set_to_none=True)
    return res


def gen_probe(key: str, maps: int) -> dict:
    tok = tokenizer(key)
    model = load(key)
    model.eval()
    gen = Generator(model, tok, Grammars(tok, model.config.vocab_size))
    exs = dev_examples()[:maps]
    res = {"model": key, "maps": [], "batches": {}}
    greedy = {}
    for ex in exs:
        t0 = time.perf_counter()
        g = greedy[ex.name] = gen.greedy(ex)
        a = gen.answer(g.tokens)
        res["maps"].append({"map": ex.name, "greedy_seconds": round(time.perf_counter() - t0, 2),
                            "greedy_tokens_per_s": round(len(g.tokens) / (time.perf_counter() - t0), 1),
                            "greedy_tokens": len(g.tokens), "finished": g.finished,
                            "valid": a is not None and not problems(a, ex.s),
                            "relations": len(a["relations"]) if a else None})
        print(res["maps"][-1], flush=True)
    for n in (8, 4, 2):
        torch.cuda.reset_peak_memory_stats()
        t0, ok, total, generated = time.perf_counter(), 0, 0, 0
        try:
            for ex in exs:                                  # sampling only; the greedy answers are reused
                prefix, outs = gen.samples(ex, greedy[ex.name], n, seed=0)
                for o in outs:
                    a = gen.answer(prefix + o.tokens)
                    ok += a is not None and not problems(a, ex.s)
                    total += 1
                    generated += len(o.tokens)
        except torch.OutOfMemoryError as e:
            res["batches"][n] = {"error": "out of memory"}
            torch.cuda.empty_cache()
            continue
        sec = time.perf_counter() - t0
        res["batches"][n] = {"seconds": round(sec, 1), "samples": total, "valid": ok,
                             "samples_per_min": round(60 * total / sec, 1), "tokens_per_s": round(generated / sec, 1),
                             "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 2),
                             "peak_driver_used_gib": round(driver_used(), 2)}
        print(n, res["batches"][n], flush=True)
    return res


def main():
    key, steps, maps = arg("model", "qwen3-8b"), arg("steps", 20), arg("maps", 5)
    path = OUT / f"probe_{key}.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    with timed("a3", f"A3.1_probe:{key}") as row:
        if "--gen" in sys.argv:
            report["generation"] = gen_probe(key, maps)
        elif "--sweep" in sys.argv:
            report["length_sweep"] = length_sweep(key, [600, 1200, 1800, 2600])
        else:
            report["training"] = train_probe(key, steps)
            row.update({k: report["training"][k] for k in ("tokens_per_s", "peak_reserved_gib", "longest_tokens")})
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report.get("training", {}).items() if k != "losses"}, indent=1, default=float))


if __name__ == "__main__":
    main()
