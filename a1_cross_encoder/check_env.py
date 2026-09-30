"""Phase 0 smoke test: can this machine fine-tune ModernBERT on the GPU?

Uses random token ids only (no data, no scoring). For each probe it runs a few real
training steps (bf16 autocast, fp32 master weights, AdamW) and reports peak VRAM and
step time, so the base-vs-large choice (D006) and the attention kernel (D008) rest on
measurements.

    # from /mnt/c/Argument Mining inside WSL (D008)
    ~/.venvs/dialam/bin/python -m a1_cross_encoder.check_env
"""
import importlib.util
import platform
import time

import torch
import transformers
from transformers import AutoModelForSequenceClassification

transformers.logging.set_verbosity_error()  # the fresh 5-way head is expected, not a warning

NUM_LABELS = 5  # A->B Inference, B->A Inference, Conflict, Rephrase, None
BASE, LARGE = "answerdotai/ModernBERT-base", "answerdotai/ModernBERT-large"
PROBES = [
    # (model id, sequence length, batch size, gradient checkpointing, attention, torch.compile)
    (BASE, 2048, 1, False, "sdpa", False),
    (BASE, 2048, 1, False, "sdpa", True),
    (BASE, 2048, 1, False, "flex_attention", False),
    (BASE, 2048, 1, False, "flash_attention_2", False),
    (LARGE, 1024, 2, True, "sdpa", False),
]
WARMUP_STEPS, TIMED_STEPS = 2, 5


def report_system():
    print(f"python        {platform.python_version()}")
    print(f"torch         {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"transformers  {transformers.__version__}")
    assert torch.cuda.is_available(), "CUDA not available"
    props = torch.cuda.get_device_properties(0)
    arch = f"sm_{props.major}{props.minor}"
    print(f"gpu           {props.name}, {props.total_memory / 2**30:.1f} GiB, {arch}")
    assert arch in torch.cuda.get_arch_list(), f"{arch} not in torch build {torch.cuda.get_arch_list()}"
    assert torch.cuda.is_bf16_supported(), "bf16 not supported"
    # A kernel must actually launch on this architecture, not just import.
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    assert torch.isfinite(x @ x).all()
    print("bf16 matmul   ok")


def probe(model_id, seq_len, batch, grad_ckpt, attn, compile_):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=NUM_LABELS, attn_implementation=attn
    ).cuda()
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    if compile_:  # transformers 5.x dropped ModernBERT's `reference_compile`; use torch's own
        model.compile()
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # Random ids above the special-token range, so nothing is treated as padding.
    ids = torch.randint(1000, model.config.vocab_size - 1000, (batch, seq_len), device="cuda")
    mask = torch.ones_like(ids)
    labels = torch.randint(0, NUM_LABELS, (batch,), device="cuda")

    times = []
    for step in range(WARMUP_STEPS + TIMED_STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if step >= WARMUP_STEPS:
            times.append(time.perf_counter() - t0)
    assert torch.isfinite(loss), "loss is not finite"

    step_s = sorted(times)[len(times) // 2]
    return {
        "params_M": round(n_params),
        "attn": model.config._attn_implementation,
        "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "step_s": round(step_s, 3),
        "tokens_per_s": round(batch * seq_len / step_s),
    }


def main():
    report_system()
    for model_id, seq_len, batch, grad_ckpt, attn, compile_ in PROBES:
        name = (f"{model_id.split('/')[-1]} len={seq_len} batch={batch} ckpt={grad_ckpt} "
                f"attn={attn} compile={compile_}")
        if attn == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
            print(f"{name:95s} SKIPPED (flash_attn not installed)")
            continue
        try:
            print(f"{name:95s} {probe(model_id, seq_len, batch, grad_ckpt, attn, compile_)}")
        except torch.OutOfMemoryError:
            print(f"{name:95s} OUT OF MEMORY")
        except Exception as e:  # a kernel option that fails here is a finding, not a crash
            print(f"{name:95s} FAILED: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
