"""Qwen3 in 4-bit (NF4) with LoRA adapters, the answer-only loss, and constrained generation (D065).

Memory path, fixed in the plan:
- the model is wrapped in PEFT directly, without `prepare_model_for_kbit_training`, so the 16-bit embeddings
  and output layer are never copied to 32-bit;
- the loss reads hidden states at answer positions only and feeds them to Liger's fused linear cross-entropy,
  so full-vocabulary logits (152k words) are never stored for the prompt;
- gradient checkpointing; adapters train in 32-bit, the frozen layers stay in 4-bit.

Loading (D068): the 4-bit model is saved once to the Linux filesystem and loaded from there (2.6 GB read)
instead of quantizing the 8 GB original on every load, whose file cache WSL kept from Windows.
`--save-quantized` writes it and checks that every tensor equals the quantize-on-load model's.

    python -m a3_finetune.model --save-quantized --model=qwen3-4b      -> results/a3/quantized_check.json
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELS = {"qwen3-8b": "Qwen/Qwen3-8B", "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507"}
QUANTIZED = Path.home() / ".cache" / "a3"            # Linux side of WSL: fast, outside the repo
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    _flce = LigerFusedLinearCrossEntropyLoss()
except Exception:                                          # pragma: no cover - fall back to plain logits
    _flce = None


def tokenizer(key: str):
    return AutoTokenizer.from_pretrained(MODELS[key])


def quantized_dir(key: str) -> Path:
    return QUANTIZED / f"{MODELS[key].split('/')[-1]}-nf4"


def load_base(key: str, saved: bool = True):
    """The 4-bit base model: from the saved 4-bit copy when it exists, else quantized from the original."""
    if saved and quantized_dir(key).exists():
        return AutoModelForCausalLM.from_pretrained(quantized_dir(key), dtype=torch.bfloat16, device_map={"": 0},
                                                    attn_implementation="sdpa")
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16)
    return AutoModelForCausalLM.from_pretrained(MODELS[key], quantization_config=q, dtype=torch.bfloat16,
                                                device_map={"": 0}, attn_implementation="sdpa")


def load(key: str, adapter=None, train: bool = False):
    model = load_base(key)
    if train:
        from peft import LoraConfig, get_peft_model
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(**LORA))
    elif adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    return model


def answer_loss(model, ids: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Mean cross-entropy over answer tokens (labels != -100) and their count. ids, labels: [1, T] on the GPU."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=ids).last_hidden_state[:, :-1]
    target = labels[:, 1:]
    keep = target != -100
    h, y = hidden[keep], target[keep]
    if _flce is not None:
        loss = _flce(base.lm_head.weight, h, y)
    else:
        loss = F.cross_entropy(base.lm_head(h).float(), y)
    return loss, int(keep.sum())


def save_quantized(key: str) -> dict:
    """Quantize on load, save the 4-bit model, reload it, and compare every tensor."""
    import gc
    from common import RESULTS
    model = load_base(key, saved=False)
    model.save_pretrained(quantized_dir(key))
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    after = {k: v.detach().cpu() for k, v in load_base(key).state_dict().items()}
    differ = [k for k in before if k not in after or before[k].dtype != after[k].dtype
              or before[k].shape != after[k].shape or not torch.equal(before[k], after[k])]
    size = sum(f.stat().st_size for f in quantized_dir(key).glob("*.safetensors"))
    res = {"model": MODELS[key], "saved_to": str(quantized_dir(key)), "gib_on_disk": round(size / 2**30, 2),
           "tensors": len(before), "missing_after_reload": sorted(set(before) - set(after))[:10],
           "extra_after_reload": sorted(set(after) - set(before))[:10], "tensors_differing": differ[:10],
           "identical": not differ and set(before) == set(after)}
    (RESULTS / "a3" / "quantized_check.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    return res


if __name__ == "__main__":
    if "--save-quantized" in sys.argv:
        key = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--model=")), "qwen3-4b")
        print(json.dumps(save_quantized(key), indent=1))
