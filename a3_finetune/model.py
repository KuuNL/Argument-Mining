"""Qwen3 in 4-bit (NF4) with LoRA adapters, the answer-only loss, and constrained generation (D065).

Memory path, fixed in the plan:
- the model is wrapped in PEFT directly, without `prepare_model_for_kbit_training`, so the 16-bit embeddings
  and output layer are never copied to 32-bit;
- the loss reads hidden states at answer positions only and feeds them to Liger's fused linear cross-entropy,
  so full-vocabulary logits (152k words) are never stored for the prompt;
- gradient checkpointing; adapters train in 32-bit, the frozen layers stay in 4-bit.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELS = {"qwen3-8b": "Qwen/Qwen3-8B", "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507"}
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    _flce = LigerFusedLinearCrossEntropyLoss()
except Exception:                                          # pragma: no cover - fall back to plain logits
    _flce = None


def tokenizer(key: str):
    return AutoTokenizer.from_pretrained(MODELS[key])


def load(key: str, adapter=None, train: bool = False):
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODELS[key], quantization_config=q, dtype=torch.bfloat16,
                                                 device_map={"": 0}, attn_implementation="sdpa")
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
