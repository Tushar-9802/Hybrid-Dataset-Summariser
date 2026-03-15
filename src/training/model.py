"""
model.py — QLoRA Model Setup + LoRA+ Optimizer Factory
Loads Mistral-7B with 4-bit NF4 quantization, applies LoRA,
creates asymmetric optimizer (LoRA+) with separate LR for A and B matrices.

Hardware target: RTX 5070 Ti (16GB VRAM, Blackwell sm_120)
"""

import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

log = logging.getLogger(__name__)

MODEL_ID = "mistralai/Mistral-7B-v0.1"


def setup_tokenizer(model_id: str = MODEL_ID):
    """Load and configure Mistral tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_model(config: dict, adapter_path: str = None):
    """
    Load Mistral-7B with 4-bit quantization and LoRA.

    Args:
        config: dict with keys: model.name, lora.*, hardware.*
        adapter_path: if resuming, path to existing LoRA adapter

    Returns:
        (model, tokenizer)
    """
    model_name = config.get("model", {}).get("name", MODEL_ID)

    log.info(f"Loading tokenizer: {model_name}")
    tokenizer = setup_tokenizer(model_name)

    # 4-bit NF4 quantization — BF16 compute (native on Blackwell)
    compute_dtype = getattr(torch, config.get("model", {}).get("compute_dtype", "bfloat16"))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.get("model", {}).get("double_quant", True),
    )

    log.info(f"Loading model: {model_name} (4-bit NF4, compute={compute_dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )

    # Gradient checkpointing — mandatory for 16GB VRAM
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # incompatible with gradient checkpointing

    if adapter_path and Path(adapter_path).exists():
        # Resume from existing adapter
        log.info(f"Loading existing LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        # Fresh LoRA
        lora_cfg = config.get("lora", {})
        peft_config = LoraConfig(
            r=lora_cfg.get("rank", 32),
            lora_alpha=lora_cfg.get("alpha", 64),
            lora_dropout=lora_cfg.get("dropout", 0.1),
            target_modules=lora_cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            task_type="CAUSAL_LM",
            bias="none",
        )
        log.info(f"Applying LoRA: r={peft_config.r}, alpha={peft_config.lora_alpha}, "
                 f"modules={peft_config.target_modules}")
        model = get_peft_model(model, peft_config)

    # Log trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # VRAM report
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        log.info(f"VRAM: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    return model, tokenizer


def setup_optimizer(model, config: dict):
    """
    LoRA+ optimizer: asymmetric learning rates for A and B matrices.

    ηB = λ · ηA where λ = config.lora_plus.lr_B / config.lora_plus.lr_A

    Uses bitsandbytes 8-bit AdamW to save ~50% optimizer state VRAM.
    """
    import bitsandbytes as bnb

    lora_plus = config.get("lora_plus", {})
    lr_A = lora_plus.get("lr_A", 1e-4)
    lr_B = lora_plus.get("lr_B", 8e-4)
    weight_decay = config.get("optimizer", {}).get("weight_decay", 0.01)

    log.info(f"LoRA+ optimizer: lr_A={lr_A}, lr_B={lr_B} (ratio={lr_B/lr_A:.0f}x)")

    param_groups = {"A": [], "B": [], "other": []}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "lora_A" in name:
            param_groups["A"].append(param)
        elif "lora_B" in name:
            param_groups["B"].append(param)
        else:
            param_groups["other"].append(param)

    optimizer_groups = []
    if param_groups["A"]:
        optimizer_groups.append({
            "params": param_groups["A"], "lr": lr_A, "weight_decay": weight_decay,
        })
    if param_groups["B"]:
        optimizer_groups.append({
            "params": param_groups["B"], "lr": lr_B, "weight_decay": weight_decay,
        })
    if param_groups["other"]:
        optimizer_groups.append({
            "params": param_groups["other"], "lr": lr_A, "weight_decay": weight_decay,
        })

    log.info(f"Param groups: A={len(param_groups['A'])}, B={len(param_groups['B'])}, "
             f"other={len(param_groups['other'])}")

    optimizer = bnb.optim.AdamW8bit(optimizer_groups)

    return optimizer


def setup_scheduler(optimizer, total_steps: int, config: dict):
    """Cosine scheduler with warmup."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

    sched_cfg = config.get("scheduler", {})
    warmup_ratio = sched_cfg.get("warmup_ratio", 0.05)
    min_lr_ratio = sched_cfg.get("min_lr_ratio", 0.1)

    warmup_steps = max(1, int(total_steps * warmup_ratio))
    cosine_steps = total_steps - warmup_steps

    log.info(f"Scheduler: {warmup_steps} warmup + {cosine_steps} cosine (total {total_steps})")

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=min_lr_ratio * 1e-4)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    return scheduler


def save_checkpoint(model, optimizer, scheduler, epoch, step, path: str, extra: dict = None):
    """Save LoRA adapter + training state."""
    ckpt_dir = Path(path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter (PEFT native format — loadable by evaluate.py)
    model.save_pretrained(str(ckpt_dir))
    log.info(f"Saved adapter to {ckpt_dir}")

    # Save training state (optimizer, scheduler, step)
    state = {
        "epoch": epoch,
        "global_step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if extra:
        state.update(extra)

    torch.save(state, ckpt_dir / "training_state.pt")
    log.info(f"Saved training state (epoch={epoch}, step={step})")


def load_training_state(path: str, optimizer, scheduler):
    """Load optimizer/scheduler state for resume."""
    state_path = Path(path) / "training_state.pt"
    if not state_path.exists():
        log.warning(f"No training state at {state_path}")
        return 0, 0

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    log.info(f"Resumed from epoch={state['epoch']}, step={state['global_step']}")
    return state["epoch"], state["global_step"]