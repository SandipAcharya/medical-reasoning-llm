"""
qlora.py
────────
Attaches LoRA adapters to the quantized base model via PEFT,
and provides save / load utilities for the trained adapter weights.

Why LoRA on a quantized model (QLoRA)?
  The 4-bit base weights are frozen. LoRA injects tiny trainable rank-r
  matrices (A, B) beside each target linear layer. Gradients flow only
  through these ~24M trainable params, not the 3B frozen base params.
  Memory: ~8 GB peak on T4 vs ~24 GB for full fine-tuning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from omegaconf import DictConfig
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import PreTrainedModel

from .base import ModelConfig, load_base_model, load_tokenizer

logger = logging.getLogger(__name__)


# ─── Config ────────────────────────────────────────────────────────────────────

@dataclass
class QLoRAConfig:
    """Mirrors the `lora:` block in training_config.yaml."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> "QLoRAConfig":
        return cls(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
            target_modules=list(cfg.lora.target_modules),
        )

    def to_peft_config(self) -> LoraConfig:
        """Convert to a PEFT LoraConfig object."""
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.target_modules,
        )

    @property
    def scaling(self) -> float:
        """LoRA scaling factor: alpha / r."""
        return self.lora_alpha / self.r


# ─── Public API ────────────────────────────────────────────────────────────────

def attach_lora(
    model: PreTrainedModel,
    lora_config: QLoRAConfig,
    gradient_checkpointing: bool = True,
) -> PreTrainedModel:
    """
    Prepare a quantized model for k-bit training and attach LoRA adapters.

    Steps:
    1. `prepare_model_for_kbit_training` — enables gradient flow through
       the quantized base and casts layer norms to float32.
    2. Enable gradient checkpointing to reduce VRAM at the cost of ~20%
       slower training.
    3. Inject LoRA adapter layers via `get_peft_model`.

    Parameters
    ----------
    model : PreTrainedModel
        The quantized base model (loaded with load_in_4bit=True).
    lora_config : QLoRAConfig
        LoRA hyperparameters.
    gradient_checkpointing : bool
        Strongly recommended True for T4 (16 GB VRAM) training.

    Returns
    -------
    PeftModel
        The model with LoRA adapters attached and ready for training.
    """
    logger.info(
        "Attaching LoRA: r=%d | alpha=%d | scaling=%.2f | dropout=%.2f | targets=%s",
        lora_config.r,
        lora_config.lora_alpha,
        lora_config.scaling,
        lora_config.lora_dropout,
        lora_config.target_modules,
    )

    # Step 1: Prepare quantized model for gradient flow
    model = prepare_model_for_kbit_training(model)

    # Step 2: Gradient checkpointing
    if gradient_checkpointing:
        model.enable_input_require_grads()
        logger.info("Gradient checkpointing: enabled")

    # Step 3: Inject LoRA adapters
    peft_config = lora_config.to_peft_config()
    model = get_peft_model(model, peft_config)

    _log_trainable_params(model)
    return model


def save_adapter(
    model: PeftModel,
    output_dir: Union[str, Path],
    adapter_name: str = "default",
) -> Path:
    """
    Save only the LoRA adapter weights (not the full model).

    The saved directory will contain:
      - adapter_model.safetensors  (~100 MB vs ~6 GB for the full model)
      - adapter_config.json
      - tokenizer files

    Parameters
    ----------
    model : PeftModel
        The fine-tuned model with attached adapters.
    output_dir : str | Path
        Directory to save adapter weights.
    adapter_name : str
        PEFT adapter name (default: "default").

    Returns
    -------
    Path
        The directory where the adapter was saved.
    """
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(save_path))
    logger.info("LoRA adapter saved to: %s", save_path)

    # Log adapter file sizes
    for f in save_path.iterdir():
        size_mb = f.stat().st_size / 1024**2
        logger.info("  %s  (%.1f MB)", f.name, size_mb)

    return save_path


def load_adapter(
    base_model: PreTrainedModel,
    adapter_path: Union[str, Path],
    is_trainable: bool = False,
) -> PeftModel:
    """
    Load a previously saved LoRA adapter onto a base model.

    Parameters
    ----------
    base_model : PreTrainedModel
        The base model (should match the one used for training).
    adapter_path : str | Path
        Directory containing the saved adapter weights.
    is_trainable : bool
        Set True only if continuing to train. False for inference.

    Returns
    -------
    PeftModel
        The base model with the adapter loaded.
    """
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir.resolve()}\n"
            f"Run training first: python scripts/train.py"
        )

    logger.info("Loading adapter from: %s", adapter_dir)
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        is_trainable=is_trainable,
    )
    logger.info("Adapter loaded successfully")
    return model


def merge_and_unload(model: PeftModel, output_dir: Optional[Union[str, Path]] = None) -> PreTrainedModel:
    """
    Merge LoRA weights into the base model and return a standard model.

    Useful for deployment — merged model runs without PEFT overhead.
    Note: requires enough RAM/VRAM to hold the full float16 model (~6 GB).

    Parameters
    ----------
    model : PeftModel
        The fine-tuned PEFT model.
    output_dir : optional str | Path
        If provided, save the merged model here.

    Returns
    -------
    PreTrainedModel
        The merged (non-PEFT) model.
    """
    logger.info("Merging LoRA weights into base model (requires ~6 GB VRAM)...")
    merged = model.merge_and_unload()
    logger.info("Merge complete")

    if output_dir:
        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(save_path))
        logger.info("Merged model saved to: %s", save_path)

    return merged


# ─── Logging helper ────────────────────────────────────────────────────────────

def _log_trainable_params(model: PeftModel) -> None:
    """Log number of trainable vs frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    logger.info(
        "Parameter summary:\n"
        "  Trainable (LoRA adapters) : %10d  (%.4f%%)\n"
        "  Frozen    (quantized base): %10d  (%.4f%%)\n"
        "  Total                     : %10d",
        trainable, 100.0 * trainable / total,
        frozen, 100.0 * frozen / total,
        total,
    )