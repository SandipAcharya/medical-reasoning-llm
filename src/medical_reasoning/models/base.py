"""
base.py
───────
Loads the base Qwen2.5-3B-Instruct model with optional 4-bit NF4
quantization via BitsAndBytes. Also handles tokenizer loading and
the critical pad_token setup required for Qwen models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


# ─── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Mirrors the `model:` + `quantization:` blocks in training_config.yaml."""

    name: str = "Qwen/Qwen2.5-3B-Instruct"
    trust_remote_code: bool = True
    torch_dtype: str = "float16"

    # BitsAndBytes quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True

    # Tokenizer
    padding_side: str = "right"        # right-padding needed for causal LM training
    add_eos_token: bool = True

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> "ModelConfig":
        return cls(
            name=cfg.model.name,
            trust_remote_code=cfg.model.trust_remote_code,
            torch_dtype=cfg.model.torch_dtype,
            load_in_4bit=cfg.quantization.load_in_4bit,
            bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=cfg.quantization.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
        )


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert a string dtype name to a torch.dtype."""
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(
            f"Unsupported dtype: {dtype_str!r}. Choose from {list(mapping.keys())}"
        )
    return mapping[dtype_str]


def _build_bnb_config(cfg: ModelConfig) -> Optional[BitsAndBytesConfig]:
    """Build a BitsAndBytesConfig from ModelConfig, or None if not quantizing."""
    if not cfg.load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_str_to_dtype(cfg.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


# ─── Public API ────────────────────────────────────────────────────────────────

def load_tokenizer(
    model_name: str,
    padding_side: str = "right",
    trust_remote_code: bool = True,
) -> PreTrainedTokenizer:
    """
    Load and configure the tokenizer for Qwen2.5-Instruct.

    Qwen2.5 uses `<|endoftext|>` as both EOS and PAD by default, which
    causes issues during training. This function ensures pad_token is
    correctly set.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    padding_side : str
        "right" for training (causal LM), "left" for batch inference.
    trust_remote_code : bool
        Required for some models with custom tokenizer code.

    Returns
    -------
    PreTrainedTokenizer
    """
    logger.info("Loading tokenizer: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
    )

    # Qwen2.5: ensure pad_token is set (use eos_token if missing)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token (%r)", tokenizer.eos_token)

    logger.info(
        "Tokenizer loaded | vocab_size=%d | pad_token=%r | eos_token=%r",
        tokenizer.vocab_size,
        tokenizer.pad_token,
        tokenizer.eos_token,
    )
    return tokenizer


def load_base_model(
    config: ModelConfig,
    device_map: str = "auto",
) -> PreTrainedModel:
    """
    Load the base causal LM with optional 4-bit NF4 quantization.

    Parameters
    ----------
    config : ModelConfig
        Model and quantization configuration.
    device_map : str
        "auto" distributes across available GPUs. "cuda:0" pins to one GPU.

    Returns
    -------
    PreTrainedModel
        The loaded (possibly quantized) model.

    Notes
    -----
    With QLoRA (load_in_4bit=True), the model is NOT trainable as-is.
    You must call `attach_lora()` afterward to add trainable LoRA adapters.
    The frozen 4-bit base weights serve only as the reference for the
    adapter's low-rank updates.
    """
    bnb_config = _build_bnb_config(config)
    dtype = _str_to_dtype(config.torch_dtype)

    logger.info("Loading base model: %s", config.name)
    if bnb_config:
        logger.info(
            "QLoRA: 4-bit NF4 quantization | double_quant=%s | compute_dtype=%s",
            config.bnb_4bit_use_double_quant,
            config.bnb_4bit_compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        quantization_config=bnb_config,
        torch_dtype=dtype if not config.load_in_4bit else None,
        device_map=device_map,
        trust_remote_code=config.trust_remote_code,
    )

    # Disable model caching — not needed during training, wastes memory
    model.config.use_cache = False
    # Enable gradient checkpointing compatibility with QLoRA
    model.config.pretraining_tp = 1

    _log_model_info(model, config.name)
    return model


def load_model_and_tokenizer(
    config: ModelConfig,
    device_map: str = "auto",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Convenience function: load both model and tokenizer in one call.

    Returns
    -------
    (model, tokenizer)
    """
    tokenizer = load_tokenizer(
        config.name,
        padding_side="right",
        trust_remote_code=config.trust_remote_code,
    )
    model = load_base_model(config, device_map=device_map)
    return model, tokenizer


# ─── Logging helper ────────────────────────────────────────────────────────────

def _log_model_info(model: PreTrainedModel, name: str) -> None:
    """Log parameter counts and memory footprint."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        "Model loaded: %s | total_params=%.2fB | trainable_params=%d (%.2f%%)",
        name,
        total_params / 1e9,
        trainable_params,
        100.0 * trainable_params / total_params if total_params else 0,
    )

    # GPU memory
    if torch.cuda.is_available():
        mem_mb = torch.cuda.memory_allocated() / 1024**2
        logger.info("GPU memory after load: %.1f MB", mem_mb)