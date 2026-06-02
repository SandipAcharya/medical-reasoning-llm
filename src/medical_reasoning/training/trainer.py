"""
trainer.py
──────────
Wraps HuggingFace TRL's SFTTrainer with:
  - Automatic hyperparameter setup from YAML config
  - Gradient checkpointing
  - Automatic adapter saving on completion
  - Rich logging of training progress
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from datasets import DatasetDict
from omegaconf import DictConfig
from peft import PeftModel
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)


# ─── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Mirrors the `training:` + `sft:` blocks in training_config.yaml."""

    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False
    optim: str = "paged_adamw_32bit"
    logging_steps: int = 25
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    dataloader_num_workers: int = 2
    remove_unused_columns: bool = False
    report_to: str = "none"
    push_to_hub: bool = False
    hub_model_id: str = ""
    # SFT-specific
    max_seq_length: int = 2048
    dataset_text_field: str = "text"
    packing: bool = False

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> "TrainingConfig":
        t = cfg.training
        s = cfg.sft
        return cls(
            output_dir=cfg.paths.output_dir,
            num_train_epochs=t.num_train_epochs,
            per_device_train_batch_size=t.per_device_train_batch_size,
            per_device_eval_batch_size=t.per_device_eval_batch_size,
            gradient_accumulation_steps=t.gradient_accumulation_steps,
            learning_rate=t.learning_rate,
            lr_scheduler_type=t.lr_scheduler_type,
            warmup_ratio=t.warmup_ratio,
            weight_decay=t.weight_decay,
            max_grad_norm=t.max_grad_norm,
            fp16=t.fp16,
            bf16=t.bf16,
            optim=t.optim,
            logging_steps=t.logging_steps,
            eval_steps=t.eval_steps,
            save_steps=t.save_steps,
            save_total_limit=t.save_total_limit,
            load_best_model_at_end=t.load_best_model_at_end,
            metric_for_best_model=t.metric_for_best_model,
            greater_is_better=t.greater_is_better,
            dataloader_num_workers=t.dataloader_num_workers,
            remove_unused_columns=t.remove_unused_columns,
            report_to=t.report_to,
            push_to_hub=t.push_to_hub,
            hub_model_id=t.hub_model_id,
            max_seq_length=cfg.data.max_seq_length,
            dataset_text_field=s.dataset_text_field,
            packing=s.packing,
        )

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @property
    def adapter_output_dir(self) -> str:
        return str(Path(self.output_dir) / "final_adapter")


# ─── Trainer ───────────────────────────────────────────────────────────────────

class MedicalSFTTrainer:
    """
    Wrapper around TRL SFTTrainer for medical chain-of-thought fine-tuning.

    Usage
    -----
    >>> trainer = MedicalSFTTrainer(model, tokenizer, splits, config)
    >>> trainer.train()
    >>> trainer.save()
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset_splits: DatasetDict,
        config: TrainingConfig,
        callbacks: Optional[list] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.splits = dataset_splits
        self.config = config
        self.callbacks = callbacks or []
        self._trainer: Optional[SFTTrainer] = None

    def build(self) -> "MedicalSFTTrainer":
        """Instantiate the SFTTrainer. Call before train()."""
        self._log_training_plan()

        sft_config = self._build_sft_config()

        self._trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.splits["train"],
            eval_dataset=self.splits.get("validation"),
            args=sft_config,
            callbacks=self.callbacks if self.callbacks else None,
        )

        logger.info("SFTTrainer built successfully")
        return self

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns
        -------
        dict
            Training metrics from the final epoch.
        """
        if self._trainer is None:
            self.build()

        logger.info("=" * 60)
        logger.info("TRAINING START")
        logger.info("  Model        : %s", self.config.output_dir)
        logger.info("  Train samples: %d", len(self.splits["train"]))
        logger.info("  Epochs       : %d", self.config.num_train_epochs)
        logger.info("  Eff. batch   : %d", self.config.effective_batch_size)
        logger.info("  LR           : %g", self.config.learning_rate)
        logger.info("=" * 60)

        result = self._trainer.train()

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("  Train loss   : %.4f", result.training_loss)
        logger.info("  Total steps  : %d", result.global_step)
        logger.info("  Runtime (min): %.1f", result.metrics.get("train_runtime", 0) / 60)
        logger.info("=" * 60)

        return result.metrics

    def save(self, output_dir: Optional[str] = None) -> Path:
        """
        Save the LoRA adapter to disk.

        Parameters
        ----------
        output_dir : str, optional
            Override the output directory from config.

        Returns
        -------
        Path
            Directory where the adapter was saved.
        """
        if self._trainer is None:
            raise RuntimeError("Must call train() before save()")

        save_dir = Path(output_dir or self.config.adapter_output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self._trainer.save_model(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))

        logger.info("Adapter saved to: %s", save_dir)
        return save_dir

    def evaluate(self) -> dict:
        """Run evaluation on the validation split and return metrics."""
        if self._trainer is None:
            raise RuntimeError("Must call build() or train() before evaluate()")
        return self._trainer.evaluate()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_sft_config(self) -> SFTConfig:
        """Build the SFTConfig (extends TrainingArguments for TRL)."""
        cfg = self.config
        output_run_dir = str(Path(cfg.output_dir) / "checkpoints")

        return SFTConfig(
            output_dir=output_run_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            fp16=cfg.fp16,
            bf16=cfg.bf16,
            optim=cfg.optim,
            logging_steps=cfg.logging_steps,
            eval_strategy="steps",
            eval_steps=cfg.eval_steps,
            save_strategy="steps",
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            load_best_model_at_end=cfg.load_best_model_at_end,
            metric_for_best_model=cfg.metric_for_best_model,
            greater_is_better=cfg.greater_is_better,
            dataloader_num_workers=cfg.dataloader_num_workers,
            remove_unused_columns=cfg.remove_unused_columns,
            report_to=cfg.report_to,
            push_to_hub=cfg.push_to_hub,
            hub_model_id=cfg.hub_model_id if cfg.push_to_hub else "",
            # SFT-specific
            max_seq_length=cfg.max_seq_length,
            dataset_text_field=cfg.dataset_text_field,
            packing=cfg.packing,
        )

    def _log_training_plan(self) -> None:
        """Print training configuration summary."""
        cfg = self.config
        total_steps = (
            len(self.splits["train"])
            // cfg.effective_batch_size
            * cfg.num_train_epochs
        )
        logger.info(
            "Training plan:\n"
            "  Samples/epoch  : %d\n"
            "  Steps/epoch    : ~%d\n"
            "  Total steps    : ~%d\n"
            "  Eval every     : %d steps\n"
            "  Save every     : %d steps",
            len(self.splits["train"]),
            len(self.splits["train"]) // cfg.effective_batch_size,
            total_steps,
            cfg.eval_steps,
            cfg.save_steps,
        )