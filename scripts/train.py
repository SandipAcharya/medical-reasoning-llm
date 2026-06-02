"""
scripts/train.py
────────────────
CLI entry point for training the medical reasoning model.

Usage
-----
    python scripts/train.py --config config/training_config.yaml

    # Override individual hyperparameters:
    python scripts/train.py \
        --config config/training_config.yaml \
        --num_samples 2000 \
        --num_epochs 1 \
        --output_dir ./results/quick_test
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

# ── Make src importable when run as a script ───────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.medical_reasoning.data.dataset import DataConfig, MedicalReasoningDataset
from src.medical_reasoning.data.preprocessor import ChatTemplatePreprocessor, PromptConfig
from src.medical_reasoning.data.utils import load_config, count_parameters, ensure_dir
from src.medical_reasoning.models.base import ModelConfig, load_model_and_tokenizer
from src.medical_reasoning.models.qlora import QLoRAConfig, attach_lora
from src.medical_reasoning.training.trainer import MedicalSFTTrainer, TrainingConfig
from src.medical_reasoning.training.callbacks import (
    SampleGenerationCallback,
    RichProgressCallback,
)

load_dotenv()
app = typer.Typer(help="Train the Medical Reasoning LLM via QLoRA.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Main command ──────────────────────────────────────────────────────────────

@app.command()
def main(
    config: str = typer.Option(
        "config/training_config.yaml",
        "--config", "-c",
        help="Path to training_config.yaml",
    ),
    output_dir: str = typer.Option(
        None, "--output_dir", "-o",
        help="Override output directory from config",
    ),
    num_samples: int = typer.Option(
        None, "--num_samples", "-n",
        help="Override number of training samples",
    ),
    num_epochs: int = typer.Option(
        None, "--num_epochs", "-e",
        help="Override number of training epochs",
    ),
    skip_eval_base: bool = typer.Option(
        False, "--skip_eval_base",
        help="Skip base model evaluation (saves time)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry_run",
        help="Load everything and stop before training (sanity check)",
    ),
) -> None:
    """
    Fine-tune Qwen2.5-3B-Instruct on medical chain-of-thought QA via QLoRA.
    """
    _banner()

    # ── 1. Load & patch config ─────────────────────────────────────────────────
    logger.info("Loading config: %s", config)
    cfg = load_config(config)

    if output_dir:
        cfg.paths.output_dir = output_dir
    if num_samples:
        cfg.data.num_samples = num_samples
    if num_epochs:
        cfg.training.num_train_epochs = num_epochs

    ensure_dir(cfg.paths.output_dir)
    ensure_dir(cfg.paths.log_dir)

    # ── 2. Load dataset ────────────────────────────────────────────────────────
    logger.info("Step 1/5 — Loading dataset")
    data_cfg = DataConfig.from_omegaconf(cfg)
    ds_loader = MedicalReasoningDataset(data_cfg)
    splits = ds_loader.get_splits()
    logger.info("Dataset summary: %s", ds_loader.summary())

    # ── 3. Load model + tokenizer ──────────────────────────────────────────────
    logger.info("Step 2/5 — Loading model and tokenizer")
    model_cfg = ModelConfig.from_omegaconf(cfg)
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    # ── 4. Preprocess dataset ──────────────────────────────────────────────────
    logger.info("Step 3/5 — Preprocessing dataset")
    prompt_cfg = PromptConfig(
        system_message=cfg.prompt.system_message,
        reasoning_header=cfg.prompt.reasoning_header,
        answer_header=cfg.prompt.answer_header,
        question_column=cfg.data.question_column,
        cot_column=cfg.data.cot_column,
        answer_column=cfg.data.answer_column,
    )
    preprocessor = ChatTemplatePreprocessor(
        tokenizer=tokenizer,
        prompt_config=prompt_cfg,
        max_seq_length=cfg.data.max_seq_length,
    )

    formatted_splits = {
        "train": preprocessor.apply_to_dataset(splits["train"]),
        "validation": preprocessor.apply_to_dataset(splits["validation"]),
        "test": splits["test"],    # Keep raw for evaluation
    }

    if dry_run:
        logger.info("DRY RUN complete — all components loaded successfully.")
        logger.info("Train: %d | Val: %d | Test: %d",
                    len(formatted_splits["train"]),
                    len(formatted_splits["validation"]),
                    len(splits["test"]))
        raise typer.Exit(0)

    # ── 5. Attach LoRA ─────────────────────────────────────────────────────────
    logger.info("Step 4/5 — Attaching LoRA adapters")
    lora_cfg = QLoRAConfig.from_omegaconf(cfg)
    model = attach_lora(model, lora_cfg, gradient_checkpointing=True)

    param_info = count_parameters(model)
    logger.info(
        "Trainable params: %d (%.4f%% of total %d)",
        param_info["trainable"],
        param_info["trainable_pct"],
        param_info["total"],
    )

    # ── 6. Build callbacks ─────────────────────────────────────────────────────
    callbacks = [
        RichProgressCallback(),
        SampleGenerationCallback(
            tokenizer=tokenizer,
            system_message=cfg.prompt.system_message,
            log_path=str(Path(cfg.paths.output_dir) / "sample_generations.txt"),
        ),
    ]

    # ── 7. Train ───────────────────────────────────────────────────────────────
    logger.info("Step 5/5 — Training")
    train_cfg = TrainingConfig.from_omegaconf(cfg)
    trainer = MedicalSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset_splits=formatted_splits,
        config=train_cfg,
        callbacks=callbacks,
    )

    metrics = trainer.train()
    adapter_path = trainer.save()

    logger.info("Training metrics: %s", metrics)
    logger.info("Adapter saved to: %s", adapter_path)
    logger.info("\n✅  Training complete! Run evaluation with:")
    logger.info("   python scripts/evaluate.py --adapter_path %s", adapter_path)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _banner() -> None:
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║        Medical Reasoning LLM — QLoRA Training            ║\n"
        "║  Qwen2.5-3B-Instruct × medical-o1-reasoning-SFT         ║\n"
        "╚══════════════════════════════════════════════════════════╝\n"
    )


if __name__ == "__main__":
    app()