"""
scripts/evaluate.py
───────────────────
CLI entry point for evaluating a trained adapter on the held-out test set.

Usage
-----
    python scripts/evaluate.py \\
        --adapter_path results/final_adapter \\
        --config config/training_config.yaml

    # Compare fine-tuned vs base model:
    python scripts/evaluate.py \\
        --adapter_path results/final_adapter \\
        --compare_base
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.medical_reasoning.data.dataset import DataConfig, MedicalReasoningDataset
from src.medical_reasoning.data.utils import load_config, ensure_dir
from src.medical_reasoning.evaluation.evaluator import MedicalEvaluator
from src.medical_reasoning.inference.pipeline import MedicalReasoningPipeline

load_dotenv()
app = typer.Typer(help="Evaluate a trained Medical Reasoning LLM adapter.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@app.command()
def main(
    adapter_path: str = typer.Option(
        ..., "--adapter_path", "-a",
        help="Path to the saved LoRA adapter directory",
    ),
    config: str = typer.Option(
        "config/training_config.yaml",
        "--config", "-c",
        help="Path to training_config.yaml",
    ),
    output_dir: str = typer.Option(
        "results", "--output_dir", "-o",
        help="Directory to save the evaluation report",
    ),
    num_qualitative: int = typer.Option(
        10, "--num_qualitative",
        help="Number of qualitative examples to include in report",
    ),
    compare_base: bool = typer.Option(
        False, "--compare_base",
        help="Also evaluate the base model (no adapter) for delta computation",
    ),
    max_new_tokens: int = typer.Option(
        1024, "--max_new_tokens",
        help="Max tokens to generate per example",
    ),
) -> None:
    """
    Evaluate the fine-tuned medical reasoning model on the held-out test split.
    Generates a JSON report with metrics and qualitative examples.
    """
    _banner()
    ensure_dir(output_dir)

    # ── 1. Load config & dataset ───────────────────────────────────────────────
    cfg = load_config(config)
    data_cfg = DataConfig.from_omegaconf(cfg)

    logger.info("Loading test split...")
    loader = MedicalReasoningDataset(data_cfg)
    splits = loader.get_splits()
    test_ds = splits["test"]
    logger.info("Test set size: %d", len(test_ds))

    # ── 2. Evaluate fine-tuned model ───────────────────────────────────────────
    logger.info("Loading fine-tuned model from: %s", adapter_path)
    ft_pipeline = MedicalReasoningPipeline.from_pretrained(
        base_model=cfg.model.name,
        adapter_path=adapter_path,
        load_in_4bit=True,
        max_new_tokens=max_new_tokens,
    )

    evaluator = MedicalEvaluator(
        model=ft_pipeline.model,
        tokenizer=ft_pipeline.tokenizer,
        system_message=cfg.prompt.system_message,
        reasoning_header=cfg.prompt.reasoning_header,
        answer_header=cfg.prompt.answer_header,
        max_new_tokens=max_new_tokens,
    )

    ft_bundle = evaluator.evaluate(
        test_dataset=test_ds,
        question_col=cfg.data.question_column,
        cot_col=cfg.data.cot_column,
        answer_col=cfg.data.answer_column,
        model_name="fine-tuned-qlora",
        num_qualitative=num_qualitative,
    )

    ft_report_path = evaluator.save_report(
        ft_bundle,
        Path(output_dir) / "eval_report_finetuned.json",
    )
    logger.info("Fine-tuned report saved: %s", ft_report_path)
    print(f"\n{ft_bundle}\n")

    # ── 3. Optional: evaluate base model ──────────────────────────────────────
    if compare_base:
        logger.info("Evaluating base model (no adapter)...")
        del ft_pipeline  # Free VRAM

        base_pipeline = MedicalReasoningPipeline.from_pretrained(
            base_model=cfg.model.name,
            adapter_path=None,     # no adapter
            load_in_4bit=True,
            max_new_tokens=max_new_tokens,
        )

        base_evaluator = MedicalEvaluator(
            model=base_pipeline.model,
            tokenizer=base_pipeline.tokenizer,
            system_message=cfg.prompt.system_message,
            reasoning_header=cfg.prompt.reasoning_header,
            answer_header=cfg.prompt.answer_header,
            max_new_tokens=max_new_tokens,
        )

        base_bundle = base_evaluator.evaluate(
            test_dataset=test_ds,
            question_col=cfg.data.question_column,
            cot_col=cfg.data.cot_column,
            answer_col=cfg.data.answer_column,
            model_name="base-qwen2.5-3b",
            num_qualitative=num_qualitative,
        )

        base_report_path = base_evaluator.save_report(
            base_bundle,
            Path(output_dir) / "eval_report_base.json",
        )
        logger.info("Base model report saved: %s", base_report_path)

        _print_comparison(base_bundle, ft_bundle)

    logger.info("\n✅  Evaluation complete!")


def _print_comparison(base, finetuned) -> None:
    """Print a side-by-side comparison table."""
    def delta(ft_val, base_val):
        if isinstance(ft_val, float) and isinstance(base_val, float):
            d = ft_val - base_val
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:.4f}"
        return "—"

    metrics = [
        ("Accuracy",            "accuracy"),
        ("ROUGE-L F1",          "rouge_l_fmeasure"),
        ("Chain Completeness",  "chain_completeness_rate"),
        ("Avg Chain Len (tok)", "avg_chain_length_tokens"),
        ("Empty Rate",          "empty_rate"),
    ]

    print("\n" + "=" * 65)
    print(f"{'Metric':<28} {'Base':>10} {'Fine-tuned':>12} {'Δ':>10}")
    print("=" * 65)
    for label, attr in metrics:
        b = getattr(base, attr, "—")
        f = getattr(finetuned, attr, "—")
        d = delta(f, b)
        b_str = f"{b:.4f}" if isinstance(b, float) else str(b)
        f_str = f"{f:.4f}" if isinstance(f, float) else str(f)
        print(f"  {label:<26} {b_str:>10} {f_str:>12} {d:>10}")
    print("=" * 65 + "\n")


def _banner() -> None:
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║        Medical Reasoning LLM — Evaluation                ║\n"
        "╚══════════════════════════════════════════════════════════╝\n"
    )


if __name__ == "__main__":
    app()