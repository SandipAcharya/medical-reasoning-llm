"""
scripts/infer.py
────────────────
CLI inference tool. Ask the trained model any clinical question.

Usage
-----
    # Single question (interactive):
    python scripts/infer.py --adapter_path results/final_adapter

    # Single question (non-interactive):
    python scripts/infer.py \\
        --adapter_path results/final_adapter \\
        --question "45yo male, crushing chest pain, ST elevation V1-V4..."

    # Batch from JSON file:
    python scripts/infer.py \\
        --adapter_path results/final_adapter \\
        --input_file data/custom_questions.json \\
        --output_file results/predictions.json
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

from src.medical_reasoning.data.utils import load_config
from src.medical_reasoning.inference.pipeline import MedicalReasoningPipeline

load_dotenv()
app = typer.Typer(help="Run inference with a trained Medical Reasoning LLM.")

logging.basicConfig(
    level=logging.WARNING,  # Suppress INFO during interactive use
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@app.command()
def main(
    adapter_path: str = typer.Option(
        ..., "--adapter_path", "-a",
        help="Path to the saved LoRA adapter",
    ),
    config: str = typer.Option(
        "config/training_config.yaml",
        "--config", "-c",
    ),
    question: Optional[str] = typer.Option(
        None, "--question", "-q",
        help="Single clinical question (use quotes). If omitted, enters interactive mode.",
    ),
    input_file: Optional[str] = typer.Option(
        None, "--input_file",
        help="JSON file with list of questions: [{\"question\": \"...\"}]",
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output_file",
        help="JSON file to write batch predictions to",
    ),
    max_new_tokens: int = typer.Option(1024, "--max_new_tokens"),
    load_in_4bit: bool = typer.Option(True, "--load_in_4bit"),
    base_model_only: bool = typer.Option(
        False, "--base_only",
        help="Run the base model without adapter (for comparison)",
    ),
) -> None:
    """
    Interactive or batch inference with the fine-tuned medical reasoning model.
    """
    _banner()

    cfg = load_config(config)

    # Load pipeline
    pipeline = MedicalReasoningPipeline.from_pretrained(
        base_model=cfg.model.name,
        adapter_path=None if base_model_only else adapter_path,
        load_in_4bit=load_in_4bit,
        system_message=cfg.prompt.system_message,
        max_new_tokens=max_new_tokens,
    )
    print("✓ Model loaded\n")

    # ── Mode: single question (CLI arg) ───────────────────────────────────────
    if question:
        result = pipeline.reason(question)
        print(result)
        return

    # ── Mode: batch from file ─────────────────────────────────────────────────
    if input_file:
        with open(input_file) as f:
            items = json.load(f)

        questions = [item["question"] for item in items]
        print(f"Running batch inference on {len(questions)} questions...")
        results = pipeline.reason_batch(questions)

        output = [
            {
                "question": r.question,
                "reasoning_chain": r.reasoning_chain,
                "final_answer": r.final_answer,
                "generation_time_s": r.generation_time_s,
                "num_tokens_generated": r.num_tokens_generated,
            }
            for r in results
        ]

        out_path = Path(output_file or "results/predictions.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"✓ Predictions saved to: {out_path}")
        return

    # ── Mode: interactive REPL ────────────────────────────────────────────────
    _interactive_loop(pipeline)


def _interactive_loop(pipeline: MedicalReasoningPipeline) -> None:
    """Run an interactive REPL for clinical question answering."""
    print("=" * 70)
    print("  MEDICAL REASONING LLM — Interactive Mode")
    print("  Type a clinical scenario and press Enter.")
    print("  Commands: 'quit' or 'exit' to stop | 'clear' to reset")
    print("=" * 70)
    print()

    while True:
        try:
            question = input("Clinical Question > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        if question.lower() == "clear":
            print("\n" * 3)
            continue
        if not question:
            continue

        print("\nGenerating...\n")
        result = pipeline.reason(question)
        print(result)


def _banner() -> None:
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════╗\n"
        "║        Medical Reasoning LLM — Inference                 ║\n"
        "╚══════════════════════════════════════════════════════════╝\n"
    )


if __name__ == "__main__":
    app()