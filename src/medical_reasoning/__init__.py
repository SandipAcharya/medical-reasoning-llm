"""
medical_reasoning
=================
QLoRA fine-tuning of Qwen2.5-3B-Instruct for clinical chain-of-thought reasoning.

Subpackages
-----------
data        — dataset loading, preprocessing, chat-template formatting
models      — model loading with BitsAndBytes quantization + LoRA attachment
training    — SFTTrainer wrapper, callbacks, training loop
evaluation  — ROUGE-L, accuracy, chain quality metrics
inference   — clean inference pipeline for production use
"""

__version__ = "0.1.0"
__author__ = "Sandip Acharya"
__license__ = "MIT"

from pathlib import Path

# Convenience: project root regardless of where code is called from
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"