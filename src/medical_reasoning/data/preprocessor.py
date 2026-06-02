"""
preprocessor.py
───────────────
Converts raw dataset rows into the Qwen2.5-Instruct chat format with
the physician system prompt, reasoning chain, and final answer baked in.

The final `text` field fed to SFTTrainer looks like:

    <|im_start|>system
    You are an expert physician...
    <|im_end|>
    <|im_start|>user
    A 45-year-old male presents with crushing chest pain...
    <|im_end|>
    <|im_start|>assistant
    Let me reason through this step by step:
    <reasoning chain here>
    Final Answer:
    <answer here>
    <|im_end|>
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


# ─── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PromptConfig:
    """Mirrors the `prompt:` block in training_config.yaml."""

    system_message: str = (
        "You are an expert physician with deep knowledge of internal medicine, "
        "cardiology, neurology, and emergency medicine. When presented with a "
        "clinical scenario, you reason through it systematically before providing "
        "your final answer. Your reasoning should include: symptom analysis, "
        "relevant anatomy and physiology, differential diagnosis with reasoning, "
        "and evidence-based management. Always separate your thinking process "
        "from your final answer."
    )
    reasoning_header: str = "Let me reason through this step by step:"
    answer_header: str = "Final Answer:"

    question_column: str = "Question"
    cot_column: str = "Complex_CoT"
    answer_column: str = "Response"


# ─── Preprocessor ──────────────────────────────────────────────────────────────

class ChatTemplatePreprocessor:
    """
    Applies the Qwen2.5-Instruct chat template to each dataset row.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        The model tokenizer — used for `apply_chat_template`.
    prompt_config : PromptConfig
        System prompt and header strings.
    max_seq_length : int
        Sequences longer than this are discarded (not truncated mid-reasoning).
    text_output_column : str
        Name of the output column written to the dataset. Default: "text".
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_config: PromptConfig,
        max_seq_length: int = 2048,
        text_output_column: str = "text",
    ) -> None:
        self.tokenizer = tokenizer
        self.cfg = prompt_config
        self.max_seq_length = max_seq_length
        self.text_output_column = text_output_column

    # ── Public API ─────────────────────────────────────────────────────────────

    def format_example(self, row: Dict[str, Any]) -> str:
        """
        Convert a single dataset row to a fully-formatted chat string.

        The assistant turn combines the reasoning chain and the final answer,
        separated by the configured headers so evaluation can parse them apart.
        """
        question = str(row.get(self.cfg.question_column, "")).strip()
        cot = str(row.get(self.cfg.cot_column, "")).strip()
        answer = str(row.get(self.cfg.answer_column, "")).strip()

        assistant_content = (
            f"{self.cfg.reasoning_header}\n"
            f"{cot}\n\n"
            f"{self.cfg.answer_header}\n"
            f"{answer}"
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.cfg.system_message},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def apply_to_dataset(
        self,
        dataset: Dataset,
        remove_original_columns: bool = True,
        filter_long: bool = True,
        num_proc: int = 4,
    ) -> Dataset:
        """
        Apply chat formatting to an entire HuggingFace Dataset.

        Parameters
        ----------
        dataset : Dataset
            The raw dataset split (train / val / test).
        remove_original_columns : bool
            Drop original columns after formatting (saves memory).
        filter_long : bool
            Remove samples whose formatted text exceeds max_seq_length tokens.
        num_proc : int
            Number of parallel workers for the map operation.

        Returns
        -------
        Dataset
            Dataset with a single `text` column (or self.text_output_column).
        """
        logger.info(
            "Applying chat template to %d samples (num_proc=%d)...",
            len(dataset), num_proc,
        )

        original_columns = dataset.column_names

        formatted = dataset.map(
            self._format_row,
            num_proc=num_proc,
            desc="Formatting chat template",
        )

        if remove_original_columns:
            cols_to_drop = [c for c in original_columns if c != self.text_output_column]
            formatted = formatted.remove_columns(cols_to_drop)

        if filter_long:
            before = len(formatted)
            formatted = formatted.filter(
                self._is_within_length_limit,
                num_proc=num_proc,
                desc="Filtering long sequences",
            )
            dropped = before - len(formatted)
            if dropped:
                logger.warning(
                    "Dropped %d / %d samples exceeding max_seq_length=%d",
                    dropped, before, self.max_seq_length,
                )

        logger.info(
            "Formatted dataset size: %d samples", len(formatted)
        )
        return formatted

    def extract_answer(self, full_text: str) -> str:
        """
        Extract just the final answer from a full model output string.
        Useful during evaluation to compare against gold answers.
        """
        marker = self.cfg.answer_header
        if marker in full_text:
            return full_text.split(marker, 1)[-1].strip()
        # Fallback: return last paragraph
        return full_text.strip().split("\n\n")[-1].strip()

    def extract_reasoning(self, full_text: str) -> str:
        """
        Extract just the reasoning chain (between headers) from model output.
        """
        start_marker = self.cfg.reasoning_header
        end_marker = self.cfg.answer_header

        if start_marker in full_text and end_marker in full_text:
            reasoning = full_text.split(start_marker, 1)[-1]
            reasoning = reasoning.split(end_marker, 1)[0]
            return reasoning.strip()

        return full_text.strip()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _format_row(self, row: Dict[str, Any]) -> Dict[str, str]:
        """Wrapper for Dataset.map — adds the `text` column."""
        return {self.text_output_column: self.format_example(row)}

    def _is_within_length_limit(self, row: Dict[str, Any]) -> bool:
        """Return True if the formatted text fits within max_seq_length tokens."""
        text = row.get(self.text_output_column, "")
        # Use fast tokenizer without special tokens for speed
        token_count = len(self.tokenizer.encode(text, add_special_tokens=False))
        return token_count <= self.max_seq_length

    def __repr__(self) -> str:
        return (
            f"ChatTemplatePreprocessor("
            f"max_seq_length={self.max_seq_length}, "
            f"output_column={self.text_output_column!r})"
        )