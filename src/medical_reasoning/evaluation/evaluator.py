"""
evaluator.py
────────────
Full evaluation loop: runs inference on the test split, parses outputs,
computes all metrics, and generates a structured JSON report with
qualitative examples.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Union

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from .metrics import (
    MetricsBundle,
    compute_accuracy,
    compute_chain_stats,
    compute_rouge_l,
)

logger = logging.getLogger(__name__)


class MedicalEvaluator:
    """
    Evaluates a fine-tuned medical reasoning model on the test split.

    Parameters
    ----------
    model : PreTrainedModel
        The fine-tuned model (with or without adapters loaded).
    tokenizer : PreTrainedTokenizer
    system_message : str
        Physician system prompt used during training.
    reasoning_header : str
        String that marks the start of the reasoning chain in the output.
    answer_header : str
        String that marks the start of the final answer in the output.
    max_new_tokens : int
        Token budget for generation per example.
    batch_size : int
        Number of examples to generate in one forward pass.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        system_message: str,
        reasoning_header: str = "Let me reason through this step by step:",
        answer_header: str = "Final Answer:",
        max_new_tokens: int = 1024,
        batch_size: int = 1,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.system_message = system_message
        self.reasoning_header = reasoning_header
        self.answer_header = answer_header
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_dataset: Dataset,
        question_col: str = "Question",
        cot_col: str = "Complex_CoT",
        answer_col: str = "Response",
        model_name: str = "fine-tuned",
        num_qualitative: int = 10,
    ) -> MetricsBundle:
        """
        Run full evaluation and return a MetricsBundle.

        Parameters
        ----------
        test_dataset : Dataset
            The held-out test split (with original raw columns).
        question_col, cot_col, answer_col : str
            Column names in the dataset.
        model_name : str
            Label for this model in the report.
        num_qualitative : int
            Number of qualitative examples to save.

        Returns
        -------
        MetricsBundle
        """
        logger.info("Starting evaluation on %d examples...", len(test_dataset))
        start = time.time()

        questions = test_dataset[question_col]
        gold_cots = test_dataset[cot_col]
        gold_answers = test_dataset[answer_col]

        # Generate outputs
        generated_full = self._generate_batch(questions)

        # Parse outputs into reasoning + answer components
        generated_chains = [self._extract_reasoning(o) for o in generated_full]
        generated_answers = [self._extract_answer(o) for o in generated_full]

        # Compute metrics
        accuracy_metrics = compute_accuracy(generated_answers, list(gold_answers))
        rouge_metrics = compute_rouge_l(generated_chains, list(gold_cots))
        chain_metrics = compute_chain_stats(
            generated_chains,
            min_steps=3,
            tokenizer=self.tokenizer,
        )

        bundle = MetricsBundle(
            model_name=model_name,
            **accuracy_metrics,
            **rouge_metrics,
            **chain_metrics,
        )

        elapsed = (time.time() - start) / 60
        logger.info("Evaluation complete in %.1f minutes", elapsed)
        logger.info("\n%s", bundle)

        # Store qualitative examples in extra
        bundle.extra["qualitative_examples"] = self._build_qualitative(
            questions, gold_cots, gold_answers,
            generated_chains, generated_answers,
            n=num_qualitative,
        )

        return bundle

    def save_report(
        self,
        bundle: MetricsBundle,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Save the evaluation report as a JSON file.

        Parameters
        ----------
        bundle : MetricsBundle
        output_path : str | Path

        Returns
        -------
        Path
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": bundle.model_name,
            "metrics": {
                "accuracy": bundle.accuracy,
                "num_correct": bundle.num_correct,
                "num_total": bundle.num_total,
                "rouge_l_precision": bundle.rouge_l_precision,
                "rouge_l_recall": bundle.rouge_l_recall,
                "rouge_l_fmeasure": bundle.rouge_l_fmeasure,
                "chain_completeness_rate": bundle.chain_completeness_rate,
                "avg_chain_length_tokens": bundle.avg_chain_length_tokens,
                "avg_steps_per_chain": bundle.avg_steps_per_chain,
                "empty_rate": bundle.empty_rate,
            },
            "qualitative_examples": bundle.extra.get("qualitative_examples", []),
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Evaluation report saved: %s", path)
        return path

    # ── Private helpers ────────────────────────────────────────────────────────

    def _generate_batch(self, questions: List[str]) -> List[str]:
        """Run inference on all questions and return raw generated strings."""
        outputs = []
        self.model.eval()

        for q in tqdm(questions, desc="Generating", unit="example"):
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": q},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(decoded)

        return outputs

    def _extract_answer(self, text: str) -> str:
        """Parse the final answer from a generated output."""
        if self.answer_header in text:
            return text.split(self.answer_header, 1)[-1].strip()
        return text.strip().split("\n\n")[-1].strip()

    def _extract_reasoning(self, text: str) -> str:
        """Parse the reasoning chain from a generated output."""
        start = self.reasoning_header
        end = self.answer_header
        if start in text and end in text:
            chain = text.split(start, 1)[-1].split(end, 1)[0]
            return chain.strip()
        if end in text:
            return text.split(end, 1)[0].strip()
        return text.strip()

    def _build_qualitative(
        self,
        questions: List[str],
        gold_cots: List[str],
        gold_answers: List[str],
        pred_chains: List[str],
        pred_answers: List[str],
        n: int = 10,
    ) -> List[dict]:
        """Build a list of n qualitative example dicts for the report."""
        examples = []
        for i in range(min(n, len(questions))):
            examples.append(
                {
                    "index": i,
                    "question": questions[i],
                    "gold_answer": gold_answers[i],
                    "generated_answer": pred_answers[i],
                    "answer_correct": (
                        gold_answers[i].lower().strip()
                        == pred_answers[i].lower().strip()
                    ),
                    "gold_reasoning_excerpt": gold_cots[i][:500],
                    "generated_reasoning_excerpt": pred_chains[i][:500],
                }
            )
        return examples