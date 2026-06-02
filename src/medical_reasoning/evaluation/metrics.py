"""
metrics.py
──────────
All evaluation metrics:

  compute_rouge_l       — ROUGE-L between generated and gold CoT
  compute_accuracy      — exact-match on final answer
  compute_chain_stats   — structural quality of reasoning chains
  MetricsBundle         — aggregated result container
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from rouge_score import rouge_scorer


# ─── Normalisation ─────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """
    Normalize text for evaluation:
    - lowercase
    - strip punctuation (except hyphens in medical terms)
    - collapse whitespace
    """
    text = text.lower().strip()
    # Remove punctuation except hyphen
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Individual Metrics ────────────────────────────────────────────────────────

def compute_rouge_l(
    predictions: List[str],
    references: List[str],
    use_stemmer: bool = False,
) -> dict:
    """
    Compute ROUGE-L between lists of predicted and reference strings.

    Parameters
    ----------
    predictions : list of str
        Generated reasoning chains / answers.
    references : list of str
        Gold reasoning chains / answers.
    use_stemmer : bool
        Apply Porter stemmer before scoring.

    Returns
    -------
    dict with keys: rouge_l_precision, rouge_l_recall, rouge_l_fmeasure
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions ({len(predictions)}) and references ({len(references)}) "
            f"must have the same length."
        )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)

    p_scores, r_scores, f_scores = [], [], []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        p_scores.append(score["rougeL"].precision)
        r_scores.append(score["rougeL"].recall)
        f_scores.append(score["rougeL"].fmeasure)

    return {
        "rouge_l_precision": round(sum(p_scores) / len(p_scores), 4),
        "rouge_l_recall": round(sum(r_scores) / len(r_scores), 4),
        "rouge_l_fmeasure": round(sum(f_scores) / len(f_scores), 4),
    }


def compute_accuracy(
    predictions: List[str],
    references: List[str],
    normalize: bool = True,
) -> dict:
    """
    Compute exact-match accuracy on final answers.

    Parameters
    ----------
    predictions : list of str
        Predicted final answers (after parsing out the reasoning chain).
    references : list of str
        Gold final answers.
    normalize : bool
        Apply text normalization before comparison.

    Returns
    -------
    dict with keys: accuracy, num_correct, num_total
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions ({len(predictions)}) and references ({len(references)}) "
            f"must have the same length."
        )

    fn = _normalize if normalize else (lambda x: x)
    correct = sum(fn(p) == fn(r) for p, r in zip(predictions, references))

    return {
        "accuracy": round(correct / len(predictions), 4) if predictions else 0.0,
        "num_correct": correct,
        "num_total": len(predictions),
    }


def compute_chain_stats(
    generated_chains: List[str],
    min_steps: int = 3,
    tokenizer=None,
) -> dict:
    """
    Compute structural quality statistics for reasoning chains.

    Parameters
    ----------
    generated_chains : list of str
        The reasoning chain portion of each generated output.
    min_steps : int
        Minimum number of sentences to consider a chain "complete".
    tokenizer : optional
        If provided, compute token lengths; otherwise use character lengths.

    Returns
    -------
    dict with keys:
      - chain_completeness_rate  : fraction with >= min_steps sentences
      - avg_chain_length_chars   : mean character length
      - avg_chain_length_tokens  : mean token count (if tokenizer provided)
      - empty_rate               : fraction of empty/very short chains
    """
    lengths_chars = [len(c) for c in generated_chains]
    n = len(generated_chains)

    # Count sentences as a proxy for reasoning steps
    def count_steps(text: str) -> int:
        return max(1, len(re.split(r"[.!?]\s+", text.strip())))

    step_counts = [count_steps(c) for c in generated_chains]
    complete = sum(s >= min_steps for s in step_counts)
    empty = sum(len(c.strip()) < 20 for c in generated_chains)

    result = {
        "chain_completeness_rate": round(complete / n, 4) if n else 0.0,
        "avg_chain_length_chars": round(sum(lengths_chars) / n, 1) if n else 0.0,
        "avg_steps_per_chain": round(sum(step_counts) / n, 2) if n else 0.0,
        "empty_rate": round(empty / n, 4) if n else 0.0,
    }

    if tokenizer is not None:
        lengths_tokens = [
            len(tokenizer.encode(c, add_special_tokens=False))
            for c in generated_chains
        ]
        result["avg_chain_length_tokens"] = round(sum(lengths_tokens) / n, 1) if n else 0.0

    return result


# ─── MetricsBundle ─────────────────────────────────────────────────────────────

@dataclass
class MetricsBundle:
    """
    Container for all evaluation metrics from a single eval run.

    Attributes
    ----------
    accuracy : float
        Exact-match accuracy on final answers.
    rouge_l_fmeasure : float
        ROUGE-L F1 between generated and gold reasoning chains.
    chain_completeness_rate : float
        Fraction of outputs with >= min_steps reasoning steps.
    avg_chain_length_tokens : float
        Mean token count of generated reasoning chains.
    num_total : int
        Number of examples evaluated.
    model_name : str
        Identifier for the model (for comparison tables).
    extra : dict
        Any additional metrics (e.g. per-category breakdowns).
    """

    accuracy: float = 0.0
    rouge_l_precision: float = 0.0
    rouge_l_recall: float = 0.0
    rouge_l_fmeasure: float = 0.0
    chain_completeness_rate: float = 0.0
    avg_chain_length_chars: float = 0.0
    avg_chain_length_tokens: float = 0.0
    avg_steps_per_chain: float = 0.0
    empty_rate: float = 0.0
    num_correct: int = 0
    num_total: int = 0
    model_name: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"MetricsBundle({self.model_name})\n"
            f"  Accuracy               : {self.accuracy:.4f} "
            f"({self.num_correct}/{self.num_total})\n"
            f"  ROUGE-L F1             : {self.rouge_l_fmeasure:.4f}\n"
            f"  Chain completeness     : {self.chain_completeness_rate:.4f}\n"
            f"  Avg chain len (tokens) : {self.avg_chain_length_tokens:.1f}\n"
            f"  Empty chain rate       : {self.empty_rate:.4f}"
        )