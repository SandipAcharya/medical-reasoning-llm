"""
tests/test_evaluation.py
────────────────────────
Unit tests for all evaluation metrics.
Run with: pytest tests/test_evaluation.py -v
No GPU or model download required.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.medical_reasoning.evaluation.metrics import (
    MetricsBundle,
    _normalize,
    compute_accuracy,
    compute_chain_stats,
    compute_rouge_l,
)


# ─── Normalisation ─────────────────────────────────────────────────────────────

class TestNormalize:
    def test_lowercase(self):
        assert _normalize("STEMI") == "stemi"

    def test_strips_punctuation(self):
        assert _normalize("STEMI.") == "stemi"
        assert _normalize("Diagnosis: MI!") == "diagnosis  mi"

    def test_preserves_hyphen(self):
        assert _normalize("ST-elevation") == "st-elevation"

    def test_collapses_whitespace(self):
        assert _normalize("  too  many   spaces  ") == "too  many   spaces"


# ─── Accuracy ──────────────────────────────────────────────────────────────────

class TestComputeAccuracy:
    def test_perfect_accuracy(self):
        preds = ["STEMI", "Appendicitis", "Pneumonia"]
        refs  = ["STEMI", "Appendicitis", "Pneumonia"]
        result = compute_accuracy(preds, refs)
        assert result["accuracy"] == 1.0
        assert result["num_correct"] == 3
        assert result["num_total"] == 3

    def test_zero_accuracy(self):
        result = compute_accuracy(["a", "b"], ["x", "y"])
        assert result["accuracy"] == 0.0
        assert result["num_correct"] == 0

    def test_partial_accuracy(self):
        result = compute_accuracy(["stemi", "wrong"], ["STEMI", "correct"])
        assert result["accuracy"] == 0.5
        assert result["num_correct"] == 1

    def test_normalization_makes_match(self):
        result = compute_accuracy(["STEMI."], ["stemi"], normalize=True)
        assert result["accuracy"] == 1.0

    def test_no_normalization(self):
        result = compute_accuracy(["STEMI"], ["stemi"], normalize=False)
        assert result["accuracy"] == 0.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            compute_accuracy(["a", "b"], ["x"])

    def test_empty_inputs(self):
        result = compute_accuracy([], [])
        assert result["accuracy"] == 0.0
        assert result["num_total"] == 0


# ─── ROUGE-L ───────────────────────────────────────────────────────────────────

class TestComputeRougeL:
    def test_perfect_rouge(self):
        text = "The patient has inferior STEMI secondary to RCA occlusion."
        result = compute_rouge_l([text], [text])
        assert result["rouge_l_fmeasure"] == pytest.approx(1.0, abs=0.001)

    def test_zero_rouge(self):
        result = compute_rouge_l(
            ["completely unrelated output"],
            ["xyz abc def ghi jkl mno"],
        )
        assert result["rouge_l_fmeasure"] < 0.5

    def test_partial_rouge(self):
        pred = "The patient has inferior STEMI and needs PCI."
        ref = "The patient has inferior STEMI secondary to RCA occlusion and needs immediate PCI."
        result = compute_rouge_l([pred], [ref])
        assert 0 < result["rouge_l_fmeasure"] < 1.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            compute_rouge_l(["a", "b"], ["x"])

    def test_returns_all_three_components(self):
        result = compute_rouge_l(["hello world"], ["hello world"])
        assert "rouge_l_precision" in result
        assert "rouge_l_recall" in result
        assert "rouge_l_fmeasure" in result

    def test_batch_averaging(self):
        preds = ["exact match", "totally wrong"]
        refs  = ["exact match", "completely different text here"]
        result = compute_rouge_l(preds, refs)
        # Mean should be between the two extremes
        assert 0 < result["rouge_l_fmeasure"] < 1.0


# ─── Chain Statistics ──────────────────────────────────────────────────────────

class TestComputeChainStats:
    def test_complete_chains(self):
        chains = [
            "First, I note the chest pain. Second, the ECG shows elevation. "
            "Third, this indicates STEMI. Fourth, PCI is indicated.",
        ] * 5
        result = compute_chain_stats(chains, min_steps=3)
        assert result["chain_completeness_rate"] == 1.0

    def test_empty_chains(self):
        result = compute_chain_stats(["", " ", "x"])
        assert result["empty_rate"] > 0

    def test_avg_length(self):
        chains = ["a" * 100, "a" * 200]
        result = compute_chain_stats(chains)
        assert result["avg_chain_length_chars"] == pytest.approx(150.0)

    def test_empty_list(self):
        result = compute_chain_stats([])
        assert result["chain_completeness_rate"] == 0.0
        assert result["empty_rate"] == 0.0

    def test_returns_all_expected_keys(self):
        result = compute_chain_stats(["some reasoning chain here."])
        assert "chain_completeness_rate" in result
        assert "avg_chain_length_chars" in result
        assert "avg_steps_per_chain" in result
        assert "empty_rate" in result


# ─── MetricsBundle ─────────────────────────────────────────────────────────────

class TestMetricsBundle:
    def test_default_instantiation(self):
        bundle = MetricsBundle()
        assert bundle.accuracy == 0.0
        assert bundle.model_name == ""

    def test_to_dict(self):
        bundle = MetricsBundle(accuracy=0.75, model_name="test")
        d = bundle.to_dict()
        assert d["accuracy"] == 0.75
        assert d["model_name"] == "test"

    def test_str_representation(self):
        bundle = MetricsBundle(
            accuracy=0.8,
            rouge_l_fmeasure=0.65,
            chain_completeness_rate=0.9,
            num_correct=80,
            num_total=100,
            model_name="fine-tuned",
        )
        s = str(bundle)
        assert "0.8000" in s
        assert "0.6500" in s
        assert "fine-tuned" in s