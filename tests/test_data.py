"""
tests/test_data.py
──────────────────
Unit tests for the data pipeline (no model required).
Run with: pytest tests/test_data.py -v
"""

import sys
from pathlib import Path

import pytest
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.medical_reasoning.data.dataset import DataConfig, MedicalReasoningDataset
from src.medical_reasoning.data.utils import (
    dataset_statistics,
    ensure_dir,
    load_config,
    split_dataset,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

def _make_dummy_dataset(n: int = 100) -> Dataset:
    """Create a minimal dataset that matches the expected schema."""
    return Dataset.from_dict(
        {
            "Question": [f"What is diagnosis {i}?" for i in range(n)],
            "Complex_CoT": [f"Reasoning for case {i}..." for i in range(n)],
            "Response": [f"Diagnosis {i}" for i in range(n)],
        }
    )


# ─── DataConfig tests ──────────────────────────────────────────────────────────

class TestDataConfig:
    def test_default_instantiation(self):
        cfg = DataConfig()
        assert cfg.num_samples == 5_000
        assert cfg.train_ratio + cfg.val_ratio + cfg.test_ratio == pytest.approx(1.0)

    def test_invalid_ratios_raise(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            DataConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_valid_custom_ratios(self):
        cfg = DataConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        assert cfg.train_ratio + cfg.val_ratio + cfg.test_ratio == pytest.approx(1.0)


# ─── Split tests ───────────────────────────────────────────────────────────────

class TestSplitDataset:
    def test_split_sizes(self):
        ds = _make_dummy_dataset(100)
        splits = split_dataset(ds, 0.8, 0.1, 0.1, seed=42)
        assert len(splits["train"]) == 80
        assert len(splits["validation"]) == 10
        assert len(splits["test"]) == 10

    def test_split_is_reproducible(self):
        ds = _make_dummy_dataset(100)
        s1 = split_dataset(ds, seed=42)
        s2 = split_dataset(ds, seed=42)
        assert s1["train"][0] == s2["train"][0]

    def test_split_differs_with_different_seed(self):
        ds = _make_dummy_dataset(100)
        s1 = split_dataset(ds, seed=42)
        s2 = split_dataset(ds, seed=99)
        # At least some examples should differ
        assert s1["train"][0] != s2["train"][0] or s1["train"][1] != s2["train"][1]

    def test_invalid_ratios_raise(self):
        ds = _make_dummy_dataset(100)
        with pytest.raises(ValueError):
            split_dataset(ds, 0.5, 0.5, 0.5)

    def test_correct_keys(self):
        ds = _make_dummy_dataset(50)
        splits = split_dataset(ds)
        assert set(splits.keys()) == {"train", "validation", "test"}


# ─── Dataset statistics tests ──────────────────────────────────────────────────

class TestDatasetStatistics:
    def test_basic_statistics(self):
        ds = _make_dummy_dataset(50)
        # Add a text column
        ds = ds.map(lambda x: {"text": x["Question"] + " " + x["Response"]})
        stats = dataset_statistics(ds, text_column="text")
        assert "overall" in stats
        assert stats["overall"]["count"] == 50
        assert stats["overall"]["min"] <= stats["overall"]["mean"]
        assert stats["overall"]["mean"] <= stats["overall"]["max"]

    def test_dataset_dict_statistics(self):
        from datasets import DatasetDict
        ds = _make_dummy_dataset(100)
        ds = ds.map(lambda x: {"text": x["Question"]})
        splits = split_dataset(ds)
        stats = dataset_statistics(splits, text_column="text")
        assert "train" in stats
        assert "validation" in stats
        assert "test" in stats
        assert "overall" in stats


# ─── MedicalReasoningDataset validation tests ──────────────────────────────────

class TestMedicalReasoningDatasetValidation:
    def test_missing_column_raises(self):
        """Simulates a dataset that is missing the CoT column."""
        # We can't easily mock the HF download in unit tests,
        # but we can test the validation method directly.
        bad_ds = Dataset.from_dict(
            {"Question": ["q1"], "Response": ["a1"]}  # Missing Complex_CoT
        )
        cfg = DataConfig(num_samples=1)
        loader = MedicalReasoningDataset(cfg)
        with pytest.raises(ValueError, match="missing required columns"):
            loader._validate_columns(bad_ds)

    def test_subset_does_not_exceed_dataset_size(self):
        ds = _make_dummy_dataset(10)
        cfg = DataConfig(num_samples=999)
        loader = MedicalReasoningDataset(cfg)
        subset = loader._subset(ds)
        assert len(subset) == 10  # Capped at dataset size

    def test_repr(self):
        cfg = DataConfig()
        loader = MedicalReasoningDataset(cfg)
        r = repr(loader)
        assert "MedicalReasoningDataset" in r
        assert "not loaded" in r


# ─── Config loading tests ─────────────────────────────────────────────────────

class TestLoadConfig:
    def test_loads_training_config(self):
        cfg = load_config("config/training_config.yaml")
        assert hasattr(cfg, "model")
        assert hasattr(cfg, "training")
        assert hasattr(cfg, "lora")
        assert cfg.model.name == "Qwen/Qwen2.5-3B-Instruct"

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("config/does_not_exist.yaml")


# ─── Utilities ─────────────────────────────────────────────────────────────────

class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "nested" / "new_dir"
        assert not new_dir.exists()
        result = ensure_dir(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_idempotent(self, tmp_path):
        d = tmp_path / "existing"
        ensure_dir(d)
        ensure_dir(d)  # Should not raise
        assert d.exists()