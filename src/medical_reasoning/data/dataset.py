"""
dataset.py
──────────
Loads and caches the FreedomIntelligence/medical-o1-reasoning-SFT dataset,
draws a reproducible stratified subset, and returns train/val/test splits
as HuggingFace Dataset objects ready for the preprocessor.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# ─── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Mirrors the `data:` block in training_config.yaml."""

    hf_dataset_name: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
    hf_dataset_split: str = "train"
    num_samples: int = 5_000
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    max_seq_length: int = 2048
    question_column: str = "Question"
    cot_column: str = "Complex_CoT"
    answer_column: str = "Response"
    seed: int = 42

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> "DataConfig":
        return cls(
            hf_dataset_name=cfg.data.hf_dataset_name,
            hf_dataset_split=cfg.data.hf_dataset_split,
            num_samples=cfg.data.num_samples,
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
            test_ratio=cfg.data.test_ratio,
            max_seq_length=cfg.data.max_seq_length,
            question_column=cfg.data.question_column,
            cot_column=cfg.data.cot_column,
            answer_column=cfg.data.answer_column,
            seed=cfg.experiment.seed,
        )

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {total:.4f}"
            )


# ─── Main class ────────────────────────────────────────────────────────────────

class MedicalReasoningDataset:
    """
    Loads and manages the medical-o1-reasoning-SFT dataset.

    Usage
    -----
    >>> ds = MedicalReasoningDataset(config)
    >>> splits = ds.get_splits()
    >>> print(splits["train"][0])
    """

    REQUIRED_COLUMNS = {"Question", "Complex_CoT", "Response"}

    def __init__(self, config: DataConfig, cache_dir: Optional[str] = None) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self._raw: Optional[Dataset] = None
        self._splits: Optional[DatasetDict] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self) -> "MedicalReasoningDataset":
        """Download (or load from cache) and subset the raw dataset."""
        logger.info("Loading dataset: %s", self.config.hf_dataset_name)

        raw_full = load_dataset(
            self.config.hf_dataset_name,
            split=self.config.hf_dataset_split,
            cache_dir=self.cache_dir,
        )
        logger.info("Full dataset size: %d samples", len(raw_full))

        self._validate_columns(raw_full)
        self._raw = self._subset(raw_full)
        logger.info(
            "Using %d / %d samples (seed=%d)",
            len(self._raw),
            len(raw_full),
            self.config.seed,
        )
        return self

    def get_splits(self) -> DatasetDict:
        """Return the train / val / test DatasetDict. Loads if not already loaded."""
        if self._raw is None:
            self.load()

        if self._splits is None:
            self._splits = self._create_splits(self._raw)

        return self._splits

    def get_raw(self) -> Dataset:
        """Return the raw (un-split) subset."""
        if self._raw is None:
            self.load()
        return self._raw  # type: ignore[return-value]

    def summary(self) -> dict:
        """Return a dict of dataset statistics for logging."""
        splits = self.get_splits()
        return {
            "train_size": len(splits["train"]),
            "val_size": len(splits["validation"]),
            "test_size": len(splits["test"]),
            "total": len(splits["train"]) + len(splits["validation"]) + len(splits["test"]),
            "question_col": self.config.question_column,
            "cot_col": self.config.cot_column,
            "answer_col": self.config.answer_column,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _validate_columns(self, ds: Dataset) -> None:
        """Raise informative error if expected columns are missing."""
        actual = set(ds.column_names)
        missing = self.REQUIRED_COLUMNS - actual
        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {missing}.\n"
                f"Available columns: {actual}\n"
                f"Check the question_column / cot_column / answer_column "
                f"settings in your config."
            )

    def _subset(self, ds: Dataset) -> Dataset:
        """Draw a reproducible random subset of `num_samples`."""
        n = min(self.config.num_samples, len(ds))
        if n == len(ds):
            logger.warning(
                "num_samples (%d) >= dataset size (%d); using all samples.",
                self.config.num_samples,
                len(ds),
            )
            return ds

        random.seed(self.config.seed)
        indices = random.sample(range(len(ds)), n)
        return ds.select(sorted(indices))

    def _create_splits(self, ds: Dataset) -> DatasetDict:
        """Split the dataset into train / val / test deterministically."""
        n = len(ds)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)
        # Test gets the remainder to avoid rounding loss
        n_test = n - n_train - n_val

        # Shuffle once with fixed seed, then slice
        shuffled = ds.shuffle(seed=self.config.seed)

        train_ds = shuffled.select(range(0, n_train))
        val_ds = shuffled.select(range(n_train, n_train + n_val))
        test_ds = shuffled.select(range(n_train + n_val, n_train + n_val + n_test))

        logger.info(
            "Splits → train: %d | val: %d | test: %d",
            len(train_ds), len(val_ds), len(test_ds),
        )

        return DatasetDict(
            {
                "train": train_ds,
                "validation": val_ds,
                "test": test_ds,
            }
        )

    def __repr__(self) -> str:
        status = "loaded" if self._splits else "not loaded"
        return (
            f"MedicalReasoningDataset("
            f"source={self.config.hf_dataset_name!r}, "
            f"num_samples={self.config.num_samples}, "
            f"status={status})"
        )