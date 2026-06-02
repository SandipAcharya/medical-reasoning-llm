"""
utils.py
────────
Shared utilities: config loading, dataset statistics, and helpers
used across the data pipeline.
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ─── Config Loading ────────────────────────────────────────────────────────────

def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load a YAML config file into an OmegaConf DictConfig.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    DictConfig
        Parsed configuration with dot-access and merge support.

    Example
    -------
    >>> cfg = load_config("config/training_config.yaml")
    >>> print(cfg.model.name)
    Qwen/Qwen2.5-3B-Instruct
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path.resolve()}\n"
            f"Make sure you're running from the project root directory."
        )

    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = OmegaConf.create(raw)
    logger.debug("Loaded config from: %s", path)
    return cfg


def merge_configs(*config_paths: Union[str, Path]) -> DictConfig:
    """
    Merge multiple YAML configs. Later configs override earlier ones.

    Example
    -------
    >>> cfg = merge_configs("config/model_config.yaml", "config/training_config.yaml")
    """
    merged = OmegaConf.create({})
    for path in config_paths:
        cfg = load_config(path)
        merged = OmegaConf.merge(merged, cfg)
    return merged


# ─── Dataset Utilities ─────────────────────────────────────────────────────────

def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> DatasetDict:
    """
    Split a Dataset into train / validation / test splits.

    Parameters
    ----------
    dataset : Dataset
        The full dataset to split.
    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.0.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    DatasetDict with keys "train", "validation", "test".
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.4f}")

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    shuffled = dataset.shuffle(seed=seed)

    return DatasetDict(
        {
            "train": shuffled.select(range(0, n_train)),
            "validation": shuffled.select(range(n_train, n_train + n_val)),
            "test": shuffled.select(range(n_train + n_val, n_train + n_val + n_test)),
        }
    )


def dataset_statistics(
    dataset: Union[Dataset, DatasetDict],
    text_column: str = "text",
    tokenizer=None,
) -> Dict[str, Any]:
    """
    Compute token-length statistics for a dataset or DatasetDict.

    Parameters
    ----------
    dataset : Dataset | DatasetDict
        The dataset to analyze.
    text_column : str
        Column containing the formatted text.
    tokenizer : optional
        If provided, compute token lengths. Otherwise use character lengths.

    Returns
    -------
    dict with keys: mean, median, min, max, p95, p99 (and split breakdown
    if DatasetDict is passed).
    """
    def _lengths(ds: Dataset) -> list[int]:
        texts = ds[text_column] if text_column in ds.column_names else []
        if tokenizer is not None:
            return [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
        return [len(t) for t in texts]

    def _stats(lengths: list[int]) -> dict:
        if not lengths:
            return {}
        sorted_l = sorted(lengths)
        n = len(sorted_l)
        return {
            "count": n,
            "mean": round(statistics.mean(lengths), 1),
            "median": statistics.median(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "p95": sorted_l[int(0.95 * n)],
            "p99": sorted_l[int(0.99 * n)],
        }

    unit = "tokens" if tokenizer else "chars"

    if isinstance(dataset, DatasetDict):
        result: Dict[str, Any] = {"unit": unit}
        all_lengths: list[int] = []
        for split_name, split_ds in dataset.items():
            lengths = _lengths(split_ds)
            result[split_name] = _stats(lengths)
            all_lengths.extend(lengths)
        result["overall"] = _stats(all_lengths)
        return result
    else:
        lengths = _lengths(dataset)
        return {"unit": unit, "overall": _stats(lengths)}


def print_sample(
    dataset: Union[Dataset, DatasetDict],
    idx: int = 0,
    split: str = "train",
    text_column: str = "text",
    max_chars: int = 1500,
) -> None:
    """
    Pretty-print a single example from the dataset.

    Parameters
    ----------
    dataset : Dataset | DatasetDict
    idx : int
        Index of the example.
    split : str
        Which split to use if DatasetDict.
    text_column : str
        Column to display.
    max_chars : int
        Truncate output after this many characters.
    """
    if isinstance(dataset, DatasetDict):
        ds = dataset[split]
    else:
        ds = dataset

    row = ds[idx]
    text = row.get(text_column, str(row))

    separator = "─" * 80
    print(f"\n{separator}")
    print(f"  Sample #{idx} | Split: {split} | Column: {text_column}")
    print(separator)
    print(text[:max_chars])
    if len(text) > max_chars:
        print(f"\n... [{len(text) - max_chars} more characters]")
    print(f"{separator}\n")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_parameters(model) -> Dict[str, int]:
    """
    Count trainable vs total parameters in a PyTorch model.

    Returns
    -------
    dict with keys: trainable, total, trainable_pct
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": round(100.0 * trainable / total, 4) if total else 0.0,
    }