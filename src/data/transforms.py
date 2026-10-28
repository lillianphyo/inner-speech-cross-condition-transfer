"""Train-only normalisation and the shuffled-label control."""
from __future__ import annotations

import numpy as np


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Per-channel z-score using train statistics; apply to train and every other array."""
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0
    return tuple(((a - mean) / std).astype(np.float32) for a in (train, *others))


def shuffle_labels(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute the labels to break the trial-to-class correspondence."""
    y2 = np.asarray(y).copy()
    rng.shuffle(y2)
    return y2
