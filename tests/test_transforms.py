"""Train-only z-score and the shuffled-label control."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.transforms import shuffle_labels, standardize


def test_standardize_uses_train_statistics_for_every_array():
    rng = np.random.default_rng(0)
    train = rng.normal(3.0, 2.0, size=(50, 4, 32))
    test = rng.normal(3.0, 2.0, size=(10, 4, 32))
    train_z, test_z = standardize(train, test)
    assert np.allclose(train_z.mean(axis=(0, 2)), 0.0, atol=1e-5)
    assert np.allclose(train_z.std(axis=(0, 2)), 1.0, atol=1e-4)
    mean, std = train.mean(axis=(0, 2), keepdims=True), train.std(axis=(0, 2), keepdims=True)
    assert np.allclose(test_z, (test - mean) / std, atol=1e-5)
    assert train_z.dtype == test_z.dtype == np.float32


def test_standardize_leaves_a_flat_channel_finite():
    train = np.ones((5, 2, 8))
    (train_z,) = standardize(train)
    assert np.isfinite(train_z).all()


def test_shuffle_labels_permutes_a_copy_reproducibly():
    y = np.repeat(np.arange(4), 25)
    shuffled = shuffle_labels(y, np.random.default_rng(0))
    assert np.array_equal(shuffled, shuffle_labels(y, np.random.default_rng(0)))
    assert sorted(shuffled) == sorted(y)
    assert not np.array_equal(shuffled, y)
    assert np.array_equal(y, np.repeat(np.arange(4), 25))
