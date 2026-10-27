"""Filter spread between neighbouring epochs and class correlation of adjacent trials."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.inner_speech.filter_spread import neighbour_share, same_class_rate

ONSETS = np.array([100, 300, 500, 700])
LO, HI = -20, 80


def _moving_average(width):
    kernel = np.ones(width) / width
    return lambda x: np.stack([np.convolve(row, kernel, mode="same") for row in x])


def test_identity_filter_leaves_no_share_in_the_neighbours():
    x = np.random.default_rng(0).standard_normal((3, 900))
    shares = neighbour_share(x, ONSETS, LO, HI, lambda s: s.copy(), [1, 2])
    assert shares.shape == (4,)
    assert np.allclose(shares, 0.0)


def test_filter_longer_than_the_gap_leaks_into_the_neighbours():
    x = np.random.default_rng(0).standard_normal((3, 900))
    shares = neighbour_share(x, ONSETS, LO, HI, _moving_average(301), [1, 2])
    assert np.all(shares > 0.01) and np.all(shares < 1.0)


def test_filter_shorter_than_the_gap_does_not_leak():
    x = np.random.default_rng(0).standard_normal((3, 900))
    shares = neighbour_share(x, ONSETS, LO, HI, _moving_average(21), [1, 2])
    assert np.allclose(shares, 0.0)


def test_edge_trials_are_refused():
    x = np.zeros((1, 900))
    with pytest.raises(ValueError, match="neighbour"):
        neighbour_share(x, ONSETS, LO, HI, lambda s: s, [0])


def test_same_class_rate_counts_adjacent_pairs_within_a_condition_in_time_order():
    events = np.array([
        [400, 1, 0, 1],
        [100, 0, 0, 1],
        [200, 0, 0, 1],
        [300, 1, 0, 1],
        [500, 1, 1, 1],
        [600, 1, 1, 1],
    ])
    rate, pairs = same_class_rate(events)
    assert pairs == 4
    assert rate == pytest.approx(3 / 4)
