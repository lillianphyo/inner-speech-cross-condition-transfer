"""Train/test overlap guard: the check that would have caught the ds003626 condition leak."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.leakage import assert_disjoint


def _trials(n, seed):
    return np.random.default_rng(seed).standard_normal((n, 4, 32)).astype(np.float32)


def test_assert_disjoint_accepts_separate_trial_sets():
    assert_disjoint(_trials(20, seed=0), _trials(20, seed=1))


def test_assert_disjoint_rejects_shared_trials():
    train, test = _trials(20, seed=0), _trials(20, seed=1)
    leaked = np.concatenate([test, train[:3]])
    with pytest.raises(ValueError, match="3 trials"):
        assert_disjoint(train, leaked)
