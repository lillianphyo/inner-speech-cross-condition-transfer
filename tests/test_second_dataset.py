"""Overt-only and LOSO arms of the second-dataset experiment, on synthetic binary subjects."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.inner_speech.exp09_second_dataset import LOSO_ARMS, WITHIN_ARMS, loso_rows, within_rows

CONFIG = dict(batch_size=16, num_epochs=30, learning_rate=1e-2, weight_decay=0.0, patience=30)
N_CH, N_TIMES = 8, 128


def _subject(n_per_class, rng, code):
    X = rng.standard_normal((2 * n_per_class, N_CH, N_TIMES)).astype(np.float32)
    y = np.repeat(np.arange(2), n_per_class)
    for c in range(2):
        X[y == c, code[c]] *= 4.0
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def test_within_subject_overt_is_decodable_and_its_shuffle_is_not():
    rng = np.random.default_rng(0)
    X, y = _subject(40, rng, code=[0, 1])
    rows = {row["arm"]: row for row in within_rows(X, y, model_name="ShallowConvNet", seed=0, config=CONFIG)}
    assert set(rows) == set(WITHIN_ARMS)
    assert rows["overt_only"]["n_test"] == 16 and rows["overt_only"]["chance"] == 0.5
    assert rows["overt_only"]["accuracy"] > 0.75
    assert rows["shuffle_overt"]["accuracy"] < 0.7
    assert 0.5 <= rows["overt_only"]["majority_rate"] <= 1.0


def test_loso_transfers_a_shared_code_and_never_trains_on_the_held_out_subject():
    rng = np.random.default_rng(0)
    data = {f"sub-{i:02d}": _subject(20, rng, code=[0, 1]) for i in range(1, 5)}
    rows = {row["arm"]: row for row in loso_rows(data, "sub-02", model_name="ShallowConvNet", seed=0, config=CONFIG)}
    assert set(rows) == set(LOSO_ARMS)
    assert rows["overt_loso"]["validation_subject"] == "sub-03"
    assert rows["overt_loso"]["n_train"] == 80 and rows["overt_loso"]["n_test"] == 40
    assert rows["overt_loso"]["accuracy"] > 0.75
    assert rows["shuffle_overt_loso"]["accuracy"] < 0.7


def test_loso_fails_on_a_subject_specific_code():
    rng = np.random.default_rng(0)
    data = {f"sub-{i:02d}": _subject(20, rng, code=[2 * i - 2, 2 * i - 1]) for i in range(1, 5)}
    rows = {row["arm"]: row for row in loso_rows(data, "sub-01", model_name="ShallowConvNet", seed=0, config=CONFIG, arms=("overt_loso",))}
    assert rows["overt_loso"]["accuracy"] < 0.7
