"""LOSO harness on synthetic subjects: a class code shared across subjects must transfer to the
held-out subject, a subject-specific code must not, and the held-out subject never trains."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.inner_speech.loso import ARMS, loso_folds, run_loso_arms

CONFIG = dict(batch_size=16, num_epochs=30, learning_rate=1e-2, weight_decay=0.0, patience=30)
N_CH, N_TIMES = 8, 128
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04"]


def _condition(n_per_class, rng, code):
    X = rng.standard_normal((4 * n_per_class, N_CH, N_TIMES)).astype(np.float32)
    y = np.repeat(np.arange(4), n_per_class)
    for c in range(4):
        X[y == c, code[c]] *= 4.0
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def _data(shared: bool, seed=0):
    rng = np.random.default_rng(seed)
    data = {}
    for i, subject in enumerate(SUBJECTS):
        code = [0, 1, 2, 3] if shared else [(c + i) % 4 for c in range(4)]
        data[subject] = {"pronounced": _condition(10, rng, code), "inner_speech": _condition(15, rng, code)}
    return data


def _rows(shared: bool, arms=ARMS):
    rows = run_loso_arms(_data(shared), "sub-01", model_name="ShallowConvNet", seed=0, config=CONFIG, arms=arms)
    return {row["arm"]: row for row in rows}


def test_folds_hold_out_one_subject_and_validate_on_the_next():
    folds = loso_folds(SUBJECTS)
    assert folds[0] == ("sub-01", "sub-02", ["sub-03", "sub-04"])
    assert folds[-1] == ("sub-04", "sub-01", ["sub-02", "sub-03"])
    assert all(held_out not in train and validation not in train for held_out, validation, train in folds)


def test_shared_code_transfers_across_subjects_and_shuffle_collapses():
    rows = _rows(shared=True)
    assert rows["inner_loso"]["accuracy"] > 0.6
    assert rows["overt_loso"]["accuracy"] > 0.6
    assert rows["transfer_loso"]["accuracy"] > 0.6
    assert rows["shuffle_inner_loso"]["accuracy"] < 0.55


def test_subject_specific_code_stays_at_chance_on_the_held_out_subject():
    rows = _rows(shared=False, arms=("inner_loso",))
    assert rows["inner_loso"]["accuracy"] < 0.5


def test_rows_score_every_held_out_trial_and_train_on_two_subjects():
    rows = _rows(shared=True)
    assert rows["inner_loso"]["n_test"] == 60 and rows["inner_loso"]["n_train"] == 120
    assert rows["overt_loso"]["n_test"] == 40 and rows["overt_loso"]["n_train"] == 80
    assert rows["transfer_loso"]["n_test"] == 60 and rows["transfer_loso"]["n_train"] == 80
    assert all(row["validation_subject"] == "sub-02" for row in rows.values())


def test_held_out_trials_in_training_are_refused():
    data = _data(shared=True)
    data["sub-03"] = data["sub-01"]
    with pytest.raises(ValueError, match="both the train and the test set"):
        run_loso_arms(data, "sub-01", model_name="ShallowConvNet", seed=0, config=CONFIG, arms=("inner_loso",))


def test_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="unknown arm"):
        run_loso_arms(_data(True), "sub-01", model_name="ShallowConvNet", seed=0, config=CONFIG, arms=("loso",))
