"""Leave-one-subject-out arms for ds003626: nothing from the held-out subject touches training
or model selection. The validation subject is the next subject in the list, so the eight remaining
subjects train the model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.inner_speech.honest_transfer import CHANCE, fit_and_score
from src.data.leakage import assert_disjoint
from src.data.transforms import shuffle_labels

ARMS = ("inner_loso", "overt_loso", "transfer_loso", "shuffle_inner_loso")
CONDITION = {"inner_loso": "inner_speech", "overt_loso": "pronounced", "transfer_loso": "pronounced",
             "shuffle_inner_loso": "inner_speech"}
TEST_CONDITION = {"inner_loso": "inner_speech", "overt_loso": "pronounced", "transfer_loso": "inner_speech",
                  "shuffle_inner_loso": "inner_speech"}


def loso_folds(subjects: list[str]) -> list[tuple[str, str, list[str]]]:
    """(held-out, validation, training subjects) for every subject; validation = the next subject."""
    folds = []
    for i, held_out in enumerate(subjects):
        validation = subjects[(i + 1) % len(subjects)]
        folds.append((held_out, validation, [s for s in subjects if s not in (held_out, validation)]))
    return folds


def _pool(data: dict, subjects: list[str], condition: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([data[s][condition][0] for s in subjects])
    y = np.concatenate([data[s][condition][1] for s in subjects])
    return X, y


def run_loso_arms(data: dict, held_out: str, *, model_name: str, seed: int, config: dict, device: str = "cpu",
                  arms: tuple[str, ...] = ARMS) -> list[dict]:
    """`data[subject][condition] = (X, y)`; returns one row per arm scored on all trials of `held_out`."""
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arm(s): {sorted(unknown)}")
    subjects = list(data)
    _, validation, training = next(fold for fold in loso_folds(subjects) if fold[0] == held_out)
    rng = np.random.default_rng(seed)
    rows = []
    for arm in arms:
        train = _pool(data, training, CONDITION[arm])
        val = data[validation][CONDITION[arm]]
        test = data[held_out][TEST_CONDITION[arm]]
        assert_disjoint(train[0], test[0])
        assert_disjoint(val[0], test[0])
        if arm == "shuffle_inner_loso":
            train = (train[0], shuffle_labels(train[1], rng))
            val = (val[0], shuffle_labels(val[1], rng))
        scores = fit_and_score(model_name, train, val, {"test": test}, config=config, device=device, seed=seed)
        rows.append(dict(arm=arm, validation_subject=validation, n_train=len(train[1]), n_test=len(test[1]),
                         chance=CHANCE, **scores["test"]))
    return rows
