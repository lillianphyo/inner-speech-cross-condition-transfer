"""Classical baselines on synthetic separable data, through the same arms as the networks."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.inner_speech.classical import CLASSICAL, GRID, classical_fit_and_score
from experiments.inner_speech.honest_transfer import run_subject_arms
from experiments.inner_speech.loso import run_loso_arms

N_CH, N_TIMES = 8, 128


def _condition(n_per_class, rng, signal=True, code=(0, 1, 2, 3)):
    X = rng.standard_normal((4 * n_per_class, N_CH, N_TIMES)).astype(np.float32)
    y = np.repeat(np.arange(4), n_per_class)
    if signal:
        for c in range(4):
            X[y == c, code[c]] *= 4.0
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


@pytest.mark.parametrize("name", CLASSICAL)
def test_classical_separates_a_variance_code_and_reports_the_chosen_setting(name):
    rng = np.random.default_rng(0)
    train, val, test = (_condition(20, rng) for _ in range(3))
    scores = classical_fit_and_score(name, train, val, {"test": test}, seed=0)["test"]
    assert scores["accuracy"] > 0.8
    assert set(scores) == {"accuracy", "balanced_accuracy", "f1_macro", "kappa", "setting"}
    assert scores["setting"] in GRID[name]


def test_classical_stays_at_chance_without_signal():
    rng = np.random.default_rng(1)
    train, val, test = (_condition(20, rng, signal=False) for _ in range(3))
    scores = classical_fit_and_score("CSP+LDA", train, val, {"test": test}, seed=0)["test"]
    assert scores["accuracy"] < 0.5


def test_unknown_classical_model_is_refused():
    rng = np.random.default_rng(0)
    train = _condition(5, rng)
    with pytest.raises(ValueError, match="Unknown classical model"):
        classical_fit_and_score("PCA", train, train, {"t": train}, seed=0)


def test_subject_arms_dispatch_to_the_classical_models():
    rng = np.random.default_rng(0)
    X_pron, y_pron = _condition(20, rng)
    X_inner, y_inner = _condition(30, rng)
    rows = {row["arm"]: row for row in run_subject_arms(
        X_pron, y_pron, X_inner, y_inner, model_name="Riemannian", seed=0, config={},
        arms=("target_only", "transfer"))}
    assert rows["target_only"]["accuracy"] > 0.8
    assert rows["transfer"]["accuracy"] > 0.8
    assert {row["setting"] for row in rows.values()} <= set(GRID["Riemannian"])


def test_loso_arms_dispatch_to_the_classical_models():
    rng = np.random.default_rng(0)
    data = {f"sub-0{i}": {"pronounced": _condition(10, rng), "inner_speech": _condition(15, rng)} for i in range(1, 5)}
    rows = {row["arm"]: row for row in run_loso_arms(
        data, "sub-01", model_name="CSP+LDA", seed=0, config={}, arms=("inner_loso", "transfer_loso"))}
    assert rows["inner_loso"]["accuracy"] > 0.8
    assert rows["transfer_loso"]["accuracy"] > 0.8
    assert rows["inner_loso"]["setting"] in GRID["CSP+LDA"]
