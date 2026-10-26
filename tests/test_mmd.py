"""MMD domain-adaptation arm of the honest transfer harness."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.inner_speech.exp02_cross_dataset import get_model
from experiments.inner_speech.honest_transfer import MMD_WEIGHT, run_subject_arms
from experiments.inner_speech.mmd import mmd_rbf, mmd_regularizer
from tests.test_honest_transfer import CONFIG, N_CH, N_TIMES, _condition


def test_mmd_is_near_zero_for_one_distribution_and_positive_for_shifted_ones():
    rng = torch.Generator().manual_seed(0)
    a, b = torch.randn(64, 10, generator=rng), torch.randn(64, 10, generator=rng)
    same = mmd_rbf(a, b).item()
    shifted = mmd_rbf(a, b + 3.0).item()
    assert abs(same) < 0.05
    assert shifted > 10 * max(same, 1e-3)


@pytest.mark.parametrize("name", ["EEGNet", "ShallowConvNet"])
def test_network_features_feed_the_classifier(name):
    model = get_model(name, N_CH, N_TIMES, 4).eval()
    x = torch.randn(5, N_CH, N_TIMES)
    assert torch.allclose(model(x), model.fc(model.features(x)))


def test_regularizer_returns_a_differentiable_scalar():
    rng = np.random.default_rng(0)
    source = rng.standard_normal((40, N_CH, N_TIMES)).astype(np.float32)
    target = rng.standard_normal((30, N_CH, N_TIMES)).astype(np.float32) + 2.0
    model = get_model("ShallowConvNet", N_CH, N_TIMES, 4)
    penalty = mmd_regularizer(source, target, weight=0.5, batch_size=16, device="cpu", rng=rng)(model)
    assert penalty.shape == () and penalty.requires_grad and torch.isfinite(penalty)


def _rows(shared_signal: bool, seed=0):
    rng = np.random.default_rng(seed)
    X_pron, y_pron = _condition(20, rng, signal=True)
    X_inner, y_inner = _condition(30, rng, signal=shared_signal)
    rows = run_subject_arms(
        X_pron, y_pron, X_inner, y_inner,
        model_name="ShallowConvNet", seed=seed, config=CONFIG, device="cpu", arms=("mmd", "transfer"),
    )
    return {row["arm"]: row for row in rows}


def test_mmd_arm_transfers_a_shared_code_and_records_its_weight():
    rows = _rows(shared_signal=True)
    assert rows["mmd"]["accuracy"] > 0.6
    assert rows["mmd"]["n_test"] == rows["transfer"]["n_test"]
    assert rows["mmd"]["mmd_weight"] == MMD_WEIGHT
    assert "mmd_weight" not in rows["transfer"]


def test_mmd_arm_cannot_invent_an_inner_speech_code():
    rows = _rows(shared_signal=False)
    assert rows["mmd"]["accuracy"] < 0.5


def test_mmd_arm_is_refused_for_classical_models():
    rng = np.random.default_rng(0)
    X_pron, y_pron = _condition(20, rng, signal=True)
    X_inner, y_inner = _condition(20, rng, signal=True)
    with pytest.raises(ValueError, match="mmd"):
        run_subject_arms(X_pron, y_pron, X_inner, y_inner, model_name="CSP+LDA", seed=0, arms=("mmd",))
