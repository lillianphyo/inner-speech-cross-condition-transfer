"""Honest cross-condition transfer harness, on synthetic separable data.

Two synthetic regimes: a class code shared by both conditions (transfer must work) and a
class code present only in the pronounced condition (inner speech must stay at chance).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.inner_speech.exp02_cross_dataset import get_model
from experiments.inner_speech.exp05_honest_transfer import MODELS, WINDOWS, append_rows
from experiments.inner_speech.honest_transfer import ARMS, run_subject_arms, split_indices, summarize

CONFIG = dict(batch_size=16, num_epochs=30, learning_rate=1e-2, weight_decay=0.0, patience=30)
N_CH, N_TIMES = 8, 128


def _condition(n_per_class, rng, signal: bool):
    X = rng.standard_normal((4 * n_per_class, N_CH, N_TIMES)).astype(np.float32)
    y = np.repeat(np.arange(4), n_per_class)
    if signal:
        for c in range(4):
            X[y == c, c] *= 4.0
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def _rows(shared_signal: bool, seed=0, arms=ARMS):
    rng = np.random.default_rng(seed)
    X_pron, y_pron = _condition(20, rng, signal=True)
    X_inner, y_inner = _condition(30, rng, signal=shared_signal)
    rows = run_subject_arms(
        X_pron, y_pron, X_inner, y_inner,
        model_name="ShallowConvNet", seed=seed, config=CONFIG, device="cpu", arms=arms,
    )
    return {row["arm"]: row for row in rows}


def test_split_indices_is_stratified_and_disjoint():
    y = np.repeat(np.arange(4), 25)
    train, val, test = split_indices(y, (0.6, 0.2, 0.2), np.random.default_rng(0))
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
    assert sorted(np.concatenate([train, val, test])) == list(range(100))
    assert np.bincount(y[test]).tolist() == [5, 5, 5, 5]


def test_shared_signal_transfers_and_shuffle_collapses():
    rows = _rows(shared_signal=True)
    assert rows["transfer"]["accuracy"] > 0.6
    assert rows["target_only"]["accuracy"] > 0.6
    assert rows["reciprocal"]["accuracy"] > 0.6
    assert rows["shuffle_transfer"]["accuracy"] < 0.5
    assert rows["shuffle_target"]["accuracy"] < 0.5


def test_pronounced_only_signal_does_not_reach_inner_speech():
    rows = _rows(shared_signal=False)
    assert rows["transfer"]["accuracy"] < 0.5
    assert rows["target_only"]["accuracy"] < 0.5


def test_every_inner_arm_scores_the_held_out_fifth_and_reciprocal_scores_all_pronounced_trials():
    rows = _rows(shared_signal=True)
    n_test = {rows[arm]["n_test"] for arm in ("target_only", "transfer", "combined", "shuffle_target", "shuffle_transfer")}
    assert n_test == {24}
    assert rows["reciprocal"]["n_test"] == 80


def test_source_only_scores_a_held_out_fifth_of_pronounced_trials_and_its_shuffle_collapses():
    rows = _rows(shared_signal=False, arms=("source_only", "shuffle_source"))
    assert set(rows) == {"source_only", "shuffle_source"}
    assert rows["source_only"]["n_test"] == rows["shuffle_source"]["n_test"] == 16
    assert rows["source_only"]["n_train"] == 48
    assert rows["source_only"]["accuracy"] > 0.6
    assert rows["shuffle_source"]["accuracy"] < 0.5


def test_arms_filter_runs_only_the_requested_arms():
    rows = _rows(shared_signal=True, arms=("transfer", "shuffle_transfer"))
    assert set(rows) == {"transfer", "shuffle_transfer"}


def test_unknown_arm_is_refused():
    rng = np.random.default_rng(0)
    X_pron, y_pron = _condition(20, rng, signal=True)
    X_inner, y_inner = _condition(30, rng, signal=True)
    with pytest.raises(ValueError, match="unknown arm"):
        run_subject_arms(X_pron, y_pron, X_inner, y_inner, model_name="ShallowConvNet", seed=0,
                         config=CONFIG, device="cpu", arms=("overt",))


def test_overlapping_conditions_are_refused():
    rng = np.random.default_rng(0)
    X, y = _condition(20, rng, signal=True)
    with pytest.raises(ValueError, match="both the train and the test set"):
        run_subject_arms(X, y, X, y, model_name="ShallowConvNet", seed=0, config=CONFIG, device="cpu")


def test_append_rows_widens_the_header_for_a_new_metric(tmp_path):
    out = tmp_path / "sub-01.csv"
    append_rows(out, pd.DataFrame([dict(subject="sub-01", arm="transfer", accuracy=0.3)]))
    append_rows(out, pd.DataFrame([dict(subject="sub-01", arm="source_only", accuracy=0.4, kappa=0.1)]))
    frame = pd.read_csv(out)
    assert list(frame.columns) == ["subject", "arm", "accuracy", "kappa"]
    assert frame["kappa"].isna().tolist() == [True, False]
    assert frame["accuracy"].tolist() == [0.3, 0.4]


def test_summarize_reports_ci_and_test_against_chance():
    rows = [
        dict(subject=f"sub-{s:02d}", window="action", model="EEGNet", seed=seed, arm="transfer",
             train_fraction=1.0, n_test=100, accuracy=0.25 + 0.01 * s, chance=0.25)
        for s in range(1, 9) for seed in (0, 1)
    ]
    summary = summarize(pd.DataFrame(rows)).iloc[0]
    assert summary["n_subjects"] == 8
    assert summary["min_seeds"] == 2
    assert summary["p_holm"] == summary["p_above_chance"]
    assert summary["mean_accuracy"] == pytest.approx(0.295)
    assert summary["ci95_low"] < summary["mean_accuracy"] < summary["ci95_high"]
    assert summary["p_above_chance"] < 0.05


def test_summarize_counts_subjects_above_their_binomial_threshold():
    rows = [
        dict(subject=f"sub-{s:02d}", window="action", model="EEGNet", seed=0, arm="transfer",
             train_fraction=1.0, n_test=200, accuracy=accuracy, chance=0.25)
        for s, accuracy in enumerate((0.26, 0.29, 0.30, 0.31, 0.35), start=1)
    ]
    summary = summarize(pd.DataFrame(rows)).iloc[0]
    assert summary["binomial_threshold"] == pytest.approx(stats.binom.ppf(0.95, 200, 0.25) / 200)
    assert summary["subjects_above_binomial"] == 2


def test_holm_correction_scales_the_smallest_p_by_the_number_of_cells():
    rows = [
        dict(subject=f"sub-{s:02d}", window=window, model="EEGNet", seed=0, arm="transfer",
             train_fraction=1.0, n_test=100, accuracy=0.25 + effect * s, chance=0.25)
        for window, effect in (("action", 0.01), ("cue", 0.0005), ("full", -0.01)) for s in range(1, 9)
    ]
    summary = summarize(pd.DataFrame(rows)).set_index("window")
    assert summary.loc["action", "p_holm"] == pytest.approx(3 * summary.loc["action", "p_above_chance"])
    assert (summary["p_holm"] >= summary["p_above_chance"]).all()
    assert summary["p_holm"].max() <= 1.0


def _fraction_rows(shared_signal: bool, arms, fractions, model_name="ShallowConvNet", seed=0):
    rng = np.random.default_rng(seed)
    X_pron, y_pron = _condition(20, rng, signal=True)
    X_inner, y_inner = _condition(30, rng, signal=shared_signal)
    rows = run_subject_arms(
        X_pron, y_pron, X_inner, y_inner, model_name=model_name, seed=seed, config=CONFIG, device="cpu",
        arms=arms, train_fractions=fractions,
    )
    return {(row["arm"], row["train_fraction"]): row for row in rows}


def test_finetune_arms_use_the_target_only_subsets_and_score_the_held_out_fifth():
    rows = _fraction_rows(True, ("target_only", "finetune", "shuffle_finetune"), (0.5, 1.0))
    assert set(rows) == {(arm, f) for arm in ("target_only", "finetune", "shuffle_finetune") for f in (0.5, 1.0)}
    for fraction in (0.5, 1.0):
        assert rows[("finetune", fraction)]["n_train"] == rows[("target_only", fraction)]["n_train"]
        assert rows[("shuffle_finetune", fraction)]["n_train"] == rows[("target_only", fraction)]["n_train"]
        assert rows[("finetune", fraction)]["n_test"] == 24
    assert rows[("target_only", 0.5)]["n_train"] == 36
    assert rows[("finetune", 1.0)]["accuracy"] > 0.6


def test_finetune_finds_nothing_when_inner_speech_carries_no_signal():
    rows = _fraction_rows(False, ("finetune",), (1.0,))
    assert rows[("finetune", 1.0)]["accuracy"] < 0.5


def test_finetune_arms_are_refused_for_classical_models():
    with pytest.raises(ValueError, match="network"):
        _fraction_rows(True, ("finetune",), (1.0,), model_name="CSP+LDA")


def test_concentration_window_ends_at_cue_onset_and_fits_every_network():
    tmin, tmax = WINDOWS["concentration"]
    assert tmax <= 0.0 and tmin == WINDOWS["cue"][0] and tmax <= WINDOWS["action"][0]
    n_times = int((tmax - tmin) * 256) + 1
    for name in MODELS:
        model = get_model(name, 128, n_times, 4).eval()
        with torch.no_grad():
            assert model(torch.zeros(3, 128, n_times)).shape == (3, 4)
