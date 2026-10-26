"""Held-out-session mode of the within-subject harness."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.inner_speech.honest_transfer import run_subject_arms, session_split
from tests.test_honest_transfer import CONFIG, _condition

SESSIONS = np.array(["ses-01", "ses-02", "ses-03"])


def _with_sessions(n_per_class, rng, signal):
    X, y = _condition(n_per_class, rng, signal=signal)
    return X, y, SESSIONS[np.arange(len(y)) % 3]


def test_session_split_holds_out_exactly_that_session_and_stratifies_the_rest():
    y = np.repeat(np.arange(4), 30)
    sessions = SESSIONS[np.arange(120) % 3]
    train, val, test = session_split(y, sessions, "ses-02", np.random.default_rng(0))
    assert set(sessions[test]) == {"ses-02"} and len(test) == 40
    assert not (set(train) | set(val)) & set(test) and not set(train) & set(val)
    assert len(train) + len(val) == 80
    assert np.bincount(y[val]).tolist() == [4, 4, 4, 4]


def test_session_split_refuses_an_unknown_or_total_session():
    y = np.repeat(np.arange(4), 3)
    with pytest.raises(ValueError):
        session_split(y, np.array(["ses-01"] * 12), "ses-02", np.random.default_rng(0))
    with pytest.raises(ValueError):
        session_split(y, np.array(["ses-01"] * 12), "ses-01", np.random.default_rng(0))


def test_session_mode_scores_only_the_held_out_session_and_keeps_transfer():
    rng = np.random.default_rng(0)
    X_pron, y_pron, s_pron = _with_sessions(24, rng, signal=True)
    X_inner, y_inner, s_inner = _with_sessions(30, rng, signal=True)
    rows = run_subject_arms(
        X_pron, y_pron, X_inner, y_inner, model_name="ShallowConvNet", seed=0, config=CONFIG,
        arms=("target_only", "transfer", "reciprocal", "source_only", "shuffle_target"),
        held_out_session="ses-03", sessions=(s_pron, s_inner),
    )
    rows = {row["arm"]: row for row in rows}
    assert all(row["session"] == "ses-03" for row in rows.values())
    assert rows["target_only"]["n_test"] == rows["transfer"]["n_test"] == int((s_inner == "ses-03").sum())
    assert rows["reciprocal"]["n_test"] == rows["source_only"]["n_test"] == int((s_pron == "ses-03").sum())
    assert rows["transfer"]["accuracy"] > 0.6 and rows["source_only"]["accuracy"] > 0.6
    assert rows["shuffle_target"]["accuracy"] < 0.5


def test_default_mode_is_unchanged_without_a_session():
    rng = np.random.default_rng(1)
    X_pron, y_pron = _condition(20, rng, signal=True)
    X_inner, y_inner = _condition(30, rng, signal=True)
    rows = {row["arm"]: row for row in run_subject_arms(
        X_pron, y_pron, X_inner, y_inner, model_name="ShallowConvNet", seed=0, config=CONFIG, arms=("reciprocal",),
    )}
    assert "session" not in rows["reciprocal"] and rows["reciprocal"]["n_test"] == 80
