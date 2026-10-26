"""EXG artefact arms: trial-wise regression, signal selection, aligned EXG loading and the exp10 runner."""
import sys
from argparse import Namespace
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.inner_speech.exg import LIP_CHANNELS, SIGNALS, make_signal, regress_out
from experiments.inner_speech.exp10_exg_arms import load_signals, run_subject
from src.data.inner_speech_loader import InnerSpeechDataset
from tests.test_inner_speech_loader import N_CH, SFREQ, _write_session

N_EXG = 8
EXG_NAMES = [f"EXG{i + 1}" for i in range(N_EXG)]


def _mixture(rng, n_trials=6, n_scalp=5, n_times=128):
    cortical = rng.standard_normal((n_trials, n_scalp, n_times)).astype(np.float32)
    exg = rng.standard_normal((n_trials, N_EXG, n_times)).astype(np.float32)
    mixing = rng.standard_normal((n_trials, n_scalp, N_EXG)).astype(np.float32)
    scalp = cortical + np.einsum("ick,ikt->ict", mixing, exg) + 3.0
    return cortical, exg, scalp


def test_regress_out_removes_the_exg_mixture_and_keeps_the_rest():
    cortical, exg, scalp = _mixture(np.random.default_rng(0))
    residual = regress_out(scalp, exg)
    assert residual.shape == scalp.shape and residual.dtype == scalp.dtype
    for i in range(len(scalp)):
        assert np.abs(exg[i] @ residual[i].T).max() < 1e-2
        assert abs(residual[i].mean()) < 1e-3
    relative_error = np.linalg.norm(residual - cortical) / np.linalg.norm(cortical)
    assert relative_error < 0.35
    assert np.linalg.norm(scalp - cortical) > 5 * np.linalg.norm(residual - cortical)


def test_regress_out_is_trial_wise():
    cortical, exg, scalp = _mixture(np.random.default_rng(1))
    before = regress_out(scalp, exg)
    exg[1] *= 10.0
    after = regress_out(scalp, exg)
    assert np.array_equal(before[0], after[0])
    assert not np.array_equal(before[1], after[1])


def test_regress_out_rejects_misaligned_trials():
    cortical, exg, scalp = _mixture(np.random.default_rng(2))
    with pytest.raises(ValueError):
        regress_out(scalp[:-1], exg)
    with pytest.raises(ValueError):
        regress_out(scalp, exg[..., :-1])


def test_make_signal_selects_by_name():
    cortical, exg, scalp = _mixture(np.random.default_rng(3))
    reversed_names = EXG_NAMES[::-1]
    assert make_signal("exg", scalp, exg, EXG_NAMES) is exg
    lips = make_signal("lips", scalp, exg, reversed_names)
    assert lips.shape[1] == 2
    assert np.array_equal(lips[:, 0], exg[:, reversed_names.index(LIP_CHANNELS[0])])
    assert np.array_equal(make_signal("scalp_regressed", scalp, exg, EXG_NAMES), regress_out(scalp, exg))
    with pytest.raises(ValueError):
        make_signal("scalp", scalp, exg, EXG_NAMES)
    assert set(SIGNALS) == {"exg", "lips", "scalp_regressed"}


def _write_exg(root: Path, subject: str, session: str, scale: float = 1.0):
    session_dir = root / "derivatives" / subject / session
    eeg = mne.read_epochs(session_dir / f"{subject}_{session}_eeg-epo.fif", preload=True, verbose=False)
    rng = np.random.default_rng(int(session[-1]))
    data = scale * rng.standard_normal((len(eeg), N_EXG, len(eeg.times)))
    info = mne.create_info(EXG_NAMES, SFREQ, "eeg")
    exg = mne.EpochsArray(data, info, events=eeg.events, tmin=eeg.tmin, event_id=eeg.event_id, verbose=False)
    exg.save(session_dir / f"{subject}_{session}_exg-epo.fif", verbose=False)


N_PER_SESSION = {"pronounced": 20, "inner_speech": 20, "visualized": 4}


@pytest.fixture
def exg_root(tmp_path):
    for i, session in enumerate(["ses-01", "ses-02"]):
        _write_session(tmp_path, "sub-01", session, offset=100 * i, n_per_session=N_PER_SESSION)
        _write_exg(tmp_path, "sub-01", session)
    return tmp_path


def test_loader_reads_exg_epochs_with_the_shared_events(exg_root):
    eeg = InnerSpeechDataset(str(exg_root), condition="all")
    exg = InnerSpeechDataset(str(exg_root), condition="all", epochs_suffix="exg")
    assert exg.ch_names == EXG_NAMES and eeg.ch_names != EXG_NAMES
    assert np.array_equal(eeg.labels, exg.labels)
    assert eeg.conditions_list == exg.conditions_list and eeg.session_ids == exg.session_ids
    assert exg.data.shape == (len(eeg), N_EXG, eeg.data.shape[-1])


def test_load_signals_returns_aligned_arrays_in_the_window(exg_root):
    data = load_signals(str(exg_root), "sub-01", "action")
    n_times = int(2.5 * SFREQ) + 1
    assert data["scalp"].shape == (len(data["labels"]), N_CH, n_times)
    assert data["exg"].shape == (len(data["labels"]), N_EXG, n_times)
    assert data["exg_names"] == EXG_NAMES
    assert set(data["conditions"]) == {"pronounced", "inner_speech", "visualized"}


def test_load_signals_rejects_misaligned_exg(exg_root):
    exg_file = exg_root / "derivatives/sub-01/ses-02/sub-01_ses-02_exg-epo.fif"
    exg = mne.read_epochs(exg_file, preload=True, verbose=False)
    exg[:-1].save(exg_file, overwrite=True, verbose=False)
    with pytest.raises(ValueError):
        load_signals(str(exg_root), "sub-01", "action")


def test_run_subject_writes_one_row_per_signal_and_arm_and_resumes(exg_root, tmp_path):
    args = Namespace(
        data_root=str(exg_root), out_dir=str(tmp_path / "out"), windows=["action"], signals=list(SIGNALS),
        models=["Riemannian"], seeds=[0], arms=["source_only", "target_only"], band=None, device="cpu",
    )
    (tmp_path / "out").mkdir()
    run_subject(args, "sub-01")
    frame = pd.read_csv(tmp_path / "out" / "sub-01.csv")
    assert sorted(frame["signal"].unique()) == sorted(SIGNALS)
    assert len(frame) == len(SIGNALS) * 2
    assert set(frame["arm"]) == {"source_only", "target_only"}
    assert frame["n_test"].gt(0).all()
    run_subject(args, "sub-01")
    assert len(pd.read_csv(tmp_path / "out" / "sub-01.csv")) == len(frame)
