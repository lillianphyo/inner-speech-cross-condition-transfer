"""Raw-rebuilt ds003626 epochs: authors' fixed steps only, trial-local, aligned with the derivatives."""
import pickle
import sys
from pathlib import Path

import mne
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.inner_speech.raw_epochs import DECIM, TMAX, TMIN, rebuild_session, session_epochs
from src.data.inner_speech_loader import InnerSpeechDataset

SFREQ = 1024
SCALP = ["A1", "A2", "A3"]
EXG = [f"EXG{i}" for i in range(1, 9)]
EVENT_ID = dict(Arriba=31, Abajo=32, Derecha=33, Izquierda=34)
ONSETS_S = (10.0, 30.0, 50.0, 70.0)
N_TIMES = int((TMAX - TMIN) * SFREQ / DECIM) + 1


def _raw(seed=0, duration=90.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * SFREQ)) / SFREQ
    reference = 50e-6 * rng.standard_normal(len(t)).cumsum() / 100
    data = np.zeros((len(SCALP) + len(EXG), len(t)))
    data[: len(SCALP)] = reference + 5e-6 * np.sin(2 * np.pi * 10 * t)
    data[len(SCALP):] = 1e-6 * rng.standard_normal((len(EXG), len(t)))
    data[len(SCALP)] = data[len(SCALP) + 1] = reference
    return mne.io.RawArray(data, mne.create_info(SCALP + EXG, SFREQ, "eeg"), verbose=False)


def _events():
    codes = [31, 33, 32, 34]
    return np.array([[int(s * SFREQ), 0, c] for s, c in zip(ONSETS_S, codes)])


def test_epochs_follow_the_authors_layout_and_drop_the_external_channels():
    epochs = session_epochs(_raw(), _events(), EVENT_ID)
    assert epochs.ch_names == SCALP
    assert epochs.info["sfreq"] == SFREQ / DECIM
    assert len(epochs.times) == N_TIMES and epochs.tmin == TMIN
    assert np.array_equal(epochs.events, _events())


def test_ear_lobe_reference_removes_the_shared_signal():
    X = session_epochs(_raw(), _events(), EVENT_ID).get_data()
    t = np.arange(N_TIMES) / (SFREQ / DECIM) + TMIN
    sine = 5e-6 * np.sin(2 * np.pi * 10 * (t[None, :] + np.array(ONSETS_S)[:, None]))
    residual = X[:, 0] - sine
    assert np.sqrt((residual ** 2).mean()) < 0.1 * np.sqrt((sine ** 2).mean())


def test_each_epoch_depends_only_on_its_own_stretch_of_recording():
    raw = _raw()
    before = session_epochs(raw.copy(), _events(), EVENT_ID).get_data()
    perturbed = raw.copy()
    perturbed._data[: len(SCALP), int(49 * SFREQ):int(53 * SFREQ)] += 1e-3
    after = session_epochs(perturbed, _events(), EVENT_ID).get_data()
    assert np.allclose(before[0], after[0]) and np.allclose(before[3], after[3])
    assert not np.allclose(before[2], after[2])


def _derivatives(root: Path, n_events=len(ONSETS_S)):
    session_dir = root / "derivatives" / "sub-01" / "ses-01"
    session_dir.mkdir(parents=True)
    events = _events()[:n_events]
    template = session_epochs(_raw(), events, EVENT_ID)
    template.save(session_dir / "sub-01_ses-01_eeg-epo.fif", verbose=False)
    classes = events[:, 2] - 31
    table = np.column_stack([events[:, 0], classes, [0, 1, 0, 1][:n_events], np.ones(n_events, int)])
    (session_dir / "sub-01_ses-01_events.dat").write_bytes(pickle.dumps(table))
    return root


def test_rebuilt_tree_loads_with_the_same_trials_and_labels(tmp_path):
    deriv = _derivatives(tmp_path / "deriv")
    out = tmp_path / "noica"
    rebuild_session(_raw(seed=1), deriv, out, "sub-01", "ses-01")
    original = InnerSpeechDataset(str(deriv), subjects=["sub-01"], sessions=["ses-01"], condition="all")
    rebuilt = InnerSpeechDataset(str(out), subjects=["sub-01"], sessions=["ses-01"], condition="all")
    assert np.array_equal(original.labels, rebuilt.labels)
    assert original.conditions_list == rebuilt.conditions_list
    assert rebuilt.data.shape == original.data.shape and rebuilt.ch_names == original.ch_names


def test_rebuild_refuses_a_recording_that_loses_trials(tmp_path):
    deriv = _derivatives(tmp_path / "deriv")
    with pytest.raises(ValueError):
        rebuild_session(_raw(duration=72.0), deriv, tmp_path / "noica", "sub-01", "ses-01")
