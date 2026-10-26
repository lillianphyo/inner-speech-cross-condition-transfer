"""Condition-separation tests for the ds003626 loader, on a synthetic derivatives tree.

The .fif event names in ds003626 carry only the word (Arriba/Abajo/Derecha/Izquierda);
the condition lives in column 2 of the companion `*_events.dat`. Every trial is tagged
with its own index so overlap between conditions is detectable.
"""
import pickle
import sys
from pathlib import Path

import mne
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.inner_speech_loader import InnerSpeechDataset

SFREQ = 64
N_CH = 4
N_TIMES = int(4.5 * SFREQ) + 1
N_PER_SESSION = {"pronounced": 4, "inner_speech": 8, "visualized": 8}
CONDITION_CODE = {"pronounced": 0, "inner_speech": 1, "visualized": 2}
FIF_CODES = np.array([31, 32, 33, 34])


def _write_session(root: Path, subject: str, session: str, offset: int, with_events: bool = True,
                   n_per_session: dict = N_PER_SESSION):
    conditions = np.concatenate([np.full(n, CONDITION_CODE[c]) for c, n in n_per_session.items()])
    n = len(conditions)
    classes = np.arange(n) % 4
    data = np.zeros((n, N_CH, N_TIMES))
    data += (offset + np.arange(n))[:, None, None]
    data[:, 1, :] = np.arange(N_TIMES)
    info = mne.create_info([f"A{i + 1}" for i in range(N_CH)], SFREQ, "eeg")
    fif_events = np.column_stack([np.arange(n) * N_TIMES * 2, np.zeros(n, int), FIF_CODES[classes]])
    epochs = mne.EpochsArray(
        data, info, events=fif_events, tmin=-0.5,
        event_id=dict(Arriba=31, Abajo=32, Derecha=33, Izquierda=34), verbose=False,
    )
    session_dir = root / "derivatives" / subject / session
    session_dir.mkdir(parents=True)
    epochs.save(session_dir / f"{subject}_{session}_eeg-epo.fif", verbose=False)
    if with_events:
        events = np.column_stack([fif_events[:, 0], classes, conditions, np.full(n, int(session[-1]))])
        with open(session_dir / f"{subject}_{session}_events.dat", "wb") as f:
            pickle.dump(events, f)
    return classes, conditions


@pytest.fixture
def dataset_root(tmp_path):
    for i, session in enumerate(["ses-01", "ses-02"]):
        _write_session(tmp_path, "sub-01", session, offset=100 * i)
    return tmp_path


def _trial_ids(ds: InnerSpeechDataset) -> set:
    return set(ds.data[:, 0, 0].astype(int).tolist())


def test_conditions_are_disjoint_and_complete(dataset_root):
    loaded = {c: InnerSpeechDataset(str(dataset_root), condition=c) for c in N_PER_SESSION}
    ids = {c: _trial_ids(ds) for c, ds in loaded.items()}

    for c, n in N_PER_SESSION.items():
        assert len(loaded[c]) == 2 * n
        assert set(loaded[c].conditions_list) == {c}
    assert not ids["pronounced"] & ids["inner_speech"]
    assert not ids["pronounced"] & ids["visualized"]
    assert not ids["inner_speech"] & ids["visualized"]

    everything = InnerSpeechDataset(str(dataset_root), condition="all")
    assert _trial_ids(everything) == ids["pronounced"] | ids["inner_speech"] | ids["visualized"]


def test_channel_names_follow_the_fif(dataset_root):
    ds = InnerSpeechDataset(str(dataset_root), condition="all")
    assert ds.ch_names == ["A1", "A2", "A3", "A4"]


def test_sampling_rate_follows_the_fif(dataset_root):
    assert InnerSpeechDataset(str(dataset_root), condition="inner_speech").sfreq == SFREQ


def test_labels_follow_events_file(dataset_root):
    ds = InnerSpeechDataset(str(dataset_root), condition="inner_speech", sessions=["ses-01"])
    trial_index = ds.data[:, 0, 0].astype(int)
    assert np.array_equal(ds.labels, trial_index % 4)


def test_missing_events_file_raises(tmp_path):
    _write_session(tmp_path, "sub-01", "ses-01", offset=0, with_events=False)
    with pytest.raises(FileNotFoundError):
        InnerSpeechDataset(str(tmp_path), condition="pronounced")


def test_events_file_disagreeing_with_fif_raises(tmp_path):
    _write_session(tmp_path, "sub-01", "ses-01", offset=0)
    events_file = tmp_path / "derivatives/sub-01/ses-01/sub-01_ses-01_events.dat"
    events = pickle.loads(events_file.read_bytes())
    events[:, 1] = (events[:, 1] + 1) % 4
    events_file.write_bytes(pickle.dumps(events))
    with pytest.raises(ValueError):
        InnerSpeechDataset(str(tmp_path), condition="pronounced")


def test_unknown_condition_raises(dataset_root):
    with pytest.raises(ValueError):
        InnerSpeechDataset(str(dataset_root), condition="overt")


def test_time_window_crops_relative_to_cue_onset(dataset_root):
    full = InnerSpeechDataset(str(dataset_root), condition="inner_speech")
    action = InnerSpeechDataset(str(dataset_root), condition="inner_speech", tmin=0.5, tmax=3.0)
    cue = InnerSpeechDataset(str(dataset_root), condition="inner_speech", tmin=-0.5, tmax=0.5)

    assert full.get_num_timepoints() == N_TIMES
    assert action.get_num_timepoints() == int(2.5 * SFREQ) + 1
    assert cue.get_num_timepoints() == int(1.0 * SFREQ) + 1
    assert len(action) == len(full)
    assert (action.data[0, 1, 0], action.data[0, 1, -1]) == (SFREQ, 3.5 * SFREQ)
    assert (cue.data[0, 1, 0], cue.data[0, 1, -1]) == (0, SFREQ)
