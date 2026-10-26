"""ds003626 scalp epochs rebuilt from the raw recordings with no step fitted across trials.

The dataset authors' derivatives were cleaned by an independent component analysis fitted on all
trials of a session, so training and test trials of that session share its unmixing. This rebuild
repeats their published fixed steps (ear-lobe reference, 50 Hz notch, 0.5-100 Hz zero-phase FIR on the
continuous recording, -0.5 to 4.0 s around cue onset, decimation by 4, per-epoch mean removal) and
leaves the decomposition out. Trials, word codes and the events table come from the derivatives, so the
output is a derivatives tree that InnerSpeechDataset reads unchanged.

    python experiments/inner_speech/raw_epochs.py --raw-root F:/datasets/ds003626 --out-root F:/datasets/ds003626_noica
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.inner_speech_loader import InnerSpeechDataset

REFERENCE = ["EXG1", "EXG2"]
EXTERNAL = [f"EXG{i}" for i in range(1, 9)]
NOTCH_HZ = 50.0
BAND_HZ = (0.5, 100.0)
TMIN, TMAX = -0.5, 4.0
DECIM = 4
SESSIONS = ("ses-01", "ses-02", "ses-03")


def session_epochs(raw: mne.io.BaseRaw, events: np.ndarray, event_id: dict) -> mne.Epochs:
    """Scalp epochs of one session; modifies `raw` in place."""
    raw.load_data()
    raw.set_eeg_reference(ref_channels=REFERENCE, verbose=False)
    raw.notch_filter(NOTCH_HZ, verbose=False)
    raw.filter(*BAND_HZ, verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, exclude=EXTERNAL)
    return mne.Epochs(
        raw, events, event_id=event_id, tmin=TMIN, tmax=TMAX, picks=picks, baseline=None,
        detrend=0, decim=DECIM, preload=True, reject_by_annotation=False, verbose=False,
    )


def rebuild_session(raw: mne.io.BaseRaw, deriv_root: Path, out_root: Path, subject: str, session: str) -> Path:
    """Write the rebuilt epochs and the events table of `subject`/`session` under `out_root`."""
    source = Path(deriv_root) / "derivatives" / subject / session
    stem = f"{subject}_{session}"
    template = mne.read_epochs(source / f"{stem}_eeg-epo.fif", preload=False, verbose=False)
    epochs = session_epochs(raw, template.events, template.event_id)
    if not (
        np.array_equal(epochs.events, template.events)
        and epochs.ch_names == template.ch_names
        and np.allclose(epochs.times, template.times)
    ):
        raise ValueError(f"{stem}: rebuilt epochs do not match the derivative trials, channels or times")
    target = Path(out_root) / "derivatives" / subject / session
    target.mkdir(parents=True, exist_ok=True)
    out = target / f"{stem}_eeg-epo.fif"
    epochs.save(out, overwrite=True, verbose=False)
    shutil.copy2(source / f"{stem}_events.dat", target / f"{stem}_events.dat")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--deriv-root", help="derivatives tree to align with; defaults to --raw-root")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--subjects", nargs="+", default=InnerSpeechDataset.SUBJECTS)
    parser.add_argument("--sessions", nargs="+", default=list(SESSIONS), choices=SESSIONS)
    args = parser.parse_args()
    deriv_root = Path(args.deriv_root or args.raw_root)
    for subject in args.subjects:
        for session in args.sessions:
            done = Path(args.out_root) / "derivatives" / subject / session / f"{subject}_{session}_events.dat"
            if done.exists():
                continue
            bdf = Path(args.raw_root) / subject / session / "eeg" / f"{subject}_{session}_task-innerspeech_eeg.bdf"
            out = rebuild_session(mne.io.read_raw_bdf(bdf, preload=True, verbose=False), deriv_root, args.out_root, subject, session)
            print(f"{subject} {session}: {out}", flush=True)


if __name__ == "__main__":
    main()
