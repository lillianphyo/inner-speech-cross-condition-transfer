"""How much of a filtered epoch comes from its neighbours' recording, and whether neighbours share a class.

The no-ICA rebuild (raw_epochs.py) band-passes the continuous recording with a zero-phase FIR whose
impulse response is longer than the gap between epochs. The in-band activity of one trial is its window of
the filtered recording; filtering it again alone gives the part of every other epoch that comes from it,
since the filter is close to idempotent in its pass band. Masking the raw recording instead would measure
the step edges of its DC offset, which the inter-trial intervals cancel in the real signal.

    python experiments/inner_speech/filter_spread.py --raw-root /path/ds003626 --subject sub-01 --session ses-01
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.inner_speech.raw_epochs import BAND_HZ, NOTCH_HZ, REFERENCE, TMAX, TMIN

CHANNELS = ("A1", "B1", "C1", "D1", "A19", "C21", "D15", "B10")


def _window(signal: np.ndarray, onset: int, lo: int, hi: int) -> np.ndarray:
    segment = signal[:, onset + lo: onset + hi]
    return segment - segment.mean(axis=1, keepdims=True)


def neighbour_share(x: np.ndarray, onsets: np.ndarray, lo: int, hi: int, filt, trials) -> np.ndarray:
    """Norm of the part of each neighbouring mean-removed epoch that comes from trial k's in-band activity,
    relative to that epoch's full filtered norm; two values (before, after) per trial in `trials`."""
    if min(trials) < 1 or max(trials) > len(onsets) - 2:
        raise ValueError("every trial needs a neighbour on both sides")
    filtered = filt(x)
    shares = []
    for k in trials:
        alone = np.zeros_like(filtered)
        alone[:, onsets[k] + lo: onsets[k] + hi] = filtered[:, onsets[k] + lo: onsets[k] + hi]
        from_k = filt(alone)
        for j in (k - 1, k + 1):
            shares.append(np.linalg.norm(_window(from_k, onsets[j], lo, hi))
                          / np.linalg.norm(_window(filtered, onsets[j], lo, hi)))
    return np.array(shares)


def same_class_rate(events: np.ndarray) -> tuple[float, int]:
    """Share of time-adjacent trial pairs of the same condition that are of the same class.
    `events` rows are (sample, class, condition, session), as in the *_events.dat files."""
    ordered = events[np.argsort(events[:, 0])]
    same_condition = ordered[1:, 2] == ordered[:-1, 2]
    same_class = (ordered[1:, 1] == ordered[:-1, 1]) & same_condition
    return float(same_class.sum() / same_condition.sum()), int(same_condition.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--session", default="ses-01")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.raw_root)
    stem = f"{args.subject}_{args.session}"

    raw = mne.io.read_raw_bdf(root / args.subject / args.session / "eeg" / f"{stem}_task-innerspeech_eeg.bdf",
                              preload=True, verbose=False)
    raw.set_eeg_reference(ref_channels=REFERENCE, verbose=False)
    raw.pick(list(CHANNELS))
    sfreq = raw.info["sfreq"]
    onsets = mne.read_epochs(root / "derivatives" / args.subject / args.session / f"{stem}_eeg-epo.fif",
                             preload=False, verbose=False).events[:, 0]
    lo, hi = int(TMIN * sfreq), int(TMAX * sfreq)

    def filt(signal):
        signal = mne.filter.notch_filter(signal, sfreq, NOTCH_HZ, verbose=False)
        return mne.filter.filter_data(signal, sfreq, *BAND_HZ, verbose=False)

    kernel = mne.filter.create_filter(None, sfreq, *BAND_HZ, verbose=False)
    taps = len(kernel)
    gaps = (onsets[1:] + lo - (onsets[:-1] + hi)) / sfreq
    lag = np.abs(np.arange(taps) - taps // 2) / sfreq
    tail = np.linalg.norm(kernel[lag >= gaps.min()]) / np.linalg.norm(kernel)
    print(f"band-pass kernel norm at lags beyond the shortest gap: {tail:.2e}")
    trials = np.sort(np.random.default_rng(args.seed).choice(np.arange(1, len(onsets) - 1), args.n_trials,
                                                             replace=False))
    shares = neighbour_share(raw.get_data(), onsets, lo, hi, filt, trials)
    print(f"band-pass taps {taps} (half-span {taps / 2 / sfreq:.2f} s); "
          f"gap between epochs {gaps.min():.2f}-{np.median(gaps):.2f} s (min-median)")
    print(f"neighbour share of epoch norm: median {np.median(shares):.5f}, max {shares.max():.5f}")

    files = sorted(glob.glob(str(root / "derivatives" / "sub-*" / "ses-*" / "*_events.dat")))
    rates = [same_class_rate(np.load(f, allow_pickle=True)) for f in files]
    pairs = sum(n for _, n in rates)
    pooled = sum(r * n for r, n in rates) / pairs
    print(f"adjacent same-condition pairs {pairs} in {len(files)} sessions; same class {pooled:.4f} (chance 0.25)")


if __name__ == "__main__":
    main()
