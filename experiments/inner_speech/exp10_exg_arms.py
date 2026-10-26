"""EXG artefact arms of the Sensors study on ds003626 (reviewers' E1).

The same within-subject arms as exp05, run on three signals instead of the scalp montage: the eight
external electrodes (`exg`), the lip pair (`lips`) and the scalp with the EXG regressed out trial by
trial (`scalp_regressed`). The EXG epochs share the events table of the EEG epochs and are checked
to carry the same labels, conditions and sessions before any split. One CSV per subject under
--out-dir with a `signal` column, resumable per (window, signal, model, seed).

    python experiments/inner_speech/exp10_exg_arms.py --data-root F:/datasets/ds003626 --subjects sub-01
    python experiments/inner_speech/exp10_exg_arms.py --summarize
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.inner_speech.band import bandpass
from experiments.inner_speech.classical import CLASSICAL
from experiments.inner_speech.exg import SIGNALS, make_signal
from experiments.inner_speech.exp05_honest_transfer import MODELS, WINDOWS, append_rows
from experiments.inner_speech.honest_transfer import ARMS, DEFAULT_CONFIG, run_subject_arms, summarize
from src.data.inner_speech_loader import InnerSpeechDataset
from src.utils.run_config import ensure_same_run_config

DEFAULT_ARMS = ("source_only", "shuffle_source", "target_only", "shuffle_target", "transfer", "shuffle_transfer")
RUN_KEYS = ["window", "signal", "model", "seed"]
SUMMARY_KEYS = ("window", "signal", "model", "arm", "train_fraction")


def load_signals(root: str, subject: str, window: str, band: tuple[float, float] | None = None) -> dict:
    """Scalp and EXG trials of `subject` in `window`, aligned trial by trial, with labels and conditions."""
    tmin, tmax = WINDOWS[window]
    scalp, exg = (
        InnerSpeechDataset(root, subjects=[subject], condition="all", tmin=tmin, tmax=tmax, epochs_suffix=suffix)
        for suffix in ("eeg", "exg")
    )
    aligned = (
        np.array_equal(scalp.labels, exg.labels)
        and scalp.conditions_list == exg.conditions_list
        and scalp.session_ids == exg.session_ids
        and scalp.sfreq == exg.sfreq
    )
    if not aligned:
        raise ValueError(f"{subject}: the EXG epochs are not aligned with the EEG epochs")
    X_scalp, X_exg = scalp.data, exg.data
    if band is not None:
        X_scalp, X_exg = bandpass(X_scalp, scalp.sfreq, *band), bandpass(X_exg, exg.sfreq, *band)
    return dict(
        scalp=X_scalp, exg=X_exg, exg_names=list(exg.ch_names),
        labels=np.asarray(scalp.labels), conditions=np.array(scalp.conditions_list),
    )


def condition_arrays(X: np.ndarray, data: dict) -> tuple:
    pron, inner = (data["conditions"] == name for name in ("pronounced", "inner_speech"))
    return X[pron], data["labels"][pron], X[inner], data["labels"][inner]


def run_subject(args, subject: str) -> None:
    out = Path(args.out_dir) / f"{subject}.csv"
    done = {}
    if out.exists():
        done = pd.read_csv(out).groupby(RUN_KEYS)["arm"].agg(set).to_dict()

    for window in args.windows:
        pending = [
            (signal, model, seed, missing) for signal in args.signals for model in args.models for seed in args.seeds
            if (missing := tuple(arm for arm in args.arms if arm not in done.get((window, signal, model, seed), set())))
        ]
        if not pending:
            continue
        data = load_signals(args.data_root, subject, window, args.band)
        signals = {signal: make_signal(signal, data["scalp"], data["exg"], data["exg_names"]) for signal in args.signals}
        for signal, model, seed, missing in pending:
            rows = run_subject_arms(
                *condition_arrays(signals[signal], data),
                model_name=model, seed=seed, config=DEFAULT_CONFIG, device=args.device, arms=missing,
            )
            frame = pd.DataFrame([
                dict(subject=subject, window=window, signal=signal, model=model, seed=seed, **row) for row in rows
            ])
            append_rows(out, frame)
            print(f"{subject} {window} {signal} {model} seed={seed}: "
                  + "  ".join(f"{row['arm']}={row['accuracy']:.3f}" for row in rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/ds003626")
    parser.add_argument("--out-dir", default="runs/exg_arms")
    parser.add_argument("--subjects", nargs="+", default=InnerSpeechDataset.SUBJECTS)
    parser.add_argument("--windows", nargs="+", default=["action"], choices=list(WINDOWS))
    parser.add_argument("--signals", nargs="+", default=list(SIGNALS), choices=list(SIGNALS))
    parser.add_argument("--models", nargs="+", default=list(MODELS) + list(CLASSICAL), choices=list(MODELS) + list(CLASSICAL))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(ARMS))
    parser.add_argument("--band", nargs=2, type=float, metavar=("LOW", "HIGH"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        results = pd.concat(pd.read_csv(path) for path in sorted(out_dir.glob("sub-*.csv")))
        summary = summarize(results, SUMMARY_KEYS)
        summary.to_csv(out_dir / "summary.csv", index=False)
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        return

    ensure_same_run_config(out_dir, dict(band=args.band, training=DEFAULT_CONFIG, windows=WINDOWS), extensible=("windows",))
    torch.set_num_threads(args.threads)
    for subject in args.subjects:
        run_subject(args, subject)


if __name__ == "__main__":
    main()
