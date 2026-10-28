"""Honest within-subject cross-condition transfer on ds003626.

One CSV per subject under --out-dir, resumable per (window, model, seed). Run subjects in
separate processes to parallelise; `--summarize` pools every per-subject CSV.

    python experiments/inner_speech/exp05_honest_transfer.py --subjects sub-01 sub-02
    python experiments/inner_speech/exp05_honest_transfer.py --summarize
    python experiments/inner_speech/exp05_honest_transfer.py --band 8 30 --out-dir runs/honest_transfer_8_30
    python experiments/inner_speech/exp05_honest_transfer.py --arms target_only finetune shuffle_finetune --out-dir runs/finetune
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
from experiments.inner_speech.honest_transfer import ARMS, DEFAULT_CONFIG, NETWORK_ONLY, run_subject_arms, summarize
from src.data.inner_speech_loader import InnerSpeechDataset
from src.utils.run_config import ensure_same_run_config

WINDOWS = {"full": (None, None), "action": (0.5, 3.0), "cue": (-0.5, 0.5), "concentration": (-0.5, 0.0)}
MODELS = ("EEGNet", "ShallowConvNet")
RUN_KEYS = ["window", "model", "seed"]


def load_conditions(root: str, subject: str, window: str, band: tuple[float, float] | None = None) -> dict:
    """Pronounced and inner-speech trials of `subject` in `window`, band-passed to `band` Hz when given."""
    tmin, tmax = WINDOWS[window]
    dataset = InnerSpeechDataset(root, subjects=[subject], condition="all", tmin=tmin, tmax=tmax)
    conditions = np.array(dataset.conditions_list)
    sessions = np.array(dataset.session_ids)
    X = dataset.data if band is None else bandpass(dataset.data, dataset.sfreq, *band)
    data = {
        name: (X[conditions == name], dataset.labels[conditions == name])
        for name in ("pronounced", "inner_speech")
    }
    data["sessions"] = {name: sessions[conditions == name] for name in ("pronounced", "inner_speech")}
    data["ch_names"] = dataset.ch_names
    return data


def append_rows(out: Path, frame: pd.DataFrame) -> None:
    """Append `frame` to the CSV at `out`, widening the header when the metrics gained a column."""
    if out.exists():
        frame = pd.concat([pd.read_csv(out), frame], ignore_index=True)
    frame.to_csv(out, index=False)


def run_subject(args, subject: str) -> None:
    out = Path(args.out_dir) / f"{subject}.csv"
    done = {}
    if out.exists():
        done = pd.read_csv(out).groupby(RUN_KEYS)["arm"].agg(set).to_dict()

    for window in args.windows:
        pending = [
            (model, seed, missing) for model in args.models for seed in args.seeds
            if (missing := tuple(
                arm for arm in args.arms
                if arm not in done.get((window, model, seed), set()) and not (arm in NETWORK_ONLY and model in CLASSICAL)
            ))
        ]
        if not pending:
            continue
        data = load_conditions(args.data_root, subject, window, args.band)
        for model, seed, missing in pending:
            rows = run_subject_arms(
                *data["pronounced"], *data["inner_speech"],
                model_name=model, seed=seed, config=DEFAULT_CONFIG, device=args.device,
                train_fractions=tuple(args.train_fractions), arms=missing,
            )
            frame = pd.DataFrame([dict(subject=subject, window=window, model=model, seed=seed, **row) for row in rows])
            append_rows(out, frame)
            print(f"{subject} {window} {model} seed={seed}: "
                  + "  ".join(f"{row['arm']}={row['accuracy']:.3f}" for row in rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/ds003626")
    parser.add_argument("--out-dir", default="runs/honest_transfer")
    parser.add_argument("--subjects", nargs="+", default=InnerSpeechDataset.SUBJECTS)
    parser.add_argument("--windows", nargs="+", default=list(WINDOWS), choices=list(WINDOWS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS) + list(CLASSICAL))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--train-fractions", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--band", nargs=2, type=float, metavar=("LOW", "HIGH"),
                        help="zero-phase band-pass in Hz applied to every trial before z-scoring")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        results = pd.concat(pd.read_csv(path) for path in sorted(out_dir.glob("sub-*.csv")))
        summary = summarize(results)
        summary.to_csv(out_dir / "summary.csv", index=False)
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        return

    ensure_same_run_config(
        out_dir, dict(band=args.band, train_fractions=args.train_fractions, training=DEFAULT_CONFIG, windows=WINDOWS),
        extensible=("windows",),
    )
    torch.set_num_threads(args.threads)
    for subject in args.subjects:
        run_subject(args, subject)


if __name__ == "__main__":
    main()
