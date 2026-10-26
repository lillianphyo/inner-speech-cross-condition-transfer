"""Within-subject arms of exp05 with a held-out recording session as the test set (ds003626 has three).

Every (subject, session) is a fold: the session's trials of a condition are the test set, the other
two sessions supply training and validation. One CSV per subject under --out-dir, resumable per
(window, model, seed, session); `--summarize` pools every per-subject CSV (mean over seeds and sessions).

    python experiments/inner_speech/exp08_session_split.py --subjects sub-01 sub-02
    python experiments/inner_speech/exp08_session_split.py --summarize
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.inner_speech.classical import CLASSICAL
from experiments.inner_speech.exp05_honest_transfer import MODELS, WINDOWS, append_rows, load_conditions
from experiments.inner_speech.honest_transfer import ARMS, DEFAULT_CONFIG, run_subject_arms, summarize
from src.data.inner_speech_loader import InnerSpeechDataset
from src.utils.run_config import ensure_same_run_config

RUN_KEYS = ["window", "model", "seed", "session"]


def run_subject(args, subject: str) -> None:
    out = Path(args.out_dir) / f"{subject}.csv"
    done = {}
    if out.exists():
        done = pd.read_csv(out).groupby(RUN_KEYS)["arm"].agg(set).to_dict()

    for window in args.windows:
        pending = [
            (model, seed, session, missing)
            for model in args.models for seed in args.seeds for session in args.sessions
            if (missing := tuple(
                arm for arm in args.arms
                if arm not in done.get((window, model, seed, session), set()) and not (arm == "mmd" and model in CLASSICAL)
            ))
        ]
        if not pending:
            continue
        data = load_conditions(args.data_root, subject, window, args.band)
        sessions = (data["sessions"]["pronounced"], data["sessions"]["inner_speech"])
        for model, seed, session, missing in pending:
            rows = run_subject_arms(
                *data["pronounced"], *data["inner_speech"],
                model_name=model, seed=seed, config=DEFAULT_CONFIG, device=args.device,
                train_fractions=tuple(args.train_fractions), arms=missing,
                held_out_session=session, sessions=sessions,
            )
            frame = pd.DataFrame([dict(subject=subject, window=window, model=model, seed=seed, **row) for row in rows])
            append_rows(out, frame)
            print(f"{subject} {window} {model} seed={seed} {session}: "
                  + "  ".join(f"{row['arm']}={row['accuracy']:.3f}" for row in rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/ds003626")
    parser.add_argument("--out-dir", default="runs/session_split")
    parser.add_argument("--subjects", nargs="+", default=InnerSpeechDataset.SUBJECTS)
    parser.add_argument("--sessions", nargs="+", default=list(InnerSpeechDataset.SESSIONS))
    parser.add_argument("--windows", nargs="+", default=["action"], choices=list(WINDOWS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS) + list(CLASSICAL))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--train-fractions", nargs="+", type=float, default=[1.0])
    parser.add_argument("--band", nargs=2, type=float, metavar=("LOW", "HIGH"))
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

    ensure_same_run_config(out_dir, dict(
        band=args.band, sessions=args.sessions, train_fractions=args.train_fractions, training=DEFAULT_CONFIG, windows=WINDOWS,
    ))
    torch.set_num_threads(args.threads)
    for subject in args.subjects:
        run_subject(args, subject)


if __name__ == "__main__":
    main()
