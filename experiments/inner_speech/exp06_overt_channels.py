"""Overt-only and inner-only accuracy on ds003626 under position-based channel subsets.

Action window only. One CSV per subject under --out-dir, resumable per (model, seed, condition).

    python experiments/inner_speech/exp06_overt_channels.py --subjects sub-01 sub-02
    python experiments/inner_speech/exp06_overt_channels.py --summarize
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.inner_speech.classical import CLASSICAL
from experiments.inner_speech.exp05_honest_transfer import MODELS, WINDOWS, append_rows, load_conditions
from experiments.inner_speech.honest_transfer import DEFAULT_CONFIG, run_subject_arms, summarize
from experiments.inner_speech.overt_channels import channel_conditions
from src.data.inner_speech_loader import InnerSpeechDataset
from src.utils.run_config import ensure_same_run_config

WINDOW = "action"
ARMS = ("source_only", "target_only")
RUN_KEYS = ["model", "seed", "condition"]
SUMMARY_KEYS = ("condition", "n_channels", "model", "arm")


def run_subject(args, subject: str) -> None:
    out = Path(args.out_dir) / f"{subject}.csv"
    done = set()
    if out.exists():
        done = set(map(tuple, pd.read_csv(out)[RUN_KEYS].drop_duplicates().itertuples(index=False)))

    data = load_conditions(args.data_root, subject, WINDOW)
    conditions = channel_conditions(data["ch_names"], args.n_random, np.random.default_rng(args.channel_seed))
    for model in args.models:
        for seed in args.seeds:
            for name, members in conditions.items():
                if (model, seed, name) in done:
                    continue
                picks = [data["ch_names"].index(channel) for channel in members]
                (X_pron, y_pron), (X_inner, y_inner) = data["pronounced"], data["inner_speech"]
                rows = run_subject_arms(
                    X_pron[:, picks], y_pron, X_inner[:, picks], y_inner,
                    model_name=model, seed=seed, config=DEFAULT_CONFIG, device=args.device, arms=ARMS,
                )
                frame = pd.DataFrame([
                    dict(subject=subject, window=WINDOW, model=model, seed=seed, condition=name,
                         n_channels=len(members), channels="|".join(members), **row)
                    for row in rows
                ])
                append_rows(out, frame)
                print(f"{subject} {model} seed={seed} {name}: "
                      + "  ".join(f"{row['arm']}={row['accuracy']:.3f}" for row in rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/ds003626")
    parser.add_argument("--out-dir", default="runs/overt_channels")
    parser.add_argument("--subjects", nargs="+", default=InnerSpeechDataset.SUBJECTS)
    parser.add_argument("--models", nargs="+", default=["EEGNet"], choices=list(MODELS) + list(CLASSICAL))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-random", type=int, default=3)
    parser.add_argument("--channel-seed", type=int, default=0)
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

    ensure_same_run_config(out_dir, dict(
        window=WINDOWS[WINDOW], training=DEFAULT_CONFIG, n_random=args.n_random, channel_seed=args.channel_seed,
    ))
    torch.set_num_threads(args.threads)
    for subject in args.subjects:
        run_subject(args, subject)


if __name__ == "__main__":
    main()
