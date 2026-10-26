"""Leave-one-subject-out on ds003626, action window. One CSV per held-out subject under --out-dir,
resumable per (model, seed, arm).

    python experiments/inner_speech/exp07_loso.py --device cuda
    python experiments/inner_speech/exp07_loso.py --summarize
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.inner_speech.classical import CLASSICAL
from experiments.inner_speech.exp05_honest_transfer import MODELS, WINDOWS, append_rows, load_conditions
from experiments.inner_speech.honest_transfer import DEFAULT_CONFIG, summarize
from experiments.inner_speech.loso import ARMS, run_loso_arms
from src.data.inner_speech_loader import InnerSpeechDataset
from src.utils.run_config import ensure_same_run_config

WINDOW = "action"
RUN_KEYS = ["model", "seed"]
SUMMARY_KEYS = ("window", "model", "arm")


def run_held_out(args, data: dict, subject: str) -> None:
    out = Path(args.out_dir) / f"{subject}.csv"
    done = {}
    if out.exists():
        done = pd.read_csv(out).groupby(RUN_KEYS)["arm"].agg(set).to_dict()
    for model in args.models:
        for seed in args.seeds:
            missing = tuple(arm for arm in args.arms if arm not in done.get((model, seed), set()))
            if not missing:
                continue
            rows = run_loso_arms(data, subject, model_name=model, seed=seed, config=DEFAULT_CONFIG,
                                 device=args.device, arms=missing)
            frame = pd.DataFrame([dict(subject=subject, window=WINDOW, model=model, seed=seed, **row) for row in rows])
            append_rows(out, frame)
            print(f"{subject} {model} seed={seed}: "
                  + "  ".join(f"{row['arm']}={row['accuracy']:.3f}" for row in rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/ds003626")
    parser.add_argument("--out-dir", default="runs/loso")
    parser.add_argument("--subjects", nargs="+", default=InnerSpeechDataset.SUBJECTS,
                        help="held-out subjects to run; the pool is always every subject")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS) + list(CLASSICAL))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
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

    ensure_same_run_config(out_dir, dict(window=WINDOWS[WINDOW], training=DEFAULT_CONFIG,
                                         pool=InnerSpeechDataset.SUBJECTS))
    torch.set_num_threads(args.threads)
    data = {subject: load_conditions(args.data_root, subject, WINDOW) for subject in InnerSpeechDataset.SUBJECTS}
    for subject in args.subjects:
        run_held_out(args, data, subject)


if __name__ == "__main__":
    main()
