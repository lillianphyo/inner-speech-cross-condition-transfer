"""Overt-speech reference arms of the Sensors study on a second dataset (ds006104, Moreira et al. 2025).

Per subject: overt-only (60/20/20 stratified) with its shuffled-label control, on the binary place
and voicing tasks. Across subjects: leave-one-subject-out (validation = next subject) on place,
voicing and the 20-class syllable task, with its shuffled control. Both broadband (1--40 Hz
preprocessing) and 8--30 Hz. Same decoders, normalisation and
statistics as exp05/exp07. Epochs are cached per subject (500 Hz, -0.5 to 0.2 s around speech
onset, scalp channels common to every subject).

    python experiments/inner_speech/exp09_second_dataset.py --data-root F:/speech-decoding/eeg_classification
    python experiments/inner_speech/exp09_second_dataset.py --summarize
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.inner_speech.band import bandpass
from experiments.inner_speech.channels import align_channels, scalp_channels
from experiments.inner_speech.classical import CLASSICAL
from experiments.inner_speech.exp05_honest_transfer import MODELS, append_rows
from experiments.inner_speech.honest_transfer import DEFAULT_CONFIG, fit_and_score, summarize
from experiments.inner_speech.loso import loso_folds
from src.data.datasets import StudyDataset
from src.data.leakage import assert_disjoint
from src.data.preprocessing import EEGPreprocessor, create_articulatory_labels, create_phoneme_labels
from src.data.splits import split_indices
from src.data.transforms import shuffle_labels
from src.utils.run_config import ensure_same_run_config

WITHIN_TASKS = ("place", "voicing")
LOSO_TASKS = ("place", "voicing", "phoneme")
WITHIN_ARMS = ("overt_only", "shuffle_overt")
LOSO_ARMS = ("overt_loso", "shuffle_overt_loso")
BANDS = {"broadband": None, "8_30": (8.0, 30.0)}
RUN_KEYS = ["task", "band", "model", "seed"]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "journal_extension" / "config_journal.yaml"
SFREQ = 500.0


def subject_epochs(study: StudyDataset, subject_id: str, h_freq: float, cache: Path):
    """One subject's epochs at SFREQ with its own channel names; cached per subject."""
    if not cache.exists():
        recording = study.loader.load_raw_eeg(subject_id)
        step, remainder = divmod(recording.info["sfreq"], SFREQ)
        if remainder or h_freq >= SFREQ / 2:
            raise ValueError(f"Cannot decimate {recording.info['sfreq']} Hz epochs to {SFREQ} Hz without aliasing.")
        subject = study.load_subject(subject_id)
        if subject.data.shape[1] != len(recording.ch_names):
            raise ValueError(f"{subject_id}: epochs have {subject.data.shape[1]} channels, recording lists {len(recording.ch_names)}.")
        cache.parent.mkdir(parents=True, exist_ok=True)
        subject.trial_info.assign(subject=subject_id).to_csv(cache.with_suffix(".csv"), index=False)
        np.savez(cache, X=subject.data[:, :, ::int(step)], ch_names=np.array(recording.ch_names))
        del study.subject_datasets[subject_id]
    stored = np.load(cache, allow_pickle=False)
    return stored["X"], pd.read_csv(cache.with_suffix(".csv")), stored["ch_names"].tolist()


def task_labels(trial_info: pd.DataFrame, task: str, task_type: str) -> np.ndarray:
    if task == "phoneme":
        return create_phoneme_labels(trial_info, task_type)[0]
    return create_articulatory_labels(trial_info, task)[0]
SUMMARY_KEYS = ("task", "band", "model", "arm")


def load_subjects(data_root: str, cache_dir: Path, max_subjects: int | None) -> dict[str, dict]:
    """`{subject: {"X": trials x common scalp channels x samples, "info": trial table}}`."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preprocessing = {key: config["preprocessing"][key] for key in ("l_freq", "h_freq", "epoch_tmin", "epoch_tmax", "baseline")}
    preprocessing["reject_threshold"] = None
    ensure_same_run_config(cache_dir, dict(
        preprocessing=preprocessing, sfreq=SFREQ,
        study=config["study"], task_type=config["task_type"], condition=config["condition"],
    ))
    study = StudyDataset(
        data_root=data_root, study=config["study"], task_type=config["task_type"], condition=config["condition"],
        max_subjects=max_subjects,
        preprocessor=EEGPreprocessor(
            l_freq=preprocessing["l_freq"], h_freq=preprocessing["h_freq"],
            epoch_tmin=preprocessing["epoch_tmin"], epoch_tmax=preprocessing["epoch_tmax"],
            baseline=tuple(preprocessing["baseline"]), reject_threshold=None,
        ),
    )
    loaded = {
        subject_id: subject_epochs(study, subject_id, preprocessing["h_freq"], cache_dir / f"{subject_id}.npz")
        for subject_id in study.subject_ids
    }
    pooled, common = align_channels([
        (X[:, [names.index(c) for c in scalp_channels(names)]], scalp_channels(names)) for X, _, names in loaded.values()
    ])
    data, start = {}, 0
    for subject_id, (X, info, _) in loaded.items():
        data[subject_id] = dict(X=pooled[start:start + len(X)], info=info, task_type=config["task_type"])
        start += len(X)
    print(f"{len(data)} subjects {list(data)}, {len(common)} common scalp channels, {pooled.shape[2]} samples", flush=True)
    return data


def labels_for(entry: dict, task: str) -> np.ndarray:
    return task_labels(entry["info"], task, entry["task_type"])


def banded(X: np.ndarray, band: tuple[float, float] | None) -> np.ndarray:
    return X if band is None else bandpass(X, SFREQ, *band)


def _row(arm, train_y, test_y, metrics, **extra) -> dict:
    classes = len(np.unique(train_y))
    return dict(arm=arm, n_train=len(train_y), n_test=len(test_y), chance=1.0 / classes,
                majority_rate=float(np.bincount(test_y).max() / len(test_y)), **extra, **metrics)


def within_rows(X: np.ndarray, y: np.ndarray, *, model_name: str, seed: int, config: dict = DEFAULT_CONFIG,
                device: str = "cpu", arms: tuple[str, ...] = WITHIN_ARMS) -> list[dict]:
    """Overt-only (stratified 60/20/20) and its shuffled-label control for one subject and task."""
    rng = np.random.default_rng(seed)
    train, val, test = split_indices(y, (0.6, 0.2, 0.2), rng)
    assert_disjoint(X[train], X[test])
    fit = dict(config=config, device=device, seed=seed, n_classes=len(np.unique(y)))
    rows = []
    for arm in arms:
        y_train, y_val = (y[train], y[val]) if arm == "overt_only" else (shuffle_labels(y[train], rng), shuffle_labels(y[val], rng))
        scores = fit_and_score(model_name, (X[train], y_train), (X[val], y_val), {"test": (X[test], y[test])}, **fit)
        rows.append(_row(arm, y[train], y[test], scores["test"]))
    return rows


def loso_rows(data: dict[str, tuple[np.ndarray, np.ndarray]], held_out: str, *, model_name: str, seed: int,
              config: dict = DEFAULT_CONFIG, device: str = "cpu", arms: tuple[str, ...] = LOSO_ARMS) -> list[dict]:
    """`data[subject] = (X, y)`; held-out subject never trains or selects; validation = next subject."""
    subjects = list(data)
    _, validation, training = next(fold for fold in loso_folds(subjects) if fold[0] == held_out)
    X_train = np.concatenate([data[s][0] for s in training])
    y_train = np.concatenate([data[s][1] for s in training])
    X_val, y_val = data[validation]
    X_test, y_test = data[held_out]
    assert_disjoint(X_train, X_test)
    assert_disjoint(X_val, X_test)
    rng = np.random.default_rng(seed)
    fit = dict(config=config, device=device, seed=seed, n_classes=len(np.unique(y_train)))
    rows = []
    for arm in arms:
        tr, va = (y_train, y_val) if arm == "overt_loso" else (shuffle_labels(y_train, rng), shuffle_labels(y_val, rng))
        scores = fit_and_score(model_name, (X_train, tr), (X_val, va), {"test": (X_test, y_test)}, **fit)
        rows.append(_row(arm, y_train, y_test, scores["test"], validation_subject=validation))
    return rows


def run(args, data: dict[str, dict]) -> None:
    out_dir = Path(args.out_dir)
    for subject in args.subjects or list(data):
        out = out_dir / f"{subject}.csv"
        done = {}
        if out.exists():
            done = pd.read_csv(out).groupby(RUN_KEYS)["arm"].agg(set).to_dict()
        for band_name in args.bands:
            band = BANDS[band_name]
            for task in sorted(set(args.within_tasks) | set(args.loso_tasks)):
                for model in args.models:
                    for seed in args.seeds:
                        finished = done.get((task, band_name, model, seed), set())
                        wanted = (WITHIN_ARMS if task in args.within_tasks else ()) + (LOSO_ARMS if task in args.loso_tasks else ())
                        missing = tuple(arm for arm in wanted if arm not in finished)
                        if not missing:
                            continue
                        rows = []
                        within = tuple(a for a in missing if a in WITHIN_ARMS)
                        if within:
                            X = banded(data[subject]["X"], band)
                            rows += within_rows(X, labels_for(data[subject], task), model_name=model, seed=seed,
                                                device=args.device, arms=within)
                        loso = tuple(a for a in missing if a in LOSO_ARMS)
                        if loso:
                            pooled = {s: (banded(entry["X"], band), labels_for(entry, task)) for s, entry in data.items()}
                            rows += loso_rows(pooled, subject, model_name=model, seed=seed, device=args.device, arms=loso)
                        frame = pd.DataFrame([dict(subject=subject, task=task, band=band_name, model=model, seed=seed, **row) for row in rows])
                        append_rows(out, frame)
                        print(f"{subject} {task} {band_name} {model} seed={seed}: "
                              + "  ".join(f"{row['arm']}={row['accuracy']:.3f}" for row in rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="auto")
    parser.add_argument("--out-dir", default="runs/second_dataset")
    parser.add_argument("--cache-dir", default="runs/second_dataset/cache")
    parser.add_argument("--subjects", nargs="*", default=None, help="held-out / within subjects to run (default: all)")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--within-tasks", nargs="*", default=list(WITHIN_TASKS), choices=list(WITHIN_TASKS))
    parser.add_argument("--loso-tasks", nargs="*", default=list(LOSO_TASKS), choices=list(LOSO_TASKS))
    parser.add_argument("--bands", nargs="+", default=list(BANDS), choices=list(BANDS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS) + list(CLASSICAL))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        results = pd.concat(pd.read_csv(path) for path in sorted(out_dir.glob("*.csv")) if path.name != "summary.csv")
        summary = summarize(results, SUMMARY_KEYS)
        summary.to_csv(out_dir / "summary.csv", index=False)
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        return

    ensure_same_run_config(out_dir, dict(bands=BANDS, training=DEFAULT_CONFIG, sfreq=SFREQ, max_subjects=args.max_subjects))
    torch.set_num_threads(args.threads)
    data = load_subjects(args.data_root, Path(args.cache_dir), args.max_subjects)
    run(args, data)


if __name__ == "__main__":
    main()
