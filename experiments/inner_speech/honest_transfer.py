"""Within-subject cross-condition transfer on ds003626 with a real train/test separation.

Every arm that reports inner-speech accuracy scores the
same held-out inner-speech trials, model selection uses a validation split that is never the
test split, and the pronounced and inner sets are checked to share no trial.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.inner_speech.classical import CLASSICAL, classical_fit_and_score
from experiments.inner_speech.mmd import mmd_regularizer
from experiments.inner_speech.exp02_cross_dataset import evaluate_model, get_model, train_model
from src.data.leakage import assert_disjoint
from src.data.splits import split_indices
from src.data.transforms import shuffle_labels, standardize

N_CLASSES = 4
CHANCE = 1.0 / N_CLASSES
DEFAULT_CONFIG = dict(batch_size=32, num_epochs=100, learning_rate=1e-3, weight_decay=1e-4, patience=15)
ARMS = (
    "target_only", "reciprocal", "transfer", "combined", "shuffle_target", "shuffle_transfer",
    "source_only", "shuffle_source", "mmd", "finetune", "shuffle_finetune",
)
NETWORK_ONLY = ("mmd", "finetune", "shuffle_finetune")
MMD_WEIGHT = 0.1


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def fit(build_model, train, val, others, *, config, device, seed, regularizer=None, adapt_to=None):
    """Train `build_model(n_channels, n_times)` on z-scored `train`, select on `val`.

    `regularizer(model)` is added to every training loss; `adapt_to` (unlabelled trials) adds an
    MMD penalty between its features and the training features. Returns the model, the z-scored
    training array and the arrays in `others` on the same scale.
    """
    torch.manual_seed(seed)
    X_train, X_val, *X_others = standardize(train[0], val[0], *others, *([adapt_to] if adapt_to is not None else []))
    if adapt_to is not None:
        mmd = mmd_regularizer(
            X_train, X_others.pop(), weight=MMD_WEIGHT, batch_size=config["batch_size"], device=device,
            rng=np.random.default_rng(seed),
        )
        base = regularizer
        regularizer = mmd if base is None else (lambda model: base(model) + mmd(model))
    model = train_model(
        build_model(X_train.shape[1], X_train.shape[2]),
        _loader(X_train, train[1], config["batch_size"], shuffle=True),
        _loader(X_val, val[1], config["batch_size"], shuffle=False),
        torch.device(device), config, None, regularizer,
    )
    return model, X_train, X_others


def fit_and_score(
    model_name, train, val, tests, *, config, device, seed, n_classes=N_CLASSES, adapt_to=None,
) -> dict[str, dict]:
    """Train on `train`, select on `val`, score every set in `tests` (name -> (X, y))."""
    if model_name in CLASSICAL:
        if adapt_to is not None:
            raise ValueError(f"the mmd arm needs a network, not {model_name}")
        return classical_fit_and_score(model_name, train, val, tests, seed=seed)
    model, _, X_tests = fit(
        lambda n_channels, n_times: get_model(model_name, n_channels, n_times, n_classes),
        train, val, [X for X, _ in tests.values()], config=config, device=device, seed=seed, adapt_to=adapt_to,
    )
    return _score(model, tests, X_tests, config=config, device=device)


def _score(model, tests, X_tests, *, config, device) -> dict[str, dict]:
    return {
        name: evaluate_model(model, _loader(X_test, y_test, config["batch_size"], shuffle=False), torch.device(device))
        for (name, (_, y_test)), X_test in zip(tests.items(), X_tests)
    }


def pretrain(model_name, source_train, source_val, *, config, device, seed, n_classes=N_CLASSES):
    """Train a network on the source condition and select it on the source validation set."""
    if model_name in CLASSICAL:
        raise ValueError(f"the finetune arms need a network, not {model_name}")
    model, _, _ = fit(
        lambda n_channels, n_times: get_model(model_name, n_channels, n_times, n_classes),
        source_train, source_val, [], config=config, device=device, seed=seed,
    )
    return model


def finetune_and_score(pretrained, target_train, target_val, tests, *, config, device, seed) -> dict[str, dict]:
    """Continue training a copy of `pretrained` on `target_train`, select on `target_val`, score `tests`."""
    model, _, X_tests = fit(
        lambda n_channels, n_times: copy.deepcopy(pretrained),
        target_train, target_val, [X for X, _ in tests.values()], config=config, device=device, seed=seed,
    )
    return _score(model, tests, X_tests, config=config, device=device)


def run_subject_arms(
    X_pron: np.ndarray,
    y_pron: np.ndarray,
    X_inner: np.ndarray,
    y_inner: np.ndarray,
    *,
    model_name: str,
    seed: int,
    config: dict = DEFAULT_CONFIG,
    device: str = "cpu",
    train_fractions: tuple[float, ...] = (1.0,),
    arms: tuple[str, ...] = ARMS,
    held_out_session: str | None = None,
    sessions: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[dict]:
    """Score the requested `arms` for one subject.

    `source_only` (pronounced -> held-out pronounced) uses its own 60/20/20 split of the
    pronounced trials, drawn from a separate generator so the other arms' splits never move.
    `mmd` is `transfer` plus an MMD penalty towards the unlabelled inner-speech training trials.
    `finetune` pretrains on the pronounced training set, then continues training on the same
    inner-speech subsets as `target_only`, selecting on the inner validation set;
    `shuffle_finetune` pretrains on permuted pronounced labels.
    With `held_out_session` and `sessions = (sessions_pron, sessions_inner)`, every test set is the
    held-out session of its condition and training/validation come from the other sessions (80/20).
    """
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arm(s): {sorted(unknown)}")
    assert_disjoint(X_pron, X_inner)
    rng = np.random.default_rng(seed)
    if held_out_session is None:
        train_i, val_i, test_i = split_indices(y_inner, (0.6, 0.2, 0.2), rng)
        train_p, val_p = split_indices(y_pron, (0.8, 0.2), rng)
        pron_test = np.arange(len(y_pron))
    else:
        sessions_pron, sessions_inner = sessions
        train_i, val_i, test_i = session_split(y_inner, sessions_inner, held_out_session, rng)
        train_p, val_p, pron_test = session_split(y_pron, sessions_pron, held_out_session, rng)

    inner_train, inner_val = (X_inner[train_i], y_inner[train_i]), (X_inner[val_i], y_inner[val_i])
    inner_test = {"inner": (X_inner[test_i], y_inner[test_i])}
    pron_train, pron_val = (X_pron[train_p], y_pron[train_p]), (X_pron[val_p], y_pron[val_p])
    fit = dict(config=config, device=device, seed=seed)

    def shuffled(split):
        return split[0], shuffle_labels(split[1], rng)

    def row(arm, metrics, n_train, n_test, train_fraction=1.0):
        fields = dict(arm=arm, train_fraction=train_fraction, n_train=n_train, n_test=n_test, chance=CHANCE, **metrics)
        return fields if held_out_session is None else dict(session=held_out_session, **fields)

    rows = []
    subsets = {}
    for fraction in train_fractions:
        keep, _ = split_indices(inner_train[1], (fraction, 1.0 - fraction), rng)
        subsets[fraction] = (inner_train[0][keep], inner_train[1][keep])
    if {"target_only", "reciprocal"} & set(arms):
        for fraction, subset in subsets.items():
            score_pronounced = fraction == 1.0 and "reciprocal" in arms
            tests = dict(inner_test, pronounced=(X_pron[pron_test], y_pron[pron_test])) if score_pronounced else inner_test
            scores = fit_and_score(model_name, subset, inner_val, tests, **fit)
            if "target_only" in arms:
                rows.append(row("target_only", scores["inner"], len(subset[1]), len(test_i), fraction))
            if score_pronounced:
                rows.append(row("reciprocal", scores["pronounced"], len(subset[1]), len(pron_test)))

    combined = (np.concatenate([pron_train[0], inner_train[0]]), np.concatenate([pron_train[1], inner_train[1]]))
    inner_arms = [
        ("transfer", pron_train, pron_val),
        ("combined", combined, inner_val),
        ("shuffle_target", shuffled(inner_train), shuffled(inner_val)),
        ("shuffle_transfer", shuffled(pron_train), shuffled(pron_val)),
    ]
    for arm, train, val in inner_arms:
        if arm in arms:
            scores = fit_and_score(model_name, train, val, inner_test, **fit)
            rows.append(row(arm, scores["inner"], len(train[1]), len(test_i)))
    if "mmd" in arms:
        scores = fit_and_score(model_name, pron_train, pron_val, inner_test, adapt_to=inner_train[0], **fit)
        rows.append(dict(row("mmd", scores["inner"], len(pron_train[1]), len(test_i)), mmd_weight=MMD_WEIGHT))

    if {"source_only", "shuffle_source"} & set(arms):
        rng_source = np.random.default_rng([seed, 1])
        if held_out_session is None:
            train_s, val_s, test_s = split_indices(y_pron, (0.6, 0.2, 0.2), rng_source)
        else:
            train_s, val_s, test_s = session_split(y_pron, sessions_pron, held_out_session, rng_source)
        source_train, source_val = (X_pron[train_s], y_pron[train_s]), (X_pron[val_s], y_pron[val_s])
        source_test = {"pronounced": (X_pron[test_s], y_pron[test_s])}
        for arm, train, val in [
            ("source_only", source_train, source_val),
            ("shuffle_source", (source_train[0], shuffle_labels(source_train[1], rng_source)),
             (source_val[0], shuffle_labels(source_val[1], rng_source))),
        ]:
            if arm in arms:
                scores = fit_and_score(model_name, train, val, source_test, **fit)
                rows.append(row(arm, scores["pronounced"], len(train_s), len(test_s)))

    if {"finetune", "shuffle_finetune"} & set(arms):
        rng_finetune = np.random.default_rng([seed, 2])
        sources = {
            "finetune": (pron_train, pron_val),
            "shuffle_finetune": (
                (pron_train[0], shuffle_labels(pron_train[1], rng_finetune)),
                (pron_val[0], shuffle_labels(pron_val[1], rng_finetune)),
            ),
        }
        for arm, (source_train, source_val) in sources.items():
            if arm not in arms:
                continue
            pretrained = pretrain(model_name, source_train, source_val, **fit)
            for fraction, subset in subsets.items():
                scores = finetune_and_score(pretrained, subset, inner_val, inner_test, **fit)
                rows.append(row(arm, scores["inner"], len(subset[1]), len(test_i), fraction))
    return rows


def session_split(y: np.ndarray, sessions: np.ndarray, held_out: str, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(train, val, test) indices: test = every trial of `held_out`, train/val = a stratified 80/20 split of the rest."""
    test = np.flatnonzero(sessions == held_out)
    rest = np.flatnonzero(sessions != held_out)
    if len(test) == 0 or len(rest) == 0:
        raise ValueError(f"session {held_out!r} must hold some trials and leave some for training")
    train_rel, val_rel = split_indices(y[rest], (0.8, 0.2), rng)
    return rest[train_rel], rest[val_rel], test


SUMMARY_KEYS = ("window", "model", "arm", "train_fraction")


def summarize(results: pd.DataFrame, keys: tuple[str, ...] = SUMMARY_KEYS) -> pd.DataFrame:
    """Across-subject mean accuracy, 95 % t-interval and one-sided Wilcoxon test against chance.

    One cell per distinct `keys` value. `p_holm` corrects `p_above_chance` over every cell of the
    table; `min_seeds` is the smallest number of seeds any subject contributes to the cell.
    `binomial_threshold` is the per-subject 95 % binomial chance threshold (Combrisson & Jerbi 2015)
    for the smallest test set in the cell; `subjects_above_binomial` counts subjects whose mean
    accuracy exceeds the threshold for their own test size.
    """
    keys = list(keys)
    per_subject = results.groupby(keys + ["subject"], as_index=False).agg(
        accuracy=("accuracy", "mean"), chance=("chance", "mean"), seeds=("seed", "nunique"), n_test=("n_test", "min"),
    )
    per_subject["threshold"] = stats.binom.ppf(0.95, per_subject["n_test"], per_subject["chance"]) / per_subject["n_test"]

    def describe(group: pd.DataFrame) -> dict:
        accuracy, chance = group["accuracy"].to_numpy(), group["chance"].iloc[0]
        half_width = stats.t.ppf(0.975, len(accuracy) - 1) * stats.sem(accuracy) if len(accuracy) > 1 else np.nan
        differs = len(accuracy) > 1 and np.any(accuracy != chance)
        p_value = stats.wilcoxon(accuracy - chance, alternative="greater").pvalue if differs else np.nan
        return dict(
            n_subjects=len(accuracy), min_seeds=int(group["seeds"].min()), mean_accuracy=accuracy.mean(), chance=chance,
            ci95_low=accuracy.mean() - half_width, ci95_high=accuracy.mean() + half_width,
            p_above_chance=p_value, subjects_above_chance=int((accuracy > chance).sum()),
            binomial_threshold=group["threshold"].max(),
            subjects_above_binomial=int((group["accuracy"] > group["threshold"]).sum()),
        )

    summary = pd.DataFrame([dict(zip(keys, key), **describe(group)) for key, group in per_subject.groupby(keys)])
    tested = summary["p_above_chance"].notna()
    order = summary.loc[tested, "p_above_chance"].sort_values()
    scaled = order * np.arange(len(order), 0, -1)
    summary.loc[order.index, "p_holm"] = scaled.cummax().clip(upper=1.0)
    return summary
