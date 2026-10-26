"""Paper tables: paired effects with t-interval and minimum detectable effect, the appendix
longtable, and byte-for-byte regeneration of the committed tables from runs/."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.inner_speech.paper_tables import (
    appendix_table,
    effect_rows,
    load_runs,
    main,
    paired_effect,
    signed,
)


def test_signed_uses_latex_minus():
    assert signed(-0.94) == "$-$0.9"
    assert signed(0.75) == "+0.8"
    assert signed(0.0) == "+0.0"


def test_paired_effect_matches_t_interval_and_mde():
    diff = np.array([0.01, 0.03, -0.02, 0.04, 0.00, 0.02, 0.05, -0.01, 0.03, 0.02])
    b = pd.Series(0.25, index=[f"sub-{i:02d}" for i in range(1, 11)])
    mean, low, high, mde = paired_effect(b + diff, b)
    d = diff * 100
    se = d.std(ddof=1) / np.sqrt(10)
    assert mean == pytest.approx(d.mean())
    assert high - mean == pytest.approx(stats.t.ppf(0.975, 9) * se)
    assert mean - low == pytest.approx(stats.t.ppf(0.975, 9) * se)
    assert mde == pytest.approx((stats.t.ppf(0.95, 9) + stats.t.ppf(0.80, 9)) * se)


def test_paired_effect_pairs_by_participant_not_position():
    a = pd.Series([0.30, 0.40], index=["sub-02", "sub-01"])
    b = pd.Series([0.20, 0.30, 0.99], index=["sub-01", "sub-02", "sub-03"])
    mean, *_ = paired_effect(a, b)
    assert mean == pytest.approx(((0.40 - 0.20) + (0.30 - 0.30)) / 2 * 100)


def _runs(arms, models, seeds, fractions=(1.0,)):
    rows = []
    for subject in [f"sub-{i:02d}" for i in range(1, 4)]:
        for model in models:
            for arm in arms:
                for fraction in fractions:
                    for seed in seeds:
                        rows.append(dict(subject=subject, window="action", model=model, seed=seed, arm=arm,
                                         train_fraction=fraction, accuracy=0.25 + 0.01 * seed))
    return pd.DataFrame(rows)


def test_effect_rows_marks_models_without_finetune_runs():
    models = ("EEGNet", "ShallowConvNet", "CSP+LDA", "Riemannian")
    within = _runs(["target_only", "transfer", "shuffle_transfer"], models, [0, 1])
    finetune = _runs(["target_only", "finetune"], models[:2], [0], fractions=(0.25, 0.5, 1.0))
    rows = effect_rows(within, finetune)
    assert len(rows) == 6
    assert rows[0].startswith(r"Inner-only $-$ chance & ")
    assert all(row.endswith(r" & -- & -- \\") for row in rows[3:])
    assert r"Fine-tune $-$ inner-only (25\%)" in rows[3]


def test_appendix_table_orders_and_labels_cells():
    summary = pd.DataFrame([
        dict(window=w, model=m, arm=a, train_fraction=f, mean_accuracy=0.3, ci95_low=0.21, ci95_high=0.39,
             p_above_chance=0.0123, p_holm=0.5, subjects_above_chance=7, binomial_threshold=0.4166,
             subjects_above_binomial=1)
        for w in ("cue", "full")
        for m in ("Riemannian", "EEGNet")
        for a, f in (("reciprocal", 1.0), ("target_only", 0.5), ("source_only", 1.0))
    ])
    body = [line for line in appendix_table(summary).splitlines() if line.startswith(("full", "cue"))]
    assert [line.split(" & ")[:3] for line in body[:3]] == [
        ["full", "EEGNet", "overt-only"], ["full", "EEGNet", "inner-only"], ["full", "EEGNet", "reciprocal"]]
    assert body[3].startswith("full & Riemannian")
    assert body[6].startswith("cue & EEGNet")
    assert body[1] == r"full & EEGNet & inner-only & 0.50 & 0.300 [0.21, 0.39] & 0.012 & 0.50 & 7 & 0.417 & 1 \\"
    assert body[0].split(" & ")[3] == ""


def test_committed_tables_regenerate_from_runs(tmp_path):
    main(["--runs", str(ROOT / "runs"), "--out-dir", str(tmp_path)])
    for name in ("supp_within.tex", "effect_rows.tex"):
        assert (tmp_path / name).read_text() == (ROOT / "tables" / name).read_text()


def test_load_runs_pools_every_participant():
    runs = load_runs(ROOT / "runs" / "finetune")
    assert sorted(runs.subject.unique()) == [f"sub-{i:02d}" for i in range(1, 11)]
