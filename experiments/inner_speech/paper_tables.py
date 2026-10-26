"""Rebuild the paired-effect table (tab:effect) and the appendix longtable (Table A1) from runs/.

    python experiments/inner_speech/paper_tables.py --runs runs --out-dir tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CHANCE = 0.25
MODELS = ("EEGNet", "ShallowConvNet", "CSP+LDA", "Riemannian")
WINDOWS = ("full", "action", "cue")
ARM_LABELS = {
    "source_only": "overt-only",
    "shuffle_source": "shuffled overt",
    "target_only": "inner-only",
    "shuffle_target": "shuffled inner",
    "transfer": "transfer",
    "shuffle_transfer": "shuffled transfer",
    "mmd": "transfer + MMD",
    "combined": "combined",
    "reciprocal": "reciprocal",
}
FINETUNE_FRACTIONS = (0.25, 0.5, 1.0)
HEADER_ROW = (r"\textbf{Window} & \textbf{Model} & \textbf{Arm} & \textbf{Frac.} & \textbf{Mean [95\% CI]} & $p$ & "
              r"$\tilde{p}$ & $n_{>0.25}$ & $\theta$ & $n_{>\theta}$ \\")
APPENDIX_HEAD = "\n".join([
    r"\setlength{\tabcolsep}{3.5pt}",
    r"\setlength{\LTcapwidth}{\textwidth}",
    r"\begin{longtable}{llllcccccc}",
    r"\caption{Every cell of the subject-dependent evaluation (Table~\ref{tab:within} shows the action-window subset). "
    r"Mean accuracy over ten participants with 95\% $t$-CI; $p$, one-sided Wilcoxon against 0.250; $\tilde{p}$, "
    r"Holm-corrected over the 126 cells; $n_{>0.25}$, participants above chance; $\theta$, the largest per-participant "
    r"binomial threshold in the cell; $n_{>\theta}$, participants above their own threshold. Calibration fraction shown "
    r"for inner-only only.\label{tab:supp}}\\",
    r"\toprule", HEADER_ROW, r"\midrule", r"\endfirsthead",
    r"\multicolumn{10}{l}{\emph{Table A1, continued.}}\\",
    r"\toprule", HEADER_ROW, r"\midrule", r"\endhead",
    r"\bottomrule", r"\endlastfoot",
])


def load_runs(run_dir: Path) -> pd.DataFrame:
    return pd.concat([pd.read_csv(path) for path in sorted(Path(run_dir).glob("sub-*.csv"))], ignore_index=True)


def per_participant(runs: pd.DataFrame, model: str, arm: str, fraction: float | None = None) -> pd.Series:
    """Action-window accuracy of one model and arm, mean over seeds, indexed by participant."""
    cell = runs[(runs.window == "action") & (runs.model == model) & (runs.arm == arm)]
    if fraction is not None:
        cell = cell[cell.train_fraction == fraction]
    return cell.groupby("subject").accuracy.mean()


def paired_effect(a: pd.Series, b: pd.Series) -> tuple[float, float, float, float]:
    """Mean a - b in percentage points over shared participants, its 95% t-interval and the one-sided
    alpha 0.05 / power 0.80 minimum detectable effect (manuscript eq:mde)."""
    diff = (a - b).dropna() * 100
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    mde = (stats.t.ppf(0.95, n - 1) + stats.t.ppf(0.80, n - 1)) * se
    return diff.mean(), diff.mean() - half, diff.mean() + half, mde


def signed(value: float) -> str:
    text = f"{value:+.1f}"
    return "$-$" + text[1:] if text.startswith("-") else text


def effect_cell(a: pd.Series, b: pd.Series) -> str:
    mean, low, high, mde = paired_effect(a, b)
    return f"{signed(mean)} [{signed(low)}, {signed(high)}]; {mde:.1f}"


def effect_rows(within: pd.DataFrame, finetune: pd.DataFrame) -> list[str]:
    rows = {r"Inner-only $-$ chance": [], r"Transfer $-$ inner-only": [], r"Transfer $-$ shuffled transfer": []}
    for model in MODELS:
        inner = per_participant(within, model, "target_only", 1.0)
        transfer = per_participant(within, model, "transfer")
        rows[r"Inner-only $-$ chance"].append(effect_cell(inner, inner * 0 + CHANCE))
        rows[r"Transfer $-$ inner-only"].append(effect_cell(transfer, inner))
        rows[r"Transfer $-$ shuffled transfer"].append(effect_cell(transfer, per_participant(within, model, "shuffle_transfer")))
    for fraction in FINETUNE_FRACTIONS:
        label = rf"Fine-tune $-$ inner-only ({fraction:.0%})".replace("%", r"\%")
        rows[label] = [
            effect_cell(per_participant(finetune, model, "finetune", fraction),
                        per_participant(finetune, model, "target_only", fraction))
            if model in set(finetune.model) else "--"
            for model in MODELS
        ]
    return [" & ".join([label, *cells]) + r" \\" for label, cells in rows.items()]


def appendix_table(summary: pd.DataFrame) -> str:
    order = summary.assign(
        _w=summary.window.map(WINDOWS.index),
        _m=summary.model.map(MODELS.index),
        _a=summary.arm.map(list(ARM_LABELS).index),
    ).sort_values(["_w", "_m", "_a", "train_fraction"])
    lines = [APPENDIX_HEAD]
    for r in order.itertuples():
        fraction = f"{r.train_fraction:.2f}" if r.arm == "target_only" else ""
        lines.append(
            f"{r.window} & {r.model} & {ARM_LABELS[r.arm]} & {fraction} & "
            f"{r.mean_accuracy:.3f} [{r.ci95_low:.2f}, {r.ci95_high:.2f}] & {r.p_above_chance:.3f} & {r.p_holm:.2f} & "
            f"{r.subjects_above_chance} & {r.binomial_threshold:.3f} & {r.subjects_above_binomial} \\\\"
        )
    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out-dir", default="tables")
    args = parser.parse_args(argv)
    runs, out = Path(args.runs), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "supp_within.tex").write_text(appendix_table(pd.read_csv(runs / "honest_transfer" / "summary.csv")))
    rows = effect_rows(load_runs(runs / "honest_transfer"), load_runs(runs / "finetune"))
    (out / "effect_rows.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
