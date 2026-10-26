"""External-electrode (EXG) signals of ds003626 for the artefact arms: the eight EXG channels alone,
the lip pair alone, and the scalp channels with the EXG regressed out trial by trial.
"""
from __future__ import annotations

import numpy as np

LIP_CHANNELS = ("EXG7", "EXG8")
SIGNALS = ("exg", "lips", "scalp_regressed")


def regress_out(X_scalp: np.ndarray, X_exg: np.ndarray) -> np.ndarray:
    """Residual of every scalp channel after a least-squares fit on the EXG channels (plus intercept)
    of the same trial. No other trial enters the fit, so no split can leak through it."""
    if X_scalp.shape[0] != X_exg.shape[0] or X_scalp.shape[-1] != X_exg.shape[-1]:
        raise ValueError(f"scalp {X_scalp.shape} and EXG {X_exg.shape} trials do not line up")
    residual = np.empty_like(X_scalp)
    ones = np.ones((1, X_exg.shape[-1]))
    for i in range(len(X_scalp)):
        regressors = np.concatenate([X_exg[i], ones]).T.astype(np.float64)
        target = X_scalp[i].T.astype(np.float64)
        beta, *_ = np.linalg.lstsq(regressors, target, rcond=None)
        residual[i] = (target - regressors @ beta).T
    return residual


def make_signal(name: str, X_scalp: np.ndarray, X_exg: np.ndarray, exg_names: list[str]) -> np.ndarray:
    if name == "exg":
        return X_exg
    if name == "lips":
        return X_exg[:, [exg_names.index(channel) for channel in LIP_CHANNELS]]
    if name == "scalp_regressed":
        return regress_out(X_scalp, X_exg)
    raise ValueError(f"unknown signal {name!r}; choose from {SIGNALS}")
