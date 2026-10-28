"""Classical baselines for the ds003626 arms: CSP+LDA and Riemannian tangent-space logistic
regression, fit on the same z-scored trials as the networks and selected on the validation split.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.inner_speech.exp02_cross_dataset import compute_metrics
from src.data.transforms import standardize

CLASSICAL = ("CSP+LDA", "Riemannian")
GRID = {"CSP+LDA": (4, 8, 16), "Riemannian": (0.1, 1.0, 10.0)}


def build_classical(name: str, setting, seed: int):
    from pyriemann.estimation import Covariances
    from pyriemann.spatialfilters import CSP
    from pyriemann.tangentspace import TangentSpace

    if name == "CSP+LDA":
        return make_pipeline(Covariances("oas"), CSP(nfilter=setting, log=True),
                             LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    if name == "Riemannian":
        return make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"),
                             LogisticRegression(C=setting, max_iter=2000, random_state=seed))
    raise ValueError(f"Unknown classical model: {name}")


def classical_fit_and_score(name: str, train, val, tests: dict, *, seed: int) -> dict[str, dict]:
    """Fit every grid setting on `train`, keep the one that scores best on `val`, score every set in `tests`."""
    if name not in GRID:
        raise ValueError(f"Unknown classical model: {name}")
    X_train, X_val, *X_tests = standardize(train[0], val[0], *[X for X, _ in tests.values()])
    X_train, X_val = X_train.astype(np.float64), X_val.astype(np.float64)
    fitted = {setting: build_classical(name, setting, seed).fit(X_train, train[1]) for setting in GRID[name]}
    best = max(fitted, key=lambda setting: np.mean(fitted[setting].predict(X_val) == val[1]))
    model = fitted[best]
    return {
        key: dict(compute_metrics(np.asarray(y_test), model.predict(X_test.astype(np.float64))), setting=best)
        for (key, (_, y_test)), X_test in zip(tests.items(), X_tests)
    }
