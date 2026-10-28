"""Maximum mean discrepancy penalty between network features of two unlabelled trial sets.

Re-check of the earlier domain-adaptation claim on the fixed loader: an RBF-kernel
MMD (median-heuristic bandwidth) between the features of a random source batch and a random
target batch is added to the classification loss at every step, as `train_model`'s regularizer.
"""
from __future__ import annotations

import numpy as np
import torch


def mmd_rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Biased MMD^2 estimate between feature batches `a` and `b` with a Gaussian kernel."""
    both = torch.cat([a, b])
    distances = torch.cdist(both, both) ** 2
    bandwidth = distances.detach()[distances.detach() > 0].median().clamp(min=1e-6)
    kernel = torch.exp(-distances / bandwidth)
    n = len(a)
    return kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2 * kernel[:n, n:].mean()


def mmd_regularizer(X_source: np.ndarray, X_target: np.ndarray, *, weight: float, batch_size: int, device, rng):
    """Return `regularizer(model)` = `weight` x MMD between `model.features` of fresh random batches."""
    source = torch.from_numpy(X_source).to(device)
    target = torch.from_numpy(X_target).to(device)

    def regularizer(model):
        picks_s = torch.from_numpy(rng.choice(len(source), min(batch_size, len(source)), replace=False)).to(device)
        picks_t = torch.from_numpy(rng.choice(len(target), min(batch_size, len(target)), replace=False)).to(device)
        return weight * mmd_rbf(model.features(source[picks_s]), model.features(target[picks_t]))

    return regularizer
