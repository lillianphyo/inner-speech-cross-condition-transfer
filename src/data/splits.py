"""Stratified, disjoint index splits."""
from __future__ import annotations

import numpy as np


def split_indices(y: np.ndarray, fractions: tuple[float, ...], rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Stratified, disjoint index splits; the last split takes the remainder of each class."""
    parts = [[] for _ in fractions]
    for label in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == label))
        bounds = np.round(np.cumsum(fractions[:-1]) * len(members)).astype(int)
        for part, chunk in zip(parts, np.split(members, bounds)):
            part.extend(chunk.tolist())
    return tuple(np.array(sorted(part)) for part in parts)
