"""Named channel sets and montage helpers for the channel-subset and second-dataset arms.

`left_primary_strip` covers left primary motor and somatosensory cortex under the C row;
`left_sensorimotor` adds the FC and CP neighbours that scalp EEG cannot separate from it.
10-10 to cortex mapping after Homan 1987 and Koessler 2009.
"""
from __future__ import annotations

import re

import numpy as np

NON_SCALP = ("EOG", "M1", "M2", "BIP1", "BIP2")

KNOWLEDGE_SETS = {
    "left_primary_strip": ["C5", "C3", "C1"],
    "left_sensorimotor": ["FC5", "FC3", "FC1", "C5", "C3", "C1", "CP5", "CP3", "CP1"],
}
MIRROR_NAMES = {
    "left_primary_strip": "right_primary_strip",
    "left_sensorimotor": "right_sensorimotor",
}


def align_channels(recordings: list[tuple[np.ndarray, list[str]]]) -> tuple[np.ndarray, list[str]]:
    """Pool epochs from recordings with different montages, keeping the channels present in all of them.

    Channels are matched by name and returned in the first recording's order.
    """
    common = [name for name in recordings[0][1] if all(name in names for _, names in recordings)]
    pooled = np.concatenate([X[:, [names.index(name) for name in common]] for X, names in recordings])
    return pooled, common


def scalp_channels(ch_names: list[str]) -> list[str]:
    return [name for name in ch_names if name not in NON_SCALP]


def mirror(names: list[str]) -> list[str]:
    """Other-hemisphere homologues: odd electrode numbers become the next even number."""
    def flip(name: str) -> str:
        match = re.fullmatch(r"([A-Za-z]+)(\d+)", name)
        if not match:
            return name
        number = int(match.group(2))
        return f"{match.group(1)}{number + 1 if number % 2 else number - 1}"

    return [flip(name) for name in names]


def random_subsets(ch_names: list[str], size: int, n_draws: int, rng: np.random.Generator) -> list[list[str]]:
    scalp = scalp_channels(ch_names)
    draws: list[list[str]] = []
    while len(draws) < n_draws:
        draw = sorted(rng.choice(scalp, size=size, replace=False).tolist(), key=scalp.index)
        if draw not in draws:
            draws.append(draw)
    return draws
