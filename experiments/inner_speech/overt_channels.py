"""Channel-subset conditions for the overt-vs-inner check on ds003626 (BioSemi 128).

ds003626 names its electrodes A1–D32 and the derivatives carry no montage, so every 10-10
label of the knowledge sets is mapped to the nearest BioSemi 128 electrode of MNE's standard
montages. The peripheral ring (frontal, temporal and occipital rim) is where articulation
EMG enters scalp EEG: an overt-speech score there that matches the central strip points to
muscle rather than cortex.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.channel_selection.channels import KNOWLEDGE_SETS, MIRROR_NAMES, mirror, random_subsets

MAX_DISTANCE_M = 0.025
SETS = ("left_primary_strip", "left_sensorimotor")
PERIPHERAL_1010 = [
    "Fp1", "Fp2", "AF7", "AF8", "F7", "F8", "FT7", "FT8", "T7", "T8",
    "TP7", "TP8", "P7", "P8", "O1", "O2",
]


def nearest_biosemi(labels: list[str], ch_names: list[str], max_distance: float = MAX_DISTANCE_M) -> list[str]:
    """Map 10-10 `labels` to distinct BioSemi 128 electrodes among `ch_names`, by 3-D distance."""
    biosemi = mne.channels.make_standard_montage("biosemi128").get_positions()["ch_pos"]
    standard = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    names = [name for name in ch_names if name in biosemi]
    positions = np.array([biosemi[name] for name in names])
    chosen: list[str] = []
    for label in labels:
        distances = np.linalg.norm(positions - standard[label], axis=1)
        index = int(distances.argmin())
        if distances[index] > max_distance:
            raise ValueError(f"No BioSemi electrode within {max_distance * 100:.1f} cm of {label}.")
        if names[index] in chosen:
            raise ValueError(f"{label} maps to {names[index]}, already taken by an earlier label.")
        chosen.append(names[index])
    return chosen


def channel_conditions(ch_names: list[str], n_random: int, rng: np.random.Generator) -> dict[str, list[str]]:
    """All channels, the two knowledge sets with their right-hemisphere mirrors, the peripheral ring,
    and `n_random` size-matched random draws for each knowledge-set size."""
    conditions = {"all": list(ch_names)}
    for name in SETS:
        conditions[name] = nearest_biosemi(KNOWLEDGE_SETS[name], ch_names)
        conditions[MIRROR_NAMES[name]] = nearest_biosemi(mirror(KNOWLEDGE_SETS[name]), ch_names)
    conditions["peripheral"] = nearest_biosemi(PERIPHERAL_1010, ch_names)
    for size in sorted({len(KNOWLEDGE_SETS[name]) for name in SETS}):
        for draw_index, draw in enumerate(random_subsets(ch_names, size, n_random, rng)):
            conditions[f"random_{size}_{draw_index}"] = draw
    return conditions
