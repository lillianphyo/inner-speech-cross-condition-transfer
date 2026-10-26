"""Position-based channel conditions for the ds003626 BioSemi 128 cap."""
import sys
from pathlib import Path

import mne
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.inner_speech.overt_channels import channel_conditions, nearest_biosemi

BIOSEMI = mne.channels.make_standard_montage("biosemi128")
CH_NAMES = BIOSEMI.ch_names
POSITIONS = BIOSEMI.get_positions()["ch_pos"]


def _x(names):
    return np.array([POSITIONS[name][0] for name in names])


def test_left_labels_map_to_distinct_left_hemisphere_electrodes():
    chosen = nearest_biosemi(["C5", "C3", "C1"], CH_NAMES)
    assert len(set(chosen)) == 3
    assert (_x(chosen) < 0).all()
    assert nearest_biosemi(["Cz"], CH_NAMES) == ["A1"]


def test_far_label_is_refused():
    with pytest.raises(ValueError, match="within"):
        nearest_biosemi(["C3"], CH_NAMES, max_distance=0.001)


def test_two_labels_sharing_an_electrode_are_refused():
    with pytest.raises(ValueError, match="already taken"):
        nearest_biosemi(["C3", "C3"], CH_NAMES)


def test_channel_conditions_have_the_expected_sizes_and_sides():
    conditions = channel_conditions(CH_NAMES, n_random=2, rng=np.random.default_rng(0))
    assert len(conditions["all"]) == 128
    assert len(conditions["left_primary_strip"]) == len(conditions["right_primary_strip"]) == 3
    assert len(conditions["left_sensorimotor"]) == len(conditions["right_sensorimotor"]) == 9
    assert len(conditions["peripheral"]) == 16
    assert len(set(conditions["peripheral"])) == 16
    assert (_x(conditions["left_sensorimotor"]) < 0).all()
    assert (_x(conditions["right_sensorimotor"]) > 0).all()
    assert not set(conditions["left_primary_strip"]) & set(conditions["right_primary_strip"])
    assert {name for name in conditions if name.startswith("random_")} == {
        "random_3_0", "random_3_1", "random_9_0", "random_9_1",
    }
    assert len(conditions["random_9_0"]) == 9
    assert all(set(members) <= set(CH_NAMES) for members in conditions.values())
