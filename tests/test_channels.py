"""Named channel sets and montage helpers (ds006104 ANT 64-channel montage)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.inner_speech.channels import KNOWLEDGE_SETS, align_channels, mirror, random_subsets, scalp_channels

ANT_64 = [
    "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "M1", "T7", "C3", "Cz",
    "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "POz", "O1", "O2", "EOG",
    "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6", "FC3", "FCz", "FC4", "C5", "C1", "C2", "C6", "CP3",
    "CP4", "P5", "P1", "P2", "P6", "PO5", "PO3", "PO4", "PO6", "FT7", "FT8", "TP7", "TP8", "PO7", "PO8",
    "Oz", "BIP1", "BIP2",
]


def test_scalp_channels_drop_eye_mastoid_and_bipolar():
    scalp = scalp_channels(ANT_64)
    assert len(scalp) == 61
    assert not {"EOG", "M1", "M2", "BIP1", "BIP2"} & set(scalp)


def test_knowledge_sets_exist_in_the_montage_and_have_planned_sizes():
    assert {name: len(members) for name, members in KNOWLEDGE_SETS.items()} == {
        "left_primary_strip": 3, "left_sensorimotor": 9,
    }
    for members in KNOWLEDGE_SETS.values():
        assert set(members) <= set(scalp_channels(ANT_64))


def test_mirror_swaps_hemisphere_and_keeps_midline():
    assert mirror(["FC5", "C3", "CP1", "F7", "Cz", "FT7"]) == ["FC6", "C4", "CP2", "F8", "Cz", "FT8"]
    assert set(mirror(KNOWLEDGE_SETS["left_sensorimotor"])) <= set(ANT_64)


def test_random_subsets_are_scalp_only_distinct_and_reproducible():
    draws = random_subsets(ANT_64, size=9, n_draws=10, rng=np.random.default_rng(0))
    again = random_subsets(ANT_64, size=9, n_draws=10, rng=np.random.default_rng(0))
    assert draws == again
    assert len({tuple(draw) for draw in draws}) == 10
    scalp = set(scalp_channels(ANT_64))
    assert all(len(set(draw)) == 9 and set(draw) <= scalp for draw in draws)


def test_align_channels_matches_by_name_across_montages_with_different_order_and_size():
    first = (np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4), ["Fz", "C3", "Cz"])
    second_names = ["X1", "Cz", "C3", "Fz", "X2"]
    second = (np.stack([np.full((5, 4), 100.0, dtype=np.float32) + np.arange(5)[:, None]]), second_names)

    pooled, common = align_channels([first, second])

    assert common == ["Fz", "C3", "Cz"]
    assert pooled.shape == (3, 3, 4)
    assert np.array_equal(pooled[:2], first[0])
    assert pooled[2, :, 0].tolist() == [103.0, 102.0, 101.0]


def test_align_channels_drops_a_channel_missing_from_any_recording():
    first = (np.zeros((1, 3, 2), dtype=np.float32), ["Fz", "C3", "Cz"])
    second = (np.ones((1, 2, 2), dtype=np.float32), ["Cz", "Fz"])
    pooled, common = align_channels([first, second])
    assert common == ["Fz", "Cz"]
    assert pooled.shape == (2, 2, 2)
