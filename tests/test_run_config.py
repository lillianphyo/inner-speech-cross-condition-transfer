"""Results directories refuse to be resumed with different settings."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.run_config import ensure_same_run_config


def test_same_settings_resume_and_changed_settings_are_refused(tmp_path):
    out_dir = tmp_path / "run"
    ensure_same_run_config(out_dir, dict(folds=5, subsets={"all": ["C3", "Cz"]}))
    ensure_same_run_config(out_dir, dict(subsets={"all": ["C3", "Cz"]}, folds=5))
    with pytest.raises(ValueError, match="subsets"):
        ensure_same_run_config(out_dir, dict(folds=5, subsets={"all": ["C3", "Cz"], "llm_x_3": ["C3"]}))
    with pytest.raises(ValueError, match="folds"):
        ensure_same_run_config(out_dir, dict(folds=2, subsets={"all": ["C3", "Cz"]}))


def test_extensible_tables_may_grow_but_shared_entries_must_agree(tmp_path):
    out_dir = tmp_path / "run"
    ensure_same_run_config(out_dir, dict(folds=5, windows={"action": [0.5, 3.0]}), extensible=("windows",))
    ensure_same_run_config(out_dir, dict(folds=5, windows={"action": [0.5, 3.0], "cue": [-0.5, 0.5]}), extensible=("windows",))
    ensure_same_run_config(out_dir, dict(folds=5, windows={"cue": [-0.5, 0.5]}), extensible=("windows",))
    stored = json.loads((out_dir / "run_config.json").read_text())
    assert stored["windows"] == {"action": [0.5, 3.0], "cue": [-0.5, 0.5]}
    with pytest.raises(ValueError, match="windows"):
        ensure_same_run_config(out_dir, dict(folds=5, windows={"action": [0.0, 3.0]}), extensible=("windows",))
    with pytest.raises(ValueError, match="windows"):
        ensure_same_run_config(out_dir, dict(folds=5, windows={"cue": [-0.5, 0.5]}))
