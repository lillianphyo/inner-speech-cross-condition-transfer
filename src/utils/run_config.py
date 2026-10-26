"""Guard against resuming a results directory with different settings."""
from __future__ import annotations

import json
from pathlib import Path


def _differs(stored, current, extensible: bool) -> bool:
    if extensible and isinstance(stored, dict) and isinstance(current, dict):
        return any(stored[name] != current[name] for name in stored.keys() & current.keys())
    return stored != current


def ensure_same_run_config(out_dir: Path, config: dict, extensible: tuple[str, ...] = ()) -> None:
    """Record the settings on first use; raise if `out_dir` already holds results made with other settings.

    Keys in `extensible` hold name-to-definition tables that may gain entries between runs: shared
    entries must agree, and the stored table grows to the union.
    """
    current = json.loads(json.dumps(config, sort_keys=True))
    path = out_dir / "run_config.json"
    if not path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2, sort_keys=True))
        return
    stored = json.loads(path.read_text())
    changed = sorted(
        key for key in stored.keys() | current.keys()
        if _differs(stored.get(key), current.get(key), key in extensible)
    )
    if changed:
        raise ValueError(
            f"{out_dir} holds results made with different settings ({', '.join(changed)}). "
            "Resuming would mix them or skip work; use a new output directory."
        )
    grown = {key: {**stored[key], **current[key]} for key in extensible if key in stored and key in current}
    if any(grown[key] != stored[key] for key in grown):
        path.write_text(json.dumps({**stored, **grown}, indent=2, sort_keys=True))
