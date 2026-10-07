"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import json


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8-sig")
    if config_path.suffix.lower() == ".json":
        config = json.loads(text)
    else:
        import yaml

        config = yaml.safe_load(text) or {}

    target = config.get("target", {})
    if target.get("source") in {"file", "points_file"} and target.get("points_path"):
        points_path = Path(target["points_path"])
        if not points_path.is_absolute():
            local_candidate = config_path.parent / points_path
            repository_candidate = config_path.parent.parent / points_path
            points_path = local_candidate if local_candidate.exists() else repository_candidate
        target["points_path"] = str(points_path.resolve())
    return config


def deep_copy_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config)
