# utils/config_loader.py
# -*- coding: utf-8 -*-

from pathlib import Path

import yaml


def get_project_root():
    """Return the tic_tac_toe project root directory."""
    return Path(__file__).resolve().parents[1]


def resolve_project_path(relative_path):
    """Resolve a path relative to the tic_tac_toe project root."""
    return get_project_root() / relative_path


def load_config(path="config/config.yaml"):
    """Load a YAML config file relative to the project root."""
    config_path = resolve_project_path(path)

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
