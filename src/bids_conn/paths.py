"""Load data locations from config/paths.yaml."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO_ROOT / "config" / "paths.yaml"


def get_paths(config_file: Path = CONFIG_FILE) -> dict[str, Path]:
    """Return a dict of named paths; relative entries are resolved from the project root."""
    if not config_file.exists():
        raise FileNotFoundError(
            f"{config_file} not found. Copy config/paths.example.yaml to config/paths.yaml and edit it."
        )
    with open(config_file) as f:
        entries = yaml.safe_load(f) or {}

    paths = {}
    for name, value in entries.items():
        p = Path(value).expanduser()
        paths[name] = p if p.is_absolute() else REPO_ROOT / p
    return paths
