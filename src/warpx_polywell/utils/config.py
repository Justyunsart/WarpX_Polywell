"""
Loads project settings from a .env file at the project root.
Call load_config() once at startup; access values via get_config().
"""
import os
from pathlib import Path

_config: dict = {}

def load_config() -> dict:
    """
    Reads the .env file from the project root (two levels above src/utils/).
    Populates the module-level config dict and returns it.
    Missing keys fall back to defaults.
    """
    global _config

    root = Path(__file__).resolve().parent.parent.parent
    env_file = root / ".env"

    if env_file.is_file():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                _config[key.strip()] = value.strip()

    # Apply defaults for missing keys
    _config.setdefault("STORAGE_BACKEND", "local")
    _config.setdefault("LOCAL_OUTPUT_DIR", str(root / "output"))
    _config.setdefault("GOOGLE_DRIVE_FOLDER_ID", "")

    return _config


def get_config() -> dict:
    """Returns the current config, loading it first if not yet loaded."""
    if not _config:
        load_config()
    return _config
