import os
from pathlib import Path

import yaml

from app.config.schema import AppConfig

_DEFAULT_RELATIVE = Path(__file__).resolve().parent / "default.yaml"


def load_app_config(path: Path | str | None = None) -> AppConfig:
    """Load and validate application config from YAML."""
    if path is None:
        env_path = os.environ.get("APP_CONFIG_PATH")
        resolved = Path(env_path) if env_path else _DEFAULT_RELATIVE
    else:
        resolved = Path(path)
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw)
