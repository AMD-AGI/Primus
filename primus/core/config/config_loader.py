from pathlib import Path
from typing import Any, Dict

from primus.core.config import toml_loader, yaml_loader


def parse_config(path: str) -> Dict[str, Any]:
    """
    Unified config loader for Primus.

    Supports:
      - YAML (.yaml / .yml) via yaml_loader.parse_yaml
      - TOML (.toml) via toml_loader.parse_toml
    """
    suffix = Path(path).suffix.lower()

    if suffix in {".yaml", ".yml"}:
        return yaml_loader.parse_yaml(path)
    if suffix == ".toml":
        return toml_loader.parse_toml(path)

    raise ValueError(
        f"[PrimusConfig] Unsupported config format for '{path}'. "
        "Supported extensions are: .yaml, .yml, .toml"
    )
