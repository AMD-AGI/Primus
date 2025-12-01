import os
import re
from typing import Any, Dict

ENV_PATTERN = re.compile(r"\${([^:{}]+)(?::([^}]*))?}")


def parse_toml(path: str) -> Dict[str, Any]:
    """
    Load TOML configuration with environment variable replacement.

    Notes:
        - This is intentionally minimal and does NOT yet support the
          'extends' preset mechanism that YAML configs use.
        - If you need 'extends' composition, prefer YAML configs for now.
    """
    cfg = _load_toml(path)
    cfg = _resolve_env(cfg)

    # For now, explicitly reject 'extends' in TOML to avoid silently
    # diverging from YAML semantics.
    if isinstance(cfg, dict) and "extends" in cfg:
        raise NotImplementedError(
            "[PrimusConfig] TOML configs do not yet support 'extends'. "
            "Please use YAML if you need preset composition."
        )

    return cfg or {}


def _load_toml(path: str) -> Dict[str, Any]:
    # Python 3.11+ has tomllib in stdlib; otherwise fall back to tomli if installed.
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:  # pragma: no cover - depends on runtime Python
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime Python
            raise ImportError(
                "TOML support requires Python 3.11+ (tomllib) or the 'tomli' package.\n"
                "Install tomli via `pip install tomli`, or use a YAML config instead."
            ) from exc

    with open(path, "rb") as f:
        return tomllib.load(f)


def _resolve_env(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env(v) for v in obj]
    if isinstance(obj, str):
        return _resolve_env_in_string(obj)
    return obj


def _resolve_env_in_string(s: str) -> Any:
    """Replace ${VAR} and ${VAR:default} patterns."""

    def replace_match(m: "re.Match[str]") -> str:
        var, default = m.group(1), m.group(2)

        # ${VAR} → must exist
        if default is None:
            if var not in os.environ:
                raise ValueError(f"Environment variable '{var}' is required but not set.")
            return os.environ[var]

        # ${VAR:default} → use default
        return os.environ.get(var, default)

    replaced = ENV_PATTERN.sub(replace_match, s)
    return _try_numeric(replaced) if replaced != s else replaced


def _try_numeric(v: str) -> Any:
    try:
        if re.fullmatch(r"-?\d+", v):
            return int(v)
        return float(v)
    except ValueError:
        return v
