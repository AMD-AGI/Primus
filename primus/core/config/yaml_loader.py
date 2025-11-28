import os
import re

import yaml

from primus.core.config.merge_utils import deep_merge

ENV_PATTERN = re.compile(r"\${([^:{}]+)(?::([^}]*))?}")


def parse_yaml(path: str) -> dict:
    """Load YAML with env replacement and extends merging."""
    cfg = _load_yaml(path)
    if cfg is None:
        raise ValueError(f"YAML configuration file '{path}' is empty or invalid.")
    cfg = _resolve_env(cfg)
    cfg = _apply_extends(path, cfg)  # main logic (deep merge presets)
    return cfg


# ================================================================
# 1. Load YAML
# ================================================================
def _load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


# ================================================================
# 2. Resolve environment variables
# ================================================================
def _resolve_env(obj):
    if isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env(v) for v in obj]
    if isinstance(obj, str):
        return _resolve_env_in_string(obj)
    return obj


def _resolve_env_in_string(s: str):
    """
    Replace environment variable patterns in a string.

    This function replaces `${VAR}` and `${VAR:default}` patterns in the input string
    with the value of the corresponding environment variable, or a default value if provided.

    Parameters
    ----------
    s : str
        The input string possibly containing environment variable patterns.

    Returns
    -------
    str or int or float
        The string with environment variables replaced. If the result is a numeric value,
        it is converted to int or float.

    Raises
    ------
    ValueError
        If a required environment variable is not set and no default is provided.

    Examples
    --------
    >>> os.environ["FOO"] = "bar"
    >>> _resolve_env_in_string("Value: ${FOO}")
    'Value: bar'
    >>> _resolve_env_in_string("Value: ${MISSING:default}")
    'Value: default'
    >>> _resolve_env_in_string("${NUM}")
    # If os.environ["NUM"] = "42", returns 42 (int)
    42
    """
    def replace_match(m):
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


def _try_numeric(v: str):
    """
    Attempt to convert a string value to int or float.
    Returns the original string if conversion fails.
    """
    try:
        if re.fullmatch(r"-?\d+", v):
            return int(v)
        return float(v)
    except ValueError:
        return v


# ================================================================
# 3. extends: preset composition (deep merge overlay)
# ================================================================
def _apply_extends(path: str, cfg: dict):
    """
    New preset system:

        extends:
          - preset1.yaml
          - preset2.yaml

    merge order:
        result = deep_merge(preset1, preset2)
        result = deep_merge(result, current_cfg)
    """
    if "extends" not in cfg:
        return cfg

    merged = {}

    for ext in cfg["extends"]:
        ext_path = os.path.join(os.path.dirname(path), ext)
        ext_cfg = parse_yaml(ext_path)
        merged = deep_merge(merged, ext_cfg)  # later preset overrides earlier

    local_cfg = {k: v for k, v in cfg.items() if k != "extends"}

    merged = deep_merge(merged, local_cfg)  # current file highest priority

    return merged
