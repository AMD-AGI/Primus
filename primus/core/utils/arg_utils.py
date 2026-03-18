###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Argument parsing utilities for Primus CLI.
"""


def _coerce_cli_value_modern(raw_value):
    """Convert common CLI literals to bool/int/float/string."""
    value = raw_value
    try:
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        if "." in value:
            try:
                return float(value)
            except ValueError:
                pass
        try:
            return int(value)
        except ValueError:
            return value
    except AttributeError:
        return value


def _coerce_cli_value_legacy(raw_value):
    """Convert CLI literals using legacy `_parse_kv_overrides` behavior."""
    value = raw_value
    if not isinstance(value, str):
        return value

    lower_val = value.lower()
    if lower_val == "true":
        return True
    if lower_val == "false":
        return False

    # Keep compatibility with legacy `_parse_kv_overrides`, which used eval
    # for non-boolean values (e.g., None, lists, dicts, quoted strings).
    try:
        return eval(value, {}, {})
    except Exception:
        return value


def parse_cli_overrides(overrides: list, type_mode: str = "modern") -> dict:
    """
    Parse CLI override arguments.

    Supported formats:
        - "key=value"
        - "--key=value"
        - "nested.key=value"
        - "--key value" (common CLI style, converted internally to "key=value")

    Args:
        overrides: List of raw CLI override tokens, e.g.:
            ["lr=0.001", "batch_size=32"]
            ["--train_iters", "10"]
        type_mode: Type inference strategy.
            - "modern": bool/int/float/string (original parse_cli_overrides behavior)
            - "legacy": bool + eval fallback (old _parse_kv_overrides behavior)

    Returns:
        Dictionary with parsed key-value pairs

    Examples:
        >>> parse_cli_overrides(["lr=0.001", "batch_size=32"])
        {"lr": 0.001, "batch_size": 32}

        >>> parse_cli_overrides(["model.layers=24"])
        {"model": {"layers": 24}}

        >>> parse_cli_overrides(["use_cache=true", "verbose=False"])
        {"use_cache": True, "verbose": False}

    Type Inference Rules:
        - modern: bool/int/float/string
        - legacy: bool + eval fallback

    Nested Keys:
        - Dot notation creates nested dictionaries
        - "model.layers=24" becomes {"model": {"layers": 24}}
        - Multiple nested keys merge into the same parent dict
    """
    # First normalize tokens to "key=value" form.
    normalized: list[str] = []
    i = 0
    while i < len(overrides):
        item = overrides[i]
        if not isinstance(item, str):
            normalized.append(str(item))
            i += 1
            continue

        item = item.strip()
        if not item:
            i += 1
            continue

        # Already in key=value form (including "--key=value")
        if "=" in item:
            key, value = item.split("=", 1)
            key = key.lstrip("-").strip()
            value = value.strip()
            normalized.append(f"{key}={value}")
            i += 1
            continue

        # Handle "--key value" → "key=value"
        if (
            item.startswith("--")
            and i + 1 < len(overrides)
            and isinstance(overrides[i + 1], str)
            and not overrides[i + 1].startswith("--")
        ):
            key = item.lstrip("-")
            value = overrides[i + 1]
            normalized.append(f"{key}={value}")
            i += 2
            continue

        # Handle bare "--flag" as boolean true.
        if item.startswith("--"):
            key = item.lstrip("-")
            normalized.append(f"{key}=true")
            i += 1
            continue

        # Fallback: invalid format, emit warning and skip
        print(f"[Primus] Warning: Skipping invalid override format: {item}")
        i += 1

    if type_mode not in ("modern", "legacy"):
        raise ValueError(f"Unsupported type_mode: {type_mode}")
    coerce_fn = _coerce_cli_value_modern if type_mode == "modern" else _coerce_cli_value_legacy

    result: dict = {}
    for item in normalized:
        key, value = item.split("=", 1)
        key = key.strip()
        value = coerce_fn(value.strip())

        # Handle nested keys (e.g., model.layers -> {"model": {"layers": ...}})
        if "." in key:
            keys = key.split(".")
            current = result
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            result[key] = value

    return result
