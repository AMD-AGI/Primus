###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Argument parsing utilities for Primus CLI.
"""


def parse_cli_overrides(overrides: list) -> dict:
    """
    Parse CLI override arguments.

    Supported formats:
        - "key=value"
        - "nested.key=value"
        - "--key value" (common CLI style, converted internally to "key=value")

    Args:
        overrides: List of raw CLI override tokens, e.g.:
            ["lr=0.001", "batch_size=32"]
            ["--train_iters", "10"]

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
        - Boolean: "true"/"false" (case-insensitive) -> bool
        - Integer: digits or negative digits -> int
        - Float: contains decimal point -> float
        - String: everything else remains as string

    Nested Keys:
        - Dot notation creates nested dictionaries
        - "model.layers=24" becomes {"model": {"layers": 24}}
        - Multiple nested keys merge into the same parent dict
    """
    # First normalise tokens to "key=value" form.
    normalized: list[str] = []
    i = 0
    while i < len(overrides):
        item = overrides[i]

        # Already in key=value form (including "--key=value")
        if "=" in item:
            normalized.append(item)
            i += 1
            continue

        # Handle "--key value" → "key=value"
        if item.startswith("--") and i + 1 < len(overrides) and "=" not in overrides[i + 1]:
            key = item.lstrip("-")
            value = overrides[i + 1]
            normalized.append(f"{key}={value}")
            i += 2
            continue

        # Fallback: invalid format, emit warning and skip
        print(f"[Primus] Warning: Skipping invalid override format: {item}")
        i += 1

    result: dict = {}
    for item in normalized:
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Try to convert to appropriate type
        try:
            # Try boolean
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            # Try float (handles negative values as well)
            elif "." in value:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string if float conversion fails
            else:
                # Fallback to integer parsing (including negative ints)
                try:
                    value = int(value)
                except ValueError:
                    pass  # Keep as string if int conversion fails
        except AttributeError:
            # Non-string values are left as-is
            pass

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
