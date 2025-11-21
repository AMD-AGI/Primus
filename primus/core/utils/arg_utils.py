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
    Parse CLI override arguments in the format key=value.

    Args:
        overrides: List of strings in the format ["key=value", "nested.key=value", ...]

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
    result = {}
    for item in overrides:
        if "=" not in item:
            print(f"[Primus] Warning: Skipping invalid override format: {item}")
            continue

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Try to convert to appropriate type
        try:
            # Try boolean
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            # Try int
            elif value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                value = int(value)
            # Try float
            elif "." in value:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string
        except (ValueError, AttributeError):
            pass  # Keep as string

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
