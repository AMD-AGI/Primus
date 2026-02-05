###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import copy
from typing import Any, Dict


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries with selective copying optimization.

    Rules:
      - override wins (override overwrites base)
      - nested dicts are merged recursively
      - non-dict values replaced directly
      - override can introduce new fields

    Optimization: Avoids deep-copying the entire base dict upfront.
    Instead, only recursively merges nested dict branches, while non-dict
    values are assigned by reference. This reduces memory allocations
    for configs with many nested dicts.

    Note: This creates a new dict with references to non-dict values from base
    and override. If base/override contain mutable non-dict objects and you
    need true isolation, consider copy.deepcopy() on the result.

    Example:
        base = {"a": 1, "b": {"x": 10, "y": 20}}
        override = {"b": {"y": 999}, "c": 3}

        deep_merge(base, override) → {
            "a": 1,
            "b": {"x": 10, "y": 999},
            "c": 3,
        }
    """
    result = {}

    # Copy base keys, recursively merging if needed
    for key, val in base.items():
        if key not in override:
            result[key] = val
        elif isinstance(val, dict) and isinstance(override[key], dict):
            # Recursive merge for nested dicts
            result[key] = deep_merge(val, override[key])
        else:
            # Override value wins
            result[key] = override[key]

    # Add new keys from override
    for key in override:
        if key not in base:
            result[key] = override[key]

    return result


def shallow_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shallow merge:
      - Only top-level keys
      - override wins for direct keys
      - No recursive merging

    Example:
        base = {"a": 1, "b": {"x": 10}}
        override = {"b": {"y": 20}}

        shallow_merge(base, override) → {
            "a": 1,
            "b": {"y": 20},   # entire dict replaced
        }
    """
    result = copy.deepcopy(base)
    result.update(override)
    return result
