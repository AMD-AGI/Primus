###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import copy
from typing import Any, Dict


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Rules:
      - override wins (override overwrites base)
      - nested dicts are merged recursively
      - non-dict values replaced directly
      - override can introduce new fields

    Example:
        base = {"a": 1, "b": {"x": 10, "y": 20}}
        override = {"b": {"y": 999}, "c": 3}

        deep_merge(base, override) → {
            "a": 1,
            "b": {"x": 10, "y": 999},
            "c": 3,
        }
    """
    result = copy.deepcopy(base)

    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)  # recursive merge
        else:
            result[key] = val  # override or append

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
