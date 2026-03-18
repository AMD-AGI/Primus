###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Patch Utility Functions.

Provides:
    - Wildcard version matching
    - Semantic version parsing
    - Extended version range comparison (>=, <=, between, wildcard)
"""

import re
from typing import Optional, Tuple

# -----------------------------------------------------------------------------
# Wildcard Version Matching
# -----------------------------------------------------------------------------


def version_matches(version: str, pattern: str) -> bool:
    """
    Wildcard string matching using '*' for prefix/contains patterns.

    Examples:
        version_matches("0.8.1", "0.8.*")    -> True
        version_matches("commit:945abc", "*abc") -> True
        version_matches("1.2.3", "1.2.3")    -> True
    """
    if not version:
        return False

    if "*" not in pattern:
        return version == pattern

    regex = pattern.replace(".", r"\.").replace("*", ".*")
    return re.fullmatch(regex, version) is not None


# -----------------------------------------------------------------------------
# Semantic Version Parsing
# -----------------------------------------------------------------------------


def _parse_semver(version: str) -> Optional[Tuple[int, ...]]:
    """
    Parse semantic version string 'X.Y.Z' into a tuple (X, Y, Z).

    Accepts prerelease suffixes like '0.15.0rc8' by extracting the
    leading numeric portion of each segment.
    """
    if not version:
        return None

    parts = version.split(".")
    nums = []
    for p in parts:
        m = re.match(r"^(\d+)", p)
        if not m:
            return None
        nums.append(int(m.group(1)))
    return tuple(nums)


# -----------------------------------------------------------------------------
# Extended Version Comparison
# -----------------------------------------------------------------------------


def version_in_range(version: str, pattern: str) -> bool:
    """
    Extended version filtering supporting:

        - Exact match:     "0.8.0"
        - Wildcard:        "0.8.*"
        - Comparator:      ">=0.8.0", "<0.9.0", "<=1.2.3", ">1.1.0"
        - Range (between): "0.8.0~0.8.5"

    If the pattern is not a comparator or a range, fallback to wildcard/exact.
    """
    if not version:
        return False

    # (1) Between range: "0.8.0~0.8.5"
    if "~" in pattern:
        lo, hi = pattern.split("~", 1)
        v = _parse_semver(version)
        lo_v = _parse_semver(lo)
        hi_v = _parse_semver(hi)
        if not (v and lo_v and hi_v):
            return False
        return lo_v <= v <= hi_v

    # (2) Comparators: >=, <=, >, <
    m = re.match(r"^(<=|>=|<|>)([\d\.]+)$", pattern)
    if m:
        op, target = m.groups()
        v = _parse_semver(version)
        t = _parse_semver(target)
        if not (v and t):
            return False

        if op == ">=":
            return v >= t
        if op == "<=":
            return v <= t
        if op == ">":
            return v > t
        if op == "<":
            return v < t

    # (3) Fallback to wildcard/exact match
    return version_matches(version, pattern)
