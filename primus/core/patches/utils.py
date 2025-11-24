###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Patch Utility Functions

This module provides utility functions for patch operations.
"""

import re

# ============================================================================
# Version Matching
# ============================================================================


def version_matches(version: str, pattern: str) -> bool:
    """
    Check if version matches a pattern.

    Patterns:
        - Exact match: "0.8.0"
        - Prefix match: "0.8.*"
        - Contains: "*2024-09*"

    Args:
        version: Version string to check
        pattern: Pattern to match against

    Returns:
        True if version matches pattern
    """
    if "*" not in pattern:
        return version == pattern

    # Simple wildcard matching
    regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
    return re.match(f"^{regex_pattern}$", version) is not None
