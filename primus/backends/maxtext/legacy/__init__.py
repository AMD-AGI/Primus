###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Legacy compatibility layer for Aug-version MaxText (date-versioned, e.g. ``2025.07.24``).

This package contains Primus-specific implementations adapted to the older
MaxText APIs.  When the old MaxText version is eventually retired, this entire
``legacy`` package can be removed.

NOTE: Files here are intentionally duplicated from their current-version
counterparts for isolation.  If you fix a bug in the current version,
check whether the same bug exists here.
"""
