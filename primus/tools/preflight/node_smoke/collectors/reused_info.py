###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Reused gpu/host/network info collectors from the rest of the preflight tree.

These collectors already work without a global PG and produce ``Finding``
objects (level='fail' counts as a node failure). We import each one
lazily so a missing dependency in one section doesn't cascade.
"""

from __future__ import annotations

from typing import Any, Dict

from ..shell_utils import _findings_to_dicts


def _collect_reused_info() -> Dict[str, Any]:
    """Run the existing host/gpu/network info collectors. They already work
    without a global PG and produce ``Finding`` objects (level='fail' counts
    as a node failure)."""
    section: Dict[str, Any] = {"gpu_info": [], "host_info": [], "network_info": []}
    try:
        from primus.tools.preflight.gpu.info import collect_gpu_info

        section["gpu_info"] = _findings_to_dicts(collect_gpu_info())
    except Exception as e:
        section["gpu_info"] = [
            {"level": "warn", "message": "collect_gpu_info raised", "details": {"error": str(e)}}
        ]
    try:
        from primus.tools.preflight.host.info import collect_host_info

        section["host_info"] = _findings_to_dicts(collect_host_info())
    except Exception as e:
        section["host_info"] = [
            {"level": "warn", "message": "collect_host_info raised", "details": {"error": str(e)}}
        ]
    try:
        from primus.tools.preflight.network.info import collect_network_info

        # expect_distributed=False so we don't WARN about a missing world PG.
        section["network_info"] = _findings_to_dicts(
            collect_network_info(expect_distributed=False)
        )
    except Exception as e:
        section["network_info"] = [
            {"level": "warn", "message": "collect_network_info raised", "details": {"error": str(e)}}
        ]
    return section
