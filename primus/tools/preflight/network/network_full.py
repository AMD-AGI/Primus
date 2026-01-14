###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List

from .network_probe import probe_network
from .utils import Finding


def run_network_full_checks() -> Dict[str, Any]:
    """
    Level: full

    Verify runtime process group sanity (best-effort).
    """
    probe = probe_network()
    findings: List[Finding] = []

    runtime: Dict[str, Any] = {"pg_backend": None, "pg_init_ok": True, "pg_error": None}

    try:
        import torch  # type: ignore
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            runtime["pg_backend"] = dist.get_backend()
            runtime["pg_init_ok"] = True
        else:
            # If distributed intent is detected but PG is not initialized, treat as WARN.
            if bool(probe.intent.get("is_distributed")):
                runtime["pg_init_ok"] = False
                runtime["pg_error"] = "Process group not initialized"
                findings.append(Finding("warn", "Runtime process group not initialized", runtime))
    except Exception as e:
        # torch not available or dist import failed
        runtime["pg_init_ok"] = False
        runtime["pg_error"] = str(e)
        findings.append(Finding("warn", "Runtime process group sanity unavailable", {"error": str(e)}))

    findings.append(Finding("info", "Runtime process group sanity", {"runtime": runtime}))

    return {"probe": probe, "findings": findings}
