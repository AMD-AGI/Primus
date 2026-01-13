###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import os
from typing import Any, Dict, List

from .network_probe import probe_network
from .utils import Finding


def run_network_full_checks(*, comm_sanity: bool = False) -> Dict[str, Any]:
    """
    Level: full

    Verify runtime process group sanity (best-effort) and optionally run a minimal
    allreduce sanity test (no perf measurement).
    """
    probe = probe_network()
    findings: List[Finding] = []

    runtime: Dict[str, Any] = {"pg_backend": None, "pg_init_ok": True, "pg_error": None}
    runtime_comm: Dict[str, Any] = {"allreduce_tested": False, "allreduce_ok": None, "allreduce_error": None}

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

        # Optional minimal allreduce test
        if comm_sanity:
            runtime_comm["allreduce_tested"] = True
            if dist.is_available() and dist.is_initialized():
                try:
                    # Use current device for NCCL collectives.
                    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                    if torch.cuda.is_available():
                        torch.cuda.set_device(local_rank)
                        t = torch.ones((1,), device="cuda", dtype=torch.float32)
                    else:
                        t = torch.ones((1,), device="cpu", dtype=torch.float32)
                    dist.all_reduce(t)
                    runtime_comm["allreduce_ok"] = True
                except Exception as e:
                    runtime_comm["allreduce_ok"] = False
                    runtime_comm["allreduce_error"] = str(e)
                    findings.append(Finding("warn", "Allreduce sanity failed (warn-only)", {"error": str(e)}))
            else:
                runtime_comm["allreduce_ok"] = None
                runtime_comm["allreduce_error"] = "Process group not initialized"
    except Exception as e:
        # torch not available or dist import failed
        runtime["pg_init_ok"] = False
        runtime["pg_error"] = str(e)
        findings.append(Finding("warn", "Runtime process group sanity unavailable", {"error": str(e)}))

    findings.append(Finding("info", "Runtime process group sanity", {"runtime": runtime}))
    if comm_sanity:
        findings.append(Finding("info", "Minimal communication sanity", {"runtime_comm": runtime_comm}))

    return {"probe": probe, "findings": findings}
