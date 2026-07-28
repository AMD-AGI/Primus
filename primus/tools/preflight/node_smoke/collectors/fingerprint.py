###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tier 1 -- A. Software-stack fingerprint (drift detection happens at aggregate)."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

from ..shell_utils import _parse_os_release_pretty, _read_text


def _collect_node_fingerprint() -> Dict[str, Any]:
    """Collect a deterministic, hashable fingerprint of the software stack
    on this node so the aggregator can detect drift across the cluster.

    Every value is best-effort: missing tools / files become ``None`` rather
    than raising. The aggregator skips ``None`` values when computing the
    cluster majority for a given key.
    """
    fp: Dict[str, Any] = {}

    # Kernel + OS
    try:
        fp["kernel"] = os.uname().release
    except Exception:
        fp["kernel"] = None
    fp["os_release"] = _parse_os_release_pretty()
    fp["python"] = sys.version.split()[0]

    # ROCm / HIP / amdgpu
    fp["rocm"] = _read_text("/opt/rocm/.info/version") or None
    fp["amdgpu_driver"] = _read_text("/sys/module/amdgpu/version") or None

    # PyTorch + (R)CCL
    try:
        import torch  # type: ignore

        fp["torch"] = getattr(torch, "__version__", None)
        fp["torch_hip"] = getattr(getattr(torch, "version", None), "hip", None)
        try:
            v = torch.cuda.nccl.version()  # type: ignore[attr-defined]
            if isinstance(v, tuple):
                fp["rccl"] = ".".join(str(x) for x in v)
            else:
                fp["rccl"] = str(v)
        except Exception:
            fp["rccl"] = None

        # Locate librccl.so under torch's lib dir for a stable per-node path.
        try:
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            for n in sorted(os.listdir(torch_lib)):
                if n.startswith("librccl.so"):
                    fp["rccl_path"] = os.path.join(torch_lib, n)
                    break
        except Exception:
            pass
    except Exception:
        fp["torch"] = None
        fp["torch_hip"] = None
        fp["rccl"] = None

    # Per-IB-device firmware + HCA model fingerprints. Both are critical for
    # detecting "1 of N nodes flashed differently" silent regressions.
    nic_fw: Dict[str, str] = {}
    nic_hca: Dict[str, str] = {}
    ib_root = "/sys/class/infiniband"
    if os.path.isdir(ib_root):
        try:
            for dev in sorted(os.listdir(ib_root)):
                fw = _read_text(os.path.join(ib_root, dev, "fw_ver"))
                if fw:
                    nic_fw[dev] = fw
                hca = _read_text(os.path.join(ib_root, dev, "hca_type"))
                if hca:
                    nic_hca[dev] = hca
        except Exception:
            pass
    fp["nic_fw"] = nic_fw or None
    fp["nic_hca"] = nic_hca or None

    return fp
