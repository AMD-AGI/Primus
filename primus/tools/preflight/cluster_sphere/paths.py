###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import os
from pathlib import Path


def resolve_cluster_sphere_root() -> Path:
    """
    Return the Primus Cluster Sphere integration root: the directory
    ``primus/tools/preflight/cluster_sphere/`` (this package).

    RDMA env recommendations and ``ib_write_bw`` orchestration are implemented
    in Python under that tree.

    **Override (optional):** if ``PRIMUS_CLUSTER_SPHERE_ROOT`` is set to an
    existing directory, that path is returned instead (e.g. fork or vendor
    staging). ``DIST_INF_COOKBOOK_ROOT`` is no longer read.
    """
    env_override = os.environ.get("PRIMUS_CLUSTER_SPHERE_ROOT", "").strip()
    if env_override:
        p = Path(env_override).expanduser().resolve()
        if p.is_dir():
            return p

    return Path(__file__).resolve().parent
