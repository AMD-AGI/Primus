###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tier 1 collectors -- one module per Tier 1 sub-section.

* ``dmesg``        : recent dmesg error scan
* ``fingerprint``  : Tier 1 A -- software-stack fingerprint
* ``nics``         : Tier 1 B -- NIC / RDMA roll-call
* ``host_limits``  : Tier 1 C -- host limits (ulimit, /dev/shm, NUMA, governor)
* ``gpu_low_level``: Tier 1 D-1 -- per-GPU ECC / clocks / power via amd-smi
* ``xgmi``         : Tier 1 D-2 -- XGMI topology matrix
* ``clock``        : Tier 1 E -- wall time + time-daemon active states
* ``rocm_smi``     : Tier 1 F + fallbacks -- rocm-smi self-latency and
                     amd-smi fallback parsers
* ``gpu_processes``: Tier 1 G -- foreign / leaked process detection
* ``tooling``      : tooling-availability inventory
* ``reused_info``  : reused gpu/host/network info collectors
"""

from __future__ import annotations
