# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""ODC runtime configuration.

The ODC primitives (``odc.primitives.*``) are decoupled library code with no
knowledge of Primus' config system. Historically they read their tuning knobs
straight from ``os.environ`` (ODC_P2P_BACKEND, ODC_GDA_*, ...), which violates
the Primus rule that formal feature logic must be config-driven, not env-driven.

This leaf module (imports only stdlib, so it can be populated BEFORE the heavy
``odc.primitives`` modules are first imported) holds a single ``OdcRuntimeConfig``
instance. The Primus ODC integration patch reads the ``odc_*`` config items from
the trainer config and calls :func:`set_config` at ``before_train`` -- i.e. after
the Primus config is available but before ``odc.primitives`` (and their
import-time backend selection) run. The primitives then read values from here
instead of the environment.

Defaults below are byte-for-byte the previous env defaults, so an unset config
reproduces the prior behaviour exactly.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OdcRuntimeConfig:
    # --- symmetric-memory backend selection (was ODC_P2P_BACKEND) ---
    # "mori" (default) | "rocshmem"
    p2p_backend: str = "mori"
    # MORI init method (was ODC_MORI_INIT): "pg" (default) | "uid"
    mori_init: str = "pg"
    # BufferSplitter max global buffer size in bytes (was ODC_MAX_BUFFER_SIZE).
    max_buffer_size: int = 64 * 1024 * 1024

    # --- rocSHMEM backend (was ODC_ROCSHMEM_GDA / ODC_ROCSHMEM_LIB) ---
    # GPU-direct (GDA) device-kernel cross-node path; required for multi-node.
    rocshmem_gda: bool = False
    # Optional path to a monolithic librs_host_gda.so ctypes override; None ->
    # consume the rocSHMEM ops from Primus-Turbo.
    rocshmem_lib: Optional[str] = None

    # --- GDA reduce-scatter tuning knobs ---
    # reduce-scatter kernel grid blocks (was ODC_GDA_RS_BLOCKS).
    gda_rs_blocks: int = 64
    # peer-pipeline batch depth (was PRIMUS_TURBO_ODC_GDA_PIPE). ALSO consumed by
    # the Primus-Turbo device kernel via getenv, so set_config bridges it back to
    # the PRIMUS_TURBO_ODC_GDA_PIPE env var for the C++ side.
    gda_pipe: int = 1
    # defer the cross-node reduce to once-per-minibatch (was ODC_GDA_DEFER_REDUCE):
    # "auto" (default: on iff multi-node, i.e. n_pes > local_world_size) | "1" | "0".
    gda_defer_reduce: str = "auto"
    # cross-node write-visibility warm-up mode (was ODC_GDA_WARMUP_MODE):
    # "strided" (default) | "full" | "hdp" | "fence" | "hdpfence".
    gda_warmup_mode: str = "strided"
    # strided warm-up page stride in bytes (was ODC_GDA_STRIDE_BYTES).
    gda_stride_bytes: int = 65536


_CONFIG = OdcRuntimeConfig()


def get_config() -> OdcRuntimeConfig:
    """Return the process-wide ODC runtime config (populated by set_config)."""
    return _CONFIG


def set_config(**kwargs) -> OdcRuntimeConfig:
    """Update the ODC runtime config in place from keyword overrides.

    Only known fields are applied; None values are ignored (keep the default).
    Returns the config instance. Also bridges gda_pipe back to the
    PRIMUS_TURBO_ODC_GDA_PIPE env var, which the Primus-Turbo device kernel reads
    via getenv (the one knob that must stay visible to the C++ side).
    """
    import os

    valid = set(OdcRuntimeConfig.__dataclass_fields__)
    for k, v in kwargs.items():
        if k in valid and v is not None:
            setattr(_CONFIG, k, v)
    # Bridge the pipe depth to the env var the turbo C++ kernel reads.
    os.environ["PRIMUS_TURBO_ODC_GDA_PIPE"] = str(int(_CONFIG.gda_pipe))
    return _CONFIG
