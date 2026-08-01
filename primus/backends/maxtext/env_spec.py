###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText backend environment specification (single source of truth).

This module declares *all* MaxText/JAX performance and architecture environment
that Primus is responsible for. It is consumed by :class:`MaxTextAdapter` via
``env_defaults()`` and applied by the base adapter before JAX/XLA is imported.

Previously this env was duplicated across three places (``examples/run_pretrain.sh``,
``runner/helpers/hooks/train/pretrain/maxtext/prepare.py``, and the adapter's old
``_apply_arch_env_defaults``). It now lives here only.

Provenance: these values mirror the known-good MaxText launch env scripts at
https://github.com/ROCm/MAD/tree/develop/scripts/jax-maxtext/env_scripts . Those
scripts are identical across GPU archs *except* for two single-variable
exceptions, which are the only arch-gated entries below:

  * gfx950 (MI350X/MI355X): ``RCCL_WARP_SPEED_AUTO=0``  (WarpSpeed default-on can NaN)
  * gfx942 (MI300X):        ``HSA_NO_SCRATCH_RECLAIM=1``

Everything else — including all ``XLA_FLAGS`` knobs and the NVTE/HIP/HSA tunables —
is architecture-agnostic (``arch="all"``).

Precedence (see env_registry): per-config ``env:`` > outer/shell env > these
defaults > image-baked. ``XLA_FLAGS`` is merged per-flag so the managed knobs win
over an image-baked ``XLA_FLAGS`` (which may carry ``--xla_gpu_autotune_level=0``)
while preserving any unrelated baked flags.
"""

from __future__ import annotations

import os
from typing import List

from primus.core.backend.env_registry import (
    ARCH_ALL,
    ARCH_GFX942,
    ARCH_GFX950,
    MODE_XLA_MERGE,
    EnvVar,
)


def _build_xla_flags() -> str:
    """Assemble the managed ``XLA_FLAGS`` string.

    The autotune level is parameterized via ``XLA_GPU_AUTOTUNE_LEVEL`` (default 4).
    MUST be >= 1: with autotuning off (=0, as baked into some rock/TheRock images)
    XLA/hipBLASLt always picks the default fp8 GEMM kernel, which overflows on the
    fine-grained MoE expert einsums (qwen3-30B-A3B, deepseek-v2-lite) -> NaN loss on
    ~every fp8 run. autotune_level>=4 lets XLA pick a numerically stable fp8 kernel.
    """
    autotune_level = os.getenv("XLA_GPU_AUTOTUNE_LEVEL", "4")
    flags = (
        "--xla_gpu_memory_limit_slop_factor=95 "
        "--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 "
        "--xla_gpu_enable_command_buffer='' "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_all_gather_combine_threshold_bytes=8589934592 "
        "--xla_gpu_enable_triton_gemm=false "
        "--xla_gpu_enable_cublaslt=true "
        f"--xla_gpu_autotune_level={autotune_level} "
        "--xla_gpu_enable_all_gather_combine_by_dim=false"
    )
    if os.getenv("DUMP_HLO", "0") == "1":
        dump_dir = os.getenv("DUMP_HLO_DIR", "output/xla_dump_hlo")
        flags += f" --xla_dump_to={dump_dir}"
    return flags


def maxtext_env_defaults() -> List[EnvVar]:
    """Return the declarative MaxText env defaults for the current run.

    Built fresh on each call so it reflects the live values of the few
    parameterizing inputs (``XLA_GPU_AUTOTUNE_LEVEL``, ``DUMP_HLO``, ``NNODES``,
    ``MASTER_ADDR``/``MASTER_PORT``).
    """
    entries: List[EnvVar] = [
        # ---- XLA / JAX ----
        EnvVar("XLA_FLAGS", _build_xla_flags(), mode=MODE_XLA_MERGE,
               note="managed XLA knobs incl. autotune (fp8 MoE NaN fix)"),
        EnvVar("XLA_PYTHON_CLIENT_MEM_FRACTION", ".97",
               note="avoid HSA OOM during multi-node"),
        EnvVar("TF_CPP_MIN_LOG_LEVEL", "2",
               note="suppress benign JAX/MaxText shutdown errors"),
        # ---- Transformer Engine (NVTE) ----
        EnvVar("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1"),
        EnvVar("NVTE_USE_HIPBLASLT", "1"),
        EnvVar("NVTE_FUSED_ATTN", "1"),
        EnvVar("NVTE_FUSED_ATTN_CK", "1"),
        EnvVar("NVTE_FUSED_ATTN_AOTRITON", "0"),
        EnvVar("NVTE_CK_USES_BWD_V3", "1"),
        EnvVar("NVTE_CK_USES_FWD_V3", "1"),
        EnvVar("NVTE_CK_IS_V3_ATOMIC_FP32", "0"),
        EnvVar("NVTE_CK_HOW_V3_BF16_CVT", "2"),
        # ---- AMD GPU / HIP / HSA ----
        EnvVar("GPU_MAX_HW_QUEUES", "2"),
        EnvVar("HIP_FORCE_DEV_KERNARG", "1"),
        EnvVar("HSA_FORCE_FINE_GRAIN_PCIE", "1"),
        # NOTE: NCCL_DEBUG is deliberately NOT managed here. It is purely
        # diagnostic (logs the RCCL version once) and every launcher already sets
        # a shared empty default (base_env.sh / run_pretrain.sh), so owning it in
        # the adapter would only create a cosmetic empty-vs-VERSION cross-path diff.
        # ---- Architecture-gated (the ONLY two arch differences) ----
        EnvVar("RCCL_WARP_SPEED_AUTO", "0", arch=ARCH_GFX950,
               note="gfx950 WarpSpeed default-on can cause NaN losses"),
        EnvVar("HSA_NO_SCRATCH_RECLAIM", "1", arch=ARCH_GFX942,
               note="gfx942 scratch-reclaim stability"),
    ]

    # Multi-node JAX coordinator: derive from MASTER_ADDR/PORT so both the
    # run_pretrain.sh path and the primus-cli path get a coordinator without any
    # shell plumbing. Single-node runs deliberately leave these unset so MaxText
    # uses the local single-controller path (no jax.distributed.initialize).
    try:
        nnodes = int(os.getenv("NNODES", "1"))
    except ValueError:
        nnodes = 1
    if nnodes > 1:
        entries.append(EnvVar("JAX_COORDINATOR_IP", os.getenv("MASTER_ADDR", "localhost")))
        entries.append(EnvVar("JAX_COORDINATOR_PORT", os.getenv("MASTER_PORT", "1234")))

    return entries
