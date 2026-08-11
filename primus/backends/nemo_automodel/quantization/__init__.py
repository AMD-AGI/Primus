###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Model-agnostic low-precision hooks for the AutoModel diffusion backend.

Everything here applies to any diffusion model the recipe can build (FP8 GEMM,
FP8 attention, non-deterministic bf16 attention), which is why it lives outside
``models/``.

Deliberately no re-exports: each module is env-gated and pulls in optional
dependencies (primus_turbo, aiter) at import time. The trainer imports them
lazily by dotted path so a missing dependency degrades to a skipped hook rather
than an import error on a default run.
"""
