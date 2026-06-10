###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Inference (serving) projection for Primus.

This package mirrors the training-oriented :mod:`performance_projection` and
:mod:`memory_projection` packages but models *autoregressive inference*:

  * **Forward-only** compute — no backward pass, optimizer or gradients.
  * **Two phases** — prefill (prompt → first token, drives TTFT) and decode
    (autoregressive generation, drives inter-token latency / throughput).
  * **KV cache** memory — the dominant inference-time memory term.
  * **Serving features** — chunked prefill, KV-cache quantization,
    batching / concurrency, and speculative decoding.

The CLI entry point is :func:`launch_projection_from_cli`, wired from
``primus projection inference`` (see ``primus/cli/subcommands/projection.py``).
"""

from .launcher import launch_projection_from_cli

__all__ = ["launch_projection_from_cli"]
