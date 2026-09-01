###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared helpers for Megatron-Bridge checkpoint conversion hooks.

Use ``run_convert_checkpoints.py`` (or ``run_bridge_convert_checkpoints()``)
so convert-phase patches run in the **same** Python process as the bridge
converter. Applying patches in a separate ``python -c`` invocation does not
carry over to a subsequent ``convert_checkpoints.py`` subprocess.
"""


def apply_bridge_convert_patches() -> int:
    """Register + run all ``phase="convert"`` Megatron-Bridge patches.

    Must be called in the conversion hook process BEFORE importing or running
    ``megatron.bridge`` conversion entrypoints (e.g.
    ``convert_checkpoints.py import``).
    """
    import importlib

    importlib.import_module("primus.backends.megatron_bridge.patches.transformers_rope_theta_patches")
    from primus.core.patches import run_patches

    return run_patches(
        backend="megatron_bridge",
        phase="convert",
    )
