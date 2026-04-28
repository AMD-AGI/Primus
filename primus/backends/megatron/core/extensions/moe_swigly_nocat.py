###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Runtime monkeypatch — MoE SwiGLU forward+backward without cat/split.

The Triton kernels and autograd Function live in
:mod:`primus.backends.megatron.core.extensions.triton_swiglu`. This module
only contains the install hook that monkey-patches
``megatron.core.transformer.moe.experts.GroupedMLP.activation_func_with_probs``
to call ``triton_swiglu_with_probs`` instead of the default
``F.silu(gate) * up * probs`` Python composition.

The default composition gets fused by Inductor into
``triton_poi_fused__to_copy_cat_mul_silu_silu_backward_split_1`` in bwd,
which round-trips ``d_gate / d_up`` through a ``cat`` + ``split`` on stream 0
(see slab/notes/2026-04/2026-04-25_gptoss_23_swiglu_nocat_triton_verify.md).
The Triton kernel writes both gradients directly into a single ``[N, 2H]``
``dx`` buffer.

Activation
----------
* ``PRIMUS_MOE_SWIGLU_NOCAT=1`` (default ON in
  ``small_llm_moe_pretraining/primus/config_MI355X_*.sh``).
* Falls back silently when the MLP is not GLU-gated or the activation is
  not ``torch.nn.functional.silu``.

Rollback
--------
Set ``PRIMUS_MOE_SWIGLU_NOCAT=0`` and restart the trainer. No Megatron
state is mutated until a ``GroupedMLP`` is constructed.
"""
from __future__ import annotations

import os
import sys

import torch.nn.functional as F


def _enabled() -> bool:
    v = os.environ.get("PRIMUS_MOE_SWIGLU_NOCAT", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _log(msg: str) -> None:
    if os.environ.get("RANK", "0") == "0":
        print(f"[moe_swiglu_nocat] {msg}", file=sys.stderr, flush=True)


_INSTALLED = False


def install() -> bool:
    """Install the monkeypatch. Returns True on success."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if not _enabled():
        _log("disabled (set PRIMUS_MOE_SWIGLU_NOCAT=1 to enable)")
        return False

    try:
        from primus.backends.megatron.core.extensions.triton_swiglu import (
            triton_swiglu_with_probs,
        )
    except ImportError as exc:
        _log(f"could not import triton_swiglu_with_probs: {exc}; abort install")
        return False

    try:
        from megatron.core.transformer.moe.experts import GroupedMLP
    except Exception as exc:
        _log(f"failed to import GroupedMLP: {exc}; abort install")
        return False

    if getattr(GroupedMLP, "_swiglu_nocat_patched", False):
        _INSTALLED = True
        return True

    _orig_init = GroupedMLP.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        cfg = getattr(self, "config", None)
        if cfg is None:
            return
        if not getattr(cfg, "gated_linear_unit", False):
            return
        if getattr(cfg, "activation_func", None) is not F.silu:
            return
        self.activation_func_with_probs = triton_swiglu_with_probs

    GroupedMLP.__init__ = _patched_init
    GroupedMLP._swiglu_nocat_patched = True
    _INSTALLED = True
    _log("patched GroupedMLP.activation_func_with_probs with Triton no-cat op")
    return True
