###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Make megatron's CPU optimizer offload usable on a device where TE's FusedAdam hangs.

Offload matters here because optimizer state is what puts full-depth Gemma 4 out
of reach on a single GPU. Moving that state to host memory is the only lever that
changes the shape of the problem rather than trimming the model.

The obstruction is that the offload path insists on TE's FusedAdam, and TE's
FusedAdam does not return on this device (see ``gemma4_local_spec`` for the
symptom: ``optimizer.step()`` pinned at full GPU utilisation with an unchanging
``multi_tensor_apply`` stack, driver logging ``MES(0, 0) ring buffer is full``).
So turning offload on without also moving the GPU-side optimizer off FusedAdam
converts a memory problem into a hang.

Why the two existing knobs do not get you there
-----------------------------------------------
``megatron/core/optimizer/__init__.py`` has a config field that looks like
exactly the right switch, ``use_torch_optimizer_for_cpu_offload``
(``optimizer_config.py:350``), and the offload branch honours it -- and then
immediately throws the result away for adam (lines 511-517)::

    gpu_optimizer_cls = Adam if config.optimizer == 'adam' else SGD
    cpu_optimizer_cls = CPUAdam if config.optimizer == 'adam' else CPUSGD
    if config.use_torch_optimizer_for_cpu_offload:
        gpu_optimizer_cls = cpu_optimizer_cls        # line 514: honoured...
    if config.optimizer == 'adam':
        gpu_optimizer_cls = Adam                     # line 516: ...then clobbered

Line 516 is unconditional within the adam case, so for ``optimizer='adam'`` --
which precision-aware optimisation already asserts is the only allowed value --
``use_torch_optimizer_for_cpu_offload`` has no effect whatsoever. It is not that
the flag is wrong for us; it is inert. This looks like a plain bug and is worth
reporting upstream: a field whose entire purpose is to avoid the fused optimizer
silently cannot.

The other knob, ``USING_PYTORCH_OPTIMIZER``, is what ``gemma4_local_spec`` flips
to keep the *non-offload* path on torch. The offload branch never reads it. It
is consulted only at line 556, inside the ``elif config.optimizer == 'adam'``
that offload skips entirely. So the existing ``PRIMUS_GEMMA4_TORCH_OPTIM``
escape hatch does not cover offload, and enabling offload would silently undo it.

What this patch does instead
----------------------------
It rebinds the module-level ``Adam`` name in ``megatron.core.optimizer`` to
``torch.optim.AdamW``. That name is bound once at import time, preferring TE's
FusedAdam (lines 14-35), and line 516 reads it. Rebinding is safe because that
name has only two readers: line 516 here, and line 560, which is reachable only
when ``USING_PYTORCH_OPTIMIZER`` is false -- and ``gemma4_local_spec`` sets it
true. There are no ``isinstance`` checks against it.

Precision-aware optimisation stays *on* for offload, which is the opposite of
what ``gemma4_local_spec`` does for the non-offload path, and the difference is
real rather than inconsistent. ``gemma4_local_spec`` turns it off because
precision-aware relies on FusedAdam's master weights and a bare torch Adam has
none. Under offload the master weights come from ``HybridDeviceOptimizer``
itself, which is passed ``param_update_in_fp32=True`` unconditionally (line 540)
and maintains its own fp32 copies plus the ``decoupled_grad`` plumbing that
precision-aware expects. Megatron agrees: ``optimizer_config.py:444`` returns
early from the "precision-aware requires TE FusedAdam" check when
``optimizer_cpu_offload`` is set, precisely because HDO supplies it. Keeping
precision-aware on is also the point -- without it the fp32 master weights stay
resident on the GPU and much of the saving is given back.

Numerics: the CPU side was already ``torch.optim.AdamW`` upstream
(``CPUAdam``, line 11). This makes the GPU side the same class, so both halves
of the hybrid optimizer now run identical math and the split point stops being
observable in the update rule. That is a narrowing of behaviour, not a widening.

Left deliberately alone: ``optimizer_defaults`` still carries
``bias_correction=True`` and ``fused=True`` (lines 523-524). Those ride inside
the param-group dicts rather than as constructor kwargs, so torch's AdamW keeps
the keys it recognises and ignores ``bias_correction``. ``fused=True`` selects
torch's fused AdamW, which is the upstream intent for the CPU optimizer and is
supported for fp32 params on the GPU side; if it ever proves unavailable the
symptom will be a loud constructor error, not a silent fallback.
"""

from __future__ import annotations

import os
from typing import Any

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCHED_ATTR = "_primus_gemma4_cpu_offload_patched"

_ENV = "PRIMUS_GEMMA4_CPU_OFFLOAD"


def offload_fraction() -> float | None:
    """Read the requested offload fraction, or None when offload is not requested.

    Accepts a bare ``1`` as a convenience for "offload everything", since that is
    the interesting setting and writing ``1`` rather than ``1.0`` is the obvious
    mistake to make. ``0`` means "explicitly requested nothing", which is treated
    as not requested rather than as an error.

    This is exposed rather than private because ``gemma4_local_spec`` has to ask
    the same question to decide whether to clear the precision-aware flag, and
    routing that through the environment keeps the two patches independent of the
    order they happen to be applied in.
    """
    raw = os.environ.get(_ENV)
    if raw is None or not raw.strip():
        return None

    try:
        value = float(raw.strip())
    except ValueError:
        log_rank_0(f"[Patch:gemma4.cpu_offload] {_ENV}={raw!r} is not a number; ignoring")
        return None

    if not 0.0 <= value <= 1.0:
        log_rank_0(f"[Patch:gemma4.cpu_offload] {_ENV}={raw!r} is outside [0, 1]; ignoring")
        return None

    return value or None


def _use_torch_gpu_optimizer() -> None:
    """Point the offload path's GPU optimizer class at torch's AdamW."""
    import sys

    import torch

    module = sys.modules.get("megatron.core.optimizer")
    if module is None:
        # Nothing to rebind yet. Unlike the config fields below there is no
        # useful partial action available: the name will be bound to FusedAdam
        # when the module is eventually imported, and this patch will not run
        # again. Say so, because the run would otherwise hang later with no hint.
        log_rank_0(
            "[Patch:gemma4.cpu_offload] megatron.core.optimizer not imported yet, so the "
            "offload path will use TE FusedAdam for the GPU half. Expect a hang on this device."
        )
        return

    current = getattr(module, "Adam", None)
    if current is torch.optim.AdamW:
        return

    module.Adam = torch.optim.AdamW
    log_rank_0(
        "[Patch:gemma4.cpu_offload] rebound megatron.core.optimizer.Adam "
        f"({getattr(current, '__name__', current)} -> torch.optim.AdamW) so "
        "HybridDeviceOptimizer's GPU half is not TE FusedAdam"
    )


def _enable_offload(container: Any, fraction: float) -> None:
    opt_cfg = getattr(container, "optimizer", None)
    if opt_cfg is None:
        log_rank_0("[Patch:gemma4.cpu_offload] no optimizer config on the container; nothing to do")
        return

    opt_cfg.optimizer_cpu_offload = True
    opt_cfg.optimizer_offload_fraction = fraction

    # Records the intent even though megatron clobbers it at line 516. If that
    # bug is ever fixed upstream this becomes the mechanism and the rebind below
    # becomes a no-op, which is the right way round.
    opt_cfg.use_torch_optimizer_for_cpu_offload = True

    # HDO supplies the master weights, so precision-aware is both allowed and
    # wanted here -- see the module docstring.
    if not getattr(opt_cfg, "use_precision_aware_optimizer", False):
        opt_cfg.use_precision_aware_optimizer = True
        log_rank_0(
            "[Patch:gemma4.cpu_offload] set use_precision_aware_optimizer=True "
            "(HybridDeviceOptimizer provides the master weights; keeps fp32 copies off the GPU)"
        )

    # Both of these are asserted rather than defaulted downstream, so a mismatch
    # would surface as a bare AssertionError a long way from its cause.
    if not getattr(opt_cfg, "use_distributed_optimizer", False):
        opt_cfg.use_distributed_optimizer = True
        log_rank_0(
            "[Patch:gemma4.cpu_offload] set use_distributed_optimizer=True "
            "(asserted by use_precision_aware_optimizer)"
        )

    if not getattr(opt_cfg, "decoupled_weight_decay", True):
        opt_cfg.decoupled_weight_decay = True
        log_rank_0(
            "[Patch:gemma4.cpu_offload] set decoupled_weight_decay=True "
            "(asserted by the CPU offload branch, which is AdamW-only)"
        )

    log_rank_0(
        f"[Patch:gemma4.cpu_offload] optimizer_cpu_offload=True, offload_fraction={fraction} "
        "(optimizer state moves to host memory)"
    )

    _use_torch_gpu_optimizer()


@register_patch(
    "gemma4.cpu_offload",
    backend="megatron_bridge",
    phase="setup",
    description="Enable CPU optimizer offload with a torch GPU optimizer instead of TE FusedAdam",
)
def patch_gemma4_cpu_offload(ctx: PatchContext) -> None:
    fraction = offload_fraction()
    if fraction is None:
        return

    try:
        from primus.backends.megatron_bridge import (
            megatron_bridge_posttrain_trainer,
            megatron_bridge_pretrain_trainer,
        )
    except Exception:
        return

    for module in (megatron_bridge_pretrain_trainer, megatron_bridge_posttrain_trainer):
        original = getattr(module, "load_recipe_config", None)
        if original is None or getattr(original, _PATCHED_ATTR, False):
            continue

        def load_recipe_config(backend_args, _original=original, _fraction=fraction):
            container = _original(backend_args)
            _enable_offload(container, _fraction)
            return container

        setattr(load_recipe_config, _PATCHED_ATTR, True)
        module.load_recipe_config = load_recipe_config
