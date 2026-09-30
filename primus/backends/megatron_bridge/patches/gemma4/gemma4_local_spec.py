###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Make ``model.transformer_impl = "local"`` actually take effect for Gemma 4.

Setting ``transformer_impl`` to ``local`` on a Gemma 4 config silently does
nothing. ``Gemma4ModelProvider`` binds the layer spec at dataclass-construction
time (``gemma4_provider.py:322``)::

    transformer_layer_spec: Union[Callable, object] = field(
        default_factory=lambda: partial(_gemma4_block_spec, use_transformer_engine=HAVE_TE)
    )

``HAVE_TE`` is module-level and true whenever TransformerEngine imports, and
``transformer_impl`` is never consulted. So the spec is already frozen to the
TransformerEngine layers before any override runs, and the config ends up
self-contradictory: ``transformer_impl='local'`` next to a spec partial carrying
``use_transformer_engine=True``. The spec wins, TE layers get built, and the
only hint is a log line nobody reads.

This is worth fixing upstream rather than only here: ``transformer_impl`` is a
standard megatron knob and a user setting it has every reason to expect the TE
path to be off. Silently ignoring it is worse than rejecting it.

Why it matters on MI455X specifically
-------------------------------------
On the ``gfx1250-20260910`` image hipBLASLt cannot plan even a plain square bf16
GEMM and returns ``HIPBLAS_STATUS_INVALID_VALUE``; torch works only because
``TORCH_BLAS_PREFER_HIPBLASLT=0`` reroutes it to rocBLAS. TransformerEngine
calls hipBLASLt *directly* from ``rocm_gemm.hip`` and has no rocBLAS fallback
(the only NVTE GEMM knobs in this build select among CK / CUTLASS / HipKittens
*grouped* GEMM backends), so a TE layer fails with::

    rocm_gemm.hip:1603 in function hipblaslt_gemm: HIPBLASLT Error: 3

``Error: 3`` is ``HIPBLAS_STATUS_INVALID_VALUE`` -- the same status the probe
saw. Avoiding TE is therefore the only way to run on this image, which makes
``transformer_impl=local`` load-bearing rather than a tuning preference.

Numerics: this changes which *implementation* computes each layer, not the
mathematics. Megatron's local layers are the reference implementations; expect
small floating-point differences from TE's fused kernels, not different math.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCHED_ATTR = "_primus_gemma4_local_spec_patched"


def _rebind_spec(container: Any) -> None:
    model = getattr(container, "model", None)
    if model is None:
        return

    impl = getattr(model, "transformer_impl", None)

    # The optimizer switch is deliberately independent of the layer spec. TE's
    # FusedAdam does not return on this device: optimizer.step() stays pinned at
    # full GPU utilisation with an unchanging py-spy stack in
    # multi_tensor_apply.py:23, while the driver repeatedly logs "MES(0, 0) ring
    # buffer is full". So any run that builds TE layers -- which is exactly the
    # case this function otherwise declines to touch -- still needs a way to keep
    # the optimizer on torch. Without it, testing TE means risking a host reboot.
    if os.environ.get("PRIMUS_GEMMA4_TORCH_OPTIM") == "1":
        _use_torch_optimizer(container)
        log_rank_0(
            "[Patch:gemma4.local_spec] PRIMUS_GEMMA4_TORCH_OPTIM=1: forcing torch Adam "
            f"(transformer_impl={impl!r}); TE FusedAdam hangs on this device"
        )

    if impl != "local":
        return

    spec = getattr(model, "transformer_layer_spec", None)
    if spec is None:
        return

    # Only touch the exact shape we understand: a partial over the Gemma 4 block
    # spec that is currently bound to TE. Anything else (a hand-supplied spec, an
    # already-local partial) is left alone rather than guessed at.
    if not isinstance(spec, partial):
        log_rank_0(
            "[Patch:gemma4.local_spec] transformer_impl='local' but transformer_layer_spec "
            f"is {type(spec).__name__}, not a functools.partial -- leaving it alone. "
            "TransformerEngine layers may still be built."
        )
        return

    if spec.keywords.get("use_transformer_engine") is False:
        return  # already local, nothing to do

    model.transformer_layer_spec = partial(
        spec.func, *spec.args, **{**spec.keywords, "use_transformer_engine": False}
    )

    # The full-layer spec is a separate switch that also forces TE layers.
    if getattr(model, "use_transformer_engine_full_layer_spec", False):
        model.use_transformer_engine_full_layer_spec = False
        log_rank_0("[Patch:gemma4.local_spec] also cleared use_transformer_engine_full_layer_spec")

    log_rank_0(
        "[Patch:gemma4.local_spec] transformer_impl='local': rebound transformer_layer_spec "
        "to use_transformer_engine=False (the provider had frozen it to True)"
    )

    _disable_te_general_gemm()
    _use_torch_optimizer(container)


def _disable_te_general_gemm() -> None:
    """Route megatron's opportunistic TE GEMM call sites back to torch.

    The layer spec does not cover everything. Two call sites reach for TE's
    GEMM directly whenever the symbol merely *imports*, regardless of
    ``transformer_impl``:

      * ``moe/moe_utils.py:1326`` (router gating, fwd and bwd) --
        ``if te_general_gemm is not None and router_dtype != torch.float64``
      * ``tensor_parallel/layers.py:662-668``

    Both fall back to ``torch.mm`` / ``torch.addmm`` when the symbol is None, so
    setting it to None is the intended escape hatch rather than a hack. It is
    also the only one available here: TE on ROCm calls hipBLASLt directly and
    exposes no rocBLAS fallback, so on an image with broken hipBLASLt any TE
    GEMM is fatal no matter how the layers are built.

    Note moe_utils binds the symbol into its own namespace at import time, so
    patching only the extensions module would miss the router.
    """
    targets = [
        "megatron.core.extensions.transformer_engine",
        "megatron.core.transformer.moe.moe_utils",
    ]
    import sys

    for name in targets:
        module = sys.modules.get(name)
        if module is None:
            continue  # not imported yet; its own import-time guard will apply
        if getattr(module, "te_general_gemm", None) is not None:
            module.te_general_gemm = None
            log_rank_0(f"[Patch:gemma4.local_spec] disabled te_general_gemm in {name}")


def _use_torch_optimizer(container: Any) -> None:
    """Take megatron's Torch-optimizer branch instead of TE's FusedAdam.

    ``megatron/core/optimizer/__init__.py:13-35`` picks the Adam implementation
    at import time, preferring TE's FusedAdam, then Apex's, then Torch's. It
    records the choice in a module-level ``USING_PYTORCH_OPTIMIZER`` flag, and
    ``get_megatron_optimizer`` reads that flag rather than re-checking imports::

        if USING_PYTORCH_OPTIMIZER:
            adam_cls = torch.optim.AdamW if config.decoupled_weight_decay else torch.optim.Adam
        else:
            kwargs["adam_w_mode"] = config.decoupled_weight_decay
            adam_cls = Adam

    So flipping the flag is sufficient and is the same path a machine without TE
    or Apex takes. It also correctly skips the FusedAdam-specific master-weights
    handling at line 1117.

    Why bother: TE's FusedAdam hung the device. The optimizer step sat in
    ``multi_tensor_apply/__init__.py:23`` indefinitely at full GPU utilisation
    with an unchanging stack, while the driver logged ``MES(0, 0) ring buffer is
    full`` repeatedly -- multi_tensor_apply submits very many small operations,
    and the firmware queue could not drain them. A plain torch Adam over the same
    parameters completes promptly, so this is a hang rather than slow arithmetic.
    """
    import sys

    # Two independent things happen here, and they must not be coupled. Flipping
    # the module flag needs megatron.core.optimizer to be imported already;
    # clearing the config field needs only the container. Gating both on the
    # import would make the config change depend on patch-vs-import ordering,
    # which is not something this patch controls -- and a half-applied switch
    # (torch Adam requested, precision-aware still on) is worse than either
    # outcome, because precision-aware optimisation relies on the FusedAdam
    # master-weight path that is no longer there.
    module = sys.modules.get("megatron.core.optimizer")
    if module is None:
        log_rank_0(
            "[Patch:gemma4.local_spec] megatron.core.optimizer not imported yet; "
            "its own import-time choice applies. Clearing the config field regardless."
        )
    elif getattr(module, "USING_PYTORCH_OPTIMIZER", None) is not True:
        module.USING_PYTORCH_OPTIMIZER = True
        log_rank_0(
            "[Patch:gemma4.local_spec] forced USING_PYTORCH_OPTIMIZER=True "
            "(torch.optim.Adam instead of TE FusedAdam)"
        )

    # Precision-aware optimizer is a FusedAdam feature (it relies on the fused
    # kernel's master-weight support), so it cannot come along for the ride. It
    # lives on the optimizer section, not the model section.
    #
    # Unless CPU offload is in play, in which case HybridDeviceOptimizer supplies
    # the master weights itself and precision-aware is wanted -- clearing it there
    # would push the fp32 copies back onto the GPU, which is the memory that
    # offload exists to reclaim. Asking the environment rather than inspecting the
    # config keeps this independent of which of the two patches runs first.
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_cpu_offload import (
        offload_fraction,
    )

    if offload_fraction() is not None:
        log_rank_0(
            "[Patch:gemma4.local_spec] leaving use_precision_aware_optimizer alone; "
            "CPU offload is requested and provides its own master weights"
        )
        return

    opt_cfg = getattr(container, "optimizer", None)
    if opt_cfg is not None and getattr(opt_cfg, "use_precision_aware_optimizer", False):
        opt_cfg.use_precision_aware_optimizer = False
        log_rank_0(
            "[Patch:gemma4.local_spec] cleared use_precision_aware_optimizer "
            "(requires TE FusedAdam master weights)"
        )


@register_patch(
    "gemma4.local_spec",
    backend="megatron_bridge",
    phase="setup",
    description="Honour transformer_impl='local' by rebinding the TE-bound Gemma 4 layer spec",
)
def patch_gemma4_local_spec(ctx: PatchContext) -> None:
    try:
        from primus.backends.megatron_bridge import (
            megatron_bridge_posttrain_trainer,
            megatron_bridge_pretrain_trainer,
        )
    except Exception:
        return

    # Wraps load_recipe_config the same way gemma4.config.overrides does. That
    # patch registers first (import order in __init__), so it wraps first and
    # this wrapper sits outside it -- meaning PRIMUS_GEMMA4_SET has already been
    # applied by the time _rebind_spec reads transformer_impl. That ordering is
    # required: transformer_impl=local is normally set via PRIMUS_GEMMA4_SET.
    for module in (megatron_bridge_pretrain_trainer, megatron_bridge_posttrain_trainer):
        original = getattr(module, "load_recipe_config", None)
        if original is None or getattr(original, _PATCHED_ATTR, False):
            continue

        def load_recipe_config(backend_args, _original=original):
            container = _original(backend_args)
            _rebind_spec(container)
            return container

        setattr(load_recipe_config, _PATCHED_ATTR, True)
        module.load_recipe_config = load_recipe_config
