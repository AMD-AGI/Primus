###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Patch ``get_megatron_optimizer`` to install MXFP4 weight de-oscillation.

When MXFP4 training is driven through the Primus-Turbo FP4 autocast path and
``weight_deosc`` is enabled, this patch wraps every
``DistributedOptimizer.step_with_ready_grads`` so the de-oscillation detector
runs right after the optimizer updates the fp32 master and all-gathers the bf16
model weight. See ``primus.backends.megatron.core.optimizer.weight_deosc`` for
the algorithm and rationale.
"""

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _is_weight_deosc_can_patch(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    fp4 = bool(getattr(args, "fp4", False))
    use_turbo_fp4 = bool(getattr(args, "use_turbo_fp4_autocast", False))
    deosc = bool(getattr(args, "weight_deosc", False))
    return fp4 and use_turbo_fp4 and deosc and is_primus_turbo_can_patch(ctx)


@register_patch(
    "megatron.turbo.weight_deosc",
    backend="megatron",
    phase="before_train",
    description="Install MXFP4 weight de-oscillation on the distributed optimizer.",
    condition=_is_weight_deosc_can_patch,
)
def patch_get_megatron_optimizer_weight_deosc(ctx: PatchContext) -> None:
    try:
        import megatron.training.training as training_module
    except ImportError as e:
        log_rank_0(f"[Patch:megatron.turbo.weight_deosc] Skip (Megatron not available): {e}")
        return

    from primus.backends.megatron.core.optimizer.weight_deosc import (
        WeightDeOscConfig,
        install_weight_deosc,
    )

    args = get_args(ctx)
    config = WeightDeOscConfig(
        enable=True,
        period=int(getattr(args, "weight_deosc_period", 200)),
        ratio_threshold=float(getattr(args, "weight_deosc_ratio", 4.0)),
        start_step=int(getattr(args, "weight_deosc_start_step", 0)),
        log_freq=int(getattr(args, "weight_deosc_log_freq", 0)),
    )

    original_get_megatron_optimizer = training_module.get_megatron_optimizer

    if getattr(original_get_megatron_optimizer, "_primus_weight_deosc_wrapper", False):
        return

    def _patched_get_megatron_optimizer(*func_args, **func_kwargs):
        optimizer = original_get_megatron_optimizer(*func_args, **func_kwargs)
        try:
            install_weight_deosc(optimizer, config)
        except Exception as exc:  # never block optimizer construction
            log_rank_0(f"[Patch:megatron.turbo.weight_deosc] install failed, skipped: {exc}")
        return optimizer

    setattr(_patched_get_megatron_optimizer, "_primus_weight_deosc_wrapper", True)
    training_module.get_megatron_optimizer = _patched_get_megatron_optimizer
    log_rank_0(
        "[Patch:megatron.turbo.weight_deosc] Patched get_megatron_optimizer to install "
        f"MXFP4 weight de-oscillation (period={config.period}, ratio={config.ratio_threshold}, "
        f"start_step={config.start_step})."
    )

    _install_checkpoint_persistence()


def _checkpoint_iter_dir(checkpoints_path, iteration, release: bool = False):
    """Return the common (rank-independent) checkpoint directory for an iteration."""
    if not checkpoints_path:
        return None
    try:
        from megatron.training.checkpointing import get_checkpoint_name

        return get_checkpoint_name(checkpoints_path, iteration, release=release, return_base_dir=True)
    except Exception:
        directory = "release" if release else "iter_{:07d}".format(int(iteration))
        import os

        return os.path.join(checkpoints_path, directory)


def _install_checkpoint_persistence() -> None:
    """Wrap save_checkpoint / load_checkpoint to persist de-oscillation state.

    Writes a per-rank sidecar (``<ckpt_dir>/weight_deosc/rank_<R>.pt``) that is
    independent of the checkpoint format (works for both legacy and torch_dist).
    All work is best-effort and guarded so checkpointing can never break.
    """
    try:
        from megatron.training.global_vars import get_args
    except Exception:
        return

    from primus.backends.megatron.core.optimizer.weight_deosc import (
        load_deosc_sidecars,
        save_deosc_sidecars,
    )

    def _wrap_save(orig):
        if orig is None or getattr(orig, "_primus_deosc_ckpt_wrapper", False):
            return orig

        def _wrapped(iteration, model, optimizer, *args, **kwargs):
            ret = orig(iteration, model, optimizer, *args, **kwargs)
            try:
                a = get_args()
                ckpt_dir = _checkpoint_iter_dir(
                    getattr(a, "save", None), iteration, release=bool(kwargs.get("release", False))
                )
                save_deosc_sidecars(optimizer, ckpt_dir)
            except Exception as exc:
                log_rank_0(f"[WeightDeOsc] sidecar save skipped: {exc}")
            return ret

        setattr(_wrapped, "_primus_deosc_ckpt_wrapper", True)
        return _wrapped

    def _wrap_load(orig):
        if orig is None or getattr(orig, "_primus_deosc_ckpt_wrapper", False):
            return orig

        def _wrapped(ddp_model, optimizer, opt_param_scheduler, load_arg="load", *args, **kwargs):
            ret = orig(ddp_model, optimizer, opt_param_scheduler, load_arg, *args, **kwargs)
            try:
                a = get_args()
                iteration = ret[0] if isinstance(ret, (tuple, list)) and ret else getattr(a, "iteration", 0)
                ckpt_dir = _checkpoint_iter_dir(getattr(a, load_arg, None), iteration)
                load_deosc_sidecars(optimizer, ckpt_dir)
            except Exception as exc:
                log_rank_0(f"[WeightDeOsc] sidecar load skipped: {exc}")
            return ret

        setattr(_wrapped, "_primus_deosc_ckpt_wrapper", True)
        return _wrapped

    # Patch every namespace that holds a reference to these functions: the
    # Primus trainer imports them by value, and save_checkpoint_and_time uses
    # megatron.training.training's reference for periodic saves.
    import importlib

    for mod_name in (
        "megatron.training.training",
        "primus.modules.trainer.megatron.trainer",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, "save_checkpoint"):
            mod.save_checkpoint = _wrap_save(mod.save_checkpoint)
        if hasattr(mod, "load_checkpoint"):
            mod.load_checkpoint = _wrap_load(mod.load_checkpoint)

    log_rank_0("[Patch:megatron.turbo.weight_deosc] Installed de-oscillation checkpoint persistence.")
