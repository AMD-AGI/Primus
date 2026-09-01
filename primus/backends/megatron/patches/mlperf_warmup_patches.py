###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MLPerf Warmup Patches for Flux Training.

Wraps megatron.training.training.train_step with a one-shot hook that runs
all warmup steps in a tight loop on the first invocation, then executes the
first real training step and self-removes.  This avoids relying on Megatron's
local ``iteration`` variable (which cannot be controlled from a train_step
wrapper) and mirrors NeMo's approach of running warmup before real data is
touched.

Priority 95 ensures this hook is the outermost wrapper around the full
train_step chain (FP8 cache, delayed scaling, wall-clock timer, etc.).
Self-removal restores the inner chain intact.
"""

import logging

import torch
import torch.distributed

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

logger = logging.getLogger(__name__)


def _log(msg):
    log_rank_0(f"[MLPerf_WARMUP] {msg}")


def _warmup_enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return args is not None and getattr(args, "warmup_train_steps", 0) > 0


def _reset_fp8_te_spec(models):
    """Reset FP8 state for TransformerEngine spec modules.

    Recipe-agnostic: walks ``fp8_meta`` and only touches buffers that exist on
    the recipe-specific state object. Skips TE's ``reset_fp8_meta_tensors``
    helper because TE 2.8.0.dev0 unconditionally derefs ``.scale`` /
    ``.amax_history``, which crashes on ``Float8CurrentScalingRecipeState``
    (current/tensorwise scaling has no persistent state — see TE's
    ``fp8.py``: *"Per-tensor current quantization does not require state"*).
    """
    count = 0
    for m in models:
        for module in m.modules():
            if not hasattr(module, "fp8_initialized"):
                continue
            module.fp8_initialized = False
            count += 1
            if not hasattr(module, "fp8_meta"):
                continue
            meta = module.fp8_meta
            for key in ("scaling_fwd", "scaling_bwd"):
                if key not in meta:
                    continue
                tm = meta[key]
                if hasattr(tm, "amax_history"):
                    tm.amax_history.fill_(0.0)
                if hasattr(tm, "scale"):
                    tm.scale.fill_(1.0)
                if hasattr(tm, "scale_inv"):
                    tm.scale_inv.fill_(1.0)
    return count


def _seed_fp8_amax(models, seed_value=1.0):
    """Seed FP8 amax_history with a safe non-zero value to prevent scale=inf."""
    count = 0
    for m in models:
        for module in m.modules():
            if not hasattr(module, "fp8_meta"):
                continue
            meta = module.fp8_meta
            for key in ("scaling_fwd", "scaling_bwd"):
                if key not in meta:
                    continue
                tm = meta[key]
                if hasattr(tm, "amax_history"):
                    tm.amax_history.fill_(seed_value)
                    count += 1
    return count


def _reset_fp8_local_spec(models):
    """Reset FP8 state for local spec (tensorwise delayed-scaling) modules.

    Re-initialises per-module delayed-scaling buffers via the canonical
    ``_init_delayed_scaling_state`` helper.  The new buffers are separate
    objects from the ``_DelayedScalingRegistry``'s global tensors, so the
    pointer-check in ``_fast_update_scales`` / ``_fast_update_scales_with_history``
    will detect the mismatch and trigger ``registry.__init__(modules)`` on the
    next training step, which re-creates global tensors with
    ``_first_step = True`` and bootstraps weight amaxes from the restored weights.

    Buffers are moved to the module's device because ``_init_delayed_scaling_state``
    creates plain CPU tensors (bare ``torch.zeros`` / ``torch.tensor``).
    """
    from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
        _init_delayed_scaling_state,
    )

    _DELAYED_BUF_NAMES = (
        "scale_input",
        "scale_weight",
        "scale_grad",
        "amax_history_input",
        "amax_history_weight",
        "amax_history_grad",
        "staged_input_amax",
        "staged_grad_amax",
        "staged_weight_amax",
    )

    count = 0
    for m in models:
        for module in m.modules():
            if not getattr(module, "_use_delayed_scaling", False):
                continue
            device = module.weight.device
            _init_delayed_scaling_state(module)
            for buf_name in _DELAYED_BUF_NAMES:
                buf = module._buffers.get(buf_name)
                if buf is not None and buf.device != device:
                    module._buffers[buf_name] = buf.to(device)
            count += 1
    return count


def _neuter_optimizer(optimizer):
    """Set optimizer to no-op mode: betas=[1,1], weight_decay=0."""
    saved = []
    inner = getattr(optimizer, "optimizer", optimizer)
    _log(
        f"Neutering optimizer: type={type(optimizer).__name__}, "
        f"inner={type(inner).__name__}, "
        f"param_groups={len(inner.param_groups)}"
    )
    for group in inner.param_groups:
        state = {}
        for key in ("betas", "weight_decay", "bias_correction", "pre_mult_wd"):
            if key in group:
                state[key] = group[key]
        saved.append(state)

        if "betas" in group:
            group["betas"] = [1.0, 1.0]
        if "weight_decay" in group:
            group["weight_decay"] = 0.0
        if "bias_correction" in group:
            group["bias_correction"] = False
        if "pre_mult_wd" in group:
            group["pre_mult_wd"] = 0.0
    return saved


def _restore_optimizer(optimizer, saved):
    """Restore optimizer hyperparams (betas, weight_decay, etc.)."""
    inner = getattr(optimizer, "optimizer", optimizer)
    for group, state in zip(inner.param_groups, saved):
        for key, val in state.items():
            group[key] = val
    _log("Restored optimizer parameters")


def _reset_optimizer_state(optimizer):
    """Zero per-parameter step counters so Adam acts as if no steps occurred.

    Handles both flat ``MegatronOptimizer`` wrappers and ``ChainedOptimizer``
    which wraps several sub-optimizers.  Step counts live either in
    ``param_groups[i]["step"]`` (Apex / TE FusedAdam) or in
    ``optimizer.state[p]["step"]`` (stock PyTorch Adam).
    """

    def _reset_single(opt):
        inner = getattr(opt, "optimizer", opt)
        for group in inner.param_groups:
            if "step" in group:
                group["step"] = 0
        for state in inner.state.values():
            if isinstance(state, dict) and "step" in state:
                if isinstance(state["step"], torch.Tensor):
                    state["step"].zero_()
                else:
                    state["step"] = 0

    if hasattr(optimizer, "chained_optimizers"):
        for sub_opt in optimizer.chained_optimizers:
            _reset_single(sub_opt)
    else:
        _reset_single(optimizer)
    _log("Reset optimizer step counters")


def _build_synthetic_iterator(primus_args):
    """Build the mock Flux dataloader the warmup steps consume."""
    from torch.utils.data import DataLoader

    from primus.backends.megatron.data.dataloader import MegatronDataloaderWrapper
    from primus.backends.megatron.data.synthetic.mock_datasets import (
        PreGeneratedMockFluxSchnellDataset,
    )

    image_size = getattr(primus_args, "image_size", 256)
    vae_latent_mode = getattr(primus_args, "vae_latent_mode", "resample")
    mbs = getattr(primus_args, "micro_batch_size", 64)

    mock_dataset = PreGeneratedMockFluxSchnellDataset(
        num_samples=max(mbs * 4, 256),
        image_size=image_size,
        vae_latent_mode=vae_latent_mode,
    )
    mock_loader = DataLoader(mock_dataset, batch_size=mbs, shuffle=False, drop_last=True)
    return MegatronDataloaderWrapper(mock_loader)


def _run_warmup_and_restore(
    *,
    warmup_steps,
    train_step_fn,
    forward_step_func,
    synthetic_iter,
    model,
    optimizer,
    opt_param_scheduler,
    config,
    forward_backward_func,
    iteration=None,
):
    """Run synthetic steps, then undo every effect they had on training state.

    The caller supplies ``train_step_fn`` so this works both from inside the
    train_step chain and from the pre-data boundary, where the chain has to be
    read at call time.
    """
    import megatron.training.training as mt
    from megatron.training import get_args as megatron_get_args

    megatron_args = megatron_get_args()
    models = model if isinstance(model, (list, tuple)) else [model]
    transformer_impl = getattr(megatron_args, "transformer_impl", "local")
    use_fsdp2_fp8 = getattr(megatron_args, "use_fsdp2_fp8_all_gather", False)

    # ---- 1. Snapshot model parameters to CPU ----
    _log("Saving model parameters to CPU before warmup")
    saved_params = {}
    for m in models:
        for name, p in m.named_parameters():
            saved_params[name] = p.data.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    _log(f"Saved {len(saved_params)} parameter tensors")

    # ---- 2. Neuter optimizer ----
    saved_opt = _neuter_optimizer(optimizer)

    # ---- 3. Suppress training_log and eval during warmup ----
    saved_training_log = mt.training_log
    saved_eval = mt.evaluate_and_print_results
    mt.training_log = lambda *a, **k: None
    mt.evaluate_and_print_results = lambda *a, **k: None

    # ---- 3b. Save LR scheduler state (NeMo never steps the scheduler during warmup) ----
    saved_lr_num_steps = opt_param_scheduler.num_steps

    # ---- 3c. Install Megatron's grad-finalize callback for the warmup steps ----
    # At the pre-data boundary `train()` has not run yet, so it has not assigned
    # config.finalize_model_grads_func. That callback is the only caller of
    # finish_grad_sync(), so without it a warmup backward dispatches the
    # data-parallel reduce-scatter and nothing ever waits on it. The handle is
    # still outstanding when the first real step dispatches its own, and
    # Megatron asserts "Should not have multiple communication calls
    # outstanding at once" before a single iteration completes.
    saved_finalize = config.finalize_model_grads_func
    if saved_finalize is None:
        from megatron.core.distributed import finalize_model_grads

        config.finalize_model_grads_func = finalize_model_grads
        _log("Installed finalize_model_grads_func for warmup (unset at this boundary)")

    # ---- 4. Run warmup steps with synthetic data ----
    try:
        for step_idx in range(warmup_steps):
            _log(f"Warmup step {step_idx + 1}/{warmup_steps}")
            train_step_fn(
                forward_step_func,
                synthetic_iter,
                model,
                optimizer,
                opt_param_scheduler,
                config,
                forward_backward_func,
                iteration=iteration,
            )
    finally:
        config.finalize_model_grads_func = saved_finalize
    _log(f"Completed {warmup_steps} warmup steps")

    # ---- 5. Restore optimizer ----
    _restore_optimizer(optimizer, saved_opt)
    _reset_optimizer_state(optimizer)

    # ---- 6. Restore model parameters from CPU ----
    restored = 0
    for m in models:
        for name, p in m.named_parameters():
            if name in saved_params:
                p.data.copy_(saved_params[name])
                restored += 1
    del saved_params
    _log(f"Restored {restored} parameter tensors from CPU snapshot")

    # ---- 7. FP8 reset (spec-aware) ----
    if transformer_impl == "transformer_engine":
        te_count = _reset_fp8_te_spec(models)
        amax_count = _seed_fp8_amax(models)
        _log(f"FP8 TE reset: {te_count} modules, " f"seeded {amax_count} amax tensors")
    else:
        local_count = _reset_fp8_local_spec(models)
        _log(f"FP8 local spec reset: {local_count} modules")

    # ---- 8. FSDP2 FP8 all-gather recompute ----
    if use_fsdp2_fp8:
        try:
            from primus.backends.megatron.core.distributed.fsdp2_fp8_all_gather import (
                precompute_fp8_scales_for_fsdp,
            )

            cache_data = getattr(megatron_args, "fp8_precompute_data_cache", True)
            use_cpp = getattr(megatron_args, "use_cpp_fp8_quantize", False)
            sr = getattr(megatron_args, "fp8_all_gather_stochastic_rounding", False)
            precompute_fp8_scales_for_fsdp(
                models[0],
                cache_data=cache_data,
                use_cpp_quantize=use_cpp,
                stochastic_rounding=sr,
            )
            _log("Recomputed FSDP2 FP8 all-gather scales")
        except Exception as e:
            _log(f"FSDP2 FP8 recompute failed (non-fatal): {e}")

    # ---- 9. Reload model params in optimizer (FSDP2 BF16 master weight) ----
    if hasattr(optimizer, "reload_model_params"):
        optimizer.reload_model_params()
        _log("Called optimizer.reload_model_params()")

    # ---- 10. Post-restore NaN check ----
    nan_params = 0
    for m in models:
        for name, p in m.named_parameters():
            if p.data.is_floating_point() and torch.isnan(p.data).any():
                nan_params += 1
    _log(f"Post-restore parameter check: nan_params={nan_params}")

    # ---- 11. Zero gradients ----
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()

    # ---- 12. Reset counters ----
    megatron_args.consumed_train_samples = 0
    megatron_args.skipped_train_samples = 0
    opt_param_scheduler.num_steps = saved_lr_num_steps
    _log(
        f"Reset consumed_train_samples=0, skipped_train_samples=0, "
        f"lr_scheduler.num_steps={saved_lr_num_steps}"
    )

    # ---- 13. Restore training_log and eval ----
    mt.training_log = saved_training_log
    mt.evaluate_and_print_results = saved_eval

    # ---- 13b. Invalidate the CudaPrefetchIterator that was built around
    # the SYNTHETIC iterator during warmup step 1.
    #
    # ``patch_grad_zero_and_data_prefetch`` builds a ``CudaPrefetchIterator``
    # the first time its ``_patched_train_step`` runs and caches it in a
    # closure-local ``_prefetch_state["iter"]``.  Because warmup step 1
    # is the first call into that train_step, the prefetch iterator gets
    # bound to ``synthetic_iter``.  ``MegatronDataloaderWrapper`` is
    # cyclic (never raises ``StopIteration``), so subsequent real
    # training steps would silently keep reading from the cycling
    # synthetic dataset instead of the actual training dataset -- model
    # overfits the mock samples and val_loss on real data stays stuck
    # at ~1.38 forever.
    #
    # Dropping the cached entry forces the next train_step to rebuild
    # the prefetch wrapper around its incoming ``data_iterator`` arg
    # (the real iterator).
    try:
        from primus.backends.megatron.patches.delayed_fp8_scaling_patches import (
            reset_prefetch_state,
        )

        evicted = reset_prefetch_state()
        if evicted is None:
            _log("  Prefetch reset: no cached iterator to evict")
        else:
            _log(
                f"  Prefetch reset: evicted cached {type(evicted).__name__} "
                f"(wrapped synthetic warmup iterator) -- next train_step "
                f"will rebuild it around the real data_iterator"
            )
    except Exception as _e:
        _log(f"  Prefetch reset failed (non-fatal): {_e}")

    # ---- 14. Synchronize ----
    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _install_boundary_warmup(primus_args, warmup_steps):
    """Run warmup at the pre-data boundary, inside the initialization window.

    Used in MLPerf mode, where warmup must finish before ``run_start`` and
    therefore before the data iterators exist. Everything the warmup needs is
    captured on the way there by :mod:`mlperf_boundary`.
    """
    from primus.backends.megatron.patches import mlperf_boundary

    def _warmup_hook():
        import megatron.training.training as mt

        captured = mlperf_boundary.captured()
        model = captured.get("model")
        optimizer = captured.get("optimizer")
        opt_param_scheduler = captured.get("opt_param_scheduler")
        forward_step_func = captured.get("forward_step_func")
        missing = [
            name
            for name, value in (
                ("model", model),
                ("optimizer", optimizer),
                ("opt_param_scheduler", opt_param_scheduler),
                ("forward_step_func", forward_step_func),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(
                "MLPerf warmup runs before the data iterators are built and needs "
                "objects captured from Megatron, but these were never captured: "
                + ", ".join(missing)
                + ". The capture wrappers in mlperf_boundary did not run."
            )

        models = model if isinstance(model, (list, tuple)) else [model]
        # Read train_step now, not at install time: every other before_train
        # patch has wrapped it by the time the boundary fires.
        _run_warmup_and_restore(
            warmup_steps=warmup_steps,
            train_step_fn=mt.train_step,
            forward_step_func=forward_step_func,
            synthetic_iter=_build_synthetic_iterator(primus_args),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            config=mt.get_model_config(models[0]),
            forward_backward_func=mt.get_forward_backward_func(),
            iteration=0,
        )

    mlperf_boundary.register_pre_run_hook("mlperf_warmup", _warmup_hook, order=10)
    mlperf_boundary.install()
    log_rank_0(
        f"[Patch:mlperf_warmup] Warmup registered at the pre-data boundary " f"(warmup_steps={warmup_steps})"
    )


def _install_train_step_warmup(mt, primus_args, warmup_steps):
    """Run warmup inside the first train_step, then execute the first real step.

    The non-MLPerf path. Warmup lands after the data iterators exist, which is
    fine when no clock is running, and it keeps development recipes on the
    behavior they were tuned against.
    """
    _wrapped_chain = mt.train_step
    _warmup_done = [False]
    _synthetic_iter = [None]

    def _hooked_train_step(
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        opt_param_scheduler,
        config,
        forward_backward_func,
        iteration=None,
    ):
        if _warmup_done[0]:
            return _wrapped_chain(
                forward_step_func,
                data_iterator,
                model,
                optimizer,
                opt_param_scheduler,
                config,
                forward_backward_func,
                iteration=iteration,
            )

        if _synthetic_iter[0] is None:
            _synthetic_iter[0] = _build_synthetic_iterator(primus_args)

        _run_warmup_and_restore(
            warmup_steps=warmup_steps,
            train_step_fn=_wrapped_chain,
            forward_step_func=forward_step_func,
            synthetic_iter=_synthetic_iter[0],
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            config=config,
            forward_backward_func=forward_backward_func,
            iteration=iteration,
        )

        _log("Executing first real train_step with training data")
        result = _wrapped_chain(
            forward_step_func,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            config,
            forward_backward_func,
            iteration=iteration,
        )

        _warmup_done[0] = True
        mt.train_step = _wrapped_chain
        _log("Self-removed warmup hook, train_step = inner wrapped chain")

        return result

    _hooked_train_step._primus_warmup_hook = True
    mt.train_step = _hooked_train_step

    log_rank_0(f"[Patch:mlperf_warmup] Installed warmup hook " f"(warmup_steps={warmup_steps}, priority=95)")


@register_patch(
    "megatron.training.mlperf_warmup",
    backend="megatron",
    phase="before_train",
    description="MLPerf warmup: synthetic data steps before measured training",
    condition=_warmup_enabled,
    priority=95,
)
def patch_mlperf_warmup(ctx: PatchContext):
    """Install warmup at priority 95, so it is the outermost wrapper."""
    import megatron.training.training as mt

    if hasattr(mt.train_step, "_primus_warmup_hook"):
        return

    primus_args = get_args(ctx)
    warmup_steps = getattr(primus_args, "warmup_train_steps", 2)

    if getattr(primus_args, "mlperf_mode", False):
        _install_boundary_warmup(primus_args, warmup_steps)
    else:
        _install_train_step_warmup(mt, primus_args, warmup_steps)
