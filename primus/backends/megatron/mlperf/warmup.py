###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Synthetic-data warmup for Primus / Megatron training.

Adds optional FP4 (MXFP4 / NVFP4) warmup support on top of the original FP8
warmup helpers.  The original FP8 / BF16 path is preserved exactly:

  * No autocast wrapping by default — ``train_step`` runs at the model's
    native precision, identical to the committed pre-FP4 behaviour.
  * After warmup, ``reset_fp8_state`` + ``seed_fp8_amax(1.0)`` always runs
    (no-op for BF16 / non-FP8 modules).
  * ``ChainedOptimizer`` support (EP < num_gpus) is preserved.

FP4 mode is **opt-in** via ``WARMUP_RECIPE=fp4_mxfp4`` / ``fp4_nvfp4``.  When
set, the warmup loop is wrapped in ``te.fp8_autocast(fp8_recipe=...)`` and
an additional ``reset_fp4_state()`` pass runs after warmup.

Runs N forward+backward passes with random token sequences before the real
training loop starts.  This pre-compiles Triton / CK / hipBLASLt kernels and
amortizes:

  * distributed-optimizer FP32 main-param allocation
  * DDP gradient-bucket allocation
  * NCCL communicator init + collective autotune
  * hipBLASLt heuristic / Triton / CK JIT caches (recipe-dependent)
  * PyTorch HIP allocator block layout

After warmup:
  * Model parameters are restored from a pre-warmup snapshot.
  * Optimizer state is restored (neutered during warmup so weights never move).
  * Adam buffers (exp_avg / exp_avg_sq) are zeroed and grads cleared.
  * LR scheduler state is rolled back and re-synced (param_groups['lr']).
  * FP8 scaling state (amax_history, scale, scale_inv, fp8_initialized) is
    fully reset and seeded with safe defaults.
  * If WARMUP_RECIPE selected an FP4 recipe, FP4 state is also reset.

Controlled by:

  SYNTH_WARMUP_STEPS         default 3,  0 disables.
  WARMUP_RECIPE              default "" (no autocast, original behaviour),
                             one of:
                               "" | "bf16"
                               | "fp8_hybrid" | "fp8_e4m3"
                               | "fp4_mxfp4"  | "fp4_nvfp4"
                             Legacy alias WARMUP_FP8_RECIPE=hybrid|e4m3 also
                             accepted (maps to fp8_hybrid / fp8_e4m3).
                             FP8/BF16 models do NOT need to set this; it
                             only affects which autocast (if any) wraps the
                             warmup train_step.
  WARMUP_FP8_HISTORY_LEN     default 4, only used for fp8_* recipes.
  SYNTH_WARMUP_EMPTY_CACHE   default 1, call torch.cuda.empty_cache() per step.
"""

import os
import time
from contextlib import nullcontext

import torch
import torch.distributed

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log(msg):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        print(f"[SYNTH_WARMUP] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


class SyntheticGPTDataIterator:
    """Infinite iterator yielding random token batches for Primus/Megatron GPT.

    Primus's ``get_batch_on_this_tp_rank`` expects the iterator to yield a dict
    with ``tokens``, ``labels``, ``loss_mask``, and ``position_ids`` tensors,
    each of shape ``[mbs, seq_length]``.
    """

    def __init__(self, seq_length, micro_batch_size, vocab_size=32000):
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.vocab_size = vocab_size

    def __iter__(self):
        return self

    def __next__(self):
        mbs = self.micro_batch_size
        sl = self.seq_length
        tokens = torch.randint(0, self.vocab_size, (mbs, sl), dtype=torch.int64)
        labels = torch.randint(0, self.vocab_size, (mbs, sl), dtype=torch.int64)
        loss_mask = torch.ones(mbs, sl, dtype=torch.float32)
        position_ids = torch.arange(sl, dtype=torch.int64).unsqueeze(0).expand(mbs, -1)
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }


# ---------------------------------------------------------------------------
# FP8 state management  --  ORIGINAL from committed warmup.py.
#
# The only change vs. the committed version is one defensive line at the top
# of ``_is_delayed_scaling_recipe`` that returns False for FP4-flagged
# modules so the FP8 reset path skips them (handled by reset_fp4_state()
# instead).  No behavioural change for FP8 / BF16 / current-scaling models.
# ---------------------------------------------------------------------------


def _is_delayed_scaling_recipe(module):
    """Return True if the module uses delayed scaling (which has scale/amax_history).

    Current scaling (tensorwise) is stateless — no scale or amax_history to
    manage — so reset / seed operations must be skipped for it.
    FP4 (MXFP4 / NVFP4) modules also have no scale/amax_history; they are
    skipped here and handled by reset_fp4_state() if WARMUP_RECIPE is fp4_*.
    Returns True when recipe type cannot be determined (safe default).
    """
    # Defensive: FP4 modules use a different recipe state class with no
    # scale/amax_history. Calling FP8 helpers on them is at best a no-op
    # and at worst raises AttributeError.
    if bool(getattr(module, "fp4", False)) or hasattr(module, "fp4_initialized"):
        return False

    from transformer_engine.common.recipe import DelayedScaling

    fp8_meta = getattr(module, "fp8_meta", None)
    if fp8_meta is None:
        return True
    fwd_state = fp8_meta.get("scaling_fwd", None)
    if fwd_state is None:
        return True
    return isinstance(getattr(fwd_state, "recipe", None), DelayedScaling)


def _manual_reset_fp8_meta(module):
    """Reset FP8 amax / scale tensors when reset_fp8_meta_tensors() is absent."""
    if not hasattr(module, "fp8_meta"):
        return False
    if not _is_delayed_scaling_recipe(module):
        return False
    meta = module.fp8_meta
    reset_count = 0
    for key in ("scaling_fwd", "scaling_bwd"):
        if key not in meta:
            continue
        tensor_meta = meta[key]
        if hasattr(tensor_meta, "amax_history"):
            tensor_meta.amax_history.fill_(0.0)
            reset_count += 1
        if hasattr(tensor_meta, "scale"):
            tensor_meta.scale.fill_(1.0)
            reset_count += 1
        if hasattr(tensor_meta, "scale_inv"):
            tensor_meta.scale_inv.fill_(1.0)
            reset_count += 1
    return reset_count > 0


def reset_fp8_state(model, reset_meta_tensors=True):
    """Clear ``fp8_initialized`` on every TE layer, forcing re-init.

    When *reset_meta_tensors* is True, also zeros out amax_history and
    resets scale / scale_inv to 1.0.

    Skips FP4 modules (handled by reset_fp4_state() when WARMUP_RECIPE=fp4_*).
    """
    count = 0
    method_count = 0
    manual_count = 0

    def _reset(m):
        nonlocal count, method_count, manual_count
        if hasattr(m, "fp8_initialized"):
            m.fp8_initialized = False
            count += 1
            if reset_meta_tensors:
                if hasattr(m, "reset_fp8_meta_tensors"):
                    if _is_delayed_scaling_recipe(m):
                        m.reset_fp8_meta_tensors()
                        method_count += 1
                elif _manual_reset_fp8_meta(m):
                    manual_count += 1

    models = model if isinstance(model, (list, tuple)) else [model]
    for m in models:
        m.apply(_reset)
    _log(f"reset_fp8_state: {count} modules, " f"{method_count} via method, {manual_count} via manual reset")
    return count


def seed_fp8_amax(model, seed_value=1.0):
    """Fill amax_history with *seed_value* to prevent scale=inf after reset.

    Delayed scaling computes ``scale = fp8_max / max(amax_history)``.
    If amax_history is all-zero after reset, scale becomes inf -> NaN.
    Seeding with 1.0 gives scale = fp8_max / 1.0 ~ 448 (E4M3), which is safe.
    """
    count = 0

    def _seed(m):
        nonlocal count
        if not hasattr(m, "fp8_meta"):
            return
        if not _is_delayed_scaling_recipe(m):
            return
        meta = m.fp8_meta
        for key in ("scaling_fwd", "scaling_bwd"):
            if key not in meta:
                continue
            tensor_meta = meta[key]
            if hasattr(tensor_meta, "amax_history"):
                tensor_meta.amax_history.fill_(seed_value)
                count += 1

    models = model if isinstance(model, (list, tuple)) else [model]
    for m in models:
        m.apply(_seed)
    _log(f"seed_fp8_amax: seeded {count} amax_history tensors with {seed_value}")
    return count


# ===========================================================================
# FP4 state management  --  NEW, only invoked when WARMUP_RECIPE=fp4_*
# ===========================================================================
#
# Notes on FP4 vs FP8 in TE:
#   * MXFP4BlockScaling and NVFP4BlockScaling derive scales **per tile** from
#     the current activation/weight on every forward — there is no
#     amax_history and no global scale-inv to bias.  So no seeding step.
#   * Some TE versions store FP4 quantizer caches under fp8_meta as well
#     (under attrs like ``block_scales``, ``mxfp4_quantizer``, etc.).  We
#     try to clear those defensively; they're allowed to be missing.
#   * Different TE versions may track init via ``fp4_initialized``,
#     ``fp8_initialized`` (shared with FP8), or both.  We touch whichever
#     attribute exists.

# Per-tensor-meta attribute names that may hold FP4-specific cached state.
# Cleared opportunistically if present; missing attrs are silently skipped.
_FP4_META_ATTRS = (
    "block_scales",
    "block_scales_inv",
    "mxfp4_block_scale",
    "fp4_quantizer_state",
    "fp4_amax",
)


def _is_fp4_module(m):
    """True if a TE module is configured for FP4 (MXFP4 / NVFP4).

    FP4 modules in TE share ``fp8_initialized`` and ``fp8_meta`` with FP8 but
    populate them with an FP4 recipe state (``MXFP4BlockScalingRecipeState``,
    ``NVFP4BlockScalingRecipeState``) that has no ``scale``/``amax_history``.
    """
    if bool(getattr(m, "fp4", False)):
        return True
    if hasattr(m, "fp4_initialized"):
        return True
    meta = getattr(m, "fp8_meta", None)
    if isinstance(meta, dict):
        for key in ("scaling_fwd", "scaling_bwd"):
            tm = meta.get(key)
            if tm is None:
                continue
            cls = type(tm).__name__
            if "FP4" in cls or "Fp4" in cls:
                return True
    return False


def _manual_reset_fp4_meta(module):
    """Reset FP4 per-tile block-scale buffers when reset_fp4_meta_tensors() is absent."""
    if not hasattr(module, "fp8_meta"):
        return False
    meta = module.fp8_meta
    reset_count = 0
    for key in ("scaling_fwd", "scaling_bwd"):
        if key not in meta:
            continue
        tensor_meta = meta[key]
        for attr in _FP4_META_ATTRS:
            if not hasattr(tensor_meta, attr):
                continue
            t = getattr(tensor_meta, attr)
            if hasattr(t, "zero_"):
                try:
                    t.zero_()
                    reset_count += 1
                except Exception:
                    pass
    return reset_count > 0


def reset_fp4_state(model, reset_meta_tensors=True):
    """Clear ``fp4_initialized`` (or shared ``fp8_initialized``) on FP4 layers.

    Parallel to :func:`reset_fp8_state` but for FP4 modules.  Unlike FP8 this
    does NOT seed any history — block-scaling recipes have no amax_history,
    so TE will recompute per-tile scales from real data on the next forward.

    Returns the number of modules touched.
    """
    count = 0
    method_count = 0
    manual_count = 0

    def _reset(m):
        nonlocal count, method_count, manual_count
        if not _is_fp4_module(m):
            return

        touched = False
        if hasattr(m, "fp4_initialized"):
            m.fp4_initialized = False
            touched = True
        if hasattr(m, "fp8_initialized"):
            m.fp8_initialized = False
            touched = True
        if touched:
            count += 1

        if reset_meta_tensors:
            if hasattr(m, "reset_fp4_meta_tensors"):
                try:
                    m.reset_fp4_meta_tensors()
                    method_count += 1
                except Exception:
                    pass
            elif hasattr(m, "reset_fp8_meta_tensors"):
                # FP4 modules typically share the meta-reset method with FP8.
                try:
                    m.reset_fp8_meta_tensors()
                    method_count += 1
                except Exception:
                    if _manual_reset_fp4_meta(m):
                        manual_count += 1
            elif _manual_reset_fp4_meta(m):
                manual_count += 1

    models = model if isinstance(model, (list, tuple)) else [model]
    for m in models:
        m.apply(_reset)
    _log(f"reset_fp4_state: {count} modules, " f"{method_count} via method, {manual_count} via manual reset")
    return count


# ===========================================================================
# Warmup recipe selection  --  NEW, opt-in via WARMUP_RECIPE
# ===========================================================================


def _resolve_recipe_str():
    """Return the warmup recipe string, honouring legacy WARMUP_FP8_RECIPE.

    Default is "" (empty) → no autocast wrapping, identical to the original
    pre-FP4 warmup.  FP8/BF16 users do not need to set anything.
    """
    val = os.getenv("WARMUP_RECIPE", "").strip().lower()
    if val:
        return val
    legacy = os.getenv("WARMUP_FP8_RECIPE", "").strip().lower()
    if legacy in ("hybrid", "e4m3"):
        return f"fp8_{legacy}"
    return ""


def _build_warmup_recipe():
    """Return a TE recipe object if WARMUP_RECIPE selects one, else None.

    Supports FP8 (DelayedScaling) and FP4 (MXFP4BlockScaling, NVFP4BlockScaling).
    Both kinds are consumed by ``te.fp8_autocast(fp8_recipe=...)``.

    Returns
    -------
    (recipe, kind) where kind is one of ``"fp8"``, ``"fp4"`` or ``None``.
    """
    recipe_str = _resolve_recipe_str()
    if not recipe_str or recipe_str == "bf16":
        if recipe_str == "bf16":
            _log("WARMUP_RECIPE=bf16: no autocast wrapping")
        return None, None

    # ---- FP8 family ------------------------------------------------------
    if recipe_str.startswith("fp8_"):
        try:
            from transformer_engine.common.recipe import DelayedScaling, Format
        except ImportError:
            _log(f"WARMUP_RECIPE={recipe_str!r} but transformer_engine missing; no autocast")
            return None, None
        fmt = {"fp8_hybrid": Format.HYBRID, "fp8_e4m3": Format.E4M3}.get(recipe_str)
        if fmt is None:
            _log(f"Unknown FP8 WARMUP_RECIPE={recipe_str!r}; no autocast")
            return None, None
        history_len = int(os.getenv("WARMUP_FP8_HISTORY_LEN", "4"))
        _log(
            f"Warmup will wrap train_step in fp8_autocast"
            f"(format={recipe_str[4:]}, amax_history_len={history_len}, algo=most_recent)"
        )
        return (
            DelayedScaling(
                margin=0,
                fp8_format=fmt,
                amax_history_len=history_len,
                amax_compute_algo="most_recent",
            ),
            "fp8",
        )

    # ---- FP4 family ------------------------------------------------------
    if recipe_str.startswith("fp4_"):
        try:
            import transformer_engine.common.recipe as te_recipe
        except ImportError:
            _log(f"WARMUP_RECIPE={recipe_str!r} but transformer_engine missing; no autocast")
            return None, None

        # Try to nudge Primus's MXFP4 recipe-state patch into place if the
        # model was built without FP4 (so the patch never ran).  No-op otherwise.
        try:
            from primus.backends.megatron.core.fp4_utils import (
                _ensure_mxfp4_recipe_support,
            )

            _ensure_mxfp4_recipe_support()
        except Exception:
            pass

        cls_name = {
            "fp4_mxfp4": "MXFP4BlockScaling",
            "fp4_nvfp4": "NVFP4BlockScaling",
        }.get(recipe_str)
        if cls_name is None:
            _log(f"Unknown FP4 WARMUP_RECIPE={recipe_str!r}; no autocast")
            return None, None

        recipe_cls = getattr(te_recipe, cls_name, None)
        if recipe_cls is None:
            _log(f"WARMUP_RECIPE={recipe_str!r} but {cls_name} not in this TE build; no autocast")
            return None, None
        try:
            recipe = recipe_cls()
        except Exception as e:
            _log(f"Failed to construct {cls_name}() for warmup: {e}; no autocast")
            return None, None
        _log(f"Warmup will wrap train_step in fp8_autocast(fp8_recipe={cls_name}())")
        return recipe, "fp4"

    _log(f"Unknown WARMUP_RECIPE={recipe_str!r}; no autocast")
    return None, None


def _warmup_autocast(recipe):
    """Return a context-manager factory for the warmup loop."""
    if recipe is None:
        return nullcontext
    import transformer_engine.pytorch as te

    def _ctx():
        return te.fp8_autocast(enabled=True, fp8_recipe=recipe)

    return _ctx


# ---------------------------------------------------------------------------
# Optimizer save / restore  --  ORIGINAL from committed warmup.py
# (preserves ChainedOptimizer support for EP < num_gpus configs)
# ---------------------------------------------------------------------------


def _get_inner_optimizers(optimizer):
    """Return a list of leaf-level torch optimizers from any Megatron wrapper.

    When expert_parallel < num_gpus, Megatron creates a ChainedOptimizer with
    separate sub-optimizers for dense and expert parameters.  The previous code
    used ``optimizer.optimizer`` which asserts ``len == 1`` on ChainedOptimizer.
    This helper iterates over all chained sub-optimizers when present, and falls
    back to the single-optimizer path otherwise.
    """
    if hasattr(optimizer, "chained_optimizers"):
        inners = []
        for sub_opt in optimizer.chained_optimizers:
            inners.append(getattr(sub_opt, "optimizer", sub_opt))
        return inners
    return [getattr(optimizer, "optimizer", optimizer)]


def _neuter_optimizer(optimizer):
    """Set betas=[1,1], weight_decay=0, bias_correction=False.

    With betas=[1,1] the Adam momentum/variance stays at its initial (zero)
    value, so the effective weight update is zero.

    Handles ChainedOptimizer by iterating over all sub-optimizers.
    """
    all_saved = []
    for inner in _get_inner_optimizers(optimizer):
        saved = []
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
        all_saved.append(saved)
    return all_saved


def _restore_optimizer(optimizer, all_saved):
    """Restore optimizer state, handling ChainedOptimizer."""
    for inner, saved in zip(_get_inner_optimizers(optimizer), all_saved):
        for group, state in zip(inner.param_groups, saved):
            for key, val in state.items():
                group[key] = val
            if "step" in group:
                del group["step"]


# ---------------------------------------------------------------------------
# Model parameter snapshot  --  ORIGINAL from committed warmup.py, unchanged
# ---------------------------------------------------------------------------


def _save_model_params(models):
    """Snapshot all model parameters to CPU to avoid doubling GPU memory."""
    saved = {}
    for m in models:
        for name, p in m.named_parameters():
            saved[(id(m), name)] = p.data.to("cpu", copy=True)
    return saved


def _restore_model_params(models, saved):
    restored = 0
    for m in models:
        for name, p in m.named_parameters():
            key = (id(m), name)
            if key in saved:
                p.data.copy_(saved[key].to(p.device))
                restored += 1
    return restored


# ---------------------------------------------------------------------------
# LR scheduler save / restore  --  ORIGINAL + step(0) re-sync after restore
# ---------------------------------------------------------------------------

_SCHEDULER_KEYS = (
    "num_steps",
    "num_floating_point_operations_so_far",
)


def _save_scheduler_state(scheduler):
    if scheduler is None:
        return None
    return {k: getattr(scheduler, k) for k in _SCHEDULER_KEYS if hasattr(scheduler, k)}


def _restore_scheduler_state(scheduler, state):
    if scheduler is None or state is None:
        return
    for k, v in state.items():
        setattr(scheduler, k, v)
    # Re-sync param_groups['lr'] and ['weight_decay'] to match the restored
    # num_steps. Without this, the optimizer still holds the LR from the last
    # warmup step even though num_steps has been rewound to its pre-warmup value.
    scheduler.step(0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_synthetic_warmup(
    train_step_func,
    forward_step_func,
    model,
    optimizer,
    opt_param_scheduler,
    config,
    megatron_args,
):
    """Run *warmup_steps* forward+backward passes with synthetic data.

    Called from ``MLPerfMegatronPretrainTrainer``'s patched ``train()``
    **before** ``RUN_START`` is logged, so warmup time is excluded from the
    timed run.  All side-effects (model params, optimizer, LR scheduler,
    FP8/FP4 state) are reverted after warmup.

    Default behaviour (``WARMUP_RECIPE`` unset): no autocast wrapping,
    identical to the pre-FP4 committed version.  Set
    ``WARMUP_RECIPE=fp4_mxfp4`` or ``fp4_nvfp4`` to opt into FP4 warmup.

    Args:
        train_step_func: The (already MLPerf-patched) upstream
            ``megatron.training.training.train_step`` function.  It is called
            with the new-style signature
            ``(forward_step_func, data_iterator, model, optimizer,
            opt_param_scheduler, config, forward_backward_func, iteration=0)``.
            ``forward_backward_func`` is resolved here via
            ``get_forward_backward_func()`` (mirrors ``training.train()``).
    """
    warmup_steps = int(os.getenv("SYNTH_WARMUP_STEPS", "3"))
    if warmup_steps <= 0:
        _log(f"Skipped (SYNTH_WARMUP_STEPS={warmup_steps})")
        return

    empty_cache_each_step = os.getenv("SYNTH_WARMUP_EMPTY_CACHE", "1") not in ("0", "false", "False")

    t0 = time.time()
    _log(
        f"Starting {warmup_steps}-step synthetic warmup "
        f"(seq_len={megatron_args.seq_length}, "
        f"mbs={megatron_args.micro_batch_size}, "
        f"empty_cache_each_step={empty_cache_each_step})"
    )

    vocab_size = getattr(
        megatron_args,
        "padded_vocab_size",
        getattr(megatron_args, "vocab_size", 32000),
    )
    synth_iter = SyntheticGPTDataIterator(
        megatron_args.seq_length,
        megatron_args.micro_batch_size,
        vocab_size,
    )

    models = model if isinstance(model, (list, tuple)) else [model]

    # Resolve the forward-backward func the same way upstream
    # ``megatron.training.training.train()`` does (see training.py).  Newer
    # Megatron ``train_step`` takes this as an explicit positional argument.
    from megatron.core.pipeline_parallel import get_forward_backward_func

    forward_backward_func = get_forward_backward_func()

    # Temporarily set config fields needed by forward_backward_func.
    # Megatron's train() normally sets these, but warmup runs before it.
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel as DDP,
    )

    saved_config = {}
    for key in ("finalize_model_grads_func", "grad_scale_func", "no_sync_func"):
        saved_config[key] = getattr(config, key, None)
    if config.finalize_model_grads_func is None:
        config.finalize_model_grads_func = finalize_model_grads
    if config.grad_scale_func is None:
        config.grad_scale_func = optimizer.scale_loss
    if megatron_args.overlap_grad_reduce and config.no_sync_func is None:
        if isinstance(models[0], DDP):
            config.no_sync_func = models[0].no_sync if len(models) == 1 else [m.no_sync for m in models]

    # ---- save state ------------------------------------------------------
    _log(f"Saving {sum(p.numel() for m in models for p in m.parameters())} parameters")
    saved_params = _save_model_params(models)
    saved_opt = _neuter_optimizer(optimizer)
    saved_sched = _save_scheduler_state(opt_param_scheduler)

    # ---- decide warmup precision (default: no autocast = original path) -
    recipe, recipe_kind = _build_warmup_recipe()
    warmup_ctx_factory = _warmup_autocast(recipe)

    # ---- warmup steps ----------------------------------------------------
    for step in range(1, warmup_steps + 1):
        step_t0 = time.time()
        with warmup_ctx_factory():
            train_step_func(
                forward_step_func,
                synth_iter,
                model,
                optimizer,
                opt_param_scheduler,
                config,
                forward_backward_func,
                iteration=0,
            )
        torch.cuda.synchronize()
        if empty_cache_each_step:
            torch.cuda.empty_cache()
        _log(f"Step {step}/{warmup_steps} done in {time.time() - step_t0:.1f}s")

    # ---- restore state ---------------------------------------------------
    _log("Restoring optimizer")
    _restore_optimizer(optimizer, saved_opt)

    _log("Restoring LR scheduler")
    _restore_scheduler_state(opt_param_scheduler, saved_sched)

    _log("Restoring model parameters")
    n_restored = _restore_model_params(models, saved_params)
    del saved_params
    _log(f"Restored {n_restored} parameter tensors")

    if hasattr(optimizer, "reload_model_params"):
        optimizer.reload_model_params()
        _log("Called optimizer.reload_model_params()")

    # ---- zero Adam state buffers (defensive) ----------------------------
    # With betas=[1,1] during warmup, exp_avg / exp_avg_sq should already be
    # zero, but zero them explicitly in case any param group lacked betas.
    # Iterate over chained sub-optimizers for ChainedOptimizer support.
    zeroed_state_tensors = 0
    for inner_opt in _get_inner_optimizers(optimizer):
        for param_states in inner_opt.state.values():
            for k, v in param_states.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    v.zero_()
                    zeroed_state_tensors += 1
    _log(f"Zeroed {zeroed_state_tensors} optimizer state tensors (exp_avg / exp_avg_sq)")

    for m in models:
        m.zero_grad(set_to_none=True)
    _log("Zeroed all model gradients")

    # ---- reset FP8 (always; no-op for BF16 / non-FP8 modules) -----------
    _log("Resetting FP8 state")
    total_reset = 0
    for m in models:
        total_reset += reset_fp8_state(m, reset_meta_tensors=True)
    seed_fp8_amax(models, seed_value=1.0)

    # ---- reset FP4 (only when warmup explicitly used an FP4 recipe) -----
    total_fp4_reset = 0
    if recipe_kind == "fp4":
        _log("Resetting FP4 state (no seeding)")
        for m in models:
            total_fp4_reset += reset_fp4_state(m, reset_meta_tensors=True)

    # ---- release cached allocator blocks --------------------------------
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # ---- restore Megatron config ----------------------------------------
    for key, val in saved_config.items():
        setattr(config, key, val)

    nan_params = sum(
        1
        for m in models
        for _, p in m.named_parameters()
        if p.data.is_floating_point() and torch.isnan(p.data).any()
    )
    elapsed = time.time() - t0
    _log(
        f"Warmup complete in {elapsed:.1f}s "
        f"(fp8_reset={total_reset}, fp4_reset={total_fp4_reset}, "
        f"recipe={recipe_kind or 'native'}, nan_params={nan_params})"
    )
