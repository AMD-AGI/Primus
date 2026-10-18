###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Hooks for Megatron's optimizer step: grad clip + AdamW + model-param write in one Triton pass per main
param, and a global-norm grad NaN check.

Two switches (module state, read at every call), set by ``primus/backends/megatron/patches/
triton_adamw_step_patches.py``, which installs the hooks built here; the hooks are inert while both are off.

``set_fused(True)``
    ``clip_grad_by_total_norm_fp32`` defers the clip coefficient instead of scaling the main grads in place;
    ``MixedPrecisionOptimizer.step_with_ready_grads`` (reached by ``DistributedOptimizer`` via ``super()``)
    then runs ``triton_adamw_kernel.adamw_clip_`` once per main param (largest first): the coefficient applied
    in registers, the fp32 main param / exp_avg / exp_avg_sq updated exactly like Transformer Engine's
    FusedAdam (bitwise on gfx1250) and the bf16 model param slice written in the same pass. The grad norm is
    unchanged. Any ineligible optimizer or tensor (``ineligible_reason``; ``fused_step`` raises ``_Fallback``)
    runs the original step for that step, after applying any deferred clip with TE ``multi_tensor_scale``
    exactly like the original clip; nothing is partially updated. ``FP32Optimizer.step_with_ready_grads``
    (fp32 training, alone or inside a ``ChainedOptimizer``) applies any deferred clip the same way before its
    own step, so no deferral outlives the step of the optimizer that owns the params.

``set_nancheck(True)``
    DDP's per-bucket NaN/Inf check (``_ParamAndGradBucketGroup.check_grads`` with ``check_for_nan_or_inf``:
    one norm kernel and two host syncs per bucket, inside the last micro-batch's backward) is skipped and the
    global grad norm, which the optimizer computes anyway, is checked instead inside the clip call, before any
    parameter update (fatal RuntimeError). ``check_for_large_grads`` is never skipped. The final
    ``check_for_nan_in_loss_and_grad`` (after ``validate_args``) is re-read on first use. Narrower than the
    bucket check in two ways: grads left out of the norm (TP-duplicate params on tp_rank > 0, e.g. LayerNorm
    weights with TP > 1 and no sequence parallel) are not checked, and a rank whose optimizer has no params
    does not raise (the others do).

After a fused step the main grads hold the unclipped values; nothing reads them before the next backward
zeroes them.
"""

from __future__ import annotations

import functools
import math
import weakref

import torch
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.optimizer import clip_grads
from megatron.core.utils import to_local_if_dtensor
from torch.distributed._tensor import DTensor

from primus.backends.megatron.core.fusions._notices import warn_fallback_once

_STATE = {"fused": False, "nancheck": False, "nan_final": None}
_PENDING = {}  # id(main param) -> (param, clip coeff) deferred by the clip wrapper
_DST_CACHE = weakref.WeakKeyDictionary()  # optimizer -> (param buffer + main param signature, dst map)


def set_fused(on: bool) -> None:
    _STATE["fused"] = bool(on)


def set_nancheck(on: bool) -> None:
    _STATE["nancheck"] = bool(on)
    _STATE["nan_final"] = None


def fused_enabled() -> bool:
    return _STATE["fused"]


def nancheck_enabled() -> bool:
    if not _STATE["nancheck"]:
        return False
    if _STATE["nan_final"] is None:  # the patch gate saw the args before validate_args
        try:
            from megatron.training import get_args

            _STATE["nan_final"] = bool(getattr(get_args(), "check_for_nan_in_loss_and_grad", True))
        except Exception:  # noqa: BLE001 - no global args (unit tests): keep the switch
            _STATE["nan_final"] = True
    return _STATE["nan_final"]


class _Fallback(Exception):
    """Raised by the fused step, before any update, when the original step must run instead."""


def _flush_pending(params):
    """Apply deferred clip coefficients with TE multi_tensor_scale, exactly like the original clip."""
    by_coeff = {}
    for p in params:
        ent = _PENDING.pop(id(p), None)
        if ent is not None and p.grad is not None:
            by_coeff.setdefault(ent[1], []).append(to_local_if_dtensor(p.grad).detach())
    for coeff, grads in by_coeff.items():
        dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=grads[0].device)
        clip_grads.multi_tensor_applier(
            clip_grads.multi_tensor_scale_impl, dummy_overflow_buf, [grads, grads], coeff
        )


def _inner_params(opt):
    """Every param of the inner optimizer (the fp32 main params for a DistributedOptimizer)."""
    return [p for g in getattr(opt.optimizer, "param_groups", []) for p in g["params"]]


def ineligible_reason(self, dm):
    """None when this optimizer can take the fused step, else the reason (``"off"`` = switch off)."""
    if not fused_enabled():
        return "off"
    if not isinstance(self, dm.DistributedOptimizer):
        return "not a DistributedOptimizer"
    if getattr(self, "is_stub_optimizer", False):
        return "stub optimizer"
    if getattr(self.ddp_config, "use_megatron_fsdp", False):
        return "megatron fsdp"
    cfg = self.config
    if getattr(cfg, "use_precision_aware_optimizer", False):
        return "precision-aware optimizer"
    if getattr(cfg, "reuse_grad_buf_for_mxfp8_param_ag", False):
        return "mxfp8 param all-gather"
    inner = self.optimizer
    if type(inner).__name__ != "FusedAdam" or not type(inner).__module__.startswith("transformer_engine"):
        return f"inner optimizer {type(inner).__module__}.{type(inner).__name__}"
    if getattr(inner, "capturable", False) or getattr(inner, "master_weights", False):
        return "capturable / master-weight FusedAdam"
    if getattr(inner, "adam_w_mode", None) != 1:
        return "not AdamW mode"
    if (
        getattr(inner, "exp_avg_dtype", torch.float32) != torch.float32
        or getattr(inner, "exp_avg_sq_dtype", torch.float32) != torch.float32
    ):
        return "non-fp32 moments"
    if getattr(inner, "use_decoupled_grad", False):
        return "decoupled grads"
    return None


def dst_map(self):
    """id(shard main param) -> its slice of the model param buffer (DistributedOptimizer copy_group_params).
    Cached per optimizer while every bucket's param buffer and every shard main param stay the same objects.
    """
    sig = tuple(
        b.param_data.data_ptr() if b.param_data is not None else 0
        for buf in self.buffers
        for b in buf.buckets
    ) + tuple(
        id(p)
        for groups in (self.shard_fp32_from_float16_groups, self.shard_fp32_groups)
        for g in groups
        for p in g
    )
    ent = _DST_CACHE.get(self)
    if ent is not None and ent[0] == sig:
        return ent[1]
    dst = {}
    pairs = (
        (self.shard_fp32_from_float16_groups, self.model_float16_groups),
        (self.shard_fp32_groups, self.model_fp32_groups),
    )
    for shard_groups, model_groups in pairs:
        for sg, mg in zip(shard_groups, model_groups):
            for smp, mp in zip(sg, mg):
                if is_float8tensor(mp):
                    raise _Fallback("fp8 model param")
                world_range = self._get_model_param_range_map(mp)["gbuf_world_in_bucket"]
                gbuf_index, _, bucket_id = self.model_param_gbuf_map[mp]
                buf = self.buffers[gbuf_index].buckets[bucket_id].param_data
                if buf is None:
                    raise _Fallback("no param buffer")
                view = buf.view(-1)[world_range.start : world_range.end]
                if view.numel() != smp.nelement():
                    raise _Fallback("param range mismatch")
                dst[id(smp)] = view
    _DST_CACHE[self] = (sig, dst)
    return dst


def fused_step(self) -> None:
    """Clip (deferred coefficient) + AdamW + model-param write for every main param; raises _Fallback
    before any update."""
    from primus.backends.megatron.core.optimizer import triton_adamw_kernel as kernel

    inner = self.optimizer
    dst = dst_map(self)
    work = []
    for group in inner.param_groups:
        if len(group["params"]) == 0:
            continue
        items = []
        for p in group["params"]:
            st = inner.state[p]
            if len(st) == 0:  # first step: TE's own zero init, before the grad check, as FusedAdam.step does
                inner.initialize_state(p, False)
            g = p.grad
            if g is None:
                continue
            if "exp_avg" not in st or "exp_avg_sq" not in st:
                raise _Fallback("optimizer state not initialised")
            m, v = st["exp_avg"], st["exp_avg_sq"]
            for t in (p, g, m, v):
                if (
                    isinstance(t, DTensor)
                    or t.dtype != torch.float32
                    or not t.is_contiguous()
                    or t.numel() != p.numel()
                    or not t.is_cuda
                ):
                    raise _Fallback("tensor layout")
            d = dst.get(id(p))
            if d is None:
                raise _Fallback("main param without a model param slice")
            items.append((p, g, m, v, d))
        if isinstance(group.get("lr"), torch.Tensor):
            raise _Fallback("tensor lr")
        work.append((group, items))
    # All checks passed: update (TE FusedAdam.step semantics, then the bf16 copy). Params are
    # independent, so launch largest first: the small ones queue up behind a busy GPU.
    launches = []
    for group, items in work:
        group["step"] = group["step"] + 1 if "step" in group else 1
        beta1, beta2 = group["betas"]
        sc = kernel.te_scalars(
            group["lr"],
            beta1,
            beta2,
            group["eps"],
            group["weight_decay"],
            group["step"],
            group.get("bias_correction", True),
        )
        launches.extend((sc, it) for it in items)
    launches.sort(key=lambda x: -x[1][0].numel())
    for sc, (p, g, m, v, d) in launches:
        ent = _PENDING.pop(id(p), None)
        coeff = ent[1] if ent is not None else 1.0
        # An fp32 model param shares its main param's storage: copy it as Megatron does (a self-copy) rather
        # than add a kernel variant that stores twice to the same address.
        out = d if d.dtype in (torch.bfloat16, torch.float16) else None
        kernel.adamw_clip_(p, g, m, v, out, clip=coeff, scalars=sc, check=False)
        if out is None:
            d.data.copy_(p)


def make_clip_grad_by_total_norm_fp32(orig):
    """Wrap Megatron's clip: global-norm NaN check, then defer (fused on) or apply (fused off) the clip."""

    @functools.wraps(orig)
    def clip_grad_by_total_norm_fp32(parameters, max_norm, total_norm, use_decoupled_grad=False):
        if nancheck_enabled() and not math.isfinite(total_norm):
            raise RuntimeError(
                f"non-finite global grad norm {total_norm} (use_primus_global_grad_nan_check checks it "
                "instead of every DDP bucket); no parameter was updated"
            )
        # A list: the parameters are walked twice below, and the replaced function accepts any iterable.
        parameters = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
        for p in parameters:  # a new clip decision supersedes any stale deferral
            _PENDING.pop(id(p), None)
        if not fused_enabled() or use_decoupled_grad:
            return orig(parameters, max_norm, total_norm, use_decoupled_grad)
        clip_coeff = max_norm / (total_norm + 1.0e-6)
        if clip_coeff < 1.0:
            for p in parameters:
                if p.grad is not None:
                    assert p.grad.dtype == torch.float32
                    _PENDING[id(p)] = (p, clip_coeff)

    return clip_grad_by_total_norm_fp32


def make_step_with_ready_grads(orig, dm, key: str):
    """Wrap ``MixedPrecisionOptimizer.step_with_ready_grads``: the fused step when eligible, else the original
    step (the reason logged once under the patch ``key``)."""

    @functools.wraps(orig)
    def step_with_ready_grads(self):
        why = ineligible_reason(self, dm)
        if why is None:
            timers = self.config.timers
            if timers is not None:
                timers("optimizer-inner-step", log_level=1).start(barrier=self.config.barrier_with_L1_time)
            try:
                fused_step(self)
            except _Fallback as exc:
                why = f"fallback: {exc}"
            if timers is not None:
                timers("optimizer-inner-step").stop()
            if why is None:
                if timers is not None:  # keep the timer set of the original method
                    timers("optimizer-copy-main-to-model-params", log_level=1).start(
                        barrier=self.config.barrier_with_L1_time
                    )
                    timers("optimizer-copy-main-to-model-params").stop()
                return True
        if why != "off":
            warn_fallback_once(key, why)
        _flush_pending(_inner_params(self))
        return orig(self)

    return step_with_ready_grads


def make_flush_then_step(orig):
    """Wrap ``FP32Optimizer.step_with_ready_grads``: apply a deferred clip (as the clip would), then step."""

    @functools.wraps(orig)
    def step_with_ready_grads(self):
        if not getattr(self, "is_stub_optimizer", False) and getattr(self, "optimizer", None) is not None:
            _flush_pending(_inner_params(self))
        return orig(self)

    return step_with_ready_grads


def make_check_grads(orig):
    """Wrap ``_ParamAndGradBucketGroup.check_grads``: no NaN/Inf-only check while the global check is on."""

    @functools.wraps(orig)
    def check_grads(self, check_for_nan_or_inf, check_for_large):
        if nancheck_enabled() and not check_for_large:
            return None
        return orig(self, check_for_nan_or_inf, check_for_large)

    return check_grads
