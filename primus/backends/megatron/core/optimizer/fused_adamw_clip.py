###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused clip + AdamW + bf16 param write step, and a global-norm NaN check, for Megatron's DistributedOptimizer.

Two independent switches (module state, read at every call), enabled by the patches in
``primus/backends/megatron/patches/fused_adamw_clip_patches.py``:

``set_fused(True)``   (patch ``megatron.optimizer.fused_adamw_clip``)
    ``clip_grad_by_total_norm_fp32`` defers the clip coefficient instead of scaling the main grads in place;
    ``MixedPrecisionOptimizer.step_with_ready_grads`` (reached by ``DistributedOptimizer`` through ``super()``)
    then runs one Triton pass per main param (``triton_fused_adamw.adamw_clip_``, largest first) that applies the
    coefficient in registers, updates the fp32 main param / exp_avg / exp_avg_sq exactly like Transformer
    Engine's FusedAdam (bitwise) and writes the bf16 model param slice in the same pass. The grad norm is
    unchanged. Replaces TE ``multi_tensor_scale`` (clip) + TE ``multi_tensor_adam`` + the per-param main ->
    model copy. Any ineligible optimizer or tensor (precision-aware optimizer, FSDP, fp8 params, capturable /
    master-weight / non-AdamW FusedAdam, non-fp32 states, DTensor or non-contiguous tensors, tensor lr) runs
    the original step for that step, after applying any deferred clip with TE ``multi_tensor_scale`` exactly
    like the original clip; nothing is partially updated. ``FP32Optimizer.step_with_ready_grads`` (fp32
    training, alone or inside a ``ChainedOptimizer``) is hooked to apply any deferred clip the same way before
    its own step, so no deferral outlives the step of the optimizer that owns the params.

``set_nancheck(True)``   (patch ``megatron.ddp.global_grad_nan_check``)
    DDP's per-bucket NaN/Inf check (``_ParamAndGradBucketGroup.check_grads`` with ``check_for_nan_or_inf``:
    one norm kernel and two host syncs per bucket, inside the last micro-batch's backward) is skipped and the
    global grad norm, which the optimizer computes anyway, is checked instead inside the clip call, before any
    parameter update (fatal RuntimeError). Needs ``clip_grad > 0`` (otherwise the clip, and with it the check,
    is not called) and ``rerun_mode: disabled`` (otherwise the bucket check feeds Megatron's rerun state
    machine). ``check_for_large_grads`` is never skipped.

After a fused step the main grads hold the unclipped values; nothing reads them before the next backward
zeroes them.
"""

from __future__ import annotations

import math
import weakref

from primus.core.utils.module_utils import warning_rank_0

_STATE = {"fused": False, "nancheck": False, "last_fallback": None}
_PENDING = {}  # id(main param) -> (param, clip coeff) deferred by the clip wrapper
_DST_CACHE = weakref.WeakKeyDictionary()  # optimizer -> (param buffer signature, dst map)
HOOKED_ATTR = "__primus_fused_adamw_clip_orig__"


def set_fused(on: bool) -> None:
    _STATE["fused"] = bool(on)


def set_nancheck(on: bool) -> None:
    _STATE["nancheck"] = bool(on)


def fused_enabled() -> bool:
    return _STATE["fused"]


def nancheck_enabled() -> bool:
    return _STATE["nancheck"]


class _Fallback(Exception):
    """Raised by the fused step, before any update, when the original step must run instead."""


def _flush_pending(params):
    """Apply deferred clip scales with TE multi_tensor_scale, exactly like the original clip."""
    import torch
    from megatron.core.optimizer import clip_grads as cg
    from megatron.core.utils import to_local_if_dtensor

    by_coeff = {}
    for p in params:
        ent = _PENDING.pop(id(p), None)
        if ent is not None and p.grad is not None:
            by_coeff.setdefault(ent[1], []).append(to_local_if_dtensor(p.grad).detach())
    for coeff, grads in by_coeff.items():
        dummy = torch.zeros(1, dtype=torch.int, device="cuda")
        cg.multi_tensor_applier(cg.multi_tensor_scale_impl, dummy, [grads, grads], coeff)


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
    import torch

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
    Cached per optimizer while every bucket's param buffer stays at the same address."""
    from megatron.core.fp8_utils import is_float8tensor

    sig = tuple(
        b.param_data.data_ptr() if b.param_data is not None else 0
        for buf in self.buffers
        for b in buf.buckets
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


def fused_step(self):
    """Clip (deferred coefficient) + AdamW + bf16 write for every main param; raises _Fallback before any update."""
    import torch
    from torch.distributed._tensor import DTensor

    from primus.backends.megatron.core.optimizer import triton_fused_adamw as fa

    inner = self.optimizer
    dst = dst_map(self)
    work = []
    for group in inner.param_groups:
        if len(group["params"]) == 0:
            continue
        items = []
        for p in group["params"]:
            g = p.grad
            if g is None:
                continue
            st = inner.state[p]
            if len(st) == 0:  # first step: TE's own zero init, as FusedAdam.step does
                inner.initialize_state(p, False)
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
        sc = fa.te_scalars(
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
        out = d if d.dtype in (torch.bfloat16, torch.float16) else None
        fa.adamw_clip_(p, g, m, v, out, clip=coeff, scalars=sc, check=False)
        if out is None:
            d.data.copy_(p)
    return True


def make_clip_grad_by_total_norm_fp32(orig):
    """Wrap Megatron's clip: global-norm NaN check, then defer (fused on) or apply (fused off) the clip."""

    def clip_grad_by_total_norm_fp32(parameters, max_norm, total_norm, use_decoupled_grad=False):
        import torch

        if nancheck_enabled():
            if not math.isfinite(total_norm):
                raise RuntimeError(
                    f"[fused_adamw_clip] non-finite global grad norm {total_norm} (per-bucket NaN check "
                    "replaced by the global-norm check); no parameter was updated"
                )
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        for p in parameters:  # a new clip decision supersedes any stale deferral
            _PENDING.pop(id(p), None)
        if not fused_enabled() or use_decoupled_grad:
            return orig(parameters, max_norm, total_norm, use_decoupled_grad)
        clip_coeff = max_norm / (total_norm + 1.0e-6)
        if clip_coeff < 1.0:
            for p in parameters:
                if p.grad is not None:
                    assert p.grad.type() == "torch.cuda.FloatTensor"
                    _PENDING[id(p)] = (p, clip_coeff)

    clip_grad_by_total_norm_fp32.__wrapped__ = orig
    setattr(clip_grad_by_total_norm_fp32, HOOKED_ATTR, orig)
    return clip_grad_by_total_norm_fp32


def make_step_with_ready_grads(orig, dm):
    """Wrap ``MixedPrecisionOptimizer.step_with_ready_grads``: fused step when eligible, else the original."""

    def step_with_ready_grads(self):
        why = ineligible_reason(self, dm)
        if why is None:
            timers = self.config.timers
            if timers is not None:
                timers("optimizer-inner-step", log_level=1).start(barrier=self.config.barrier_with_L1_time)
            try:
                ok = fused_step(self)
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
                return ok
        if why != "off" and _STATE["last_fallback"] != why:
            warning_rank_0(f"[fused_adamw_clip] original optimizer step ({why})")
            _STATE["last_fallback"] = why
        _flush_pending(_inner_params(self))
        return orig(self)

    step_with_ready_grads.__wrapped__ = orig
    setattr(step_with_ready_grads, HOOKED_ATTR, orig)
    return step_with_ready_grads


def make_flush_then_step(orig):
    """Wrap ``FP32Optimizer.step_with_ready_grads``: apply any deferred clip (as the original clip), then step."""

    def step_with_ready_grads(self):
        if not getattr(self, "is_stub_optimizer", False) and getattr(self, "optimizer", None) is not None:
            _flush_pending(_inner_params(self))
        return orig(self)

    step_with_ready_grads.__wrapped__ = orig
    setattr(step_with_ready_grads, HOOKED_ATTR, orig)
    return step_with_ready_grads


def make_check_grads(orig):
    """Wrap ``_ParamAndGradBucketGroup.check_grads``: skip the NaN/Inf-only check while the global check is on."""

    def check_grads(self, check_for_nan_or_inf, check_for_large):
        if nancheck_enabled() and not check_for_large:
            return None
        return orig(self, check_for_nan_or_inf, check_for_large)

    check_grads.__wrapped__ = orig
    setattr(check_grads, HOOKED_ATTR, orig)
    return check_grads


def install_hooks() -> dict:
    """Hook Megatron's clip, the step_with_ready_grads of MixedPrecisionOptimizer and FP32Optimizer and the DDP
    bucket check (idempotent). The hooks are inert while both switches are off.
    Returns {hook name: "patched" | "already" | "absent"}."""
    import importlib

    om = importlib.import_module("megatron.core.optimizer.optimizer")
    dm = importlib.import_module("megatron.core.optimizer.distrib_optimizer")
    pb = importlib.import_module("megatron.core.distributed.param_and_grad_buffer")
    res = {}
    cur = om.clip_grad_by_total_norm_fp32
    if getattr(cur, HOOKED_ATTR, None) is None:
        om.clip_grad_by_total_norm_fp32 = make_clip_grad_by_total_norm_fp32(cur)
        res["clip_grad_by_total_norm_fp32"] = "patched"
    else:
        res["clip_grad_by_total_norm_fp32"] = "already"
    mpo = om.MixedPrecisionOptimizer
    cur = mpo.step_with_ready_grads
    if getattr(cur, HOOKED_ATTR, None) is None:
        mpo.step_with_ready_grads = make_step_with_ready_grads(cur, dm)
        res["MixedPrecisionOptimizer.step_with_ready_grads"] = "patched"
    else:
        res["MixedPrecisionOptimizer.step_with_ready_grads"] = "already"
    fpo = om.FP32Optimizer
    cur = fpo.step_with_ready_grads
    if getattr(cur, HOOKED_ATTR, None) is None:
        fpo.step_with_ready_grads = make_flush_then_step(cur)
        res["FP32Optimizer.step_with_ready_grads"] = "patched"
    else:
        res["FP32Optimizer.step_with_ready_grads"] = "already"
    group_cls = getattr(pb, "_ParamAndGradBucketGroup", None)
    if group_cls is None or not hasattr(group_cls, "check_grads"):
        res["_ParamAndGradBucketGroup.check_grads"] = "absent"
    elif getattr(group_cls.check_grads, HOOKED_ATTR, None) is None:
        group_cls.check_grads = make_check_grads(group_cls.check_grads)
        res["_ParamAndGradBucketGroup.check_grads"] = "patched"
    else:
        res["_ParamAndGradBucketGroup.check_grads"] = "already"
    return res
