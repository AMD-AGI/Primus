###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PRIMUS_WGRAD_BETA0=1: the first write of a step OVERWRITES ``main_grad`` (beta = 0) instead of Megatron clearing
the whole gradient buffer and every producer accumulating into zeros.

Removes, on the DSv4 paper-recipe proxy (1 microbatch, bf16 main_grad): the per-step buffer clear (35 bf16 fills,
1.34 ms) and the main_grad read of every first add (the bf16 ``add`` kernels, 4.94 ms in 258 launches, become
copies; the g90 dense block wgrad and the MegaMoE fp8blk wgrad write main_grad straight from their epilogue).

State, kept as Python attributes on each parameter and set eagerly every step:
  ``_primus_mg_stale``   set by ``zero_grad_buffer``; the first write of the step consumes it (beta 0).
  ``_primus_mg_owner``   sticky: some step's first write was a beta-0 overwrite -> the clear skips this slice.
  ``_primus_mg_skipped`` this step's clear skipped the slice (owner and laid out as a plain view of the buffer).
  ``_primus_fwd_claim``  set for weights whose aware writer flags ``grad_added_to_main_grad`` already in forward
                         (Turbo ``gemm_fp8`` fused path); for all others a set flag means another fused writer got
                         there first, and the policy answers beta 1.

Aware writers (ask ``primus_turbo.pytorch.ops.utils.main_grad_first_write(param)``, i.e. :func:`first_write`):
  * Megatron DDP backward hook (param.grad -> main_grad): ``copy_`` instead of ``add_`` on the first write.
  * Primus ``_bridge_weight_grad`` (multi-microbatch quantized-weight path): same.
  * Turbo ``gemm_fp8`` BLOCKWISE with ``fuse_bgrad_accum_pattern`` (g90 dense wgrad epilogue beta 0 / 1).
  * Turbo ``fused_mega_moe_fp8blk`` wgrad (vk / vkp kernels, beta 0 / 1).
Writers that know nothing of this (Megatron / TE / other Turbo fused epilogues, beta 1) stay correct for
parameters they write first: those never become owners, so their slices keep being cleared. The one unsupported
pattern is a parameter written by such a writer AFTER an aware writer took beta 0 on it in the same step -- none in
DSv4 (each weight has one use per microbatch; tied embedding/output weights are excluded outright).
``PRIMUS_WGRAD_BETA0_POISON=1`` fills skipped slices with NaN so any such reliance shows up as a NaN grad norm.

Clearing: ``_ParamAndGradBuffer.reset`` zeroes only maximal runs of non-owner parameters (an owner breaks a run;
alignment / bucket padding between non-owners is cleared with them). Padding next to owners is never written by
anyone (a reduce-scatter writes the reduction of zeros there) and stays zero from the allocation / iteration-0 clear.
An owner that receives no write in a step is zeroed right before its bucket's grad sync.
``PRIMUS_WGRAD_BETA0_FULLCLEAR=1``: keep Megatron's full clear but still run every beta-0 writer (bit-identity A/B).

CUDA graphs (full iteration): ``zero_grad_buffer`` runs eagerly outside the graph every step and resets the stale
flags; the captured writers keep the beta chosen at capture (beta 0 on the first write), which is valid whether or
not the slice was cleared, and owners are sticky, so the eager clear and the replayed writers stay consistent.
Not combinable with ``PRIMUS_TURBO_GRAD_OWNERSHIP=1`` (both own ``_ParamAndGradBuffer.reset``).
"""

from __future__ import annotations

import os

import torch

_ENABLED = os.environ.get("PRIMUS_WGRAD_BETA0", "0").strip().lower() in ("1", "true")
_POISON = os.environ.get("PRIMUS_WGRAD_BETA0_POISON", "0") == "1"
_FULLCLEAR = os.environ.get("PRIMUS_WGRAD_BETA0_FULLCLEAR", "0") == "1"
_stats = {"logged": False}


def _log(msg: str) -> None:
    try:
        from primus.core.utils.module_utils import log_rank_0

        log_rank_0(msg)
    except Exception:  # noqa: BLE001 -- CPU unit tests run without the Primus runtime
        print(msg)


# ----------------------------------------------------------------------------- policy (pure, CPU-testable)
def first_write(param) -> bool:
    """Consume the step's first-write state of ``param``: True -> the caller must overwrite main_grad (beta 0)."""
    param = getattr(param, "_primus_mg_param", param)  # quantized weight buffer -> its parameter
    if not getattr(param, "_primus_mg_stale", False):
        return False
    param._primus_mg_stale = False
    if getattr(param, "shared_embedding", False) or getattr(param, "shared", False):
        return False
    if getattr(param, "grad_added_to_main_grad", False) and not getattr(param, "_primus_fwd_claim", False):
        return False  # another fused writer already wrote main_grad this step
    param._primus_mg_owner = True
    return True


def mark_stale(params) -> None:
    for p in params:
        p._primus_mg_stale = True


def owner_slices_and_clear_runs(grad: torch.Tensor, index_map) -> tuple:
    """-> (owner params skipped by the clear, [start, end) runs to zero). ``index_map``: param -> (start, end, ...)."""
    base, item = grad.data_ptr(), grad.element_size()
    entries = sorted(((v[0], v[1], p) for p, v in index_map.items()), key=lambda t: t[0])
    owners, runs, cur = [], [], None
    for s, e, p in entries:
        mg = getattr(p, "main_grad", None)
        is_owner = (
            getattr(p, "_primus_mg_owner", False)
            and isinstance(mg, torch.Tensor)
            and mg.dtype == grad.dtype
            and mg.is_contiguous()
            and mg.numel() == e - s
            and mg.data_ptr() == base + s * item
        )
        if is_owner:
            owners.append(p)
            if cur is not None:
                runs.append(tuple(cur))
                cur = None
        elif cur is None:
            cur = [s, e]
        else:
            cur[1] = e  # merge across padding between two non-owners
    if cur is not None:
        runs.append(tuple(cur))
    return owners, runs


def clear_buffer(grad: torch.Tensor, index_map, extra_grads=()) -> bool:
    """Clear everything but owner slices. False (caller does the full clear) when nothing would be skipped."""
    owners, runs = owner_slices_and_clear_runs(grad, index_map)
    for p in index_map:
        p._primus_mg_skipped = False
    if not owners or _FULLCLEAR:
        return False
    for s, e in runs:
        grad[s:e].zero_()
    for g in extra_grads:
        g.zero_()
    for p in owners:
        p._primus_mg_skipped = True
        if _POISON:
            p.main_grad.fill_(float("nan"))
    if not _stats["logged"]:
        skipped = sum(p.main_grad.numel() for p in owners)
        _log(
            f"[Patch:wgrad_beta0] buffer numel={grad.numel()} owner slices={len(owners)} skipped={skipped} "
            f"({100.0 * skipped / max(1, grad.numel()):.1f}%), clear runs={len(runs)}" + ("  POISON" if _POISON else "")
        )
        _stats["logged"] = True
    return True


def zero_unwritten(params) -> None:
    """Owners whose clear was skipped but that got no write this step: zero them now (before any grad sync)."""
    for p in params:
        if getattr(p, "_primus_mg_skipped", False):
            if getattr(p, "_primus_mg_stale", False):
                p.main_grad.zero_()
                p._primus_mg_stale = False
                p._primus_mg_owner = False
            p._primus_mg_skipped = False


# ----------------------------------------------------------------------------- Megatron wiring
def _bucket_group_params(bucket_group):
    for bucket in bucket_group.buckets:
        yield from getattr(bucket, "params_list", getattr(bucket, "params", ()))


def _is_enabled(ctx) -> bool:
    if not _ENABLED:
        return False
    if os.environ.get("PRIMUS_TURBO_GRAD_OWNERSHIP", "0") == "1":
        raise RuntimeError("PRIMUS_WGRAD_BETA0=1 and PRIMUS_TURBO_GRAD_OWNERSHIP=1 both replace the grad-buffer clear")
    return True


def install() -> None:
    from megatron.core.distributed import distributed_data_parallel as ddp_mod
    from megatron.core.distributed.param_and_grad_buffer import (
        _ParamAndGradBucketGroup,
        _ParamAndGradBuffer,
    )

    DDP = ddp_mod.DistributedDataParallel
    if getattr(DDP, "_primus_wgrad_beta0", False):
        return
    try:
        from primus_turbo.pytorch.ops.utils import set_main_grad_first_write_policy

        set_main_grad_first_write_policy(first_write)
    except ImportError:
        _log("[Patch:wgrad_beta0] Primus-Turbo has no first-write policy hook: only the DDP hook path is beta 0")

    orig_zero = DDP.zero_grad_buffer
    orig_hook_factory = DDP._make_backward_post_hook
    orig_reset = _ParamAndGradBuffer.reset
    orig_start = _ParamAndGradBucketGroup.start_grad_sync
    orig_finish = _ParamAndGradBucketGroup.finish_grad_sync
    is_capturing = getattr(ddp_mod, "is_graph_capturing", lambda: False)

    def zero_grad_buffer(self):
        mark_stale(self.params_with_grad)
        return orig_zero(self)

    def reset(self):
        if not clear_buffer(self.grad_data, self.param_index_map, getattr(self, "extra_main_grads", ())):
            orig_reset(self)

    def _make_backward_post_hook(self, param):
        orig_hook = orig_hook_factory(self, param)

        def hook(*unused):
            if (
                not is_capturing()
                and param in self.param_to_bucket_group
                and param.grad is not None
                and not param.grad_added_to_main_grad
                and not getattr(param, "zero_out_wgrad", False)
                and first_write(param)
            ):
                # first write of the step: copy instead of add (no main_grad read, no clear needed); the original
                # hook then sees the flag, skips its add, drops param.grad and signals grad-ready as usual
                param.main_grad.copy_(param.grad.data)
                param.grad_added_to_main_grad = True
                try:
                    return orig_hook(*unused)
                finally:
                    param.grad_added_to_main_grad = False  # later microbatches add as usual
            return orig_hook(*unused)

        return hook

    def start_grad_sync(self, *a, **k):
        zero_unwritten(_bucket_group_params(self))
        return orig_start(self, *a, **k)

    def finish_grad_sync(self, *a, **k):
        zero_unwritten(_bucket_group_params(self))
        return orig_finish(self, *a, **k)

    DDP.zero_grad_buffer = zero_grad_buffer
    DDP._make_backward_post_hook = _make_backward_post_hook
    _ParamAndGradBuffer.reset = reset
    _ParamAndGradBucketGroup.start_grad_sync = start_grad_sync
    _ParamAndGradBucketGroup.finish_grad_sync = finish_grad_sync
    DDP._primus_wgrad_beta0 = True
    _log(
        "[Patch:wgrad_beta0] first write of a step overwrites main_grad (DDP hook copy_, Turbo g90 / MegaMoE fp8blk "
        "beta 0); grad-buffer clear limited to non-owner slices"
        + ("; FULLCLEAR (A/B check)" if _FULLCLEAR else "")
        + ("; POISON" if _POISON else "")
    )


try:
    from primus.core.patches import PatchContext, register_patch

    @register_patch(
        "megatron.core.distributed.wgrad_beta0",
        backend="megatron",
        phase="before_train",
        description="First write of a step overwrites main_grad (beta 0); the grad-buffer clear skips those slices.",
        condition=_is_enabled,
        priority=71,
    )
    def patch_wgrad_beta0(ctx: PatchContext) -> None:
        del ctx
        install()

except ImportError:  # CPU unit tests import the policy functions standalone
    pass
