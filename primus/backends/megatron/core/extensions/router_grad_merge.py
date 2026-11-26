###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Join the MoE router input gradient with the expert input gradient.

The MoE input feeds both the FP32 router and the experts, so autograd runs the
BF16 router dX GEMM and then adds its result to the expert input gradient.
With the BF16 router backward (``moe_router_bwd_bf16``) this module defers the
router dX GEMM to an identity autograd join on the MoE input and computes
``bf16(dY @ W) + expert_grad`` in one Triton kernel, rounding exactly like the
unfused GEMM output followed by the BF16 gradient accumulation.

Only the fused residual/RMSNorm transformer-layer forward opens the join, since
its normalized activation is consumed by the MLP alone. Any layer, MoE, or
router configuration outside the guards below runs Megatron's path unchanged.
The first fused joins are compared bitwise with the unfused sum; a mismatch
disables the join for the rest of the run.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable

import torch

from primus.core.utils.module_utils import log_rank_0, log_rank_all

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None

SUPPORTED_NUM_EXPERTS = (16, 32, 64, 128)
VERIFY_CALLS = 48
_BLOCK_M = 32
_BLOCK_N = 128

_PARENT_SCOPE = contextvars.ContextVar("primus_router_grad_merge_parent", default=None)
_ROUTE_SCOPE = contextvars.ContextVar("primus_router_grad_merge_route", default=None)
_STATS = dict(created=0, claimed=0, fused=0, identity=0, reference=0, verified=0)
_ORIGINALS = {}
_STATE = dict(disabled=False, hook_attr=None)


if triton is not None:

    @triton.jit
    def _router_merge_kernel(
        G,
        W,
        E,
        O,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        kk = tl.arange(0, K)
        row_ok = rows[:, None] < M
        col_ok = cols[None, :] < N
        g = tl.load(G + rows[:, None] * K + kk[None, :], mask=row_ok, other=0)
        w = tl.load(W + kk[:, None] * N + cols[None, :], mask=col_ok, other=0)
        rounded = tl.dot(g, w).to(tl.bfloat16).to(tl.float32)
        offsets = rows[:, None] * N + cols[None, :]
        expert = tl.load(E + offsets, mask=row_ok & col_ok, other=0).to(tl.float32)
        # One opaque FP32 add after the BF16 rounding, as in the unfused sum.
        total = tl.inline_asm_elementwise(
            "v_add_f32 $0, $1, $2",
            constraints="=v,v,v",
            args=[rounded, expert],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(O + offsets, total.to(tl.bfloat16), mask=row_ok & col_ok)


def merge_kernel(grad: torch.Tensor, weight: torch.Tensor, expert: torch.Tensor) -> torch.Tensor:
    """Return ``bf16(grad @ weight) + expert`` for BF16 ``[M, E]``, ``[E, H]``, ``[M, H]``."""
    rows, hidden = expert.shape
    output = torch.empty_like(expert)
    grid = (triton.cdiv(rows, _BLOCK_M), triton.cdiv(hidden, _BLOCK_N))
    _router_merge_kernel[grid](
        grad,
        weight,
        expert,
        output,
        rows,
        hidden,
        weight.shape[0],
        _BLOCK_M,
        _BLOCK_N,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    return output


class _MergeRecord:
    """Router-side state of one MoE forward, consumed by the input join."""

    def __init__(self, x: torch.Tensor):
        self.pointer = x.data_ptr()
        self.shape = tuple(x.shape)
        self.numel = x.numel()
        self.stream = torch.cuda.current_stream(x.device)
        self.claimed = False
        self.grad = None
        self.weight = None
        self.gemm = None
        _STATS["created"] += 1


def _reference_sum(record: _MergeRecord, expert):
    value = record.gemm(record.weight, record.grad, torch.bfloat16, layout="NN", grad=True)[0]
    value = value.view(record.shape)
    return value if expert is None else value + expert


def _merge_sum(record: _MergeRecord, expert):
    grad, weight = record.grad, record.weight
    current = torch.cuda.current_stream(grad.device)
    same_stream = current.cuda_stream == record.stream.cuda_stream
    if not same_stream:
        current.wait_stream(record.stream)
        grad.record_stream(current)
        weight.record_stream(current)
    num_experts, hidden = weight.shape
    eligible = (
        not _STATE["disabled"]
        and same_stream
        and expert is not None
        and expert.dtype == torch.bfloat16
        and expert.is_contiguous()
        and grad.is_contiguous()
        and weight.is_contiguous()
        and grad.shape[1] == num_experts
        and expert.numel() == grad.shape[0] * hidden
    )
    if not eligible:
        _STATS["reference"] += 1
        return _reference_sum(record, expert)

    flat = expert.view(-1, hidden)
    output = merge_kernel(grad, weight, flat)
    if _STATS["verified"] < VERIFY_CALLS:
        reference = record.gemm(weight, grad, torch.bfloat16, layout="NN", grad=True)[0] + flat
        if not torch.equal(output.view(torch.int16), reference.view(torch.int16)):
            _STATE["disabled"] = True
            log_rank_all(
                "[RouterGradMerge] fused join differs from the unfused BF16 sum; disabled for this run"
            )
            return reference.view(record.shape)
        _STATS["verified"] += 1
        if _STATS["verified"] == VERIFY_CALLS:
            log_rank_0(f"[RouterGradMerge] verified {VERIFY_CALLS} fused joins bitwise")
    _STATS["fused"] += 1
    return output.view(record.shape)


class _RouterGradientJoin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, record):
        ctx.record = record
        ctx.set_materialize_grads(False)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, expert):
        record = ctx.record
        if record.grad is None:
            _STATS["identity"] += 1
            return expert, None
        try:
            return _merge_sum(record, expert), None
        finally:
            record.grad = None
            record.weight = None
            record.gemm = None


def stats() -> dict:
    return dict(_STATS, disabled=_STATE["disabled"], installed=bool(_ORIGINALS))


def _only_param_gather_hooks(module) -> bool:
    if any(
        getattr(module, name, None) for name in ("_forward_hooks", "_backward_pre_hooks", "_backward_hooks")
    ):
        return False
    # DDP's parameter all-gather pre-hook ignores the module inputs, so it
    # cannot add a consumer of the joined activation.
    hook_attr = _STATE["hook_attr"]
    return all(getattr(hook, hook_attr, False) for hook in getattr(module, "_forward_pre_hooks", {}).values())


def _no_global_hooks() -> bool:
    module = torch.nn.modules.module
    return not any(
        getattr(module, name, None)
        for name in (
            "_global_forward_hooks",
            "_global_forward_pre_hooks",
            "_global_backward_hooks",
            "_global_backward_pre_hooks",
        )
    )


def _layer_eligible(layer, residual, inference_context, padding_mask) -> bool:
    from megatron.core.transformer.moe.moe_layer import MoELayer

    cfg = layer.config
    mlp = getattr(layer, "mlp", None)
    norm = getattr(layer, "pre_mlp_layernorm", None)
    if not (
        layer.training
        and torch.is_grad_enabled()
        and residual.requires_grad
        and inference_context is None
        and padding_mask is None
        and type(mlp) is MoELayer
        and norm is not None
        and not isinstance(norm, torch.nn.Identity)
    ):
        return False
    if any(
        getattr(layer, name, False)
        for name in ("recompute_mlp", "recompute_pre_mlp_layernorm", "offload_mlp_norm")
    ):
        return False
    if getattr(cfg, "cuda_graph_impl", "none") not in (None, "none"):
        return False
    if getattr(cfg, "mlp_chunks_for_prefill", 1) != 1 or getattr(cfg, "mlp_chunks_for_training", 1) != 1:
        return False
    return _no_global_hooks() and all(_only_param_gather_hooks(m) for m in (layer, mlp, norm, mlp.router))


def _moe_eligible(moe, x, intermediate_tensors, padding_mask, parent) -> bool:
    cfg = moe.config
    if (
        parent is None
        or parent["moe"] is not moe
        or intermediate_tensors is not None
        or padding_mask is not None
    ):
        return False
    if not (
        moe.training
        and torch.is_grad_enabled()
        and x.requires_grad
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.dim() == 3
        and x.shape[-1] == cfg.hidden_size
        and x.numel() < 2**31
    ):
        return False
    if (
        x.untyped_storage().data_ptr() == parent["residual_storage"]
        or x.data_ptr() != parent["private_pointer"]
    ):
        return False
    if cfg.num_moe_experts not in SUPPORTED_NUM_EXPERTS:
        return False
    if any(
        getattr(moe, name, False)
        for name in ("moe_layer_recompute", "use_shared_expert", "shared_expert_overlap")
    ):
        return False
    if any(
        getattr(cfg, name, None)
        for name in ("moe_latent_size", "moe_input_jitter_eps", "moe_shared_expert_intermediate_size")
    ):
        return False
    if getattr(cfg, "moe_router_dtype", None) != "fp32":
        return False
    if getattr(cfg, "moe_router_force_load_balancing", False):
        return False
    if set(moe.fwd_execution_map) != {"route", "expert_compute", "postprocess"}:
        return False
    if not all(
        getattr(getattr(moe, name), "__func__", None) is _ORIGINALS[name]
        for name in ("route", "preprocess", "shared_experts_compute")
    ):
        return False
    router = moe.router
    if getattr(router.forward, "__func__", None) is not _ORIGINALS["router_forward"]:
        return False
    if getattr(router.gating, "__func__", None) is not _ORIGINALS["router_gating"]:
        return False
    return all(_only_param_gather_hooks(m) for m in (moe, router))


def begin_fused_scope(layer, normalized, residual, inference_context, padding_mask):
    """Open the join around ``layer.mlp(normalized)`` in the fused layer forward.

    ``normalized`` must be consumed only by ``layer.mlp``; ``residual`` is the
    separate residual stream. Returns a token for :func:`end_fused_scope`.
    """
    if not _ORIGINALS or _STATE["disabled"]:
        return None
    if not _layer_eligible(layer, residual, inference_context, padding_mask):
        return None
    parent = dict(
        moe=layer.mlp,
        residual_storage=residual.untyped_storage().data_ptr(),
        private_pointer=normalized.data_ptr(),
    )
    if not _moe_eligible(layer.mlp, normalized, None, padding_mask, parent):
        return None
    return _PARENT_SCOPE.set(parent)


def end_fused_scope(token) -> None:
    if token is not None:
        _PARENT_SCOPE.reset(token)


def install(uses_bf16_backward: Callable) -> bool:
    """Install the join over Megatron's MoE forward and router gating function.

    ``uses_bf16_backward(ctx, te_general_gemm)`` selects router backward calls
    that run the BF16 GEMMs. Returns False, leaving Megatron untouched, when
    another patch already replaced one of the wrapped methods.
    """
    if triton is None:
        raise RuntimeError("moe_router_grad_merge requires Triton")
    from megatron.core.distributed import distributed_data_parallel as ddp_module
    from megatron.core.transformer.moe import moe_layer, moe_utils, router

    from primus.backends.megatron.patches.turbo.qk_rmsnorm_rope_patches import (
        _DDP_PARAM_GATHER_HOOK_ATTR,
        _install_ddp_param_gather_hook_marker,
    )

    moe_cls = moe_layer.MoELayer
    gating_fn = moe_utils.RouterGatingLinearFunction
    if getattr(moe_cls, "_primus_router_grad_merge", False):
        return True
    originals = dict(
        moe_forward=moe_cls.forward,
        route=moe_cls.route,
        preprocess=moe_cls.preprocess,
        shared_experts_compute=moe_cls.shared_experts_compute,
        router_forward=router.TopKRouter.forward,
        router_gating=router.Router.gating,
        gating_forward=gating_fn.forward,
        gating_backward=gating_fn.backward,
    )
    for name, module in (
        ("moe_forward", moe_layer),
        ("router_forward", router),
        ("router_gating", router),
        ("gating_forward", moe_utils),
    ):
        if originals[name].__code__.co_filename != module.__file__:
            log_rank_0(f"[RouterGradMerge] {name} is already replaced by another patch; join not installed")
            return False

    _install_ddp_param_gather_hook_marker(ddp_module.DistributedDataParallel)
    _STATE["hook_attr"] = _DDP_PARAM_GATHER_HOOK_ATTR
    _ORIGINALS.update(originals)

    def moe_forward(self, hidden_states, intermediate_tensors=None, padding_mask=None):
        parent = _PARENT_SCOPE.get()
        if not _moe_eligible(self, hidden_states, intermediate_tensors, padding_mask, parent):
            return originals["moe_forward"](self, hidden_states, intermediate_tensors, padding_mask)
        record = _MergeRecord(hidden_states)
        joined = _RouterGradientJoin.apply(hidden_states, record)
        token = _ROUTE_SCOPE.set(record)
        try:
            return originals["moe_forward"](self, joined, intermediate_tensors, padding_mask)
        finally:
            _ROUTE_SCOPE.reset(token)

    def gating_forward(ctx, inp, weight, bias, router_dtype):
        record = _ROUTE_SCOPE.get()
        if (
            record is not None
            and inp.requires_grad
            and inp.dtype == weight.dtype == torch.bfloat16
            and router_dtype == torch.float32
            and inp.is_contiguous()
            and weight.is_contiguous()
            and weight.dim() == 2
            and weight.shape[0] in SUPPORTED_NUM_EXPERTS
            and weight.shape[1] == inp.shape[-1]
            and inp.numel() == record.numel
            and inp.data_ptr() == record.pointer
            and torch.cuda.current_stream(inp.device).cuda_stream == record.stream.cuda_stream
        ):
            if record.claimed:
                raise RuntimeError("moe_router_grad_merge: multiple router consumers of one MoE input")
            record.claimed = True
            _STATS["claimed"] += 1
            ctx._primus_router_merge_record = record
        return originals["gating_forward"](ctx, inp, weight, bias, router_dtype)

    def gating_backward(ctx, grad_output):
        record = getattr(ctx, "_primus_router_merge_record", None)
        te_general_gemm = moe_utils.te_general_gemm
        if (
            record is None
            or _STATE["disabled"]
            or not uses_bf16_backward(ctx, te_general_gemm)
            or torch.is_grad_enabled()
            or torch.cuda.current_stream(grad_output.device).cuda_stream != record.stream.cuda_stream
        ):
            return originals["gating_backward"](ctx, grad_output)
        if record.grad is not None:
            raise RuntimeError("moe_router_grad_merge: reentrant backward reused the join state")
        inp, weight, bias = ctx.saved_tensors
        inp = inp.view(-1, inp.shape[-1])
        grad_output = grad_output.view(-1, grad_output.shape[-1])
        grad_output_bf16 = grad_output.to(torch.bfloat16)
        grad_weight = te_general_gemm(inp, grad_output_bf16, ctx.router_dtype, layout="NT", grad=True)[0].to(
            ctx.weight_dtype
        )
        grad_bias = grad_output.sum(dim=0).to(ctx.weight_dtype) if bias is not None else None
        record.grad = grad_output_bf16
        record.weight = weight
        record.gemm = te_general_gemm
        return None, grad_weight, grad_bias, None

    moe_cls.forward = moe_forward
    gating_fn.forward = staticmethod(gating_forward)
    gating_fn.backward = staticmethod(gating_backward)
    moe_cls._primus_router_grad_merge = True
    return True
