###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PRIMUS_FUSE2_LOSS=1 (fuse2 line, default off): the GPT loss tail in one launch each way.

Megatron's ``pretrain_gpt.loss_func`` + ``schedules.forward_step_calc_loss`` run ~20 tiny aten launches per step on
4096 fp32 values and scalars (mul, two sums, clones, int cast, cat, clamp, two in-place divides, two ``torch.ones``
loss scales with their multiplies / divides / copies) and 3-5 more in the backward (two DivBackward, MulBackward).
Here, for the plain GPT case only:

* forward: one Triton kernel computes ``s = sum(loss * mask)``, ``n = int(sum(mask))``, the report ``[s, n]`` and the
  returned loss ``(s / max(n, 1)) / num_microbatches`` (the in-place divides of ``forward_step_calc_loss``);
* backward: one kernel ``d loss_i = ((g / num_microbatches) / max(n, 1)) * mask_i`` -- the same fp32 operations in
  autograd's order, so every gradient is bit-identical; only ``s`` is summed in a different fp32 order (the logged
  loss may differ in the last bit);
* the MoE-aux / MTP loss scales are computed once (they are constants when the loss scale is a static unity, i.e.
  ``config.grad_scale_func`` is None or PRIMUS_STATIC_BF16_LOSS_SCALE_CPU=1 makes ``scale_loss`` the identity) and
  re-used instead of ``torch.ones`` + multiply + divide + copy every step.

Everything else (NaN / spiky-loss checks on, per-token loss, distillation / ModelOpt losses, legacy 2-output loss
functions, non-last pipeline stages, custom MoE / MTP scale functions, experimental attention-variant losses)
falls back to Megatron's code unchanged.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_ENABLED = os.environ.get("PRIMUS_FUSE2_LOSS", "0") == "1"


@triton.jit
def _loss_fwd_kernel(L, M, OUT, NTOK, REP, N, NUM_MB, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m_ = offs < N
    l = tl.load(L + offs, mask=m_, other=0.0).to(tl.float32)
    m = tl.load(M + offs, mask=m_, other=0.0).to(tl.float32)
    s = tl.sum(l * m, axis=0)
    n = tl.sum(m, axis=0).to(tl.int32)  # loss_mask.sum().to(torch.int): truncation of an exact integer
    tl.store(REP, s)
    tl.store(REP + 1, n.to(tl.float32))
    tl.store(NTOK, n)
    out = s / tl.maximum(n, 1).to(tl.float32)
    out = out / NUM_MB
    tl.store(OUT, out)


@triton.jit
def _loss_bwd_kernel(G, M, NTOK, DL, N, NUM_MB, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m_ = offs < N
    g = tl.load(G).to(tl.float32)
    n = tl.load(NTOK)
    g = g / NUM_MB  # DivBackward0 of ``output_tensor /= num_microbatches``
    g = g / tl.maximum(n, 1).to(tl.float32)  # DivBackward0 of ``output_tensor /= clamp(num_tokens, min=1)``
    m = tl.load(M + offs, mask=m_, other=0.0).to(tl.float32)
    tl.store(DL + offs, g * m, mask=m_)  # MulBackward0 of ``losses * loss_mask``


class _FusedGPTLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, losses, mask, num_mb):
        N = losses.numel()
        out = torch.empty((), dtype=torch.float32, device=losses.device)
        ntok = torch.empty((), dtype=torch.int32, device=losses.device)
        rep = torch.empty((2,), dtype=torch.float32, device=losses.device)
        _loss_fwd_kernel[(1,)](losses, mask, out, ntok, rep, N, float(num_mb), BLOCK=triton.next_power_of_2(N),
                               num_warps=8)
        ctx.save_for_backward(mask, ntok)
        ctx.num_mb = float(num_mb)
        ctx.shape = losses.shape
        ctx.mark_non_differentiable(ntok, rep)
        return out, ntok, rep

    @staticmethod
    def backward(ctx, g, _gn, _gr):
        mask, ntok = ctx.saved_tensors
        N = mask.numel()
        dl = torch.empty(ctx.shape, dtype=torch.float32, device=mask.device)
        _loss_bwd_kernel[(triton.cdiv(N, 1024),)](g.contiguous(), mask, ntok, dl, N, ctx.num_mb, BLOCK=1024,
                                                  num_warps=4)
        return dl, None, None


_SCALES = {}


def _static_unity(config) -> bool:
    if getattr(config, "moe_grad_scale_func", None) is not None or getattr(config, "mtp_grad_scale_func", None) is not None:
        return False
    if config.grad_scale_func is None:
        return True
    return os.environ.get("PRIMUS_STATIC_BF16_LOSS_SCALE_CPU", "0") == "1"


def install() -> None:
    import pretrain_gpt
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel import schedules
    from megatron.training import get_args

    orig_calc = schedules.forward_step_calc_loss
    orig_loss_func = pretrain_gpt.loss_func

    def _const_scale(kind, config, device, num_mb, cp):
        key = (kind, num_mb, cp, str(device))
        v = _SCALES.get(key)
        if v is None:
            if kind == "moe":
                v = schedules._get_moe_loss_scale(config, device) * cp / num_mb
            else:
                v = schedules._get_mtp_loss_scale(config, device) / num_mb
            _SCALES[key] = v
        return v

    @functools.wraps(orig_calc)
    def forward_step_calc_loss(model, output_tensor, loss_func, config, vp_stage, collect_non_loss_data,
                               num_microbatches, forward_data_store, cp_group_size=None, is_last_stage=None):
        args = get_args()
        fast = (
            isinstance(loss_func, functools.partial)
            and loss_func.func is orig_loss_func
            and len(loss_func.args) == 1
            and not collect_non_loss_data
            and not config.calculate_per_token_loss
            and getattr(args, "logits_load_dir", None) is None
            and not getattr(args, "modelopt_enabled", False)
            and not args.check_for_nan_in_loss_and_grad
            and not getattr(args, "check_for_spiky_loss", False)
            and isinstance(output_tensor, torch.Tensor)
            and output_tensor.is_cuda
            and output_tensor.dtype == torch.float32
            and output_tensor.is_contiguous()
            and _static_unity(config)
            and schedules._get_experimental_attention_variant_loss_scale_func(config) is None
        )
        if fast:
            mask = loss_func.args[0]
            fast = (
                isinstance(mask, torch.Tensor) and mask.is_cuda and mask.is_contiguous()
                and mask.numel() == output_tensor.numel() and mask.dtype in (torch.float32, torch.bfloat16)
                and output_tensor.numel() <= 65536
            )
        if fast:
            if cp_group_size is None and is_last_stage is None:
                cp_group_size = parallel_state.get_context_parallel_world_size()
                is_last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
            fast = bool(is_last_stage)
        if not fast:
            return orig_calc(model, output_tensor, loss_func, config, vp_stage, collect_non_loss_data,
                             num_microbatches, forward_data_store, cp_group_size, is_last_stage)

        from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
        from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

        if not _SCALES.get("_logged"):
            _SCALES["_logged"] = True
            log_rank_0(f"[fuse2 loss] fused loss tail active (N={output_tensor.numel()}, num_microbatches={num_microbatches})")
        out, num_tokens, rep = _FusedGPTLossFn.apply(output_tensor.view(-1), mask.view(-1), num_microbatches)
        forward_data_store.append({"lm loss": rep})
        if config.timers is not None:
            config.timers("forward-compute").stop()
        device = out.device
        if hasattr(config, "num_moe_experts") and config.num_moe_experts is not None:
            c = _const_scale("moe", config, device, num_microbatches, cp_group_size if cp_group_size is not None else 1)
            if MoEAuxLossAutoScaler.main_loss_backward_scale is not c:
                MoEAuxLossAutoScaler.set_loss_scale(c)
        if hasattr(config, "mtp_num_layers") and config.mtp_num_layers is not None:
            MTPLossAutoScaler.set_loss_scale(_const_scale("mtp", config, device, num_microbatches, 1))
        return out, num_tokens

    schedules.forward_step_calc_loss = forward_step_calc_loss


@register_patch(
    "megatron.fuse2.loss_tail",
    backend="megatron",
    phase="before_train",
    description="GPT loss tail (loss_func + forward_step_calc_loss) in one Triton launch each way (PRIMUS_FUSE2_LOSS=1).",
    condition=lambda ctx: _ENABLED,
    priority=80,
)
def patch_fuse2_loss(ctx: PatchContext) -> None:
    install()
    log_rank_0("[Patch:megatron.fuse2.loss_tail] fused GPT loss tail (PRIMUS_FUSE2_LOSS=1)")
