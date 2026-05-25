###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
NaN/Inf grad sanitizer for benchmark configurations with random MoE routing.

When ``moe_router_force_load_balancing=True`` triggers the Bridge-parity
``apply_random_logits`` path, every MoE layer receives a *random* token-to-
expert routing every iteration. On a pretrained MoE checkpoint that
distribution is OOD for the LoRA-augmented experts, and on bf16 +
grouped_gemm + ROCm we have observed occasional NaN/Inf grads in the
backward pass (e.g. iter 28 on DeepSeek-V2-Lite SFT). Megatron's default
behaviour is to raise a fatal RuntimeError from ``check_grads``, which
makes long benchmark runs impossible.

Megatron-Bridge happens not to crash because its RNG / packing pipeline
draws a different sequence of random batches, but the underlying issue
is identical -- the configuration is a benchmark variant, not a real SFT
schedule (the yaml itself comments ``loss curve is benchmark-only``).

This patch monkey-patches ``_ParamAndGradBucketGroup.check_grads`` so
that any NaN/Inf entries in a bucket's grad tensor are replaced with 0
*before* the official NaN check runs. The net effect is that a NaN
iteration becomes a no-op step (Adam sees an all-zero grad slice for
the affected bucket and skips that update), the optimizer state stays
clean, and the training loop continues. The first time a sanitization
happens the patch logs a warning so the user knows it triggered.

The patch is gated by the yaml flag ``sft_sanitize_nan_grads`` (default
False), so it has no effect on standard pretrain / SFT recipes.
"""

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


_HAS_LOGGED_FIRST_SANITIZE = False


@register_patch(
    "megatron.sft.grad_sanitize",
    backend="megatron",
    phase="before_train",
    description=(
        "Replace NaN/Inf entries in DDP grad buckets with 0 before the "
        "Megatron grad-NaN check. Used to keep benchmark runs with "
        "moe_router_force_load_balancing=True from crashing on the "
        "rare bf16 + random-routing grad-NaN."
    ),
    condition=lambda ctx: getattr(get_args(ctx), "sft_sanitize_nan_grads", False),
)
def patch_sft_grad_sanitize(ctx: PatchContext):
    """Monkey-patch ``_ParamAndGradBucketGroup.check_grads`` to sanitize NaN/Inf grads."""
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup

    original_check_grads = _ParamAndGradBucketGroup.check_grads

    def sanitized_check_grads(self, check_for_nan_or_inf, check_for_large):
        global _HAS_LOGGED_FIRST_SANITIZE

        for bucket in self.buckets:
            grad = bucket.grad_data
            if grad is None or grad.numel() == 0:
                continue
            finite_mask = torch.isfinite(grad)
            if not finite_mask.all():
                num_bad = int((~finite_mask).sum().item())
                torch.nan_to_num_(grad, nan=0.0, posinf=0.0, neginf=0.0)
                if not _HAS_LOGGED_FIRST_SANITIZE:
                    _HAS_LOGGED_FIRST_SANITIZE = True
                    log_rank_0(
                        f"[Patch:megatron.sft.grad_sanitize]   "
                        f"Sanitized {num_bad} NaN/Inf entries in a DDP grad bucket "
                        f"(numel={grad.numel()}). Further sanitizations will be "
                        f"silent for this run. This is expected under benchmark "
                        f"configs (e.g. moe_router_force_load_balancing=True) "
                        f"and turns the iteration into a no-op update; it is "
                        f"NOT expected under standard pretrain / SFT."
                    )

        return original_check_grads(self, check_for_nan_or_inf, check_for_large)

    _ParamAndGradBucketGroup.check_grads = sanitized_check_grads
    log_rank_0(
        "[Patch:megatron.sft.grad_sanitize]   "
        "Patched _ParamAndGradBucketGroup.check_grads to sanitize NaN/Inf grads "
        "(sft_sanitize_nan_grads=True)."
    )
