###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
AdamW that keeps optimizer state in FP32 even when parameters are BF16/FP16.

Why:
- DeepSpeed bf16 commonly keeps FP32 master weights / FP32 optimizer states.
- Vanilla torch.optim.AdamW will create exp_avg/exp_avg_sq with the same dtype as the parameter,
  so bf16 params -> bf16 states, which changes early-step behavior and can destabilize.

This optimizer:
- Stores exp_avg/exp_avg_sq in FP32
- Applies the AdamW update in FP32 on a FP32 view of the param
- Writes the updated value back to the original parameter dtype

Scope:
- Intended for this repo's bf16 diffusion training (single GPU or FSDP world_size=1).
- Supports common AdamW args: lr, betas, eps, weight_decay.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch


class AdamWFP32State(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamWFP32State does not support sparse gradients")

                grad = p.grad
                state = self.state[p]

                # Initialize master weights in state if not present
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(grad, dtype=torch.float32)
                    state["master_param"] = p.detach().clone().float()

                # Always use the persistent master weight for updates
                p_fp32 = state["master_param"]

                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]
                state["step"] += 1
                step: int = state["step"]

                # Decoupled weight decay (AdamW)
                if weight_decay != 0.0:
                    p_fp32.add_(p_fp32, alpha=-lr * weight_decay)

                # Adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
                step_size = lr / bias_correction1

                p_fp32.addcdiv_(exp_avg, denom, value=-step_size)

                # Write back to original dtype
                p.copy_(p_fp32.to(dtype=p.dtype))

        return loss
