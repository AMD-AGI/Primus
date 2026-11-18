###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Triton AdamW for Megatron, on the kernels from ``primus.core.kernels.triton_adam``.

TE builds on the generic ``multi_tensor_apply`` launcher cap each Adam launch at
320 workgroups; ROCm TE builds with the custom device-metadata Adam kernel are
uncapped, and against those this class is roughly on par.

``TritonFusedAdam`` subclasses TE ``FusedAdam`` so Megatron's distributed
optimizer, checkpointing and ``step``-in-param-group handling keep working
unchanged. Only the plain FP32-state path is replaced: capturable, TE-managed
master weights, decoupled grads, non-FP32 states, FP8/DTensor params, grad
scaler, closure and groups without bias correction use the TE step.
Individual gradients the kernels cannot take (dtype, layout) are updated by
torch's functional Adam in the same step.
"""

from collections import defaultdict

import torch
from torch.optim.adam import adam as torch_adam
from transformer_engine.pytorch.optimizers import FusedAdam

from primus.core.kernels.triton_adam import (
    is_batchable,
    triton_adam_step_,
    triton_multi_tensor_adam_step_,
    triton_supports,
)

# Launching every N batched tensors lets the GPU start while the host is still
# walking the remaining parameters.
_TENSORS_PER_BATCHED_LAUNCH = 512


class TritonFusedAdam(FusedAdam):
    """TE ``FusedAdam`` whose default path runs grid-stride / batched Triton kernels."""

    def __init__(self, params, *args, grid_size=0, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.grid_size = grid_size
        state_dtypes = getattr(self, "name_to_dtype_map", {})
        self._triton_enabled = (
            not self.capturable
            and not self.master_weights
            and not getattr(self, "use_decoupled_grad", False)
            and state_dtypes.get("exp_avg", torch.float32) == torch.float32
            and state_dtypes.get("exp_avg_sq", torch.float32) == torch.float32
        )
        self._batch_tables = defaultdict(dict)
        self._entries = {}
        self._params_supported = None

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._entries.clear()
        self._params_supported = None

    def _entry(self, p):
        """Static kernel inputs ``(m, v, numel, batchable, dtypes, addrs)`` of ``p``, or None."""
        state = self.state[p]
        if len(state) == 0:
            self.initialize_state(p, False)
        m, v = state["exp_avg"], state["exp_avg_sq"]
        cached = self._entries.get(p)
        if (
            cached is not None
            and cached[0] is m
            and cached[1] is v
            and (cached[2] is None or cached[2][5][0] == p.data_ptr())
        ):
            return cached[2]
        tensors = (p, m, v)
        entry = None
        if all(triton_supports(t) for t in tensors) and len({t.numel() for t in tensors}) == 1:
            entry = (
                m,
                v,
                p.numel(),
                is_batchable(tensors),
                tuple(t.dtype for t in tensors),
                tuple(t.data_ptr() for t in tensors),
            )
        self._entries[p] = (m, v, entry)
        return entry

    def _use_triton(self, closure, grad_scaler) -> bool:
        if not self._triton_enabled or closure is not None or grad_scaler is not None:
            return False
        if not all(group["bias_correction"] for group in self.param_groups):
            return False
        if self._params_supported is None:
            self._params_supported = all(
                self._entry(p) is not None for group in self.param_groups for p in group["params"]
            )
        return self._params_supported

    def step(self, closure=None, grad_scaler=None):
        if not self._use_triton(closure, grad_scaler):
            return super().step(closure=closure, grad_scaler=grad_scaler)

        for group_idx, group in enumerate(self.param_groups):
            if len(group["params"]) == 0:
                continue
            group["step"] = group.get("step", 0) + 1
            beta1, beta2 = group["betas"]
            hparams = dict(
                lr=group["lr"],
                beta1=beta1,
                beta2=beta2,
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                step=group["step"],
                bias_correction=True,
                adam_w_mode=self.adam_w_mode,
            )
            batches, launches, leftovers = defaultdict(list), defaultdict(int), []

            def launch(dtypes):
                tensors, rows = zip(*batches.pop(dtypes))
                cache = self._batch_tables[(group_idx, dtypes, launches[dtypes])]
                triton_multi_tensor_adam_step_(list(tensors), rows=list(rows), cache=cache, **hparams)
                launches[dtypes] += 1

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                entry = self._entry(p)
                if entry is None or not triton_supports(grad) or grad.numel() != entry[2]:
                    leftovers.append(p)
                    continue
                m, v, numel, batchable, (pd, md, vd), (pa, ma, va) = entry
                tensors = (p, grad, m, v)
                if batchable and is_batchable((grad,)):
                    dtypes = (pd, grad.dtype, md, vd)
                    batches[dtypes].append((tensors, [pa, grad.data_ptr(), ma, va, numel]))
                    if len(batches[dtypes]) == _TENSORS_PER_BATCHED_LAUNCH:
                        launch(dtypes)
                else:
                    triton_adam_step_(*tensors, grid_size=self.grid_size, **hparams)
            for dtypes in list(batches):
                launch(dtypes)
            if leftovers:
                self._torch_step(group, leftovers, bool(self.adam_w_mode))
        return None

    @torch.no_grad()
    def _torch_step(self, group, params, adam_w_mode):
        beta1, beta2 = group["betas"]
        torch_adam(
            params,
            [p.grad for p in params],
            [self.state[p]["exp_avg"] for p in params],
            [self.state[p]["exp_avg_sq"] for p in params],
            [],
            # torch increments these before use; the TE group step was already advanced.
            [torch.tensor(float(group["step"] - 1)) for _ in params],
            decoupled_weight_decay=adam_w_mode,
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=False,
        )
