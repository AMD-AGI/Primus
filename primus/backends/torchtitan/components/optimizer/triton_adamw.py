###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Triton AdamW for TorchTitan.

``torch.optim.AdamW(fused=True)`` launches ATen ``multi_tensor_apply`` kernels
capped at 320 workgroups, which leaves MI455X HBM far from saturated.
``TritonFusedAdamW`` runs the device-sized kernels from
``primus.core.kernels.triton_adam`` instead.

It subclasses ``torch.optim.AdamW`` and keeps its state layout (per-parameter
CPU ``step`` plus ``exp_avg`` / ``exp_avg_sq``), so DCP checkpoints stay
interchangeable with the stock optimizer. FSDP2 ``DTensor`` parameters are
updated through their local shards.

Group options the kernels do not implement (amsgrad, maximize, capturable,
differentiable, tensor hyper-parameters) and ``closure`` use the stock step.
Individual tensors the kernels cannot take (dtype, layout, placement) are
updated by torch's functional Adam in the same step.
"""

from collections import defaultdict

import torch
from torch.distributed.tensor import DTensor
from torch.optim.adam import adam as torch_adam
from torch.optim.optimizer import _get_scalar_dtype

from primus.core.kernels.triton_adam import (
    is_batchable,
    triton_adam_step_,
    triton_multi_tensor_adam_step_,
    triton_supports,
)

# Launching every N batched tensors lets the GPU start while the host is still
# walking the remaining parameters.
_TENSORS_PER_BATCHED_LAUNCH = 512


def _local(t, placements):
    if isinstance(t, DTensor):
        if t.placements != placements:
            return None
        t = t._local_tensor
    return t if triton_supports(t) else None


class TritonFusedAdamW(torch.optim.AdamW):
    """``torch.optim.AdamW`` whose default path runs grid-stride / batched Triton kernels."""

    def __init__(self, params, *args, grid_size=0, **kwargs):
        kwargs.pop("fused", None)
        kwargs.pop("foreach", None)
        super().__init__(params, *args, **kwargs)
        self.grid_size = grid_size
        self._batch_tables = defaultdict(dict)
        self._locals_cache = {}
        self._step_values = None

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # torch keeps a fused optimizer's device ``step`` on the device when loading into a
        # non-fused one; the kernels read it on the host every step.
        for state in self.state.values():
            step = state.get("step")
            if isinstance(step, torch.Tensor) and step.device.type != "cpu":
                state["step"] = step.to(device="cpu", dtype=_get_scalar_dtype())
        self._locals_cache.clear()
        self._step_values = None

    @staticmethod
    def _group_supported(group) -> bool:
        return not (
            group["amsgrad"]
            or group["maximize"]
            or group["capturable"]
            or group["differentiable"]
            or any(isinstance(group[k], torch.Tensor) for k in ("lr", "eps", "weight_decay"))
            or any(isinstance(b, torch.Tensor) for b in group["betas"])
        )

    def _state(self, p):
        state = self.state[p]
        if len(state) == 0:
            state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        return state

    def _locals(self, p, state):
        """Static kernel inputs of ``p``, or None if the kernels cannot take it.

        ``(placements, p_local, m_local, v_local, numel, batchable, dtypes, addrs)``; only the
        gradient is re-validated every step.
        """
        m, v = state["exp_avg"], state["exp_avg_sq"]
        p_local = p._local_tensor if isinstance(p, DTensor) else p
        cached = self._locals_cache.get(p)
        if (
            cached is not None
            and cached[0] is m
            and cached[1] is v
            and cached[2] is p_local
            and (cached[3] is None or cached[3][7][0] == p_local.data_ptr())
        ):
            return cached[3]
        placements = p.placements if isinstance(p, DTensor) else None
        tensors = tuple(_local(t, placements) for t in (p, m, v))
        entry = None
        if all(t is not None for t in tensors) and len({t.numel() for t in tensors}) == 1:
            if state["step"].device.type == "cpu":
                entry = (
                    placements,
                    *tensors,
                    tensors[0].numel(),
                    is_batchable(tensors),
                    tuple(t.dtype for t in tensors),
                    tuple(t.data_ptr() for t in tensors),
                )
        self._locals_cache[p] = (m, v, p_local, entry)
        return entry

    def _step_value_map(self):
        if self._step_values is None:
            items = [(p, s["step"]) for p, s in self.state.items() if "step" in s]
            values = torch.stack([s for _, s in items]).tolist() if items else []
            self._step_values = {p: v for (p, _), v in zip(items, values)}
        return self._step_values

    def step(self, closure=None):
        if closure is not None or not all(self._group_supported(g) for g in self.param_groups):
            self._step_values = None
            return super().step(closure)

        step_values = self._step_value_map()
        for group_idx, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            hparams = dict(
                lr=group["lr"],
                beta1=beta1,
                beta2=beta2,
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                bias_correction=True,
                adam_w_mode=group.get("decoupled_weight_decay", True),
            )
            stepped, leftovers = [], []
            batches, launches = defaultdict(list), defaultdict(int)

            def launch(key):
                dtypes, value = key
                tensors, rows = zip(*batches.pop(key))
                cache = self._batch_tables[(group_idx, dtypes, launches[key])]
                triton_multi_tensor_adam_step_(
                    list(tensors), rows=list(rows), cache=cache, step=value, **hparams
                )
                launches[key] += 1

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                state = self._state(p)
                entry = self._locals(p, state)
                g_local = None if entry is None else _local(grad, entry[0])
                if g_local is None or g_local.numel() != entry[4]:
                    leftovers.append((p, state))
                    continue
                _, p_local, m_local, v_local, numel, batchable, (pd, md, vd), (pa, ma, va) = entry
                value = step_values.get(p, 0.0) + 1.0
                step_values[p] = value
                stepped.append(state["step"])
                tensors = (p_local, g_local, m_local, v_local)
                if batchable and is_batchable((g_local,)):
                    key = ((pd, g_local.dtype, md, vd), value)
                    batches[key].append((tensors, [pa, g_local.data_ptr(), ma, va, numel]))
                    if len(batches[key]) == _TENSORS_PER_BATCHED_LAUNCH:
                        launch(key)
                else:
                    triton_adam_step_(*tensors, step=value, grid_size=self.grid_size, **hparams)
            for key in list(batches):
                launch(key)
            if stepped:
                torch._foreach_add_(stepped, 1.0)
            if leftovers:
                self._torch_step(group, leftovers)
                for p, _ in leftovers:
                    step_values[p] = step_values.get(p, 0.0) + 1.0
        return None

    @staticmethod
    @torch.no_grad()
    def _torch_step(group, entries):
        params = [p for p, _ in entries]
        beta1, beta2 = group["betas"]
        torch_adam(
            params,
            [p.grad for p in params],
            [s["exp_avg"] for _, s in entries],
            [s["exp_avg_sq"] for _, s in entries],
            [],
            [s["step"] for _, s in entries],
            has_complex=any(torch.is_complex(p) for p in params),
            decoupled_weight_decay=group.get("decoupled_weight_decay", True),
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=False,
        )
