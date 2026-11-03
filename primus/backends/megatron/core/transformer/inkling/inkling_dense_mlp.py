###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling dense SwiGLU MLP (HF ``InklingMLP``; used by the multi-token-prediction blocks).

``down(silu(gate(x)) * up(x)) * global_scale``: no biases, and a learned scalar
``global_scale`` (HF ``use_global_scale``).

Built on Megatron's :class:`~megatron.core.transformer.mlp.MLP`, so it inherits the backend's
tensor parallelism and checkpoint layout:

* ``linear_fc1`` is a ColumnParallelLinear of width ``2F``. Each TP rank holds
  ``cat(gate_shard, up_shard)`` (stride 2), and the sharded state dict splits gate / up so a
  checkpoint reshards across TP sizes.
* ``linear_fc2`` is a RowParallelLinear (``F -> H``) that all-reduces the output over TP.

The HF checkpoint stores gate and up interleaved in ``w13_dn`` (even rows gate, odd rows up).
The converter de-interleaves them into ``linear_fc1.weight = cat(gate, up)``.

The differences from the base class are deliberate:

* Width: the base reads ``config.ffn_hidden_size`` for ``linear_fc2``. For Inkling that is the
  MoE expert width, so the MLP is built on a shallow config copy whose ``ffn_hidden_size`` is
  ``inkling_dense_intermediate_size``.
* Precision: ``forward`` computes the SwiGLU in fp32 and casts back, as SGLang's ``swiglu``.
  The GEMMs stay in the activation dtype. ``global_scale`` multiplies the ``linear_fc2``
  output (after its TP all-reduce).
* Init: the weights are drawn from a CPU generator seeded per (layer, slot), on the full
  ``[2F, H]`` / ``[H, F]`` tensors, and each TP rank copies its own slice. This makes the
  initial model the same for every parallel layout, as the rest of the Inkling port.
* Checkpoint: ``global_scale`` is a TP-replicated parameter. The base sharded state dict only
  visits submodules, so it is added here.
"""

import copy

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
)
from megatron.core.utils import get_pg_rank, get_pg_size

from primus.backends.megatron.core.transformer.inkling.inkling_layers import (
    init_seed,
    seeded_normal,
)


def _swiglu_fp32(gate_up):
    """SwiGLU over the local ``cat(gate, up)`` columns, computed in fp32 (SGLang's ``swiglu``)."""
    gate, up = gate_up.float().chunk(2, dim=-1)
    return (F.silu(gate) * up).to(gate_up.dtype)


class InklingDenseMLP(MLP):
    """Dense SwiGLU MLP with a learned output scale, on Megatron's MLP (module docstring).

    ``forward`` returns ``(out, None)``, Megatron's ``(output, bias)`` convention. With
    selective recompute, the module ``mlp`` recomputes the whole MLP. ``moe_act`` is ignored
    here, because it names the routed experts' SwiGLU."""

    def __init__(self, config, inkling_layer_idx, ffn_hidden_size=None, tp_group=None):
        F_ = int(ffn_hidden_size or config.inkling_dense_intermediate_size)
        if not config.gated_linear_unit or config.add_bias_linear:
            raise ValueError("InklingDenseMLP: requires a gated linear unit (swiglu) without biases")
        mlp_config = copy.copy(config)  # only the MLP sees the dense width (module docstring)
        mlp_config.ffn_hidden_size = F_
        super().__init__(
            config=mlp_config,
            submodules=MLPSubmodules(
                linear_fc1=tensor_parallel.ColumnParallelLinear,
                linear_fc2=tensor_parallel.RowParallelLinear,
            ),
            ffn_hidden_size=F_,
            tp_group=tp_group,
        )
        self.ffn_hidden_size = F_
        H = config.hidden_size
        if config.perform_initialization:
            std, base = config.init_method_std, config.inkling_init_seed
            gate_up = seeded_normal(
                (2 * F_, H), std, init_seed(base, inkling_layer_idx, "dense_gate_up_proj")
            )
            down = seeded_normal((H, F_), std, init_seed(base, inkling_layer_idx, "dense_down_proj"))
            tp, rank = get_pg_size(self.tp_group), get_pg_rank(self.tp_group)
            part = F_ // tp
            rows = slice(rank * part, (rank + 1) * part)
            gate, up = gate_up.chunk(2, 0)
            with torch.no_grad():
                self.linear_fc1.weight.copy_(torch.cat((gate[rows], up[rows]), 0))
                self.linear_fc2.weight.copy_(down[:, rows])
        self.use_global_scale = bool(config.inkling_dense_use_global_scale)
        if self.use_global_scale:
            self.global_scale = torch.nn.Parameter(torch.ones(1))
        selective = config.recompute_granularity == "selective"
        self.recompute = selective and "mlp" in (config.recompute_modules or [])

    def _forward(self, x):
        h, _ = self.linear_fc1(x)
        out, _ = self.linear_fc2(_swiglu_fp32(h))
        if self.use_global_scale:
            out = out * self.global_scale.to(out.dtype)
        return out

    def forward(self, hidden_states, *_args, padding_mask=None, **_kwargs):
        del padding_mask  # a dense MLP routes nothing: every token is processed
        if self.recompute and self.training and hidden_states.requires_grad:
            return tensor_parallel.checkpoint(self._forward, False, hidden_states), None
        return self._forward(hidden_states), None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """The base's (TP-sharded fc1 / fc2, with gate / up split), plus the TP-replicated
        ``global_scale``."""
        sd = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if self.use_global_scale:
            metadata = ensure_metadata_has_dp_cp_group(metadata)
            sd.update(
                make_sharded_tensors_for_checkpoint(
                    {"global_scale": self.global_scale},
                    prefix,
                    {},
                    sharded_offsets,
                    tp_group=self.tp_group,
                    dp_cp_group=metadata["dp_cp_group"],
                )
            )
        return sd
