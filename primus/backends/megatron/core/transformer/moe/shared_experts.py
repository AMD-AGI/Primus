###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus shared-expert MLP with fused (clamped) SwiGLU.

Megatron's stock :class:`SharedExpertMLP` reaches the fused SwiGLU path via
``mlp.MLP.forward`` only through ``bias_swiglu_impl``, which does **not**
support DeepSeek-V4's pre-multiplication clamp. To keep the clamp correct,
``v4_moe`` disables ``bias_activation_fusion`` for the shared expert, which
forces the un-fused eager ``chunk``/``clamp``/``SiLU``/``mul`` path.

:class:`PrimusSharedExpertMLP` overrides the activation to call Primus's fused
clamped SwiGLU Triton kernel (:func:`swiglu_impl`), mirroring what
``PrimusGroupedMLP`` does for the routed experts. Both the normal ``forward``
and the ``--moe-shared-expert-overlap`` ``linear_fc1_forward_and_act`` paths
are covered so the clamp semantics stay identical.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from megatron.core.transformer.moe.shared_experts import (
    SharedExpertMLP,
    set_tensor_grad_fn_sequence_sr,
)
from megatron.core.typed_torch import apply_module
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

from primus.backends.megatron.core.fusions.fused_bias_swiglu import swiglu_impl


class PrimusSharedExpertMLP(SharedExpertMLP):
    """Shared-expert MLP that fuses the (clamped) SwiGLU activation."""

    def _can_fuse_swiglu(self) -> bool:
        return (
            not self.config.use_te_activation_func
            and self.config.gated_linear_unit
            and self.activation_func == F.silu
        )

    def _fused_swiglu(self, intermediate_parallel, bias_parallel):
        # dtype and the pre-mul clamp are handled inside the fused kernel.
        return swiglu_impl(
            intermediate_parallel,
            bias_parallel,
            self.config.activation_func_fp8_input_store,
            self.config.activation_func_clamp_value,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward with fused clamped SwiGLU (non-overlap path)."""
        if not self._can_fuse_swiglu():
            return super().forward(hidden_states)

        nvtx_range_push(suffix="linear_fc1")
        intermediate_parallel, bias_parallel = apply_module(self.linear_fc1)(hidden_states)
        nvtx_range_pop(suffix="linear_fc1")

        nvtx_range_push(suffix="activation")
        intermediate_parallel = self._fused_swiglu(intermediate_parallel, bias_parallel)
        nvtx_range_pop(suffix="activation")

        nvtx_range_push(suffix="linear_fc2")
        output, _ = apply_module(self.linear_fc2)(intermediate_parallel)
        nvtx_range_pop(suffix="linear_fc2")

        if self.use_shared_expert_gate:
            logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
            gate_score = torch.nn.functional.sigmoid(logits)
            output = output * gate_score
        return output

    def linear_fc1_forward_and_act(self, overlapped_comm_output=None):
        """Overlap-path FC1 + fused clamped SwiGLU activation."""
        if not self._can_fuse_swiglu():
            return super().linear_fc1_forward_and_act(overlapped_comm_output)

        assert self.config.moe_shared_expert_overlap
        assert self.cached_fc1_input is not None
        if overlapped_comm_output is not None:
            set_tensor_grad_fn_sequence_sr(overlapped_comm_output, torch.iinfo(torch.int).max)
        with torch.cuda.stream(self.stream):
            # [s, b, 4 * h/p]
            intermediate_parallel, bias_parallel = apply_module(self.linear_fc1)(self.cached_fc1_input)
            self.cached_fc1_input = None
            self.cached_fc2_input = self._fused_swiglu(intermediate_parallel, bias_parallel)
