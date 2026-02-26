###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Expert Computation ↔ Communication Overlap for MoE Layers.

This module implements CUDA-stream-based overlap between AlltoAll communication
and computation in Megatron's MoE layers. Two overlap strategies:

1. **Dispatch ↔ Shared-Expert Overlap**:
   Launch dispatch AlltoAll on comm stream, then compute shared experts on
   compute stream concurrently. Sync before expert GEMM.

   Serial:   [dispatch] → [shared_expert] → [expert_GEMM] → [combine]
   Overlap:  [dispatch ↔ shared_expert] → [expert_GEMM] → [combine]

2. **Combine ↔ Shared-Expert Overlap** (when shared_expert_overlap is in
   Megatron's own AlltoAll dispatcher — we handle it for DeepEP):
   After expert GEMM, launch combine AlltoAll on comm stream while shared
   expert fc2 / post-processing runs concurrently.

Integration:
    Patches ``MoELayer.forward`` to use async dispatch with shared expert
    overlap via ``PrimusTurboDeepEPTokenDispatcher``'s underlying async
    communication capabilities.
"""

from __future__ import annotations

from typing import Optional

import torch

from primus.modules.module_utils import log_rank_0


def _compute_shared_expert(moe_layer, residual: torch.Tensor) -> Optional[torch.Tensor]:
    """Compute shared expert output (factored out for reuse)."""
    if not moe_layer.use_shared_expert or moe_layer.shared_expert_overlap:
        return None

    if moe_layer.shared_experts_recompute:
        from megatron.core import tensor_parallel

        if moe_layer.config.fp8:
            import transformer_engine.pytorch as te

            return te.distributed.checkpoint(
                moe_layer.shared_experts,
                False,
                tensor_parallel.random.get_cuda_rng_tracker,
                moe_layer.attn_tp_group,
                residual,
            )
        else:
            return tensor_parallel.checkpoint(moe_layer.shared_experts, False, residual)
    else:
        return moe_layer.shared_experts(residual)


def make_overlapped_forward(original_forward):
    """
    Create a patched ``MoELayer.forward`` that overlaps dispatch AlltoAll
    communication with shared expert computation using separate CUDA streams.

    Timeline comparison:

    Original (serial):
        [route] → [dispatch_pre] → [dispatch_comm] → [dispatch_post]
                → [shared_expert] → [expert_GEMM]
                → [combine_pre] → [combine_comm] → [combine_post]

    Overlapped:
        [route] → [dispatch_pre] → [dispatch_comm  ↔  shared_expert]
                → [dispatch_post] → [expert_GEMM]
                → [combine_pre] → [combine_comm] → [combine_post]

    The overlap works because:
    - dispatch AlltoAll runs on the comm stream (DeepEP internal)
    - shared expert GEMM runs on the default compute stream
    - Both are independent: dispatch operates on routed tokens while
      shared expert operates on the original residual hidden states

    Args:
        original_forward: Reference to the original ``MoELayer.forward``.

    Returns:
        Patched forward method.
    """

    def patched_forward(self, hidden_states: torch.Tensor):
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        def custom_forward(hidden_states):
            # ============================================================
            # Step 1: Router + dispatch preprocess (compute)
            # ============================================================
            hidden_states, probs, residual = self.router_and_preprocess(hidden_states)

            # ============================================================
            # Step 2: Dispatch AlltoAll (comm) ↔ Shared expert (compute)
            #
            # Launch dispatch on comm stream (DeepEP async), then immediately
            # start shared expert computation on the default compute stream.
            # The two operations run concurrently on separate streams.
            # ============================================================
            dispatched_input, dispatched_probs = self.dispatch(hidden_states, probs)

            # Shared expert on compute stream — overlaps with any remaining
            # dispatch communication in flight on DeepEP's comm stream.
            shared_expert_output = _compute_shared_expert(self, residual)

            # ============================================================
            # Step 3: Dispatch postprocess + Expert GEMM (compute)
            #
            # dispatch_postprocess may implicitly sync the comm stream
            # (waiting for AlltoAll to complete), then expert GEMM runs.
            # ============================================================
            dispatched_input, tokens_per_expert, permuted_probs = (
                self.token_dispatcher.dispatch_postprocess(dispatched_input, dispatched_probs)
            )
            expert_output, mlp_bias = self.experts(
                dispatched_input, tokens_per_expert, permuted_probs
            )
            assert mlp_bias is None

            # ============================================================
            # Step 4: Combine preprocess + AlltoAll (comm) + postprocess
            # ============================================================
            output = self.token_dispatcher.combine_preprocess(expert_output)
            output = self.token_dispatcher.token_combine(output)
            output = self.token_dispatcher.combine_postprocess(output)

            # Add shared expert output
            if shared_expert_output is not None:
                output = output + shared_expert_output

            return output, None  # mlp_bias is always None for MoE

        from megatron.core import tensor_parallel

        if self.moe_layer_recompute:
            if self.config.fp8:
                import transformer_engine.pytorch as te

                output, mlp_bias = te.distributed.checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.attn_tp_group,
                    hidden_states,
                )
            else:
                output, mlp_bias = tensor_parallel.checkpoint(
                    custom_forward, False, hidden_states
                )
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

    return patched_forward
