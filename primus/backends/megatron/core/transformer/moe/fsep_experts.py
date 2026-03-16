###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
FSEP (Fully Sharded Expert Parallel) GroupedMLP.

Implements Expert parameter sharding along the FFN intermediate dimension,
with ReduceScatter instead of All-Reduce for output aggregation.

This enables load-balanced computation when Expert token assignment is skewed:
a hot Expert receiving T_hot >> T_avg tokens is computed in parallel across S
GPUs, each doing T_hot/S equivalent work, instead of one GPU doing all T_hot.

Reference: LAER-MoE (ASPLOS '26, arXiv:2602.11686)
"""

from typing import Optional

import torch
import torch.distributed as dist
from megatron.core import parallel_state, tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import get_args

from primus.backends.megatron.core.extensions.primus_turbo import (
    PrimusTurboGroupedMLP,
)
from primus.modules.module_utils import log_rank_0


# ---------------------------------------------------------------------------
# FSEP process group helpers
# ---------------------------------------------------------------------------

def get_fsep_group():
    """
    Return the process group for FSEP ReduceScatter/AllGather operations.

    FSEP reuses the expert_tensor_parallel_group as its sharding group.
    The initial constraint (enforced by fsep_args_validation patch) is:
        moe_fsep_sharding_degree == expert_tensor_parallel_size

    This means both concepts are identical and we reuse the existing group,
    avoiding the need for new process group initialization.
    """
    return parallel_state.get_expert_tensor_parallel_group()


def get_fsep_world_size() -> int:
    """Return world size of the FSEP group (equals moe_fsep_sharding_degree)."""
    return dist.get_world_size(group=get_fsep_group())


def get_fsep_rank() -> int:
    """Return current rank within the FSEP group."""
    return dist.get_rank(group=get_fsep_group())


# ---------------------------------------------------------------------------
# FSEPGroupedMLP
# ---------------------------------------------------------------------------

class FSEPGroupedMLP(PrimusTurboGroupedMLP):
    """
    Fully Sharded Expert Parallel GroupedMLP.

    Key difference from PrimusTurboGroupedMLP:
    - fc2 output aggregation uses ReduceScatter instead of All-Reduce.
    - Output shape: [T/S, H] per GPU instead of [T, H] per GPU.

    The Token Dispatcher's combine_preprocess must AllGather [T/S, H] → [T, H]
    before passing to DeepEP's _pre_combine. This is handled by
    PrimusTurboDeepEPTokenDispatcher when FSEP is enabled.

    Weight layout (same as Expert TP, since moe_fsep_sharding_degree == etp):
        weight1: [H, F/S]  (sharded along FFN intermediate dim)
        weight2: [F/S, H]  (sharded along FFN intermediate dim)
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        args = get_args()
        self.fsep_sharding_degree = args.moe_fsep_sharding_degree
        assert self.fsep_sharding_degree > 1, (
            "FSEPGroupedMLP requires moe_fsep_sharding_degree > 1"
        )

        super().__init__(num_local_experts, config, pg_collection)

        self.fsep_group = get_fsep_group()
        self.fsep_world_size = get_fsep_world_size()
        self.fsep_rank = get_fsep_rank()

        log_rank_0(
            f"[FSEP] FSEPGroupedMLP initialized: "
            f"num_local_experts={num_local_experts}, "
            f"sharding_degree={self.fsep_sharding_degree}, "
            f"weight1={tuple(self.weight1.shape)}, "
            f"weight2={tuple(self.weight2.shape)}"
        )

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """
        Forward pass with FSEP output aggregation.

        Currently uses All-Reduce (equivalent to Expert TP) to produce [T, H]
        output compatible with the existing token dispatcher. The ReduceScatter
        optimization (which produces [T/S, H] and requires dispatcher changes)
        will be enabled once the dispatcher integration is complete.

        Args:
            permuted_local_hidden_states: [T_local, H]
            tokens_per_expert: [num_local_experts] token counts
            permuted_probs: [T_local] routing probabilities

        Returns:
            (output, None) where output has shape [T_local, H].
        """
        # Step 1: Run fc1 → activation → fc2 without any collective.
        # Since weights are sharded along F (same as Expert TP), each GPU
        # computes a partial sum of the full fc2 output.
        partial_output = self._forward_no_reduce(
            permuted_local_hidden_states, tokens_per_expert, permuted_probs
        )

        # Step 2: Aggregate partial outputs across FSEP group.
        #
        # Phase 1 (current): Use All-Reduce to produce [T, H].
        #   - Identical to Expert TP semantics
        #   - Compatible with existing DeepEP token dispatcher
        #   - FSEP benefit: weights are sharded so each GPU computes less
        #
        # Phase 2 (future): Use ReduceScatter → [T/S, H], then modify
        #   dispatcher combine_preprocess to AllGather back to [T, H].
        #   - Reduces A2A gather bandwidth by S×
        #   - Requires dispatcher to handle [T/S, H] intermediate
        output = self._fsep_all_reduce(partial_output)

        return output, None

    def _forward_no_reduce(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run GEMM pipeline (fc1 → act → fc2) without final collective.

        Returns partial output [T_local, H] — the partial sum from this
        GPU's F-shard of the Expert weights. Multiple GPUs holding different
        F-shards must ReduceScatter to get the complete result.

        Handles all variants: FP8, activation_recompute, use_split_wgrad_op.
        """
        import primus_turbo.pytorch as pt
        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboLowPrecisionGlobalStateManager,
            use_split_wgrad_op,
        )

        args = get_args()

        # Apply routing probabilities on input if configured (topk=1 only)
        if self.config.moe_apply_probs_on_input:
            assert self.config.moe_router_topk == 1, (
                "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            )
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            permuted_probs = torch.ones_like(permuted_probs)

        # Handle zero-token case (keep gradient flow)
        if permuted_local_hidden_states.nelement() == 0:
            assert torch.count_nonzero(tokens_per_expert) == 0
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            if args.use_turbo_fused_act_with_probs:
                h = self.activation_func_with_probs(h, permuted_probs, tokens_per_expert)
            else:
                h = self.activation_func_with_probs(h, permuted_probs.unsqueeze(-1))
            return torch.matmul(h, w2)

        # Set up weight tensors and GEMM kwargs
        gemm_kargs = [dict(), dict()]
        if use_split_wgrad_op():
            w1 = self.weight1
            w2 = self.weight2
            gemm_kargs[0]["weight_reshape_size"] = (
                self.num_local_experts, self.config.hidden_size, -1
            )
            gemm_kargs[1]["weight_reshape_size"] = (
                self.num_local_experts, -1, self.config.hidden_size
            )
        else:
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

        tokens_per_expert = tokens_per_expert.to(w1.device)
        assert w1.is_contiguous(), "w1 must be contiguous"
        assert w2.is_contiguous(), "w2 must be contiguous"

        # fc1: [T, H] @ [num_experts, H, F/S] → [T, F/S]
        if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():
            quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
            fc1_output = pt.ops.grouped_gemm_fp8(
                permuted_local_hidden_states, w1, tokens_per_expert,
                trans_b=False, config=quant_config.data(),
            )
        else:
            fc1_output = self.grouped_gemm(
                permuted_local_hidden_states, w1, tokens_per_expert,
                trans_b=False, **(gemm_kargs[0])
            )

        # Activation (with optional recompute)
        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            if args.use_turbo_fused_act_with_probs:
                intermediate = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs,
                    fc1_output, permuted_probs, tokens_per_expert,
                )
            else:
                intermediate = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs,
                    fc1_output, permuted_probs.unsqueeze(-1),
                )
        else:
            if args.use_turbo_fused_act_with_probs:
                intermediate = self.activation_func_with_probs(
                    fc1_output, permuted_probs, tokens_per_expert
                )
            else:
                intermediate = self.activation_func_with_probs(
                    fc1_output, permuted_probs.unsqueeze(-1)
                )

        # fc2: [T, F/S] @ [num_experts, F/S, H] → [T, H] (partial sum)
        if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():
            quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
            # FP8 output may need cast before ReduceScatter
            partial_output = pt.ops.grouped_gemm_fp8(
                intermediate, w2, tokens_per_expert,
                trans_b=False, config=quant_config.data(),
            )
            # Cast to compute dtype for ReduceScatter (nccl requires float/bfloat16/float16)
            if partial_output.dtype not in (torch.float32, torch.bfloat16, torch.float16):
                partial_output = partial_output.to(torch.bfloat16)
        else:
            partial_output = self.grouped_gemm(
                intermediate, w2, tokens_per_expert,
                trans_b=False, **(gemm_kargs[1])
            )

        # Register recompute after fc2 (dW delayed for ZB/V-schedule)
        if self.activation_recompute:
            self.activation_checkpoint.discard_output_and_register_recompute(partial_output)

        return partial_output  # [T_local, H], partial sum across F-shard

    def _fsep_all_reduce(self, partial: torch.Tensor) -> torch.Tensor:
        """
        All-Reduce partial outputs across the FSEP group.

        This is the conservative aggregation strategy: each GPU gets the full
        [T, H] output after summing partial sums from all F-shard GPUs.
        Functionally identical to Expert TP's All-Reduce.

        Input:  partial [T, H]  — partial sum from this GPU's F-shard
        Output: [T, H]          — fully reduced output
        """
        S = self.fsep_world_size
        if S == 1:
            return partial

        if partial.nelement() == 0:
            return partial

        dist.all_reduce(partial, group=self.fsep_group)
        return partial

    def _fsep_reduce_scatter(self, partial: torch.Tensor) -> torch.Tensor:
        """
        ReduceScatter along the token dimension within fsep_group.

        Semantics:
          - Each of the S GPUs contributes partial[i] = X @ W_shard_i
          - Sum: full_output = Σ partial[i]  (this is what All-Reduce computes)
          - ReduceScatter: GPU_j gets full_output[T*j/S : T*(j+1)/S, :]

        Input:  partial [T, H]  — each GPU holds a different partial sum
        Output: [T/S, H]        — this GPU's share of the fully-reduced output

        Communication overlap: The ReduceScatter runs on the FSEP group which
        is intra-node (XGMI/NVLink), so it overlaps well with the subsequent
        inter-node A2A gather. The dispatcher can pipeline:
          Step 1: Expert GEMM → partial [T, H]    (compute stream)
          Step 2: ReduceScatter → [T/S, H]        (FSEP comm, intra-node)
          Step 3: A2A Gather → original tokens     (EP comm, inter-node)
        Steps 2 and 3 can overlap on different communication channels.

        Requires T % S == 0. The dispatcher pads tokens to multiples of S.
        """
        S = self.fsep_world_size
        T = partial.shape[0]

        if S == 1:
            return partial  # degenerate: no sharding

        if T == 0:
            return partial.new_zeros((0, partial.shape[1]))

        # Pad to multiple of S if needed (handles edge cases)
        if T % S != 0:
            pad_size = S - (T % S)
            partial = torch.nn.functional.pad(partial, (0, 0, 0, pad_size))
            T = partial.shape[0]

        output = torch.empty(
            T // S,
            partial.shape[1],
            dtype=partial.dtype,
            device=partial.device,
        )
        dist.reduce_scatter_tensor(output, partial, group=self.fsep_group)
        return output  # [T/S, H]

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """
        Reuse parent's sharded_state_dict since FSEP sharding degree == ETP.

        When moe_fsep_sharding_degree == expert_tensor_parallel_size (enforced
        by the validation patch), the weight tensors have identical shape and
        shard coordinates as Expert TP. No changes needed.
        """
        args = get_args()
        etp = parallel_state.get_expert_tensor_parallel_world_size()
        assert args.moe_fsep_sharding_degree == etp, (
            f"Checkpoint mismatch: moe_fsep_sharding_degree={args.moe_fsep_sharding_degree} "
            f"!= expert_tensor_parallel_size={etp}. "
            "This configuration is not yet supported for distributed checkpoint."
        )
        return super().sharded_state_dict(prefix, sharded_offsets, metadata)
