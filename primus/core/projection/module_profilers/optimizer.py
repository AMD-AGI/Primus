###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

from primus.core.projection.base_module_profiler import BaseModuleProfiler

# Adam mixed-precision memory traffic per parameter:
#   Read:  FP32 master param (4B) + FP32 grad (4B) + m (4B) + v (4B) = 16 B
#   Write: FP32 master param (4B) + m (4B) + v (4B) + BF16 param (2B) = 14 B
#   Total: 30 bytes per parameter
_ADAM_BYTES_PER_PARAM = 30


class OptimizerProfiler(BaseModuleProfiler):
    """Estimates optimizer step time for Adam/AdamW.

    The optimizer step is HBM-bandwidth-bound.  For each parameter, Adam
    reads/writes 30 bytes (mixed-precision training with BF16 forward, FP32
    master weights).  With ``distributed_optimizer`` or FSDP the optimizer
    state is sharded across DP ranks, so each GPU only updates
    ``N_params / dp_size`` parameters.

    This profiler requires a GEMM simulation backend to obtain HBM bandwidth
    (``hbm_bandwidth_gbps`` property) so the estimate automatically scales
    across architectures (MI300X 5.3 TB/s, MI325X 6.0 TB/s, MI355X 8.0 TB/s,
    etc.) without maintaining a separate lookup table.
    """

    def __init__(self, config, sub_profilers=None, gemm_backend=None):
        super().__init__(config, sub_profilers)
        self._gemm_backend = gemm_backend

    # ------------------------------------------------------------------
    # BaseModuleProfiler interface
    # ------------------------------------------------------------------

    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        """Return total parameter count on this GPU (post TP/PP/EP sharding)."""
        return self._count_params_per_gpu()

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        """Optimizer step has no activation memory footprint."""
        return 0

    # ------------------------------------------------------------------
    # Optimizer-specific API
    # ------------------------------------------------------------------

    def estimated_step_time_ms(
        self,
        dp_size: int = 1,
    ) -> float:
        """Estimate the optimizer step time in milliseconds.

        Args:
            dp_size: Data-parallel size. With distributed_optimizer or FSDP
                the optimizer state is sharded across DP ranks.

        Returns:
            Optimizer step time in milliseconds.
        """
        params_per_gpu = self._count_params_per_gpu()

        # --- Distributed optimizer / FSDP sharding ---
        mp_config = self.config.model_parallel_config
        use_distributed_optimizer = getattr(mp_config, "use_distributed_optimizer", False)
        use_fsdp = getattr(mp_config, "use_torch_fsdp2", False)

        if use_distributed_optimizer or use_fsdp:
            params_for_optim = params_per_gpu // max(dp_size, 1)
        else:
            params_for_optim = params_per_gpu

        # --- Compute time ---
        total_bytes = params_for_optim * _ADAM_BYTES_PER_PARAM

        hbm_bw_gbps = self._get_hbm_bandwidth_gbps()
        hbm_bw_bytes_per_ms = hbm_bw_gbps * 1e9 / 1e3  # bytes/ms

        return total_bytes / hbm_bw_bytes_per_ms

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_hbm_bandwidth_gbps(self) -> float:
        """Get HBM bandwidth from the GEMM backend.

        Raises:
            AssertionError: If no GEMM backend is set or the backend does not
                report HBM bandwidth.
        """
        assert self._gemm_backend is not None, (
            "OptimizerProfiler requires a GEMM simulation backend to obtain HBM bandwidth. "
            "Pass gemm_backend= to the constructor."
        )
        bw = self._gemm_backend.hbm_bandwidth_gbps
        assert bw is not None, (
            f"GEMM backend '{self._gemm_backend.name()}' does not report hbm_bandwidth_gbps. "
            "Ensure the target GPU architecture is specified via --gpu-arch or PRIMUS_GPU_ARCH."
        )
        return bw

    def _count_params_per_gpu(self) -> int:
        """Count total parameters per GPU after TP/PP/EP sharding."""
        model_config = self.config.model_config
        mp_config = self.config.model_parallel_config

        hidden = model_config.hidden_size
        ffn_hidden = model_config.ffn_hidden_size or (hidden * 4)
        moe_ffn = model_config.moe_ffn_hidden_size or ffn_hidden
        num_layers = model_config.num_layers
        num_experts = model_config.num_experts or 0
        moe_pattern = model_config.moe_pattern  # list of 0/1
        num_moe_layers = sum(1 for p in moe_pattern if p == 1)
        num_dense_layers = num_layers - num_moe_layers

        tp = mp_config.tensor_model_parallel_size
        pp = mp_config.pipeline_model_parallel_size
        ep = getattr(mp_config, "expert_model_parallel_size", 1) or 1

        # Attention params: Q, K, V, O -> 4 * h * h (per layer, sharded by TP)
        attn_params_per_layer = 4 * hidden * hidden // tp
        # Dense MLP: gate, up, down -> 3 * h * ffn (per layer, sharded by TP)
        dense_mlp_params_per_layer = 3 * hidden * ffn_hidden // tp
        # Expert MLP params per expert: 3 * h * moe_ffn (NOT sharded by TP normally)
        expert_tp = getattr(mp_config, "expert_tensor_parallel_size", None) or 1
        expert_mlp_params_per_expert = 3 * hidden * moe_ffn // expert_tp

        # Non-expert params across all layers (sharded by TP, PP)
        non_expert_params = num_layers * attn_params_per_layer + num_dense_layers * dense_mlp_params_per_layer
        # Expert params (sharded by EP, expert_TP, PP)
        expert_params = num_moe_layers * num_experts * expert_mlp_params_per_expert // max(ep, 1)

        # Shared experts (if any)
        shared_sz = getattr(model_config, "moe_shared_expert_intermediate_size", 0) or 0
        shared_expert_params = 0
        if shared_sz and num_moe_layers > 0:
            shared_expert_params = num_moe_layers * 3 * hidden * shared_sz // tp

        total_params_per_gpu = (non_expert_params + expert_params + shared_expert_params) // pp

        # Embedding + output layer params (only on first / last PP rank, amortise)
        vocab_size = getattr(model_config, "vocab_size", 0) or 0
        if vocab_size and pp > 0:
            embedding_params = vocab_size * hidden // tp
            output_params = vocab_size * hidden // tp
            # Amortise across PP ranks (only 1 rank holds each)
            total_params_per_gpu += (embedding_params + output_params) // pp

        return total_params_per_gpu
