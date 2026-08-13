###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
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

        # Effective HBM utilisation of the optimizer step.  The step is NOT a
        # single streaming pass at peak: it is many small per-parameter-group
        # elementwise kernels (fp32 master/m/v scattered across buckets), which
        # sustain only a fraction of peak.  Anchored to the DeepSeek-V4 Pro
        # MI355X trace (measured Adam step 93.8 ms vs a 100%-peak 19.9 ms
        # estimate -> ~0.21).  Env-tunable.
        opt_hbm_eff = float(os.getenv("PRIMUS_OPT_HBM_EFF", "0.21"))
        step_ms = total_bytes / (hbm_bw_bytes_per_ms * opt_hbm_eff)

        # Muon adds Newton-Schulz orthogonalization (extra bf16 matmuls on every
        # 2D weight matrix) on top of the memory-bound state update.  This is
        # pure GEMM compute and is modelled from first principles below: the NS
        # iteration count and per-matrix matmul shapes are fully determined by
        # the model geometry and the algorithm, so each matmul is priced with
        # the GEMM backend (no fitted constant).  On the DeepSeek-V4 Pro trace
        # this is the single largest per-iteration cost (the fused expert
        # ``gate_up`` weight ``[2*moe_ffn, hidden]`` dominates), which is why the
        # old ``PRIMUS_OPT_MUON_NS_MS`` default of 0 grossly under-predicted the
        # Muon step.  Env override retained only as an escape hatch.
        if str(getattr(self.config.model_config, "optimizer", "") or "").lower() == "muon":
            override = os.getenv("PRIMUS_OPT_MUON_NS_MS")
            if override is not None:
                step_ms += float(override)
            else:
                ns_ms = self._muon_newton_schulz_ms()
                # Distributed Muon (dist_muon = muon + use_distributed_optimizer):
                # the momentum is sharded across the data-parallel group and each
                # rank orthogonalizes only its shard, so the Newton-Schulz compute
                # scales down by the group over which the (expert-dominated) weights
                # are replicated.  Expert matrices are replicated across
                # EDP = DP / EP, which dominates; use it as the sharding factor.
                if use_distributed_optimizer or use_fsdp:
                    ep = getattr(mp_config, "expert_model_parallel_size", 1) or 1
                    edp = max(1, dp_size // max(1, ep))
                    ns_ms /= edp
                step_ms += ns_ms

        return step_ms

    # ------------------------------------------------------------------
    # Muon Newton-Schulz orthogonalization compute (first-principles)
    # ------------------------------------------------------------------

    def _muon_num_ns_steps(self) -> int:
        """Number of Newton-Schulz iterations per orthogonalization.

        Read from the config when plumbed; otherwise use the algorithm default
        (DeepSeek-V4's quintic schedule uses 10 iterations, the generic Muon
        default is 5).  This is an algorithm hyper-parameter, not a fitted
        calibration constant.
        """
        env = os.getenv("PRIMUS_OPT_MUON_NS_STEPS")
        if env is not None:
            return int(env)
        cfg = self.config.model_config
        steps = getattr(cfg, "muon_num_ns_steps", 0) or 0
        if steps:
            return int(steps)
        is_v4 = getattr(cfg, "compress_ratios", None) is not None
        return 10 if is_v4 else 5

    def _muon_weight_shapes(self):
        """Yield ``(rows, cols, count)`` for every 2D weight Muon orthogonalizes.

        Muon orthogonalizes the 2D linear weight matrices (attention Q/KV/O
        projections, dense-MLP and MoE-expert gate/up/down); 1D params, norms,
        embeddings and the output layer are handled by Adam and excluded here.
        Shapes follow the model geometry (sharding is ignored: with Muon the
        distributed optimizer is not used, so each rank orthogonalizes the full
        matrices it owns; on a single-GPU proxy that is all of them).
        """
        m = self.config.model_config
        mp = self.config.model_parallel_config

        hidden = m.hidden_size
        ffn = m.ffn_hidden_size or (hidden * 4)
        moe_ffn = m.moe_ffn_hidden_size or ffn
        heads = m.num_attention_heads
        hd = m.kv_channels or (hidden // max(heads, 1))
        n_d = heads * hd
        q_lora = getattr(m, "q_lora_rank", 0) or 0
        kv_lora = getattr(m, "kv_lora_rank", 0) or 0
        o_lora = getattr(m, "o_lora_rank", 0) or 0
        o_groups = max(1, getattr(m, "o_groups", 1) or 1)
        swiglu = bool(getattr(m, "swiglu", False))
        gate_up_mult = 2 if swiglu else 1

        num_layers = m.num_layers
        moe_pattern = m.moe_pattern or [0] * num_layers
        num_moe_layers = sum(1 for p in moe_pattern if p == 1)
        num_dense_layers = num_layers - num_moe_layers

        # --- Pipeline sharding (L1 fix) ---
        # Muon's Newton-Schulz step runs independently on every PP rank over only
        # the weights that rank owns; the per-iteration optimizer wall-clock is
        # therefore the slowest (max-loaded) stage.  Previously these counts used
        # the FULL model layer count, so the Muon step did not shrink with PP and
        # dominated the projected iteration time.  Use the critical-
        # stage layer count ceil(num_layers / pp): for balanced / near-balanced
        # layouts (e.g. DSv4-Pro 61 layers over PP8=7,7,8,8,8,8,8,7 or
        # PP16=3,3,4..4,3) this equals the max stage's layer count, and it stays
        # consistent with _count_params_per_gpu's per-rank (// pp) sharding.
        pp = getattr(mp, "pipeline_model_parallel_size", 1) or 1
        if pp > 1 and num_layers > 0:
            layers_per_rank = (num_layers + pp - 1) // pp  # ceil = critical stage
            moe_frac = num_moe_layers / num_layers
            num_moe_layers = moe_frac * layers_per_rank
            num_dense_layers = layers_per_rank - num_moe_layers
            num_layers = layers_per_rank  # attention projections are per-layer

        ep = getattr(mp, "expert_model_parallel_size", 1) or 1
        num_experts = m.num_experts or 0
        experts_per_gpu = (num_experts // max(ep, 1)) if num_experts else 0

        shapes = []  # (rows, cols, count)

        # ---- Attention projections (per layer) ----
        # V4 uses LoRA-factored Q/KV and a grouped LoRA O projection; fall back
        # to dense projections when the LoRA ranks are not configured.
        if q_lora > 0:
            shapes.append((q_lora, hidden, num_layers))  # Q down
            shapes.append((n_d, q_lora, num_layers))  # Q up
        else:
            shapes.append((n_d, hidden, num_layers))  # Q
        if kv_lora > 0:
            shapes.append((kv_lora, hidden, num_layers))  # KV down
            shapes.append((n_d, kv_lora, num_layers))  # KV up
        else:
            shapes.append((n_d, hidden, num_layers))  # KV
        if o_lora > 0:
            shapes.append((o_groups * o_lora, n_d, num_layers))  # O down
            shapes.append((hidden, o_groups * o_lora, num_layers))  # O up
        else:
            shapes.append((hidden, n_d, num_layers))  # O

        # ---- Dense-MLP layers ----
        if num_dense_layers > 0:
            shapes.append((gate_up_mult * ffn, hidden, num_dense_layers))  # gate/up
            shapes.append((hidden, ffn, num_dense_layers))  # down

        # ---- MoE expert layers (routed experts + shared expert) ----
        if num_moe_layers > 0 and experts_per_gpu > 0:
            shapes.append((gate_up_mult * moe_ffn, hidden, num_moe_layers * experts_per_gpu))
            shapes.append((hidden, moe_ffn, num_moe_layers * experts_per_gpu))
        shared_sz = getattr(m, "moe_shared_expert_intermediate_size", 0) or 0
        if num_moe_layers > 0 and shared_sz:
            shapes.append((gate_up_mult * shared_sz, hidden, num_moe_layers))
            shapes.append((hidden, shared_sz, num_moe_layers))

        return shapes

    def _muon_newton_schulz_ms(self) -> float:
        """First-principles Muon Newton-Schulz compute time (ms) for one step.

        Each 2D weight ``W`` (reduced to ``[m, n]`` with ``m = min(dim)``,
        ``n = max(dim)``) is orthogonalized with ``num_ns_steps`` quintic
        Newton-Schulz iterations.  Each iteration is three bf16 matmuls::

            A = X @ Xᵀ      -> (m, m, k=n)
            AA = A @ A       -> (m, m, k=m)
            X  = ... + B @ X -> (m, n, k=m)

        priced individually with the GEMM backend, so the per-matmul efficiency
        (tile/wave quantization, small-K overheads) comes from the same model
        used everywhere else — no calibration constant.
        """
        if self._gemm_backend is None:
            return 0.0
        steps = self._muon_num_ns_steps()
        if steps <= 0:
            return 0.0

        total_ms = 0.0
        for rows, cols, count in self._muon_weight_shapes():
            if rows <= 0 or cols <= 0 or count <= 0:
                continue
            mm = min(rows, cols)
            nn = max(rows, cols)
            try:
                t_xxt = self._gemm_backend.simulate_gemm(mm, mm, nn, "bf16").forward_time_ms
                t_aa = self._gemm_backend.simulate_gemm(mm, mm, mm, "bf16").forward_time_ms
                t_bx = self._gemm_backend.simulate_gemm(mm, nn, mm, "bf16").forward_time_ms
            except Exception:
                continue
            total_ms += count * steps * (t_xxt + t_aa + t_bx)
        return total_ms

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
        vocab_size = getattr(model_config, "padded_vocab_size", 0) or 0
        if vocab_size and pp > 0:
            embedding_params = vocab_size * hidden // tp
            output_params = vocab_size * hidden // tp
            # Amortise across PP ranks (only 1 rank holds each)
            total_params_per_gpu += (embedding_params + output_params) // pp

        return total_params_per_gpu
