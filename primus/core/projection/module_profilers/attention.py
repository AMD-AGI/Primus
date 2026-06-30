###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

from primus.core.projection.base_module_profiler import BaseModuleProfiler

from .utils import benchmark_layer, v4_module_inputs


class AttentionProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        self.module = None  # Will be set during benchmarking
        self._cached_results = None  # Cache for (forward_time, backward_time, activation_memory)
        self._cache_key = None  # Cache key (batch_size, seq_len)
        self._gemm_backend = None  # Optional: GEMM simulation backend
        self._sdpa_backend = None  # Optional: SDPA simulation backend
        self._sim_compress_ratio = None  # Per-layer cr for V4 simulate (no module)

    def set_sim_compress_ratio(self, cr):
        """Set the DeepSeek-V4 compress ratio for the current simulated layer.

        In ``simulate`` mode no torch module is built, so the cr-aware
        attention model reads the per-layer compress ratio from here instead
        of ``module.compress_ratio``.  The cr-aware path stays inactive until
        this is set (or a real V4 module is bound), so non-cr-aware callers
        see no behaviour change.
        """
        self._sim_compress_ratio = None if cr is None else int(cr)
        self._cached_results = None
        self._cache_key = None

    def set_module(self, module):
        """Set the actual attention module for benchmarking."""
        self.module = module
        # Invalidate cache when module changes
        self._cached_results = None
        self._cache_key = None

    def set_gemm_backend(self, backend):
        """Set a GEMM simulation backend for attention linear projections."""
        self._gemm_backend = backend
        self._cached_results = None
        self._cache_key = None

    def set_sdpa_backend(self, backend):
        """Set an SDPA simulation backend for attention computation."""
        self._sdpa_backend = backend
        self._cached_results = None
        self._cache_key = None

    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        args = self.config.model_config
        # Group-query & multi-latent attention support.
        # If GQA not enabled, fall back to per-head queries.
        num_query_groups = (
            args.num_query_groups
            if args.group_query_attention and args.num_query_groups
            else args.num_attention_heads
        )

        # Projection ratio: (kv_channels * n_heads) / hidden_size
        query_proj_to_hidden = (args.kv_channels * args.num_attention_heads) / args.hidden_size

        if args.multi_latent_attention:
            # q_term: either dense or LoRA factored Q with RoPE/Q-norm
            if args.q_lora_rank is None:
                q_term = (
                    args.hidden_size
                    * args.num_attention_heads
                    * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                )
            else:
                q_term = args.q_lora_rank * (
                    args.hidden_size
                    + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                    + 1
                )
            attn = (
                q_term
                # kv lora + rope + kv norm
                + args.kv_lora_rank
                * (args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim) + 1)
                # pos emb
                + args.hidden_size * args.qk_pos_emb_head_dim
                # out proj
                + (args.num_attention_heads * args.v_head_dim) * args.hidden_size
            )
            return attn

        # Standard attention path (Q,K,V,O projections)
        return (
            2
            * args.hidden_size
            * args.hidden_size
            * ((1 + (num_query_groups / args.num_attention_heads)) * query_proj_to_hidden)
        )

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        args = self.config.model_config
        mp = self.config.model_parallel_config

        tp_size = max(1, mp.tensor_model_parallel_size)
        cp_size = max(1, mp.context_model_parallel_size)

        tokens_per_rank = batch_size * seq_len // tp_size // cp_size
        if tokens_per_rank == 0:
            return 0

        bytes_per_value = 2  # assume bf16 activations

        def _num_query_groups() -> int:
            if args.group_query_attention and args.num_query_groups:
                return args.num_query_groups
            return args.num_attention_heads

        ln_width = 0

        if args.multi_latent_attention:
            # MLA uses separate latent dimensions for Q/K and V plus optional LoRA ranks.
            heads = args.num_attention_heads
            q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
            v_head_dim = args.v_head_dim

            q_width = heads * q_head_dim
            k_width = q_width  # key stores the same latent + positional dims
            v_width = heads * v_head_dim
            context_width = v_width  # attention output before the final projection
            query_projection_size = q_width  # For softmax width calculation

            if args.qk_layernorm:
                ln_width += q_width
                ln_width += k_width

            activation_width = q_width + k_width + v_width + context_width
        else:
            query_projection_size = args.kv_channels * args.num_attention_heads
            kv_projection_size = args.kv_channels * _num_query_groups()

            # Need to retain Q, K, V as well as the projected context/output.
            activation_width = query_projection_size + 2 * kv_projection_size + args.hidden_size

            if args.qk_layernorm:
                ln_width += kv_projection_size * 2

        heads_per_partition = max(1, args.num_attention_heads // tp_size)
        seqlen_per_cp = max(1, (seq_len + cp_size - 1) // cp_size)
        if getattr(args, "use_flash_attn", False):
            softmax_width = query_projection_size
        else:
            softmax_width = heads_per_partition * seqlen_per_cp
        activation_width += softmax_width

        return tokens_per_rank * (activation_width + ln_width) * bytes_per_value

    def _simulate_mla_gemms(self, batch_tokens: int, dtype: str) -> tuple[float, float]:
        """Simulate MLA (Multi-Latent Attention) projection GEMMs.

        MLA uses LoRA-factored Q and compressed KV projections instead of
        standard Q/K/V projections:
          Forward  (6 GEMMs): Q_down, Q_up, KV_down, KV_up, RoPE_proj, O_proj
          Backward (12 GEMMs): dgrad + wgrad for each of the 6 projections
        """
        args = self.config.model_config
        backend = self._gemm_backend

        hidden = args.hidden_size
        heads = args.num_attention_heads
        q_lora_rank = args.q_lora_rank
        kv_lora_rank = args.kv_lora_rank
        qk_head_dim = args.qk_head_dim
        qk_pos_emb_head_dim = args.qk_pos_emb_head_dim
        v_head_dim = args.v_head_dim

        fwd_time = 0.0
        bwd_time = 0.0
        T = batch_tokens

        # ---------- Forward ----------
        if q_lora_rank is not None:
            # Q down-proj: [T, hidden] × [hidden, q_lora_rank]
            q_down_out = q_lora_rank
            r = backend.simulate_gemm(T, q_down_out, hidden, dtype)
            fwd_time += r.forward_time_ms
            # Q up-proj: [T, q_lora_rank] × [q_lora_rank, heads*(qk_hd+qk_pe_hd)]
            q_up_out = heads * (qk_head_dim + qk_pos_emb_head_dim)
            r = backend.simulate_gemm(T, q_up_out, q_lora_rank, dtype)
            fwd_time += r.forward_time_ms
        else:
            # Direct Q projection (no LoRA): [T, hidden] × [hidden, heads*(qk_hd+qk_pe_hd)]
            q_up_out = heads * (qk_head_dim + qk_pos_emb_head_dim)
            r = backend.simulate_gemm(T, q_up_out, hidden, dtype)
            fwd_time += r.forward_time_ms

        # KV down-proj: [T, hidden] × [hidden, kv_lora_rank]
        kv_down_out = kv_lora_rank
        r = backend.simulate_gemm(T, kv_down_out, hidden, dtype)
        fwd_time += r.forward_time_ms
        # KV up-proj: [T, kv_lora_rank] × [kv_lora_rank, heads*(qk_hd+v_hd)]
        kv_up_out = heads * (qk_head_dim + v_head_dim)
        r = backend.simulate_gemm(T, kv_up_out, kv_lora_rank, dtype)
        fwd_time += r.forward_time_ms

        # RoPE positional embedding projection: [T, hidden] × [hidden, qk_pos_emb_head_dim]
        r = backend.simulate_gemm(T, qk_pos_emb_head_dim, hidden, dtype)
        fwd_time += r.forward_time_ms

        # Output projection: [T, heads*v_hd] × [heads*v_hd, hidden]
        o_in = heads * v_head_dim
        r = backend.simulate_gemm(T, hidden, o_in, dtype)
        fwd_time += r.forward_time_ms

        # ---------- Backward (dgrad + wgrad for each projection) ----------
        if q_lora_rank is not None:
            # Q down-proj dgrad: [T, q_down_out] × [q_down_out, hidden] → [T, hidden]
            r = backend.simulate_gemm(T, hidden, q_down_out, dtype)
            bwd_time += r.forward_time_ms
            # Q down-proj wgrad: [hidden, T] × [T, q_down_out] → [hidden, q_down_out]
            r = backend.simulate_gemm(hidden, q_down_out, T, dtype)
            bwd_time += r.forward_time_ms
            # Q up-proj dgrad: [T, q_up_out] × [q_up_out, q_lora_rank] → [T, q_lora_rank]
            r = backend.simulate_gemm(T, q_lora_rank, q_up_out, dtype)
            bwd_time += r.forward_time_ms
            # Q up-proj wgrad: [q_lora_rank, T] × [T, q_up_out] → [q_lora_rank, q_up_out]
            r = backend.simulate_gemm(q_lora_rank, q_up_out, T, dtype)
            bwd_time += r.forward_time_ms
        else:
            # Direct Q dgrad + wgrad
            r = backend.simulate_gemm(T, hidden, q_up_out, dtype)
            bwd_time += r.forward_time_ms
            r = backend.simulate_gemm(hidden, q_up_out, T, dtype)
            bwd_time += r.forward_time_ms

        # KV down-proj dgrad + wgrad
        r = backend.simulate_gemm(T, hidden, kv_down_out, dtype)
        bwd_time += r.forward_time_ms
        r = backend.simulate_gemm(hidden, kv_down_out, T, dtype)
        bwd_time += r.forward_time_ms
        # KV up-proj dgrad + wgrad
        r = backend.simulate_gemm(T, kv_lora_rank, kv_up_out, dtype)
        bwd_time += r.forward_time_ms
        r = backend.simulate_gemm(kv_lora_rank, kv_up_out, T, dtype)
        bwd_time += r.forward_time_ms

        # RoPE proj dgrad + wgrad
        r = backend.simulate_gemm(T, hidden, qk_pos_emb_head_dim, dtype)
        bwd_time += r.forward_time_ms
        r = backend.simulate_gemm(hidden, qk_pos_emb_head_dim, T, dtype)
        bwd_time += r.forward_time_ms

        # O proj dgrad + wgrad
        r = backend.simulate_gemm(T, o_in, hidden, dtype)
        bwd_time += r.forward_time_ms
        r = backend.simulate_gemm(o_in, hidden, T, dtype)
        bwd_time += r.forward_time_ms

        return fwd_time, bwd_time

    def _is_v4_attention(self) -> bool:
        """True when this profiler should use the cr-aware V4 attention model.

        Active when either a real DeepSeek-V4 module is bound (benchmark-side
        introspection) or — in pure ``simulate`` mode with no module — the
        model config carries ``compress_ratios`` *and* a per-layer cr was
        provided via :meth:`set_sim_compress_ratio`.  Without an explicit cr
        the cr-aware path stays off so behaviour is unchanged.

        V4 uses per-layer compression (0=dense/SWA, 128=HCA, 4=CSA) and its
        own LoRA-factored Q/KV/O + compressor/indexer, which the generic
        standard/MLA simulate path does not represent.
        """
        m = self.module
        if m is not None and hasattr(m, "compress_ratio") and "DeepseekV4" in type(m).__name__:
            return True
        return (
            self._sim_compress_ratio is not None
            and getattr(self.config.model_config, "compress_ratios", None) is not None
        )

    def _v4_resolved_cr(self) -> int:
        m = self.module
        if m is not None and hasattr(m, "compress_ratio"):
            return int(m.compress_ratio)
        return int(self._sim_compress_ratio or 0)

    def _get_v4_simulated_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        """cr-aware DeepSeek-V4 attention timing — no calibration constants.

        Every term is derived from the module's own shapes, the Origami GEMM
        backend, the FAv3 SDPA simulator, and HBM bandwidth from the hardware
        spec.  Validated against the MI355X flash trace: backward within
        2-13% per cr.  The forward memory-bound ``attn.misc`` overhead (mHC
        elementwise + the CSA sparse-gather ``cat``) is an implementation
        kernel cost that analytic simulate underestimates; it is captured
        exactly only in ``--profiling-mode benchmark`` (measured layers).
        """
        m = self.module
        args = self.config.model_config
        mp = self.config.model_parallel_config
        tp_size = max(1, mp.tensor_model_parallel_size)
        cp_size = max(1, mp.context_model_parallel_size)
        tcfg = getattr(m, "config", None)

        def cfg(attr, default, *aliases):
            # Prefer the real module, then its TransformerConfig, then args.
            for src in (m, tcfg, args):
                for name in (attr, *aliases):
                    if src is not None and getattr(src, name, None) is not None:
                        return getattr(src, name)
            return default

        cr = self._v4_resolved_cr()
        hc = max(1, int(cfg("hc_mult", 1)))
        hidden = int(cfg("hidden_size", 0))
        heads = int(cfg("num_heads", 0, "num_attention_heads"))
        hd = int(cfg("head_dim", 0, "kv_channels"))
        q_lora = int(cfg("q_lora_rank", 0) or 0)
        o_lora = int(cfg("o_lora_rank", 0) or 0)
        o_groups = max(1, int(cfg("o_groups", 1) or 1))
        swa = int(cfg("attn_sliding_window", 0) or 0)
        index_topk = int(cfg("index_topk", 0) or 0)
        ihd = int(cfg("index_head_dim", 128) or 128)
        inh = int(cfg("index_n_heads", heads) or heads)
        bpe = 2  # bf16 activations

        # hc-expanded token count per rank (the v4_flops ``s_eff`` convention:
        # V4 runs attention once per mHC stream).
        T = (batch_size * seq_len // tp_size // cp_size) * hc
        n_d = heads * hd
        dt = "fp8" if getattr(self.config.model_config, "fp8", None) else "bf16"
        hbm = (getattr(self._gemm_backend, "hbm_bandwidth_gbps", None) or 5300.0) * 1e9

        def g(mm, nn, kk):
            return self._gemm_backend.simulate_gemm(mm, nn, kk, dt).forward_time_ms

        # ---- attn.proj : V4 LoRA Q/KV/O GEMMs (cr-independent) ----
        if self._gemm_backend is not None:
            if q_lora > 0:
                proj_fwd = g(T, q_lora, hidden) + g(T, n_d, q_lora) + g(T, hd, hidden)
            else:
                proj_fwd = g(T, n_d, hidden) + g(T, hd, hidden)
            if o_lora > 0:
                proj_fwd += g(T, o_lora, n_d) + g(T, hidden, o_groups * o_lora)
            else:
                proj_fwd += g(T, hidden, n_d)
        else:
            proj_fwd = 0.0
        proj_bwd = 2.0 * proj_fwd

        # ---- attn.core : FAv3 SDPA with cr-aware visible KV ----
        core_fwd = core_bwd = 0.0
        if self._sdpa_backend is not None:
            if cr == 0:
                s_k = swa or seq_len
            elif cr == 128:
                s_k = (swa or 0) + max(1, seq_len // 128)
            elif cr == 4:
                s_k = (swa or 0) + (index_topk or seq_len)
            else:
                s_k = seq_len
            sd = self._sdpa_backend.simulate_sdpa(
                batch_size=hc, num_heads=heads, seq_len=seq_len, head_dim=hd,
                causal=True, dtype="bf16", seq_len_kv=s_k,
            )
            core_fwd, core_bwd = sd.forward_time_ms, sd.backward_time_ms

        # ---- compressor (cr>0) ----
        comp_fwd = comp_bwd = 0.0
        if cr > 0 and self._gemm_backend is not None:
            coff = 2 if cr == 4 else 1
            comp_fwd = g(T, coff * hd, hidden)
            comp_bwd = 2.0 * comp_fwd

        # ---- indexer (cr==4 only): projections + pool scoring ----
        idx_fwd = idx_bwd = idx_score = 0.0
        if cr == 4 and self._gemm_backend is not None:
            pool = max(1, T // cr)
            idx_fwd = g(T, ihd, hidden) + g(T, inh * ihd, ihd) + g(T, inh, hidden) + g(T, 2 * ihd, hidden)
            idx_bwd = 2.0 * idx_fwd
            idx_score = (2.0 * T * inh * pool * ihd) / 4768.0e12 * 1e3  # fp8 peak

        # ---- CSA sparse-gather memory traffic (cr==4) ----
        # gathered = [hc, seq, index_topk, head_dim] (single-latent, shared
        # across heads); memory-bound copy ~= 2x bytes / HBM bandwidth.
        gather_fwd = gather_bwd = 0.0
        if cr == 4:
            gathered_bytes = hc * seq_len * (index_topk or 0) * hd * bpe
            gather_fwd = 2.0 * gathered_bytes / hbm * 1e3
            gather_bwd = gather_fwd

        # ---- attn.norm : RoPE + q/k norms (memory-bound) ----
        norm_bytes = T * n_d * bpe
        norm_fwd = 3.0 * norm_bytes / hbm * 1e3
        norm_bwd = 4.0 * norm_bytes / hbm * 1e3

        fwd_time = proj_fwd + core_fwd + comp_fwd + idx_fwd + idx_score + gather_fwd + norm_fwd
        bwd_time = proj_bwd + core_bwd + comp_bwd + idx_bwd + gather_bwd + norm_bwd
        activation_memory = self.estimated_activation_memory(batch_size, seq_len)
        return (fwd_time, bwd_time, activation_memory)

    def _get_simulated_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        """Get simulated results from GEMM + SDPA simulation backends."""
        if self._is_v4_attention():
            return self._get_v4_simulated_results(batch_size, seq_len)

        args = self.config.model_config
        mp = self.config.model_parallel_config
        tp_size = max(1, mp.tensor_model_parallel_size)
        cp_size = max(1, mp.context_model_parallel_size)

        batch_tokens = batch_size * seq_len // tp_size // cp_size
        slen_per_cp = seq_len // cp_size

        fwd_time = 0.0
        bwd_time = 0.0

        # 1. Simulate linear projection GEMMs using GEMM backend
        if self._gemm_backend is not None:
            gemm_dtype = "fp8" if getattr(args, "fp8", None) else "bf16"

            if getattr(args, "multi_latent_attention", False):
                # MLA: LoRA-factored Q and compressed KV projections
                # 6 forward GEMMs + 12 backward GEMMs
                mla_fwd, mla_bwd = self._simulate_mla_gemms(batch_tokens, gemm_dtype)
                fwd_time += mla_fwd
                bwd_time += mla_bwd
            else:
                # Standard attention: Q, K, V, O projections
                # 4 forward GEMMs + 8 backward GEMMs
                num_query_groups = (
                    args.num_query_groups
                    if args.group_query_attention and args.num_query_groups
                    else args.num_attention_heads
                )
                gemm_result = self._gemm_backend.simulate_attention_gemms(
                    batch_tokens=batch_tokens,
                    hidden_size=args.hidden_size,
                    num_attention_heads=args.num_attention_heads,
                    kv_channels=args.kv_channels,
                    num_query_groups=num_query_groups,
                    dtype=gemm_dtype,
                )
                fwd_time += gemm_result.forward_time_ms
                bwd_time += gemm_result.backward_time_ms

        # 2. Simulate SDPA core computation using SDPA backend
        if self._sdpa_backend is not None:
            heads_per_rank = max(1, args.num_attention_heads // tp_size)

            if getattr(args, "multi_latent_attention", False):
                # MLA: Q·Kᵀ uses qk_head_dim + qk_pos_emb_head_dim (e.g. 192),
                #       P·V  uses v_head_dim (e.g. 128).
                sdpa_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
                sdpa_head_dim_v = args.v_head_dim
            else:
                sdpa_head_dim = args.kv_channels
                sdpa_head_dim_v = None  # same as head_dim

            sdpa_result = self._sdpa_backend.simulate_sdpa(
                batch_size=batch_size,
                num_heads=heads_per_rank,
                seq_len=slen_per_cp,
                head_dim=sdpa_head_dim,
                causal=True,
                dtype="bf16",
                head_dim_v=sdpa_head_dim_v,
            )
            fwd_time += sdpa_result.forward_time_ms
            bwd_time += sdpa_result.backward_time_ms

        activation_memory = self.estimated_activation_memory(batch_size, seq_len)
        return (fwd_time, bwd_time, activation_memory)

    def _get_benchmark_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        """Get or compute benchmark results (cached)."""
        cache_key = (batch_size, seq_len)

        if self._cached_results is None or self._cache_key != cache_key:
            if self._gemm_backend is not None or self._sdpa_backend is not None:
                # Use simulation mode
                self._cached_results = self._get_simulated_results(batch_size, seq_len)
            else:
                # Use actual GPU benchmarking
                # Context parallel / Sequence parallel adjustment
                cp_size = self.config.model_parallel_config.context_model_parallel_size
                # Effective sequence length per rank if CP is used
                slen_per_cp = seq_len // cp_size

                hidden = self.config.model_config.hidden_size
                tcfg = getattr(self.module, "config", None)
                hc_mult = getattr(tcfg, "hc_mult", 1)
                # DeepSeek-V4 attention has a different signature
                # (forward(hidden[B,S,D], position_ids[B,S])); feed V4-aware
                # inputs so the real V4 attention path is exercised instead of
                # crashing / falling back. Non-V4 modules use the stock inputs.
                v4 = v4_module_inputs(self.module, batch_size, seq_len, hidden, hc_mult, "attention")
                if v4 is not None:
                    ishapes, fkwargs = v4
                    self._cached_results = benchmark_layer(
                        self.module, ishapes, transformer_config=tcfg, forward_kwargs=fkwargs
                    )
                else:
                    self._cached_results = benchmark_layer(
                        self.module,
                        [
                            (seq_len, batch_size, hidden),
                            ((1, 1, slen_per_cp, seq_len), torch.bool),
                        ],
                    )
            self._cache_key = cache_key
        return self._cached_results

    def measured_forward_time(self, batch_size: int, seq_len: int) -> float:
        forward_time, _, _ = self._get_benchmark_results(batch_size, seq_len)
        return forward_time

    def measured_backward_time(self, batch_size: int, seq_len: int) -> float:
        _, backward_time, _ = self._get_benchmark_results(batch_size, seq_len)
        return backward_time

    def measured_activation_memory(self, batch_size: int, seq_len: int) -> int:
        _, _, activation_memory = self._get_benchmark_results(batch_size, seq_len)
        return activation_memory
