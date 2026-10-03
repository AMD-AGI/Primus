###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from typing import Optional

import torch

from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.training_config import gemm_dtype_from_config

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

        # --- DSv4-Pro-aware attention activation (v4 correction) --------------
        # V4 attention is neither stock MHA nor stock MLA: (1) a single SHARED KV
        # latent (kv_channels-wide, broadcast to all heads at compute time -> only
        # the latent is stored, NOT heads*kv_channels); (2) it always runs a
        # flash/FAv3 kernel (use_v4_triton_attention) that never materialises the
        # [B,H,S,S] score matrix; (3) it carries hc_mult parallel residual streams
        # ([B,S,K,D]) between layers.  The generic "standard" branch below
        # over-counts KV by ~num_heads and (when use_flash_attn is unset) adds a
        # phantom heads*seqlen softmax term; both are corrected here and the mHC
        # K-stream residual is added back.  Validated vs the real V4 backend
        # (deepseek_v4_attention single-latent KV + v4 flash kernels; SubA).
        _is_v4_mem = (not args.multi_latent_attention) and (
            getattr(args, "compress_ratios", None) is not None
            or bool(getattr(args, "use_v4_triton_attention", False))
        )
        if _is_v4_mem:
            hc = max(1, int(getattr(args, "hc_mult", 1) or 1))
            q_proj = args.kv_channels * args.num_attention_heads  # per-head Q (stored post up-proj)
            kv_latent = args.kv_channels  # single shared KV latent (K=V)
            flash_softmax = q_proj  # FAv3 logsumexp stats; no [B,H,S,S]
            mhc_residual = max(0, hc - 1) * args.hidden_size  # extra K-1 residual streams
            activation_width = q_proj + 2 * kv_latent + args.hidden_size + flash_softmax + mhc_residual
            return tokens_per_rank * activation_width * bytes_per_value

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

        Compute-bound terms (LoRA Q/KV/O GEMMs, compressor, indexer) come from
        the GEMM backend; the SDPA core from the FAv3 simulator; memory-bound
        terms (RoPE + q/k-norm + mHC elementwise, CSA gather) from explicit
        activation byte traffic and HBM bandwidth.

        The memory-bound elementwise block (norm + misc) is derived from first
        principles by default (:meth:`_v4_memory_bound_first_principles`): a
        byte-for-byte enumeration of every real elementwise kernel at one
        sustained-HBM efficiency, with the selective-recompute forward replay
        handled by the transformer-layer profiler (``bwd += attn_fwd``).  The
        legacy trace-anchored fudge (``ELEM_EFF=0.15`` / 8 / 30 passes) is
        available for A/B only via ``PRIMUS_ATTN_CALIBRATED=1``.
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

        # Base (un-expanded) token count per rank.  mHC collapses the K=hc
        # parallel residual streams into a single [B, S, D] stream BEFORE the
        # attention sub-block runs (DeepseekV4HybridLayer._hc_apply ->
        # HyperMixer.collapse), so every attention projection GEMM, the SDPA
        # core, the compressor/indexer, and the in-attention elementwise ops
        # execute at the base token count -- NOT S*hc.  The hc factor is a
        # memory-domain cost (a hc-wide saved residual + collapse/expand
        # bandwidth), captured separately in the memory-bound block below; it
        # is NOT a compute multiplier on the attention block.
        T = batch_size * seq_len // tp_size // cp_size
        n_d = heads * hd
        dt = gemm_dtype_from_config(self.config.model_config)
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
            # Split visible KV into the SWA local band (constant ~swa keys per
            # query) and the sparse pool/top-K (triangular over the causal past).
            #   cr 0 = dense+SWA, 128 = HCA (SWA + seq/128 pool), 4 = CSA (SWA + top-K).
            if cr == 0:
                swa_k, sparse_k = (swa or seq_len), 0
            elif cr == 128:
                swa_k, sparse_k = (swa or 0), max(1, seq_len // 128)
            elif cr == 4:
                # A query can never select more than the compressed pool holds,
                # so clamp top-K to ceil(seq/cr) the same way the HCA branch does.
                csa_pool = max(1, -(-seq_len // cr))
                swa_k, sparse_k = (swa or 0), min(index_topk or csa_pool, csa_pool)
            else:
                swa_k, sparse_k = 0, seq_len
            # #2 robustness clamp: a query can never see more than all keys.
            swa_k = min(swa_k, seq_len)
            sparse_k = min(sparse_k, seq_len)
            # The SDPA sim applies a flat causal_factor=0.5 to the whole visible
            # KV, correct only for the triangular sparse/dense part.  #9
            # (PRIMUS_ATTN_SWA_FULLBAND=1): the SWA band is a constant-width
            # per-query band (NOT triangular), so price it at full cost by
            # pre-inflating the visible KV by +swa_k -> after the 0.5 factor the
            # realised work is swa_k*1 + sparse_k*0.5.  Default off (0) keeps the
            # audited baseline; on for the more-faithful A/B.
            if os.getenv("PRIMUS_ATTN_SWA_FULLBAND", "0") == "1":
                s_k = min(2 * swa_k + sparse_k, 2 * seq_len)
            else:
                s_k = min(swa_k + sparse_k, seq_len)
            # The attention core runs once on the collapsed [B, S, D] stream,
            # so the batch axis is the real microbatch -- not hc (the streams
            # are already collapsed away before attention).
            sd = self._sdpa_backend.simulate_sdpa(
                batch_size=batch_size,
                num_heads=heads,
                seq_len=seq_len,
                head_dim=hd,
                causal=True,
                dtype="bf16",
                seq_len_kv=s_k,
            )
            core_fwd, core_bwd = sd.forward_time_ms, sd.backward_time_ms

        # ---- compressor (cr>0) ----
        comp_fwd = comp_bwd = 0.0
        if cr > 0 and self._gemm_backend is not None:
            coff = 2 if cr == 4 else 1
            # #7: the Compressor has TWO projections (compressor.py) -- wkv
            # (hidden->coff*hd) AND wgate (hidden->coff*hd) for the softmax-
            # weighted window pooling.  Only wkv was counted -> compressor ~2x
            # under.  Charge both same-shape GEMMs.
            comp_fwd = 2.0 * g(T, coff * hd, hidden)
            comp_bwd = 2.0 * comp_fwd

        # ---- indexer (cr==4 only): projections + pool scoring ----
        # #3: DSA TRAINS the indexer via a KL-distillation aux loss (megatron dev
        # dsa.py/csa.py + DeepSeek-V3.2 DSA paper) -> it HAS a backward pass.
        # Primus's shipped model freezes it for engineering reasons
        # (deepseek_v4_attention.py requires_grad_(False) unless
        # PRIMUS_V4_INDEXER_TRAINABLE=1), but a FAITHFUL projection of real
        # DSv4-Pro training must count the indexer training cost, so the default
        # here is now TRAINABLE.  Set PRIMUS_V4_INDEXER_TRAINABLE=0 to model the
        # frozen-indexer engineering variant instead.
        idx_trainable = os.getenv("PRIMUS_V4_INDEXER_TRAINABLE", "1") == "1"
        idx_fwd = idx_bwd = idx_score = 0.0
        idx_aux_fwd = idx_aux_bwd = 0.0
        if cr == 4 and self._gemm_backend is not None:
            pool = max(1, T // cr)
            idx_fwd = g(T, ihd, hidden) + g(T, inh * ihd, ihd) + g(T, inh, hidden) + g(T, 2 * ihd, hidden)
            idx_bwd = 2.0 * idx_fwd if idx_trainable else 0.0
            # Indexer QK scoring: fp8 when the fp8 indexer is enabled, else bf16.
            # Size it to the TARGET arch's S&F peak for that dtype (arch-specific)
            # via the backend's peak_flops(); fall back to the MI355X anchors so
            # behaviour is unchanged for backends/arches without an S&F table.
            score_dt = "fp8" if bool(cfg("use_v4_fp8_indexer", False)) else "bf16"
            peak = None
            _pf = getattr(self._gemm_backend, "peak_flops", None)
            if callable(_pf):
                peak = _pf(score_dt)
            if not peak:
                peak = 4768.0e12 if score_dt == "fp8" else 2384.0e12  # MI355X anchors
            idx_score = (2.0 * T * inh * pool * ihd) / peak * 1e3
            # #3 KL-distillation aux loss (only when the indexer is trained): a
            # softmax over the [T, pool] indexer score distribution + the target
            # softmax + the KL reduction, memory-bound (~3 passes fwd, ~2x bwd,
            # bf16).  The indexer proj/score GRADIENTS are idx_bwd / idx_score_bwd
            # above; this is the extra loss-head traffic on top.
            if idx_trainable:
                idx_aux_fwd = 3.0 * T * pool * bpe / hbm * 1e3
                idx_aux_bwd = 2.0 * idx_aux_fwd

        # ---- CSA sparse gather/scatter memory traffic (cr==4) ----
        # Sized on the number of (query, selected-slot) VISITS, not on
        # ``index_topk`` alone:
        #   * a query can never select more slots than the compressed pool holds
        #     (``pool_len = ceil(seq/cr)``), and
        #   * causality caps query ``t`` at ``t/cr`` slots, so the early prefix
        #     selects fewer than ``k_cap`` -> ``avail`` is the mean fill ratio.
        #     For the DSv4-Pro shape (S=4096, cr=4, topk=1024) the pool IS the
        #     whole visible history, so avail=0.5 -- the top-K is degenerate and
        #     only causality limits it.
        # The pool itself is tiny (``pool_len * hd * bpe`` ~ 1 MiB/sample) and is
        # re-read by every query, so the read side is served by L2 / Infinity
        # Cache, NOT HBM: only the bytes that are actually MATERIALIZED count.
        gather_fwd = gather_bwd = 0.0
        if cr == 4:
            pool_len = max(1, -(-seq_len // cr))
            k_cap = min(index_topk or pool_len, pool_len)
            if seq_len >= cr * k_cap:
                avail = 1.0 - (cr * k_cap) / (2.0 * seq_len)
            else:
                avail = seq_len / (2.0 * cr * k_cap)
            # CP shards the query axis; TP does NOT shard the single shared KV
            # latent (pool + top-K indices are head-agnostic -> replicated).
            gather_tokens = batch_size * max(1, seq_len // cp_size)
            # Buffers are allocated over ALL k slots (masked ones are written as
            # zeros); only the valid slots are read back downstream.
            visit_bytes = gather_tokens * k_cap * hd * bpe
            used_bytes = visit_bytes * avail
            pool_bytes = batch_size * pool_len * hd * bpe
            # Only ``use_v4_triton_csa_attention`` routes to
            # v4_csa_attention_from_pool (the in-kernel gather).  The tilelang /
            # flydsl flags are in-kernel sub-options of the *already-gathered*
            # wrapper and do NOT enable the from-pool path on their own
            # (DeepseekV4Attention._csa_forward gates on the Triton flag only),
            # so a tilelang-only config still runs the eager materialised gather.
            csa_fused = bool(cfg("use_v4_triton_csa_attention", False)) or os.getenv(
                "PRIMUS_USE_V4_TRITON_CSA_ATTENTION", ""
            ).lower() in ("1", "true", "yes")
            if csa_fused:
                # Forward materialises nothing: the pool tile is gathered into
                # LDS/registers and shared across all heads (BLOCK_H spans the
                # head axis), so the only HBM cost is one cold read of the pool.
                gather_fwd = pool_bytes / hbm * 1e3
                # Backward cannot avoid the dpool round trip: the sparse BWD
                # writes a per-visit ``dpool_partial`` buffer (input dtype, every
                # k slot) and a segmented-reduction kernel reads the valid slots
                # back into dpool[B, pool_len, hd].  Separate kernels, so this
                # does NOT overlap the SDPA core.  ``eta`` is the sustained HBM
                # fraction of the strided write + permuted read.
                eta = float(os.getenv("PRIMUS_ATTN_CSA_SCATTER_ETA", "0.6"))
                gather_bwd = (visit_bytes + used_bytes) / (eta * hbm) * 1e3
            else:
                # Eager path materialises [B, S, K, hd]: the gather writes it,
                # the ``* valid`` mask is a second read+write pass, and the
                # attention kernel reads it back.
                gather_fwd = 4.0 * visit_bytes / hbm * 1e3
                # Its VJP is the expensive half: the mask backward is another
                # read+write, and torch.gather's VJP allocates zeros over the
                # EXPANDED pool ([B, S, pool_len, hd]), scatter-adds the valid
                # slots into it and reduces it back over the query axis -- so the
                # dominant term scales with pool_len (~S/cr), not with k_cap.
                expanded_bytes = gather_tokens * pool_len * hd * bpe
                gather_bwd = (3.0 * visit_bytes + 2.0 * used_bytes + 2.0 * expanded_bytes) / hbm * 1e3

        # ---- attn.norm + attn.misc : memory-bound elementwise block ----
        # DEFAULT is the first-principles byte-traffic model
        # (:meth:`_v4_memory_bound_first_principles`): it enumerates every real
        # elementwise kernel (q/kv RMS, per-head q-RMS, partial RoPE,
        # transpose->contiguous, mHC expand/collapse, HCA cat) with explicit
        # read+write HBM traffic at a single sustained-BW efficiency, replacing
        # both ``norm`` and ``misc`` -- no per-term fitted pass counts.
        #
        # The legacy trace-anchored fudge (``norm`` = 3/4 passes over T*n_d;
        # ``misc`` = 8/30 passes over T*hidden at ``ELEM_EFF=0.15``) is retained
        # ONLY as an opt-out for A/B comparison via ``PRIMUS_ATTN_CALIBRATED=1``.
        mhc_fwd = mhc_bwd = 0.0
        if os.getenv("PRIMUS_ATTN_CALIBRATED", "0") != "1":
            # #4: split the memory-bound block into three transparent buckets --
            # norm (q/kv/per-head RMS), mHC (expand/collapse + compute_weights +
            # Sinkhorn), and misc (RoPE/transpose/HCA-cat) -- instead of hiding
            # norm (=0) and mHC inside a single over-conservative misc term.
            (norm_fwd, norm_bwd), (mhc_fwd, mhc_bwd), (misc_fwd, misc_bwd) = (
                self._v4_memory_bound_first_principles(
                    T=T, n_d=n_d, hidden=hidden, hc=hc, cr=cr, q_lora=q_lora, hd=hd, bpe=bpe, hbm=hbm
                )
            )
        else:
            # ---- legacy calibrated fallback (PRIMUS_ATTN_CALIBRATED=1) ----
            # attn.norm : RoPE + q/k norms (memory-bound).
            norm_bytes = T * n_d * bpe
            norm_fwd = 3.0 * norm_bytes / hbm * 1e3
            norm_bwd = 4.0 * norm_bytes / hbm * 1e3
            # attn.misc : generic-elementwise glue over the hc-expanded HIDDEN
            # stream at a fitted 15% sustained-HBM efficiency.  Superseded by the
            # first-principles model above; kept for A/B only.  All env-tunable.
            ELEM_EFF = float(os.getenv("PRIMUS_ATTN_ELEM_EFF", "0.15"))
            MISC_FWD_PASSES = float(os.getenv("PRIMUS_ATTN_MISC_FWD_PASSES", "8"))
            MISC_BWD_PASSES = float(os.getenv("PRIMUS_ATTN_MISC_BWD_PASSES", "30"))
            hidden_bytes = T * hidden * bpe
            misc_fwd = MISC_FWD_PASSES * hidden_bytes / (ELEM_EFF * hbm) * 1e3
            misc_bwd = MISC_BWD_PASSES * hidden_bytes / (ELEM_EFF * hbm) * 1e3

        # Indexer score backward (cr==4): only when the indexer is trained
        # (frozen by default -> no gradient path; see idx_trainable above).
        idx_score_bwd = 2.0 * idx_score if idx_trainable else 0.0

        fwd_time = (
            proj_fwd
            + core_fwd
            + comp_fwd
            + idx_fwd
            + idx_score
            + idx_aux_fwd
            + gather_fwd
            + norm_fwd
            + mhc_fwd
            + misc_fwd
        )
        bwd_time = (
            proj_bwd
            + core_bwd
            + comp_bwd
            + idx_bwd
            + idx_score_bwd
            + idx_aux_bwd
            + gather_bwd
            + norm_bwd
            + mhc_bwd
            + misc_bwd
        )
        if os.getenv("PRIMUS_PRINT_LAYER_BREAKDOWN") and int(os.getenv("RANK", "0")) == 0:
            print(
                "[BRKDN] attn cr=%d proj_f=%.4f proj_b=%.4f core_f=%.4f core_b=%.4f comp_f=%.4f comp_b=%.4f "
                "idx_f=%.4f idx_b=%.4f gather_f=%.4f gather_b=%.4f norm_f=%.4f norm_b=%.4f "
                "mhc_f=%.4f mhc_b=%.4f misc_f=%.4f misc_b=%.4f"
                % (
                    cr,
                    proj_fwd,
                    proj_bwd,
                    core_fwd,
                    core_bwd,
                    comp_fwd,
                    comp_bwd,
                    idx_fwd + idx_score + idx_aux_fwd,
                    idx_bwd + idx_score_bwd + idx_aux_bwd,
                    gather_fwd,
                    gather_bwd,
                    norm_fwd,
                    norm_bwd,
                    mhc_fwd,
                    mhc_bwd,
                    misc_fwd,
                    misc_bwd,
                )
            )
        activation_memory = self.estimated_activation_memory(batch_size, seq_len)
        return (fwd_time, bwd_time, activation_memory)

    def _v4_memory_bound_first_principles(
        self,
        *,
        T: int,
        n_d: int,
        hidden: int,
        hc: int,
        cr: int,
        q_lora: int,
        hd: int,
        bpe: int,
        hbm: float,
    ) -> tuple[float, float, list]:
        """First-principles byte-traffic model of the attention memory-bound block.

        Replaces the calibrated ``norm`` + ``misc`` terms (the ``ELEM_EFF=0.15``
        / 8-fwd / 30-bwd fudge) with an explicit enumeration of every real
        elementwise kernel in :class:`DeepseekV4Attention.forward`, each costed
        as ``bytes_read + bytes_written`` at its bf16 HBM footprint divided by a
        single sustained-HBM efficiency.

        Grounding (verified against the module + Triton kernels):

        * Elementwise kernels are FUSED (per-head RMS keeps fp32 in registers;
          the P32 fix keeps RoPE bf16 end-to-end), so each logical op moves its
          operands through HBM once -- i.e. ``read + write`` at the bf16 size,
          NOT a double-width fp32 materialization.
        * A fused reduction (RMS mean-of-squares) reads its input once and
          writes a negligible ``[..., 1]`` stat, so its RMS is ``read + write``
          of the normalized tensor = 2 bf16 passes.
        * Partial RoPE rewrites the whole ``head_dim`` (nope + rotated) into a
          fresh contiguous tensor -> read + write of the full q / k tensor.
        * This backward is the TRUE gradient computation only (each op's VJP:
          input + upstream grad -> grad-in).  It does NOT include the selective
          -recompute forward replay: the transformer-layer profiler already adds
          the whole attention forward (``bwd_time += attn_fwd``) when
          ``recompute_granularity == "selective"``, and that ``attn_fwd``
          already contains this block's ``fwd_ms``.  Folding a replay in here
          too would double-count it.

        ``eta`` (``PRIMUS_ATTN_FP_ETA``, default ``_ACTIVATION_BW_FRACTION``
        = 0.566) is the sustained fraction of peak HBM these medium streaming /
        strided kernels achieve -- the SAME constant the MoE SwiGLU activation
        and the transformer-layer residual use, so no attention-specific
        efficiency is introduced.

        Returns three ``(fwd_ms, bwd_ms)`` pairs -- ``(norm, mhc, misc)`` -- so
        the report can attribute the norm / mHC / misc buckets separately (norm
        was previously hidden as 0 and mHC buried inside misc).
        """
        # #4: near-peak efficiency for the highly-fusible norm/RoPE/transpose/mHC
        # elementwise stream.  These small-buffer, cache-friendly kernels run
        # close to peak HBM, so the SwiGLU-derived 0.566 (_ACTIVATION_BW_FRACTION)
        # systematically over-priced them (~1.7x).  Default ~0.9 (near-peak);
        # PRIMUS_ATTN_FP_ETA is still honored so the old 0.566 is one env away.
        misc_eta = float(os.getenv("PRIMUS_ATTN_MISC_ETA", os.getenv("PRIMUS_ATTN_FP_ETA", "0.9")))
        # T is the base (un-expanded) token count B*S: the in-attention
        # elementwise kernels (q/kv RMS, RoPE, transpose) run on the collapsed
        # base-S stream.  mHC operates on the [M, hc, hidden] residual, so the
        # un-expanded token count M == T here (the hc width enters only via the
        # per-op stream-count coefficients on the mHC expand/collapse/HCA rows).
        M = max(1, T)

        # HCA fraction: cr==128 layers pay the extra cat([local, pool]) that
        # materializes the broadcast K and V across all heads.  When this helper
        # is called per-layer we know the exact cr; weight = 1.0 for cr==128.
        hca = 1.0 if cr == 128 else 0.0

        tnd = T * n_d * bpe  # per-pass bf16 bytes of the n_d-wide q / attn-out tensor
        tqr = T * q_lora * bpe  # q-LoRA-wide tensor (q_layernorm)
        thd = T * hd * bpe  # single-latent KV-wide tensor (kv_layernorm, RoPE-k)

        # Each entry: (name, fwd_bytes, bwd_bytes).  read+write => factor 2;
        # a reduction's stat write is negligible so RMS is still ~read+write.
        ops = [
            # q-LoRA RMSNorm (q_layernorm on width q_lora).
            ("q_layernorm RMS", 2 * tqr, 3 * tqr),
            # Per-head parameter-less q-RMS on the full n_d-wide tensor.
            ("q per-head RMS (n_d)", 2 * tnd, 3 * tnd),
            # Partial interleaved RoPE on q: full-tensor contiguous rewrite.
            ("RoPE q (n_d)", 2 * tnd, 2 * tnd),
            # Single-latent KV RMSNorm + RoPE (width hd, shared across heads).
            ("kv_layernorm RMS", 2 * thd, 3 * thd),
            ("RoPE k (hd)", 2 * thd, 2 * thd),
            # Attention output transpose(1,2)->contiguous copy (n_d-wide).
            ("out transpose->contiguous (n_d)", 2 * tnd, 2 * tnd),
            # mHC HyperConnection expand: new[M,K,D]=post*out+comb@x
            #   fwd read x[M,K,D]+out[M,D], write new[M,K,D] => (2K+1)*M*D
            #   bwd read g,x[M,K,D]+out[M,D], write dx[M,K,D]+dout[M,D] => (3K+2)*M*D
            ("mHC expand", (2 * hc + 1) * M * hidden * bpe, (3 * hc + 2) * M * hidden * bpe),
            # mHC collapse (reduce K streams -> 1): read [M,K,D] write [M,D].
            ("mHC collapse", (hc + 1) * M * hidden * bpe, (hc + 1) * M * hidden * bpe),
            # HCA-only cat([local, pool]) for K and V: materializes the
            # broadcast (H-replicated) local KV + pool.  ~2x (K and V), read+write,
            # per un-expanded token over n_d; only on cr==128 layers.
            ("HCA cat K/V", hca * 2.0 * 2 * M * n_d * bpe, hca * 2.0 * M * n_d * bpe),
        ]

        # ---- group the ops into norm / mHC / misc, each priced at misc_eta ----
        norm_names = {"q_layernorm RMS", "q per-head RMS (n_d)", "kv_layernorm RMS"}
        mhc_names = {"mHC expand", "mHC collapse"}

        def _grp_ms(names):
            fb = sum(o[1] for o in ops if o[0] in names)
            bb = sum(o[2] for o in ops if o[0] in names)
            return fb / (misc_eta * hbm) * 1e3, bb / (misc_eta * hbm) * 1e3

        norm_ms = _grp_ms(norm_names)
        misc_ms = _grp_ms({o[0] for o in ops} - norm_names - mhc_names)
        mhc_bw_f, mhc_bw_b = _grp_ms(mhc_names)

        # mHC compute_weights (was entirely missing -> mHC under-counted, esp.
        # backward): the mapping_proj GEMM reads the K*D-wide residual, then a
        # per-stream RMS + a Sinkhorn(n_iters) normalization.  Sinkhorn is
        # launch-latency-bound (tiny [M,K,K] fp32 reduces), and its backward
        # re-runs the forward (SinkhornKnopp.backward).
        cw_fwd = cw_bwd = 0.0
        if hc > 1 and self._gemm_backend is not None:
            dt = gemm_dtype_from_config(self.config.model_config)
            cw = self._gemm_backend.simulate_gemm(M, hc * hc + 2 * hc, hc * hidden, dt).forward_time_ms
            cw_fwd += cw
            cw_bwd += 2.0 * cw
            cw_mem = 2.0 * M * hc * hidden * bpe  # RMS over the K*D-wide input
            cw_fwd += cw_mem / (misc_eta * hbm) * 1e3
            cw_bwd += 1.5 * cw_mem / (misc_eta * hbm) * 1e3
            n_iters = int(os.getenv("PRIMUS_HC_SINKHORN_ITERS", "20"))
            sink_ms = (1.0 + 2.0 * max(0, n_iters - 1)) * 0.75 / 1e3  # ~0.75us/launch
            cw_fwd += sink_ms
            cw_bwd += sink_ms

        mhc_ms = (mhc_bw_f + cw_fwd, mhc_bw_b + cw_bwd)
        return norm_ms, mhc_ms, misc_ms

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
            gemm_dtype = gemm_dtype_from_config(args)

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
