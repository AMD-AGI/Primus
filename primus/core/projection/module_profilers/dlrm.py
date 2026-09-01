###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DLRM-v4 (TorchRec / HSTU ranker) top-level profiler.

Assembles the workload profiler tree for a generative recommender:

    sparse embeddings  ->  (dense bottom MLP)  ->  N x HSTU layers  ->  over MLP

and provides both the **memory-projection** contract (``estimated_num_params``,
``estimated_activation_memory``, ``get_num_bytes_per_param``) and a first-cut
**throughput** estimate (:meth:`project_step`) priced with the shared GEMM/SDPA
simulation backends plus the embedding all-to-all collective.

Registered under the ``torchrec_dlrm`` framework via the workload registry.
"""

import os
from typing import List, Optional, Tuple

from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.module_profilers.hstu import HSTULayerProfiler
from primus.core.projection.module_profilers.sparse_embedding import (
    SparseEmbeddingProfiler,
)
from primus.core.projection.profiler_spec import ModuleProfilerSpec
from primus.core.projection.simulation_backends.base import (
    resolve_hbm_bytes_per_ms,
    resolve_peak_tflops,
)
from primus.core.projection.training_config import gemm_dtype_from_config

# Extra fp32 optimizer-state words carried *per embedding row* (not per element).
_EMB_OPTIMIZER_ROW_WORDS = {
    "rowwise_adagrad": 1,
    "row_wise_adagrad": 1,
    "exact_row_wise_adagrad": 1,
    "rowwise": 1,
}
# Extra fp32 optimizer-state words carried *per embedding element*.
_EMB_OPTIMIZER_ELEM_WORDS = {
    "adagrad": 1,
    "sgd": 0,
    "adam": 2,
    "adamw": 2,
}


def _as_list(val) -> List[int]:
    if val is None:
        return []
    if isinstance(val, str):
        try:
            val = eval(val)
        except Exception:
            return []
    if isinstance(val, (int, float)):
        return [int(val)]
    return [int(x) for x in val]


def _mlp_layers(input_dim: int, widths: List[int]) -> List[Tuple[int, int]]:
    """Return [(in, out), ...] GEMM shapes for an MLP stack."""
    layers = []
    prev = input_dim
    for w in widths:
        if prev > 0 and w > 0:
            layers.append((prev, w))
        prev = w
    return layers


class DLRMProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        self._gemm_backend = None
        self._sdpa_backend = None

    # -- backend wiring --------------------------------------------------------
    def set_simulation_backends(self, gemm_backend=None, sdpa_backend=None):
        self._gemm_backend = gemm_backend
        self._sdpa_backend = sdpa_backend
        hstu = self.sub_profilers.get("hstu_layer") if self.sub_profilers else None
        if hstu is not None and hasattr(hstu, "set_simulation_backends"):
            hstu.set_simulation_backends(gemm_backend, sdpa_backend)
        emb = self.sub_profilers.get("sparse_embedding") if self.sub_profilers else None
        if emb is not None:
            if hasattr(emb, "set_gemm_backend") and gemm_backend is not None:
                emb.set_gemm_backend(gemm_backend)
            if hasattr(emb, "set_simulation_mode"):
                emb.set_simulation_mode(True)

    # -- geometry helpers ------------------------------------------------------
    def _num_layers(self) -> int:
        return max(1, int(self.config.model_config.num_layers or 1))

    def _dense_dim(self) -> int:
        mc = self.config.model_config
        return int(mc.hidden_size or mc.embedding_dim or 0)

    def _bottom_mlp_layers(self) -> List[Tuple[int, int]]:
        mc = self.config.model_config
        return _mlp_layers(int(mc.dense_input_dim or 0), _as_list(mc.dlrm_bottom_mlp))

    def _over_mlp_layers(self) -> List[Tuple[int, int]]:
        mc = self.config.model_config
        # Interaction input ~= (#tables + 1 dense) x D collapsed to D by HSTU output.
        n_tables = int(mc.num_embedding_tables or 0)
        dim = int(mc.embedding_dim or mc.hidden_size or 0)
        inter_dim = (n_tables + 1) * dim if dim else 0
        return _mlp_layers(inter_dim, _as_list(mc.dlrm_over_mlp))

    def _preprocessor_layers(self) -> List[Tuple[int, int]]:
        """Per-token input-preprocessor GEMM shapes (fuses contextual features).

        An explicit ``dlrm_preprocessor_gemms`` list of ``[in, out]`` pairs (one
        per branch/linear) takes precedence and prices the real multi-branch
        ContextualPreprocessor at its true FLOPs; otherwise a collapsed
        ``dlrm_preprocessor_mlp`` chain is used.
        """
        mc = self.config.model_config
        explicit = getattr(mc, "dlrm_preprocessor_gemms", None)
        if explicit:
            if isinstance(explicit, str):
                try:
                    explicit = eval(explicit)
                except Exception:
                    explicit = None
            if explicit:
                return [(int(a), int(b)) for a, b in explicit if int(a) > 0 and int(b) > 0]
        widths = _as_list(mc.dlrm_preprocessor_mlp)
        if not widths:
            return []
        dim = int(mc.embedding_dim or mc.hidden_size or 0)
        in_dim = int(mc.dlrm_preprocessor_input_dim or 0)
        if in_dim <= 0:  # default: (contextual features + 1 content) x D
            n_ctx = max(0, int(mc.num_embedding_tables or 0) - 3)  # 3 sequence tables
            in_dim = (n_ctx + 1) * dim
        return _mlp_layers(in_dim, widths)

    def _head_layers(self) -> List[Tuple[int, int]]:
        """Per-sample multitask prediction-head GEMM shapes (one tower)."""
        mc = self.config.model_config
        widths = _as_list(mc.dlrm_prediction_head_mlp)
        if not widths:
            return []
        return _mlp_layers(int(mc.embedding_dim or mc.hidden_size or 0), widths)

    def _mlp_params(self, layers: List[Tuple[int, int]]) -> int:
        return int(sum(i * o for i, o in layers))

    # -- params ----------------------------------------------------------------
    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        total = 0
        emb = self.sub_profilers["sparse_embedding"]
        total += emb.estimated_num_params(rank)
        hstu = self.sub_profilers["hstu_layer"]
        total += self._num_layers() * hstu.estimated_num_params(rank)
        total += self._mlp_params(self._bottom_mlp_layers())
        total += self._mlp_params(self._over_mlp_layers())
        total += self._mlp_params(self._preprocessor_layers())
        n_tasks = max(1, int(getattr(self.config.model_config, "dlrm_num_tasks", 1) or 1))
        total += n_tasks * self._mlp_params(self._head_layers())
        return int(total)

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        act = 0
        act += self.sub_profilers["sparse_embedding"].estimated_activation_memory(batch_size, seq_len)
        act += self._num_layers() * self.sub_profilers["hstu_layer"].estimated_activation_memory(
            batch_size, seq_len
        )
        return int(act)

    def get_num_bytes_per_param(self) -> float:
        """Bytes-per-parameter for the static (weights+grad+optimizer) block.

        DLRM memory is dominated by the sparse tables, whose per-param cost
        differs from the dense blocks: embedding tables are stored at
        ``embedding_param_bytes`` with a single fp32 optimizer moment
        (row-wise Adagrad is standard), while the small dense HSTU/MLP block
        uses the usual bf16 params+grad + fp32 Adam state.  We return the
        param-count-weighted blend so the memory reporter's
        ``params x bytes_per_param`` stays representative.
        """
        mc = self.config.model_config
        emb_params = self.sub_profilers["sparse_embedding"].estimated_num_params(None)
        dense_params = max(
            0,
            self._num_layers() * self.sub_profilers["hstu_layer"].estimated_num_params(None)
            + self._mlp_params(self._bottom_mlp_layers())
            + self._mlp_params(self._over_mlp_layers()),
        )
        # Embedding optimizer state.  A *row-wise* optimizer (the sparse-table
        # default, e.g. RowWiseAdagrad) keeps one fp32 scalar per row, i.e.
        # 4/dim bytes per parameter -- negligible -- not a full fp32 moment per
        # element.  Charging a full moment doubles the dominant memory term and
        # produces false OOM verdicts.
        opt = str(getattr(mc, "embedding_optimizer", "rowwise_adagrad") or "rowwise_adagrad").lower()
        dim = max(1, int(mc.embedding_dim or mc.hidden_size or 1))
        if opt in _EMB_OPTIMIZER_ROW_WORDS:
            opt_bytes = _EMB_OPTIMIZER_ROW_WORDS[opt] * 4.0 / dim  # per-row fp32 / dim
        else:
            opt_bytes = _EMB_OPTIMIZER_ELEM_WORDS.get(opt, 1) * 4.0  # per-element fp32
        emb_bpp = float(int(mc.embedding_param_bytes or 4)) + opt_bytes
        dense_bpp = 4.0 + 10.0  # bf16 param+grad + fp32 Adam (2+4+4)
        total = emb_params + dense_params
        if total <= 0:
            return dense_bpp
        return (emb_params * emb_bpp + dense_params * dense_bpp) / total

    # -- throughput ------------------------------------------------------------
    def _mlp_step_ms(self, layers: List[Tuple[int, int]], batch: int, dtype: str) -> Tuple[float, float]:
        fwd = bwd = 0.0
        for in_dim, out_dim in layers:
            g = self._gemm_backend.simulate_gemm(batch, out_dim, in_dim, dtype=dtype)
            fwd += g.forward_time_ms
            bwd += g.backward_time_ms or (2.0 * g.forward_time_ms)
        return fwd, bwd

    def _embedding_a2a_ms(self, batch: int, seq_len: int) -> float:
        """Embedding all-to-all: exchange looked-up rows across the sharded world.

        A generative ranker is **not pooled** -- ``item_id``/``artist_id``/
        ``album_id`` are sequence features looked up once per valid position, so
        the payload scales with sequence length, not a single pooled vector per
        table.  We size it from the pooling-factor list (lookups per sample,
        summed over tables) that the embedding profiler already consumes.
        """
        mc = self.config.model_config
        n_tables = int(mc.num_embedding_tables or 0)
        dim = int(mc.embedding_dim or mc.hidden_size or 0)
        world = int(os.getenv("NNODES", "1")) * int(os.getenv("GPUS_PER_NODE", "8"))
        if world <= 1 or n_tables == 0 or dim == 0:
            return 0.0
        try:
            from primus.core.projection.module_profilers import (
                collective_args,
                collective_model,
            )

            emb = self.sub_profilers.get("sparse_embedding") if self.sub_profilers else None
            rows_per_sample = 0
            if emb is not None and hasattr(emb, "_tables"):
                _, _, pooling = emb._tables()
                rows_per_sample = int(sum(pooling))
            if rows_per_sample <= 0:
                rows_per_sample = n_tables  # fall back to pooled (1/table)
            gpn = int(os.getenv("GPUS_PER_NODE", "8"))
            nnodes = max(1, world // gpn)
            cargs = collective_args.get_default_args(num_nodes=nnodes, gpus_per_node=gpn)
            # Per-token payload: looked-up rows per sample x local batch x D,
            # exchanged as fp16 (qcomm a2a compresses to fp16/bf16).
            msg_bytes = batch * rows_per_sample * dim * 2
            us = collective_model.alltoall(cargs, msg_bytes, world, groups=["dp"])
            return float(us) / 1000.0  # us -> ms
        except Exception:
            return 0.0

    def _dense_flops_per_rank(self, local_bs: int, slen: int) -> float:
        """Per-rank dense forward FLOPs (HSTU GEMMs + attention core + MLPs).

        Used to emit MFU.  Excludes sparse-embedding gather (memory-bound, not a
        FLOP metric).  The attention core scales as E[L^2]; the GEMMs scale as
        E[L] tokens.
        """
        mc = self.config.model_config
        D = int(mc.hidden_size or mc.embedding_dim or 0)
        H = int(mc.hstu_num_heads or mc.num_attention_heads or 1)
        d_qk = int(mc.hstu_qk_dim or mc.kv_channels or (D // max(1, H)))
        d_v = int(mc.hstu_v_dim or d_qk)
        tp = max(1, int(self.config.model_parallel_config.tensor_model_parallel_size or 1))
        heads_pr = max(1, H // tp)

        mean = min(1.0, max(0.01, float(getattr(mc, "hstu_fill_factor", 1.0) or 1.0)))
        std = max(0.0, float(getattr(mc, "hstu_fill_factor_std", 0.0) or 0.0))
        mean_seq = max(1, int(round(slen * mean)))
        attn_seq = max(1, int(round(slen * min(1.0, (mean * mean + std * std) ** 0.5))))
        tokens = local_bs * mean_seq

        uvqk_n = heads_pr * (2 * d_qk + 2 * d_v)
        uvqk = 2.0 * tokens * uvqk_n * D
        out_in = int(getattr(mc, "hstu_output_input_dim", 0) or 0) or (heads_pr * d_v)
        out = 2.0 * tokens * D * out_in
        # Attention: QK^T + (softmax.V), causal ~= 0.5 of the dense triangle.
        attn = 0.5 * 2.0 * (2.0 * local_bs * heads_pr * attn_seq * attn_seq * (d_qk + d_v))
        hstu_fwd = self._num_layers() * (uvqk + out + attn)

        mlp_fwd = 0.0
        for in_dim, out_dim in self._bottom_mlp_layers() + self._over_mlp_layers():
            mlp_fwd += 2.0 * local_bs * in_dim * out_dim
        # Prediction-head towers run per sample and are replicated per task.
        n_tasks = max(1, int(getattr(mc, "dlrm_num_tasks", 1) or 1))
        for in_dim, out_dim in self._head_layers():
            mlp_fwd += 2.0 * n_tasks * local_bs * in_dim * out_dim
        # Preprocessor runs per valid token.
        for in_dim, out_dim in self._preprocessor_layers():
            mlp_fwd += 2.0 * tokens * in_dim * out_dim
        return hstu_fwd + mlp_fwd

    def project_step(self, batch_size: Optional[int] = None, seq_len: Optional[int] = None) -> dict:
        """First-cut per-step timing + throughput for a DLRM-v4 training step.

        Returns a dict with forward/backward/comm ms, step ms, samples/s, and
        MFU/HFU.  Requires simulation backends (call ``set_simulation_backends``
        first).
        """
        if self._gemm_backend is None or self._sdpa_backend is None:
            raise RuntimeError("DLRMProfiler.project_step requires simulation backends.")
        rc = self.config.runtime_config
        mc = self.config.model_config
        local_bs = int(batch_size or rc.micro_batch_size or 1)
        slen = int(seq_len or rc.sequence_length or mc.hstu_max_seq_len or 1)
        dtype = gemm_dtype_from_config(mc)

        emb = self.sub_profilers["sparse_embedding"]
        hstu = self.sub_profilers["hstu_layer"]

        fwd = emb.measured_forward_time(local_bs, slen)
        bwd = emb.measured_backward_time(local_bs, slen)

        layer_fwd = hstu.measured_forward_time(local_bs, slen)
        layer_bwd = hstu.measured_backward_time(local_bs, slen)
        fwd += self._num_layers() * layer_fwd
        bwd += self._num_layers() * layer_bwd

        # Bottom / over / prediction-head MLPs run per SAMPLE; the input
        # preprocessor runs per valid TOKEN (its M is the token count).
        mean = min(1.0, max(0.01, float(getattr(mc, "hstu_fill_factor", 1.0) or 1.0)))
        tokens = local_bs * max(1, int(round(slen * mean)))
        n_tasks = max(1, int(getattr(mc, "dlrm_num_tasks", 1) or 1))

        bmf, bmb = self._mlp_step_ms(self._bottom_mlp_layers(), local_bs, dtype)
        omf, omb = self._mlp_step_ms(self._over_mlp_layers(), local_bs, dtype)
        pmf, pmb = self._mlp_step_ms(self._preprocessor_layers(), tokens, dtype)
        hmf, hmb = self._mlp_step_ms(self._head_layers(), local_bs, dtype)
        hmf, hmb = n_tasks * hmf, n_tasks * hmb
        dense_mlp_fwd = bmf + omf + pmf + hmf
        dense_mlp_bwd = bmb + omb + pmb + hmb
        fwd += dense_mlp_fwd
        bwd += dense_mlp_bwd

        # Per-role breakdown (mirrors the Kineto trace buckets) for calibration.
        nL = self._num_layers()
        hc = hstu.measured_component_times(local_bs, slen)
        role_gemm = nL * (hc["gemm_fwd"] + hc["gemm_bwd"]) + (dense_mlp_fwd + dense_mlp_bwd)
        role_attn = nL * (hc["attn_fwd"] + hc["attn_bwd"])
        role_elem = nL * (hc["elem_fwd"] + hc["elem_bwd"])
        role_embed = emb.measured_forward_time(local_bs, slen) + emb.measured_backward_time(local_bs, slen)

        # Embedding all-to-all: only the *exposed* (non-overlapped) fraction is
        # on the critical path.
        comm_raw = self._embedding_a2a_ms(local_bs, slen)
        # NB: a bare ``or 1.0`` would turn a deliberate 0.0 (fully overlapped)
        # back into 1.0, so read the value explicitly before clamping.
        _exp = getattr(mc, "dlrm_comm_exposed_fraction", 1.0)
        exposed = min(1.0, max(0.0, 1.0 if _exp is None else float(_exp)))
        # The bandwidth term is a fair estimate of wire time, but traces show the
        # sparse collectives are dominated by peer-wait (barrier) cost that no
        # byte model produces; add it as a measured synchronization term.
        sync = max(0.0, float(getattr(mc, "dlrm_collective_sync_ms", 0.0) or 0.0))
        comm = comm_raw * exposed + sync

        # Optional data-path glue (measured inputs): H2D input copy, device
        # memcpy (jagged pack/unpack + a2a staging), and local reduce kernels.
        h2d = max(0.0, float(getattr(mc, "dlrm_h2d_ms", 0.0) or 0.0))
        memcpy = max(0.0, float(getattr(mc, "dlrm_memcpy_ms", 0.0) or 0.0))
        reduce = max(0.0, float(getattr(mc, "dlrm_reduce_ms", 0.0) or 0.0))

        # Framework glue: jagged pack/unpack + bf16<->fp32 casts/copies + bias/
        # norm reductions.  Memory-bound streaming whose per-token byte traffic
        # is a structural property of the implementation; scales with tokens.
        bpt = max(0.0, float(getattr(mc, "dlrm_glue_bytes_per_token", 0.0) or 0.0))
        glue = 0.0
        if bpt > 0:
            gfrac = min(1.0, max(0.05, float(getattr(mc, "dlrm_glue_hbm_fraction", 0.6) or 0.6)))
            peak_hbm_bytes_per_ms = resolve_hbm_bytes_per_ms(
                gemm_backend=self._gemm_backend,
                sdpa_backend=self._sdpa_backend,
                fallback_gbps=4000.0,
            )
            glue = tokens * bpt / (peak_hbm_bytes_per_ms * gfrac)

        step_ms = fwd + bwd + comm + h2d + memcpy + reduce + glue

        world = int(os.getenv("NNODES", "1")) * int(os.getenv("GPUS_PER_NODE", "8"))
        global_bs = int(rc.global_batch_size or (local_bs * world))
        samples_per_s = (global_bs / (step_ms / 1000.0)) if step_ms > 0 else 0.0

        # MFU/HFU: dense fwd+bwd FLOPs (bwd ~= 2x fwd) over the achieved step.
        peak_tflops = resolve_peak_tflops(self._gemm_backend, self._sdpa_backend, dtype="bf16")
        dense_fwd_flops = self._dense_flops_per_rank(local_bs, slen)
        dense_step_flops = 3.0 * dense_fwd_flops  # fwd + 2x bwd
        achieved_tflops = (dense_step_flops / (step_ms / 1000.0)) / 1e12 if step_ms > 0 else 0.0
        mfu = (achieved_tflops / peak_tflops) if peak_tflops else None

        return {
            "forward_ms": fwd,
            "backward_ms": bwd,
            "comm_ms": comm,
            "comm_ms_unoverlapped": comm_raw,
            "comm_sync_ms": sync,
            "h2d_ms": h2d,
            "glue_ms": glue,
            "step_ms": step_ms,
            "hstu_layer_fwd_ms": layer_fwd,
            "hstu_layer_bwd_ms": layer_bwd,
            "num_layers": self._num_layers(),
            "local_batch_size": local_bs,
            "global_batch_size": global_bs,
            "world_size": world,
            "samples_per_s": samples_per_s,
            "samples_per_s_per_gpu": samples_per_s / max(1, world),
            "achieved_tflops_per_gpu": achieved_tflops,
            "peak_tflops_per_gpu": peak_tflops,
            # No activation recompute is modelled, so HFU == MFU here.
            "mfu": mfu,
            "hfu": mfu,
            # Per-role breakdown for calibration against measured kernel time.
            "roles": {
                "attention": role_attn,
                "dense_gemm": role_gemm,
                "collectives": comm,
                "embedding": role_embed,
                "elementwise": role_elem,
                "memcpy": memcpy,
                "reduce": reduce,
                "glue": glue,
            },
        }

    # -- perf-path compatibility (unused by memory path) -----------------------
    def measured_forward_time(self, batch_size: int, seq_len: int) -> float:
        return self.project_step(batch_size, seq_len)["forward_ms"]

    def measured_backward_time(self, batch_size: int, seq_len: int) -> float:
        return self.project_step(batch_size, seq_len)["backward_ms"]

    def measured_activation_memory(self, batch_size: int, seq_len: int) -> int:
        return self.estimated_activation_memory(batch_size, seq_len)


def get_dlrm_profiler_spec(config) -> ModuleProfilerSpec:
    """Top-level profiler spec for a DLRM-v4 (TorchRec/HSTU) ranker."""
    return ModuleProfilerSpec(
        profiler=DLRMProfiler,
        config=config,
        sub_profiler_specs={
            "sparse_embedding": SparseEmbeddingProfiler,
            "hstu_layer": HSTULayerProfiler,
        },
    )
