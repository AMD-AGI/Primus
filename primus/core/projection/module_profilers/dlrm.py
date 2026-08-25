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
from primus.core.projection.training_config import gemm_dtype_from_config


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
        emb_bpp = float(int(mc.embedding_param_bytes or 4)) + 4.0  # param + 1 fp32 moment
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

    def _embedding_a2a_ms(self, batch: int) -> float:
        """Embedding all-to-all: exchange pooled outputs across the sharded world."""
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

            gpn = int(os.getenv("GPUS_PER_NODE", "8"))
            nnodes = max(1, world // gpn)
            cargs = collective_args.get_default_args(num_nodes=nnodes, gpus_per_node=gpn)
            msg_bytes = batch * n_tables * dim * 2  # pooled bf16 payload per rank
            us = collective_model.alltoall(cargs, msg_bytes, world, groups=["dp"])
            return float(us) / 1000.0  # us -> ms
        except Exception:
            return 0.0

    def project_step(self, batch_size: Optional[int] = None, seq_len: Optional[int] = None) -> dict:
        """First-cut per-step timing + throughput for a DLRM-v4 training step.

        Returns a dict with forward/backward/comm ms, step ms, and samples/s.
        Requires simulation backends (call ``set_simulation_backends`` first).
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

        bmf, bmb = self._mlp_step_ms(self._bottom_mlp_layers(), local_bs, dtype)
        omf, omb = self._mlp_step_ms(self._over_mlp_layers(), local_bs, dtype)
        fwd += bmf + omf
        bwd += bmb + omb

        comm = self._embedding_a2a_ms(local_bs)
        step_ms = fwd + bwd + comm

        world = int(os.getenv("NNODES", "1")) * int(os.getenv("GPUS_PER_NODE", "8"))
        global_bs = int(rc.global_batch_size or (local_bs * world))
        samples_per_s = (global_bs / (step_ms / 1000.0)) if step_ms > 0 else 0.0

        return {
            "forward_ms": fwd,
            "backward_ms": bwd,
            "comm_ms": comm,
            "step_ms": step_ms,
            "hstu_layer_fwd_ms": layer_fwd,
            "hstu_layer_bwd_ms": layer_bwd,
            "num_layers": self._num_layers(),
            "local_batch_size": local_bs,
            "global_batch_size": global_bs,
            "world_size": world,
            "samples_per_s": samples_per_s,
            "samples_per_s_per_gpu": samples_per_s / max(1, world),
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
