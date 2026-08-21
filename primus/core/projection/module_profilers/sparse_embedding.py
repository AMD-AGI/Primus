###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Sparse-embedding profiler for DLRM-v4 / TorchRec rankers.

Unlike a language model's single dense vocab table, a production ranker holds
many large embedding tables (11 tables / ~560 GB of parameters for Yambda-5B)
that are sharded across the *whole* world by TorchRec/DMP.  The parameter count
is therefore dominated by these tables and is asymmetric across ranks, and the
runtime cost is a **memory-bound sparse gather** (random-access row reads +
pooling), not a GEMM.

This profiler models, first-principles:

* **Params** -- sum(rows_t * dim) over tables, sharded across the embedding shard
  group (row/table-wise sharding spreads them over the world; data sharding
  replicates them per rank).
* **Memory tiering** -- ``embedding_hbm_fraction`` of the table bytes live in HBM;
  the rest lives on DDR/UVM and is streamed over the host link.  Reported via
  :meth:`param_bytes_by_tier` so the memory model can account for the DDR tier
  separately instead of assuming everything sits in HBM.
* **Compute** -- forward gather = (sum pooling_factor * dim * param_bytes) random
  reads + pooled-output write, priced at an effective (random-access) HBM
  bandwidth; the DDR-resident share is priced at the (much lower) host-link
  bandwidth.  Backward is the scatter-add of the same footprint.
"""

import os
from typing import Optional

from primus.core.projection.base_module_profiler import BaseModuleProfiler

# Random-access sparse gather sustains only a fraction of peak HBM bandwidth.
_GATHER_HBM_FRACTION = 0.30
_PEAK_HBM_BYTES_PER_MS = 4.0e9  # ~4 TB/s (MI300-class), bytes/ms
# Host link (DDR/UVM streaming of non-resident tables), bytes/ms.
_HOST_LINK_BYTES_PER_MS = 6.0e7  # ~60 GB/s effective PCIe/xGMI host transfer


def _as_int_list(val, n, default):
    """Coerce a per-table field (list | "[...]" | scalar | None) to a length-n list."""
    if val is None:
        return [default] * n
    if isinstance(val, str):
        try:
            val = eval(val)  # config values are trusted (same pattern as moe_layer_freq)
        except Exception:
            val = default
    if isinstance(val, (int, float)):
        return [int(val)] * n
    lst = [int(x) for x in val]
    if len(lst) < n:
        lst = lst + [default] * (n - len(lst))
    return lst[:n]


class SparseEmbeddingProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        self._simulation_mode = True  # DLRM path is analytical (no eager module)
        self._cached = None
        self._cache_key = None

    # -- compatibility hooks (mirror EmbeddingProfiler / leaf profilers) -------
    def set_module(self, module):
        self._cached = None
        self._cache_key = None

    def set_simulation_mode(self, enabled: bool = True):
        self._simulation_mode = enabled

    def set_gemm_backend(self, backend):
        # Embedding gather is memory-bound; the GEMM backend is only used to read
        # a peak-HBM-bandwidth number when available.
        self._gemm_backend = backend

    # -- table geometry --------------------------------------------------------
    def _tables(self):
        mc = self.config.model_config
        n = int(mc.num_embedding_tables or 0)
        if n <= 0:
            return [], 0, []
        dim = int(mc.embedding_dim or mc.hidden_size or 0)
        rows = mc.embedding_table_rows
        if rows is None:
            total = int(mc.embedding_total_rows or 0)
            per = total // n if n else 0
            rows = [per] * n
        else:
            rows = _as_int_list(rows, n, int((mc.embedding_total_rows or 0) // max(1, n)))
        pooling = _as_int_list(mc.embedding_pooling_factor, n, int(mc.embedding_default_pooling_factor or 1))
        return rows, dim, pooling

    def _shard_group_size(self) -> int:
        """Number of ranks the embedding tables are sharded across.

        Row/table-wise sharding (the TorchRec/DMP default) spreads tables over
        the whole world; ``data`` sharding replicates them per rank.
        """
        mc = self.config.model_config
        if str(getattr(mc, "embedding_sharding", "row")).lower() == "data":
            return 1
        world = int(os.getenv("NNODES", "1")) * int(os.getenv("GPUS_PER_NODE", "8"))
        return max(1, world)

    # -- params ----------------------------------------------------------------
    def _total_embedding_params(self) -> int:
        rows, dim, _ = self._tables()
        return int(sum(r * dim for r in rows))

    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        total = self._total_embedding_params()
        if rank is None:
            return total
        return total // self._shard_group_size()

    def param_bytes_by_tier(self, rank: Optional[int] = 0) -> tuple[int, int]:
        """Return (hbm_bytes, ddr_bytes) of embedding params for this rank."""
        mc = self.config.model_config
        params = self.estimated_num_params(rank)
        total_bytes = params * int(mc.embedding_param_bytes or 4)
        hbm_frac = float(getattr(mc, "embedding_hbm_fraction", 1.0) or 1.0)
        hbm_frac = min(1.0, max(0.0, hbm_frac))
        hbm = int(total_bytes * hbm_frac)
        return hbm, total_bytes - hbm

    # -- activation ------------------------------------------------------------
    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        rows, dim, _ = self._tables()
        n = len(rows)
        # Pooled output: one D-vector per table per sample (bf16 activations).
        return int(batch_size * n * dim * 2)

    # -- compute ---------------------------------------------------------------
    def _get_simulated_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        rows, dim, pooling = self._tables()
        n = len(rows)
        param_bytes = int(self.config.model_config.embedding_param_bytes or 4)
        hbm_frac = min(
            1.0, max(0.0, float(getattr(self.config.model_config, "embedding_hbm_fraction", 1.0) or 1.0))
        )

        # Rows gathered per sample across all tables (sum pooling factor), times
        # the local batch, each row = dim * param_bytes.
        rows_gathered = batch_size * sum(pooling)
        gather_bytes = rows_gathered * dim * param_bytes
        output_bytes = batch_size * n * dim * 2  # pooled write (bf16)

        hbm_bytes = gather_bytes * hbm_frac + output_bytes
        ddr_bytes = gather_bytes * (1.0 - hbm_frac)

        fwd = hbm_bytes / (_PEAK_HBM_BYTES_PER_MS * _GATHER_HBM_FRACTION)
        if ddr_bytes > 0:
            fwd += ddr_bytes / _HOST_LINK_BYTES_PER_MS
        fwd = max(0.01, fwd)
        # Backward is the scatter-add of the same gathered rows (read-modify-write).
        bwd = 2.0 * fwd
        return (fwd, bwd, self.estimated_activation_memory(batch_size, seq_len))

    def _results(self, batch_size: int, seq_len: int):
        key = (batch_size, seq_len)
        if self._cached is None or self._cache_key != key:
            self._cached = self._get_simulated_results(batch_size, seq_len)
            self._cache_key = key
        return self._cached

    def measured_forward_time(self, batch_size: int, seq_len: int) -> float:
        return self._results(batch_size, seq_len)[0]

    def measured_backward_time(self, batch_size: int, seq_len: int) -> float:
        return self._results(batch_size, seq_len)[1]

    def measured_activation_memory(self, batch_size: int, seq_len: int) -> int:
        return self._results(batch_size, seq_len)[2]
