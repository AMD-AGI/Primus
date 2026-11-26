###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

from primus.core.projection import kernel_report
from primus.core.projection.base_module_profiler import BaseModuleProfiler

_FALLBACK_HBM_BW_GBPS = 5300.0
_BW_EFFICIENCY = 0.65


# HBM traffic per logit element (forward, backward) and the bytes per element saved for the
# backward, by implementation. "native" is Megatron's jit-fused vocab-parallel CE: it upcasts
# the logits to fp32 and keeps exp(logits) for the backward (the traffic matches the fused
# Triton kernels of a gpt-oss MI355X trace). "te" is Transformer Engine's parallel CE, which
# writes the gradient into the bf16 logits the output layer already holds. Unfused is the same
# math as native, run as separate eager ops.
_CE_IMPLS = {
    "unfused": (38.0, 22.0, 4),
    "native": (22.0, 14.0, 4),
    "te": (6.0, 4.0, 0),
}


def ce_impl(model_config) -> str:
    if not getattr(model_config, "cross_entropy_loss_fusion", False):
        return "unfused"
    impl = getattr(model_config, "cross_entropy_fusion_impl", None) or "native"
    return impl if impl in _CE_IMPLS else "native"


class LossProfiler(BaseModuleProfiler):
    """Analytical model of the cross-entropy loss over the vocab-parallel logits.

    The loss is memory bound: time is the implementation's traffic per logit element over
    the achievable HBM bandwidth, and its activation is whatever it keeps for the backward.
    """

    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        self._gemm_backend = None

    def set_gemm_backend(self, backend):
        self._gemm_backend = backend

    # ----- params / memory -----

    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        return 0

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        mc = self.config.model_config
        tp = self.config.model_parallel_config.tensor_model_parallel_size
        cp = self.config.model_parallel_config.context_model_parallel_size
        tokens = batch_size * seq_len // cp
        vocab_per_rank = mc.padded_vocab_size // tp
        return tokens * vocab_per_rank * _CE_IMPLS[ce_impl(mc)][2]

    # ----- timing -----

    def _hbm_bw_gbps(self) -> float:
        if self._gemm_backend is not None:
            bw = getattr(self._gemm_backend, "hbm_bandwidth_gbps", None)
            if bw is not None:
                return bw
        return _FALLBACK_HBM_BW_GBPS

    def _loss_time_ms(self, batch_size: int, seq_len: int) -> tuple[float, float]:
        """Return (forward_ms, backward_ms) for the cross-entropy loss."""
        mc = self.config.model_config
        fwd_bytes, bwd_bytes, _ = _CE_IMPLS[ce_impl(mc)]
        tp = self.config.model_parallel_config.tensor_model_parallel_size
        cp = self.config.model_parallel_config.context_model_parallel_size
        tokens = batch_size * seq_len // cp
        vocab_per_rank = mc.padded_vocab_size // tp
        elements = tokens * vocab_per_rank
        eff_bw = self._hbm_bw_gbps() * _BW_EFFICIENCY  # GB/s

        fwd_ms = fwd_bytes * elements / (eff_bw * 1e6)
        bwd_ms = bwd_bytes * elements / (eff_bw * 1e6)
        dims = {"tokens": tokens, "vocab": vocab_per_rank}
        return (
            kernel_report.term("cross_entropy", "fwd", fwd_ms, kind="loss", **dims),
            kernel_report.term("cross_entropy", "bwd", bwd_ms, kind="loss", **dims),
        )

    def estimated_forward_time(self, batch_size: int, seq_len: int) -> float:
        fwd, _ = self._loss_time_ms(batch_size, seq_len)
        return fwd

    def estimated_backward_time(self, batch_size: int, seq_len: int) -> float:
        _, bwd = self._loss_time_ms(batch_size, seq_len)
        return bwd

    def measured_forward_time(self, batch_size: int, seq_len: int) -> float:
        return self.estimated_forward_time(batch_size, seq_len)

    def measured_backward_time(self, batch_size: int, seq_len: int) -> float:
        return self.estimated_backward_time(batch_size, seq_len)
