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

    def benchmark_loss(self, loss_owner, vocab_per_rank: int, batch_size: int, seq_len: int) -> None:
        """Time the model's own ``compute_language_model_loss`` fwd + bwd.

        "Fused" cross-entropy still runs as separate kernels after the output
        GEMM (softmax statistics over the vocab, then the logits gradient), so
        it is measured rather than assumed free; whichever loss backend the run
        installs (Megatron fused CE, Primus-Turbo CE) is what gets timed.
        """
        import torch

        from primus.core.projection.module_profilers.utils import _bench_iter_count

        cp = self.config.model_parallel_config.context_model_parallel_size
        s = seq_len // cp

        def _measure():
            logits = torch.randn(s, batch_size, vocab_per_rank, device="cuda", dtype=torch.bfloat16)
            labels = torch.randint(0, vocab_per_rank, (batch_size, s), device="cuda")
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]

            def step(timed):
                x = logits.detach().requires_grad_(True)
                if timed:
                    ev[0].record()
                loss = loss_owner.compute_language_model_loss(labels, x)
                if timed:
                    ev[1].record()
                loss.backward(torch.ones_like(loss))
                if timed:
                    ev[2].record()

            num_warmup, num_iters = _bench_iter_count(3, 10)
            for _ in range(num_warmup):
                step(False)
            fwd = bwd = 0.0
            for _ in range(num_iters):
                step(True)
                torch.cuda.synchronize()
                fwd += ev[0].elapsed_time(ev[1])
                bwd += ev[1].elapsed_time(ev[2])
            return fwd / num_iters, bwd / num_iters

        self._measured = _measure()
        torch.cuda.empty_cache()

    def measured_forward_time(self, batch_size: int, seq_len: int) -> float:
        measured = getattr(self, "_measured", None)
        return measured[0] if measured else self.estimated_forward_time(batch_size, seq_len)

    def measured_backward_time(self, batch_size: int, seq_len: int) -> float:
        measured = getattr(self, "_measured", None)
        return measured[1] if measured else self.estimated_backward_time(batch_size, seq_len)
