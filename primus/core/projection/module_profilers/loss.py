###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

from primus.core.projection.base_module_profiler import BaseModuleProfiler

_FALLBACK_HBM_BW_GBPS = 5300.0
_BW_EFFICIENCY = 0.65


class LossProfiler(BaseModuleProfiler):
    """Analytical model for cross-entropy loss computation time.

    Without fusion the loss reads the full logits tensor produced by the
    output layer, computes a softmax reduction over the vocab dimension,
    and then computes the negative-log-likelihood.  With TE-fused
    cross-entropy (``cross_entropy_loss_fusion=True``) the softmax + CE
    are folded into the output GEMM pipeline so that the separate logits
    read is eliminated.

    Memory passes (per token, per-rank):
      Forward  – 3 passes over ``vocab_per_rank`` elements:
        1. read logits for max subtraction (numerical stability)
        2. read logits to compute exp / sum
        3. write probabilities (or directly compute NLL)
      Backward – 3 passes:
        1. read softmax output
        2. read upstream grad
        3. write logits grad
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
        if getattr(mc, "cross_entropy_loss_fusion", False):
            return 0
        tp = self.config.model_parallel_config.tensor_model_parallel_size
        cp = self.config.model_parallel_config.context_model_parallel_size
        tokens = batch_size * seq_len // cp
        vocab_per_rank = mc.padded_vocab_size // tp
        return tokens * vocab_per_rank * 4  # fp32 softmax output

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
        if getattr(mc, "cross_entropy_loss_fusion", False):
            return 0.0, 0.0

        tp = self.config.model_parallel_config.tensor_model_parallel_size
        cp = self.config.model_parallel_config.context_model_parallel_size
        tokens = batch_size * seq_len // cp
        vocab_per_rank = mc.padded_vocab_size // tp
        bytes_per_el = 2  # bf16 logits

        eff_bw = self._hbm_bw_gbps() * _BW_EFFICIENCY  # GB/s
        tensor_bytes = tokens * vocab_per_rank * bytes_per_el

        fwd_ms = 3.0 * tensor_bytes / (eff_bw * 1e6)
        bwd_ms = 3.0 * tensor_bytes / (eff_bw * 1e6)
        return fwd_ms, bwd_ms

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
