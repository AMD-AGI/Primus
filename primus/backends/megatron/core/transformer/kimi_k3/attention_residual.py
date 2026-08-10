###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 attention residuals.

Kimi K3 replaces the ordinary ``x = x + sublayer(x)`` residual with a
**learned softmax mixture over a growing set of cross-layer checkpoints
plus the running residual stream**. The reference implementation is
``_apply_attn_res`` (``modeling_kimi_linear.py:1075-1088``), driven by
``KimiDecoderLayer._forward_attn_residual`` (``:973-1046``) twice per
layer and by ``KimiLinearModel._apply_output_attn_res`` (``:1226-1233``)
once after the stack.

This module owns the mixer only; the per-layer bookkeeping (when a
checkpoint is appended, when the running sum resets) lives in
``kimi_k3_block.KimiK3Layer``.

Three details of the reference are easy to get wrong, so they are
called out here and pinned by ``test_attention_residual.py``:

1. **The output mixes the un-normalised candidates.** ``v`` is the raw
   ``cat([block_residual, prefix_sum])``; ``k`` is its RMS-normalised
   form. The normalisation feeds the *scores* only
   (``modeling_kimi_linear.py:1083`` vs ``:1087``). Mixing ``k`` instead
   would rescale every candidate to unit RMS and destroy the residual
   stream's magnitude.
2. **The scorer is rank-1.** ``score_weight`` is the elementwise product
   of the RMSNorm gain ``[hidden]`` and the ``[1, hidden]`` projection
   weight (``:1084``), so scoring costs one dot product per candidate
   and ``2 * hidden`` parameters. The two factors could be folded into a
   single vector at build time; they are kept separate because the
   released checkpoint stores them separately as
   ``*_res_norm.weight`` / ``*_res_proj.weight``.
3. **All of it runs in fp32** and casts back once at the end (``:1081``,
   ``:1088``).

Layout note. The reference flattens ``[batch, seq]`` into a single token
axis and carries ``block_residual`` as ``[num_tokens, num_blocks,
hidden]``. Megatron is sequence-first, so this module keeps the tokens
as ``[seq, batch]`` and the candidate axis where the reference has it:
``prefix_sum`` is ``[..., hidden]`` and ``block_residual`` is
``[..., num_blocks, hidden]``. ``torch.matmul`` broadcasts over the
leading dims identically either way, so the arithmetic is unchanged --
``test_attention_residual.py`` asserts bit-equality against the
reference with the leading dims flattened.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels import (
    ATTN_RES_BACKENDS,
    accum_dtype,
    fused_score_weight,
    resolve_attn_res_backend,
)

__all__ = ["AttentionResidualMixer", "AttentionResidualHead"]


class AttentionResidualMixer(MegatronModule):
    """One attention-residual mixer: ``_apply_attn_res`` with its parameters.

    Args:
        config: the runtime config. ``hidden_size`` and
            ``layernorm_epsilon`` are read from it unless overridden.
        hidden_size: candidate width. Defaults to ``config.hidden_size``.
        eps: RMSNorm epsilon. Defaults to ``config.layernorm_epsilon``,
            which the Kimi K3 config keeps in step with ``norm_epsilon``
            (``kimi_k3_transformer_config.py:220-222``).

    Parameters:
        norm_weight: ``[hidden]``, HF ``*_res_norm.weight``. Ones at init,
            so a fresh mixer scores every candidate by the projection
            direction alone.
        proj_weight: ``[1, hidden]``, HF ``*_res_proj.weight``. Kept 2-D
            to match ``nn.Linear(hidden, 1, bias=False)``'s layout, which
            is what the reference squeezes at
            ``modeling_kimi_linear.py:1084``.

    Neither parameter is tensor-parallel: the residual stream is
    full-width on every TP rank, and the score reduces over the whole
    hidden axis, so the mixer needs no collective. Under sequence
    parallelism each rank holds a different slice of the token axis, so
    the two gradients *are* rank-dependent and must be all-reduced across
    TP -- which is exactly what ``sequence_parallel=True`` on a parameter
    asks ``finalize_model_grads`` to do (same treatment as the router's
    weight, ``router.py:79``).
    """

    def __init__(
        self,
        config: TransformerConfig,
        *,
        hidden_size: Optional[int] = None,
        eps: Optional[float] = None,
        params_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config=config)

        self.hidden_size = int(hidden_size if hidden_size is not None else config.hidden_size)
        self.eps = float(eps if eps is not None else config.layernorm_epsilon)

        # Stored in the model dtype and up-cast inside forward, which is what
        # the reference does: both factors are ordinary bf16 module parameters
        # and only ``score_weight`` is fp32 (modeling_kimi_linear.py:1084).
        # Storing them fp32 would not survive Float16Module's blanket
        # ``module.bfloat16()`` anyway (module.py:443-444).
        if params_dtype is None:
            params_dtype = config.params_dtype
        if (
            device is None
            and torch.cuda.is_available()
            and not getattr(config, "use_cpu_initialization", False)
        ):
            device = torch.cuda.current_device()

        self.norm_weight = nn.Parameter(torch.ones(self.hidden_size, dtype=params_dtype, device=device))
        self.proj_weight = nn.Parameter(torch.empty(1, self.hidden_size, dtype=params_dtype, device=device))

        sequence_parallel = bool(getattr(config, "sequence_parallel", False))
        for param in (self.norm_weight, self.proj_weight):
            setattr(param, "sequence_parallel", sequence_parallel)

        # Resolve the compute kernel once, at construction, following the
        # ``KimiDeltaAttention`` idiom (``kimi_delta_attention.py:287-295``): a
        # missing optional dependency then surfaces while the model is being
        # built rather than on the first forward, and the per-step dispatch
        # disappears from the hot path.
        self.backend_name = str(getattr(config, "attn_res_backend", "eager") or "eager")
        if self.backend_name not in ATTN_RES_BACKENDS:
            raise ValueError(
                f"attn_res_backend must be one of {list(ATTN_RES_BACKENDS)}; " f"got {self.backend_name!r}."
            )
        self.attn_res_backend = resolve_attn_res_backend(self.backend_name)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise the rank-1 scorer.

        ``norm_weight`` is ones (a fresh RMSNorm is the identity) and
        ``proj_weight`` follows ``config.init_method`` -- the same
        treatment ``KimiPreTrainedModel._init_weights`` gives an
        ``nn.Linear`` (``modeling_kimi_linear.py:1063-1068``), which is
        ``normal_(0, initializer_range)``.

        A zero ``proj_weight`` would also be defensible (uniform mixing
        at step 0) but it makes the two mixers in a layer numerically
        identical, which hides a wiring bug that swaps them.
        """
        if not getattr(self.config, "perform_initialization", True):
            return
        with torch.no_grad():
            self.norm_weight.fill_(1.0)
            init_method = getattr(self.config, "init_method", None)
            if init_method is None:
                self.proj_weight.normal_(mean=0.0, std=float(self.config.init_method_std))
            else:
                init_method(self.proj_weight)

    @staticmethod
    def _accum_dtype(dtype: torch.dtype) -> torch.dtype:
        """fp32, unless the operand is already wider.

        The reference spells this ``.float()`` unconditionally
        (``modeling_kimi_linear.py:1081``), which is an up-cast for every
        dtype Kimi K3 actually trains in (bf16 / fp16 / fp32) and a
        *down*-cast in fp64. Promoting instead is bit-identical on all
        three real cases and keeps the module differentiable under
        ``torch.autograd.gradcheck``, which needs fp64 end to end.
        """
        return accum_dtype(dtype)

    def score_weight(self, dtype: Optional[torch.dtype] = None) -> Tensor:
        """The fused rank-1 scoring vector, ``[hidden]``.

        ``modeling_kimi_linear.py:1084``. Exposed so tests and a future
        checkpoint adapter can check the factorisation without
        re-deriving it.
        """
        return fused_score_weight(self.norm_weight, self.proj_weight, dtype)

    def forward(self, prefix_sum: Tensor, block_residual: Tensor) -> Tensor:
        """Mix the running residual stream with the block checkpoints.

        The arithmetic lives in the backend resolved at construction
        (:mod:`.attn_res_kernels`); ``eager`` is the pure-PyTorch reference and
        the permanent oracle, ``flydsl`` is one fused kernel per direction. Both
        take the two scorer factors separately, so neither has a private copy of
        the ``norm_weight ⊙ proj_weight`` factorisation.

        Args:
            prefix_sum: ``[*, hidden]`` -- the running residual stream.
            block_residual: ``[*, num_blocks, hidden]`` -- the cross-layer
                checkpoints. ``num_blocks == 0`` is legal and makes the
                result a no-op softmax over a single candidate, i.e.
                ``prefix_sum`` itself; the caller normally skips the call
                entirely in that case (``:987``).

        Returns:
            ``[*, hidden]``, in ``prefix_sum``'s dtype.
        """
        if block_residual.shape[-1] != prefix_sum.shape[-1]:
            raise ValueError(
                f"block_residual hidden {block_residual.shape[-1]} != "
                f"prefix_sum hidden {prefix_sum.shape[-1]}"
            )
        return self.attn_res_backend(prefix_sum, block_residual, self.norm_weight, self.proj_weight, self.eps)


class AttentionResidualHead(AttentionResidualMixer):
    """The single post-stack mix (``_apply_output_attn_res``, ``:1226-1233``).

    Identical arithmetic to :class:`AttentionResidualMixer` with its own
    ``output_attn_res_{norm,proj}`` parameters; it exists as a separate
    class for the same reason DeepSeek-V4 splits ``HyperHead`` off
    ``HyperMixer`` (``hyper_connection.py:244,418``) -- the block builds
    exactly one of them, on the ``post_process`` stage only, so a type
    check is the cheapest way to assert that.
    """
