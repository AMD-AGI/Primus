###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi Delta Attention (KDA) — the linear-attention mixer of Kimi K3.

KDA is Gated DeltaNet with a **per-channel** forget gate in place of the
per-head scalar one. The state recurrence, with ``S ∈ R^{K×V}``,
per-channel retention ``α_t = exp(g_t) ∈ R^K`` and per-head write
strength ``β_t ∈ R``::

    S_t = (I − β_t k_t k_tᵀ) · Diag(α_t) · S_{t−1} + β_t k_t v_tᵀ
    o_t = S_tᵀ q_t

The decay is applied *first* and the delta correction is taken against
the already-decayed state, making the transition
``Diag(α_t) − β_t k_t (α_t ⊙ k_t)ᵀ`` — diagonal-plus-rank-1 with both
low-rank vectors tied to ``k``. The full derivation, the chunkwise form
and the numerical contract live in
:mod:`.kda_kernels._eager.reference`; this module is the parameterised
Megatron wrapper around it.

Structure (transcribed from ``KimiDeltaAttention`` in the HF
``modeling_kimi_linear.py``, which the Kimi K3 text backbone reuses):

* separate ``q``/``k``/``v`` projections, each followed by its **own**
  short depthwise causal convolution (kernel 4) and SiLU;
* ``q``/``k`` L2-normalised, ``q`` scaled by ``K ** -0.5``;
* a low-rank (``hidden → head_dim → H·K``) per-channel log-decay gate,
  bounded as ``g = −5 · sigmoid(exp(A_log) · (z + dt_bias))``;
* a per-head write strength ``β = sigmoid(b_proj(x))``;
* a sigmoid-gated head-wise RMSNorm on the output, then ``o_proj``.

Layout follows Megatron's ``GatedDeltaNet``
(``megatron/core/ssm/gated_delta_net.py:70-75``): input and output are
``[s, b, h]``; the body works in ``[b, s, h, d]``.

Tensor parallelism
------------------
Sharding is over **heads**, exactly as ``GatedDeltaNet`` does
(``gated_delta_net.py:160-199``): every per-head / per-channel parameter
is allocated at its local-TP width and flagged
``tensor_model_parallel=True``, the three convolutions are built over
local channels, ``q``/``k``/``v``/``f_b``/``g``/``b`` projections are
column-parallel with ``gather_output=False``, and ``o_proj`` is
row-parallel with ``input_is_parallel=True``. The one projection that
must **not** be sharded is ``f_a_proj``: it produces the shared
``head_dim``-wide latent that ``f_b_proj`` expands, so it is built
duplicated, following the MLA low-rank down-projection idiom at
``multi_latent_attention.py:409-419``.

The arithmetic is therefore TP-correct for ``tp_size > 1``, but only
``tp_size == 1`` is exercised by the unit tests in this work package —
multi-rank numerical validation belongs with the layer/block assembly
work package.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import nvtx_range_pop, nvtx_range_push
from torch import Tensor, nn

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
    KDA_BACKENDS,
    kda_gate,
    kda_l2norm,
    resolve_kda_backend,
)

__all__ = [
    "KimiGatedRMSNorm",
    "KimiDeltaAttentionSubmodules",
    "KimiDeltaAttention",
]

logger = logging.getLogger(__name__)


def _param_device() -> Optional[torch.device]:
    """Device for directly-allocated parameters.

    ``GatedDeltaNet`` hardcodes ``torch.cuda.current_device()``; falling
    back to ``None`` (i.e. CPU) when no accelerator is visible is what
    lets the KDA unit tests build the module on a CPU-only host.
    """
    return torch.cuda.current_device() if torch.cuda.is_available() else None


def _duplicated_linear_kwargs(spec: Union[ModuleSpec, type]) -> dict:
    """Extra kwargs that make a linear spec **replicate** rather than shard.

    Mirrors the MLA low-rank down-projection idiom at
    ``multi_latent_attention.py:409-419``: TE's ``TELinear`` takes
    ``parallel_mode='duplicated'``, while the column-parallel classes
    reach the same result by sharding and gathering back.
    """
    cls = getattr(spec, "module", spec)
    if getattr(cls, "__name__", "") == "TELinear":
        return {"parallel_mode": "duplicated"}
    return {"gather_output": True}


class KimiGatedRMSNorm(nn.Module):
    """Head-wise RMSNorm with a sigmoid output gate.

    Numerically identical to ``fla.modules.FusedRMSNormGated(hidden_size,
    eps=eps, activation='sigmoid')``, which is what the HF reference
    instantiates as ``o_norm``: normalise over the last axis, scale by a
    learnable per-channel ``weight`` (initialised to ones, so a fresh
    norm is the identity), then multiply by ``sigmoid(gate)``. The whole
    chain runs in fp32 and the result is cast back to the input dtype,
    matching ``fla``'s kernel (which upcasts its tile on load and casts
    only on store).

    Kept as a plain module rather than reusing a Megatron norm because
    the gate multiply has to happen *inside* the fp32 region to match
    ``fla`` bit-for-bit, and because the ``weight`` parameter name then
    lines up with the released checkpoint's ``o_norm.weight``.
    """

    def __init__(
        self,
        hidden_size: Optional[int] = None,
        eps: float = 1e-5,
        *,
        config: Optional[TransformerConfig] = None,
        params_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if hidden_size is None:
            raise ValueError("KimiGatedRMSNorm requires `hidden_size`.")
        if params_dtype is None:
            params_dtype = torch.float32 if config is None else config.params_dtype
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.hidden_size, dtype=params_dtype, device=device))

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        out_dtype = x.dtype
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        y = y * self.weight.float()
        y = y * torch.sigmoid(gate.float())
        return y.to(out_dtype)


@dataclass
class KimiDeltaAttentionSubmodules:
    """Module specs for the linear layers and the gated output norm of KDA.

    ``q_proj`` / ``k_proj`` / ``v_proj`` / ``f_b_proj`` / ``g_proj`` /
    ``b_proj`` are column-parallel (``gather_output=False``); ``o_proj``
    is row-parallel (``input_is_parallel=True``); ``f_a_proj`` is the
    duplicated low-rank down-projection of the decay gate. ``out_norm``
    defaults to :class:`KimiGatedRMSNorm` so the module is self-contained
    — swap in ``fla.modules.FusedRMSNormGated`` to use the fused kernel.
    """

    q_proj: Union[ModuleSpec, type] = IdentityOp
    k_proj: Union[ModuleSpec, type] = IdentityOp
    v_proj: Union[ModuleSpec, type] = IdentityOp
    f_a_proj: Union[ModuleSpec, type] = IdentityOp
    f_b_proj: Union[ModuleSpec, type] = IdentityOp
    b_proj: Union[ModuleSpec, type] = IdentityOp
    g_proj: Union[ModuleSpec, type] = IdentityOp
    out_norm: Union[ModuleSpec, type] = KimiGatedRMSNorm
    o_proj: Union[ModuleSpec, type] = IdentityOp


class KimiDeltaAttention(MegatronModule):
    """Kimi Delta Attention layer.

    Takes input of size ``[s, b, h]`` and returns ``(output, bias)`` with
    the output the same size, matching Megatron's self-attention contract.

    Args:
        config: model config. The head geometry is read from the upstream
            linear-attention fields (``linear_num_value_heads``,
            ``linear_key_head_dim``, ``linear_value_head_dim``,
            ``linear_conv_kernel_dim``). The Kimi-K3-specific fields
            (``kda_gate_lower_bound``, ``kda_use_full_rank_gate``,
            ``kda_backend``, ``kda_chunk_size``) are read with ``getattr``
            defaults so this module works against a plain
            ``TransformerConfig`` until the config work package declares
            them. ``kda_backend`` is a string selector validated against
            :data:`KDA_BACKENDS` here, mirroring how
            ``DeepseekV4Attention`` validates ``use_v4_attention_backend``
            (``deepseek_v4_attention.py:695-730``); the ``use_`` prefix is
            dropped because it reads as a boolean and this is a choice.
        submodules: specs for the projections and the gated output norm.
        layer_number: 1-based index of this layer in the block.
        pg_collection: process groups; ``pg_collection.tp`` drives head
            sharding.
        conv_bias: bias on the three short convolutions (HF: ``False``).
        A_init_range: when given, ``A_log ~ log(U(a, b))`` as in the HF
            reference / Kimi Linear. When ``None`` (the default),
            ``A_log = 0``, i.e. ``exp(A_log) = 1``, which is what the
            Kimi K3 tech report specifies. The two sources genuinely
            disagree here; the report wins because it describes *training*
            whereas the HF initialiser is inert for an inference-only
            release.
        dt_init_range: ``dt_bias`` is initialised as
            ``inverse_softplus(dt)`` with ``dt ~ logU(dt_init_range)``.
            The HF reference leaves ``dt_bias`` genuinely uninitialised
            (``torch.empty``) and the tech report's sentence is truncated,
            so this is **unverified** — the Mamba/Mamba-2 convention is
            adopted because it is the only choice that puts the initial
            per-channel retention near 1 (``dt_bias`` around ``-6.9 ..
            -2.3`` gives ``α ≈ 0.64 .. 0.995`` under the bounded gate).
            Megatron's ``GatedDeltaNet`` instead uses ``dt_bias = 1``
            (``gated_delta_net.py:235-240``), which would start the layer
            near-total forgetting. Exposed so it can be swept.
        attn_mask_type: accepted and recorded, never used. KDA has no
            softmax and no mask tensor -- it is causal by construction,
            through the recurrence and the causal short convolution. The
            argument exists because
            ``MultiTokenPredictionLayer.__init__`` validates the inner
            layer's ``self_attention.params['attn_mask_type']` against
            ``SUPPORTED_ATTN_MASK`` (``multi_token_prediction.py:773-780``)
            and refuses to construct without it, so the KDA spec has to
            declare the param and this constructor has to tolerate it.
            DeepSeek-V4 took the identical step at its P16
            (``deepseek_v4_attention.py``, "accepts and ignores
            ``attn_mask_type``").
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: KimiDeltaAttentionSubmodules,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        conv_bias: bool = False,
        A_init_range: Optional[Tuple[float, float]] = None,
        dt_init_range: Tuple[float, float] = (1e-3, 1e-1),
        attn_mask_type=None,
        **_unused_spec_kwargs,
    ) -> None:
        del _unused_spec_kwargs
        super().__init__(config)

        self.attn_mask_type = attn_mask_type
        self.layer_number = layer_number
        self.conv_bias = conv_bias
        self.A_init_range = A_init_range
        self.dt_init_range = dt_init_range
        assert pg_collection is not None, "pg_collection must be provided for KimiDeltaAttention"
        self.pg_collection = pg_collection
        self.tp_size = self.pg_collection.tp.size()
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        self.hidden_size = config.hidden_size
        self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_heads = config.linear_num_value_heads
        # KDA is strictly MHA-shaped: one key head per value head, and the
        # gate/state live on the key axis, so the two head dims must agree.
        assert config.linear_num_key_heads == self.num_heads, (
            f"KDA requires linear_num_key_heads == linear_num_value_heads; got "
            f"{config.linear_num_key_heads} vs {self.num_heads}"
        )
        assert self.key_head_dim == self.value_head_dim, (
            f"KDA requires linear_key_head_dim == linear_value_head_dim; got "
            f"{self.key_head_dim} vs {self.value_head_dim}"
        )
        assert self.num_heads % self.tp_size == 0, (
            f"linear_num_value_heads ({self.num_heads}) must be divisible by "
            f"tensor-parallel size ({self.tp_size})"
        )
        # f_a_proj is replicated by sharding its output and gathering it back
        # (see _duplicated_linear_kwargs), which needs the gate latent width to
        # divide evenly too.
        assert self.key_head_dim % self.tp_size == 0, (
            f"linear_key_head_dim ({self.key_head_dim}) must be divisible by tensor-parallel "
            f"size ({self.tp_size}); it is the width of the replicated decay-gate latent"
        )

        # Kimi-K3-specific config, read defensively until the config WP lands.
        self.gate_lower_bound = getattr(config, "kda_gate_lower_bound", -5.0)
        self.use_full_rank_gate = getattr(config, "kda_use_full_rank_gate", True)
        self.backend_name = str(getattr(config, "kda_backend", "eager") or "eager")
        self.chunk_size = getattr(config, "kda_chunk_size", 64)
        if self.backend_name not in KDA_BACKENDS:
            raise ValueError(f"kda_backend must be one of {list(KDA_BACKENDS)}; got {self.backend_name!r}.")
        # Resolve once, at construction, following the DeepSeek-V4 attention idiom
        # (deepseek_v4_attention.py:737-771): a missing optional dependency then
        # surfaces while the model is being built rather than on the first
        # forward, and the per-step dispatch disappears from the hot path.
        self.kda_backend = resolve_kda_backend(self.backend_name)

        # Depthwise-conv1d impl, coordinated with the unified backend selector.
        # ``config.use_kimi_k3_attention_backend`` (when set) supersedes the
        # ``K3P_KDA_CONV`` env: the "fla" family uses fla ``causal_conv1d``, any
        # other family uses torch ``nn.Conv1d``. When it is None the conv defers
        # to ``K3P_KDA_CONV`` in :meth:`_short_conv` (default "default" = torch;
        # run_pretrain.sh sets it to "fla"), so the legacy behaviour is unchanged.
        # NOTE: the chunk kernel needs no handling here -- __post_init__ already
        # rewrote ``config.kda_backend`` to the unified selector, so
        # ``self.backend_name`` above reflects it.
        _attn_backend = getattr(config, "use_kimi_k3_attention_backend", None)
        self._kda_conv_use_fla = None if _attn_backend is None else (str(_attn_backend) == "fla")

        # One rank-0 line on the first KDA layer, so a run's log states plainly
        # which KDA path is live and what selected it (the B2 unified knob vs the
        # legacy kda_backend field + K3P_KDA_CONV env). Cheap and audit-friendly.
        if self.layer_number in (None, 1):
            import os as _os

            if self._kda_conv_use_fla is None:
                _conv = "fla" if _os.environ.get("K3P_KDA_CONV", "default") == "fla" else "torch"
                _src = "kda_backend + K3P_KDA_CONV (legacy)"
            else:
                _conv = "fla" if self._kda_conv_use_fla else "torch"
                _src = f"use_kimi_k3_attention_backend={_attn_backend!r}"
            try:
                import torch.distributed as _dist

                _rank0 = (not _dist.is_initialized()) or _dist.get_rank() == 0
            except Exception:
                _rank0 = True
            if _rank0:
                logger.info(
                    "[Primus:Kimi-K3] KDA backend resolved: chunk_kernel=%s conv=%s (via %s)",
                    self.backend_name,
                    _conv,
                    _src,
                )

        self.num_heads_local_tp = self.num_heads // self.tp_size
        self.qk_dim = self.key_head_dim * self.num_heads
        self.v_dim = self.value_head_dim * self.num_heads
        self.qk_dim_local_tp = self.qk_dim // self.tp_size
        self.v_dim_local_tp = self.v_dim // self.tp_size

        # --- q / k / v projections (column-parallel over heads) ----------
        self.q_proj = self._build_column_parallel(submodules.q_proj, self.qk_dim, "q_proj")
        self.k_proj = self._build_column_parallel(submodules.k_proj, self.qk_dim, "k_proj")
        self.v_proj = self._build_column_parallel(submodules.v_proj, self.v_dim, "v_proj")

        # --- three SEPARATE short depthwise causal convolutions ----------
        self.q_conv1d = self._build_conv1d(self.qk_dim_local_tp)
        self.k_conv1d = self._build_conv1d(self.qk_dim_local_tp)
        self.v_conv1d = self._build_conv1d(self.v_dim_local_tp)

        # --- per-channel log-decay gate ----------------------------------
        # f_a_proj is duplicated: its output is the shared head_dim-wide
        # latent that the column-parallel f_b_proj expands per head.
        self.f_a_proj = build_module(
            submodules.f_a_proj,
            self.hidden_size,
            self.key_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="f_a_proj",
            tp_group=self.pg_collection.tp,
            **_duplicated_linear_kwargs(submodules.f_a_proj),
        )
        self.f_b_proj = build_module(
            submodules.f_b_proj,
            self.key_head_dim,
            self.qk_dim,
            config=self._f_b_proj_config(),
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="f_b_proj",
            tp_group=self.pg_collection.tp,
        )
        self.A_log = nn.Parameter(
            torch.empty(self.num_heads_local_tp, dtype=torch.float32, device=_param_device())
        )
        setattr(self.A_log, "tensor_model_parallel", True)
        self.dt_bias = nn.Parameter(
            torch.empty(
                self.num_heads_local_tp * self.key_head_dim,
                dtype=torch.float32,
                device=_param_device(),
            )
        )
        setattr(self.dt_bias, "tensor_model_parallel", True)

        # --- per-head write strength (beta) ------------------------------
        self.b_proj = self._build_column_parallel(submodules.b_proj, self.num_heads, "b_proj")

        # --- output gate -------------------------------------------------
        # K3 sets use_full_rank_gate=True, so the gate is a single wide
        # projection rather than Kimi Linear's low-rank g_a/g_b pair.
        if not self.use_full_rank_gate:
            raise NotImplementedError(
                "Only the full-rank KDA output gate (kda_use_full_rank_gate=True, what Kimi K3 "
                "ships) is implemented; the low-rank g_a_proj/g_b_proj variant from Kimi Linear "
                "is not."
            )
        self.g_proj = self._build_column_parallel(submodules.g_proj, self.v_dim, "g_proj")

        self.out_norm = build_module(
            submodules.out_norm,
            config=self.config,
            hidden_size=self.value_head_dim,
            eps=self.config.layernorm_epsilon,
        )
        self._mark_out_norm_grads_for_tp_reduction()

        self.o_proj = build_module(
            submodules.o_proj,
            self.v_dim,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="o_proj",
            tp_group=self.pg_collection.tp,
        )

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _mark_out_norm_grads_for_tp_reduction(self) -> None:
        """Flag ``out_norm``'s gain so its gradient is summed across TP.

        ``out_norm`` is ``value_head_dim``-wide and is applied **per head**,
        so one replicated gain vector is shared by every head. With head
        sharding each rank therefore computes only the sum over *its* heads:
        a partial derivative, not the whole one. Measured at the debug shape
        in fp32, rank 0's gradient deviates from the TP=1 gradient by
        ``rel_rms`` 0.78 at TP=2 and 0.91 at TP=4 -- close to the
        ``sqrt(1 - 1/tp)`` a dropped-terms sum predicts -- while the TP sum
        matches to 1e-7 (``wp7/RESULTS.md``). Left alone, each rank's
        optimizer would apply a different partial gradient to a parameter
        that is supposed to be replicated, and the "same" weight would drift
        apart across TP ranks.

        ``sequence_parallel`` is the attribute Megatron uses to mean exactly
        "this parameter is replicated but its gradient is partial, sum it over
        the tensor-parallel group": ``_allreduce_non_tensor_model_parallel_grads``
        collects every parameter carrying it into a single coalesced
        all-reduce over ``tp_group`` (``finalize_model_grads.py:357-370``).
        The same mechanism already covers the attention-residual mixers
        (``attention_residual.py:127-129``) and upstream's router weight
        (``router.py:79``). It has to be that mechanism and not an autograd
        hook on ``.grad``: under DDP the gradient lives in ``main_grad``, and
        a ``.grad`` hook would be a silent no-op in real training.

        Note the same latent problem exists in upstream's ``GatedDeltaNet``,
        whose ``out_norm`` is built the same way (``gated_delta_net.py:
        201-207``) and carries no such flag.
        """
        if self.tp_size <= 1:
            return
        assert self.config.sequence_parallel, (
            "KimiDeltaAttention requires sequence_parallel=True whenever "
            f"tensor_model_parallel_size > 1 (got tp_size={self.tp_size}). Two independent "
            "reasons: (1) out_norm's gain is shared across heads while the heads are "
            "sharded, so its gradient is a partial sum that only "
            "_allreduce_non_tensor_model_parallel_grads reconstructs -- and that runs "
            "only under `config.sequence_parallel` (finalize_model_grads.py:358); "
            "(2) MoE + TP > 1 is refused at forward time without it anyway "
            "(moe_layer.py:484-488). Set sequence_parallel: true in the experiment yaml."
        )
        for param in self.out_norm.parameters(recurse=True):
            setattr(param, "sequence_parallel", True)

    def _f_b_proj_config(self):
        """A config copy with ``sequence_parallel`` off, for ``f_b_proj`` only.

        Every other projection here consumes ``hidden_states``, which under
        sequence parallelism is the rank's token shard ``[s/tp, b, h]``; a
        sequence-parallel column-parallel linear all-gathers that shard along
        the sequence axis before its GEMM, which is exactly right.

        ``f_b_proj`` is the one projection whose input is **not** a token
        shard. ``f_a_proj`` already gathered the sequence *and* gathered its
        output width, so ``z`` is a full-sequence, full-width ``[s, b,
        head_dim]`` tensor replicated on every TP rank. Leaving
        ``sequence_parallel`` on would make ``f_b_proj`` all-gather the
        sequence a **second** time and emit ``[s * tp, b, ...]``; KDA's own
        reshape then fails with ``shape '[b, s, h_local, d]' is invalid for
        input of size <tp * that>`` -- which is what it did before this
        override, so TP > 1 was unreachable for Kimi K3 in practice, because
        MoE + TP > 1 *requires* sequence parallelism
        (``moe_layer.py:484-488`` raises at forward time otherwise).

        Turning the flag off also fixes the backward: with
        ``sequence_parallel=False`` and ``tp_size > 1``,
        ``ColumnParallelLinear.__init__`` sets ``allreduce_dgrad``
        (``layers.py:900-902``), so the gradient w.r.t. the replicated ``z``
        is summed across TP -- which is required, since each rank's
        ``f_b_proj`` shard produces only a partial derivative of the loss with
        respect to it. ``f_a_proj``'s ``gather_from_tensor_model_parallel_region``
        then splits that complete gradient back to per-rank slices.

        A shallow copy rather than a mutation, following
        ``StableLatentMoE._latent_config`` and ``SharedExpertMLP.__init__:47``:
        the config object is shared with the rest of the layer.
        """
        if not getattr(self.config, "sequence_parallel", False):
            return self.config
        config = copy.copy(self.config)
        config.sequence_parallel = False
        return config

    def _build_column_parallel(self, spec: Union[ModuleSpec, type], output_size: int, buffer_name: str):
        return build_module(
            spec,
            self.hidden_size,
            output_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name=buffer_name,
            tp_group=self.pg_collection.tp,
        )

    def _build_conv1d(self, channels_local_tp: int) -> nn.Conv1d:
        """Depthwise causal convolution over the local-TP channel shard.

        ``padding=kernel-1`` plus the ``[..., :seq_len]`` truncation in
        :meth:`_short_conv` is the standard Megatron spelling of a causal
        depthwise conv (``gated_delta_net.py:166-175, 327``).
        """
        conv = nn.Conv1d(
            in_channels=channels_local_tp,
            out_channels=channels_local_tp,
            bias=self.conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=channels_local_tp,
            padding=self.conv_kernel_dim - 1,
            device=_param_device(),
            dtype=self.config.params_dtype,
        )
        setattr(conv.weight, "tensor_model_parallel", True)
        if self.conv_bias:
            setattr(conv.bias, "tensor_model_parallel", True)
        return conv

    def reset_parameters(self) -> None:
        """Initialise ``A_log`` and ``dt_bias``. See the class docstring."""
        if not self.config.perform_initialization:
            return
        # The TP RNG tracker only exists once a CUDA seed has been set; on a
        # CPU-only host (unit tests) there is nothing to fork.
        rng_ctx = get_cuda_rng_tracker().fork() if torch.cuda.is_available() else contextlib.nullcontext()
        with rng_ctx:
            if self.A_init_range is None:
                # Tech report §2.1.1: A_h = 0, i.e. exp(A_log) = 1.
                self.A_log.data.zero_()
            else:
                low, high = self.A_init_range
                assert 0 <= low <= high, f"A_init_range must be non-negative and ordered; got {(low, high)}"
                a = torch.empty_like(self.A_log).uniform_(low, high)
                self.A_log.data.copy_(torch.log(a))
            dt_min, dt_max = self.dt_init_range
            dt = torch.empty_like(self.dt_bias).uniform_(math.log(dt_min), math.log(dt_max)).exp()
            # inverse softplus: log(exp(dt) - 1), spelled stably
            self.dt_bias.data.copy_(dt + torch.log(-torch.expm1(-dt)))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _short_conv(self, x: Tensor, conv: nn.Conv1d, seq_len: int) -> Tensor:
        """Causal depthwise conv + SiLU on a ``[b, s, d]`` tensor."""
        import os

        # Priority: the yaml-level use_kimi_k3_attention_backend (resolved to
        # self._kda_conv_use_fla in __init__) wins; when it is None the legacy
        # K3P_KDA_CONV env decides (default "default" = torch nn.Conv1d).
        use_fla_conv = self._kda_conv_use_fla
        if use_fla_conv is None:
            use_fla_conv = os.environ.get("K3P_KDA_CONV", "default") == "fla"

        if use_fla_conv:
            import fla.modules.convolution as fc  # type: ignore

            fn = getattr(fc, "causal_conv1d", None) or getattr(fc, "causal_conv1d_fn", None)
            if fn is None:
                raise RuntimeError("K3P_KDA_CONV=fla but fla.modules.convolution has no causal conv")
            channels = conv.weight.shape[0]
            kernel = conv.weight.shape[-1]
            out = fn(
                x.contiguous(),
                conv.weight.view(channels, kernel),
                bias=conv.bias,
                activation="silu",
            )
            if isinstance(out, tuple):
                out = out[0]
            return out[..., :seq_len, :]

        y = conv(x.transpose(1, 2))[..., :seq_len]
        return F.silu(y.transpose(1, 2))

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[Any] = None,
        sequence_len_offset: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass.

        Args:
            hidden_states: ``[s, b, h]``.
            attention_mask: unused — KDA is intrinsically causal and this
                work package does not support padded or packed batches.

        Returns:
            ``(output, bias)`` with ``output`` of shape ``[s, b, h]``.
        """
        del attention_mask, key_value_states, attention_bias, sequence_len_offset
        if inference_context is not None:
            raise NotImplementedError("KimiDeltaAttention does not support inference caching yet.")
        if packed_seq_params is not None:
            raise NotImplementedError("KimiDeltaAttention does not support packed sequences yet.")

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size

        # --- projections, then three separate causal convs --------------
        nvtx_range_push(suffix="kda_in_proj")
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        # s b d -> b s d
        q, k, v = (x.transpose(0, 1) for x in (q, k, v))
        nvtx_range_pop(suffix="kda_in_proj")

        nvtx_range_push(suffix="kda_conv1d")
        q = self._short_conv(q, self.q_conv1d, seq_len)
        k = self._short_conv(k, self.k_conv1d, seq_len)
        v = self._short_conv(v, self.v_conv1d, seq_len)
        nvtx_range_pop(suffix="kda_conv1d")

        q = q.reshape(batch, seq_len, self.num_heads_local_tp, self.key_head_dim)
        k = k.reshape(batch, seq_len, self.num_heads_local_tp, self.key_head_dim)
        v = v.reshape(batch, seq_len, self.num_heads_local_tp, self.value_head_dim)
        q = kda_l2norm(q.contiguous())
        k = kda_l2norm(k.contiguous())

        # --- gate and write strength ------------------------------------
        nvtx_range_push(suffix="kda_gate")
        z, _ = self.f_a_proj(hidden_states)
        z, _ = self.f_b_proj(z)
        z = z.transpose(0, 1).reshape(batch, seq_len, self.num_heads_local_tp, self.key_head_dim)
        g = kda_gate(z, self.A_log, self.dt_bias, self.gate_lower_bound)
        beta, _ = self.b_proj(hidden_states)
        beta = torch.sigmoid(beta.transpose(0, 1).float())
        nvtx_range_pop(suffix="kda_gate")

        # --- the delta-rule recurrence ----------------------------------
        nvtx_range_push(suffix="kda_delta_rule")
        core_out, _ = self.kda_backend(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            scale=None,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            chunk_size=self.chunk_size,
        )
        nvtx_range_pop(suffix="kda_delta_rule")

        # --- sigmoid-gated head-wise RMSNorm, then o_proj ---------------
        nvtx_range_push(suffix="kda_out_norm")
        gate, _ = self.g_proj(hidden_states)
        gate = gate.transpose(0, 1).reshape(batch, seq_len, self.num_heads_local_tp, self.value_head_dim)
        out = self.out_norm(core_out, gate)
        nvtx_range_pop(suffix="kda_out_norm")

        # b s h d -> s b (h d)
        out = out.reshape(batch, seq_len, self.v_dim_local_tp).transpose(0, 1).contiguous()
        nvtx_range_push(suffix="kda_o_proj")
        out, out_bias = self.o_proj(out)
        nvtx_range_pop(suffix="kda_o_proj")
        return out, out_bias

    # ------------------------------------------------------------------
    # Distributed checkpointing
    # ------------------------------------------------------------------

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None, tp_group=None
    ) -> ShardedStateDict:
        """Sharded state dict; ``A_log`` / ``dt_bias`` / conv weights are TP-sharded on axis 0."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        tp_group = tp_group if tp_group is not None else self.pg_collection.tp

        sharded_state_dict: ShardedStateDict = {}
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={"A_log": 0, "dt_bias": 0},
            sharded_offsets=sharded_offsets,
            tp_group=tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )

        for name, module in self.named_children():
            if name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                axis_map = {"weight": 0}
                if self.conv_bias:
                    axis_map["bias"] = 0
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module.state_dict(prefix="", keep_vars=True),
                    f"{prefix}{name}.",
                    axis_map,
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=metadata["dp_cp_group"],
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
                )
            sharded_state_dict.update(module_sharded_sd)

        return sharded_state_dict
