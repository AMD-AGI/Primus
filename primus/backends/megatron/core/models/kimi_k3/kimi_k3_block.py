###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Portions of this file are adapted from Moonshot AI Kimi-Linear
# (https://huggingface.co/moonshotai/Kimi-K3), modeling_kimi_linear.py
# (KimiDecoderLayer).
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 decoder layer and block.

The layer interleaves Kimi Delta Attention with NoPE MLA
(``config.linear_attention_freq``) and threads the attention-residual
state. It is a transcription of ``KimiDecoderLayer._forward_attn_residual``
(``modeling_kimi_linear.py:973-1046``) onto Megatron's sequence-first
``[s, b, h]`` convention, with ``KimiLinearModel.forward``'s stack-level
bookkeeping (``:1188-1217``) in the block.

Why this is not a plain ``TransformerLayer``
--------------------------------------------

Two tensors flow between layers, ``(hidden_states, block_residual)``, but
upstream's ``TransformerLayer.forward`` returns ``(hidden_states,
context)`` and pipeline P2P carries a single ``[s, b, h]`` tensor.
DeepSeek-V4 hit the identical problem with its K parallel
HyperConnection streams and solved it by folding the extra axis into the
sequence axis at PP boundaries (``deepseek_v4_block.py:341-429``,
``:802-806``). This file copies that solution:
:func:`_lift_res_in` / :func:`_lower_res_out` are the K3 analogues of
``_lift_streams_in`` / ``_lower_streams_out``.

The K3 wrinkle is that the extra axis **grows with depth**. It is not
runtime state, though: a checkpoint is appended exactly when
``layer_idx % attn_res_block_size == 0`` (``:995``), so the number of
checkpoints in flight on entry to any layer is the pure function
:func:`attn_res_num_blocks_before` of its *global* index. Nothing but the
padded tensor has to cross a stage boundary, and the padding width is
``config.attn_res_num_blocks_max``.

Shape contract
--------------

* ``pre_process`` stage: ``hidden_states`` arrives ``[s, b, h]`` and
  ``block_residual`` is created empty, ``[s, b, 0, h]`` -- genuinely
  zero-width, which is what makes layer 0 skip its pre-attention mix
  (``:1190-1192`` and ``:987``).
* Non-first stage: ``hidden_states`` arrives ``[(1 + nb_max) * s, b, h]``
  and is unfolded.
* ``post_process`` stage: the head collapses the state and the block
  returns ``[s, b, h]``.
* Non-final stage: the block returns the folded
  ``[(1 + nb_max) * s, b, h]``.

At PP=1 ``pre_process`` and ``post_process`` are both true, so only the
first and third bullets are exercised. PP > 1 additionally needs the
pipeline scheduler to expect the folded wire shape, which is installed by
``primus/backends/megatron/patches/kimi_k3_pp_shape_patches.py`` (WP7);
without it the receiving stage's :func:`_lift_res_in` raises on the first
microbatch, because PyTorch P2P validates ``numel`` rather than shape.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import make_viewless_tensor
from torch import Tensor

from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
    KimiK3TransformerConfig,
)

logger = logging.getLogger(__name__)

__all__ = [
    "KimiK3LayerSubmodules",
    "KimiK3TransformerBlockSubmodules",
    "KimiK3Layer",
    "KimiK3TransformerBlock",
    "attn_res_num_blocks_before",
    "_lift_res_in",
    "_lower_res_out",
]


# ---------------------------------------------------------------------------
# Attention-residual bookkeeping (pure functions of the layer index)
# ---------------------------------------------------------------------------


def attn_res_num_blocks_before(layer_idx: int, block_size: Optional[int]) -> int:
    """Checkpoints in flight on entry to the 0-indexed ``layer_idx``.

    A checkpoint is appended by every layer whose index is a multiple of
    ``block_size`` (``modeling_kimi_linear.py:995``), so on entry to layer
    ``L`` the count is ``|{L' < L : L' % block_size == 0}| =
    ceil(L / block_size)``.

    With ``block_size = 4`` over 8 layers this is
    ``[0, 1, 1, 1, 1, 2, 2, 2]`` on *entry*, and appends land at layers 0
    and 4 -- the trace the block-level test asserts.

    Being a pure function of the global index is what makes the pipeline
    seam cheap: no fill count has to be sent between stages, only the
    padded tensor.
    """
    if not block_size:
        return 0
    return -(-int(layer_idx) // int(block_size))


# ---------------------------------------------------------------------------
# Submodule dataclasses
# ---------------------------------------------------------------------------


@dataclass
class KimiK3LayerSubmodules(TransformerLayerSubmodules):
    """Spec tree for one Kimi K3 decoder layer.

    Extends :class:`TransformerLayerSubmodules` -- as
    :class:`DeepseekV4HybridLayerSubmodules` does
    (``deepseek_v4_block.py:296-318``) -- so Megatron spec-lifecycle code
    that inspects ``submodules_config`` needs no K3 branch. The four core
    slots keep their upstream names (``input_layernorm`` /
    ``self_attention`` / ``pre_mlp_layernorm`` / ``mlp``).

    K3 adds the two attention-residual mixers. Both are ``None`` when
    ``attn_res_block_size`` is unset, in which case the layer degrades to
    the ordinary ``x = x + sublayer(x)`` residual.

    The inherited cross-attention and BDA slots stay at their defaults:
    K3 has no cross-attention, and the attention-residual mix replaces the
    bias-dropout-add residual entirely (``:987-1046`` has no dropout).
    """

    attn_res_mixer: Optional[Union[ModuleSpec, type]] = None
    mlp_res_mixer: Optional[Union[ModuleSpec, type]] = None


@dataclass
class KimiK3TransformerBlockSubmodules:
    """Spec tree for the Kimi K3 decoder block.

    ``attn_res_head`` is the post-stack mix
    (``KimiLinearModel._apply_output_attn_res``, ``:1226-1233``) and is
    built on the ``post_process`` stage only, mirroring how V4 builds
    ``hyper_head`` (``deepseek_v4_layer_specs.py:688``,
    ``deepseek_v4_block.py:899-913``).
    """

    layer_specs: Optional[List[ModuleSpec]] = None
    attn_res_head: Optional[Union[ModuleSpec, type]] = None
    final_layernorm: Optional[Union[ModuleSpec, type]] = None


# ---------------------------------------------------------------------------
# Attention-residual <-> sequence-axis packing (the PP P2P shape carrier)
# ---------------------------------------------------------------------------


def _lift_res_in(
    hidden_states: Tensor,
    *,
    pre_process: bool,
    num_blocks: int,
    num_blocks_max: int,
) -> Tuple[Tensor, Tensor]:
    """Split the block's input into ``(hidden_states, block_residual)``.

    Args:
        hidden_states: ``[s, b, h]`` on the first stage;
            ``[(1 + num_blocks_max) * s, b, h]`` on later stages, as
            packed by :func:`_lower_res_out`.
        pre_process: ``True`` on the first pipeline stage.
        num_blocks: checkpoints in flight on entry to this stage's first
            layer, i.e. :func:`attn_res_num_blocks_before` of its global
            index. Slots beyond it are padding and are dropped here.
        num_blocks_max: ``config.attn_res_num_blocks_max``; 0 disables the
            mechanism.

    Returns:
        ``([s, b, h], [s, b, num_blocks, h])``.
    """
    if num_blocks_max <= 0:
        empty = hidden_states.new_zeros(*hidden_states.shape[:-1], 0, hidden_states.shape[-1])
        return hidden_states, empty

    if pre_process:
        if num_blocks != 0:
            raise ValueError(
                f"the first pipeline stage must start with zero checkpoints, got {num_blocks}"
            )
        # modeling_kimi_linear.py:1190-1192 -- genuinely zero-width, so
        # layer 0's pre-attention mix is skipped rather than mixing a zero.
        seq, batch, hidden = hidden_states.shape
        return hidden_states, hidden_states.new_zeros(seq, batch, 0, hidden)

    packed, batch, hidden = hidden_states.shape
    stride = 1 + num_blocks_max
    if packed % stride != 0:
        raise ValueError(
            f"PP boundary tensor first dim {packed} is not divisible by "
            f"1 + attn_res_num_blocks_max = {stride}; the previous stage did not pack "
            "the attention-residual state via _lower_res_out."
        )
    seq = packed // stride
    unfolded = hidden_states.view(stride, seq, batch, hidden)
    # [num_blocks, s, b, h] -> [s, b, num_blocks, h] so the mixer sees the
    # candidate axis where _apply_attn_res has it.
    block_residual = unfolded[1 : 1 + num_blocks].permute(1, 2, 0, 3).contiguous()
    return unfolded[0].contiguous(), block_residual


def _lower_res_out(
    hidden_states: Tensor,
    block_residual: Tensor,
    *,
    post_process: bool,
    num_blocks_max: int,
) -> Tensor:
    """Fold ``(hidden_states, block_residual)`` back into one 3-D tensor.

    Args:
        hidden_states: ``[s, b, h]``. On the final stage this is already
            the collapsed output of :class:`AttentionResidualHead`.
        block_residual: ``[s, b, num_blocks, h]``. Ignored on the final
            stage; zero-padded to ``num_blocks_max`` otherwise.
        post_process: ``True`` on the final pipeline stage.
        num_blocks_max: padding width.

    Returns:
        ``[s, b, h]`` on the final stage, else
        ``[(1 + num_blocks_max) * s, b, h]`` -- a 3-D tensor of constant
        shape, which is all standard PP P2P kernels need.
    """
    if post_process or num_blocks_max <= 0:
        return hidden_states

    seq, batch, hidden = hidden_states.shape
    num_blocks = block_residual.shape[-2]
    if num_blocks > num_blocks_max:
        raise ValueError(f"num_blocks {num_blocks} exceeds attn_res_num_blocks_max {num_blocks_max}")

    # [s, b, nb, h] -> [nb, s, b, h], then pad the unused slots. The pad is
    # what keeps the P2P shape constant across stages; the receiving stage
    # slices it back off using attn_res_num_blocks_before.
    blocks = block_residual.permute(2, 0, 1, 3)
    if num_blocks < num_blocks_max:
        pad = blocks.new_zeros(num_blocks_max - num_blocks, seq, batch, hidden)
        blocks = torch.cat((blocks, pad), dim=0)
    packed = torch.cat((hidden_states.unsqueeze(0), blocks), dim=0)
    return packed.reshape((1 + num_blocks_max) * seq, batch, hidden)


# ---------------------------------------------------------------------------
# One Kimi K3 decoder layer
# ---------------------------------------------------------------------------


class KimiK3Layer(TransformerLayer):
    """One Kimi K3 decoder layer.

    Subclasses :class:`TransformerLayer` for type identity -- upstream
    ``isinstance(layer, TransformerLayer)`` checks and the
    ``BaseTransformerLayer`` utilities then apply -- but bypasses its
    ``__init__``, exactly as ``DeepseekV4HybridLayer`` does
    (``deepseek_v4_block.py:504-516``). The parent builds a BDA residual
    and a cross-attention slot that K3 does not have, and its
    ``forward`` has the wrong arity.

    Args:
        config: the runtime Kimi K3 config.
        submodules: :class:`KimiK3LayerSubmodules`.
        layer_idx: 0-indexed **global** layer index. Selects the attention
            variant and drives the checkpoint-append schedule, so it must
            be the global index even under pipeline parallelism.
        is_kda_layer: whether this layer is KDA. Recorded for
            introspection; the spec builder has already chosen the
            attention module from it.
        layer_number: 1-based index Megatron uses for the aux-loss
            tracker and sharded state dict. Defaults to ``layer_idx + 1``.
        pg_collection: process groups. Threaded explicitly to the MLP --
            see :meth:`_build_mlp`.
        use_attn_residuals: override the attention-residual mechanism for
            this layer alone. ``None`` (every decoder layer) derives it
            from ``config.attn_res_block_size``. Only the Multi-Token
            Prediction layer passes ``False``: it runs *after*
            :class:`AttentionResidualHead` has already collapsed the
            checkpoint set into the hidden state it consumes, so there is
            no ``block_residual`` left to mix and a softmax over the one
            remaining candidate would be the identity with two dead
            parameters. See ``kimi_k3_mtp_specs.py``.
        is_mtp_layer: whether this layer belongs to the MTP block.
            Forwarded to the MoE and on to its router, whose aux-loss
            tracker keys MTP depths separately from decoder layers
            (``router.py:454``, ``:468``, ``:531``, ``:541``).
    """

    def __init__(
        self,
        config: KimiK3TransformerConfig,
        submodules: KimiK3LayerSubmodules,
        *,
        layer_idx: int,
        is_kda_layer: bool = False,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        use_attn_residuals: Optional[bool] = None,
        is_mtp_layer: bool = False,
        **_unused_build_kwargs,
    ) -> None:
        del _unused_build_kwargs
        MegatronModule.__init__(self, config=config)

        if pg_collection is None:
            # MultiTokenPredictionLayer builds its inner layer without
            # forwarding pg_collection (multi_token_prediction.py:835-841), so
            # the MTP path always lands here. use_mpu_process_groups() reads
            # parallel_state, i.e. the real groups -- not the same thing as the
            # get_default_pg_collection() fallback DECISIONS.md warns about for
            # the MoE.
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.vp_stage = vp_stage

        self.layer_idx = int(layer_idx)
        self.is_kda_layer = bool(is_kda_layer)
        self.layer_number = int(layer_number) if layer_number is not None else self.layer_idx + 1
        self.is_mtp_layer = bool(is_mtp_layer)
        self.hidden_dropout = config.hidden_dropout
        # Cache for the parent's sharded_state_dict / recompute helpers.
        self.submodules_config = submodules

        self.attn_res_block_size = int(getattr(config, "attn_res_block_size", 0) or 0)
        self.use_attn_residuals = (
            self.attn_res_block_size > 0 if use_attn_residuals is None else bool(use_attn_residuals)
        )
        if self.use_attn_residuals and self.attn_res_block_size <= 0:
            raise ValueError(
                f"layer {self.layer_idx} was built with use_attn_residuals=True but "
                "config.attn_res_block_size is unset; the append schedule is undefined."
            )
        self.appends_checkpoint = (
            self.use_attn_residuals and self.layer_idx % self.attn_res_block_size == 0
        )
        self.num_blocks_in = (
            attn_res_num_blocks_before(self.layer_idx, self.attn_res_block_size)
            if self.use_attn_residuals
            else 0
        )

        hidden_size = int(config.hidden_size)
        norm_eps = float(config.layernorm_epsilon)

        self.input_layernorm = build_module(
            submodules.input_layernorm, config=config, hidden_size=hidden_size, eps=norm_eps
        )
        self.self_attention = build_module(
            submodules.self_attention,
            config=config,
            layer_number=self.layer_number,
            pg_collection=pg_collection,
        )
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm, config=config, hidden_size=hidden_size, eps=norm_eps
        )
        self.mlp = self._build_mlp(submodules.mlp, pg_collection=pg_collection)
        self.is_moe_layer = isinstance(self.mlp, MoELayer)

        if self.use_attn_residuals:
            assert submodules.mlp_res_mixer is not None, (
                "attn_res_block_size is set but the layer spec carries no mlp_res_mixer; "
                "every layer runs the pre-MLP mix (:1028-1033)."
            )
            self.mlp_res_mixer = build_module(submodules.mlp_res_mixer, config=config)
            # The pre-attention mix is skipped while no checkpoint exists,
            # i.e. on layer 0 only (:987). The reference still allocates
            # self_attention_res_{norm,proj} there (:908-917) and never
            # reaches them; building them would leave two parameters that
            # can never receive a gradient, which permanently disarms the
            # "every parameter gets a grad" check that is the cheapest test
            # for an unwired submodule. Drop them instead -- the same
            # structural choice V4 makes for attn_hc / ffn_hc at
            # hc_mult == 1 (deepseek_v4_block.py:689-691).
            if self.num_blocks_in > 0:
                assert submodules.attn_res_mixer is not None, (
                    f"layer {self.layer_idx} enters with {self.num_blocks_in} checkpoint(s) "
                    "and so runs the pre-attention mix, but the spec carries no "
                    "attn_res_mixer."
                )
                self.attn_res_mixer = build_module(submodules.attn_res_mixer, config=config)
            else:
                self.attn_res_mixer = None
        else:
            self.attn_res_mixer = None
            self.mlp_res_mixer = None

    # ------------------------------------------------------------------

    def _build_mlp(self, mlp_spec, *, pg_collection: ProcessGroupCollection) -> nn.Module:
        """Build the FFN, forwarding the process groups it actually needs.

        Upstream gates the ``pg_collection`` forward on an **identity**
        check against ``(MoELayer, TEGroupedMLP, SequentialMLP)``
        (``transformer_layer.py:374``), so ``StableLatentMoE`` -- a
        ``MoELayer`` subclass -- does not match and silently falls back to
        ``get_default_pg_collection()`` (``moe_layer.py:170-171``). A
        ``issubclass`` test is what makes the explicitly-threaded groups
        reach it. The adjacent ``isinstance`` at ``:402`` does match, so
        ``is_moe_layer`` would have come out right either way.
        """
        kwargs = {}
        module = getattr(mlp_spec, "module", mlp_spec)
        if isinstance(module, type) and issubclass(module, MoELayer):
            kwargs["pg_collection"] = pg_collection
            kwargs["is_mtp_layer"] = self.is_mtp_layer
        elif isinstance(module, type) and issubclass(module, MLP):
            assert hasattr(pg_collection, "tp"), "a TP process group is required for the dense MLP"
            kwargs["tp_group"] = pg_collection.tp

        mlp = build_module(mlp_spec, config=self.config, **kwargs)
        if hasattr(mlp, "set_layer_number"):
            # The router's aux-loss tracker indexes by layer number
            # (router.py:464-479) and the spec deliberately carries none,
            # so it has to be set after the build (transformer_layer.py:396).
            mlp.set_layer_number(self.layer_number)
        return mlp

    @staticmethod
    def _add_bias(output: Tensor, bias: Optional[Tensor]) -> Tensor:
        """Fold a sublayer's separated bias back into its output.

        K3's residual is a softmax mixture rather than an add, so there is
        nothing for upstream's fused ``bias_dropout_add`` to fuse with;
        the bias has to be applied before the mix. It is ``None`` on every
        K3 path (``add_bias_linear`` is false and the MLA / KDA output
        projections are bias-free), so this is a guard, not a hot path.
        """
        if bias is None:
            return output
        return output + bias

    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        block_residual: Optional[Tensor] = None,
        packed_seq_params=None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run one layer.

        Transcribes ``KimiDecoderLayer._forward_attn_residual``
        (``modeling_kimi_linear.py:984-1046``). ``prefix_sum`` is the
        running residual stream; ``block_residual`` is the growing set of
        cross-layer checkpoints.

        Args:
            hidden_states: ``[s, b, h]``.
            attention_mask: forwarded to the attention module. KDA ignores
                it (it is intrinsically causal); MLA passes it to
                ``core_attention``.
            block_residual: ``[s, b, num_blocks, h]``. ``None`` or
                zero-width means no checkpoints yet.
            packed_seq_params: forwarded to the attention module.

        Returns:
            ``(hidden_states, block_residual)``. The arity matches
            upstream's ``(hidden_states, context)`` -- and V4's
            ``(hidden, None)`` (``deepseek_v4_block.py:781``) -- so
            callers that unpack two values keep working; the second
            element carries the checkpoint state instead of a
            cross-attention context, because K3 has no cross-attention.
        """
        del kwargs  # rotary / inference kwargs: K3 is NoPE and training-only

        if not self.use_attn_residuals:
            attn_out = self._add_bias(
                *self.self_attention(
                    self.input_layernorm(hidden_states),
                    attention_mask=attention_mask,
                    packed_seq_params=packed_seq_params,
                )
            )
            hidden_states = hidden_states + attn_out
            mlp_out = self._add_bias(*self.mlp(self.pre_mlp_layernorm(hidden_states)))
            return hidden_states + mlp_out, block_residual

        if block_residual is None:
            seq, batch, hidden = hidden_states.shape
            block_residual = hidden_states.new_zeros(seq, batch, 0, hidden)

        # The runtime state and the index-derived schedule must agree; they
        # are computed independently (one by the appends in this loop, one by
        # attn_res_num_blocks_before) precisely so that a drift is caught
        # rather than silently changing which candidates get mixed.
        if block_residual.shape[-2] != self.num_blocks_in:
            raise AssertionError(
                f"layer {self.layer_idx} received {block_residual.shape[-2]} checkpoints but "
                f"its index implies {self.num_blocks_in}; the append schedule has drifted "
                "from attn_res_num_blocks_before."
            )

        prefix_sum: Optional[Tensor] = hidden_states

        # Pre-attention mix. Skipped only while no checkpoint exists, i.e.
        # at layer 0 (:987), which is also the only layer built without an
        # attn_res_mixer.
        if self.attn_res_mixer is not None:
            hidden_states = self.attn_res_mixer(prefix_sum, block_residual)

        # Append this block's checkpoint and reset the running sum (:995-998).
        if self.appends_checkpoint:
            block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)
            prefix_sum = None

        attn_out = self._add_bias(
            *self.self_attention(
                self.input_layernorm(hidden_states),
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
            )
        )
        # :1023-1026 -- after a reset the stream restarts from the sublayer
        # output rather than from the (discarded) pre-attention hidden.
        prefix_sum = attn_out if prefix_sum is None else prefix_sum + attn_out

        # Pre-MLP mix (:1028-1033). block_residual is non-empty here on
        # every layer, because layer 0 has just appended its own.
        hidden_states = self.mlp_res_mixer(prefix_sum, block_residual)

        mlp_out = self._add_bias(*self.mlp(self.pre_mlp_layernorm(hidden_states)))
        prefix_sum = prefix_sum + mlp_out

        return prefix_sum, block_residual


# ---------------------------------------------------------------------------
# The Kimi K3 decoder block
# ---------------------------------------------------------------------------


class KimiK3TransformerBlock(TransformerBlock):
    """The Kimi K3 decoder stack.

    Subclasses :class:`TransformerBlock` for type identity and its
    sharded-state-dict surface, bypassing its ``__init__`` for the same
    reasons ``DeepseekV4TransformerBlock`` does
    (``deepseek_v4_block.py:825-830``): the parent requires a real
    ``pg_collection`` and runs upstream-specific layer construction, while
    the spec tree plus the lift / lower helpers give the same
    functionality and stay CPU-instantiable.
    """

    def __init__(
        self,
        config: KimiK3TransformerConfig,
        spec=None,
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        submodules: Optional[KimiK3TransformerBlockSubmodules] = None,
    ) -> None:
        MegatronModule.__init__(self, config=config)

        self.spec = spec
        self.submodules = submodules
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        # The parent's __init__ is bypassed, so every attribute the parent's
        # *other* methods read has to be set here. ``tp_group`` is one:
        # ``TransformerBlock.sharded_state_dict`` passes it to
        # ``sharded_state_dict_default`` for every child that is not
        # ``self.layers`` (``transformer_block.py:953-963``), i.e. for
        # ``attn_res_head`` and ``final_layernorm``. Without it **saving a
        # Kimi K3 checkpoint raises** ``AttributeError: 'KimiK3TransformerBlock'
        # object has no attribute 'tp_group'`` -- at the first save_interval,
        # long after the run looks healthy. ``DeepseekV4TransformerBlock`` has
        # the same gap (``deepseek_v4_block.py:838`` sets only pg_collection).
        self.tp_group = pg_collection.tp
        # Required by the pipeline schedules, same contract as the parent.
        self.input_tensor = None

        self.num_blocks_max = int(config.attn_res_num_blocks_max)
        self.use_attn_residuals = self.num_blocks_max > 0

        # PP > 1 additionally needs the pipeline scheduler to expect the folded
        # wire shape [(1 + num_blocks_max) * s, b, h] rather than [s, b, h].
        # That lives in
        # primus/backends/megatron/patches/kimi_k3_pp_shape_patches.py, which
        # is gated on model_type == "kimi_k3" and attn_res_block_size > 0 and
        # PP > 1, and is applied at the `before_train` phase -- i.e. before any
        # model is built, so it is already installed by the time this runs.
        # It is not asserted here: this block is CPU-instantiable in unit tests
        # that have no Primus patch context at all, and an unpatched scheduler
        # fails loudly and immediately anyway (the receiving stage's
        # _lift_res_in rejects a first dim that is not divisible by
        # 1 + num_blocks_max, at :213-218).

        layer_specs = submodules.layer_specs if submodules is not None else None
        assert layer_specs, "Kimi K3 requires non-empty submodules.layer_specs."
        self.layers = nn.ModuleList()
        self.global_layer_indices: List[int] = []
        for local_idx, layer_spec in enumerate(layer_specs):
            layer = build_module(layer_spec, config=config, pg_collection=pg_collection)
            self.layers.append(layer)
            self.global_layer_indices.append(int(getattr(layer, "layer_idx", local_idx)))
        self.layer_offset = self.global_layer_indices[0] if self.global_layer_indices else 0

        hidden_size = int(config.hidden_size)
        norm_eps = float(config.layernorm_epsilon)

        # The post-stack mix lives on the final stage only (:1215-1217).
        if self.use_attn_residuals and self.post_process:
            head_spec = submodules.attn_res_head if submodules is not None else None
            assert head_spec is not None, (
                "the post_process stage needs an attn_res_head spec: without it the "
                "output_attn_res_{proj,norm} parameters are missing and the stack's final "
                "mix silently degrades to the identity."
            )
            self.attn_res_head = build_module(head_spec, config=config)
        else:
            self.attn_res_head = None

        if self.post_layer_norm and self.post_process:
            final_norm_spec = submodules.final_layernorm if submodules is not None else None
            assert final_norm_spec is not None, "Kimi K3 requires a final_layernorm spec."
            self.final_layernorm = build_module(
                final_norm_spec, config=config, hidden_size=hidden_size, eps=norm_eps
            )
        else:
            self.final_layernorm = None

        logger.info(
            "[Primus:Kimi-K3] decoder block: %d local layers %s, pre_process=%s post_process=%s, "
            "attn_res_num_blocks_max=%d",
            len(self.layers),
            self.global_layer_indices,
            pre_process,
            post_process,
            self.num_blocks_max,
        )

    # ------------------------------------------------------------------

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Pipeline-parallel hook: stash the tensor from the previous stage."""
        self.input_tensor = input_tensor

    @property
    def num_layers_per_pipeline_rank(self) -> int:
        """Compatibility shim for upstream debug / recompute code."""
        return len(self.layers)

    def attn_res_block_count_trace(self) -> List[int]:
        """``block_residual.shape[-2]`` on entry to each local layer.

        The value the per-layer bookkeeping test asserts on. Derived from
        the layer indices alone, which is the point: if the forward's
        appends ever drift from this schedule, the trace and the runtime
        shapes disagree.
        """
        return [layer.num_blocks_in for layer in self.layers]

    # ------------------------------------------------------------------

    def _recompute_local_layer_indices(self) -> Optional[set]:
        """Local indices of the layers to activation-checkpoint.

        Same semantics as ``DeepseekV4TransformerBlock``
        (``deepseek_v4_block.py:956-1001``): Primus's explicit
        ``recompute_layer_ids`` wins, then ``recompute_method``.
        """
        if not self.training:
            return None
        cfg = self.config
        if getattr(cfg, "recompute_granularity", None) != "full":
            return None
        n_local = len(self.layers)
        if n_local == 0:
            return None

        recompute_layer_ids = getattr(cfg, "recompute_layer_ids", None)
        if recompute_layer_ids:
            wanted = {int(i) for i in recompute_layer_ids}
            local = {
                local_idx
                for local_idx, global_idx in enumerate(self.global_layer_indices)
                if int(global_idx) in wanted
            }
            return local or None

        num = int(getattr(cfg, "recompute_num_layers", 0) or 0)
        if num <= 0:
            return None
        method = getattr(cfg, "recompute_method", None) or "block"
        if method == "block":
            return set(range(min(num, n_local)))
        if method == "uniform":
            return set(range(n_local))
        raise ValueError(f"Invalid recompute_method for Kimi K3: {method!r}")

    def _layer_fp8_context(self, global_idx: int):
        """Per-layer fp8 / fp4 autocast (no-op when both are off).

        This block overrides the parent's forward, so the per-layer
        ``get_fp8_context`` wrapping the stock loop applies is not
        inherited and has to be re-applied -- the same reason V4 does it
        (``deepseek_v4_block.py:1003-1032``). Resolved as a module
        attribute at call time so Primus's ``before_train`` fp8 patch is
        honoured.
        """
        if getattr(self.config, "fp4", None):
            from megatron.core import fp4_utils

            return fp4_utils.get_fp4_context(self.config, global_idx)
        if not self.config.fp8:
            return nullcontext()
        from megatron.core import fp8_utils

        return fp8_utils.get_fp8_context(self.config, global_idx)

    def _forward_layer_checkpointed(
        self,
        layer: KimiK3Layer,
        hidden_states: Tensor,
        block_residual: Tensor,
        attention_mask: Optional[Tensor],
        packed_seq_params,
        global_idx: int,
    ) -> Tuple[Tensor, Tensor]:
        """Run one layer under activation checkpointing.

        Both grad-carrying tensors are passed as checkpointed args and
        both are returned: ``CheckpointFunction`` filters its outputs to
        the ones that require grad (``random.py:571-578``), so a two-tensor
        signature is supported. The fp8 context is entered *inside* the
        closure so the recomputed forward matches the original.
        """
        from megatron.core import tensor_parallel

        def _run(hidden: Tensor, blocks: Tensor):
            with self._layer_fp8_context(global_idx):
                return layer(
                    hidden,
                    attention_mask,
                    block_residual=blocks,
                    packed_seq_params=packed_seq_params,
                )

        return tensor_parallel.checkpoint(
            _run,
            self.config.distribute_saved_activations,
            hidden_states,
            block_residual,
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        **kwargs,
    ) -> Tensor:
        """Run the Kimi K3 decoder. See the module docstring for shapes.

        The ``rotary_pos_*`` kwargs are accepted and ignored: K3 is NoPE
        everywhere, and its full-attention layers get that from a
        zero-width frequency table inside the attention module rather than
        from a missing argument here.
        """
        del rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin
        del sequence_len_offset, kwargs

        if inference_context is not None:
            raise NotImplementedError(
                "Kimi K3 has no inference cache yet: KimiDeltaAttention needs a recurrent "
                "state cache (KimiDynamicCache in the reference) that this work package "
                "does not build."
            )

        if not self.pre_process:
            hidden_states = self.input_tensor if self.input_tensor is not None else hidden_states
        if hidden_states is None:
            raise ValueError("KimiK3TransformerBlock.forward received no hidden_states tensor")

        hidden_states, block_residual = _lift_res_in(
            hidden_states,
            pre_process=self.pre_process,
            num_blocks=attn_res_num_blocks_before(
                self.layer_offset, getattr(self.config, "attn_res_block_size", 0)
            ),
            num_blocks_max=self.num_blocks_max,
        )

        recompute_local = self._recompute_local_layer_indices()
        for local_idx, layer in enumerate(self.layers):
            global_idx = self.global_layer_indices[local_idx]
            if recompute_local is not None and local_idx in recompute_local:
                hidden_states, block_residual = self._forward_layer_checkpointed(
                    layer,
                    hidden_states,
                    block_residual,
                    attention_mask,
                    packed_seq_params,
                    global_idx,
                )
            else:
                with self._layer_fp8_context(global_idx):
                    hidden_states, block_residual = layer(
                        hidden_states,
                        attention_mask,
                        block_residual=block_residual,
                        packed_seq_params=packed_seq_params,
                    )

        if self.attn_res_head is not None:
            hidden_states = self.attn_res_head(hidden_states, block_residual)

        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)

        out = _lower_res_out(
            hidden_states,
            block_residual,
            post_process=self.post_process,
            num_blocks_max=self.num_blocks_max,
        )
        return make_viewless_tensor(inp=out, requires_grad=out.requires_grad, keep_graph=True)
