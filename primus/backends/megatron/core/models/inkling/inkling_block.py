###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling decoder layer and block (sequence-first ``[s, b, h]``)."""

from megatron.core import tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlock,
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import get_pg_rank

from primus.backends.megatron.core.transformer.inkling.inkling_dense_mlp import (
    InklingDenseMLP,
)
from primus.backends.megatron.core.transformer.inkling.inkling_layers import (
    InklingAttention,
    InklingRMSNorm,
    InklingShortConv,
)
from primus.backends.megatron.core.transformer.inkling.inkling_moe import InklingMoE


class InklingDecoderLayer(TransformerLayer):
    """x + attn_sconv(attn(norm(x))); then h + mlp_sconv(moe(norm(h))).

    Built by Megatron's :class:`TransformerLayer` constructor (process groups, the
    global layer number, the cuda-graph / offload bookkeeping; FSDP2 and
    Megatron-FSDP wrap ``TransformerLayer`` units) with identity slots, then fills
    the slots with Inkling's modules: its layer has short convs on both residual
    branches and no bias-dropout-add (Inkling has no hidden dropout) or
    cross-attention.

    ``layer_number`` is the 1-based index within this pipeline (virtual) stage, as
    ``TransformerBlock`` passes it; ``self.layer_number`` is Megatron's 1-based
    global number (distributed-checkpoint keys); ``self.layer_number - 1`` is the
    0-based global index (``inkling_local_layer_ids`` and the init seeds).

    Selective recompute follows Megatron's module names: ``core_attn`` (attention core,
    in :class:`InklingAttention`), ``layernorm`` (both pre-norms, output-discarding as in
    ``TransformerLayer``: the norm output is freed after its consumer ran and recomputed by a
    gradient hook), ``moe`` (the whole MoE layer), ``moe_act`` (the expert SwiGLU) and
    ``shared_experts`` (the shared experts), the last three in :class:`InklingMoE`. ``mlp``
    covers the dense MLP of the MTP depths only (:class:`InklingDenseMLP`).

    ``is_mtp_layer`` builds the block of a multi-token-prediction depth (Megatron passes
    ``layer_number`` = depth + 1): the same layer with a dense SwiGLU MLP (HF / SGLang force
    ``dense_mlp_idx`` past it) and the depth's own attention layout (sliding with the MTP window
    for the depths in ``inkling_mtp_local_layer_ids``, else global).
    """

    def __init__(
        self,
        config,
        layer_number=1,
        pg_collection=None,
        vp_stage=None,
        submodules=None,
        is_mtp_layer=False,
        **_kwargs,
    ):
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        if config.hidden_dropout:
            raise ValueError(
                f"Inkling has no hidden (residual) dropout: set hidden_dropout: 0.0 "
                f"(got {config.hidden_dropout})"
            )
        super().__init__(
            config=config,
            submodules=TransformerLayerSubmodules(),
            layer_number=layer_number,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=is_mtp_layer,
        )
        self.vp_stage = vp_stage
        if is_mtp_layer:
            # MTP depth d (``self.layer_number`` = d + 1, Megatron's MTP numbering): a dense MLP,
            # sliding attention when d is in inkling_mtp_local_layer_ids (with the MTP window /
            # heads), and init seeds after the decoder's (layer index num_layers + d).
            depth = self.layer_number - 1
            idx = config.num_layers + depth
            sliding = depth in (config.inkling_mtp_local_layer_ids or [])
            attn_kwargs = dict(
                sliding_window=config.inkling_mtp_window,
                swa_num_attention_heads=config.inkling_mtp_swa_heads,
            )
        else:
            idx = self.layer_number - 1  # 0-based global layer index (local-layer ids, init seeds)
            sliding = idx in (config.inkling_local_layer_ids or [])
            attn_kwargs = {}
        self.is_moe_layer = not is_mtp_layer
        eps = config.layernorm_epsilon
        H = config.hidden_size
        self.input_layernorm = InklingRMSNorm(H, eps)
        self.self_attention = InklingAttention(
            config, self.layer_number, idx, sliding, self.pg_collection, **attn_kwargs
        )
        # replicated over TP: the TP group sets their checkpoint replica ids
        tp_group = self.pg_collection.tp
        self.attn_sconv = InklingShortConv(
            H, config.inkling_sconv_kernel_size, tp_group=tp_group, impl=config.inkling_sconv_impl
        )
        self.pre_mlp_layernorm = InklingRMSNorm(H, eps)
        if is_mtp_layer:
            self.mlp = InklingDenseMLP(config, idx, tp_group=self.pg_collection.tp)
        else:
            self.mlp = InklingMoE(config, self.layer_number, idx, self.pg_collection)
        self.mlp_sconv = InklingShortConv(
            H, config.inkling_sconv_kernel_size, tp_group=tp_group, impl=config.inkling_sconv_impl
        )
        selective = config.recompute_granularity == "selective"
        modules = set(config.recompute_modules or []) if selective else set()
        self.recompute_norms = "layernorm" in modules

    def _norm(self, norm, x):
        """``(norm(x), checkpoint)``; with layernorm recompute (training) the checkpoint is a
        ``CheckpointWithoutOutput`` whose output the caller discards after its consumer ran."""
        if not (self.recompute_norms and self.training and x.requires_grad):
            return norm(x), None
        ckpt = tensor_parallel.CheckpointWithoutOutput(fp8=None)  # None = no fp8 (refused)
        return ckpt.checkpoint(norm, x), ckpt

    def forward(self, hidden_states, attention_mask=None, context=None, inference_context=None, **kwargs):
        """Megatron's layer contract: ``(hidden_states, context)``."""
        if inference_context is not None or kwargs.get("inference_params") is not None:
            raise NotImplementedError("Inkling: KV-cache inference is not supported")
        if kwargs.get("packed_seq_params") is not None:
            raise NotImplementedError("Inkling: packed sequences are not supported")
        padding_mask = kwargs.get("padding_mask")
        x, ckpt = self._norm(self.input_layernorm, hidden_states)
        attn, _ = self.self_attention(x, attention_mask=attention_mask)
        if ckpt is not None:  # free the norm output; recompute it when attn's gradient arrives
            ckpt.discard_output_and_register_recompute(attn)
        h = hidden_states + self.attn_sconv(attn)
        y, ckpt = self._norm(self.pre_mlp_layernorm, h)
        moe_out, _ = self.mlp(y, padding_mask=padding_mask)  # InklingMoE refuses a mask
        if ckpt is not None:
            ckpt.discard_output_and_register_recompute(moe_out)
        out = h + self.mlp_sconv(moe_out)
        return out, context


class InklingBlock(TransformerBlock):
    """The Inkling decoder stack for this pipeline stage (and virtual stage).

    Built by Megatron's :class:`TransformerBlock` constructor from a
    ``TransformerBlockSubmodules`` spec: one :class:`InklingDecoderLayer` spec per
    layer of this stage and Inkling's RMSNorm as the final norm (on the
    post-process stage only). The forward is Megatron's: full recompute
    (``uniform`` / ``block``, ``recompute_num_layers``), pipeline input tensors,
    viewless outputs and the final norm all come from ``TransformerBlock.forward``.
    """

    def __init__(self, config, pre_process=True, post_process=True, pg_collection=None, vp_stage=None):
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        n = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=get_pg_rank(pg_collection.pp))
        spec = TransformerBlockSubmodules(
            layer_specs=[ModuleSpec(module=InklingDecoderLayer)] * n, layer_norm=InklingRMSNorm
        )
        super().__init__(
            config=config,
            spec=spec,
            post_layer_norm=True,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Per-layer keys (``layers.{global}.``): sliding and global layers have different
        shapes (KV heads, relative-bias extent), so they cannot be stacked along a layer axis."""
        metadata = dict(metadata or {})
        metadata["non_homogeneous_layers"] = True
        return super().sharded_state_dict(prefix, sharded_offsets, metadata)
