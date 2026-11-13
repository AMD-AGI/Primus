###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mistral Large 4 (``mistral_large4_text``) builder and ``model_provider``."""

import copy
from typing import Optional

from megatron.core.transformer.spec_utils import import_module

from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layer_specs import (
    get_mistral_large4_decoder_block_spec,
)
from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
    holds_megatron_checkpoint,
)
from primus.backends.megatron.core.models.mistral_large4.mistral_large4_model import (
    MistralLarge4Model,
)
from primus.backends.megatron.core.models.mistral_large4.mistral_large4_transformer_config import (
    MistralLarge4TransformerConfig,
)

__all__ = ["build_mistral_large4_model", "mistral_large4_builder", "model_provider"]


def build_mistral_large4_model(
    config,
    *,
    vocab_size,
    max_sequence_length,
    pre_process=True,
    post_process=True,
    untie_embeddings_and_output_weights=True,
    vp_stage=None,
    pg_collection=None,
    spec=None,
):
    """The model from a config, without Megatron's global args."""
    spec = spec or get_mistral_large4_decoder_block_spec(config, vp_stage=vp_stage)
    return MistralLarge4Model(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=True,
        share_embeddings_and_output_weights=not untie_embeddings_and_output_weights,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )


def mistral_large4_config_from_args(args) -> MistralLarge4TransformerConfig:
    """``core_transformer_config_from_args`` with this model's config class, passed explicitly.

    Megatron swaps any requested class for ``MLATransformerConfig`` when
    ``args.multi_latent_attention`` is set, and accepts ``rope_type`` other than ``rope`` only
    then. The call therefore gets a copy of ``args`` with ``multi_latent_attention`` off and no
    ``rope_type``; ``__post_init__`` turns MLA back on and ``rope_type`` is set to ``yarn`` below.
    Nothing global is changed.
    """
    # megatron.training imports are function-local: importing it initialises CUDA, and the
    # model-provider registration (import_utils) imports this module on CPU-only hosts
    from megatron.training.arguments import core_transformer_config_from_args

    rope_type = getattr(args, "rope_type", None)
    if rope_type not in (None, "yarn"):
        raise ValueError(f"Mistral Large 4 uses YaRN rope (rope_type yarn), got {rope_type!r}")
    local = copy.copy(args)
    local.multi_latent_attention = False
    local.rope_type = None
    config = core_transformer_config_from_args(local, config_class=MistralLarge4TransformerConfig)
    if not isinstance(config, MistralLarge4TransformerConfig):
        raise TypeError(
            f"core_transformer_config_from_args built {type(config).__name__}, not MistralLarge4TransformerConfig"
        )
    config.rope_type = "yarn"
    seq_length = getattr(args, "seq_length", None)
    config.pp_carrier_seq_length = int(seq_length) if seq_length else None
    return config


def mistral_large4_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    from megatron.training import print_rank_0  # see mistral_large4_config_from_args

    print_rank_0("[Primus:Mistral-Large-4] building MistralLarge4Model...")
    if config is not None and not isinstance(config, MistralLarge4TransformerConfig):
        # Megatron's training loop passes the generic config it built from args; this model needs
        # its own class (per-layer lists, indexer), rebuilt from the same args
        print_rank_0(
            f"[Primus:Mistral-Large-4] rebuilding {type(config).__name__} as MistralLarge4TransformerConfig"
        )
        config = None
    if config is None:
        config = mistral_large4_config_from_args(args)
    print_rank_0(
        "[Primus:Mistral-Large-4] per-layer layout: "
        f"no_rope_layers={list(config.no_rope_layers)} indexer_types={list(config.indexer_types)} "
        f"kv_source_layers={list(config.kv_source_layers)}"
    )
    assert not getattr(
        args, "use_legacy_models", False
    ), "Mistral Large 4 requires use_legacy_models=False (Mcore only)."
    check_position_layout(args)
    check_indexer_source(args, config)
    spec_name = getattr(args, "spec", None)
    spec = import_module(spec_name) if spec_name is not None else None
    model = build_mistral_large4_model(
        config,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        untie_embeddings_and_output_weights=args.untie_embeddings_and_output_weights,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
        spec=spec,
    )
    check_stage_starts(model, args)
    if _experts_on_expert_groups(args):
        mark_experts_expert_parallel(model)
    if getattr(args, "freeze_routed_experts", False):
        n = freeze_routed_experts(model)
        print_rank_0(f"[Primus:Mistral-Large-4] freeze_routed_experts: {n} routed-expert tensors frozen")
    return model


# the parameter-name part of the routed experts (MoELayer.experts; the shared expert is
# .mlp.shared_experts.): freeze_routed_experts and mark_experts_expert_parallel select the same set
ROUTED_EXPERT_NAME_PART = ".mlp.experts."


def _is_routed_expert(name: str) -> bool:
    return ROUTED_EXPERT_NAME_PART in name


def freeze_routed_experts(model) -> int:
    """Freeze the routed experts (``.mlp.experts.``; the router, shared expert, attention, norms,
    embedding and head stay trainable); returns how many tensors were frozen.

    For fine-tuning on fewer GPUs than full training needs: the routed experts are ~97% of the
    parameters, so freezing them leaves no gradient or optimizer state for them (the 16-layer cut
    fine-tunes on one 8-GPU node with EP8)."""
    n = 0
    for name, param in model.named_parameters():
        if _is_routed_expert(name):
            param.requires_grad_(False)
            n += 1
    return n


def check_stage_starts(model, args) -> None:
    """At build time: a pipeline stage (or virtual stage) whose first layer is a ``shared``
    indexer layer has no top-k to reuse, which the sparse path (sequence > index_topk) needs.
    Refuse that layout when ``seq_length`` can exceed ``index_topk`` (it bounds every document,
    packed or not), instead of failing on the first long batch."""
    seq_length = int(getattr(args, "seq_length", 0) or 0)
    blocks = [m for m in model.modules() if getattr(m, "_stage_starts_on_shared", False)]
    if not blocks:
        return
    cfg = blocks[0].config
    if seq_length > int(cfg.index_topk):
        first = blocks[0].layer_indices[0]
        raise ValueError(
            f"Mistral Large 4: this pipeline stage starts on layer {first}, a 'shared' indexer layer "
            f"(indexer_types), and seq_length {seq_length} > index_topk {cfg.index_topk} runs sparse "
            "attention, which needs the top-k of the layer before it: choose a pipeline split "
            "(decoder_first/last_pipeline_num_layers, pipeline_model_parallel_layout) whose stages "
            "start on full-indexer layers"
        )


def check_indexer_source(args, cfg) -> None:
    """Build-time half of the frozen-indexer guard (the other half,
    ``MistralLarge4Attention._check_indexer_loaded``, checks at the first sparse forward that the
    indexer weights are not their init values, which also catches a run resumed from its own
    checkpoint of a from-scratch run).

    The DSA indexer is frozen (the reference runs it under ``no_grad`` and trains no indexer
    loss), so its top-k is only meaningful with weights loaded from a checkpoint. With
    ``seq_length`` > ``index_topk`` the top-k selects the attended keys; without a checkpoint
    (``load`` / ``pretrained_checkpoint``) that selection would come from a random, never-trained
    indexer for the whole run. Refuse it. ``seq_length`` <= ``index_topk`` is dense causal
    attention (every key is selected), so from-scratch training is accepted there."""
    seq_length = int(getattr(args, "seq_length", 0) or 0)
    if seq_length <= int(cfg.index_topk):
        return
    if holds_megatron_checkpoint(getattr(args, "load", None)) or holds_megatron_checkpoint(
        getattr(args, "pretrained_checkpoint", None)
    ):
        return
    if getattr(args, "perform_initialization", True) is False:
        # weights are filled after the build (the converter / a parity harness); the first sparse
        # forward still checks that the indexer was loaded (MistralLarge4Attention)
        from megatron.training import print_rank_0

        print_rank_0(
            "[Primus:Mistral-Large-4] perform_initialization false: the DSA indexer must be loaded before "
            "sparse attention (checked at the first sparse forward)"
        )
        return
    if getattr(args, "allow_untrained_indexer", False):
        from megatron.training import print_rank_0

        print_rank_0(
            "[Primus:Mistral-Large-4] WARNING: allow_untrained_indexer: sparse DSA attention selects keys "
            "with a random, frozen indexer (code-path test; the selection does not train)"
        )
        return
    raise ValueError(
        f"Mistral Large 4: seq_length {seq_length} > index_topk {cfg.index_topk} runs DSA sparse "
        "attention, whose top-k comes from the frozen indexer (not trained, as in the reference). "
        "Without a checkpoint (load / pretrained_checkpoint) the indexer is random and the "
        "selection never improves. Load the converted release, or train from scratch with "
        f"seq_length <= {cfg.index_topk} (dense causal attention); allow_untrained_indexer: true "
        "accepts it for code-path tests."
    )


def check_position_layout(args) -> None:
    """Rope / SSMax positions are the block's own (0..s-1, or per document with packed THD
    sequences); per-document positions or masks in an unpacked batch are not supported."""
    packed = bool(getattr(args, "use_packed_attention", False))
    for flag in ("reset_position_ids", "reset_attention_mask"):
        if getattr(args, flag, False) and not packed:
            raise NotImplementedError(
                f"Mistral Large 4: {flag} without packed (THD) sequences is not supported; rope / SSMax "
                "positions restart per document only with packed sequences"
            )


def _experts_on_expert_groups(args) -> bool:
    """With the plain (non-distributed) optimizer, a checkpoint holds one optimizer per
    parameter kind: dense, plus expert when the experts use the expert process groups. Megatron
    puts the experts on those groups only for EP > 1 (or ETP != TP), so an EP1 checkpoint has
    one optimizer and an EP2 run expects two, and the checkpoint does not load across EP sizes.
    Mistral Large 4 keeps its experts on the expert groups at every EP size in that case (with
    EP1 and ETP == TP the expert data-parallel group is the data-parallel group: same
    reductions)."""
    return not (
        getattr(args, "use_distributed_optimizer", False)
        or getattr(args, "use_megatron_fsdp", False)
        or getattr(args, "use_torch_fsdp2", False)
    )


def mark_experts_expert_parallel(model) -> int:
    """Mark the routed experts' parameters as expert-parallel (``allreduce = False``), as
    Megatron does for EP > 1; returns how many were marked."""
    n = 0
    for name, param in model.named_parameters():
        if _is_routed_expert(name) and getattr(param, "allreduce", True):
            param.allreduce = False
            n += 1
    return n


def model_provider(
    model_builder=None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config=None,
    pg_collection=None,
):
    from megatron.training import get_args  # see mistral_large4_config_from_args

    model_builder = model_builder or mistral_large4_builder
    return model_builder(
        get_args(), pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection
    )
