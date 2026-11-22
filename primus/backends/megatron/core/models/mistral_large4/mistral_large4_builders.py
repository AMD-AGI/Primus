###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mistral Large 4 (``mistral_large4_text``) builder and ``model_provider``."""

import copy

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
from primus.core.utils.flag_utils import bool_flag

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
    # its reader (the decoder block's _expected_local_seq) treats an unset or 0 length as unknown
    config.pp_carrier_seq_length = getattr(args, "seq_length", None)
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
    check_sparse_context_parallel(args, config)
    check_indexer_source(args, config)
    check_stage_starts(config, args)
    if getattr(args, "spec", None) is not None:
        # the block spec carries the per-layer NoPE / indexer / KV-sharing layout: a generic layer spec
        # would build another model
        raise ValueError(
            f"Mistral Large 4 builds its own decoder block spec; spec={args.spec!r} is not supported (unset it)"
        )
    model = build_mistral_large4_model(
        config,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        untie_embeddings_and_output_weights=args.untie_embeddings_and_output_weights,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )
    if _experts_on_expert_groups(args):
        mark_experts_expert_parallel(model)
    if bool_flag(getattr(args, "freeze_routed_experts", False), "freeze_routed_experts", False):
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

    For fine-tuning on fewer GPUs than full training needs: the routed experts are most of the
    parameters, so freezing them leaves no gradient or optimizer state for them."""
    n = 0
    for name, param in model.named_parameters():
        if _is_routed_expert(name):
            param.requires_grad_(False)
            n += 1
    return n


def _runs_sparse(args, cfg) -> bool:
    """Whether ``args.seq_length`` can run the sparse DSA path (sequence > ``index_topk``; the
    sequence length bounds every document, packed or not)."""
    return int(getattr(args, "seq_length", 0) or 0) > int(cfg.index_topk)


def check_sparse_context_parallel(args, cfg) -> None:
    """At build time, before any checkpoint is loaded: the sparse DSA path (seq_length >
    index_topk) does not run with context parallelism; the block's first sparse forward refuses it
    too, after the whole model is built and loaded. A shorter seq_length runs dense attention with CP."""
    cp = int(getattr(args, "context_parallel_size", 1) or 1)
    if cp > 1 and _runs_sparse(args, cfg):
        raise NotImplementedError(
            f"Mistral Large 4 DSA sparse attention (seq_length {args.seq_length} > index_topk {cfg.index_topk}) "
            f"with context_parallel_size {cp} is not supported; use context_parallel_size 1 or "
            "seq_length <= index_topk"
        )


def stage_start_layers(config) -> list:
    """The first layer (0-based) of every pipeline stage and virtual stage, from the config alone
    (the same offsets and layer counts Megatron's TransformerBlock builds with); stages without
    layers are left out."""
    pp = int(getattr(config, "pipeline_model_parallel_size", 1) or 1)
    vpp = getattr(config, "virtual_pipeline_model_parallel_size", None)
    layout = getattr(config, "pipeline_model_parallel_layout", None)
    starts = []
    if layout is not None:
        from megatron.core.transformer.enums import LayerType

        if isinstance(layout, (str, list)):
            from megatron.core.transformer.pipeline_parallel_layer_layout import (
                PipelineParallelLayerLayout,
            )

            layout = PipelineParallelLayerLayout(layout, pp)

        offset = 0
        for v in range(len(layout.layout[0])):
            for p in range(pp):
                n = list(layout.layout[p][v]).count(LayerType.decoder)
                if n:
                    starts.append(offset)
                offset += n
        return starts
    if pp == 1:
        return [0] if config.num_layers else []
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    vp_stages = range(int(vpp)) if vpp else [None]
    for v in vp_stages:
        for p in range(pp):
            if get_num_layers_to_build(config, vp_stage=v, pp_rank=p):
                starts.append(get_transformer_layer_offset(config, vp_stage=v, pp_rank=p))
    return starts


def check_stage_starts(config, args) -> None:
    """Before the model is built (no stage is allocated yet): a pipeline stage (or virtual
    stage) whose first layer is a ``shared`` indexer layer has no top-k to reuse, which the sparse
    path (sequence > index_topk) needs. Refuse that layout when ``seq_length`` can exceed
    ``index_topk`` (it bounds every document, packed or not), on every rank alike, instead of
    after a stage of the model is built or on the first long batch. The error lists the PP sizes
    (an even split, no VPP) whose stages all start on full-indexer layers for this config
    (``valid_even_pp_sizes``)."""
    if not _runs_sparse(args, config):
        return
    bad = [i for i in stage_start_layers(config) if config.indexer_types[i] == "shared"]
    if bad:
        raise ValueError(
            f"Mistral Large 4: pipeline stages start on layers {bad}, 'shared' indexer layers "
            f"(indexer_types), and seq_length {args.seq_length} > index_topk {config.index_topk} runs "
            "sparse attention, which needs the top-k of the layer before it: choose a pipeline split "
            "(pipeline_model_parallel_size, decoder_first/last_pipeline_num_layers, "
            "pipeline_model_parallel_layout) whose stages start on full-indexer layers (for these "
            f"{config.num_layers} layers with an even split: PP {valid_even_pp_sizes(config)})"
        )


def valid_even_pp_sizes(config) -> list:
    """The pipeline sizes that split ``config.num_layers`` evenly (no VPP) with every stage
    starting on a ``full`` indexer layer (for the release's 72 layers: 1, 2, 3, 6, 9, 18)."""
    n = int(config.num_layers)
    return [
        pp
        for pp in range(1, n + 1)
        if n % pp == 0 and all(config.indexer_types[s] != "shared" for s in range(0, n, n // pp))
    ]


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
    if not _runs_sparse(args, cfg):
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
    if bool_flag(getattr(args, "allow_untrained_indexer", False), "allow_untrained_indexer", False):
        from megatron.training import print_rank_0

        print_rank_0(
            "[Primus:Mistral-Large-4] WARNING: allow_untrained_indexer: sparse DSA attention selects keys "
            "with a random, frozen indexer (code-path test; the selection does not train)"
        )
        return
    raise ValueError(
        f"Mistral Large 4: seq_length {args.seq_length} > index_topk {cfg.index_topk} runs DSA sparse "
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


# The provider a family package exports (registry.py): builder.family_model_provider's signature,
# defaulting to this model's builder. builder.py is imported on the call, not at module import:
# it imports megatron.training, which pulls in optional GPU-probing packages (fla, mamba_ssm) that
# fail on a host without a GPU; the converter and CPU tests import this module.
def model_provider(
    model_builder=None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    from primus.backends.megatron.core.models.builder import family_model_provider

    return family_model_provider(mistral_large4_builder)(
        model_builder, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection
    )


model_provider.default_builder = mistral_large4_builder
