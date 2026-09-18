###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Allow non-expert TE CUDA graphs with Primus-Turbo MXFP4 MoE.

Megatron's TE CUDA-graph helper derives its quantization metadata from the
global transformer config.  A model with ``fp4_recipe=mxfp4`` therefore asks
TE for an FP4 recipe even when the graph scope is only attention.  TE does not
own the MXFP4 expert path in this configuration, and its recipe builder rejects
MXFP4 before graph capture starts.

For the narrowly gated Primus-Turbo configuration below, attention and router
work are implemented outside TE while the uncaptured expert path owns MXFP4.
Keep the global FP4 flag intact while Megatron builds sample inputs and graph
runners so Primus-Turbo MXFP4 linears remain enabled. Strip only the TE
quantization kwargs after input preparation, before TE captures the graph.
"""

from functools import wraps
from importlib import import_module

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _enum_value(value):
    return getattr(value, "value", value)


def _scope_values(scope):
    if scope is None:
        return set()
    if isinstance(scope, str):
        scope = [scope]
    return {str(_enum_value(value)).lower() for value in scope}


_NON_EXPERT_GRAPH_SCOPES = {"attn", "moe_router", "moe_preprocess"}
_STATIC_REPLAY_KWARGS = (
    "rotary_pos_emb",
    "rotary_pos_cos",
    "rotary_pos_sin",
    "rotary_pos_cos_sin",
)


def _same_tensor_tree_signature(lhs, rhs):
    """Return whether two tensor trees have the same graph-input signature."""
    try:
        import torch
    except ImportError:
        return False

    if isinstance(lhs, torch.Tensor) or isinstance(rhs, torch.Tensor):
        return (
            isinstance(lhs, torch.Tensor)
            and isinstance(rhs, torch.Tensor)
            and lhs.shape == rhs.shape
            and lhs.dtype == rhs.dtype
            and lhs.device == rhs.device
            and lhs.layout == rhs.layout
        )
    if isinstance(lhs, (tuple, list)) or isinstance(rhs, (tuple, list)):
        return (
            type(lhs) is type(rhs)
            and len(lhs) == len(rhs)
            and all(_same_tensor_tree_signature(a, b) for a, b in zip(lhs, rhs))
        )
    return lhs is None and rhs is None


def _cache_static_replay_kwargs(helper, make_graphed_callables_kwargs):
    """Attach TE's immutable capture inputs to each layer and graph index.

    TE copies every user input into its static graph buffers unless the replay
    tensor already has the capture tensor's data pointer. RoPE tensors are
    immutable for this fixed-sequence training workload, so replaying with the
    exact capture objects safely avoids one device copy per layer. Attention
    masks are intentionally excluded: runtime ``None`` is materialized as a
    zero mask, while the capture sample can contain a causal mask.
    """
    sample_kwargs = make_graphed_callables_kwargs.get("sample_kwargs")
    if not sample_kwargs:
        return

    num_layers_accumulated = 0
    for layers in helper.callables_per_chunk:
        for layer_number, layer in enumerate(layers):
            per_graph_kwargs = []
            for batch_number in range(helper.num_microbatches):
                if helper.config.overlap_moe_expert_parallel_comm:
                    graph_idx = (
                        num_layers_accumulated + layer_number
                    ) * helper.num_microbatches + batch_number
                else:
                    graph_idx = (
                        num_layers_accumulated * helper.num_microbatches
                        + batch_number * len(layers)
                        + layer_number
                    )
                static_kwargs = sample_kwargs[graph_idx]
                per_graph_kwargs.append(
                    {
                        key: static_kwargs[key]
                        for key in _STATIC_REPLAY_KWARGS
                        if key in static_kwargs and static_kwargs[key] is not None
                    }
                )
            layer._primus_te_static_replay_kwargs = per_graph_kwargs
        num_layers_accumulated += len(layers)


def _is_mxfp4_nonexpert_graph(config) -> bool:
    """Return whether an MXFP4 TE graph excludes expert computation."""
    fp4_recipe = str(_enum_value(getattr(config, "fp4_recipe", ""))).lower()
    scopes = _scope_values(getattr(config, "cuda_graph_scope", None))
    return (
        getattr(config, "cuda_graph_impl", "none") == "transformer_engine"
        and bool(scopes)
        and scopes <= _NON_EXPERT_GRAPH_SCOPES
        and bool(getattr(config, "fp4", False))
        and fp4_recipe == "mxfp4"
    )


def _is_turbo_mxfp4_nonexpert_graph(config) -> bool:
    """Return whether TE quantization metadata is irrelevant to this graph."""

    return (
        _is_mxfp4_nonexpert_graph(config)
        and bool(getattr(config, "enable_primus_turbo", False))
        and bool(getattr(config, "use_turbo_attention", False))
        and bool(getattr(config, "use_turbo_gemm", False))
    )


def _can_patch(ctx: PatchContext) -> bool:
    return _is_turbo_mxfp4_nonexpert_graph(get_args(ctx))


@register_patch(
    "megatron.turbo.mxfp4_attention_cudagraph",
    backend="megatron",
    phase="before_train",
    description="Disable irrelevant TE FP4 metadata for Turbo non-expert CUDA graphs",
    condition=_can_patch,
)
def patch_mxfp4_attention_cudagraph(ctx: PatchContext):
    from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
    from megatron.core.transformer.transformer_layer import TransformerLayer

    original_get_input_data = TECudaGraphHelper._get_cuda_graph_input_data
    original_create_cudagraphs = TECudaGraphHelper.create_cudagraphs
    original_get_replay_args = TransformerLayer._get_te_cuda_graph_replay_args

    @wraps(original_get_input_data)
    def get_input_data_without_te_fp4(self):
        # Do not clear config.fp4 while _get_sample_arguments creates graph
        # runners. _CudaGraphRunner snapshots that flag and the Turbo dense
        # projections consult it during warmup/capture; clearing it here would
        # silently graph BF16 projections in an otherwise MXFP4 workload.
        # The helper imports get_fp4_recipe inside its nested kwargs builder.
        # MXFP4 is owned by Primus-Turbo here, so suppress only that TE recipe
        # lookup.  Keeping config.fp4 set is essential: sample argument setup
        # constructs _CudaGraphRunner and snapshots the low-precision mode.
        fp4_utils = import_module("megatron.core.fp4_utils")

        original_get_fp4_recipe = fp4_utils.get_fp4_recipe
        fp4_utils.get_fp4_recipe = lambda _config: None
        try:
            sample_args, kwargs = original_get_input_data(self)
        finally:
            fp4_utils.get_fp4_recipe = original_get_fp4_recipe

        kwargs["fp8_enabled"] = False
        kwargs.pop("fp8_recipe", None)
        kwargs.pop("fp8_weight_caching", None)
        kwargs.pop("fp8_group", None)
        _cache_static_replay_kwargs(self, kwargs)
        return sample_args, kwargs

    TECudaGraphHelper._get_cuda_graph_input_data = get_input_data_without_te_fp4

    @wraps(original_create_cudagraphs)
    def create_cudagraphs_with_turbo_fp4(self):
        # make_graphed_callables(fp8_enabled=False) correctly prevents TE from
        # interpreting MXFP4 as a TE-owned recipe, but its capture warmups then
        # run without Megatron's outer per-layer FP4 context.  Turbo linears
        # consult PrimusTurboLowPrecisionGlobalStateManager, not config.fp4,
        # so that missing context silently sends attention QKV/O projections
        # through their BF16 fallback during capture.  The fallback kernels are
        # then permanently baked into the graphs.
        #
        # Recreate the same Primus FP4 context used by eager TransformerBlock
        # execution around the whole TE capture.  TE may nest a disabled TE
        # autocast internally, but it does not modify Turbo's separate FP4
        # enable flag.  Expert computation remains outside the selected graph
        # scopes and is unaffected.
        from primus.backends.megatron.core.fp4_utils import get_fp4_context

        with get_fp4_context(self.config):
            return original_create_cudagraphs(self)

    TECudaGraphHelper.create_cudagraphs = create_cudagraphs_with_turbo_fp4

    @wraps(original_get_replay_args)
    def get_replay_args_with_static_attention_inputs(self, *args, **kwargs):
        cudagraph_args, cudagraph_kwargs = original_get_replay_args(self, *args, **kwargs)
        static_inputs = getattr(self, "_primus_te_static_replay_kwargs", None)
        if not static_inputs:
            return cudagraph_args, cudagraph_kwargs

        graph_idx = getattr(self, "current_microbatch", 0) % len(static_inputs)
        for key, static_value in static_inputs[graph_idx].items():
            replay_value = cudagraph_kwargs.get(key)
            if _same_tensor_tree_signature(replay_value, static_value):
                cudagraph_kwargs[key] = static_value
        return cudagraph_args, cudagraph_kwargs

    TransformerLayer._get_te_cuda_graph_replay_args = (
        get_replay_args_with_static_attention_inputs
    )
    log_rank_0(
        "[Patch:megatron.turbo.mxfp4_attention_cudagraph] "
        "Disabled TE FP4 metadata, preserved Turbo FP4 capture context, and "
        "reused immutable RoPE inputs for non-expert CUDA graphs"
    )
