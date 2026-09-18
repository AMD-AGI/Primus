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

    original_get_input_data = TECudaGraphHelper._get_cuda_graph_input_data

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
        return sample_args, kwargs

    TECudaGraphHelper._get_cuda_graph_input_data = get_input_data_without_te_fp4
    log_rank_0(
        "[Patch:megatron.turbo.mxfp4_attention_cudagraph] "
        "Disabled TE FP4 metadata for non-expert Turbo CUDA graphs"
    )
