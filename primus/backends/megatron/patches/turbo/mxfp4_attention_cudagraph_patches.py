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
Hide the global FP4 flag only while Megatron constructs the graph input
metadata. This lets TE graph any combination of attention, router, and MoE
preprocessing with quantization autocast disabled, then restores MXFP4 before
capture and training continue.
"""

from functools import wraps

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
        # Primus-only Turbo flags are not propagated to TransformerConfig.
        # The patch registration condition already established that this is a
        # Turbo run; only recheck the graph fields available on self.config.
        if not _is_mxfp4_nonexpert_graph(self.config):
            return original_get_input_data(self)

        original_fp4 = self.config.fp4
        self.config.fp4 = None
        try:
            return original_get_input_data(self)
        finally:
            self.config.fp4 = original_fp4

    TECudaGraphHelper._get_cuda_graph_input_data = get_input_data_without_te_fp4
    log_rank_0(
        "[Patch:megatron.turbo.mxfp4_attention_cudagraph] "
        "Disabled TE FP4 metadata for non-expert Turbo CUDA graphs"
    )
