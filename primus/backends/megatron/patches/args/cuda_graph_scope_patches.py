###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_kv_rank_0


@register_patch(
    "megatron.args.cuda_graph_scope",
    backend="megatron",
    phase="build_args",
    description="Convert cuda_graph_scope string entries to CudaGraphScope enums",
)
def patch_cuda_graph_scope(ctx: PatchContext):
    """
    Convert cuda_graph_scope from a list of strings to CudaGraphScope enums.

    Entries that are not ``"full"`` are mapped via ``CudaGraphScope[scope]``;
    ``"full"`` is kept as-is so Megatron's own validation can handle it.
    Defaults to ``[]`` when the attribute is missing or ``None``.
    """
    from megatron.core.transformer.enums import CudaGraphScope

    
    args = ctx.extra.get("backend_args", {})
    if not args:
        return

    value = getattr(args, "cuda_graph_scope", None)
    if value is None:
        args.cuda_graph_scope = []
        log_kv_rank_0(
            "[Patch:megatron.args.cuda_graph_scope] -cuda_graph_scope",
            f"{args.cuda_graph_scope}",
        )
        return

    if isinstance(value, list):
        args.cuda_graph_scope = [
            CudaGraphScope[scope] if scope != "full" else scope for scope in value
        ]
    elif isinstance(value, str) and value != "full":
        args.cuda_graph_scope = [CudaGraphScope[value]]
    # "full" string is left as-is for Megatron's own deprecation handling

    log_kv_rank_0(
        "[Patch:megatron.args.cuda_graph_scope] -cuda_graph_scope",
        f"{args.cuda_graph_scope}",
    )
