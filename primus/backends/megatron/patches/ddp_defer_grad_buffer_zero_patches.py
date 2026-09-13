###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Skip the full DDP gradient-buffer fill at the start of a step.

With ``defer_grad_buffer_zero``, the contiguous main-grad storage is left
dirty. The backward hook overwrites each parameter slice on its first gradient
and uses ``add_`` only for later contributions. Direct-to-main-grad producers
must set ``grad_added_to_main_grad`` and ``main_grad_initialized`` before the
hook runs.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


@register_patch(
    "megatron.args.defer_grad_buffer_zero",
    backend="megatron",
    phase="setup",
    description="Expose --defer-grad-buffer-zero on Megatron's distributed argparse group.",
)
def patch_defer_grad_buffer_zero_arg(ctx: PatchContext) -> None:
    try:
        import megatron.training.arguments as margs
    except ImportError:
        return

    orig = getattr(margs, "_add_distributed_args", None)
    if orig is None or getattr(orig, "_primus_defer_grad_buffer_zero", False):
        return

    def _add_distributed_args(parser):
        parser = orig(parser)
        group = parser.add_argument_group(title="distributed")
        group.add_argument(
            "--defer-grad-buffer-zero",
            action="store_true",
            default=False,
            help=(
                "Overwrite each main-grad slice on its first gradient instead of "
                "clearing the full contiguous gradient buffer before backward."
            ),
        )
        return parser

    _add_distributed_args._primus_defer_grad_buffer_zero = True
    margs._add_distributed_args = _add_distributed_args
    log_rank_0("[Patch:megatron.args.defer_grad_buffer_zero] added --defer-grad-buffer-zero")


@register_patch(
    "megatron.ddp.defer_grad_buffer_zero",
    backend="megatron",
    phase="before_train",
    description="Overwrite first gradients into main_grad instead of filling the full buffer.",
    condition=lambda ctx: bool(getattr(get_args(ctx), "defer_grad_buffer_zero", False)),
)
def patch_defer_grad_buffer_zero(ctx: PatchContext) -> None:
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel as DDP,
    )
    from megatron.core.transformer.cuda_graphs import is_graph_capturing

    orig_init = DDP.__init__
    orig_make_hook = DDP._make_backward_post_hook
    orig_zero = DDP.zero_grad_buffer

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.ddp_config.defer_grad_buffer_zero = True
        config = getattr(self, "config", None)
        if getattr(config, "cuda_graph_impl", "none") == "transformer_engine":
            raise ValueError(
                "defer_grad_buffer_zero is incompatible with Transformer Engine CUDA graphs: "
                "their replay path does not reset per-parameter first-gradient state"
            )

    def _make_backward_post_hook(self, param):
        orig_hook = orig_make_hook(self, param)

        def hook(*unused):
            if is_graph_capturing():
                return orig_hook(*unused)
            if (
                param in self.param_to_bucket_group
                and param.grad is not None
                and (
                    not param.grad_added_to_main_grad
                    or getattr(param, "zero_out_wgrad", False)
                )
                and not getattr(param, "main_grad_initialized", False)
            ):
                param.main_grad.copy_(param.grad.data)
                param.main_grad_initialized = True
                param.grad = None
                if self.ddp_config.overlap_grad_reduce:
                    self.param_to_bucket_group[param].register_grad_ready(
                        param, self.force_all_reduce
                    )
                return
            return orig_hook(*unused)

        return hook

    def zero_grad_buffer(self):
        if getattr(self.config, "cuda_graph_impl", "none") != "transformer_engine":
            for param in self.params_with_grad:
                param.grad_added_to_main_grad = False
                param.main_grad_initialized = False
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.reset()

    DDP.__init__ = __init__
    DDP._make_backward_post_hook = _make_backward_post_hook
    DDP.zero_grad_buffer = zero_grad_buffer
    log_rank_0(
        "[Patch:megatron.ddp.defer_grad_buffer_zero] first-gradient copy into main_grad; "
        "full buffer reset skipped"
    )
