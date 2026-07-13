###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DDP overlap_param_gather + torch.compile compatibility patches.

When using per_block torch.compile with Megatron DDP's overlap_param_gather,
the original per-sub-module hook registration causes Dynamo to trace through
hooks containing NCCL side effects, leading to assertion errors.

This patch uses two-tier hook registration:
  - Layer hooks (recurse=True): one hook per transformer layer (children of
    ModuleList). These fire in the eager __call__ wrapper before the compiled
    forward, causing zero graph breaks inside compiled layers.
  - Non-layer hooks (recurse=False): hooks on all modules outside compiled
    regions (embeddings, projections, norms). These preserve original Megatron
    behavior and run entirely in eager mode.
"""

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


@register_patch(
    "megatron.ddp.overlap_param_gather_compile",
    backend="megatron",
    phase="before_train",
    description=(
        "Patch DDP forward pre-hooks for overlap_param_gather + torch.compile "
        "compatibility using two-tier hook registration."
    ),
    # Distinct from the FSDP2 fp8-cache patch (priority=45) to avoid a
    # registration-order-dependent tie-break. The two are mutually exclusive
    # (DDP vs FSDP2), so the exact relative order is not load-bearing.
    priority=44,
    condition=lambda ctx: (
        getattr(get_args(ctx), "use_distributed_optimizer", False)
        and getattr(get_args(ctx), "overlap_param_gather", False)
        and getattr(getattr(get_args(ctx), "torch_compile", None), "enable", False)
        and not getattr(get_args(ctx), "disable_ddp_compile_patches", False)
    ),
)
def patch_ddp_overlap_param_gather_for_compile(ctx: PatchContext) -> None:
    """Patch DistributedDataParallel to use two-tier hook registration.

    Monkey-patches class methods on DDP so that both the initial hook
    registration in __init__ and subsequent enable/disable cycles in the
    training loop use the patched logic.
    """
    try:
        from megatron.core.distributed.distributed_data_parallel import (
            DistributedDataParallel as DDP,
        )
        from megatron.core.transformer.cuda_graphs import is_graph_capturing

        def _get_overlap_hook_modules(self):
            """Partition modules into layer-tier and non-layer-tier for hooks.

            Layer modules: children of nn.ModuleList containers (transformer
            layers). These get hooks with recurse=True.

            Non-layer modules: everything else (embeddings, projections, norms).
            These get hooks with recurse=False (original Megatron behavior).
            Modules that are descendants of a layer module are excluded (they
            are covered by the layer hook's recurse=True).
            """
            # Preserve module-traversal order (deterministic) while de-duplicating;
            # iterating a plain set would make hook-registration order vary run to
            # run, which hurts reproducibility/debuggability.
            layer_modules = []
            seen = set()
            for module in self.module.modules():
                if isinstance(module, torch.nn.ModuleList):
                    for child in module.children():
                        if child not in seen:
                            seen.add(child)
                            layer_modules.append(child)

            inside_layer = set()
            for layer in layer_modules:
                for sub in layer.modules():
                    inside_layer.add(sub)

            non_layer_modules = [m for m in self.module.modules() if m not in inside_layer]

            return layer_modules, non_layer_modules

        def _make_forward_pre_hook(self, recurse=False):
            """Create a forward pre-hook parameterized by recurse depth."""

            def hook(module, *unused):
                if not self.use_forward_hook:
                    raise RuntimeError("Should use pre-hook only when overlap_param_gather is True")

                if is_graph_capturing():
                    return

                for param in module.parameters(recurse=recurse):
                    if param not in self.param_to_bucket_group:
                        continue
                    if not param.requires_grad:
                        raise RuntimeError("Bucketed param in forward pre-hook must require grad")

                    skip_next_bucket_dispatch = (
                        self.ddp_config.align_param_gather or self.overlap_param_gather_with_optimizer_step
                    )
                    self.param_to_bucket_group[param].finish_param_sync(
                        skip_next_bucket_dispatch=skip_next_bucket_dispatch
                    )

            return hook

        def enable_forward_pre_hook(self):
            """Register two-tier forward pre-hooks for param all-gather overlap."""
            if not self.use_forward_hook:
                raise RuntimeError("enable_forward_pre_hook requires use_forward_hook=True")
            if len(self.remove_forward_pre_hook_handles) != 0:
                raise RuntimeError("Forward pre-hooks already registered")

            layer_modules, non_layer_modules = self._get_overlap_hook_modules()

            layer_hook = self._make_forward_pre_hook(recurse=True)
            for module in layer_modules:
                self.remove_forward_pre_hook_handles[module] = module.register_forward_pre_hook(layer_hook)

            non_layer_hook = self._make_forward_pre_hook(recurse=False)
            for module in non_layer_modules:
                self.remove_forward_pre_hook_handles[module] = module.register_forward_pre_hook(
                    non_layer_hook
                )

        def disable_forward_pre_hook(self, param_sync: bool = True):
            """Remove all forward pre-hooks (both tiers)."""
            if not self.use_forward_hook:
                raise RuntimeError("disable_forward_pre_hook requires use_forward_hook=True")
            for module, handle in list(self.remove_forward_pre_hook_handles.items()):
                handle.remove()
            self.remove_forward_pre_hook_handles.clear()

            if param_sync:
                self.start_param_sync(force_sync=True)

        DDP._get_overlap_hook_modules = _get_overlap_hook_modules
        DDP._make_forward_pre_hook = _make_forward_pre_hook
        DDP.enable_forward_pre_hook = enable_forward_pre_hook
        DDP.disable_forward_pre_hook = disable_forward_pre_hook

        log_rank_0(
            "[Patch:megatron.ddp.overlap_param_gather_compile] "
            "Patched DDP with two-tier hook registration for "
            "overlap_param_gather + torch.compile compatibility"
        )

    except Exception as e:
        import traceback

        log_rank_0(
            f"[Patch:megatron.ddp.overlap_param_gather_compile] "
            f"ERROR: Failed to patch DDP: {type(e).__name__}: {e}"
        )
        log_rank_0(f"Traceback: {traceback.format_exc()}")
        # Re-raise: silently leaving DDP unpatched here would run training with
        # incorrect forward pre-hooks (Dynamo graph breaks / NCCL side effects),
        # so fail loudly instead of degrading correctness.
        raise RuntimeError(
            "Failed to apply DDP overlap_param_gather + torch.compile patch; "
            "see log above for the underlying error."
        ) from e
