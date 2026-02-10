###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron-FSDP Patches

This module contains patches for Megatron's native FSDP implementation:
- DeviceMesh API compatibility for PyTorch 2.10+

These patches are separate from PyTorch FSDP2 (torch.distributed.fsdp),
which is Megatron's own FSDP implementation.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0


@register_patch(
    "megatron.fsdp.device_mesh",
    backend="megatron",
    phase="before_train",
    description=(
        "Fix Megatron-FSDP DeviceMesh API compatibility for PyTorch 2.10+ "
        "by updating get_mesh_names to use new API."
    ),
    condition=lambda ctx: getattr(get_args(ctx), "use_megatron_fsdp", False),
)
def patch_megatron_fsdp_device_mesh(ctx: PatchContext):
    """
    Patch Megatron FSDP's get_mesh_names to work with PyTorch 2.10+ DeviceMesh API.

    Issue:
        Megatron-LM's megatron_fsdp/utils.py uses the outdated
        `_mesh_resources.child_to_root_mapping` attribute which was removed
        in PyTorch 2.10. The new API uses `_root_mesh` and `_flatten_mapping`.

    Solution:
        Replace get_mesh_names with a patched version that uses the new PyTorch 2.10+
        DeviceMesh API methods.
    """
    log_rank_0("[Patch:megatron.fsdp.device_mesh] Patching Megatron FSDP DeviceMesh " "API compatibility...")

    try:
        from megatron.core.distributed.fsdp.src.megatron_fsdp import utils

        # Check if the module has the function we need to patch
        if not hasattr(utils, "get_mesh_names"):
            warning_rank_0("[Patch:megatron.fsdp.device_mesh] get_mesh_names not found, skipping patch")
            return

        # Define the patched function
        def patched_get_mesh_names(device_mesh):
            """
            Get dimension names from a DeviceMesh, including submesh names.

            This is the patched version that works with PyTorch 2.10+.
            """
            # Get the root mesh using the new API
            root_mesh = device_mesh._get_root_mesh()

            # Start with the device_mesh's own dimension names
            result = list(device_mesh.mesh_dim_names or [])

            # Collect submesh dimension names from flattened mapping
            # In PyTorch 2.10+, submeshes are tracked via _flatten_mapping
            if hasattr(root_mesh, "_flatten_mapping") and root_mesh._flatten_mapping:
                for mesh_dim, submesh in root_mesh._flatten_mapping.items():
                    # Check if this submesh matches our device_mesh
                    if submesh == device_mesh or (
                        hasattr(submesh, "_dim_group_names") and submesh._dim_group_names
                    ):
                        # Add submesh dimension names
                        if hasattr(submesh, "mesh_dim_names") and submesh.mesh_dim_names:
                            result.extend(submesh.mesh_dim_names)

            return result

        # Apply the monkey patch
        utils.get_mesh_names = patched_get_mesh_names

        log_rank_0(
            "[Patch:megatron.fsdp.device_mesh] Megatron FSDP DeviceMesh patch "
            "applied successfully (PyTorch 2.10+ compatibility)"
        )

    except ImportError as e:
        warning_rank_0(
            f"[Patch:megatron.fsdp.device_mesh] Megatron FSDP not available, " f"skipping patch: {e}"
        )
    except Exception as e:
        warning_rank_0(f"[Patch:megatron.fsdp.device_mesh] Failed to patch Megatron FSDP " f"DeviceMesh: {e}")
