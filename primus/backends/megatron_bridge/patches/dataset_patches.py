###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron-Bridge Dataset Patches

This module patches Megatron-Bridge's default dataset configurations to use
updated dataset paths compatible with HuggingFace's namespace reorganization.
"""

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.bridge.recipes.utils.finetune_utils",
    backend="megatron_bridge",
    phase="setup",
    description="Patch default_squad_config to use rajpurkar/squad (HF namespace migration)",
    condition=lambda ctx: True,  # Always apply this patch
)
def patch_default_squad_config(ctx: PatchContext):
    """
    Patch default_squad_config to use 'rajpurkar/squad' instead of 'squad'.

    Background:
        HuggingFace reorganized dataset namespaces in Jan 2026, enforcing
        'owner/dataset' format. The old 'squad' path now returns 404.

    Changes:
        - Line 84: dataset_name="squad" → dataset_name="rajpurkar/squad"

    Impact:
        - Fixes 404 errors when using default squad dataset
        - Enables finetuning recipes to work out-of-the-box
        - No breaking changes (squad dataset still used, just updated path)
    """
    log_rank_0("[Megatron-Bridge Patch] Updating default_squad_config to use 'rajpurkar/squad'...")

    import megatron.bridge.recipes.utils.finetune_utils as finetune_utils
    from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
    from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
    from megatron.bridge.data.hf_processors import process_squad_example

    # Save original function for reference
    finetune_utils.default_squad_config

    def patched_default_squad_config(seq_length: int, packed_sequence: bool = False) -> HFDatasetConfig:
        """
        Patched version of default_squad_config using rajpurkar/squad.

        This is identical to the original except for the dataset_name.
        """
        if packed_sequence:
            # Packed sequence configuration
            dataset_kwargs = {"pad_to_max_length": True}
            packed_sequence_specs = PackedSequenceSpecs(packed_sequence_size=seq_length)
        else:
            # Standard configuration
            dataset_kwargs = {}
            packed_sequence_specs = None

        # Use 'batch' sampler for variable-length finetuning
        dataloader_type = "batch"

        return HFDatasetConfig(
            dataset_name="rajpurkar/squad",  # ✅ PATCHED: Updated from "squad"
            process_example_fn=process_squad_example,
            seq_length=seq_length,
            seed=5678,  # Different from pretrain seed
            dataloader_type=dataloader_type,
            num_workers=1,
            do_validation=True,
            do_test=False,
            val_proportion=0.1,
            dataset_kwargs=dataset_kwargs,
            packed_sequence_specs=packed_sequence_specs,
            rewrite=False,
        )

    # Apply the patch
    finetune_utils.default_squad_config = patched_default_squad_config

    log_rank_0("[Megatron-Bridge Patch] ✅ default_squad_config patched successfully")
