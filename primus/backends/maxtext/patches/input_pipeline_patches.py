###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText Input Pipeline Patches

Replaces MaxText's HuggingFace data-processing helpers with Primus
implementations that add additional preprocessing, custom tokenisation,
and robust iterator support.
"""

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0


@register_patch(
    patch_id="maxtext.input_pipeline",
    backend="maxtext",
    phase="setup",
    description="Replace MaxText HF data-processing functions with Primus implementations (Dec version)",
    condition=lambda ctx: True,  # Always enabled
    backend_versions=["0.1.1"],
)
def patch_input_pipeline(ctx: PatchContext) -> None:
    """
    Monkey-patch HuggingFace data-processing functions in MaxText's
    ``input_pipeline`` with Primus versions.
    """
    log_rank_0("[Patch:maxtext.input_pipeline] Patching HF data processing...")

    import MaxText.input_pipeline._hf_data_processing as orig_hf_data_processing
    import MaxText.input_pipeline.input_pipeline_interface as orig_input_pipeline_interface

    from primus.backends.maxtext.input_pipeline._hf_data_processing import (
        make_hf_eval_iterator,
        make_hf_train_iterator,
        preprocessing_pipeline,
    )

    orig_hf_data_processing.preprocessing_pipeline = preprocessing_pipeline
    orig_hf_data_processing.make_hf_train_iterator = make_hf_train_iterator
    orig_hf_data_processing.make_hf_eval_iterator = make_hf_eval_iterator

    orig_input_pipeline_interface.make_hf_train_iterator = make_hf_train_iterator
    orig_input_pipeline_interface.make_hf_eval_iterator = make_hf_eval_iterator

    warning_rank_0("[Patch:maxtext.input_pipeline] HF data processing patched successfully.")


# ---------------------------------------------------------------------------
# Legacy patch for Aug-version MaxText (2025.*)
# ---------------------------------------------------------------------------


@register_patch(
    patch_id="maxtext.input_pipeline.legacy",
    backend="maxtext",
    phase="setup",
    description="Replace MaxText HF data-processing functions with Primus legacy implementations (Aug version)",
    condition=lambda ctx: True,  # Always enabled
    backend_versions=["2025.*"],
)
def patch_input_pipeline_legacy(ctx: PatchContext) -> None:
    """
    Monkey-patch HuggingFace data-processing functions in MaxText's
    ``input_pipeline`` with legacy Primus versions (uses ``CustomPackAndBatchOperation``,
    ``max_segments``, ``PadToMaxLength``, etc.).
    """
    log_rank_0("[Patch:maxtext.input_pipeline.legacy] Patching HF data processing (legacy)...")

    import MaxText.input_pipeline._hf_data_processing as orig_hf_data_processing
    import MaxText.input_pipeline.input_pipeline_interface as orig_input_pipeline_interface

    from primus.backends.maxtext.legacy.input_pipeline._hf_data_processing import (
        make_hf_eval_iterator,
        make_hf_train_iterator,
        preprocessing_pipeline,
    )

    orig_hf_data_processing.preprocessing_pipeline = preprocessing_pipeline
    orig_hf_data_processing.make_hf_train_iterator = make_hf_train_iterator
    orig_hf_data_processing.make_hf_eval_iterator = make_hf_eval_iterator

    orig_input_pipeline_interface.make_hf_train_iterator = make_hf_train_iterator
    orig_input_pipeline_interface.make_hf_eval_iterator = make_hf_eval_iterator

    warning_rank_0("[Patch:maxtext.input_pipeline.legacy] HF data processing patched successfully.")
