###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial
from types import MethodType
from typing import Callable, Optional, Union

import megatron.legacy.model
import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.models.mamba import MambaModel
from megatron.training import get_args

from primus.backends.megatron.patches.core.extensions.logits_processor import (
    fused_softcap,
)

from primus.core.utils.import_utils import lazy_import  # isort: skip

g_final_logit_softcapping: Optional[float] = None
original_compute_language_model_loss: Optional[MethodType] = None


def wrapped_compute_language_model_loss(self, labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    global g_final_logit_softcapping
    assert g_final_logit_softcapping is not None

    logits = logits.float()
    fused_softcap(logits, g_final_logit_softcapping)

    global original_compute_language_model_loss
    return original_compute_language_model_loss(labels, logits)


# def primus_gpt_builder(args, pre_process, post_process, vp_stage=None, config=None):
def primus_model_provider(
    model_provider: Callable, pre_process=True, post_process=True, vp_stage: Optional[int] = None
) -> Union[GPTModel, megatron.legacy.model.GPTModel, MambaModel]:
    # get model
    model = model_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

    args = get_args()
    if args.final_logit_softcapping is not None and args.final_logit_softcapping > 0.0:

        global g_final_logit_softcapping
        g_final_logit_softcapping = args.final_logit_softcapping

        # save original func
        global original_compute_language_model_loss
        original_compute_language_model_loss = model.compute_language_model_loss

        # wrap with logits softcapping
        model.compute_language_model_loss = MethodType(wrapped_compute_language_model_loss, model)

    return model


def get_model_provider():
    """
    Resolve model_provider across Megatron versions.

    - New:   model_provider + gpt_builder
    - Mid:   model_provider only
    - Old:   pretrain_gpt.model_provider
    """
    # Try to import model_provider
    model_provider = lazy_import(
        ["model_provider", "pretrain_gpt"], "model_provider", log_prefix="[Primus][MegatronCompat]"
    )

    # Try to import gpt_builder (only exists in newer versions)
    try:
        gpt_builder = lazy_import(["gpt_builders"], "gpt_builder", log_prefix="[Primus][MegatronCompat]")
        return partial(model_provider, gpt_builder)
    except ImportError:
        return model_provider


# def get_model_provider(args):
#     """
#     Get the appropriate model provider function for Megatron.

#     This function determines which model architecture to use based on args
#     and returns a provider function that Megatron's pretrain() expects.

#     Args:
#         args: Megatron argument namespace

#     Returns:
#         Callable: Model provider function
#     """
#     from megatron.training.arguments import core_transformer_config_from_args

#     def model_provider(pre_process=True, post_process=True, vp_stage=None):
#         """Build the model."""
#         config = core_transformer_config_from_args(args)

#         # Determine model type
#         if hasattr(args, "model_type") and args.model_type == "mamba":
#             from megatron.core.models.mamba import MambaModel

#             model = MambaModel(config=config, pre_process=pre_process, post_process=post_process)
#         else:
#             # Default to GPT model
#             from megatron.core.models.gpt import GPTModel

#             model = GPTModel(
#                 config=config,
#                 num_tokentypes=0,
#                 parallel_output=True,
#                 pre_process=pre_process,
#                 post_process=post_process,
#             )

#         return model

#     return model_provider
