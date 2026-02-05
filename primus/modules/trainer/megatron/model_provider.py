###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from types import MethodType
from typing import Callable, Optional, Union

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.models.mamba import MambaModel
from megatron.training import get_args

from primus.backends.megatron.core.extensions.logits_processor import fused_softcap

import megatron.legacy.model  # isort: skip

g_final_logit_softcapping: Optional[float] = None
original_compute_language_model_loss: Optional[MethodType] = None


def wrapped_compute_language_model_loss(self, labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    global g_final_logit_softcapping
    assert g_final_logit_softcapping is not None

    logits = logits.float()
    fused_softcap(logits, g_final_logit_softcapping)

    global original_compute_language_model_loss
    return original_compute_language_model_loss(labels, logits)


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
