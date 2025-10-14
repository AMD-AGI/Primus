
from types import MethodType
from typing import List, Optional, Tuple, Union
import torch
from megatron.core.models.gpt import GPTModel
from megatron.training import get_args
from pretrain_gpt import model_provider as megatron_model_provider

import megatron.legacy.model  # isort: skip
from primus.backends.megatron.core.extensions.logits_processor import fused_softcap

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
    pre_process=True, post_process=True, vp_stage: Optional[int] = None
) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    # get model
    model = megatron_model_provider(
       pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
    
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