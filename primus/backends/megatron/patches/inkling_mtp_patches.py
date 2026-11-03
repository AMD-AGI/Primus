###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling MTP: keep the MTP stage's copy of the trunk ``embed_norm`` in sync.

The MTP depths embed the rolled tokens with the shared word embedding and the trunk's
``embed_norm``. With pipeline parallelism the MTP stage (the last) holds copies of both. Megatron
all-reduces the word-embedding gradient between the first stage and the MTP stage
(``_allreduce_word_embedding_grads``); this patch wraps ``finalize_model_grads`` so that the
``embed_norm`` gradient is all-reduced the same way, right after it. Both copies start equal (ones,
or the same checkpoint key) and get the same update.
"""

import functools

from primus.core.patches import PatchContext, get_args, register_patch

_PATCH_KEY = "megatron.inkling.mtp_embed_norm_grads"


def _wrap(finalize_model_grads):
    from megatron.core.utils import get_model_config

    from primus.backends.megatron.core.models.inkling.inkling_model import (
        allreduce_mtp_embed_norm_grads,
    )

    @functools.wraps(finalize_model_grads)
    def wrapped(model, *args, **kwargs):
        # after the original, as Megatron's word-embedding all-reduce runs after the DP grad sync
        # (an async DP reduce may still own the buffer before it). Both stages were scaled by the
        # same 1 / num_tokens, so the sum is unchanged; the optimizer clips / steps afterwards.
        out = finalize_model_grads(model, *args, **kwargs)
        allreduce_mtp_embed_norm_grads(model, get_model_config(model[0]))
        return out

    wrapped._inkling_mtp_wrapper = True
    return wrapped


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    priority=102,  # after the zero-bubble patch that replaces finalize_model_grads
    description="Inkling MTP: all-reduce the trunk embed_norm gradient with its MTP-stage copy.",
    condition=lambda ctx: getattr(get_args(ctx), "model_type", None) == "inkling"
    and bool(getattr(get_args(ctx), "mtp_num_layers", None))
    and int(getattr(get_args(ctx), "pipeline_model_parallel_size", 1) or 1) > 1,
)
def patch_inkling_mtp_embed_norm_grads(ctx: PatchContext):
    import importlib

    import megatron.core.distributed as dist_module
    import megatron.training.training as training_module

    # Not ``import megatron.core.distributed.finalize_model_grads as m``: the package re-exports
    # the function under the submodule's name, so that binds the function, not the module.
    fmg_module = importlib.import_module("megatron.core.distributed.finalize_model_grads")

    for module in (training_module, dist_module, fmg_module):
        current = getattr(module, "finalize_model_grads", None)
        if current is not None and not getattr(current, "_inkling_mtp_wrapper", False):
            module.finalize_model_grads = _wrap(current)
