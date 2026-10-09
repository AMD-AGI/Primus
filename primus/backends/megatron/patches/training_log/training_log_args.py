###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Bind the arguments of a ``megatron.training.training.training_log`` call by name.

``train()`` calls ``training_log`` mostly positionally, and the Primus wrappers around it take ``*args, **kwargs``,
so a wrapper that needs one argument (``iteration``, ``grad_norm``, ...) binds the call against this signature.
"""

import inspect


def _training_log_signature(
    loss_dict,
    total_loss_dict,
    learning_rate,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    grad_norm,
    params_norm,
    num_zeros_in_grad,
    max_attention_logit,
    pg_collection=None,
    is_first_iteration=False,
):
    """Megatron's ``training_log`` signature."""


TRAINING_LOG_SIGNATURE = inspect.signature(_training_log_signature)


def bind_training_log_args(args, kwargs):
    """``inspect.BoundArguments`` for a training_log call, or None if the call does not fit the signature."""
    try:
        bound = TRAINING_LOG_SIGNATURE.bind(*args, **kwargs)
    except TypeError:
        return None
    if not isinstance(bound.arguments.get("iteration"), int):
        return None
    return bound
