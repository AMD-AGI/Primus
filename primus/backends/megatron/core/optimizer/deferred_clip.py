###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Hand the global-norm clip coefficient from gradient clipping to the Adam step that reads the gradients.

Megatron clips by scaling every gradient in place with the clip coefficient, one read and one write
of each, and the Adam step reads them again right after. An optimizer that marks its parameters with
``mark_folding`` takes the coefficient instead: the clip leaves it on the parameters
(``defer``) and the optimizer applies it as it reads each gradient, to the value the in-place scaling
would have left (fp32 product, rounded to the gradient's dtype). Any step that does not fold it
calls ``apply_eagerly`` first, which does that scaling, so the result never depends on the path.
"""

import torch

_FOLDS = "_primus_adam_folds_clip"
_PENDING = "_primus_deferred_clip_coeff"

# set by the optimizer patch once folding is safe for the run (e.g. no zero-count of the clipped grads)
ENABLED = False


def mark_folding(params) -> None:
    """``params`` belong to an optimizer whose step applies a deferred clip coefficient."""
    for p in params:
        setattr(p, _FOLDS, True)


def defer(params, coeff: torch.Tensor) -> bool:
    """Leave ``coeff`` (one-element fp32, on the device) on ``params`` for their optimizer step, if all
    of them belong to a folding optimizer; False (nothing done) otherwise."""
    params = list(params)
    if not ENABLED or not params or not all(getattr(p, _FOLDS, False) for p in params):
        return False
    for p in params:
        setattr(p, _PENDING, coeff)
    return True


def take(param):
    """The coefficient deferred to ``param`` (cleared), or None."""
    return param.__dict__.pop(_PENDING, None)


def apply_eagerly(params, use_decoupled_grad: bool) -> None:
    """Scale the gradients of ``params`` by their pending coefficients in place, as clipping would have."""
    for p in params:
        coeff = take(p)
        if coeff is None:
            continue
        grad = getattr(p, "decoupled_grad", None) if use_decoupled_grad else p.grad
        if grad is not None:
            grad.mul_(coeff)
