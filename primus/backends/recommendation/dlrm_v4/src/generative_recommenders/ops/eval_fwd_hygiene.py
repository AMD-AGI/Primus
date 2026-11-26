# pyre-strict
"""DLRM_HSTU_FP8_EVAL_FWD_HYGIENE: the eval forward stops paying for training.

The eval pass runs ``model.eval()`` under ``torch.no_grad()``
(dlrm_v4/train/utils.py ``_run_eval_window``). Its forward still did work
whose only consumer is a backward or the training telemetry, and it declined
two forward fast paths that were gated on grad mode rather than on what they
compute. With this switch on:

  * ops/position.py: the POSENC_FAST alpha fold and the POSENC_CONCAT_FOLD
    contextual-concat fold (yambda_5b.gin items (13), (20)(c)) also serve
    no-grad forwards. Both kernels compute bitwise the values of the unfolded
    concat_2D_jagged + ``* alpha`` + position-kernel sequence they replace.
  * ops/triton/triton_position.py: the index sorts (and the arange, where it
    runs) that only the table-gradient reduction of the backward reads are
    skipped when no backward can run.
  * ops/triton/triton_hstu_linear.py: the FP8 output-projection wgrad no
    longer saves its (1,) dequant-scale tensor when no backward can run.
  * modules/dlrm_hstu.py: eval batches no longer stash the per-batch FLOPs
    (lazy or eager; read only by the train perf line).

Sub-switch DLRM_HSTU_FP8_EVAL_FWD_HYGIENE_LOSS (acts only with the main
switch on): modules/multitask_module.py: eval batches also skip the multitask
loss (DlrmHSTU.main_forward reads it only to build the training aux losses;
in eval it is discarded). Predictions, labels and weights, which the eval
metrics read, are unchanged either way. Kept separate because it removes a
computation the eval pass ran (not only backward/telemetry bookkeeping).

Whether a backward can run is decided OUTSIDE each torch.autograd.Function
(``backward_possible``): inside ``Function.forward`` grad mode is always off,
and ``ctx.needs_input_grad`` is True for parameters even under no_grad.
Training forwards (grad enabled, parameters require grad) take exactly the
code they took before and do not read either switch.

Read per call; armed only by the value "1"; absent or "0" = off = the code
before this item. Not FP8 math; the prefix is what launch_slurm.sh forwards.
"""

import os
from typing import List, Optional

import torch

EVAL_FWD_HYGIENE_ENV: str = "DLRM_HSTU_FP8_EVAL_FWD_HYGIENE"
EVAL_FWD_HYGIENE_LOSS_ENV: str = "DLRM_HSTU_FP8_EVAL_FWD_HYGIENE_LOSS"
_ARMED_LOGGED: List[bool] = [False]
_LOSS_LOGGED: List[bool] = [False, False]  # armed line, ignored line


@torch.jit.ignore
def eval_fwd_hygiene_enabled() -> bool:
    """$DLRM_HSTU_FP8_EVAL_FWD_HYGIENE == "1" (read per call; logs one arming line).

    Callers consult it only on forwards that cannot run a backward (or, for the
    module-level items, only in eval mode), so the line marks the first such
    forward of the process.
    """
    on = os.environ.get(EVAL_FWD_HYGIENE_ENV, "0") == "1"
    if on and not _ARMED_LOGGED[0]:
        _ARMED_LOGGED[0] = True
        print(
            "[eval-fwd-hygiene] DLRM_HSTU_FP8_EVAL_FWD_HYGIENE=1 armed: no-grad "
            "forwards take the posenc alpha/concat folds; posenc index sorts, "
            "FP8 out-wgrad scale save and FLOPs stash skipped when no "
            "backward/training can read them (bit-identical outputs); eval "
            "multitask loss: "
            + (
                "skipped (DLRM_HSTU_FP8_EVAL_FWD_HYGIENE_LOSS=1)"
                if os.environ.get(EVAL_FWD_HYGIENE_LOSS_ENV, "0") == "1"
                else "computed (DLRM_HSTU_FP8_EVAL_FWD_HYGIENE_LOSS off)"
            ),
            flush=True,
        )
    return on


@torch.jit.ignore
def eval_fwd_hygiene_loss_enabled() -> bool:
    """$DLRM_HSTU_FP8_EVAL_FWD_HYGIENE_LOSS == "1" and the main switch on (read
    per call). Called only by eval-mode multitask forwards; the sub-switch is
    read first, so with it off the main switch is not read here.
    """
    if os.environ.get(EVAL_FWD_HYGIENE_LOSS_ENV, "0") != "1":
        return False
    on = eval_fwd_hygiene_enabled()
    if on and not _LOSS_LOGGED[0]:
        _LOSS_LOGGED[0] = True
        print(
            "[eval-fwd-hygiene] DLRM_HSTU_FP8_EVAL_FWD_HYGIENE_LOSS=1 armed: "
            "eval forwards skip the multitask loss (read only as the training "
            "aux loss; predictions, labels and weights unchanged)",
            flush=True,
        )
    elif not on and not _LOSS_LOGGED[1]:
        _LOSS_LOGGED[1] = True
        print(
            "[eval-fwd-hygiene] DLRM_HSTU_FP8_EVAL_FWD_HYGIENE_LOSS=1 ignored: "
            "needs DLRM_HSTU_FP8_EVAL_FWD_HYGIENE=1 (the eval multitask loss "
            "is computed)",
            flush=True,
        )
    return on


def backward_possible(*tensors: Optional[torch.Tensor]) -> bool:
    """Whether autograd records a graph through an op on these inputs.

    The same rule torch.autograd.Function.apply uses (grad mode on and any
    input requires grad). Must be called by the wrapper, before ``apply``:
    inside ``forward`` the grad mode is always off. Called only from
    @torch.jit.unused wrappers.
    """
    if not torch.is_grad_enabled():
        return False
    for t in tensors:
        if t is not None and t.requires_grad:
            return True
    return False
