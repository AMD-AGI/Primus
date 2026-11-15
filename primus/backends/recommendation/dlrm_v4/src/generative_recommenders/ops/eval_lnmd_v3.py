# pyre-strict
"""DLRM_HSTU_FP8_EVAL_LNMD_V3: the eval LN*U*dropout forward on the v3 kernel.

DECLARED NUMERICS CHANGE (eval only), default OFF; yambda_5b.gin binds it OFF.

The training LN*U*dropout forward moved to the persistent one-wave-per-row
kernels (yambda_5b.gin items (20)(b) v2, (28)(c) v3, (28)(e) maskgen), but
those launches sit in the training branch of
ops/triton/triton_hstu_linear.py ``_triton_layer_norm_mul_dropout_fwd_impl``
only. A call with ``training=False`` (the eval pass: ``model.eval()`` under
``torch.no_grad()``, dlrm_v4/train/utils.py ``_run_eval_window``) still runs
the legacy per-row ``_ln_mul_dropout_fwd`` (grid N, two waves per row).

With this switch on, a ``training=False`` call that meets the eligibility the
training forward requires for v3 (separated-RNG GPU, not
FUSE_OUTPUT_LN_RNG_BLACKWELL, item (20)(b)'s ``_use_lnmd_fwd_v2`` condition:
ROCm, DLRM_HSTU_FP8_LNMD_FWD_V2 on, N >= 2^20 rows, a sigmoid activation; a
full-width block ``BLOCK_D == D``; item (28)(c)'s ``_use_lnmd_fwd_v3``)
launches ``_ln_mul_dropout_fwd_rng_v3`` with the training branch's arguments
and ``TRAINING=False``: the training forward's per-row arithmetic with the
dropout step removed (no mask is read; the empty mask placeholder is passed
as the training branch passes its mask). Every other call keeps the legacy
kernel (in yambda-5b: the targets-only layer, N = batch rows, no sigmoid).

NOT bit-identical to the legacy eval kernel: one-wave vs two-wave row
reductions give last-bit mean/rstd differences, and a small fraction of the
e4m3 y bytes move by one code. A precision change of the eval pass; training
forwards pass ``training=True`` (output dropout 0.1) and never read the switch.
A config with output dropout 0 runs its TRAINING forward with
``training=False`` too (HSTUComputeOutputFunction) and would be affected.

Read per call; armed only by the value "1"; absent or "0" = off = the code
before this item. Not FP8 math; the prefix is what launch_slurm.sh forwards.
"""

import os
from typing import List

import torch
from generative_recommenders.ops.utils import arm_print_all_ranks

EVAL_LNMD_V3_ENV: str = "DLRM_HSTU_FP8_EVAL_LNMD_V3"
_LOGGED: List[bool] = [False, False]  # armed line, ignored line


@torch.jit.ignore
def eval_lnmd_v3_enabled() -> bool:
    """$DLRM_HSTU_FP8_EVAL_LNMD_V3 == "1" (read per call, no logging).

    Consulted only by LN*U*dropout forwards with training=False, so a
    training forward (training=True) never reads it.
    """
    return os.environ.get(EVAL_LNMD_V3_ENV, "0") == "1"


@torch.jit.ignore
def log_eval_lnmd_v3_armed() -> None:
    """One line per process (every rank, tagged [rank N]) at the first eval
    LN*U*dropout forward that runs v3."""
    if _LOGGED[0]:
        return
    _LOGGED[0] = True
    arm_print_all_ranks(
        "armed OK: [eval-lnmd-v3] DLRM_HSTU_FP8_EVAL_LNMD_V3=1: eval "
        "(training=False) LN*U*dropout forwards with the v3 eligibility run "
        "_ln_mul_dropout_fwd_rng_v3 TRAINING=False instead of the legacy "
        "per-row _ln_mul_dropout_fwd (declared numerics change: not bitwise "
        "to the legacy kernel)"
    )


@torch.jit.ignore
def log_eval_lnmd_v3_ignored(reason: str) -> None:
    """One line per process (every rank, tagged [rank N]) when the switch is
    set but v3 cannot run on this configuration at all (not for calls refused
    only by their shape, which take the legacy kernel as in training)."""
    if _LOGGED[1]:
        return
    _LOGGED[1] = True
    arm_print_all_ranks(
        "[eval-lnmd-v3] DLRM_HSTU_FP8_EVAL_LNMD_V3=1 ignored: "
        + reason
        + " (eval LN*U*dropout forwards run the legacy _ln_mul_dropout_fwd)"
    )
