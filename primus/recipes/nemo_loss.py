###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
NeMo-equivalent training/validation loss for Primus (port of
``MaskedTokenLossReduction.forward + reduce``).

Background
----------
NeMo MLPerf (`mlperf_code_llama2_70b_0430/src/custom_llama.py`) overrides
``training_loss_reduction`` / ``validation_loss_reduction`` on the model with
``MaskedTokenLossReduction``:

* per microbatch:  per-sample token-mean
    ``loss_for_ub[i] = sum(losses[i] * mask[i]) / sum(mask[i])``
* across microbatches:  ``concat(per_sample_means).mean()``

This is **not** equivalent to Megatron-Bridge's default ``masked_next_token_loss``
which computes a *global* token-mean across all tokens in all samples. When
samples have very different numbers of unmasked tokens (typical for SFT/LoRA
runs), the two objectives differ and produce different gradient directions,
which is one source of the post-healing loss-drop divergence we see vs NeMo.

Implementation
--------------
We expose a NeMo-equivalent per-microbatch loss function that returns the
3-tuple Megatron-Core's pipeline schedule expects::

    (loss, num_tokens, {"lm loss": [loss_sum, count]})

with values chosen so that the schedule's:

    output_tensor /= clamp(num_tokens, 1)
    output_tensor /= num_microbatches

ends up driving backward with::

    grad_total = mean over (microbatch, sample) of d(per_sample_mean)/dW

which is exactly NeMo's ``MaskedTokenLossReduction.reduce``:
``concat(per_sample_means).mean()``.

We accomplish this by returning per microbatch:

* ``loss = sum_over_samples(per_sample_mean)``       (a scalar)
* ``num_tokens = B``                                 (microbatch sample count)
* ``reporting_loss = [loss, B]``                     (for ``[sum, count]`` aggregation)

Megatron-Core then divides by ``B`` (giving microbatch sample-mean) and by
``num_microbatches`` (giving the global mean of microbatch-sample-means as
the backward target), which is byte-identical to NeMo's reduction up to
floating-point reduction order.

Reporting parity
----------------
Megatron-Bridge's training & eval aggregation already computes
``loss_sum / count`` over (microbatch, DP-rank). With our return shape, that
ratio is::

    sum_{rank, microbatch} sum_samples(per_sample_mean)
    --------------------------------------------------- = mean per_sample_mean
    sum_{rank, microbatch} B

which matches NeMo's reported training/validation loss when every microbatch
on every rank has the same ``B`` (the standard Megatron setup).

Activation
----------
Auto-installs at import time. Disable by setting ``PRIMUS_NEMO_LOSS=0``.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Tuple, Union

import torch
from megatron.core import parallel_state
from megatron.core.rerun_state_machine import get_rerun_state_machine

from primus.modules.module_utils import log_rank_0


# Same value Megatron-Bridge uses in ``masked_next_token_loss``.
_SPIKY_LOSS_FACTOR: int = 10

# Module-level flag so callers can verify the patch landed.
_INSTALLED: bool = False


def _enabled() -> bool:
    """Return True when the NeMo-equivalent loss should be active."""
    flag = os.environ.get("PRIMUS_NEMO_LOSS", "1").strip().lower()
    return flag not in ("0", "false", "no", "off")


def nemo_masked_next_token_loss(
    loss_mask: torch.Tensor,
    output_tensor: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    check_for_nan_in_loss: bool = True,
    check_for_spiky_loss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """NeMo-equivalent per-microbatch loss.

    Mirrors ``MaskedTokenLossReduction.forward`` (mlperf_code_llama2_70b_0430):

      masked_losses    = per_token_losses * loss_mask
      num_valid_tokens = loss_mask.sum(dim=1)
      loss_for_ub      = masked_losses.sum(dim=1) / clamp(num_valid_tokens, 1)
      loss_for_ub[num_valid_tokens == 0] = 0

    Returns a 3-tuple compatible with Megatron-Core's pipeline schedule that
    yields a backward equivalent to NeMo's
    ``concat(per_sample_means).mean()`` reduction (see module docstring).

    Args:
        loss_mask: Loss mask captured at forward-step construction time. Used
            unless ``output_tensor`` is a ``(losses, dynamic_mask)`` tuple
            (LLaVA-style), in which case the dynamic mask wins.
        output_tensor: Per-token loss tensor from the model
            (shape ``(B, S_local)``) or a ``(losses, mask)`` tuple.
        check_for_nan_in_loss: Validate against NaN via the rerun state machine.
        check_for_spiky_loss: Validate against unexpectedly large losses.

    Returns:
        Tuple of (loss_sum_over_samples, num_samples, {"lm loss": [sum, count]}).
    """
    # LLaVA / packed-seq style override.
    if isinstance(output_tensor, tuple):
        per_token_losses = output_tensor[0]
        loss_mask = output_tensor[1]
    else:
        per_token_losses = output_tensor

    per_token_losses = per_token_losses.float()
    loss_mask = loss_mask.float()

    # Force 2D: (B, S_local). The base loss takes a single ``view(-1)``; here
    # we need per-row reductions so reshape using the mask's batch dim.
    batch_size = loss_mask.shape[0]
    per_token_losses = per_token_losses.view(batch_size, -1)
    loss_mask = loss_mask.view(batch_size, -1)

    # Context-parallel handling, byte-equivalent to NeMo's
    # ``MaskedTokenLossReduction.forward``: all-reduce the mask so
    # ``num_valid_tokens`` reflects the whole sequence, compute the local
    # per-sample sum / total-count, then all-reduce ``loss_for_ub`` so each
    # CP rank has the full per-sample mean.
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size > 1:
        torch.distributed.all_reduce(
            loss_mask, group=parallel_state.get_context_parallel_group()
        )

    masked_losses = per_token_losses * loss_mask
    num_valid_tokens = loss_mask.sum(dim=1)
    safe_denom = torch.clamp(num_valid_tokens, min=1.0)
    loss_for_ub = masked_losses.sum(dim=1) / safe_denom
    loss_for_ub = torch.where(
        num_valid_tokens == 0, torch.zeros_like(loss_for_ub), loss_for_ub
    )

    if cp_size > 1:
        torch.distributed.all_reduce(
            loss_for_ub, group=parallel_state.get_context_parallel_group()
        )

    # Microbatch contribution: schedule will divide by ``num_tokens`` (=B)
    # and ``num_microbatches`` to produce the desired backward target.
    loss_sum = loss_for_ub.sum()
    num_samples = torch.tensor(
        loss_for_ub.numel(), dtype=torch.int, device=loss_for_ub.device
    )

    # NaN / Inf / spiky-loss validation (parity with masked_next_token_loss).
    rerun_state_machine = get_rerun_state_machine()
    if check_for_nan_in_loss:
        rerun_state_machine.validate_result(
            result=loss_sum,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss_sum,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
    if check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss_sum,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=_SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,
            fatal=False,
        )

    reporting_loss = torch.cat(
        [
            loss_sum.clone().detach().view(1),
            num_samples.detach().view(1).float(),
        ]
    )
    return (loss_sum, num_samples, {"lm loss": reporting_loss})


def create_nemo_masked_next_token_loss_function(
    loss_mask: torch.Tensor,
    check_for_nan_in_loss: bool,
    check_for_spiky_loss: bool,
) -> partial:
    """Factory matching Megatron-Bridge's ``create_masked_next_token_loss_function``."""
    return partial(
        nemo_masked_next_token_loss,
        loss_mask,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
    )


def install_nemo_loss_if_enabled() -> bool:
    """Monkey-patch Megatron-Bridge step modules to use the NeMo-equivalent loss.

    Affected bindings:

    * ``megatron.bridge.training.vlm_step._create_loss_function`` ->
      ``create_nemo_masked_next_token_loss_function`` (used by
      ``MegatronBridgePosttrainTrainer.train()`` -> ``finetune(...,
      forward_step_func=vlm_step.forward_step)``).
    * ``megatron.bridge.training.gpt_step.masked_next_token_loss`` ->
      ``nemo_masked_next_token_loss`` (used by other gpt_step-based recipes
      and any test callers).

    Idempotent. Gated by env var ``PRIMUS_NEMO_LOSS`` (default: enabled;
    disable with ``PRIMUS_NEMO_LOSS=0``).

    Returns True when patching took effect, False if disabled or already
    installed.
    """
    global _INSTALLED
    if _INSTALLED:
        return False
    if not _enabled():
        log_rank_0(
            "[primus.nemo_loss] PRIMUS_NEMO_LOSS disabled; using Megatron-Bridge "
            "default global-token-mean loss"
        )
        return False

    patched_any = False

    try:
        from megatron.bridge.training import vlm_step as _vlm

        _vlm._create_loss_function = create_nemo_masked_next_token_loss_function
        log_rank_0(
            "[primus.nemo_loss] Patched megatron.bridge.training.vlm_step._create_loss_function "
            "-> NeMo MaskedTokenLossReduction-equivalent (per-sample token-mean)"
        )
        patched_any = True
    except Exception as e:  # pragma: no cover - defensive
        log_rank_0(f"[primus.nemo_loss] Skipping vlm_step patch: {e!r}")

    try:
        from megatron.bridge.training import gpt_step as _gpt

        _gpt.masked_next_token_loss = nemo_masked_next_token_loss
        log_rank_0(
            "[primus.nemo_loss] Patched megatron.bridge.training.gpt_step.masked_next_token_loss "
            "-> NeMo per-sample-mean variant"
        )
        patched_any = True
    except Exception as e:  # pragma: no cover - defensive
        log_rank_0(f"[primus.nemo_loss] Skipping gpt_step patch: {e!r}")

    _INSTALLED = patched_any
    return patched_any


# Auto-install on import. Importing this module from ``llama2_custom.py``
# (or any recipe entry point) is sufficient to activate NeMo-equivalent
# loss reporting + gradient semantics for the entire training run.
install_nemo_loss_if_enabled()
