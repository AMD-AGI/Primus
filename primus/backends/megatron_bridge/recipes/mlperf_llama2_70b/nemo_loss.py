###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
NeMo-equivalent training/validation loss for Primus (exact port of
``MaskedTokenLossReduction.forward + reduce``).

Background
----------
NeMo MLPerf (`mlperf_code_llama2_70b_0430/src/custom_llama.py`) overrides
``training_loss_reduction`` / ``validation_loss_reduction`` on the model
with ``MaskedTokenLossReduction``. The class has **two distinct paths**
depending on its constructor flags:

1. ``validation_step=False`` (training) **or** ``val_drop_last=True``:
    * per microbatch:  per-sample token-mean
        ``loss_for_ub[i] = sum(losses[i] * mask[i]) / sum(mask[i])``
    * across microbatches:  ``concat(per_sample_means).mean()``
2. ``validation_step=True`` AND ``val_drop_last=False`` (NeMo's default for
   eval; see ``CustomLlamaModel.validation_loss_reduction``):
    * per microbatch:  per-sample token-mean (same as path 1) **plus** a
      0/1 indicator ``num_valid_tokens_in_ub = (num_valid_tokens > 0)``,
      stacked to produce ``[loss_for_ub, valid_indicator]``.
    * across microbatches:  ``vstack(stacked).sum(dim=0)`` → returns
      ``[total_loss_sum, total_valid_sample_count]``.
    * the eval-time CP all-reduce on ``loss_mask`` / ``loss_for_ub`` is
      gated by ``disabled_cp_for_eval`` (i.e., CP is **off** for eval).

This is **not** equivalent to Megatron-Bridge's default
``masked_next_token_loss`` which computes a *global* token-mean across all
tokens in all samples. When samples have very different numbers of
unmasked tokens (typical for SFT/LoRA runs), the two objectives differ
and produce different gradient directions.

Implementation
--------------
We expose a NeMo-equivalent per-microbatch loss function that returns the
3-tuple Megatron-Core's pipeline schedule expects::

    (loss, num_tokens, {"lm loss": [loss_sum, count]})

For both paths, ``loss = loss_for_ub.sum()`` (sum over the microbatch's
per-sample means). The choice of ``num_tokens`` and the ``[sum, count]``
reporting tensor depends on the path:

* **Train / val_drop_last=True path**: ``num_tokens = num_samples = B``.
  After Megatron-Bridge's ``output /= clamp(num_tokens, 1); output /=
  num_microbatches`` and DP+microbatch ``[sum, count]`` aggregation, the
  effective backward target and reported loss are both
  ``mean over (microbatch, sample) of per_sample_mean`` -- byte-identical
  to NeMo's ``concat(per_sample_means).mean()`` when ``B`` is uniform.

* **val_drop_last=False path** (eval, NeMo default): ``num_tokens =
  num_samples = sum_i (num_valid_tokens_i > 0)``. Empty-mask samples
  (those whose entire sequence is masked) are excluded from the
  denominator -- mirroring NeMo's ``num_valid_tokens_in_ub`` indicator.
  Because ``loss_for_ub`` is already zeroed for those samples, the
  numerator (sum) is unaffected. Megatron-Bridge's eval reduction does
  ``total_loss_sum / total_valid_count`` across DP+CP+microbatch, which
  matches NeMo's ``vstack(...).sum(dim=0)`` semantics.

Train vs. eval detection
~~~~~~~~~~~~~~~~~~~~~~~~
Megatron-Bridge has a single ``forward_step`` for both train and eval and
wraps the eval pass in ``with torch.no_grad():`` (see
``megatron/bridge/training/eval.py:118``). We detect eval at call time
with ``torch.is_grad_enabled()`` -- a Megatron-Bridge-specific but stable
signal.

CP gating (``cp_eval`` / ``disabled_cp_for_eval``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NeMo's ``cp_eval`` parameter (``CustomLlamaModel.config.cp_eval``) toggles
``disabled_cp_for_eval``: when CP is **disabled** for eval, the CP
all-reduces on ``loss_mask`` and ``loss_for_ub`` must be **skipped** (the
data isn't actually sharded across CP ranks even though
``parallel_state.get_context_parallel_world_size() > 1``). We expose this
through env var ``PRIMUS_NEMO_LOSS_DISABLED_CP_FOR_EVAL`` (default ``0``,
i.e. CP-on-for-eval).

``val_drop_last`` is exposed via ``PRIMUS_NEMO_LOSS_VAL_DROP_LAST``
(default ``0``, matching NeMo's eval setup ``val_drop_last=False``).

Reporting parity
----------------
Megatron-Bridge's training & eval aggregation computes ``loss_sum /
count`` over (microbatch, DP-rank, CP-rank). With our return shape:

* Train / val_drop_last=True:
  ``sum_(rank, microbatch) sum_samples(per_sample_mean) /
   sum_(rank, microbatch) B  =  mean per_sample_mean``
* val_drop_last=False:
  ``sum_(rank, microbatch) sum_samples(per_sample_mean) /
   sum_(rank, microbatch) count_valid_samples
   = total_loss_sum / total_valid_sample_count`` (matches NeMo).

Activation
----------
Auto-installs at import time. Disable by setting ``PRIMUS_NEMO_LOSS=0``.

Env vars
~~~~~~~~
* ``PRIMUS_NEMO_LOSS``                         (default ``1``): master switch.
* ``PRIMUS_NEMO_LOSS_VAL_DROP_LAST``           (default ``0``): if ``1``,
  treat eval like train (NeMo's ``val_drop_last=True``).
* ``PRIMUS_NEMO_LOSS_DISABLED_CP_FOR_EVAL``    (default ``0``): if ``1``,
  skip the CP all-reduces during eval (NeMo's ``cp_eval`` set).
"""

from __future__ import annotations

import os
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
from megatron.core import parallel_state
from megatron.core.rerun_state_machine import get_rerun_state_machine

from primus.core.utils.module_utils import log_rank_0


def _safe_log_rank_0(msg: str) -> None:
    """Log on rank 0; fall back to print if the global logger is not ready yet."""
    try:
        log_rank_0(msg)
    except AttributeError:
        if os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) in ("0", "", None):
            print(msg, flush=True)


# Same value Megatron-Bridge uses in ``masked_next_token_loss``.
_SPIKY_LOSS_FACTOR: int = 10

# Module-level flag so callers can verify the patch landed.
_INSTALLED: bool = False


def _enabled() -> bool:
    """Return True when the NeMo-equivalent loss should be active."""
    flag = os.environ.get("PRIMUS_NEMO_LOSS", "1").strip().lower()
    return flag not in ("0", "false", "no", "off")


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var (``1/true/yes/on`` -> True; anything else -> False)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _is_validation_step() -> bool:
    """Detect whether we're inside a validation forward pass.

    Megatron-Bridge wraps the entire eval loop in
    ``with torch.no_grad():`` (``megatron/bridge/training/eval.py:118``)
    and reuses the same ``forward_step`` for both train and eval, so
    ``torch.is_grad_enabled()`` is a stable, framework-internal signal
    that distinguishes the two.
    """
    return not torch.is_grad_enabled()


def _val_drop_last() -> bool:
    """``val_drop_last`` flag, mirrored to NeMo's ``MaskedTokenLossReduction``.

    NeMo's ``CustomLlamaModel.validation_loss_reduction`` constructs the
    eval reduction with ``val_drop_last=False`` (drop_last=False so all
    eval samples count). Default here matches that. Set
    ``PRIMUS_NEMO_LOSS_VAL_DROP_LAST=1`` to fall back to the train-style
    averaging (``concat(per_sample_means).mean()`` over all microbatches).
    """
    return _env_bool("PRIMUS_NEMO_LOSS_VAL_DROP_LAST", default=False)


def _disabled_cp_for_eval() -> bool:
    """``disabled_cp_for_eval`` flag, mirrored to NeMo's ``cp_eval`` parameter.

    When CP is disabled for the eval forward pass (i.e., the model
    doesn't actually shard across CP ranks during eval, even though
    ``parallel_state.get_context_parallel_world_size() > 1``), the CP
    all-reduces on ``loss_mask`` and ``loss_for_ub`` MUST be skipped --
    otherwise we double-count along the CP dimension. NeMo encodes this
    via ``cp_eval is not None`` in ``MaskedTokenLossReduction.__init__``.
    Default here is ``False`` (CP-on-for-eval); set
    ``PRIMUS_NEMO_LOSS_DISABLED_CP_FOR_EVAL=1`` if your recipe disables
    CP during eval.
    """
    return _env_bool("PRIMUS_NEMO_LOSS_DISABLED_CP_FOR_EVAL", default=False)


def _cp_eval_value() -> Optional[int]:
    """Return NeMo's ``cp_eval`` integer value for the validation reduction.

    NeMo's ``CustomLlamaModel.validation_loss_reduction`` passes
    ``cp_eval=self.config.cp_eval``, which is the *integer*
    ``cfg.model.eval_cp`` (or ``None``). The class only checks
    ``cp_eval is not None`` to set ``disabled_cp_for_eval``, so the
    integer value itself never affects the loss math -- but we keep the
    int-or-None contract to remain byte-identical to NeMo.

    Resolution order:
      1. ``PRIMUS_NEMO_LOSS_CP_EVAL`` (preferred). Set to an integer
         (e.g. ``1``) to mark "CP is disabled for eval". Empty / unset
         leaves it as ``None``.
      2. Fallback: ``PRIMUS_NEMO_LOSS_DISABLED_CP_FOR_EVAL`` (boolean).
         If set truthy, returns ``1``; else ``None``. Provided for
         backwards compat with the function-based env var.
    """
    raw = os.environ.get("PRIMUS_NEMO_LOSS_CP_EVAL")
    if raw is not None and raw.strip():
        try:
            return int(raw)
        except ValueError:
            return 1 if raw.strip().lower() in ("1", "true", "yes", "on") else None
    return 1 if _disabled_cp_for_eval() else None


# ============================================================================
# Byte-identical port of NeMo's ``MaskedTokenLossReduction`` class.
#
# Source:
#     mlperf_code_llama2_70b_0430/src/custom_llama.py:160-221
# Original imports replaced 1:1:
#     nemo.lightning.megatron_parallel.MegatronLossReduction -> stub below
#     megatron.core.parallel_state                            -> identical
#     torch                                                   -> identical
#
# The class body is reproduced as-is so the per-microbatch loss math, the
# CP / DP all-reduce gating, and the train vs. val branching are
# guaranteed identical to NeMo. The Megatron-Bridge wiring (which
# expects a (loss, num_tokens, dict) 3-tuple instead of NeMo's
# (loss_for_ub, dict)) lives in ``nemo_masked_next_token_loss`` further
# down -- it instantiates the right train / val singleton, calls
# ``forward(batch, per_token_losses)``, and repackages the return.
# ============================================================================


class MegatronLossReduction:
    """Stub of ``nemo.lightning.megatron_parallel.MegatronLossReduction``.

    The NeMo base class is just an interface marker (its ``forward`` /
    ``reduce`` are abstract). We reproduce it as an empty class so the
    ported ``MaskedTokenLossReduction`` keeps the same MRO for any
    callers that introspect via ``isinstance``.
    """


class MaskedTokenLossReduction(MegatronLossReduction):
    """Byte-identical port of NeMo's ``MaskedTokenLossReduction``.

    Identical to ``custom_llama.MaskedTokenLossReduction`` (lines 160-221
    of ``mlperf_code_llama2_70b_0430/src/custom_llama.py``):

    * ``__init__`` signature, attribute names, and ``disabled_cp_for_eval``
      derivation are unchanged.
    * ``forward(batch, per_token_losses)`` returns NeMo's 2-tuple
      ``(loss_for_ub, dict)`` where the dict is either ``{"avg":
      reduced_loss}`` (train / ``val_drop_last=True``) or
      ``{"loss_sum_and_ub_size": <(B, 2) all-reduced tensor>}``
      (validation with ``val_drop_last=False``).
    * ``reduce(losses_reduced_per_micro_batch)`` aggregates microbatch
      outputs identically (``concat(...).mean()`` for ``"avg"``,
      ``vstack(...).sum(dim=0)`` for ``"loss_sum_and_ub_size"``).

    To use this class with Megatron-Bridge's function-based loss API
    (which expects ``(loss, num_tokens, {"lm loss": [sum, count]})``),
    go through ``nemo_masked_next_token_loss`` -- it builds an instance
    (one singleton for train, one for val), calls ``forward``, and
    repackages the return.
    """

    def __init__(
        self,
        validation_step: bool = False,
        val_drop_last: bool = True,
        cp_eval: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.validation_step = validation_step
        self.train_step = not validation_step
        self.val_drop_last = val_drop_last
        self.disabled_cp_for_eval = cp_eval is not None

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        per_token_losses: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(per_token_losses, tuple):
            per_token_losses, loss_mask = per_token_losses
            batch["loss_mask"] = loss_mask
        masked_losses = per_token_losses * batch["loss_mask"]

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size > 1 and (self.train_step or not self.disabled_cp_for_eval):
            torch.distributed.all_reduce(
                batch["loss_mask"], group=parallel_state.get_context_parallel_group()
            )

        num_valid_tokens = batch["loss_mask"].sum(1)
        loss_for_ub = torch.sum(masked_losses, dim=1) / num_valid_tokens
        loss_for_ub = torch.where(num_valid_tokens == 0, torch.zeros_like(loss_for_ub), loss_for_ub)

        if cp_size > 1 and (self.train_step or not self.disabled_cp_for_eval):
            torch.distributed.all_reduce(loss_for_ub, group=parallel_state.get_context_parallel_group())

        if self.validation_step and not self.val_drop_last:
            num_valid_tokens_in_ub = (num_valid_tokens > 0).long()
            loss_sum_and_ub_size_all_gpu = torch.stack(
                [loss_for_ub.clone().detach(), num_valid_tokens_in_ub.clone().detach()],
                dim=1,
            )
            if self.disabled_cp_for_eval:
                torch.distributed.all_reduce(loss_sum_and_ub_size_all_gpu)
            else:
                torch.distributed.all_reduce(
                    loss_sum_and_ub_size_all_gpu,
                    group=parallel_state.get_data_parallel_group(),
                )
            return loss_for_ub, {"loss_sum_and_ub_size": loss_sum_and_ub_size_all_gpu}

        reduced_loss = loss_for_ub
        return loss_for_ub, {"avg": reduced_loss}

    def reduce(self, losses_reduced_per_micro_batch: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if losses_reduced_per_micro_batch:
            if "avg" in losses_reduced_per_micro_batch[0]:
                loss_tensors_list = [loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch]
                loss_tensor = torch.concat(loss_tensors_list)
                return loss_tensor.mean()

            loss_sum_tensors_list: List[torch.Tensor] = [
                loss_sum["loss_sum_and_ub_size"] for loss_sum in losses_reduced_per_micro_batch
            ]
            loss_sum = (
                torch.vstack(loss_sum_tensors_list).sum(dim=0)
                if len(loss_sum_tensors_list) > 0
                else torch.tensor([0.0, 0.0], device=torch.cuda.current_device())
            )
            return loss_sum

        return torch.tensor(0.0, device=torch.cuda.current_device())


# ----------------------------------------------------------------------------
# Lazy singletons mirroring NeMo's ``CustomLlamaModel`` setup:
#
#     self._training_loss_reduction   = MaskedTokenLossReduction()
#     self._validation_loss_reduction = MaskedTokenLossReduction(
#         validation_step=True,
#         val_drop_last=False,
#         cp_eval=self.config.cp_eval,
#     )
#
# (See ``mlperf_code_llama2_70b_0430/src/custom_llama.py:143-157``.)
# We construct lazily because env vars may not be set at import time.
# ----------------------------------------------------------------------------

_TRAIN_REDUCTION: Optional[MaskedTokenLossReduction] = None
_VAL_REDUCTION: Optional[MaskedTokenLossReduction] = None


def _get_train_reduction() -> MaskedTokenLossReduction:
    """Return the per-process training ``MaskedTokenLossReduction`` singleton."""
    global _TRAIN_REDUCTION
    if _TRAIN_REDUCTION is None:
        _TRAIN_REDUCTION = MaskedTokenLossReduction()
    return _TRAIN_REDUCTION


def _get_val_reduction() -> MaskedTokenLossReduction:
    """Return the per-process validation ``MaskedTokenLossReduction`` singleton."""
    global _VAL_REDUCTION
    if _VAL_REDUCTION is None:
        _VAL_REDUCTION = MaskedTokenLossReduction(
            validation_step=True,
            val_drop_last=_val_drop_last(),
            cp_eval=_cp_eval_value(),
        )
    return _VAL_REDUCTION


def reset_loss_reduction_singletons() -> None:
    """Drop cached singletons so subsequent calls re-read env vars.

    Useful for tests or for recipes that reconfigure ``cp_eval`` /
    ``val_drop_last`` after import time.
    """
    global _TRAIN_REDUCTION, _VAL_REDUCTION
    _TRAIN_REDUCTION = None
    _VAL_REDUCTION = None


def nemo_masked_next_token_loss(
    loss_mask: torch.Tensor,
    output_tensor: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    check_for_nan_in_loss: bool = True,
    check_for_spiky_loss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Megatron-Bridge adapter for the byte-identical ``MaskedTokenLossReduction``.

    All loss math (CP / DP all-reduces, val_drop_last branching,
    train vs. eval gating) lives in the class above and is identical
    to NeMo. This adapter does three things:

    1. Picks the right per-process singleton (train vs. val) based on
       ``torch.is_grad_enabled()`` -- Megatron-Bridge wraps eval in
       ``with torch.no_grad():`` (``eval.py:118``) and reuses the same
       ``forward_step`` for both, so this is the only stable signal.
    2. Calls ``MaskedTokenLossReduction.forward(batch, per_token_losses)``
       with a ``batch = {"loss_mask": loss_mask}`` shim. The class
       mutates ``batch["loss_mask"]`` in-place via the CP all-reduce --
       same as NeMo, so we accept the side effect.
    3. Repackages NeMo's 2-tuple ``(loss_for_ub, dict)`` return into the
       Megatron-Core schedule's expected 3-tuple
       ``(loss, num_tokens, {"lm loss": [sum, count]})``:

       * ``"avg"`` branch (train / ``val_drop_last=True``):
           - ``loss = loss_for_ub.sum()``, ``num_tokens = B``.
             Megatron-Core computes ``loss / clamp(num_tokens, 1) /
             num_microbatches`` for backward, giving
             ``mean over (microbatch, sample) of per_sample_mean`` --
             byte-identical to NeMo's ``concat(...).mean()`` reduction
             when ``B`` is uniform across microbatches and DP ranks.
       * ``"loss_sum_and_ub_size"`` branch (val w/ ``val_drop_last=False``):
           - The class already DP-all-reduced the (B, 2) tensor inside
             ``forward``. We sum dim=0 (matches NeMo's ``vstack(...).sum(dim=0)``
             reduce) to get ``[total_loss_sum, total_valid_count]`` for
             this microbatch, then return it as the reporting tensor.
             Megatron-Bridge's eval reduction then sums it across
             microbatches and DP+CP (``eval.py:183``); since both
             entries scale by the same factor in any further reduction,
             the final ratio ``loss_sum / count`` is identical to
             NeMo's reported eval loss.

    NaN / Inf / spike validation runs on the local per-sample sum so it
    triggers regardless of which branch the class took.
    """
    if isinstance(output_tensor, tuple):
        per_token_losses, dynamic_mask = output_tensor
        loss_mask = dynamic_mask
    else:
        per_token_losses = output_tensor

    per_token_losses = per_token_losses.float()
    loss_mask = loss_mask.float()

    batch_size = loss_mask.shape[0]
    per_token_losses = per_token_losses.view(batch_size, -1)
    loss_mask = loss_mask.view(batch_size, -1)

    reduction = _get_val_reduction() if _is_validation_step() else _get_train_reduction()

    batch: Dict[str, torch.Tensor] = {"loss_mask": loss_mask}
    loss_for_ub, reduced_dict = reduction.forward(batch, per_token_losses)

    loss_sum = loss_for_ub.sum()

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

    if "loss_sum_and_ub_size" in reduced_dict:
        agg = reduced_dict["loss_sum_and_ub_size"].sum(dim=0).float()
        agg_loss_sum = agg[0]
        agg_count = agg[1]
        num_samples = agg_count.to(torch.int)
        reporting_loss = torch.cat(
            [
                agg_loss_sum.clone().detach().view(1),
                agg_count.clone().detach().view(1),
            ]
        )
        return (agg_loss_sum, num_samples, {"lm loss": reporting_loss})

    num_samples = torch.tensor(loss_for_ub.numel(), dtype=torch.int, device=loss_for_ub.device)
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
        _safe_log_rank_0(
            "[primus.nemo_loss] PRIMUS_NEMO_LOSS disabled; using Megatron-Bridge "
            "default global-token-mean loss"
        )
        return False

    patched_any = False

    try:
        from megatron.bridge.training import vlm_step as _vlm

        _vlm._create_loss_function = create_nemo_masked_next_token_loss_function
        _safe_log_rank_0(
            "[primus.nemo_loss] Patched megatron.bridge.training.vlm_step._create_loss_function "
            "-> NeMo MaskedTokenLossReduction-equivalent (per-sample token-mean)"
        )
        patched_any = True
    except Exception as e:  # pragma: no cover - defensive
        _safe_log_rank_0(f"[primus.nemo_loss] Skipping vlm_step patch: {e!r}")

    try:
        from megatron.bridge.training import gpt_step as _gpt

        _gpt.masked_next_token_loss = nemo_masked_next_token_loss
        _safe_log_rank_0(
            "[primus.nemo_loss] Patched megatron.bridge.training.gpt_step.masked_next_token_loss "
            "-> NeMo per-sample-mean variant"
        )
        patched_any = True
    except Exception as e:  # pragma: no cover - defensive
        _safe_log_rank_0(f"[primus.nemo_loss] Skipping gpt_step patch: {e!r}")

    _INSTALLED = patched_any
    return patched_any


# Auto-install on import. Importing this module from ``llama2_custom.py``
# (or any recipe entry point) is sufficient to activate NeMo-equivalent
# loss reporting + gradient semantics for the entire training run.
install_nemo_loss_if_enabled()
