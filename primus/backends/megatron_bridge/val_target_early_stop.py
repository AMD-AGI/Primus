###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Stop Megatron-Bridge pretraining once validation loss meets a target (e.g. MLPerf FLUX 0.586).

Uses monkey-patches on ``megatron.bridge.training.eval.evaluate`` and
``megatron.bridge.training.train.checkpoint_and_decide_exit`` so we do not fork third_party.

Knobs (set on ``config.train`` via YAML flat overrides):

- ``val_stop_loss`` (float): threshold; if unset, patches are not installed.
- ``val_stop_loss_key`` (str | None): substring to match a key in the validation loss dict
  (default: prefer ``lm loss``, then any key containing ``loss``).
- ``val_stop_mode`` (str): ``le`` = stop when metric <= target (MLPerf); ``ge`` = stop when >= target.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from primus.modules.module_utils import log_rank_0

_orig_evaluate: Optional[Callable[..., Any]] = None
_orig_checkpoint_and_decide_exit: Optional[Callable[..., Any]] = None
_patches_active: bool = False


def _pick_validation_metric(total_loss_dict: dict, key_substring: Optional[str]) -> Optional[float]:
    if not total_loss_dict:
        return None
    if key_substring:
        ks = key_substring.lower()
        for k, v in total_loss_dict.items():
            if ks in k.lower():
                return float(v.item()) if hasattr(v, "item") else float(v)
    for preferred in ("lm loss", "loss"):
        for k, v in total_loss_dict.items():
            if preferred in k.lower():
                return float(v.item()) if hasattr(v, "item") else float(v)
    k0 = next(iter(total_loss_dict))
    v = total_loss_dict[k0]
    return float(v.item()) if hasattr(v, "item") else float(v)


def _evaluate_with_val_target(
    state: Any,
    forward_step_func: Any,
    data_iterator: Any,
    model: Any,
    process_non_loss_data_func: Any,
    config: Any,
    verbose: bool = False,
    non_loss_data_func: Any = None,
):
    assert _orig_evaluate is not None
    result = _orig_evaluate(
        state,
        forward_step_func,
        data_iterator,
        model,
        process_non_loss_data_func,
        config,
        verbose,
        non_loss_data_func,
    )
    total_loss_dict, _collected, timelimit = result
    if timelimit or not total_loss_dict:
        return result

    target = getattr(state.cfg.train, "val_stop_loss", None)
    if target is None:
        return result

    mode = (getattr(state.cfg.train, "val_stop_mode", None) or "le").lower()
    key_sub = getattr(state.cfg.train, "val_stop_loss_key", None)
    val = _pick_validation_metric(total_loss_dict, key_sub)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stop_local = torch.tensor([0.0], device=device, dtype=torch.float32)
    if val is not None:
        t = float(target)
        hit = (mode == "le" and val <= t) or (mode == "ge" and val >= t)
        if hit:
            stop_local[0] = 1.0

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stop_local, op=dist.ReduceOp.MAX)

    if stop_local.item() > 0.5:
        state.train_state._primus_val_target_reached = True  # noqa: SLF001
        log_rank_0(
            f"Primus: validation target satisfied (metric={val}, val_stop_loss={target}, mode={mode}); "
            "stopping after this iteration."
        )

    return result


def _checkpoint_with_val_target(
    state: Any,
    model: Any,
    optimizer: Any,
    opt_param_scheduler: Any,
    num_floating_point_operations_so_far: float,
    checkpointing_context: dict[str, Any],
    train_data_iterator: Any,
) -> bool:
    assert _orig_checkpoint_and_decide_exit is not None
    should_exit = _orig_checkpoint_and_decide_exit(
        state,
        model,
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far,
        checkpointing_context,
        train_data_iterator,
    )
    if should_exit:
        return True
    if not getattr(state.train_state, "_primus_val_target_reached", False):
        return False

    from megatron.bridge.training.train import save_checkpoint_and_time

    if state.cfg.checkpoint.save:
        save_checkpoint_and_time(
            state,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far,
            checkpointing_context,
            train_data_iterator=train_data_iterator,
        )
    from megatron.bridge.training.utils.log_utils import barrier_and_log

    barrier_and_log(
        "Primus: exiting training because validation loss reached val_stop_loss "
        f"({getattr(state.cfg.train, 'val_stop_loss', None)})."
    )
    return True


def install_validation_target_early_stop(config_container: Any) -> None:
    """Install patches if ``config_container.train.val_stop_loss`` is set."""
    global _orig_evaluate, _orig_checkpoint_and_decide_exit, _patches_active

    target = getattr(config_container.train, "val_stop_loss", None)
    if target is None:
        return
    if _patches_active:
        return

    import megatron.bridge.training.eval as mb_eval
    import megatron.bridge.training.train as mb_train

    _orig_evaluate = mb_eval.evaluate
    _orig_checkpoint_and_decide_exit = mb_train.checkpoint_and_decide_exit

    mb_eval.evaluate = _evaluate_with_val_target
    mb_train.checkpoint_and_decide_exit = _checkpoint_with_val_target
    _patches_active = True
    log_rank_0(f"Primus: validation early-stop enabled (val_stop_loss={target!r}).")


def uninstall_validation_target_early_stop() -> None:
    """Restore original Megatron-Bridge functions."""
    global _orig_evaluate, _orig_checkpoint_and_decide_exit, _patches_active

    if not _patches_active:
        return

    import megatron.bridge.training.eval as mb_eval
    import megatron.bridge.training.train as mb_train

    if _orig_evaluate is not None:
        mb_eval.evaluate = _orig_evaluate
    if _orig_checkpoint_and_decide_exit is not None:
        mb_train.checkpoint_and_decide_exit = _orig_checkpoint_and_decide_exit

    _orig_evaluate = None
    _orig_checkpoint_and_decide_exit = None
    _patches_active = False
