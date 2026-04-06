###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron-Bridge validation/test logging for Primus diffusion pretrain.

Patches ``evaluate_and_print_results`` on ``eval``, ``train``, and ``pretrain`` (import-by-value).

- Avoids duplicate rank-0 lines: loss summaries use Primus ``log_rank_0`` via
  :mod:`training_log_mirror` only, not ``print_rank_0`` + mirror.
- Skips redundant post-train **validation** when the last training step already ran in-loop eval
  (same ``step`` and ``step % eval_interval == 0``).
- Optional skip of post-train **test** pass (``train.eval_skip_posttrain_test``) when val/test
  iterators are the same data (common for FLUX Energon).
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

_orig_evaluate_and_print: Optional[Callable[..., Any]] = None
_patches_active: bool = False
_patched_modules: List[Any] = []


def _patch_evaluate_and_print_modules(wrapped: Callable[..., Any]) -> None:
    """Assign ``wrapped`` everywhere ``evaluate_and_print_results`` is imported by value."""
    import megatron.bridge.training.eval as mb_eval
    import megatron.bridge.training.pretrain as mb_pretrain
    import megatron.bridge.training.train as mb_train

    global _patched_modules
    mb_eval.evaluate_and_print_results = wrapped
    mb_train.evaluate_and_print_results = wrapped
    mb_pretrain.evaluate_and_print_results = wrapped
    _patched_modules = [mb_eval, mb_train, mb_pretrain]


def _skip_redundant_posttrain_validation(state: Any, prefix: str) -> bool:
    """Megatron ``pretrain`` re-runs validation after ``train()``; skip if the loop already eval'd."""
    if "on validation set" not in prefix:
        return False
    train = state.cfg.train
    if train.train_iters is None or not train.eval_interval:
        return False
    step = state.train_state.step
    if step <= 0:
        return False
    return step == train.train_iters and (step % train.eval_interval == 0)


def _skip_posttrain_test(state: Any, prefix: str) -> bool:
    if "on test set" not in prefix:
        return False
    return bool(getattr(state.cfg.train, "eval_skip_posttrain_test", False))


def install_eval_result_logging_patches() -> None:
    """Monkey-patch ``evaluate_and_print_results`` on eval, train, and pretrain modules."""
    global _orig_evaluate_and_print, _patches_active

    if _patches_active:
        return

    import megatron.bridge.training.eval as mb_eval

    _orig_evaluate_and_print = mb_eval.evaluate_and_print_results

    def _wrapped(
        state: Any,
        prefix: str,
        forward_step_func: Any,
        data_iterator: Any,
        model: Any,
        config: Any,
        verbose: bool = False,
        write_to_tensorboard: bool = True,
        process_non_loss_data_func: Any = None,
        non_loss_data_func: Any = None,
    ) -> None:
        if _skip_redundant_posttrain_validation(state, prefix) or _skip_posttrain_test(state, prefix):
            return

        train = state.cfg.train
        if hasattr(train, "eval_verbose"):
            verbose = bool(getattr(train, "eval_verbose"))
        else:
            verbose = False

        import megatron.bridge.training.eval as mb_eval
        import megatron.bridge.utils.common_utils as cu

        saved_cu_pr_last = cu.print_rank_last
        saved_eval_pr_last = mb_eval.print_rank_last

        def _print_rank_last_rank0(message: str) -> None:
            mirror_on = os.environ.get("PRIMUS_MIRROR_MEGATRON_BRIDGE_TRAINING_LOG", "1").strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            )
            if "validation loss at" in message:
                if not mirror_on:
                    cu.print_rank_0(message)
            else:
                cu.print_rank_0(message)
            saved_eval_pr_last(message)

        cu.print_rank_last = _print_rank_last_rank0
        mb_eval.print_rank_last = _print_rank_last_rank0
        try:
            assert _orig_evaluate_and_print is not None
            _orig_evaluate_and_print(
                state,
                prefix,
                forward_step_func,
                data_iterator,
                model,
                config,
                verbose=verbose,
                write_to_tensorboard=write_to_tensorboard,
                process_non_loss_data_func=process_non_loss_data_func,
                non_loss_data_func=non_loss_data_func,
            )
        finally:
            cu.print_rank_last = saved_cu_pr_last
            mb_eval.print_rank_last = saved_eval_pr_last

    _patch_evaluate_and_print_modules(_wrapped)
    _patches_active = True


def uninstall_eval_result_logging_patches() -> None:
    global _orig_evaluate_and_print, _patches_active, _patched_modules

    if not _patches_active:
        return

    if _orig_evaluate_and_print is not None:
        for mod in _patched_modules:
            mod.evaluate_and_print_results = _orig_evaluate_and_print

    _orig_evaluate_and_print = None
    _patched_modules = []
    _patches_active = False
