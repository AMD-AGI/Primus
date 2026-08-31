###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
The single transition between MLPerf initialization and measured training.

MLPerf requires the clock to start before the implementation touches the
dataset. Megatron's ``pretrain()`` runs ``setup_model_and_optimizer``, then
builds the data iterators, then calls ``train()``, and Primus' patch phases
(``setup`` / ``build_args`` / ``before_train`` / ``after_train``) all fire
before ``pretrain()`` -- there is no phase between model build and data build.
Megatron-LM is a pinned submodule, so a new phase cannot be inserted at the
call site either.

This module therefore creates the missing seam by wrapping three Megatron
entry points from the ``before_train`` phase:

  ``pretrain``                            capture ``forward_step_func``
  ``setup_model_and_optimizer``           capture model / optimizer / scheduler
  ``build_train_valid_test_data_iterators``  fire the transition on entry

On the first call into the data-iterator builder -- before any shard, worker
or prefetcher exists -- registered pre-run hooks execute (compile warmup),
all ranks synchronize, rank 0 emits ``init_stop`` / ``run_start``, and a
second barrier holds every rank until those records exist.

Hooks are ordered so warmup always precedes the transition; the transition
itself is a separate registration so the logging patch owns what it emits.
"""

import logging

logger = logging.getLogger(__name__)

_ORIGINALS: dict = {}
_HOOKS: list = []
_TRANSITION: list = []
_CAPTURED: dict = {}
_FIRED = [False]


def register_pre_run_hook(name: str, fn, order: int = 50) -> None:
    """Register work that must finish before the clock starts.

    Hooks run in ascending ``order`` on every rank, inside the initialization
    window, with the captured model/optimizer available via :func:`captured`.
    """
    _HOOKS.append((order, name, fn))
    _HOOKS.sort(key=lambda entry: entry[0])


def set_transition(fn) -> None:
    """Register the callable that emits ``init_stop`` / ``run_start``."""
    _TRANSITION.clear()
    _TRANSITION.append(fn)


def captured() -> dict:
    """Objects captured from Megatron on the way to the boundary.

    Keys are present only once the corresponding entry point has run:
    ``forward_step_func``, ``model``, ``optimizer``, ``opt_param_scheduler``.
    """
    return _CAPTURED


def has_fired() -> bool:
    return _FIRED[0]


def reset_for_tests() -> None:
    """Drop all registrations and captures. Tests only."""
    _ORIGINALS.clear()
    _HOOKS.clear()
    _TRANSITION.clear()
    _CAPTURED.clear()
    _FIRED[0] = False


def fire() -> None:
    """Run the pre-run hooks, synchronize, emit the transition, synchronize."""
    if _FIRED[0]:
        return
    _FIRED[0] = True

    for _order, name, fn in _HOOKS:
        logger.debug("MLPerf boundary: running pre-run hook %s", name)
        fn()

    _synchronize()

    for fn in _TRANSITION:
        fn()

    # Hold every rank until the records exist, so no rank can begin opening
    # the dataset while rank 0 is still writing run_start.
    _synchronize()


def _synchronize() -> None:
    import torch
    import torch.distributed

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def install() -> None:
    """Wrap the three Megatron entry points. Idempotent."""
    if _ORIGINALS:
        return

    import megatron.training as megatron_training_pkg
    import megatron.training.training as mt

    _ORIGINALS["pretrain"] = getattr(megatron_training_pkg, "pretrain", None)
    _ORIGINALS["setup_model_and_optimizer"] = mt.setup_model_and_optimizer
    _ORIGINALS["build_train_valid_test_data_iterators"] = mt.build_train_valid_test_data_iterators

    # forward_step_func is a pretrain() argument and is reachable nowhere else
    # before train(). The trainer imports pretrain inside its run method, after
    # this phase, so rebinding the package attribute is picked up.
    if _ORIGINALS["pretrain"] is not None:
        original_pretrain = _ORIGINALS["pretrain"]

        def _capturing_pretrain(*args, **kwargs):
            if len(args) > 3:
                _CAPTURED["forward_step_func"] = args[3]
            elif "forward_step_func" in kwargs:
                _CAPTURED["forward_step_func"] = kwargs["forward_step_func"]
            return original_pretrain(*args, **kwargs)

        megatron_training_pkg.pretrain = _capturing_pretrain

    original_setup = _ORIGINALS["setup_model_and_optimizer"]

    def _capturing_setup_model_and_optimizer(*args, **kwargs):
        model, optimizer, opt_param_scheduler = original_setup(*args, **kwargs)
        _CAPTURED["model"] = model
        _CAPTURED["optimizer"] = optimizer
        _CAPTURED["opt_param_scheduler"] = opt_param_scheduler
        return model, optimizer, opt_param_scheduler

    mt.setup_model_and_optimizer = _capturing_setup_model_and_optimizer

    original_build = _ORIGINALS["build_train_valid_test_data_iterators"]

    def _boundary_build_data_iterators(*args, **kwargs):
        # Virtual pipelining calls this once per stage; only the first call is
        # the boundary, and fire() is idempotent regardless.
        fire()
        return original_build(*args, **kwargs)

    _boundary_build_data_iterators._primus_mlperf_boundary = True
    mt.build_train_valid_test_data_iterators = _boundary_build_data_iterators
