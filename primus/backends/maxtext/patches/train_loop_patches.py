###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText ``pre_train/train.py`` patches (ROCm/maxtext release/v26.10).

- ``maxtext.train.device_info`` (1c275c1e2): log every device in
  ``print_system_information`` and write ``<base_output_directory>/device_info.json``
  from ``train.initialize``.
- ``maxtext.train.sync_before_loop`` (dd8204b2d): barrier across hosts right
  before the training loop, so stragglers from init do not time out the first
  collective ("collective operation timeout" / "stop sending heartbeats").
- ``maxtext.train.sync_each_step`` (8d1d43c03): block on the updated train state
  after every step so async dispatch cannot race ahead.

``device_info`` source-patches ``train.initialize`` and therefore runs before
``maxtext.train`` (priority 50), which wraps the same function.
"""

import functools
import json
import os
import socket

from primus.backends.maxtext.patches._source_patch_utils import (
    any_in_source,
    rewrite_source,
)
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0


def _import_train(prefix):
    try:
        from maxtext.trainers.pre_train import train
        from maxtext.utils import max_utils
    except ImportError as e:
        warning_rank_0(f"{prefix} maxtext.trainers.pre_train.train not importable; skipped: {e}")
        return None, None
    return train, max_utils


def _make_save_device_information(jax):
    def save_device_information(config):
        """Convert device information to JSON format."""
        devices = jax.devices()
        device_info = {"hostname": socket.gethostname(), "devices": []}

        for device in devices:
            if device.local_hardware_id is not None:
                info = {
                    "id": device.id,
                    "local_hardware_id": device.local_hardware_id,
                    "process_index": device.process_index,
                    "device_kind": device.device_kind,
                    "platform_version": jax.extend.backend.get_backend().platform_version,
                }
                device_info["devices"].append(info)
        device_info_path = os.path.join(config.base_output_directory, "device_info.json")
        with open(device_info_path, "w") as f:
            json.dump(device_info, f, indent=4)

    return save_device_information


@register_patch(
    patch_id="maxtext.train.device_info",
    backend="maxtext",
    phase="setup",
    description="Log every device in print_system_information and save device_info.json from train.initialize",
    condition=lambda ctx: True,
    priority=40,
)
def patch_device_info(ctx: PatchContext) -> None:
    del ctx
    prefix = "[Patch:maxtext.train.device_info]"
    train, max_utils = _import_train(prefix)
    if train is None:
        return
    if hasattr(max_utils, "save_device_information"):
        log_rank_0(f"{prefix} MaxText already saves device information; not patched.")
        return

    import jax

    orig_print = max_utils.print_system_information
    max_logging = max_utils.max_logging

    @functools.wraps(orig_print)
    def print_system_information(*args, **kwargs):
        orig_print(*args, **kwargs)
        devices = jax.devices()
        max_logging.log(f"System Information: Number of devices: {len(devices)}, jax path {jax.__file__}")
        for i, device in enumerate(devices):
            if device.local_hardware_id is not None:
                max_logging.log(
                    f"System Information: Device {i}: {device.id} "
                    f"(Local id: {device.local_hardware_id}, Process index: {device.process_index})"
                )

    max_utils.print_system_information = print_system_information
    max_utils.save_device_information = _make_save_device_information(jax)
    rewrite_source(
        train,
        "initialize",
        [
            (
                "  train_utils.validate_train_config(config)\n",
                "  train_utils.validate_train_config(config)\n"
                "  max_utils.save_device_information(config)\n",
            )
        ],
    )
    log_rank_0(f"{prefix} Device logging and device_info.json enabled.")


@register_patch(
    patch_id="maxtext.train.sync_before_loop",
    backend="maxtext",
    phase="setup",
    description="Synchronize all hosts (sync_global_devices) right before the MaxText training loop",
    condition=lambda ctx: True,
)
def patch_sync_before_loop(ctx: PatchContext) -> None:
    del ctx
    prefix = "[Patch:maxtext.train.sync_before_loop]"
    train, _ = _import_train(prefix)
    if train is None:
        return
    if any_in_source(train, "train_loop", ["sync_before_training_loop"]):
        log_rank_0(f"{prefix} MaxText already synchronizes hosts before the loop; not patched.")
        return

    rewrite_source(
        train,
        "train_loop",
        [
            (
                "  _job_completed_gracefully = False\n",
                "  # Synchronize all hosts before entering the training loop.\n"
                "  # Without this barrier, timing variance during initialization (JIT compilation,\n"
                "  # profiler/logger setup, etc.) causes hosts to enter the training loop at different\n"
                "  # times. The first collective operation (data sharding in load_next_batch) then\n"
                '  # times out waiting for straggler hosts, resulting in "collective operation timeout"\n'
                '  # or "stop sending heartbeats" errors.\n'
                '  max_logging.log("====== BARRIER: Synchronizing hosts before training loop ======")\n'
                '  jax.experimental.multihost_utils.sync_global_devices("sync_before_training_loop")\n'
                '  max_logging.log("====== BARRIER PASSED: Starting training loop ======")\n'
                "\n"
                "  _job_completed_gracefully = False\n",
            )
        ],
    )
    log_rank_0(f"{prefix} Host barrier inserted before the training loop.")


@register_patch(
    patch_id="maxtext.train.sync_each_step",
    backend="maxtext",
    phase="setup",
    description="Block on the updated train state after every MaxText training step",
    condition=lambda ctx: True,
)
def patch_sync_each_step(ctx: PatchContext) -> None:
    del ctx
    prefix = "[Patch:maxtext.train.sync_each_step]"
    train, _ = _import_train(prefix)
    if train is None:
        return
    if any_in_source(train, "train_loop", ['jax.block_until_ready(jax_device_state["state"])']):
        log_rank_0(f"{prefix} MaxText already blocks on the state every step; not patched.")
        return

    import jax

    orig = train.training_loop_iteration

    @functools.wraps(orig)
    def training_loop_iteration(jax_device_state, *args, **kwargs):
        metrics = orig(jax_device_state, *args, **kwargs)
        jax.block_until_ready(jax_device_state["state"])
        return metrics

    train.training_loop_iteration = training_loop_iteration
    log_rank_0(f"{prefix} train_loop now blocks on the state after every step.")
