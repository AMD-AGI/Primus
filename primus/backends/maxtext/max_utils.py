###############################################################################
# Copyright 2023–2025 Google LLC. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
import os
import socket

import jax
from MaxText import max_logging


def print_system_information():
    """Print system information of the current environment.
    Note that this will initialize the JAX backend."""
    max_logging.log(f"System Information: Jax Version: {jax.__version__}")
    max_logging.log(f"System Information: Jaxlib Version: {jax.lib.__version__}")
    max_logging.log(f"System Information: Jax Backend: {jax.extend.backend.get_backend().platform_version}")

    devices = jax.devices()
    max_logging.log(f"System Information: Number of devices: {len(devices)}, jax path {jax.__file__}")
    for i, device in enumerate(devices):
        if device.local_hardware_id is not None:
            max_logging.log(
                f"System Information: Device {i}: {device.id} "
                f"(Local id: {device.local_hardware_id}, Process index: {device.process_index})"
            )


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
    # Save to JSON file
    device_info_path = os.path.join(config.base_output_directory, "device_info.json")
    with open(device_info_path, "w") as f:
        json.dump(device_info, f, indent=4)


def initialize_wandb_writer(config):
    if jax.process_index() != 0 or not config.enable_wandb:
        return None

    def safe_get_config(config, key, default=None):
        try:
            return getattr(config, key)
        except KeyError:
            return default

    import wandb

    if safe_get_config(config, "wandb_save_dir") is None or config.wandb_save_dir == "":
        wandb_save_dir = os.path.join(config.base_output_directory, "wandb")
    else:
        wandb_save_dir = config.wandb_save_dir

    if safe_get_config(config, "wandb_project") is None or config.wandb_project == "":
        wandb_project = os.getenv("WANDB_PROJECT", "Primus-MaxText-Pretrain")
    else:
        wandb_project = config.wandb_project
    if safe_get_config(config, "wandb_exp_name") is None or config.wandb_exp_name == "":
        wandb_exp_name = config.run_name
    else:
        wandb_exp_name = config.wandb_exp_name

    if config.enable_wandb and "WANDB_API_KEY" not in os.environ:
        max_logging.log(
            "The environment variable WANDB_API_KEY is not set. Please set it or login wandb before proceeding"
        )
        return None

    os.makedirs(wandb_save_dir, exist_ok=True)

    wandb.init(project=wandb_project, name=wandb_exp_name, dir=wandb_save_dir, config=config.get_keys())
    max_logging.log(f"WandB logging enabled: {wandb_save_dir=}, {wandb_project=}, {wandb_exp_name=}")
    return wandb


def close_wandb_writer(wandb_writer):
    if jax.process_index() == 0 and wandb_writer is not None:
        wandb_writer.finish()
