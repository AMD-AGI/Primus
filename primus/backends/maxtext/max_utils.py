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
import orbax.checkpoint as ocp
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import initialization

from MaxText import max_logging
from MaxText.max_utils import _retrieve_jax_init_info, is_gpu_backend, is_cpu_backend, get_coordinator_ip_address


def maybe_initialize_jax_distributed_system(raw_keys):
    """The best recipe to initialize the Jax Distributed System has varied over time. We keep a layer of
    indirection in MaxText to avoid breaking the call sites unnecessarily.

    Currently jax.distributed.initialize() fully works as expected!

    For CPUs, we call jax.distributed.initialize() explicitly, with the specified arguments.
    """
    if raw_keys["skip_jax_distributed_system"]:
        max_logging.log("Skipping jax distributed system due to skip_jax_distributed_system=True flag.")
        return
    if raw_keys["enable_single_controller"]:
        max_logging.log("Skipping jax distributed system since its not needed for single controller.")
        return
    if jax.distributed.is_initialized():
        max_logging.log("Jax distributed system is already initialized.")
        return
    if raw_keys["inference_benchmark_test"]:
        # Disable initialization for inference benmark test.
        return
    if raw_keys["compile_topology"]:
        # Don't initialize jax distributed with AOT compilation
        return
    if is_gpu_backend(raw_keys):
        max_logging.log("Attempting to initialize the jax distributed system for GPU backend...")
        initialize_jax_for_gpu(raw_keys)
        max_logging.log("Jax distributed system initialized on GPU!")
    elif is_cpu_backend(raw_keys):
        max_logging.log("Attempting to initialize the jax distributed system for CPU backend...")
        initialize_jax_for_cpu(raw_keys)
        max_logging.log("Jax distributed system initialized on CPUs!")
    elif raw_keys["enable_multi_tier_checkpointing"]:
        max_logging.log("Attempting to initialize the jax distributed system for multi-tier " "checkpointing...")
        initialization.initialize_multi_tier_checkpointing(
            local_checkpoint_directory=raw_keys["local_checkpoint_directory"],
            backup_interval_minutes=raw_keys["multi_tier_checkpointing_backup_interval_minutes"],
            run_name=raw_keys["run_name"],
            jax_initialization_timeout_seconds=raw_keys["jax_distributed_initialization_timeout"],
            data_parallelism=raw_keys["mtc_data_parallelism"],
            )
        max_logging.log("Jax distributed system initialized for multi-tier checkpointing!")
    elif (raw_keys["enable_checkpointing"] and raw_keys["compile_topology_num_slices"] == -1) or raw_keys[
    "hardware"
    ] == "gpu_multiprocess":
        max_logging.log("Attempting to initialize the jax distributed system...")
        if not raw_keys["enable_emergency_checkpoint"]:
            jax.distributed.initialize(
                initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
                heartbeat_timeout_seconds=raw_keys["jax_distributed_heartbeat_timeout_seconds"],
            )
        else:
            if raw_keys["hardware"] == "gpu_multiprocess":
                max_logging.log("Initializing jax distribtued to support local checkpointing with" " GPUs...")
                jax.distributed.initialize(
                    initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
                    heartbeat_timeout_seconds=raw_keys["jax_distributed_heartbeat_timeout_seconds"],
                )
                ocp.multihost.initialize_runtime_to_distributed_ids()
                ocp.multihost.initialize_distributed_to_device_ids()
            else:
                initialize_jax_for_tpu_with_emergency_checkpointing(raw_keys)
        max_logging.log("Jax distributed system initialized!")


def initialize_jax_for_gpu(raw_keys):
    """Jax distributed initialize for GPUs."""
    if os.environ.get("JAX_COORDINATOR_IP") is not None:
        coordinator_ip = str(os.getenv("JAX_COORDINATOR_IP"))
        coordinator_port = str(os.getenv("JAX_COORDINATOR_PORT"))
        jax.distributed.initialize(
            coordinator_address=f"{coordinator_ip}:{coordinator_port}",
            num_processes=int(os.getenv("NNODES")),
            process_id=int(os.getenv("NODE_RANK")),
            initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
            heartbeat_timeout_seconds=raw_keys["jax_distributed_heartbeat_timeout_seconds"],
        )
        max_logging.log(f"JAX global devices: {jax.devices()}")


def initialize_jax_for_cpu(raw_keys):
    """Jax distributed initialize for CPUs. Includes retries until the coordinator is ready."""
    coordinator_ip_address = get_coordinator_ip_address()
    coordinator_address = coordinator_ip_address + ":1234"  # JAX coordinator port used in XPK
    # Env variables to be set in XPK or otherwise
    job_index = int(os.environ.get("JOB_INDEX"))
    job_completion_index = int(os.environ.get("JOB_COMPLETION_INDEX"))
    processes_in_job = int(os.environ.get("PROCESSES_IN_JOB"))
    pid = job_index * processes_in_job + job_completion_index
    max_logging.log(f" Jax process id is {pid} ")
    # Explicit initialize is needed only for CPUs
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        process_id=pid,
        num_processes=int(os.environ.get("JAX_PROCESS_COUNT")),
        initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
        heartbeat_timeout_seconds=raw_keys["jax_distributed_heartbeat_timeout_seconds"],
    )


def initialize_jax_for_tpu_with_emergency_checkpointing(raw_keys):
    """Initialize JAX distributed runtime for TPUs when emergency checkpointing is used.
    The information required to initialize JAX distributed runtime will be written by GKE to
    the local checkpoint directory. This function retrieves that information and initializes
    JAX distributed runtime.
    """
    process_id, coordinator_address = _retrieve_jax_init_info(raw_keys)

    if process_id != "" and coordinator_address != "":
        max_logging.log(
            f"Using {process_id} as the process_id and {coordinator_address} as the"
            " coordinator_address to initialize JAX distributed runtime..."
        )
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            process_id=int(process_id),
            initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
            heartbeat_timeout_seconds=raw_keys["jax_distributed_heartbeat_timeout_seconds"],
        )

        ocp.multihost.initialize_runtime_to_distributed_ids()
        ocp.multihost.initialize_distributed_to_device_ids()


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

    wandb.init(project=wandb_project, name=wandb_exp_name, dir=wandb_save_dir, config=dict(config.get_keys()))
    max_logging.log(f"WandB logging enabled: {wandb_save_dir=}, {wandb_project=}, {wandb_exp_name=}")
    return wandb


def close_wandb_writer(wandb_writer):
    if jax.process_index() == 0 and wandb_writer is not None:
        wandb_writer.finish()
