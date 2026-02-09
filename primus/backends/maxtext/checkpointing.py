###############################################################################
# Copyright 2023–2025 Google LLC. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Any

import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as emergency_replicator_checkpoint_manager
from etils import epath
from flax.training import train_state
from MaxText import max_logging
from MaxText.checkpointing import (
    _load_full_state_from_path,
    _replica_devices,
    _restore_grain_iterator,
    load_params_from_path,
)
from MaxText.input_pipeline.input_pipeline_interface import PlaceHolderDataIterator
from MaxText.multihost_dataloading import MultiHostDataLoadIterator

Composite = ocp.args.Composite
EmergencyCheckpointManager = emergency_checkpoint_manager.CheckpointManager
EmergencyReplicatorCheckpointManager = emergency_replicator_checkpoint_manager.ReplicatorCheckpointManager


def load_state_if_possible(
    checkpoint_manager: ocp.CheckpointManager | None,
    data_iterator: MultiHostDataLoadIterator | list[MultiHostDataLoadIterator] | None,
    load_parameters_from_path: str,
    load_full_state_from_path: str,
    checkpoint_storage_concurrent_gb: int,
    abstract_unboxed_pre_state: train_state.TrainState,
    enable_single_replica_ckpt_restoring: bool | None = False,
    dataset_type: str | None = "tfds",
    step: int = -1,  # -1 means latest
    use_ocdbt=True,
    use_zarr3=True,
    enable_orbax_v1=False,
    checkpoint_conversion_fn=None,
    source_checkpoint_layout="orbax",
    expansion_factor_real_data: int = -1,
):
    """Loads TrainState as possible from the inputs.

    Args:
    checkpoint_manager: if the checkpoint_manager has a valid checkpoint, return
        that TrainState. This enables a full reload of a run in progress.
    load_parameters_from_path: if there is no checkpoint in the checkpoint
        manager, load parameters from a parameter only checkpoint at this path.
    load_full_state_from_path: if there is no checkpoint in the checkpoint
        manager, load full state from a full state checkpoint at this path.
    abstract_unboxed_pre_state: an unboxed, abstract TrainState that Orbax
        matches type against.
    enable_single_replica_ckpt_restoring: bool flag for restoring checkpoitn
        with SingleReplicaArrayHandler
    checkpoint_storage_concurrent_gb: concurrent GB for checkpoint byte I/O.
    enable_orbax_v1: bool flag for enabling Orbax v1.
    checkpoint_conversion_fn: function for converting checkpoint to Orbax v1.
    source_checkpoint_layout: Optional checkpoint context to use for loading,
    provided in string format with the default being "orbax".

    Returns:
    A tuple of (train_state, train_state_params) where full_train_state captures
        a full reload and train_state_params just the params for a partial reload.
        At most one will be non-None. Both can be None if neither checkpoint is
        set.
    """

    if checkpoint_manager is not None:
        max_logging.log("checkpoint manager exists so trying to load this run's existing checkpoint")

        step = checkpoint_manager.latest_step() if step < 0 else step
        if step is not None:
            max_logging.log(f"restoring from this run's directory step {step}")

            def map_to_pspec(data):
                if not enable_single_replica_ckpt_restoring:
                    return ocp.type_handlers.ArrayRestoreArgs(sharding=data.sharding)
                pspec = data.sharding.spec
                mesh = data.sharding.mesh
                replica_axis_index = 0
                replica_devices = _replica_devices(mesh.devices, replica_axis_index)
                replica_mesh = jax.sharding.Mesh(replica_devices, mesh.axis_names)
                single_replica_sharding = jax.sharding.NamedSharding(replica_mesh, pspec)

                return ocp.type_handlers.SingleReplicaArrayRestoreArgs(
                    sharding=jax.sharding.NamedSharding(mesh, pspec),
                    single_replica_sharding=single_replica_sharding,
                    global_shape=data.shape,
                    dtype=data.dtype,
                )

            # Cache the original ArrayHandler before potentially overriding it.
            # This is the same handler used when enable_single_replica_ckpt_restoring=False.
            original_array_handler = ocp.type_handlers.get_type_handler(jax.Array)

            # Register SingleReplicaArrayHandler globally for restore (if enabled)
            if enable_single_replica_ckpt_restoring:
                single_replica_handler = ocp.type_handlers.SingleReplicaArrayHandler(
                    replica_axis_index=0,
                    broadcast_memory_limit_bytes=1024 * 1024 * 1000,  # 1000 MB limit
                )
                ocp.type_handlers.register_type_handler(jax.Array, single_replica_handler, override=True)

            restore_args = jax.tree_util.tree_map(map_to_pspec, abstract_unboxed_pre_state)
            checkpoint_args = ocp.args.PyTreeRestore(
                item=abstract_unboxed_pre_state, restore_args=restore_args
            )

            def _restore_original_array_handler():
                """Restore the original ArrayHandler after SingleReplicaArrayHandler restore.

                This is critical because SingleReplicaArrayHandler is designed for restore only.
                Using it for saves will cause missing array_metadatas files and checkpoint failures.
                We restore the EXACT handler that was in place before, not a new instance.
                """
                if enable_single_replica_ckpt_restoring:
                    max_logging.log(
                        "Restoring original ArrayHandler after SingleReplicaArrayHandler restore..."
                    )
                    # Re-register the original handler that was cached before the override
                    ocp.type_handlers.register_type_handler(jax.Array, original_array_handler, override=True)
                    max_logging.log("Original ArrayHandler restored successfully.")

            match (checkpoint_manager, dataset_type, data_iterator):
                # Case 1: Matches if 'checkpoint_manager' is an instance of either EmergencyCheckpointManager
                # or EmergencyReplicatorCheckpointManager. The '_' indicates that 'dataset_type' and
                # 'data_iterator' can be any value and aren't used in this pattern.
                case (checkpoint_manager, _, _) if isinstance(
                    checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)
                ):
                    result = (
                        checkpoint_manager.restore(step, args=Composite(state=checkpoint_args)).state,
                        None,
                    )
                    _restore_original_array_handler()
                    return result
                # Case 2: Matches if dataset type is "grain" and the data iterator is not a
                # PlaceHolderDataIterator and a specific checkpoint file exists for the iterator
                case (
                    checkpoint_manager,
                    dataset_type,
                    data_iterator,
                ) if (
                    dataset_type == "grain"
                    and data_iterator
                    and not isinstance(data_iterator, PlaceHolderDataIterator)
                    and (checkpoint_manager.directory / str(step) / "iter").exists()
                ):
                    result = _restore_grain_iterator(
                        checkpoint_manager, step, data_iterator, checkpoint_args, expansion_factor_real_data
                    )
                    _restore_original_array_handler()
                    return result
                # Case 3: Default/Fallback case.
                # This case acts as a wildcard ('_') and matches if none of the preceding cases were met.
                case _:
                    result = (checkpoint_manager.restore(step, args=Composite(items=checkpoint_args)), None)
                    _restore_original_array_handler()
                    return result

    if load_parameters_from_path != "":
        restored_params = load_params_from_path(
            load_parameters_from_path,
            abstract_unboxed_pre_state.params,
            checkpoint_storage_concurrent_gb,
            use_ocdbt=use_ocdbt,
            use_zarr3=use_zarr3,
        )
        return None, restored_params
    elif load_full_state_from_path != "":
        max_logging.log(f"Loading full state from path: {load_full_state_from_path}")
        restored_state = _load_full_state_from_path(
            path=load_full_state_from_path,
            abstract_unboxed_pre_state=abstract_unboxed_pre_state,
            enable_orbax_v1=enable_orbax_v1,
            checkpoint_conversion_fn=checkpoint_conversion_fn,
            source_checkpoint_layout=source_checkpoint_layout,
        )
        return {"items": restored_state}, None
    else:
        max_logging.log("No existing checkpoints found, not restoring checkpoint.")
        return None, None


def create_orbax_checkpoint_manager(
    checkpoint_dir: str,
    enable_checkpointing: bool,
    use_async: bool,
    save_interval_steps: int,
    dataset_type: None | str = "tfds",
    orbax_logger: Any = None,  # pytype: disable=attribute-error
    use_ocdbt: bool = True,
    use_zarr3: bool = True,
    max_to_keep: int = 5,
):
    """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
    if not enable_checkpointing:
        max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
        return None

    max_logging.log(f"Creating checkpoint manager with ocdbt={use_ocdbt} and zarr3={use_zarr3}")

    # Base configuration for all dataset types
    item_names = ("items",)
    # we need to use ocdbt and zarr3 to control max file size in the checkpoint
    item_handlers = {"items": ocp.PyTreeCheckpointHandler(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3)}

    if dataset_type == "grain":
        item_names += ("iter",)
        item_handlers["iter"] = ocp.GrainCheckpointHandler()

    # local storage checkpoint needs parent directory created
    p = epath.Path(checkpoint_dir)
    p.mkdir(exist_ok=True, parents=True)
    manager = ocp.CheckpointManager(
        p,
        item_names=item_names,
        item_handlers=item_handlers,
        options=ocp.CheckpointManagerOptions(
            create=True,
            save_interval_steps=save_interval_steps,
            enable_async_checkpointing=use_async,
            max_to_keep=max_to_keep,
        ),
        logger=orbax_logger,
    )

    max_logging.log("Checkpoint manager created!")
    return manager
