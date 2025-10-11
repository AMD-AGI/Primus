###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import List, Optional, Union

from megatron.core.transformer.moe.moe_utils import (
    clear_aux_losses_tracker,
    get_moe_layer_wise_logging_tracker,
    reduce_aux_losses_tracker_across_ranks,
)


def track_moe_metrics(
    loss_scale: float,
    iteration: int,
    writer,
    wandb_writer=None,
    mlflow_writer=None,
    total_loss_dict=None,
    per_layer_logging=False,
    force_initialize: bool = False,
    track_names: Optional[List[str]] = None,
    num_layers: Optional[int] = None,
    moe_layer_freq: Optional[Union[int, List[int]]] = None,
):
    """Track the MoE metrics for logging."""
    # Aux loss logging
    tracker = get_moe_layer_wise_logging_tracker()
    # Initialize the tracker if force_initialize is True
    if force_initialize:
        if track_names is not None:
            for key in track_names:
                if key not in tracker:
                    tracker[key] = {}
                    tracker[key]["values"] = torch.zeros(num_layers, device="cuda")
                    tracker[key]["reduce_group"] = None
                    tracker[key]["avg_group"] = None
    reduce_aux_losses_tracker_across_ranks(track_names)

    # Get number of MoE layers
    if moe_layer_freq is None:
        num_moe_layers = num_layers
    elif isinstance(moe_layer_freq, int):
        assert isinstance(num_layers, int)
        moe_layer_pattern = [1 if (i % moe_layer_freq == 0) else 0 for i in range(num_layers)]
        num_moe_layers = sum(moe_layer_pattern)
    elif isinstance(moe_layer_freq, list):
        num_moe_layers = sum(moe_layer_freq)
    else:
        raise ValueError(f"Invalid moe_layer_freq: {moe_layer_freq}")

    if writer is not None:
        aux_losses = {k: v["values"].float() * loss_scale for k, v in tracker.items()}
        for name, loss_list in aux_losses.items():
            if total_loss_dict is not None:
                if name not in total_loss_dict:
                    total_loss_dict[name] = loss_list.sum() / num_moe_layers
                else:
                    total_loss_dict[name] += loss_list.sum() / num_moe_layers

            # currently when using add_scalars,
            # torch.utils.add_scalars makes each timer its own run, which
            # polutes the runs list, so we just add each as a scalar
            writer.add_scalar(name, loss_list.sum() / num_moe_layers, iteration)
            if per_layer_logging:
                for i, loss in enumerate(loss_list.tolist()):
                    writer.add_scalar(f"moe/{name}_layer_{i}", loss, iteration)

            # W&B logging lacks support for logging multiple scalars simultaneously.
            # As a workaround, we log each scalar individually first, then we can create
            # a custom panel to manually group them to a single plot.
            if wandb_writer:
                wandb_writer.log({f"{name}": loss_list.sum() / num_moe_layers}, iteration)
                if per_layer_logging:
                    wandb_writer.log(
                        {f"moe/{name}_layer_{i}": loss for i, loss in enumerate(loss_list.tolist())},
                        iteration,
                    )
            if mlflow_writer:
                mlflow_writer.log_metric(f"{name}", loss_list.mean(), iteration)
                if per_layer_logging:
                    mlflow_writer.log_metrics(
                        {f"moe/{name}_layer_{i}": loss for i, loss in enumerate(loss_list.tolist())},
                        iteration,
                    )

    clear_aux_losses_tracker()
