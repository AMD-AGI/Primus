###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Forward step for WAN (2.1 / 2.2) video diffusion training.

Mirrors ``flux_forward_step_func`` but for 5D video latents, and returns the
scheduler's training target and per-timestep weight alongside the prediction so
the trainer can apply the weighted flow-matching loss.

Random draws happen in a fixed order -- noise first, then timesteps -- because
both come from the ambient RNG, so any reordering changes the whole stream and
with it every loss value.
"""

from typing import Optional, Tuple

import torch


def wan_forward_step_func(
    data_iterator,
    model,
    scheduler,
    timestep_window: Optional[Tuple[float, float]] = None,
    boundary_timestep: Optional[float] = None,
    loss_weighting: str = "diffsynth",
):
    """Run one WAN training forward pass.

    Args:
        data_iterator: Iterator yielding ``EncodedWanTaskEncoder`` batches.
        model: ``Wan`` or ``Wan2_2`` instance.
        scheduler: ``WanFlowMatchScheduler``.
        timestep_window: Optional ``(lo, hi)`` in ``[0, 1]`` restricting sampled
            timesteps to a sub-window, for training one WAN 2.2 expert per job.
        boundary_timestep: Optional WAN 2.2 routing boundary in post-shift
            timestep space. Ignored by single-transformer ``Wan``.
        loss_weighting: ``"diffsynth"`` for the scheduler's per-timestep weight,
            ``"uniform"`` for an unweighted objective.

    Returns:
        Tuple ``(noise_pred, clean_latents, noise, target, weight, loss_mask,
        metrics, is_validation)``.
    """
    from megatron.core.parallel_state import (
        get_pipeline_model_parallel_rank,
        get_pipeline_model_parallel_world_size,
        get_tensor_model_parallel_world_size,
    )

    pp_size = get_pipeline_model_parallel_world_size()
    if pp_size > 1:
        pp_rank = get_pipeline_model_parallel_rank()
        if pp_rank not in (0, pp_size - 1):
            # WAN enforces PP=1, but stay harmless on PP middle ranks.
            dummy = torch.tensor(0.0, device="cuda", requires_grad=True)
            return dummy, dummy, dummy, dummy, dummy, None, {}, False

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size != 1:
        raise RuntimeError(
            f"WAN requires tensor_model_parallel_size=1, got tp_size={tp_size}. "
            "WanConfig.validate() enforces this; this is a defense-in-depth check."
        )

    if model.config.bf16:
        compute_dtype = torch.bfloat16
    elif model.config.fp16:
        compute_dtype = torch.float16
    else:
        compute_dtype = model.config.params_dtype

    assert data_iterator is not None, (
        "WAN forward step requires a data_iterator (TP=1 path). "
        "Make sure the dataset provider sets is_distributed=True."
    )

    batch = next(data_iterator)
    if not isinstance(batch, dict):
        raise TypeError(f"[WanForwardStep] Expected batch to be a dict, got {type(batch)}.")

    required_keys = ("latents", "encoder_hidden_states")
    missing = [key for key in required_keys if key not in batch]
    if missing:
        raise KeyError(f"[WanForwardStep] Batch missing required keys: {missing}. Got: {list(batch.keys())}")

    for key in list(batch.keys()):
        if isinstance(batch[key], torch.Tensor):
            if batch[key].is_floating_point():
                batch[key] = batch[key].to(dtype=compute_dtype, device="cuda", non_blocking=True)
            elif not batch[key].is_cuda:
                batch[key] = batch[key].cuda(non_blocking=True)

    latents = batch["latents"]
    encoder_hidden_states = batch["encoder_hidden_states"]
    loss_mask = batch.get("loss_mask")
    is_validation = "timestep" in batch and batch.get("validation", False)

    with torch.no_grad():
        if "noise" in batch and isinstance(batch["noise"], torch.Tensor):
            noise = batch["noise"].to(dtype=compute_dtype, device="cuda", non_blocking=True)
        else:
            noise = torch.randn_like(latents, dtype=compute_dtype)

        if is_validation and "timestep" in batch:
            timesteps = batch["timestep"].to(device="cuda", dtype=torch.long)
        else:
            timesteps = scheduler.sample_training_timesteps(
                batch_size=latents.shape[0],
                device=latents.device,
                timestep_window=timestep_window,
            )

        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        target = scheduler.training_target(latents, noise, timesteps)

        # Condition on the inference-aligned post-shift timestep, which is also
        # the axis the weight table is indexed on.
        model_timesteps = scheduler.conditioning_timestep(timesteps, device=latents.device)

        if loss_weighting == "uniform":
            weight = torch.ones(latents.shape[0], device=latents.device, dtype=latents.dtype)
        else:
            weight = scheduler.training_weight(model_timesteps).to(device=latents.device, dtype=latents.dtype)

    # The routing boundary is compared against the conditioned timestep, so it
    # has to live in the same post-shift space.
    boundary_tensor: Optional[torch.Tensor] = None
    if boundary_timestep is not None:
        boundary_tensor = torch.full_like(model_timesteps, fill_value=float(boundary_timestep))

    with torch.amp.autocast("cuda", enabled=True, dtype=compute_dtype):
        noise_pred = model(
            hidden_states=noisy_latents,
            timestep=model_timesteps,
            encoder_hidden_states=encoder_hidden_states,
            boundary_timestep=boundary_tensor,
        )

    metrics = {
        "batch_size": latents.shape[0],
        "latent_channels": latents.shape[1],
        "latent_t": latents.shape[2],
        "latent_h": latents.shape[3],
        "latent_w": latents.shape[4],
        "text_seq_len": encoder_hidden_states.shape[1],
        "avg_timestep": timesteps.float().mean(),
    }

    return (
        noise_pred,
        latents,
        noise,
        target,
        weight,
        loss_mask,
        metrics,
        is_validation,
    )


__all__ = ["wan_forward_step_func"]
