# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Adapted from NeMo's Flux training architecture.

"""
Forward step functions for diffusion model training.

This module provides forward step implementations for different
diffusion models, handling the training loop logic.

Supported data formats (framework-standard keys):
    - Pre-encoded: latents, prompt_embeds, pooled_prompt_embeds, text_ids (optional)
    - Raw: images, txt - encodes on-the-fly

Architecture follows functional composition for clarity and testability.
"""

import hashlib
import json
import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import torch

from primus.backends.megatron.core.models.diffusion.flux.utils import (
    generate_image_position_ids,
    generate_text_position_ids,
    pack_latents,
    unpack_latents,
)
from primus.backends.megatron.training.diffusion.noise_utils import (
    apply_flow_matching_noise,
)
from primus.backends.megatron.training.diffusion.timestep_sampling import (
    LogitNormalSampler,
)

logger = logging.getLogger(__name__)
_EMITTED_MODEL_WEIGHT_ITERATIONS: set[int] = set()
_MODEL_WEIGHT_AUDIT_CONTEXT_UNSET = object()


def _emit_batch_fingerprint(batch: dict, step_count: int, *, is_training: bool = True) -> None:
    """Emit a rank-local sample-key digest for explicit continuity audits."""
    if os.getenv("PRIMUS_AUDIT_BATCH_FINGERPRINTS") != "1":
        return
    if not is_training:
        return
    if os.getenv("PRIMUS_SYNTHETIC_WARMUP_ACTIVE") == "1":
        return

    fingerprint = batch.get("_audit_sample_key_sha256")
    sample_count = batch.get("_audit_sample_count")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
        or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        raise RuntimeError(
            "Batch-fingerprint audit was requested but the Energon batch has no "
            "valid sample-key fingerprint"
        )

    from megatron.core import parallel_state

    global_rank = (
        torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.getenv("RANK", "-1"))
    )
    payload = {
        "global_rank": global_rank,
        "data_parallel_rank": parallel_state.get_data_parallel_rank(),
        "step": int(step_count),
        "sample_count": sample_count,
        "sample_keys_sha256": fingerprint,
    }
    # Emit through logging, not print: fd 1 does not survive to the run log on
    # this launch path, so a bare print leaves the audit reporting zero markers
    # on a healthy run. Logging and fd 2 both survive.
    logger.info("PRIMUS_BATCH_FINGERPRINT=%s", json.dumps(payload, sort_keys=True))


def _parse_model_weight_steps(value: str) -> set[int]:
    """Parse completed training iterations requested for weight auditing."""
    steps = set()
    for token in value.split(","):
        token = token.strip()
        if not token or not token.isdigit() or int(token) <= 0:
            raise RuntimeError(
                "PRIMUS_AUDIT_MODEL_WEIGHT_STEPS must be a comma-separated " "list of positive integers"
            )
        step = int(token)
        if step in steps:
            raise RuntimeError("PRIMUS_AUDIT_MODEL_WEIGHT_STEPS contains duplicate step " f"{step}")
        steps.add(step)
    if not steps:
        raise RuntimeError("PRIMUS_AUDIT_MODEL_WEIGHT_STEPS is empty")
    return steps


def _sample_model_weights(model, sample_size: int) -> dict:
    """Return deterministic, low-overhead per-parameter weight samples."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    metadata = []
    sampled_tensors = []
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        tensor = parameter.detach().reshape(-1)
        if tensor.numel() == 0:
            continue
        count = min(sample_size, tensor.numel())
        indices = torch.arange(count, device=tensor.device, dtype=torch.int64) * tensor.numel() // count
        sampled = tensor.index_select(0, indices).to(dtype=torch.float32)
        metadata.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "numel": parameter.numel(),
                "sample_count": count,
                "requires_grad": parameter.requires_grad,
            }
        )
        sampled_tensors.append(sampled)

    if not sampled_tensors:
        raise RuntimeError("model-weight audit found no parameters")
    devices = {tensor.device for tensor in sampled_tensors}
    if len(devices) != 1:
        raise RuntimeError(
            "model-weight audit requires all sampled parameters on one device, "
            f"found {sorted(map(str, devices))}"
        )

    combined = torch.cat(sampled_tensors).cpu()
    parameters = []
    offset = 0
    total_sum = 0.0
    total_sum_squares = 0.0
    total_absmax = 0.0
    total_nonfinite_count = 0
    all_finite = True
    for item in metadata:
        count = item["sample_count"]
        sample = combined[offset : offset + count]
        offset += count
        finite_mask = torch.isfinite(sample)
        nonfinite_count = int((~finite_mask).sum().item())
        finite = nonfinite_count == 0
        sample_sum = float(sample.double().sum().item()) if finite else None
        sample_sum_squares = float(sample.double().square().sum().item()) if finite else None
        sample_absmax = float(sample.abs().max().item()) if finite else None
        item.update(
            {
                "sample_finite": finite,
                "sample_nonfinite_count": nonfinite_count,
                "sample_sum": sample_sum,
                "sample_sum_squares": sample_sum_squares,
                "sample_absmax": sample_absmax,
                "sample_sha256": hashlib.sha256(sample.contiguous().numpy().tobytes()).hexdigest(),
            }
        )
        parameters.append(item)
        all_finite = all_finite and finite
        total_nonfinite_count += nonfinite_count
        if finite:
            total_sum += sample_sum
            total_sum_squares += sample_sum_squares
            total_absmax = max(total_absmax, sample_absmax)

    return {
        "parameter_count": len(parameters),
        "parameter_numel": sum(item["numel"] for item in parameters),
        "sample_count": combined.numel(),
        "sample_finite": all_finite,
        "sample_nonfinite_count": total_nonfinite_count,
        "sample_sum": total_sum if all_finite else None,
        "sample_sum_squares": total_sum_squares if all_finite else None,
        "sample_absmax": total_absmax if all_finite else None,
        "parameters": parameters,
    }


def _model_weight_iteration_coordinate() -> tuple[int, int, int, int]:
    """Return canonical completed/next iterations and current run metadata.

    Megatron restores ``args.iteration`` from the checkpoint before entering
    the training loop. The pinned Megatron training loop then records its
    active completed-iteration coordinate in ``args.curr_iteration`` before
    every ``train_step`` while leaving ``args.iteration`` at the restored
    baseline. Prefer that active coordinate when present and retain the
    restored value as the resume-safe fallback.

    Forward-call counts are deliberately excluded: gradient accumulation can
    change, and Megatron may replay a forward before any optimizer update.
    Neither event is allowed to shift the Megatron training-loop coordinate.
    """
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.training import get_args

    args = get_args()
    restored_iteration = getattr(args, "iteration", None)
    if (
        isinstance(restored_iteration, bool)
        or not isinstance(restored_iteration, int)
        or restored_iteration < 0
    ):
        raise RuntimeError("model-weight audit requires args.iteration to be a nonnegative integer")

    completed_iteration = getattr(args, "curr_iteration", restored_iteration)
    if (
        isinstance(completed_iteration, bool)
        or not isinstance(completed_iteration, int)
        or completed_iteration < 0
    ):
        raise RuntimeError(
            "model-weight audit requires args.curr_iteration to be a nonnegative integer when present"
        )
    if completed_iteration < restored_iteration:
        raise RuntimeError(
            "model-weight audit requires args.curr_iteration to be at least the restored args.iteration"
        )

    train_iters = getattr(args, "train_iters", None)
    if isinstance(train_iters, bool) or not isinstance(train_iters, int) or train_iters <= 0:
        raise RuntimeError("model-weight audit requires args.train_iters to be a positive integer")

    num_microbatches = get_num_microbatches()
    if isinstance(num_microbatches, bool) or not isinstance(num_microbatches, int) or num_microbatches <= 0:
        raise RuntimeError(
            "model-weight audit requires get_num_microbatches() to return a " "positive integer"
        )
    return completed_iteration, completed_iteration + 1, num_microbatches, train_iters


def _reject_json_constant(value: str):
    raise ValueError(f"non-standard JSON constant {value}")


def _encode_strict_json(payload: dict) -> bytes:
    """Encode one deterministic JSON object with no non-finite constants."""
    if not isinstance(payload, dict):
        raise TypeError("model-weight audit payload must be a JSON object")
    return (
        json.dumps(
            payload,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _read_strict_json(path: Path) -> dict:
    """Read a regular, non-symlink JSON object and reject non-finite values."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"model-weight audit output is not a regular file: {path}")
        handle = os.fdopen(descriptor)
        descriptor = None
        with handle:
            payload = json.load(handle, parse_constant=_reject_json_constant)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(payload, dict):
        raise RuntimeError(f"model-weight audit output is not a JSON object: {path}")
    # json.load(parse_constant=...) rejects NaN/Infinity tokens. Re-encoding
    # additionally rejects valid JSON numbers that overflow to Python infinity
    # (for example 1e9999).
    _encode_strict_json(payload)
    return payload


def _model_weight_summary_identity(payload: dict) -> tuple[dict, bool]:
    """Return restart identity while validating replay-variant provenance."""
    provenance_fields = {"forward_step_count", "num_microbatches"}
    present_fields = provenance_fields.intersection(payload)
    if present_fields and present_fields != provenance_fields:
        missing = sorted(provenance_fields - present_fields)
        raise RuntimeError(
            "model-weight audit output has incomplete replay provenance; " f"missing fields: {missing}"
        )

    has_provenance = bool(present_fields)
    identity = dict(payload)
    if has_provenance:
        for field in sorted(provenance_fields):
            value = identity.pop(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RuntimeError(f"model-weight audit output field {field} must be a positive integer")
    return identity, has_provenance


def _write_model_weight_summary_once(output: Path, payload: dict) -> None:
    """Publish one strict JSON record atomically without overwriting."""
    encoded = _encode_strict_json(payload)
    identity, has_provenance = _model_weight_summary_identity(payload)
    encoded_identity = _encode_strict_json(identity)
    descriptor, temporary_value = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_value)
    try:
        os.fchmod(descriptor, 0o600)
        handle = os.fdopen(descriptor, "wb")
        descriptor = None
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError:
            existing = _read_strict_json(output)
            existing_identity, existing_has_provenance = _model_weight_summary_identity(existing)
            if (
                existing_has_provenance != has_provenance
                or _encode_strict_json(existing_identity) != encoded_identity
            ):
                raise RuntimeError(
                    "model-weight audit output already exists with different " f"content: {output}"
                )
        else:
            directory_flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                directory_flags |= os.O_DIRECTORY
            directory_descriptor = os.open(output.parent, directory_flags)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _model_weight_audit_context(step_count: int, *, is_training: bool):
    """Validate audit configuration on every rank before rank-zero selection."""
    raw_steps = os.getenv("PRIMUS_AUDIT_MODEL_WEIGHT_STEPS")
    if raw_steps is None:
        return None
    if not is_training or os.getenv("PRIMUS_SYNTHETIC_WARMUP_ACTIVE") == "1":
        return None

    completed_iterations = _parse_model_weight_steps(raw_steps)
    (
        completed_training_iteration,
        next_training_iteration,
        num_microbatches,
        train_iters,
    ) = _model_weight_iteration_coordinate()
    terminal_iterations = sorted(step for step in completed_iterations if step >= train_iters)
    if terminal_iterations:
        first_terminal = terminal_iterations[0]
        raise RuntimeError(
            "model-weight audit cannot safely sample completed iteration "
            f"{first_terminal} with train_iters={train_iters}: overlap-param-gather "
            "buffers are refreshed by the next model forward, so selected completed "
            f"iteration {first_terminal} requires training through at least {first_terminal + 1}"
        )

    raw_sample_size = os.getenv("PRIMUS_AUDIT_MODEL_WEIGHT_SAMPLE_SIZE", "256")
    if not raw_sample_size.isdigit() or not 1 <= int(raw_sample_size) <= 4096:
        raise RuntimeError("PRIMUS_AUDIT_MODEL_WEIGHT_SAMPLE_SIZE must be an integer in [1, 4096]")
    sample_size = int(raw_sample_size)

    output_value = os.getenv("PRIMUS_AUDIT_MODEL_WEIGHT_PATH")
    if not output_value:
        raise RuntimeError("PRIMUS_AUDIT_MODEL_WEIGHT_PATH is required when weight auditing is enabled")
    output_directory = Path(output_value)
    if not output_directory.is_absolute():
        raise RuntimeError("PRIMUS_AUDIT_MODEL_WEIGHT_PATH must be absolute")

    if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count <= 0:
        raise RuntimeError("model-weight audit requires step_count to be a positive integer")
    forward_step_count = step_count

    # Every deterministic check above runs on every training rank. Branching
    # earlier can leave peers entering model collectives after rank zero fails.
    global_rank = (
        torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.getenv("RANK", "-1"))
    )
    if global_rank != 0:
        return None

    return {
        "completed_iterations": completed_iterations,
        "global_rank": global_rank,
        "completed_training_iteration": completed_training_iteration,
        "next_training_iteration": next_training_iteration,
        "num_microbatches": num_microbatches,
        "sample_size": sample_size,
        "output_directory": output_directory,
        "forward_step_count": forward_step_count,
    }


def _emit_model_weight_summary(
    model,
    step_count: int,
    *,
    is_training: bool = True,
    audit_context=_MODEL_WEIGHT_AUDIT_CONTEXT_UNSET,
) -> None:
    """Write one rank-0 sampled weight summary at requested training steps."""
    context = (
        _model_weight_audit_context(step_count, is_training=is_training)
        if audit_context is _MODEL_WEIGHT_AUDIT_CONTEXT_UNSET
        else audit_context
    )
    if context is None:
        return
    completed_iterations = context["completed_iterations"]
    global_rank = context["global_rank"]
    completed_training_iteration = context["completed_training_iteration"]
    next_training_iteration = context["next_training_iteration"]
    num_microbatches = context["num_microbatches"]
    sample_size = context["sample_size"]
    output_directory = context["output_directory"]
    forward_step_count = context["forward_step_count"]
    if (
        completed_training_iteration not in completed_iterations
        or completed_training_iteration in _EMITTED_MODEL_WEIGHT_ITERATIONS
    ):
        return

    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / f"completed_iteration_{completed_training_iteration:07d}.json"

    payload = {
        "version": 1,
        "global_rank": global_rank,
        "completed_training_iteration": completed_training_iteration,
        "next_training_iteration": next_training_iteration,
        "forward_step_count": forward_step_count,
        # Audit convention: zero means the first eligible audit forward for
        # this completed iteration. It is not arithmetic on cumulative calls.
        "microbatch_index": 0,
        "num_microbatches": num_microbatches,
        "sample_size_per_parameter": sample_size,
        **_sample_model_weights(model, sample_size),
    }
    try:
        _write_model_weight_summary_once(output, payload)
    except BaseException:
        _EMITTED_MODEL_WEIGHT_ITERATIONS.discard(completed_training_iteration)
        raise

    _EMITTED_MODEL_WEIGHT_ITERATIONS.add(completed_training_iteration)
    logger.info(
        "PRIMUS_MODEL_WEIGHT_SUMMARY=%s",
        json.dumps(
            {
                "path": str(output),
                "sample_count": payload["sample_count"],
                "sample_finite": payload["sample_finite"],
                "completed_training_iteration": completed_training_iteration,
                "next_training_iteration": next_training_iteration,
                "forward_step_count": forward_step_count,
            },
            sort_keys=True,
            allow_nan=False,
        ),
    )


def _emit_model_weight_summary_after_forward(
    model_output,
    model,
    step_count: int,
    *,
    is_training: bool = True,
    audit_context=_MODEL_WEIGHT_AUDIT_CONTEXT_UNSET,
):
    """Audit an already-evaluated model forward and return its output unchanged.

    Using this helper as a wrapper around ``model(...)`` makes Python finish the
    model call and all forward pre-hooks before auditing. This is required for
    current distributed-optimizer buffers when ``overlap_param_gather`` is on.
    The real call site carries its pre-forward validated context into this
    wrapper, which still runs before backward and the optimizer update.
    """
    _emit_model_weight_summary(
        model,
        step_count,
        is_training=is_training,
        audit_context=audit_context,
    )
    return model_output


def prepare_flux_latents(
    latents: torch.Tensor,
    scheduler,
    img_ids: Optional[torch.Tensor] = None,
    guidance_scale: Optional[float] = None,
    use_guidance_embed: bool = False,
    timestep_sampler=None,  # Optional: custom timestep sampler
    pregenerated_noise: Optional[torch.Tensor] = None,
    pregenerated_timesteps: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Prepare latents for Flux training forward pass.

    This function:
        1. Generates img_ids if not provided (for robustness)
        2. Samples timesteps using configurable sampling strategy
        3. Adds noise to latents using flow matching
        4. Packs latents into sequence format
        5. Prepares guidance embeddings (if enabled)

    Args:
        latents: Clean latent tensor (B, C, H, W)
        scheduler: Flow matching scheduler
        img_ids: Image position IDs (B, H*W/4, 3). If None, will be generated
        guidance_scale: Guidance scale value (for CFG)
        use_guidance_embed: Whether to use guidance embedding
        timestep_sampler: Optional custom timestep sampler
                          (default: LogitNormalSampler)
        pregenerated_noise: If provided, use this noise instead of sampling.
            Used by deterministic comparison tests to ensure identical inputs.
        pregenerated_timesteps: If provided, use these timesteps (in [0,1] range)
            instead of sampling. Used by deterministic comparison tests.

    Returns:
        Tuple containing:
            - clean_latents: Original latents (for target computation)
            - noise: Sampled noise
            - packed_noisy_latents: Noisy latents in packed format
            - img_ids: Image position IDs
            - guidance_vec: Guidance vector (or None)
            - timesteps: Sampled timesteps (in [0, num_train_timesteps] range)
            - sigma_1d: Raw sigma in [0, 1] range, shape [B]. Pass directly to
              the model as timesteps_norm to avoid bf16 round-trip precision loss.

    Reference:
        NeMo's prepare_image_latent_like_reference()
    """
    batch_size, num_channels, height, width = latents.shape
    device = latents.device
    dtype = latents.dtype

    # Use default sampler if not provided
    if timestep_sampler is None:
        timestep_sampler = LogitNormalSampler()

    # Generate img_ids if not provided (for robustness with variable sizes)
    if img_ids is None:
        img_ids = generate_image_position_ids(batch_size, height, width, device, dtype)

    if pregenerated_noise is not None:
        noise = pregenerated_noise.to(device=device, dtype=dtype)
    else:
        noise = torch.randn_like(latents, device=device, dtype=dtype)

    if pregenerated_timesteps is not None:
        sigma = pregenerated_timesteps.to(device=device, dtype=dtype)
        timesteps = sigma * scheduler.num_train_timesteps
    else:
        timesteps, sigma = timestep_sampler.sample(batch_size, device, scheduler)

    # Convert sigma to correct dtype
    sigma = sigma.to(dtype=dtype)

    # Save 1D sigma [B] before unsqueezing — used as timesteps_norm to avoid
    # the bf16 round-trip (sigma * 1000 / 1000) that corrupts ~2.5% of values.
    sigma_1d = sigma.clone()

    # Broadcast sigma to match latent dimensions
    while len(sigma.shape) < latents.ndim:
        sigma = sigma.unsqueeze(-1)

    # Flow matching forward process: x_t = (1 - sigma) * x_0 + sigma * noise
    noisy_latents = apply_flow_matching_noise(latents, noise, sigma)

    # Pack latents into sequence format
    packed_noisy_latents = pack_latents(noisy_latents)

    # Prepare guidance embedding (if enabled)
    if use_guidance_embed and guidance_scale is not None:
        guidance_vec = torch.full(
            (batch_size,),
            guidance_scale,
            device=device,
            dtype=dtype,
        )
    else:
        guidance_vec = None

    return (
        latents,
        noise,
        packed_noisy_latents,
        img_ids,
        guidance_vec,
        timesteps,
        sigma_1d,
    )


# NOTE: kept as an eager alias — torch.compile breaks CUDA RNG reproducibility
# (compiled torch.randn_like produces different values than eager mode with the
# same generator state). prepare_flux_latents only contains small ops (randn,
# rand, element-wise), so compile overhead exceeds any fusion benefit. Eager
# also matches NeMo's RNG sequence for cross-framework convergence comparison.
_eager_prepare_flux_latents = prepare_flux_latents


def _is_validation_forward(model, _batch=None) -> bool:
    """Classify from rank-consistent model state, not rank-local batch data."""
    return not model.training


def _pregenerated_diffusion_inputs(
    batch,
    *,
    tp_size,
    is_validation,
    compute_dtype,
    tensor_parallel,
):
    """Return deterministic inputs without splitting TP collective participation."""
    if tp_size == 1:
        return batch.get("noise"), batch.get("timesteps")
    if not is_validation:
        return None, None

    # Evaluation mode is rank-consistent, so every TP rank enters this
    # collective even though only TP rank zero owns ``batch``.
    batch_timesteps = tensor_parallel.broadcast_data(
        ["timesteps"],
        batch,
        compute_dtype,
    ).get("timesteps")
    if not batch_timesteps.is_cuda:
        batch_timesteps = batch_timesteps.cuda(non_blocking=True)
    return None, batch_timesteps


def flux_forward_step_func(
    data_iterator,
    model,
    scheduler,
    use_guidance_embed=False,
    guidance_scale=None,
    timestep_sampler=None,
    cfg_dropout_prob=0.0,
    empty_t5_encodings=None,
    empty_clip_encodings=None,
    vae_scale=None,
    vae_shift=None,
    vae_latent_mode="presampled",
    per_step_rng_reseed=False,
    step_count=0,
):
    """
    Forward step function for Flux training with distributed data loading.

    Following Megatron's multimodal data loading pattern:
    - When TP=1 (pure DP): each rank loads data directly, no broadcast needed
    - When TP>1: only TP rank 0 has data_iterator, broadcast to other TP ranks
    - Middle PP stages return early

    This function orchestrates the training step by:
    1. Handling distributed data loading (broadcast from rank 0)
    2. Loading or encoding images (via helper function)
    3. Loading or encoding text (via helper function)
    4. Optionally applying CFG dropout (replacing text embeddings with empty encodings)
    5. Preparing latents with noise and packing (via helper function)
    6. Running model forward pass
    7. Returning model output and loss computation inputs

    Supports two data formats (follows NeMo conventions):
        1. Pre-encoded: latents, prompt_embeds, pooled_prompt_embeds, text_ids (optional)
        2. Raw: images, txt - encodes on-the-fly

    Architecture follows NeMo conventions for Flux training.

    Args:
        data_iterator: Iterator yielding training batches (None on non-dataloader ranks)
        model: Flux model instance with encoders (config.params_dtype used for data broadcasting)
        scheduler: Flow matching scheduler
        use_guidance_embed: Whether model uses guidance embedding
        guidance_scale: Guidance scale for CFG training
        timestep_sampler: Optional custom timestep sampler (default: LogitNormalSampler)
        cfg_dropout_prob: Probability of replacing text embeddings with empty encodings (default: 0.0)
        empty_t5_encodings: Pre-generated fixed empty T5 encodings (seq_len, 1, context_dim)
        empty_clip_encodings: Pre-generated fixed empty CLIP encodings (vec_in_dim,)
        vae_scale: Optional VAE latent scale factor (default: None, MLPerf uses 0.3611)
        vae_shift: Optional VAE latent shift factor (default: None, MLPerf uses 0.1159)
        vae_latent_mode: How to obtain latents from the batch (default: "presampled").
            "presampled" — use stored latents directly.
            "resample" — reconstruct latents from stored mean+logvar via
            reparameterization at every step, then apply vae_scale/vae_shift.
        per_step_rng_reseed: Reseed the default CUDA generator at each step
            to isolate training random ops from model forward RNG consumption
            (default: False).
        step_count: Monotonically increasing counter identifying this forward
            call. Used only to derive a per-step RNG seed and as audit
            provenance; training-loop coordinates come from Megatron iteration
            state. Managed by the caller (DiffusionPretrainTrainer) and
            reconstructed on resume as iteration * num_microbatches.

    Returns:
        Tuple of (noise_pred, clean_latents, noise, loss_mask, metrics_dict, is_validation)
        - noise_pred: Model output (predicted velocity) [B, C, H, W]
        - clean_latents: Original clean latents [B, C, H, W]
        - noise: Sampled noise [B, C, H, W]
        - loss_mask: Optional mask for variable-length sequences [B] or None
        - metrics_dict: Dictionary with training metrics
        - is_validation: True when the model is in evaluation mode
    """
    # Reseed default CUDA generator per step to isolate training random ops
    # (noise, timesteps, CFG dropout) from model forward RNG consumption.
    # Required because TE fused attention advances the default generator even
    # with dropout=0 when the DPA prologue patch is active.
    if per_step_rng_reseed:
        from megatron.core import parallel_state as _ps
        from megatron.training import get_args as _get_args

        _seed = _get_args().seed
        _per_rank_seed = _seed + 100 * _ps.get_data_parallel_rank()
        _step_seed = (_per_rank_seed * 10000 + step_count) % (2**63)
        torch.cuda.manual_seed(_step_seed)

    from megatron.core import tensor_parallel
    from megatron.core.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    # Pipeline parallelism (pipeline_model_parallel_size > 1) is rejected at
    # config construction for Flux (see BaseDiffusionConfig.__post_init__), so
    # no middle-pipeline-stage handling is needed here.
    # Derive compute dtype from bf16/fp16 flags rather than params_dtype.
    # When use_fsdp2_fp32_param_optimizer is active, params_dtype is FP32 (for
    # optimizer precision) but compute should still be BF16/FP16.
    if model.config.bf16:
        compute_dtype = torch.bfloat16
    elif model.config.fp16:
        compute_dtype = torch.float16
    else:
        compute_dtype = model.config.params_dtype

    tp_size = get_tensor_model_parallel_world_size()

    if tp_size == 1:
        # Pure DP: every rank has its own data iterator (is_distributed=True).
        # Skip broadcast_data overhead (~1.5ms of GPU idle from NCCL
        # self-broadcasts, GPU->CPU transfers, and .item() sync stalls).
        if data_iterator is None:
            raise RuntimeError(
                "data_iterator is None with TP=1; dataset provider must set is_distributed=True"
            )
        batch = next(data_iterator)
        if not isinstance(batch, dict):
            raise TypeError(
                f"[ForwardStep] Expected batch to be dict, got {type(batch)}. Batch value: {batch}"
            )
        if vae_latent_mode == "resample":
            required_keys = ["mean", "logvar", "prompt_embeds", "pooled_prompt_embeds"]
        else:
            required_keys = ["latents", "prompt_embeds", "pooled_prompt_embeds"]
        missing_keys = [k for k in required_keys if k not in batch]
        if missing_keys:
            raise KeyError(
                f"[ForwardStep] Batch missing required keys: {missing_keys}. "
                f"Got keys: {list(batch.keys())}. "
                f"vae_latent_mode={vae_latent_mode}"
            )

        # Cast to compute_dtype and move to CUDA in one pass
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                if batch[key].is_floating_point():
                    batch[key] = batch[key].to(dtype=compute_dtype, device="cuda", non_blocking=True)
                elif not batch[key].is_cuda:
                    batch[key] = batch[key].cuda(non_blocking=True)

        prompt_embeds = batch["prompt_embeds"]
        pooled_prompt_embeds = batch["pooled_prompt_embeds"]

        if vae_latent_mode == "resample":
            mean = batch["mean"]
            logvar = batch["logvar"]
        else:
            latents = batch["latents"]

        loss_mask = batch.get("loss_mask")
    else:
        # TP > 1: only rank 0 loads data, broadcast to other TP ranks
        if data_iterator is not None and get_tensor_model_parallel_rank() == 0:
            try:
                batch = next(data_iterator)
                if not isinstance(batch, dict):
                    raise TypeError(
                        f"[ForwardStep] Expected batch to be dict, got {type(batch)}. "
                        f"Batch value: {batch}"
                    )
                if vae_latent_mode == "resample":
                    required_keys = ["mean", "logvar", "prompt_embeds", "pooled_prompt_embeds"]
                else:
                    required_keys = ["latents", "prompt_embeds", "pooled_prompt_embeds"]
                missing_keys = [k for k in required_keys if k not in batch]
                if missing_keys:
                    raise KeyError(
                        f"[ForwardStep] Batch missing required keys: {missing_keys}. "
                        f"Got keys: {list(batch.keys())}. "
                        f"vae_latent_mode={vae_latent_mode}"
                    )
            except StopIteration:
                raise RuntimeError(
                    "[ForwardStep] Data iterator exhausted (should be infinite with "
                    "MegatronDataloaderWrapper). This indicates a bug in the dataloader wrapper."
                )
            except Exception as e:
                logger.error(f"[ForwardStep] Error getting batch: {type(e).__name__}: {e}")
                import traceback

                logger.error(f"[ForwardStep] Traceback: {traceback.format_exc()}")
                raise
        else:
            batch = None

        if batch is not None:
            for key in batch:
                if isinstance(batch[key], torch.Tensor) and batch[key].is_floating_point():
                    batch[key] = batch[key].to(dtype=compute_dtype)

        try:
            prompt_embeds = tensor_parallel.broadcast_data(["prompt_embeds"], batch, compute_dtype).get(
                "prompt_embeds"
            )
            pooled_prompt_embeds = tensor_parallel.broadcast_data(
                ["pooled_prompt_embeds"], batch, compute_dtype
            ).get("pooled_prompt_embeds")

            if vae_latent_mode == "resample":
                mean = tensor_parallel.broadcast_data(["mean"], batch, compute_dtype).get("mean")
                logvar = tensor_parallel.broadcast_data(["logvar"], batch, compute_dtype).get("logvar")
            else:
                latents = tensor_parallel.broadcast_data(["latents"], batch, compute_dtype).get("latents")
        except Exception as e:
            logger.error(f"[ForwardStep] Error broadcasting data: {type(e).__name__}: {e}")
            logger.error(
                f"[ForwardStep] batch type: {type(batch)}, "
                f"batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'N/A'}"
            )
            if isinstance(batch, dict):
                for key, value in batch.items():
                    logger.error(
                        f"[ForwardStep]   {key}: type={type(value)}, "
                        f"shape={value.shape if hasattr(value, 'shape') else 'N/A'}"
                    )
            import traceback

            logger.error(f"[ForwardStep] Traceback: {traceback.format_exc()}")
            raise

        loss_mask = None
        if batch is not None and "loss_mask" in batch:
            loss_mask = tensor_parallel.broadcast_data(["loss_mask"], batch, compute_dtype).get("loss_mask")
            if not loss_mask.is_cuda:
                loss_mask = loss_mask.cuda(non_blocking=True)

        if not prompt_embeds.is_cuda:
            prompt_embeds = prompt_embeds.cuda(non_blocking=True)
        if not pooled_prompt_embeds.is_cuda:
            pooled_prompt_embeds = pooled_prompt_embeds.cuda(non_blocking=True)

    # Obtain latents based on vae_latent_mode
    if vae_latent_mode == "resample":
        # Resample mode: reconstruct latents from posterior parameters each step
        if not mean.is_cuda:
            mean = mean.cuda(non_blocking=True)
        if not logvar.is_cuda:
            logvar = logvar.cuda(non_blocking=True)
        std = torch.exp(0.5 * logvar)
        vae_eps = torch.randn_like(mean)

        latents = mean + std * vae_eps
        # Scale/shift is always applied after resampling (raw posterior -> normalized latents)
        latents = vae_scale * (latents - vae_shift)
    else:
        # Presampled mode: use stored latents directly
        if not latents.is_cuda:
            latents = latents.cuda(non_blocking=True)

    # Validation detection.
    #
    # MLPerf v5.1 Flux1 validation spec (flux1/nemo/README.md §6 "Evaluation"):
    #   - Per-sample fixed timestep t ∈ {0/8, 1/8, ..., 7/8}
    #   - Equal sample count per timestep (29 696 / 8 = 3 712)
    #   - val_loss = mean over per-timestep means (equivalent to flat mean given
    #     equal counts).
    #
    # NeMo's official to_webdataset preserves a `timestep` integer per sample
    # from the MLCommons Arrow source. Our `primus-cli data diffusion-ingest`
    # path (pipelines/ingest.py:33 `ARROW_COLUMNS`) ingests only the 4 tensor
    # columns and writes `{"key": ...}` to the json sidecar — so our val shards
    # are MISSING the timestep field, which used to make this branch fall
    # through to the training path with uniform-random timesteps via the
    # `timestep_sampler`. That produced a *different* val_loss estimator than
    # the spec's: E_t~U[0,1][MSE] (Monte Carlo over [0,1]) vs the spec's
    # left-Riemann sum over t∈{0/8..7/8}. The two estimators are not
    # comparable, so a uniform-random val path can make val_loss converge
    # spuriously fast relative to the reference convergence point.
    #
    # Fix: when batch is in eval mode (model.training=False, set by
    # the evaluation harness via `model_module.eval()`) and lacks a `timestep`
    # field, inject equidistant timesteps deterministically by within-batch
    # index. With MBS=64, each micro-batch covers each t∈{0..7} exactly 8
    # times. Across 58 micro-batches × 8 DP ranks = 464 micro-batches → exactly
    # 3 712 samples per timestep, matching the MLPerf v5.1 spec count.
    #
    # CFG dropout during val: SUPPRESSED.
    #
    # Reference-implementation tally for "apply CFG dropout during validation":
    #   NeMo MLPerf reference (custom_flux.py): ON
    #   AMD's MLPerf submission: OFF
    #   TorchTitan flux training script: OFF
    #
    # CFG-off during validation is MLPerf-compliant under the v6.0 rules even
    # though NeMo (which generated the reference convergence point) has it on.
    # Empirically, applying CFG-during-val structurally inflates val_loss by
    # ~0.015-0.030 (the 10% unconditional samples pay a ~0.15-0.30 MSE
    # penalty), which is enough to materially shift the convergence-crossing
    # step, so we keep it off to match the submission configuration.
    # Evaluation mode is identical on every tensor-parallel rank, including
    # ranks where ``batch`` is intentionally None. Batch-local classification
    # can split ranks before model collectives when TP > 1.
    is_validation = _is_validation_forward(model, batch)
    if batch is not None and is_validation and "timestep" in batch:
        val_timesteps = batch["timestep"].to(dtype=compute_dtype) / 8.0
        batch["timesteps"] = val_timesteps
    elif batch is not None and is_validation:
        batch_size_val = pooled_prompt_embeds.shape[0]
        val_idx = torch.arange(batch_size_val, device="cuda") % 8
        batch["timestep"] = val_idx
        val_timesteps = val_idx.to(dtype=compute_dtype) / 8.0
        batch["timesteps"] = val_timesteps

    # Validate every rank before noise preparation and the expensive model
    # forward, then carry rank zero's parsed context through publication.
    # Direct emitter calls build and validate the same context themselves.
    model_weight_audit_context = _model_weight_audit_context(
        step_count,
        is_training=not is_validation,
    )

    if batch is not None:
        _emit_batch_fingerprint(
            batch,
            step_count,
            is_training=not is_validation,
        )

    # Matches NeMo's forward_step which wraps prepare_image_latent_like_reference
    # in torch.no_grad() — no gradients needed for position IDs, noise sampling,
    # timestep sampling, or latent packing.
    with torch.no_grad():
        # Generate img_ids based on latent spatial dimensions
        # NOTE: When RoPE fusion is enabled, we use batch_size=1 to satisfy Transformer Engine's
        # fused kernel constraints (freqs must have shape [S, 1, 1, D]). PyTorch broadcasting
        # applies the same position grid across all batch samples. This requires all images in
        # the batch to have the same resolution (same height/width).
        rope_fusion_batch_size = 1 if model.config.apply_rope_fusion else latents.shape[0]
        img_ids = generate_image_position_ids(
            batch_size=rope_fusion_batch_size,
            height=latents.shape[2],
            width=latents.shape[3],
            device=latents.device,
            dtype=latents.dtype,
        )

        # Generate text_ids (Flux convention: zeros for text position IDs)
        # NOTE: When RoPE fusion is enabled, use batch_size=1 for consistency with img_ids
        # (broadcasting will handle the actual batch dimension). This matches NVIDIA's MLPerf
        # implementation strategy: both txt_ids and img_ids have shape [1, seq_len, 3] with
        # RoPE fusion, allowing proper concatenation before the fused RoPE kernel.
        text_ids = generate_text_position_ids(
            batch_size=rope_fusion_batch_size,
            seq_len=prompt_embeds.shape[1],
            device=latents.device,
            dtype=latents.dtype,
        )

        # Extract pre-generated noise/timesteps from batch (deterministic tests)
        batch_noise, batch_timesteps = _pregenerated_diffusion_inputs(
            batch,
            tp_size=tp_size,
            is_validation=is_validation,
            compute_dtype=compute_dtype,
            tensor_parallel=tensor_parallel,
        )

        # Prepare latents (noise, packing, scheduling).
        # Eager wrapper — see _eager_prepare_flux_latents NOTE for why compile
        # is intentionally disabled (CUDA RNG reproducibility).
        (
            clean_latents,
            noise,
            packed_noisy_latents,
            img_ids,
            guidance_vec,
            timesteps,
            sigma_1d,
        ) = _eager_prepare_flux_latents(
            latents=latents,
            scheduler=scheduler,
            img_ids=img_ids,
            guidance_scale=guidance_scale,
            use_guidance_embed=use_guidance_embed,
            timestep_sampler=timestep_sampler,
            pregenerated_noise=batch_noise,
            pregenerated_timesteps=batch_timesteps,
        )

    # CFG dropout: randomly replace text embeddings with fixed empty encodings.
    # Placed after prepare_flux_latents so the RNG consumption order matches NeMo:
    # VAE resample → noise → timesteps → CFG dropout.
    # Applied during training only — validation uses fixed per-sample timesteps.
    if (
        not is_validation
        and cfg_dropout_prob > 0.0
        and empty_t5_encodings is not None
        and empty_clip_encodings is not None
    ):
        batch_size_cfg = pooled_prompt_embeds.shape[0]
        dropout_mask = torch.rand(batch_size_cfg, device="cuda") < cfg_dropout_prob

        empty_t5 = empty_t5_encodings.to(device="cuda", dtype=prompt_embeds.dtype, non_blocking=True)
        empty_t5 = empty_t5.squeeze(1).unsqueeze(0)

        if empty_t5.shape[1] != prompt_embeds.shape[1]:
            raise ValueError(
                f"Empty T5 encoding seq_len ({empty_t5.shape[1]}) does not match "
                f"data T5 seq_len ({prompt_embeds.shape[1]}). "
                f"Regenerate empty encodings with matching t5_max_length."
            )

        t5_mask = dropout_mask.view(-1, 1, 1).expand_as(prompt_embeds)
        prompt_embeds = torch.where(t5_mask, empty_t5.expand_as(prompt_embeds), prompt_embeds)

        empty_clip = empty_clip_encodings.to(
            device="cuda", dtype=pooled_prompt_embeds.dtype, non_blocking=True
        )
        clip_mask = dropout_mask.view(-1, 1).expand_as(pooled_prompt_embeds)
        pooled_prompt_embeds = torch.where(
            clip_mask, empty_clip.expand_as(pooled_prompt_embeds), pooled_prompt_embeds
        )

    # Transpose for Megatron format (sequence-first)
    packed_noisy_latents = packed_noisy_latents.transpose(0, 1)
    prompt_embeds = prompt_embeds.transpose(0, 1)

    # Use raw sigma directly instead of timesteps/1000 to avoid bf16 round-trip
    timesteps_norm = sigma_1d.to(dtype=packed_noisy_latents.dtype)

    with torch.amp.autocast("cuda", enabled=True, dtype=compute_dtype):
        # The model call is evaluated before the wrapper. With
        # overlap_param_gather, this guarantees every forward pre-hook has
        # refreshed its distributed-optimizer parameter buffer before sampling.
        noise_pred = _emit_model_weight_summary_after_forward(
            model(
                img=packed_noisy_latents,
                txt=prompt_embeds,
                y=pooled_prompt_embeds,
                timesteps=timesteps_norm,
                img_ids=img_ids,
                txt_ids=text_ids,
                guidance=guidance_vec,
            ),
            model,
            step_count,
            is_training=not is_validation,
            audit_context=model_weight_audit_context,
        )

        # Unpack latents from sequence format
        noise_pred = noise_pred.transpose(0, 1)  # (S, B, C*4) -> (B, S, C*4)
        noise_pred = unpack_latents(
            noise_pred,
            height=clean_latents.shape[2],
            width=clean_latents.shape[3],
        )  # -> (B, C, H, W)

    # Create metrics dict for logging
    metrics = {
        "batch_size": latents.shape[0],
        "image_height": clean_latents.shape[2] * 8,  # VAE 8x downsampling
        "image_width": clean_latents.shape[3] * 8,
        "latent_channels": clean_latents.shape[1],
        "avg_timestep": timesteps.float().mean(),
        "text_seq_len": prompt_embeds.shape[0],  # After transpose to (S, B, C)
        "img_seq_len": packed_noisy_latents.shape[0],  # After transpose to (S, B, C)
    }

    # Return model output and loss computation inputs (matching Megatron's pattern)
    return noise_pred, clean_latents, noise, loss_mask, metrics, is_validation


__all__ = [
    "prepare_flux_latents",
    "flux_forward_step_func",
]
