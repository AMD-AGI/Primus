# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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

import logging
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
            call. Used to derive a unique per-step RNG seed. Managed by the
            caller (DiffusionPretrainTrainer) and reconstructed from checkpoint
            state on resume as iteration * num_microbatches.

    Returns:
        Tuple of (noise_pred, clean_latents, noise, loss_mask, metrics_dict, is_validation)
        - noise_pred: Model output (predicted velocity) [B, C, H, W]
        - clean_latents: Original clean latents [B, C, H, W]
        - noise: Sampled noise [B, C, H, W]
        - loss_mask: Optional mask for variable-length sequences [B] or None
        - metrics_dict: Dictionary with training metrics
        - is_validation: True when batch contains "timestep" key (MLPerf validation mode)
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
    is_validation = False
    if batch is not None and "timestep" in batch:
        is_validation = True
        val_timesteps = batch["timestep"].float() / 8.0
        batch["timesteps"] = val_timesteps
    elif batch is not None and not model.training:
        is_validation = True
        batch_size_val = pooled_prompt_embeds.shape[0]
        val_idx = torch.arange(batch_size_val, device="cuda") % 8
        batch["timestep"] = val_idx
        val_timesteps = val_idx.to(dtype=compute_dtype) / 8.0
        batch["timesteps"] = val_timesteps

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
        batch_noise = None
        batch_timesteps = None
        if batch is not None:
            if tp_size == 1:
                batch_noise = batch.get("noise")
                batch_timesteps = batch.get("timesteps")
            else:
                if "noise" in batch:
                    batch_noise = tensor_parallel.broadcast_data(["noise"], batch, compute_dtype).get("noise")
                    if not batch_noise.is_cuda:
                        batch_noise = batch_noise.cuda(non_blocking=True)
                if "timesteps" in batch:
                    batch_timesteps = tensor_parallel.broadcast_data(["timesteps"], batch, compute_dtype).get(
                        "timesteps"
                    )
                    if not batch_timesteps.is_cuda:
                        batch_timesteps = batch_timesteps.cuda(non_blocking=True)

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
        noise_pred = model(
            img=packed_noisy_latents,
            txt=prompt_embeds,
            y=pooled_prompt_embeds,
            timesteps=timesteps_norm,
            img_ids=img_ids,
            txt_ids=text_ids,
            guidance=guidance_vec,
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
