# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""
HuggingFace to Primus Flux checkpoint converter.

Converts HuggingFace Diffusers Flux transformer checkpoints to
Primus/Megatron-Core compatible format.

Key Conversion:
    - HuggingFace: Separate double_blocks, single_blocks
    - Primus: Unified transformer.layers.{0-N} with TransformerBlock

    This reflects Primus's architectural enhancement using heterogeneous
    layer specifications in a single TransformerBlock container.

Usage:
    from primus.backends.megatron.core.models.diffusion.flux import convert_hf_checkpoint

    primus_state_dict = convert_hf_checkpoint(
        checkpoint_path="black-forest-labs/FLUX.1-dev",
        flux_config=config,
        save_to="primus_flux_12b.safetensors"
    )

Reference:
    - HuggingFace Diffusers checkpoint format
    - Megatron-Core TransformerBlock architecture
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

logger = logging.getLogger(__name__)


def _fuse_qkv_weights(
    config,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Fuse separate Q, K, V weight matrices into Megatron's fused QKV format.

    Megatron-Core uses grouped-query attention (GQA) format where Q, K, V
    are interleaved per attention group:
        [Q_group0, K_group0, V_group0, Q_group1, K_group1, V_group1, ...]

    Args:
        config: FluxConfig with num_attention_heads, num_query_groups
        q_weight: Query weights [hidden_size, hidden_size]
        k_weight: Key weights [hidden_size, hidden_size]
        v_weight: Value weights [hidden_size, hidden_size]

    Returns:
        Fused QKV weights [(heads_per_group + 2) * num_query_groups * head_size, hidden_size]
        in GQA interleaved format

    Reference:
        - Megatron-Core: Grouped Query Attention patterns
    """
    head_num = config.num_attention_heads
    num_query_groups = getattr(config, "num_query_groups", head_num)
    heads_per_group = head_num // num_query_groups
    hidden_size = config.hidden_size
    head_size = hidden_size // head_num

    # Get input shape from q_weight
    old_tensor_shape = q_weight.size()

    # Reshape to [num_heads, head_size, ...]
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
    new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

    q = q_weight.view(*new_q_tensor_shape)
    k = k_weight.view(*new_kv_tensor_shape)
    v = v_weight.view(*new_kv_tensor_shape)

    # Interleave by group: [Q_heads_for_group, K_group, V_group, ...]
    qkv_list = []
    for i in range(num_query_groups):
        qkv_list.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
        qkv_list.append(k[i : i + 1, :, :])
        qkv_list.append(v[i : i + 1, :, :])

    qkv = torch.cat(qkv_list, dim=0)

    # Validate shape
    if qkv.ndim != 3:
        raise ValueError(f"Expected 3D QKV tensor, got shape {qkv.shape}")
    expected_dim0 = (heads_per_group + 2) * num_query_groups
    if qkv.shape[0] != expected_dim0:
        raise ValueError(f"Expected QKV dim0 {expected_dim0}, got shape {qkv.shape}")
    if qkv.shape[1] != head_size:
        raise ValueError(f"Expected QKV head_size {head_size}, got shape {qkv.shape}")
    if qkv.shape[2] != old_tensor_shape[1]:
        raise ValueError(f"Expected QKV hidden dim {old_tensor_shape[1]}, got shape {qkv.shape}")

    # Reshape to [fused_dim, hidden_size]
    qkv = qkv.reshape(head_size * (head_num + 2 * num_query_groups), hidden_size)

    return qkv


def _fuse_qkv_bias(
    config,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    v_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Fuse Q, K, V bias terms into Megatron's fused format.

    Args:
        config: FluxConfig with num_attention_heads, num_query_groups
        q_bias: Query bias [hidden_size]
        k_bias: Key bias [hidden_size]
        v_bias: Value bias [hidden_size]

    Returns:
        Fused QKV bias [(heads_per_group + 2) * num_query_groups * head_size]
        in GQA interleaved format

    Reference:
        - Megatron-Core: Grouped Query Attention patterns
    """
    head_num = config.num_attention_heads
    num_query_groups = getattr(config, "num_query_groups", head_num)
    heads_per_group = head_num // num_query_groups
    hidden_size = config.hidden_size
    head_size = hidden_size // head_num

    # Reshape to [num_heads, head_size]
    new_q_bias_shape = (head_num, head_size)
    new_kv_bias_shape = (num_query_groups, head_size)

    q = q_bias.view(*new_q_bias_shape)
    k = k_bias.view(*new_kv_bias_shape)
    v = v_bias.view(*new_kv_bias_shape)

    # Interleave by group
    qkv_bias_list = []
    for i in range(num_query_groups):
        qkv_bias_list.append(q[i * heads_per_group : (i + 1) * heads_per_group, :])
        qkv_bias_list.append(k[i : i + 1, :])
        qkv_bias_list.append(v[i : i + 1, :])

    qkv_bias = torch.cat(qkv_bias_list, dim=0)
    qkv_bias = qkv_bias.reshape(head_size * (head_num + 2 * num_query_groups))

    return qkv_bias


# Key mapping from HuggingFace to Primus
# After TransformerBlock refactor: double_blocks and single_blocks are now transformer.layers[0-56]
FLUX_KEY_MAPPING = {
    "double_blocks": {
        "norm1.linear.weight": "adaln.adaLN_modulation.1.weight",
        "norm1.linear.bias": "adaln.adaLN_modulation.1.bias",
        "norm1_context.linear.weight": "adaln_context.adaLN_modulation.1.weight",
        "norm1_context.linear.bias": "adaln_context.adaLN_modulation.1.bias",
        "attn.norm_q.weight": "self_attention.q_layernorm.weight",
        "attn.norm_k.weight": "self_attention.k_layernorm.weight",
        "attn.norm_added_q.weight": "self_attention.added_q_layernorm.weight",
        "attn.norm_added_k.weight": "self_attention.added_k_layernorm.weight",
        "attn.to_out.0.weight": "self_attention.linear_proj.weight",
        "attn.to_out.0.bias": "self_attention.linear_proj.bias",
        "attn.to_add_out.weight": "self_attention.added_linear_proj.weight",
        "attn.to_add_out.bias": "self_attention.added_linear_proj.bias",
        "ff.net.0.proj.weight": "mlp.linear_fc1.weight",
        "ff.net.0.proj.bias": "mlp.linear_fc1.bias",
        "ff.net.2.weight": "mlp.linear_fc2.weight",
        "ff.net.2.bias": "mlp.linear_fc2.bias",
        "ff_context.net.0.proj.weight": "context_mlp.linear_fc1.weight",
        "ff_context.net.0.proj.bias": "context_mlp.linear_fc1.bias",
        "ff_context.net.2.weight": "context_mlp.linear_fc2.weight",
        "ff_context.net.2.bias": "context_mlp.linear_fc2.bias",
    },
    "single_blocks": {
        "norm.linear.weight": "adaln.adaLN_modulation.1.weight",
        "norm.linear.bias": "adaln.adaLN_modulation.1.bias",
        "proj_mlp.weight": "mlp.linear_fc1.weight",
        "proj_mlp.bias": "mlp.linear_fc1.bias",
        "attn.norm_q.weight": "self_attention.q_layernorm.weight",
        "attn.norm_k.weight": "self_attention.k_layernorm.weight",
    },
    # Root-level mappings
    "norm_out.linear.bias": "norm_out.adaLN_modulation.1.bias",
    "norm_out.linear.weight": "norm_out.adaLN_modulation.1.weight",
    "proj_out.bias": "proj_out.bias",
    "proj_out.weight": "proj_out.weight",
    "time_text_embed.guidance_embedder.linear_1.bias": "guidance_embedding.in_layer.bias",
    "time_text_embed.guidance_embedder.linear_1.weight": "guidance_embedding.in_layer.weight",
    "time_text_embed.guidance_embedder.linear_2.bias": "guidance_embedding.out_layer.bias",
    "time_text_embed.guidance_embedder.linear_2.weight": "guidance_embedding.out_layer.weight",
    "x_embedder.bias": "img_embed.bias",
    "x_embedder.weight": "img_embed.weight",
    "time_text_embed.timestep_embedder.linear_1.bias": "timestep_embedding.time_embedding.in_layer.bias",
    "time_text_embed.timestep_embedder.linear_1.weight": "timestep_embedding.time_embedding.in_layer.weight",
    "time_text_embed.timestep_embedder.linear_2.bias": "timestep_embedding.time_embedding.out_layer.bias",
    "time_text_embed.timestep_embedder.linear_2.weight": "timestep_embedding.time_embedding.out_layer.weight",
    "context_embedder.bias": "txt_embed.bias",
    "context_embedder.weight": "txt_embed.weight",
    "time_text_embed.text_embedder.linear_1.bias": "vector_embedding.in_layer.bias",
    "time_text_embed.text_embedder.linear_1.weight": "vector_embedding.in_layer.weight",
    "time_text_embed.text_embedder.linear_2.bias": "vector_embedding.out_layer.bias",
    "time_text_embed.text_embedder.linear_2.weight": "vector_embedding.out_layer.weight",
}


# Key mapping from Black Forest Labs native format to Primus format
# Reference: diffusers Flux checkpoint conversion
# This is the format used in official BFL releases (e.g., flux1-dev.safetensors, flux1-schnell.sft)
BFL_KEY_MAPPING = {
    # Root-level embeddings
    "img_in.weight": "img_embed.weight",
    "img_in.bias": "img_embed.bias",
    "txt_in.weight": "txt_embed.weight",
    "txt_in.bias": "txt_embed.bias",
    # Timestep embedding (time_in → timestep_embedding.time_embedding)
    "time_in.in_layer.weight": "timestep_embedding.time_embedding.in_layer.weight",
    "time_in.in_layer.bias": "timestep_embedding.time_embedding.in_layer.bias",
    "time_in.out_layer.weight": "timestep_embedding.time_embedding.out_layer.weight",
    "time_in.out_layer.bias": "timestep_embedding.time_embedding.out_layer.bias",
    # Vector embedding (vector_in → vector_embedding)
    "vector_in.in_layer.weight": "vector_embedding.in_layer.weight",
    "vector_in.in_layer.bias": "vector_embedding.in_layer.bias",
    "vector_in.out_layer.weight": "vector_embedding.out_layer.weight",
    "vector_in.out_layer.bias": "vector_embedding.out_layer.bias",
    # Guidance embedding (guidance_in → guidance_embedding) - optional
    "guidance_in.in_layer.weight": "guidance_embedding.in_layer.weight",
    "guidance_in.in_layer.bias": "guidance_embedding.in_layer.bias",
    "guidance_in.out_layer.weight": "guidance_embedding.out_layer.weight",
    "guidance_in.out_layer.bias": "guidance_embedding.out_layer.bias",
    # Final layer
    "final_layer.linear.weight": "proj_out.weight",
    "final_layer.linear.bias": "proj_out.bias",
    "final_layer.adaLN_modulation.1.weight": "norm_out.adaLN_modulation.1.weight",
    "final_layer.adaLN_modulation.1.bias": "norm_out.adaLN_modulation.1.bias",
}

# Block-level mappings for BFL double_blocks
BFL_DOUBLE_BLOCK_MAPPING = {
    # Note: In BFL format, QKV are already FUSED (img_attn.qkv.weight contains Q+K+V concatenated)
    # We need to UNFUSE them first, then REFUSE in Primus/Megatron GQA format
    "img_attn.qkv.weight": "self_attention.linear_qkv.weight",  # Will need special handling
    "img_attn.qkv.bias": "self_attention.linear_qkv.bias",
    "img_attn.proj.weight": "self_attention.linear_proj.weight",
    "img_attn.proj.bias": "self_attention.linear_proj.bias",
    "txt_attn.qkv.weight": "self_attention.added_linear_qkv.weight",  # Will need special handling
    "txt_attn.qkv.bias": "self_attention.added_linear_qkv.bias",
    "txt_attn.proj.weight": "self_attention.added_linear_proj.weight",
    "txt_attn.proj.bias": "self_attention.added_linear_proj.bias",
    # QK LayerNorms
    "img_attn.norm.query_norm.scale": "self_attention.q_layernorm.weight",
    "img_attn.norm.key_norm.scale": "self_attention.k_layernorm.weight",
    "txt_attn.norm.query_norm.scale": "self_attention.added_q_layernorm.weight",
    "txt_attn.norm.key_norm.scale": "self_attention.added_k_layernorm.weight",
    # Image MLPs
    "img_mlp.0.weight": "mlp.linear_fc1.weight",
    "img_mlp.0.bias": "mlp.linear_fc1.bias",
    "img_mlp.2.weight": "mlp.linear_fc2.weight",
    "img_mlp.2.bias": "mlp.linear_fc2.bias",
    # Text MLPs
    "txt_mlp.0.weight": "context_mlp.linear_fc1.weight",
    "txt_mlp.0.bias": "context_mlp.linear_fc1.bias",
    "txt_mlp.2.weight": "context_mlp.linear_fc2.weight",
    "txt_mlp.2.bias": "context_mlp.linear_fc2.bias",
    # Modulation (AdaLN)
    "img_mod.lin.weight": "adaln.adaLN_modulation.1.weight",
    "img_mod.lin.bias": "adaln.adaLN_modulation.1.bias",
    "txt_mod.lin.weight": "adaln_context.adaLN_modulation.1.weight",
    "txt_mod.lin.bias": "adaln_context.adaLN_modulation.1.bias",
}

# Block-level mappings for BFL single_blocks
BFL_SINGLE_BLOCK_MAPPING = {
    # Note: linear1 in BFL is FUSED [Q, K, V, MLP] - need to split
    "linear1.weight": None,  # Special handling: split into QKV + MLP
    "linear1.bias": None,
    # Note: linear2 in BFL is FUSED [MLP_out, proj_out] - need to split
    "linear2.weight": None,  # Special handling: split into proj_out only (simplified in some versions)
    "linear2.bias": None,
    # Modulation
    "modulation.lin.weight": "adaln.adaLN_modulation.1.weight",
    "modulation.lin.bias": "adaln.adaLN_modulation.1.bias",
    # QK LayerNorms (renamed 'norm' in BFL single blocks)
    "norm.query_norm.scale": "self_attention.q_layernorm.weight",
    "norm.key_norm.scale": "self_attention.k_layernorm.weight",
}


def detect_checkpoint_format(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Detect checkpoint format by inspecting keys.

    Args:
        state_dict: Loaded checkpoint state dictionary

    Returns:
        'bfl_native': Black Forest Labs native format (img_in, txt_in, time_in)
        'hf_diffusers': HuggingFace Diffusers format (x_embedder, context_embedder, time_text_embed)

    Reference: diffusers Flux checkpoint conversion
    The BFL native format is what Black Forest Labs releases directly.
    The HF Diffusers format is what you get after saving from diffusers library.
    """
    sample_keys = list(state_dict.keys())

    # Check for BFL native format markers
    # BFL uses: img_in, txt_in, time_in, vector_in, guidance_in
    if any(k.startswith("img_in.") for k in sample_keys):
        return "bfl_native"
    if any(k.startswith("txt_in.") for k in sample_keys):
        return "bfl_native"
    if any("time_in.in_layer" in k for k in sample_keys):
        return "bfl_native"

    # Check for HF Diffusers format markers
    # Diffusers uses: x_embedder, context_embedder, time_text_embed
    if any(k.startswith("x_embedder.") for k in sample_keys):
        return "hf_diffusers"
    if any("time_text_embed.timestep_embedder" in k for k in sample_keys):
        return "hf_diffusers"

    # Default to HF Diffusers (original behavior)
    return "hf_diffusers"


def _get_hf_token(token_file: Optional[str] = None) -> Optional[str]:
    """
    Get HuggingFace token with fallback options.

    Thin wrapper around the shared
    :func:`...preprocessing.auth.setup_hf_authentication` so the token-resolution
    priority chain (file with permission check -> HF_TOKEN env -> HF CLI login ->
    None) lives in one place. Returns None instead of raising so the converter
    can fall back to public-only access.

    Args:
        token_file: Optional path to token file

    Returns:
        Token string if found, None otherwise
    """
    from primus.backends.megatron.data.diffusion.preprocessing.auth import (
        HFAuthError,
        setup_hf_authentication,
    )

    try:
        return setup_hf_authentication(token_file=token_file, use_env=True)
    except HFAuthError:
        # Preserve fallback-to-public behavior for the converter rather than
        # hard-failing on a bad/insecure token file.
        return None


def convert_hf_checkpoint(
    checkpoint_path: Union[str, Path],
    flux_config,
    save_to: Optional[Union[str, Path]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert HuggingFace Flux checkpoint to Primus format.

    Supports both local paths and HuggingFace repo IDs. If a repo ID is provided
    (e.g., "black-forest-labs/FLUX.1-dev"), the checkpoint will be
    automatically downloaded from HuggingFace Hub.

    Args:
        checkpoint_path: Path to HF checkpoint OR HuggingFace repo ID
                        (e.g., "black-forest-labs/FLUX.1-dev/transformer")
        flux_config: FluxConfig instance with model architecture info
        save_to: Optional path to save converted checkpoint

    Returns:
        Dictionary of converted state dict (Primus format)

    Example:
        >>> from primus.backends.megatron.core.models.diffusion.flux import FluxConfig
        >>> config = FluxConfig.flux_12b()
        >>> # Auto-download from HuggingFace
        >>> primus_sd = convert_hf_checkpoint(
        ...     "black-forest-labs/FLUX.1-dev/transformer",
        ...     flux_config=config,
        ...     save_to="primus_flux_12b.safetensors"
        ... )
    """
    checkpoint_path_str = str(checkpoint_path)
    checkpoint_path_obj = Path(checkpoint_path)

    # Check if path exists locally first
    if checkpoint_path_obj.exists():
        # Local file/directory exists, use it directly
        pass
    else:
        # Path doesn't exist - check if it looks like a HuggingFace repo ID
        # HF repo IDs: don't start with /, ./, ../, and don't contain ..
        looks_like_hf_repo = (
            "/" in checkpoint_path_str
            and not checkpoint_path_str.startswith("/")
            and not checkpoint_path_str.startswith("./")
            and not checkpoint_path_str.startswith("../")
            and ".." not in checkpoint_path_str
            and not checkpoint_path_str.startswith(".")  # Avoid hidden files/dirs
        )

        if looks_like_hf_repo:
            logger.info("Detected HuggingFace repo ID: %s", checkpoint_path_str)
            logger.info("Downloading from HuggingFace Hub...")

            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required for downloading checkpoints. "
                    "Install with: pip install huggingface_hub"
                )

            # Setup authentication: .hf_token file → HF_TOKEN env → HF CLI login
            token_file = Path(__file__).parents[6] / ".hf_token"
            hf_token = None
            if token_file.exists():
                logger.info("Using HuggingFace token from project root: %s", token_file)
                hf_token = _get_hf_token(token_file=str(token_file))
            else:
                # Falls back to HF_TOKEN env var and HF CLI login
                hf_token = _get_hf_token(token_file=None)

            # Parse repo_id and subfolder
            parts = checkpoint_path_str.split("/")
            if len(parts) >= 2:
                repo_id = "/".join(parts[:2])  # e.g., "black-forest-labs/FLUX.1-dev"
                subfolder = "/".join(parts[2:]) if len(parts) > 2 else None  # e.g., "transformer"
            else:
                repo_id = checkpoint_path_str
                subfolder = None

            # Download to cache
            cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))

            try:
                local_dir = snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[f"{subfolder}/*"] if subfolder else None,
                    cache_dir=cache_dir,
                    token=hf_token,
                    resume_download=True,
                )

                # Construct path to actual checkpoint
                checkpoint_path = Path(local_dir) / subfolder if subfolder else Path(local_dir)
                logger.info("Downloaded to: %s", checkpoint_path)

            except Exception as e:
                logger.error("Download failed: %s", e)
                if "401" in str(e) or "403" in str(e):
                    raise RuntimeError(
                        f"Authentication failed. Please set HuggingFace token using one of:\n"
                        f"  1. Create .hf_token file: echo 'your_token' > .hf_token && chmod 600 .hf_token\n"
                        f"  2. Set environment variable: export HF_TOKEN=your_token\n"
                        f"  3. Login via CLI: huggingface-cli login\n"
                        f"Get your token from: https://huggingface.co/settings/tokens\n"
                        f"Accept the model license at: https://huggingface.co/{repo_id}"
                    ) from e
                raise
        else:
            # Looks like a local path that doesn't exist - raise FileNotFoundError
            raise FileNotFoundError(
                f"Checkpoint file or directory not found: {checkpoint_path_str}\n"
                f"If this is a HuggingFace repo ID, ensure it follows the format 'org/repo' or 'org/repo/subfolder'"
            )

    logger.info("Loading HuggingFace checkpoint from: %s", checkpoint_path)

    # Load HF checkpoint
    hf_state_dict = {}
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        # Load all .safetensors files in directory
        safetensor_files = list(checkpoint_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {checkpoint_path}")

        for file in safetensor_files:
            logger.info("  Loading %s...", file.name)
            hf_state_dict.update(load_safetensors(str(file)))
    elif checkpoint_path.is_file():
        hf_state_dict = load_safetensors(str(checkpoint_path))
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loaded %d keys from HuggingFace checkpoint", len(hf_state_dict))

    # Detect checkpoint format
    checkpoint_format = detect_checkpoint_format(hf_state_dict)
    logger.info("Detected checkpoint format: %s", checkpoint_format)

    # Branch based on format
    if checkpoint_format == "bfl_native":
        return _convert_bfl_checkpoint(hf_state_dict, flux_config, save_to)
    else:
        return _convert_hf_diffusers_checkpoint(hf_state_dict, flux_config, save_to)


def _convert_hf_diffusers_checkpoint(
    hf_state_dict: Dict[str, torch.Tensor],
    flux_config,
    save_to: Optional[Union[str, Path]] = None,
) -> Dict[str, torch.Tensor]:
    """Convert HuggingFace Diffusers format checkpoint to Primus format."""
    # Convert to Primus format
    primus_state_dict = {}
    num_double_blocks = -1
    num_single_blocks = -1

    # First pass: Convert simple key mappings (ONLY double blocks and root keys)
    logger.info("Converting simple key mappings (double blocks and root keys)...")
    for hf_key, value in hf_state_dict.items():
        # Skip QKV weights - will handle separately
        if any(
            x in hf_key
            for x in [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.add_q_proj",
                "attn.add_k_proj",
                "attn.add_v_proj",
            ]
        ):
            continue

        # Skip ALL single block keys - will handle in separate pass AFTER determining num_double_blocks
        if hf_key.startswith("single_transformer_blocks"):
            continue

        # Map double blocks -> transformer.layers[0-18]
        if hf_key.startswith("transformer_blocks"):
            parts = hf_key.split(".")
            idx = int(parts[1])
            sub_key = ".".join(parts[2:])
            num_double_blocks = max(idx, num_double_blocks)

            if sub_key in FLUX_KEY_MAPPING["double_blocks"]:
                # New key format: transformer.layers.{idx} instead of double_blocks.{idx}
                primus_key = f"transformer.layers.{idx}.{FLUX_KEY_MAPPING['double_blocks'][sub_key]}"
                primus_state_dict[primus_key] = value

        # Map root-level keys
        elif hf_key in FLUX_KEY_MAPPING:
            # Special handling for norm_out: swap scale/shift halves
            # HF Diffusers stores [SCALE; SHIFT], but Primus/BFL native expects [SHIFT; SCALE]
            if hf_key == "norm_out.linear.weight":
                half_size = value.shape[0] // 2
                scale_half = value[:half_size, :]  # HF first half = SCALE
                shift_half = value[half_size:, :]  # HF second half = SHIFT
                # Swap to BFL native order: [SHIFT; SCALE]
                value = torch.cat([shift_half, scale_half], dim=0)
            elif hf_key == "norm_out.linear.bias":
                half_size = value.shape[0] // 2
                scale_half = value[:half_size]  # HF first half = SCALE
                shift_half = value[half_size:]  # HF second half = SHIFT
                # Swap to BFL native order: [SHIFT; SCALE]
                value = torch.cat([shift_half, scale_half], dim=0)

            primus_state_dict[FLUX_KEY_MAPPING[hf_key]] = value

    # Detect number of single blocks
    for hf_key in hf_state_dict.keys():
        if hf_key.startswith("single_transformer_blocks"):
            parts = hf_key.split(".")
            idx = int(parts[1])
            num_single_blocks = max(idx, num_single_blocks)

    logger.info("Found %d double blocks, %d single blocks", num_double_blocks + 1, num_single_blocks + 1)

    # Second pass: Convert single block simple keys (NOW num_double_blocks is known!)
    logger.info("Converting single block simple keys...")
    for hf_key, value in hf_state_dict.items():
        if not hf_key.startswith("single_transformer_blocks"):
            continue

        # Skip QKV and proj_out - will handle separately in later passes
        if any(x in hf_key for x in ["attn.to_q", "attn.to_k", "attn.to_v", "proj_out"]):
            continue

        parts = hf_key.split(".")
        idx = int(parts[1])
        sub_key = ".".join(parts[2:])

        if sub_key in FLUX_KEY_MAPPING["single_blocks"]:
            layer_idx = num_double_blocks + 1 + idx  # NOW num_double_blocks is determined!
            primus_key = f"transformer.layers.{layer_idx}.{FLUX_KEY_MAPPING['single_blocks'][sub_key]}"
            primus_state_dict[primus_key] = value

    # Third pass: Fuse QKV weights for double blocks (now transformer.layers[0-18])
    logger.info("Fusing QKV weights for double blocks...")
    for i in range(num_double_blocks + 1):
        # Main attention QKV
        q_key = f"transformer_blocks.{i}.attn.to_q.weight"
        k_key = f"transformer_blocks.{i}.attn.to_k.weight"
        v_key = f"transformer_blocks.{i}.attn.to_v.weight"

        fused_qkv = _fuse_qkv_weights(
            flux_config, hf_state_dict[q_key], hf_state_dict[k_key], hf_state_dict[v_key]
        )
        # New key format: transformer.layers.{i} instead of double_blocks.{i}
        primus_state_dict[f"transformer.layers.{i}.self_attention.linear_qkv.weight"] = fused_qkv

        # QKV bias
        q_bias_key = f"transformer_blocks.{i}.attn.to_q.bias"
        k_bias_key = f"transformer_blocks.{i}.attn.to_k.bias"
        v_bias_key = f"transformer_blocks.{i}.attn.to_v.bias"

        fused_qkv_bias = _fuse_qkv_bias(
            flux_config, hf_state_dict[q_bias_key], hf_state_dict[k_bias_key], hf_state_dict[v_bias_key]
        )
        primus_state_dict[f"transformer.layers.{i}.self_attention.linear_qkv.bias"] = fused_qkv_bias

        # Context (added) attention QKV
        add_q_key = f"transformer_blocks.{i}.attn.add_q_proj.weight"
        add_k_key = f"transformer_blocks.{i}.attn.add_k_proj.weight"
        add_v_key = f"transformer_blocks.{i}.attn.add_v_proj.weight"

        fused_add_qkv = _fuse_qkv_weights(
            flux_config, hf_state_dict[add_q_key], hf_state_dict[add_k_key], hf_state_dict[add_v_key]
        )
        primus_state_dict[f"transformer.layers.{i}.self_attention.added_linear_qkv.weight"] = fused_add_qkv

        # Added QKV bias
        add_q_bias_key = f"transformer_blocks.{i}.attn.add_q_proj.bias"
        add_k_bias_key = f"transformer_blocks.{i}.attn.add_k_proj.bias"
        add_v_bias_key = f"transformer_blocks.{i}.attn.add_v_proj.bias"

        fused_add_qkv_bias = _fuse_qkv_bias(
            flux_config,
            hf_state_dict[add_q_bias_key],
            hf_state_dict[add_k_bias_key],
            hf_state_dict[add_v_bias_key],
        )
        primus_state_dict[f"transformer.layers.{i}.self_attention.added_linear_qkv.bias"] = fused_add_qkv_bias

    # Fourth pass: Fuse QKV and split proj_out for single blocks (now transformer.layers[19+])
    logger.info("Fusing QKV and splitting proj_out for single blocks...")
    for i in range(num_single_blocks + 1):
        # Calculate layer index with offset
        layer_idx = num_double_blocks + 1 + i

        # QKV
        q_key = f"single_transformer_blocks.{i}.attn.to_q.weight"
        k_key = f"single_transformer_blocks.{i}.attn.to_k.weight"
        v_key = f"single_transformer_blocks.{i}.attn.to_v.weight"

        fused_qkv = _fuse_qkv_weights(
            flux_config, hf_state_dict[q_key], hf_state_dict[k_key], hf_state_dict[v_key]
        )
        # New key format with offset
        primus_state_dict[f"transformer.layers.{layer_idx}.self_attention.linear_qkv.weight"] = fused_qkv

        # QKV bias
        q_bias_key = f"single_transformer_blocks.{i}.attn.to_q.bias"
        k_bias_key = f"single_transformer_blocks.{i}.attn.to_k.bias"
        v_bias_key = f"single_transformer_blocks.{i}.attn.to_v.bias"

        fused_qkv_bias = _fuse_qkv_bias(
            flux_config, hf_state_dict[q_bias_key], hf_state_dict[k_bias_key], hf_state_dict[v_bias_key]
        )
        primus_state_dict[f"transformer.layers.{layer_idx}.self_attention.linear_qkv.bias"] = fused_qkv_bias

        # Split proj_out (combined MLP+Attention output in HF)
        # HF format: [out_dim, hidden*2] where [:, :3072] is attention, [:, 3072:] is MLP
        proj_out_weight = hf_state_dict[f"single_transformer_blocks.{i}.proj_out.weight"]
        proj_out_bias = hf_state_dict[f"single_transformer_blocks.{i}.proj_out.bias"]

        # Split weight at hidden_size (3072 for Flux)
        hidden_size = flux_config.hidden_size
        attn_proj = proj_out_weight[:, :hidden_size].clone()
        mlp_proj = proj_out_weight[:, hidden_size:].clone()

        primus_state_dict[f"transformer.layers.{layer_idx}.self_attention.linear_proj.weight"] = attn_proj
        primus_state_dict[f"transformer.layers.{layer_idx}.mlp.linear_fc2.weight"] = mlp_proj

        primus_state_dict[f"transformer.layers.{layer_idx}.mlp.linear_fc2.bias"] = proj_out_bias.clone()

    logger.info("Conversion complete! Primus state dict has %d keys", len(primus_state_dict))

    # Save if requested
    if save_to:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving to: %s", save_to)
        save_safetensors(primus_state_dict, str(save_to))
        logger.info("Saved successfully!")

    return primus_state_dict


def _convert_bfl_checkpoint(
    bfl_state_dict: Dict[str, torch.Tensor],
    flux_config,
    save_to: Optional[Union[str, Path]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert Black Forest Labs native format checkpoint to Primus format.

    Reference: diffusers Flux checkpoint conversion

    Key differences from HF Diffusers format:
    - QKV are already FUSED in BFL (img_attn.qkv.weight contains Q+K+V concatenated)
    - Single blocks have fused linear1 (Q, K, V, MLP) and linear2 (proj_out)
    """
    logger.info("Converting BFL native format to Primus...")
    primus_state_dict = {}
    num_double_blocks = -1
    num_single_blocks = -1

    # First pass: Convert simple root-level mappings
    logger.info("Converting root-level keys...")
    for bfl_key, value in bfl_state_dict.items():
        # Skip block-level keys - will handle in second pass
        if bfl_key.startswith("double_blocks.") or bfl_key.startswith("single_blocks."):
            continue

        # Map root-level keys
        if bfl_key in BFL_KEY_MAPPING:
            primus_key = BFL_KEY_MAPPING[bfl_key]
            primus_state_dict[primus_key] = value

    # Detect number of blocks
    for key in bfl_state_dict.keys():
        if key.startswith("double_blocks."):
            idx = int(key.split(".")[1])
            num_double_blocks = max(idx, num_double_blocks)
        elif key.startswith("single_blocks."):
            idx = int(key.split(".")[1])
            num_single_blocks = max(idx, num_single_blocks)

    logger.info("Found %d double blocks, %d single blocks", num_double_blocks + 1, num_single_blocks + 1)

    # Second pass: Convert double blocks -> transformer.layers[0-18]
    logger.info("Converting double blocks (unfusing and refusing QKV)...")
    for i in range(num_double_blocks + 1):
        block_prefix_bfl = f"double_blocks.{i}"
        block_prefix_primus = f"transformer.layers.{i}"  # New key format

        # Convert simple mappings first
        for bfl_suffix, primus_suffix in BFL_DOUBLE_BLOCK_MAPPING.items():
            if primus_suffix is None:
                continue  # Skip special handling keys

            bfl_key = f"{block_prefix_bfl}.{bfl_suffix}"
            if bfl_key in bfl_state_dict:
                primus_key = f"{block_prefix_primus}.{primus_suffix}"

                # Special handling for QKV (need to unfuse BFL QKV, then refuse in Primus format)
                if "qkv" in bfl_suffix and ("weight" in bfl_suffix or "bias" in bfl_suffix):
                    continue  # Handle separately below
                else:
                    primus_state_dict[primus_key] = bfl_state_dict[bfl_key]

        # Handle img_attn QKV (unfuse from BFL concat format, then refuse in Primus GQA format)
        # BFL format: [Q; K; V] concatenated along dim 0
        # Reference: diffusers Flux checkpoint conversion
        img_qkv_weight = bfl_state_dict[f"{block_prefix_bfl}.img_attn.qkv.weight"]
        img_qkv_bias = bfl_state_dict[f"{block_prefix_bfl}.img_attn.qkv.bias"]

        # Unfuse: split into Q, K, V
        img_q, img_k, img_v = torch.chunk(img_qkv_weight, 3, dim=0)
        img_q_bias, img_k_bias, img_v_bias = torch.chunk(img_qkv_bias, 3, dim=0)

        # Refuse in Primus GQA format
        fused_img_qkv = _fuse_qkv_weights(flux_config, img_q, img_k, img_v)
        fused_img_qkv_bias = _fuse_qkv_bias(flux_config, img_q_bias, img_k_bias, img_v_bias)

        primus_state_dict[f"{block_prefix_primus}.self_attention.linear_qkv.weight"] = fused_img_qkv
        primus_state_dict[f"{block_prefix_primus}.self_attention.linear_qkv.bias"] = fused_img_qkv_bias

        # Handle txt_attn QKV (same process)
        txt_qkv_weight = bfl_state_dict[f"{block_prefix_bfl}.txt_attn.qkv.weight"]
        txt_qkv_bias = bfl_state_dict[f"{block_prefix_bfl}.txt_attn.qkv.bias"]

        txt_q, txt_k, txt_v = torch.chunk(txt_qkv_weight, 3, dim=0)
        txt_q_bias, txt_k_bias, txt_v_bias = torch.chunk(txt_qkv_bias, 3, dim=0)

        fused_txt_qkv = _fuse_qkv_weights(flux_config, txt_q, txt_k, txt_v)
        fused_txt_qkv_bias = _fuse_qkv_bias(flux_config, txt_q_bias, txt_k_bias, txt_v_bias)

        primus_state_dict[f"{block_prefix_primus}.self_attention.added_linear_qkv.weight"] = fused_txt_qkv
        primus_state_dict[f"{block_prefix_primus}.self_attention.added_linear_qkv.bias"] = fused_txt_qkv_bias

    # Third pass: Convert single blocks -> transformer.layers[19+]
    # BFL single blocks have:
    # - linear1: fused [Q, K, V, MLP] - need to split and handle separately
    # - linear2: just proj_out (simpler than HF Diffusers which has MLP+proj)
    # Reference: diffusers Flux checkpoint conversion
    logger.info("Converting single blocks (splitting linear1)...")
    for i in range(num_single_blocks + 1):
        block_prefix_bfl = f"single_blocks.{i}"
        # Calculate layer index with offset (num_double_blocks + 1 + i)
        layer_idx = num_double_blocks + 1 + i
        block_prefix_primus = f"transformer.layers.{layer_idx}"  # New key format

        # Convert simple mappings (modulation, norms)
        for bfl_suffix, primus_suffix in BFL_SINGLE_BLOCK_MAPPING.items():
            if primus_suffix is None:
                continue  # Skip special handling

            bfl_key = f"{block_prefix_bfl}.{bfl_suffix}"
            if bfl_key in bfl_state_dict:
                primus_key = f"{block_prefix_primus}.{primus_suffix}"
                primus_state_dict[primus_key] = bfl_state_dict[bfl_key]

        # Handle linear1: fused [Q, K, V, MLP]
        linear1_weight = bfl_state_dict[f"{block_prefix_bfl}.linear1.weight"]
        linear1_bias = bfl_state_dict[f"{block_prefix_bfl}.linear1.bias"]

        # Split along dim 0: [Q, K, V, MLP]
        hidden_size = flux_config.hidden_size
        mlp_hidden_dim = int(hidden_size * 4.0)  # mlp_ratio = 4.0
        split_sizes = (hidden_size, hidden_size, hidden_size, mlp_hidden_dim)

        q, k, v, mlp = torch.split(linear1_weight, split_sizes, dim=0)
        q_bias, k_bias, v_bias, mlp_bias = torch.split(linear1_bias, split_sizes, dim=0)

        # Fuse Q, K, V in Primus GQA format
        fused_qkv = _fuse_qkv_weights(flux_config, q, k, v)
        fused_qkv_bias = _fuse_qkv_bias(flux_config, q_bias, k_bias, v_bias)

        primus_state_dict[f"{block_prefix_primus}.self_attention.linear_qkv.weight"] = fused_qkv
        primus_state_dict[f"{block_prefix_primus}.self_attention.linear_qkv.bias"] = fused_qkv_bias

        # MLP goes to linear_fc1
        primus_state_dict[f"{block_prefix_primus}.mlp.linear_fc1.weight"] = mlp
        primus_state_dict[f"{block_prefix_primus}.mlp.linear_fc1.bias"] = mlp_bias

        # Handle linear2: In BFL, this is just proj_out (not fused with MLP like in HF Diffusers)
        linear2_weight = bfl_state_dict[f"{block_prefix_bfl}.linear2.weight"]
        linear2_bias = bfl_state_dict[f"{block_prefix_bfl}.linear2.bias"]

        # In BFL format, linear2 is [out_channels, hidden_size*2] where first half is attn proj, second is mlp proj
        # Split at hidden_size
        attn_proj = linear2_weight[:, :hidden_size].clone()
        mlp_proj = linear2_weight[:, hidden_size:].clone()

        primus_state_dict[f"{block_prefix_primus}.self_attention.linear_proj.weight"] = attn_proj
        primus_state_dict[f"{block_prefix_primus}.mlp.linear_fc2.weight"] = mlp_proj

        primus_state_dict[f"{block_prefix_primus}.mlp.linear_fc2.bias"] = linear2_bias.clone()

    logger.info("Conversion complete! Primus state dict has %d keys", len(primus_state_dict))

    # Save if requested
    if save_to:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving to: %s", save_to)
        save_safetensors(primus_state_dict, str(save_to))
        logger.info("Saved successfully!")

    return primus_state_dict


__all__ = [
    "convert_hf_checkpoint",
    "FLUX_KEY_MAPPING",
    "BFL_KEY_MAPPING",
    "BFL_DOUBLE_BLOCK_MAPPING",
    "BFL_SINGLE_BLOCK_MAPPING",
    "detect_checkpoint_format",
    "_fuse_qkv_weights",
    "_fuse_qkv_bias",
]
