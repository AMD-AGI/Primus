###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import glob
import os
from typing import Any

import torch
from safetensors.torch import load_file as safe_load_file

from primus.backends.diffusion.models.flux.adapter import FluxForTraining
from primus.backends.diffusion.models.flux.autoencoder import (
    AutoEncoderParams,
    load_autoencoder,
)
from primus.backends.diffusion.models.flux.conditioner import HFEmbedder
from primus.backends.diffusion.models.flux.configuration_flux import FluxTrainingConfig
from primus.backends.diffusion.models.flux.model import (
    Flux,
    flux_1_dev_params,
    flux_1_schnell_params,
)
from primus.backends.diffusion.models.flux.train_pipeline import (
    FluxFlowMatchTrainPipeline,
    FluxFlowMatchTrainPipelineConfig,
)
from primus.backends.diffusion.utils.log import logger
from primus.backends.diffusion.utils.train_utils import count_parameters

_FLUX_PRESET_ALIASES = {
    "flux-schnell": "flux-schnell",
    "flux.1-schnell": "flux-schnell",
    "flux1-schnell": "flux-schnell",
    "flux-dev": "flux-dev",
    "flux.1-dev": "flux-dev",
    "flux1-dev": "flux-dev",
}


def _strip_known_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "dit.", "model.")
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        stripped = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix) :]
                    changed = True
        out[stripped] = value
    return out


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        return dict(safe_load_file(path))
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")
    return obj


def _candidate_weight_files(path: str, *, default_filename: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.exists(path):
        resolved = _resolve_hf_checkpoint(path, default_filename=default_filename)
        if resolved:
            return [resolved]
    candidates: list[str] = []
    for fname in (
        "flux1-schnell.safetensors",
        "flux1-dev.safetensors",
        "dit_model.safetensors",
        "model.safetensors",
    ):
        candidate = os.path.join(path, fname)
        if os.path.exists(candidate):
            candidates.append(candidate)
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(path, "*.bin")))
    return candidates


def _resolve_hf_checkpoint(path_or_repo_file: str, *, default_filename: str) -> str | None:
    if path_or_repo_file.startswith(("/", "./", "../", "~")):
        return None
    parts = path_or_repo_file.split("/")
    if len(parts) == 2 and parts[-1].endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt")):
        return None
    if len(parts) < 2:
        return None
    if len(parts) >= 3:
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
    else:
        repo_id = path_or_repo_file
        filename = default_filename
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


def _load_flux_weights(dit: torch.nn.Module, pretrained_path: str, *, default_filename: str) -> None:
    candidates = _candidate_weight_files(pretrained_path, default_filename=default_filename)
    if not candidates:
        raise FileNotFoundError(f"No FLUX DiT weights found under {pretrained_path}")

    merged: dict[str, torch.Tensor] = {}
    for ckpt in candidates:
        merged.update(_strip_known_prefixes(_load_state_dict(ckpt)))

    result = dit.load_state_dict(merged, strict=False)
    logger.info(
        "Loaded FLUX DiT weights. "
        f"files={len(candidates)} missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}"
    )


def _build_flux_dit(params) -> Flux:
    local_rank = os.environ.get("LOCAL_RANK")
    use_cuda = local_rank is not None and torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}") if use_cuda else torch.device("cpu")
    init_seed = torch.cuda.initial_seed() if use_cuda else torch.initial_seed()
    with torch.device(device):
        dit = Flux(params)
    # Constructor defaults consume RNG even though explicit TorchTitan
    # initialization overwrites them. Reset so init_weights starts at the
    # configured common model seed, as it does with TorchTitan meta creation.
    if use_cuda:
        torch.cuda.manual_seed(init_seed)
    else:
        torch.manual_seed(init_seed)
    dit.init_weights()
    return dit


def build_flux_model(model_config: dict[str, Any]):
    """
    Build a FLUX model from the selected model preset.

    `model_preset` is injected by the registry from `model.name` for Primus
    configs such as `flux.1-dev` and `flux.1-schnell`.
    """
    cfg_dict: dict[str, Any] = dict(model_config.get("config", {}) or {})
    low_precision_provider = str(cfg_dict.get("low_precision_provider") or "").strip().lower()
    low_precision_recipe = str(cfg_dict.get("low_precision_recipe") or "").strip().lower()
    if (low_precision_provider, low_precision_recipe) not in {
        ("", ""),
        ("torchao", "mxfp8"),
        ("primus_turbo", "mxfp8"),
    }:
        raise ValueError(
            "Unsupported FLUX low-precision configuration: "
            f"provider={low_precision_provider!r}, recipe={low_precision_recipe!r}"
        )
    preset_name = str(model_config.get("model_preset") or cfg_dict.get("model_preset") or "flux.1-schnell")
    preset = _FLUX_PRESET_ALIASES.get(preset_name.lower(), preset_name)

    params_overrides = dict(cfg_dict.get("params", {}) or {})
    if preset == "flux-dev":
        params = flux_1_dev_params(**params_overrides)
    elif preset == "flux-schnell":
        params = flux_1_schnell_params(**params_overrides)
    else:
        raise ValueError(
            "Unsupported FLUX model_preset="
            f"{preset_name!r}; expected one of: 'flux.1-dev', 'flux.1-schnell'"
        )
    dit = _build_flux_dit(params)

    pretrained_path = model_config.get("load_from_pretrained_path") or model_config.get("pretrained_path")
    if pretrained_path:
        logger.info(f"Loading FLUX DiT weights from {pretrained_path}")
        default_filename = "flux1-dev.safetensors" if preset == "flux-dev" else "flux1-schnell.safetensors"
        _load_flux_weights(dit, pretrained_path, default_filename=default_filename)

    if low_precision_recipe == "mxfp8":
        if low_precision_provider == "torchao":
            try:
                from torchao.prototype.moe_training.mxfp8_linear import MXFP8Linear
            except ImportError as exc:
                raise ImportError("TorchAO MXFP8 training support is required") from exc
        else:
            try:
                from primus_turbo.pytorch.core.low_precision import (
                    Float8QuantConfig,
                    Format,
                    ScaleDtype,
                    ScalingGranularity,
                )
                from primus_turbo.pytorch.modules import Float8Linear
            except ImportError as exc:
                raise ImportError("Primus-Turbo MXFP8 training support is required") from exc
            turbo_config = Float8QuantConfig(
                format=Format.E4M3,
                granularity=ScalingGranularity.MX_BLOCKWISE,
                scale_dtype=ScaleDtype.E8M0,
                block_size=32,
            )

        replacements = []
        for fqn, module in dit.named_modules():
            if type(module) is not torch.nn.Linear:
                continue
            parts = fqn.split(".", 2)
            if len(parts) != 3:
                continue
            suffix = parts[2]
            selected = (
                parts[0] == "double_blocks"
                and suffix
                in {
                    "img_attn.qkv",
                    "img_attn.proj",
                    "img_mlp.0",
                    "img_mlp.2",
                    "txt_attn.qkv",
                    "txt_attn.proj",
                    "txt_mlp.0",
                    "txt_mlp.2",
                }
            ) or (
                parts[0] == "single_blocks" and suffix in {"linear1", "linear2"}
            )
            if selected:
                replacements.append((fqn, module, suffix.endswith("attn.qkv")))

        for fqn, module, high_precision_wgrad in replacements:
            devices = [module.weight.device] if module.weight.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                common = {
                    "bias": module.bias is not None,
                    "device": module.weight.device,
                    "dtype": module.weight.dtype,
                    "wgrad_with_hp": high_precision_wgrad,
                }
                if low_precision_provider == "torchao":
                    replacement = MXFP8Linear(module.in_features, module.out_features, **common)
                else:
                    replacement = Float8Linear(
                        module.in_features, module.out_features, config=turbo_config, **common
                    )
            with torch.no_grad():
                replacement.weight.copy_(module.weight)
                if module.bias is not None:
                    replacement.bias.copy_(module.bias)
            dit.set_submodule(fqn, replacement)

        expected = len(dit.double_blocks) * 8 + len(dit.single_blocks) * 2
        high_precision_wgrad_count = sum(item[2] for item in replacements)
        expected_high_precision = len(dit.double_blocks) * 2
        if len(replacements) != expected or high_precision_wgrad_count != expected_high_precision:
            raise RuntimeError(
                f"FLUX MXFP8 selected {len(replacements)} Linear modules "
                f"({high_precision_wgrad_count} high-precision wgrad); expected "
                f"{expected} ({expected_high_precision} high-precision wgrad)"
            )
        logger.info(
            f"Enabled {low_precision_provider} MXFP8 for {len(replacements)} FLUX block Linear modules; "
            f"wgrad=MXFP8 for {len(replacements) - high_precision_wgrad_count} and "
            f"high precision for {high_precision_wgrad_count} QKV modules"
        )

    encoder_cfg = dict(model_config.get("encoder", {}) or cfg_dict.get("encoder", {}) or {})
    dtype = torch.bfloat16
    t5_encoder = None
    clip_encoder = None
    autoencoder = None
    if encoder_cfg.get("t5_encoder"):
        t5_encoder = HFEmbedder(
            str(encoder_cfg["t5_encoder"]),
            max_length=int(encoder_cfg.get("max_t5_length", 256)),
            torch_dtype=dtype,
        )
    if encoder_cfg.get("clip_encoder"):
        clip_encoder = HFEmbedder(
            str(encoder_cfg["clip_encoder"]),
            max_length=int(encoder_cfg.get("max_clip_length", 77)),
            torch_dtype=dtype,
        )
    if encoder_cfg.get("autoencoder"):
        ae_params = AutoEncoderParams(
            resolution=int(encoder_cfg.get("resolution", 256)),
            scale_factor=float(cfg_dict.get("autoencoder_scale_factor", 0.3611)),
            shift_factor=float(cfg_dict.get("autoencoder_shift_factor", 0.1159)),
        )
        autoencoder = load_autoencoder(
            str(encoder_cfg["autoencoder"]),
            ae_params,
            dtype=dtype,
            sample_z=bool(encoder_cfg.get("sample_z", True)),
        )

    training_cfg = FluxTrainingConfig(
        model_preset=preset,
        trainable_modules=cfg_dict.get("trainable_modules", "dit"),
        guidance=None if not params.guidance_embed else float(cfg_dict.get("guidance", 1.0)),
        autoencoder_scale_factor=float(cfg_dict.get("autoencoder_scale_factor", 0.3611)),
        autoencoder_shift_factor=float(cfg_dict.get("autoencoder_shift_factor", 0.1159)),
    )
    pipeline = FluxFlowMatchTrainPipeline(
        FluxFlowMatchTrainPipelineConfig(
            autoencoder_scale_factor=training_cfg.autoencoder_scale_factor,
            autoencoder_shift_factor=training_cfg.autoencoder_shift_factor,
            guidance=training_cfg.guidance,
        )
    )
    model = FluxForTraining(
        dit=dit,
        train_pipeline=pipeline,
        model_config=training_cfg,
        autoencoder=autoencoder,
        t5_encoder=t5_encoder,
        clip_encoder=clip_encoder,
        raw_config={
            "model_config": model_config,
            "flux_params": params.to_dict(),
        },
        trainable_modules=training_cfg.trainable_modules,
    )
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Built FLUX model: total={total_params:,} trainable={trainable_params:,}")
    return model
