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

_FP8_DOUBLE_ATTN_PROJ_SUFFIXES = {
    "img_attn.proj",
    "txt_attn.proj",
}
_FP8_DOUBLE_MLP_SUFFIXES = {
    "img_mlp.0",
    "img_mlp.2",
    "txt_mlp.0",
    "txt_mlp.2",
}
_FP8_SELECTIVE_GEMM_SHAPES = {
    (8192, 3072, 12288),
    (16384, 3072, 15360),
}


@torch.library.custom_op("primus::flux_flydsl_scaled_mm", mutates_args=())
def _flux_flydsl_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    from primus_turbo.flydsl.gemm.gemm_fp8_kernel import gemm_fp8_tensorwise_flydsl_kernel

    return gemm_fp8_tensorwise_flydsl_kernel(
        a,
        a_scale,
        b.t(),
        b_scale,
        trans_a=False,
        trans_b=True,
        out_dtype=torch.bfloat16,
    )


@_flux_flydsl_scaled_mm.register_fake
def _(a, b, a_scale, b_scale):
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16)


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
    float8_recipe = str(cfg_dict.get("float8_recipe") or "").strip().lower()
    if float8_recipe not in {"", "tensorwise"}:
        raise ValueError(
            f"Unsupported FLUX float8_recipe={float8_recipe!r}; expected null or 'tensorwise'"
        )
    fp8_gemm_backend = str(cfg_dict.get("float8_gemm_backend") or "").strip().lower()
    if fp8_gemm_backend not in {"", "selective_triton", "selective_flydsl"}:
        raise ValueError(
            "Unsupported FLUX float8_gemm_backend="
            f"{fp8_gemm_backend!r}; expected null, 'selective_triton', or 'selective_flydsl'"
        )
    if fp8_gemm_backend and not float8_recipe:
        raise ValueError("FLUX float8_gemm_backend requires float8_recipe='tensorwise'")
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

    if float8_recipe:
        try:
            from torchao.float8 import (
                CastConfig,
                Float8LinearConfig,
                ScalingType,
                convert_to_float8_training,
            )
        except ImportError as exc:
            raise ImportError("TorchAO is required for FLUX tensor-wise FP8 training") from exc

        if fp8_gemm_backend:
            os.environ["PRIMUS_FLUX_FP8_GEMM_BACKEND"] = fp8_gemm_backend
        else:
            os.environ.pop("PRIMUS_FLUX_FP8_GEMM_BACKEND", None)

        if fp8_gemm_backend == "selective_triton":
            from torch._inductor.kernel import mm

            if not getattr(mm, "_PRIMUS_FLUX_SELECTIVE_TRITON", False):
                raise RuntimeError("selective_triton requires the FLUX FP8 Inductor image patch")
            logger.info(f"Using Triton FP8 GEMM for shapes {sorted(_FP8_SELECTIVE_GEMM_SHAPES)}")

        if fp8_gemm_backend == "selective_flydsl":
            import torchao.float8.float8_ops as float8_ops

            original_addmm = float8_ops.addmm_float8_unwrapped

            def selective_flydsl_addmm(
                a_data,
                a_scale,
                b_data,
                b_scale,
                output_dtype,
                output_scale=None,
                bias=None,
                use_fast_accum=False,
            ):
                shape = (a_data.shape[0], b_data.shape[1], a_data.shape[1])
                if (
                    shape in _FP8_SELECTIVE_GEMM_SHAPES
                    and output_dtype == torch.bfloat16
                    and output_scale is None
                    and bias is None
                ):
                    return _flux_flydsl_scaled_mm(
                        a_data,
                        b_data,
                        a_scale.reciprocal(),
                        b_scale.reciprocal(),
                    )
                return original_addmm(
                    a_data,
                    a_scale,
                    b_data,
                    b_scale,
                    output_dtype,
                    output_scale,
                    bias,
                    use_fast_accum,
                )

            float8_ops.addmm_float8_unwrapped = selective_flydsl_addmm
            logger.info(f"Using FlyDSL FP8 GEMM for shapes {sorted(_FP8_SELECTIVE_GEMM_SHAPES)}")

        full_wgrad_fqns: list[str] = []
        high_precision_wgrad_fqns: list[str] = []

        def module_kind(module: torch.nn.Module, fqn: str) -> str | None:
            if type(module) is not torch.nn.Linear:
                return None
            parts = fqn.split(".", 2)
            if len(parts) != 3:
                return None
            if parts[0] == "double_blocks":
                if parts[2] == "img_attn.qkv":
                    return "qkv"
                if parts[2] == "txt_attn.qkv":
                    return "qkv"
                if parts[2] in _FP8_DOUBLE_ATTN_PROJ_SUFFIXES:
                    return "full"
                if parts[2] in _FP8_DOUBLE_MLP_SUFFIXES:
                    return "full"
            if parts[0] == "single_blocks" and parts[2] in {"linear1", "linear2"}:
                return "full"
            return None

        def full_wgrad_filter(module: torch.nn.Module, fqn: str) -> bool:
            selected = module_kind(module, fqn) == "full"
            if selected:
                full_wgrad_fqns.append(fqn)
            return selected

        def high_precision_wgrad_filter(module: torch.nn.Module, fqn: str) -> bool:
            selected = module_kind(module, fqn) == "qkv"
            if selected:
                high_precision_wgrad_fqns.append(fqn)
            return selected

        dit = convert_to_float8_training(
            dit,
            module_filter_fn=full_wgrad_filter,
            config=Float8LinearConfig(
                pad_inner_dim=False,
                enable_fsdp_float8_all_gather=False,
            ),
        )
        dit = convert_to_float8_training(
            dit,
            module_filter_fn=high_precision_wgrad_filter,
            config=Float8LinearConfig(
                cast_config_input_for_grad_weight=CastConfig(
                    scaling_type=ScalingType.DISABLED
                ),
                cast_config_grad_output_for_grad_weight=CastConfig(
                    scaling_type=ScalingType.DISABLED
                ),
                pad_inner_dim=False,
                enable_fsdp_float8_all_gather=False,
            ),
        )
        expected_full_count = len(dit.double_blocks) * 6 + len(dit.single_blocks) * 2
        expected_high_precision_count = len(dit.double_blocks) * 2
        if (
            len(full_wgrad_fqns) != expected_full_count
            or len(high_precision_wgrad_fqns) != expected_high_precision_count
        ):
            raise RuntimeError(
                "FLUX FP8 converted "
                f"{len(full_wgrad_fqns)} full-wgrad and "
                f"{len(high_precision_wgrad_fqns)} high-precision-wgrad Linear modules; "
                f"expected {expected_full_count} and {expected_high_precision_count}"
            )
        logger.info(
            "Enabled TorchAO dynamic tensor-wise FP8 for "
            f"{len(full_wgrad_fqns) + len(high_precision_wgrad_fqns)} FLUX block Linear modules; "
            f"wgrad=FP8 for {len(full_wgrad_fqns)} and high precision for "
            f"{len(high_precision_wgrad_fqns)} QKV modules"
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
