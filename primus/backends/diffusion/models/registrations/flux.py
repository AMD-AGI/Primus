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
_MXFP4_RECIPES = {"pareto_a", "pareto_b", "custom"}
_MXFP4_PRECISIONS = {"bf16", "mxfp4", "mxfp8"}
_MXFP4_DOUBLE_SUFFIXES = {
    "img_attn.qkv", "img_attn.proj", "img_mlp.0", "img_mlp.2",
    "txt_attn.qkv", "txt_attn.proj", "txt_mlp.0", "txt_mlp.2",
}
_MXFP4_SINGLE_SUFFIXES = {"linear1", "linear2"}
_MXFP4_BF16_SCOPES = {
    "none": (),
    "double_img_up_2_3": (("double_blocks", "img_mlp.0", 2, 3),),
    "double_img_up": (("double_blocks", "img_mlp.0", None, None),),
    "double_img_mlp_attn_out": tuple(
        ("double_blocks", suffix, None, None)
        for suffix in ("img_mlp.0", "img_mlp.2", "img_attn.proj")
    ),
    "double_img_all": tuple(
        ("double_blocks", suffix, None, None)
        for suffix in ("img_mlp.0", "img_mlp.2", "img_attn.qkv", "img_attn.proj")
    ),
    "double_img_all_txt_mlp_down": tuple(
        ("double_blocks", suffix, None, None)
        for suffix in ("img_mlp.0", "img_mlp.2", "img_attn.qkv", "img_attn.proj", "txt_mlp.2")
    ),
    "double_img_all_txt_mlp": tuple(
        ("double_blocks", suffix, None, None)
        for suffix in (
            "img_mlp.0", "img_mlp.2", "img_attn.qkv", "img_attn.proj",
            "txt_mlp.0", "txt_mlp.2",
        )
    ),
    "double_all": tuple(
        ("double_blocks", suffix, None, None) for suffix in _MXFP4_DOUBLE_SUFFIXES
    ),
}
_MXFP4_SELECTIVE_SCOPES = {
    "none": (),
    "single_linear2": (("single_blocks", "linear2", None, None),),
    "single_linear1_early": (
        ("single_blocks", "linear2", None, None),
        ("single_blocks", "linear1", 0, 18),
    ),
    "late_txt_mlp_up": (
        ("single_blocks", "linear2", None, None),
        ("double_blocks", "txt_mlp.0", 9, 18),
    ),
}
_MXFP4_RESIDUAL_SCOPES = {
    "none": (),
    "double_img_up_0_1": (("double_blocks", "img_mlp.0", 0, 1),),
    "double_img_up_2_3": (("double_blocks", "img_mlp.0", 2, 3),),
    "double_img_up_0_3": (("double_blocks", "img_mlp.0", 0, 3),),
    "double_img_up_4_8": (("double_blocks", "img_mlp.0", 4, 8),),
    "double_img_up": (("double_blocks", "img_mlp.0", None, None),),
    "double_txt_up": (("double_blocks", "txt_mlp.0", None, None),),
    "double_up": tuple(
        ("double_blocks", suffix, None, None)
        for suffix in ("img_mlp.0", "txt_mlp.0")
    ),
    "up": tuple(
        (stack, suffix, None, None)
        for stack, suffix in (
            ("double_blocks", "img_mlp.0"),
            ("double_blocks", "txt_mlp.0"),
            ("single_blocks", "linear1"),
        )
    ),
    "mlp": tuple(
        (stack, suffix, None, None)
        for stack, suffix in (
            ("double_blocks", "img_mlp.0"), ("double_blocks", "img_mlp.2"),
            ("double_blocks", "txt_mlp.0"), ("double_blocks", "txt_mlp.2"),
            ("single_blocks", "linear1"), ("single_blocks", "linear2"),
        )
    ),
    "all": tuple(
        [("double_blocks", suffix, None, None) for suffix in _MXFP4_DOUBLE_SUFFIXES]
        + [("single_blocks", suffix, None, None) for suffix in _MXFP4_SINGLE_SUFFIXES]
    ),
}
_FP8_SELECTIVE_GEMM_SHAPES = {
    (3072, 15360, 16384),
    (8192, 3072, 3072),
    (8192, 3072, 12288),
    (8192, 9216, 3072),
    (8192, 12288, 3072),
    (16384, 3072, 15360),
    (16384, 15360, 3072),
    (16384, 21504, 3072),
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


@torch.library.custom_op("primus::flux_flydsl_natural_wgrad", mutates_args=())
def _flux_flydsl_natural_wgrad(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    grad_scale: torch.Tensor,
    input_scale: torch.Tensor,
) -> torch.Tensor:
    from primus_turbo.flydsl.gemm.gemm_fp8_kernel import gemm_fp8_tensorwise_flydsl_kernel

    return gemm_fp8_tensorwise_flydsl_kernel(
        grad_output,
        grad_scale,
        input,
        input_scale,
        trans_a=True,
        trans_b=False,
        out_dtype=torch.bfloat16,
    )


@_flux_flydsl_natural_wgrad.register_fake
def _(grad_output, input, grad_scale, input_scale):
    return torch.empty(
        (grad_output.shape[1], input.shape[1]),
        device=input.device,
        dtype=torch.bfloat16,
    )


def _mxfp4_scope_matches(fqn: str, rules) -> bool:
    parts = fqn.split(".", 2)
    if len(parts) != 3:
        return False
    stack, index_text, suffix = parts
    if not index_text.isdigit():
        return False
    index = int(index_text)
    return any(
        stack == rule_stack
        and suffix == rule_suffix
        and (first is None or first <= index)
        and (last is None or index <= last)
        for rule_stack, rule_suffix, first, last in rules
    )


def _resolve_mxfp4_strategy(
    fqn: str,
    recipe: str,
    *,
    base_precision: str = "mxfp4",
    bf16_scope: str = "none",
    selective_mxfp4_scope: str = "none",
    forward_hadamard: str = "none",
    activation_residual: str = "none",
    activation_residual_dtype: str = "bf16",
    double_block_count: int = 19,
) -> tuple[str, bool, int] | None:
    """Resolve one selected FLUX Linear to precision, paired RHT, and residual mode."""
    if recipe not in _MXFP4_RECIPES:
        raise ValueError(f"Unsupported FLUX MXFP4 recipe: {recipe!r}")
    parts = fqn.split(".", 2)
    if len(parts) != 3:
        return None
    stack, index_text, suffix = parts
    if not index_text.isdigit() or not (
        (stack == "double_blocks" and suffix in _MXFP4_DOUBLE_SUFFIXES)
        or (stack == "single_blocks" and suffix in _MXFP4_SINGLE_SUFFIXES)
    ):
        return None

    presets = {
        "pareto_a": ("mxfp8", "double_img_all", "none"),
        "pareto_b": ("mxfp8", "double_img_up", "single_linear2"),
    }
    if recipe in presets:
        base_precision, bf16_scope, selective_mxfp4_scope = presets[recipe]
        forward_hadamard = activation_residual = "none"

    if _mxfp4_scope_matches(fqn, _MXFP4_BF16_SCOPES[bf16_scope]):
        precision = "bf16"
    elif _mxfp4_scope_matches(fqn, _MXFP4_SELECTIVE_SCOPES[selective_mxfp4_scope]):
        precision = "mxfp4"
    else:
        precision = base_precision

    index = int(index_text)
    residual_selected = _mxfp4_scope_matches(
        fqn, _MXFP4_RESIDUAL_SCOPES.get(activation_residual, ())
    )
    if activation_residual == "double_img_up_early":
        residual_selected = (
            stack == "double_blocks"
            and suffix == "img_mlp.0"
            and index < double_block_count // 2
        )
    elif activation_residual == "double_img_up_late":
        residual_selected = (
            stack == "double_blocks"
            and suffix == "img_mlp.0"
            and index >= double_block_count // 2
        )

    is_mlp = "mlp" in suffix or (
        stack == "single_blocks" and suffix in _MXFP4_SINGLE_SUFFIXES
    )
    hadamard_selected = forward_hadamard == "all" or (
        forward_hadamard == "mlp" and is_mlp
    )
    use_residual = precision == "mxfp4" and residual_selected
    residual_mode = (
        {"bf16": 1, "mxfp8": 2, "mxfp4": 3}[activation_residual_dtype]
        if use_residual
        else 0
    )
    return (
        precision,
        precision == "mxfp4" and hadamard_selected and not use_residual,
        residual_mode,
    )


def _mxfp4_forward_precision(fqn: str, recipe: str) -> str | None:
    strategy = _resolve_mxfp4_strategy(fqn, recipe)
    return strategy[0] if strategy else None


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

    `model_preset` is injected by the registry from `model.name`
    configs such as `flux.1-dev` and `flux.1-schnell`.
    """
    cfg_dict: dict[str, Any] = dict(model_config.get("config", {}) or {})
    float8_recipe = str(cfg_dict.get("float8_recipe") or "").strip().lower()
    if float8_recipe not in {"", "tensorwise"}:
        raise ValueError(f"Unsupported FLUX float8_recipe={float8_recipe!r}; expected null or 'tensorwise'")
    mxfp4_recipe = str(cfg_dict.get("mxfp4_recipe") or "").strip().lower()
    if mxfp4_recipe not in {"", *_MXFP4_RECIPES}:
        raise ValueError(
            f"Unsupported FLUX mxfp4_recipe={mxfp4_recipe!r}; "
            "expected null, 'pareto_a', 'pareto_b', or 'custom'"
        )
    if float8_recipe and mxfp4_recipe:
        raise ValueError("FLUX float8_recipe and mxfp4_recipe are mutually exclusive")

    fp8_qkv_high_precision_value = str(
        cfg_dict.get("float8_qkv_wgrad_high_precision", True)
    ).lower()
    if float8_recipe and fp8_qkv_high_precision_value not in {"true", "false"}:
        raise ValueError("FLUX float8_qkv_wgrad_high_precision must be boolean")
    fp8_qkv_high_precision = fp8_qkv_high_precision_value == "true"

    mxfp4_backward_precision = str(
        cfg_dict.get("mxfp4_backward_precision", "mxfp4")
    ).lower()
    if mxfp4_recipe and mxfp4_backward_precision not in {"mxfp4", "mxfp8"}:
        raise ValueError("FLUX mxfp4_backward_precision must be 'mxfp4' or 'mxfp8'")
    mxfp8_all_gather_value = str(cfg_dict.get("mxfp8_all_gather", False)).lower()
    if mxfp4_recipe and mxfp8_all_gather_value not in {"true", "false", "1", "0"}:
        raise ValueError("FLUX mxfp8_all_gather must be boolean")
    mxfp8_all_gather = mxfp8_all_gather_value in {"true", "1"}

    mxfp4_eval_precision = str(cfg_dict.get("mxfp4_eval_precision", "same")).lower()
    if mxfp4_recipe and mxfp4_eval_precision not in {"same", "bf16"}:
        raise ValueError("FLUX mxfp4_eval_precision must be 'same' or 'bf16'")
    gradient_sr_value = str(
        cfg_dict.get("mxfp4_gradient_stochastic_rounding", False)
    ).lower()
    if mxfp4_recipe and gradient_sr_value not in {"true", "false"}:
        raise ValueError("FLUX mxfp4_gradient_stochastic_rounding must be boolean")
    mxfp4_gradient_sr = gradient_sr_value == "true"

    mxfp4_options = {
        "base_precision": str(cfg_dict.get("mxfp4_forward_precision", "mxfp4")).lower(),
        "bf16_scope": str(cfg_dict.get("mxfp4_bf16_forward_scope", "none")).lower(),
        "selective_mxfp4_scope": str(
            cfg_dict.get("mxfp4_selective_forward_scope", "none")
        ).lower(),
        "forward_hadamard": str(cfg_dict.get("mxfp4_forward_hadamard", "none")).lower(),
        "activation_residual": str(
            cfg_dict.get("mxfp4_activation_residual", "none")
        ).lower(),
        "activation_residual_dtype": str(
            cfg_dict.get("mxfp4_activation_residual_dtype", "bf16")
        ).lower(),
    }
    if mxfp4_recipe == "custom":
        valid_options = {
            "base_precision": _MXFP4_PRECISIONS,
            "bf16_scope": set(_MXFP4_BF16_SCOPES),
            "selective_mxfp4_scope": set(_MXFP4_SELECTIVE_SCOPES),
            "forward_hadamard": {"none", "all", "mlp"},
            "activation_residual": {
                *set(_MXFP4_RESIDUAL_SCOPES),
                "double_img_up_early",
                "double_img_up_late",
            },
            "activation_residual_dtype": _MXFP4_PRECISIONS,
        }
        for option, valid in valid_options.items():
            if mxfp4_options[option] not in valid:
                raise ValueError(
                    f"Unsupported FLUX MXFP4 custom {option}={mxfp4_options[option]!r}; "
                    f"expected one of {sorted(valid)}"
                )
    pure_mxfp8 = (
        mxfp4_recipe == "custom"
        and mxfp4_options["base_precision"] == "mxfp8"
        and mxfp4_options["bf16_scope"] == "none"
        and mxfp4_options["selective_mxfp4_scope"] == "none"
    )
    if mxfp4_backward_precision == "mxfp8" and not pure_mxfp8:
        raise ValueError("MXFP8 backward requires an all-MXFP8 custom forward recipe")
    if mxfp8_all_gather and (not pure_mxfp8 or mxfp4_backward_precision != "mxfp8"):
        raise ValueError("MXFP8 AllGather requires pure MXFP8 forward and backward")
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

        fp8_all_gather = os.getenv("FLUX_FP8_ALL_GATHER", "0") == "1"
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
                enable_fsdp_float8_all_gather=fp8_all_gather,
            ),
        )
        qkv_config = Float8LinearConfig(
            pad_inner_dim=False,
            enable_fsdp_float8_all_gather=fp8_all_gather,
        )
        if fp8_qkv_high_precision:
            qkv_config = Float8LinearConfig(
                cast_config_input_for_grad_weight=CastConfig(scaling_type=ScalingType.DISABLED),
                cast_config_grad_output_for_grad_weight=CastConfig(scaling_type=ScalingType.DISABLED),
                pad_inner_dim=False,
                enable_fsdp_float8_all_gather=fp8_all_gather,
            )
        dit = convert_to_float8_training(
            dit,
            module_filter_fn=high_precision_wgrad_filter,
            config=qkv_config,
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
        fp8_wgrad_count = len(full_wgrad_fqns) + (
            0 if fp8_qkv_high_precision else len(high_precision_wgrad_fqns)
        )
        high_precision_count = len(high_precision_wgrad_fqns) if fp8_qkv_high_precision else 0
        logger.info(
            "Enabled TorchAO dynamic tensor-wise FP8 for "
            f"{len(full_wgrad_fqns) + len(high_precision_wgrad_fqns)} FLUX block Linear modules; "
            f"wgrad=FP8 for {fp8_wgrad_count} and high precision for "
            f"{high_precision_count} QKV modules"
        )

    if mxfp4_recipe:
        try:
            from primus_turbo.pytorch.core.backend import (
                BackendType,
                GlobalBackendManager,
                PrecisionType,
            )

            from primus.backends.diffusion.models.flux.mxfp4 import MXFP4Linear
        except ImportError as exc:
            raise ImportError("Primus-Turbo MXFP4 training support is required") from exc

        backend = GlobalBackendManager.get_gemm_backend(PrecisionType.FP4)
        backend = getattr(backend, "backend", backend)
        preshuffle = backend == BackendType.AITER and not GlobalBackendManager.auto_tune_enabled()
        if not preshuffle:
            raise RuntimeError(
                "FLUX MXFP4 recipes require PRIMUS_TURBO_GEMM_BACKEND=FP4:AITER "
                "and PRIMUS_TURBO_AUTO_TUNE=0"
            )

        precision_counts = {"bf16": 0, "mxfp4": 0, "mxfp8": 0}
        for fqn, module in list(dit.named_modules()):
            strategy = _resolve_mxfp4_strategy(
                fqn,
                mxfp4_recipe,
                **mxfp4_options,
                double_block_count=len(dit.double_blocks),
            )
            if strategy is None or type(module) is not torch.nn.Linear:
                continue
            forward_precision, forward_hadamard, residual_mode = strategy
            dit.set_submodule(
                fqn,
                MXFP4Linear(
                    module,
                    forward_precision,
                    backward_precision=mxfp4_backward_precision,
                    mxfp8_all_gather=mxfp8_all_gather,
                    forward_hadamard=forward_hadamard,
                    activation_residual_mode=residual_mode,
                    eval_bf16=mxfp4_eval_precision == "bf16",
                    gradient_stochastic_rounding=mxfp4_gradient_sr,
                    fqn=fqn,
                ),
            )
            precision_counts[forward_precision] += 1

        expected_total = len(dit.double_blocks) * 8 + len(dit.single_blocks) * 2
        if sum(precision_counts.values()) != expected_total:
            raise RuntimeError(
                f"FLUX MXFP4 recipe selected {sum(precision_counts.values())} block Linear modules; "
                f"expected {expected_total}"
            )
        if (
            mxfp4_recipe in {"pareto_a", "pareto_b"}
            and len(dit.double_blocks) == 19
            and len(dit.single_blocks) == 38
        ):
            expected_counts = {
                "pareto_a": {"bf16": 76, "mxfp4": 0, "mxfp8": 152},
                "pareto_b": {"bf16": 19, "mxfp4": 38, "mxfp8": 171},
            }[mxfp4_recipe]
            if precision_counts != expected_counts:
                raise RuntimeError(
                    f"FLUX MXFP4 {mxfp4_recipe} routed {precision_counts}; expected {expected_counts}"
                )
        logger.info(
            f"Enabled FLUX MXFP4 {mxfp4_recipe}: forward={precision_counts}; "
            f"backward={mxfp4_backward_precision.upper()} for all selected modules; "
            f"MXFP8 AllGather={mxfp8_all_gather}"
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
