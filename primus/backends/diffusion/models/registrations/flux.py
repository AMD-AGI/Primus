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
_FLUX_QKV_SUFFIXES = {
    "img_attn.qkv",
    "txt_attn.qkv",
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
    from primus_turbo.flydsl.gemm.gemm_fp8_kernel import (
        gemm_fp8_tensorwise_flydsl_kernel,
    )

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
    from primus_turbo.flydsl.gemm.gemm_fp8_kernel import (
        gemm_fp8_tensorwise_flydsl_kernel,
    )

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


def _flux_block_kind(module: torch.nn.Module, fqn: str) -> str | None:
    """Classify a FLUX block Linear as ``"qkv"`` or ``"full"``, else None.

    The selection is deliberately identical to the FP8 recipe's, so an MXFP4 run
    quantizes exactly the same 228 modules and leaves the embeddings,
    modulation/adaLN projections and final layer in BF16.
    """
    if type(module) is not torch.nn.Linear:
        return None
    parts = fqn.split(".", 2)
    if len(parts) != 3:
        return None
    if parts[0] == "double_blocks":
        if parts[2] in _FLUX_QKV_SUFFIXES:
            return "qkv"
        if parts[2] in _FP8_DOUBLE_ATTN_PROJ_SUFFIXES:
            return "full"
        if parts[2] in _FP8_DOUBLE_MLP_SUFFIXES:
            return "full"
    if parts[0] == "single_blocks" and parts[2] in {"linear1", "linear2"}:
        return "full"
    return None


def _apply_flux_mxfp4(dit, config):
    """Swap the FLUX block Linears for MXFP4, including QKV Wgrad.

    The QKV opt-out mirrors the FP8 recipe below, which already disables the
    Wgrad casts on exactly those modules. ``FLUX_FP4_QKV_WGRAD_BF16=1`` keeps
    them in BF16; default 0 is full MXFP4.

    ``FLUX_FP4_BF16_LAST_BLOCKS`` and ``FLUX_FP4_BF16_FIRST_BLOCKS`` hold whole
    transformer blocks out of MXFP4, counting inwards from each end of the network;
    ``FLUX_FP4_BF16_MLP_ONLY=1`` narrows that to the MLP Linears of the selected
    blocks. All three default to leaving every block quantized.

    Both ends are offered because the two literatures disagree about where to spend
    the budget, and the disagreement is specifically about our format. NVFP4's first
    listed ingredient is keeping ~15% of the network higher precision "with the
    majority of high precision layers at the end of the network", and LongLive-2.0's
    video-diffusion FP4 recipe likewise holds the sensitive operations back. But the
    one study that measures FP4 sensitivity per layer and per depth finds the
    opposite of the end-of-network heuristic: sensitivity "does not universally
    localize to the final blocks", is often bimodal, and "early blocks can be highly
    sensitive, particularly under MXFP4" (arXiv 2603.08747). So the first-blocks arm
    is the one that study argues for and the last-blocks arm is the one NVIDIA does.

    The same study, and FP4DiT independently on diffusion transformers, agree on
    WHICH modules matter: MLP up/down projections dominate, attention projections are
    "substantially less sensitive". Hence MLP_ONLY, which buys roughly two thirds of
    a block's sensitivity for roughly two thirds of its cost, but concentrated on the
    part that matters. Note the single blocks fuse attention and MLP into linear1 and
    linear2, so MLP_ONLY cannot separate them there; a selected single block is held
    back whole either way.

    Budget, which is what actually constrains this: the RCP floor of 8.316M samples
    needs >=76.0 img/s/GPU for a 57-minute run, and a BF16 forward across all 228
    quantized Linears measured -35.5% on one node. A double block carries twice a
    single block's GEMM work (2x3072x36864 against 3072x36864), which splits the
    block FLOPs almost exactly in half between the 19 double and 38 single blocks, so
    one double block is ~1.2% of throughput and one single block ~0.6%. That affords
    about six blocks whole, or about ten with MLP_ONLY -- and the 15% NVFP4 asks for
    is already past it.
    """
    from dataclasses import replace

    from primus.backends.diffusion.models.quantization.mxfp4_linear import (
        convert_to_mxfp4_training,
    )

    qkv_wgrad_bf16 = os.getenv("FLUX_FP4_QKV_WGRAD_BF16", "0") == "1"
    bf16_last = int(os.getenv("FLUX_FP4_BF16_LAST_BLOCKS", "0"))
    bf16_first = int(os.getenv("FLUX_FP4_BF16_FIRST_BLOCKS", "0"))
    mlp_only = os.getenv("FLUX_FP4_BF16_MLP_ONLY", "0") == "1"
    n_double, n_single = len(dit.double_blocks), len(dit.single_blocks)
    if min(bf16_last, bf16_first) < 0 or bf16_last + bf16_first > n_double + n_single:
        raise ValueError(f"FLUX_FP4_BF16_FIRST_BLOCKS={bf16_first} + LAST_BLOCKS={bf16_last} is out of range")

    # Held-back blocks as (kind, index), walking inwards from each end of the
    # network. Network order is the double blocks then the single blocks, so
    # "first" starts at double_blocks.0 and "last" ends at the final single block.
    order = [("double_blocks", i) for i in range(n_double)]
    order += [("single_blocks", i) for i in range(n_single)]
    held_back = set(order[:bf16_first]) | set(order[len(order) - bf16_last :])

    def held(fqn: str) -> bool:
        kind, index, suffix = fqn.split(".", 2)
        if (kind, int(index)) not in held_back:
            return False
        # Double blocks name their MLP separately, so MLP_ONLY can act there.
        return not (mlp_only and kind == "double_blocks" and suffix not in _FP8_DOUBLE_MLP_SUFFIXES)

    # Enumerated before conversion, since _flux_block_kind keys off nn.Linear and the
    # swapped modules are no longer one. This replaces the arithmetic the count check
    # used to do, which does not survive a per-module selection.
    block_linears = [fqn for fqn, m in dit.named_modules() if _flux_block_kind(m, fqn) is not None]
    expected_total = n_double * 8 + n_single * 2
    if len(block_linears) != expected_total:
        raise RuntimeError(f"FLUX has {len(block_linears)} block Linear modules; expected {expected_total}")
    expected = sum(1 for fqn in block_linears if not held(fqn))

    def filter_fn(module: torch.nn.Module, fqn: str) -> bool:
        return _flux_block_kind(module, fqn) is not None and not held(fqn)

    def config_for(fqn: str):
        parts = fqn.split(".", 2)
        if qkv_wgrad_bf16 and parts[0] == "double_blocks" and parts[2] in _FLUX_QKV_SUFFIXES:
            return replace(config, wgrad="bf16")
        return None

    converted = convert_to_mxfp4_training(dit, filter_fn, config, config_for)

    if len(converted) != expected:
        raise RuntimeError(f"FLUX MXFP4 converted {len(converted)} Linear modules; expected {expected}")
    logger.info(
        f"Enabled MXFP4 for {len(converted)} of {expected_total} FLUX block Linear modules "
        f"({config.describe()}); wgrad={'BF16' if qkv_wgrad_bf16 else 'MXFP4'} for QKV modules; "
        f"held back {expected_total - len(converted)} modules across {len(held_back)} blocks "
        f"(first={bf16_first} last={bf16_last} mlp_only={int(mlp_only)})"
    )
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
        raise ValueError(f"Unsupported FLUX float8_recipe={float8_recipe!r}; expected null or 'tensorwise'")
    fp8_gemm_backend = str(cfg_dict.get("float8_gemm_backend") or "").strip().lower()
    if fp8_gemm_backend not in {"", "selective_triton", "selective_flydsl", "full_flydsl"}:
        raise ValueError(
            "Unsupported FLUX float8_gemm_backend="
            f"{fp8_gemm_backend!r}; expected null, 'selective_triton', "
            "'selective_flydsl', or 'full_flydsl'"
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

    # MXFP4 replaces the FP8 GEMMs on the same modules, so the two recipes are
    # mutually exclusive and FLUX_FP4_PASSES wins when both are configured.
    from primus.backends.diffusion.models.quantization.mxfp4_linear import (
        config_from_env as _mxfp4_config_from_env,
    )

    mxfp4_config = _mxfp4_config_from_env()
    if mxfp4_config is not None:
        dit = _apply_flux_mxfp4(dit, mxfp4_config)
        float8_recipe = ""

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

        if fp8_gemm_backend in {"selective_flydsl", "full_flydsl"}:
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
                    (fp8_gemm_backend == "full_flydsl" or shape in _FP8_SELECTIVE_GEMM_SHAPES)
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
            if fp8_gemm_backend == "full_flydsl":
                logger.info("Using FlyDSL for all FLUX FP8 scaled GEMMs")
            else:
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
        dit = convert_to_float8_training(
            dit,
            module_filter_fn=high_precision_wgrad_filter,
            config=Float8LinearConfig(
                cast_config_input_for_grad_weight=CastConfig(scaling_type=ScalingType.DISABLED),
                cast_config_grad_output_for_grad_weight=CastConfig(scaling_type=ScalingType.DISABLED),
                pad_inner_dim=False,
                enable_fsdp_float8_all_gather=fp8_all_gather,
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
