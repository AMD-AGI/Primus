###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
WAN (2.1 / 2.2) Pretrain Trainer for Primus-Megatron.

Mirrors :class:`FluxPretrainTrainer` but specialized for the WAN family:

    - Model: :class:`Wan` (single transformer: WAN 2.1 all sizes, WAN 2.2
      TI2V-5B) or :class:`Wan2_2` (dual-expert A14B).
    - Scheduler: :class:`WanFlowMatchScheduler`, which adds ``training_target``
      and ``training_weight`` to the Flux flow-matching scheduler.
    - Loss: :func:`compute_weighted_flow_matching_loss`.
    - Task encoder: :class:`EncodedWanTaskEncoder` for the pre-encoded path.

WAN runs at ``tensor_model_parallel_size=1`` and
``pipeline_model_parallel_size=1``; ``WanConfig.validate()`` enforces both.
"""

import os

import torch
import torch.nn as nn

from primus.backends.megatron.diffusion_trainer import DiffusionPretrainTrainer
from primus.backends.megatron.training.diffusion.schedulers import WanFlowMatchScheduler
from primus.core.utils.module_utils import log_rank_0

# Both attention backends re-read the flash-attention backward variant on every
# call, and each keeps its own copy of the choice: Primus-Turbo (local spec)
# reads the first variable, TransformerEngine's ROCm CK backend (te_spec) the
# second. A run uses one spec or the other, so set both rather than have the
# trainer work out which one is live.
#
# Neither default is right here. Primus' launchers override Primus-Turbo's own
# default of "1" to "0" (``examples/run_pretrain.sh`` and
# ``runner/helpers/envs/base_env.sh``), and the release Dockerfile pins
# ``NVTE_CK_IS_V3_ATOMIC_FP32=0``, which is the gfx950 value; on gfx942 the CK
# v3 backward needs fp32 atomics, as ``tools/installation/env.sh`` already spells
# out for the bare-metal path.
ATTN_ATOMIC_FP32_ENVS = (
    "PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32",
    "NVTE_CK_IS_V3_ATOMIC_FP32",
)

# TE only reaches a v3 backward with this on. It is already "1" everywhere in
# Primus; setting it stops a launcher that disabled v3 from quietly undoing the
# atomics above.
TE_CK_BWD_V3_ENV = "NVTE_CK_USES_BWD_V3"


def _apply_atomic_fp32_backward(enabled: bool) -> None:
    """Let WAN's attention backward reach the fp32-atomic ASM kernels.

    WAN's sequence length is not divisible by 64, which is what puts both
    backends in the same corner: the kernel families that serve non-padded
    shapes accumulate dQ through atomics, so with the atomic accumulator off
    there is no ASM backward to dispatch to and both fall through to ck_tile
    without saying so.

    On the local spec that leaves Primus-Turbo on ck_tile's backward instead of
    ``aiter::fmha_bwd_hd128_bf16_a32_rtna_psskddv`` over the same call count.

    On te_spec the same thing happens one layer down. TE asks for its v3
    backward (``NVTE_CK_USES_BWD_V3`` is "1" everywhere in Primus) but the
    release Dockerfile pins ``NVTE_CK_IS_V3_ATOMIC_FP32=0``, the value that suits
    gfx950, so on gfx942 v3 is unreachable and CK serves the backward from
    ck_tile. Setting the atomics gets
    ``aiter::fmha_bwd_hd128_bf16_a32_rtz_pssk_group`` instead over the same call
    count, which is the pairing ``tools/installation/env.sh`` already prescribes
    for gfx942.

    So the atomic path is what reaches the ASM kernels on both specs. The
    accumulation order is non-deterministic, so this is not free in principle;
    over a 20-step run the loss stayed within bf16 reduction noise rather than
    showing a behaviour change.

    The opt-out is the YAML key rather than the environment variables, because
    the launchers and the image export those unconditionally and so every WAN run
    arrives here with them already set to "0". Reading them back could not tell a
    deliberate choice apart from that blanket default, which is the bug this
    works around. Set ``attn_atomic_fp32: false`` to get the deterministic
    split-dQ path back.
    """
    value = "1" if enabled else "0"
    previous = {name: os.environ.get(name) for name in ATTN_ATOMIC_FP32_ENVS}
    for name in ATTN_ATOMIC_FP32_ENVS:
        os.environ[name] = value
    if enabled:
        os.environ[TE_CK_BWD_V3_ENV] = "1"

    settings = ", ".join(f"{name}={value} (was {previous[name]!r})" for name in ATTN_ATOMIC_FP32_ENVS)
    if enabled:
        log_rank_0(
            f"WAN trainer: {settings} so the attention backward reaches the "
            f"fp32-atomic ASM kernels instead of ck_tile"
        )
    else:
        log_rank_0(
            f"WAN trainer: {settings} by request (attn_atomic_fp32: false); "
            f"the attention backward will use ck_tile"
        )


class WanPretrainTrainer(DiffusionPretrainTrainer):
    """Trainer for WAN 2.1 / 2.2 video diffusion pre-training.

    YAML keys consumed (under ``overrides``):
        - ``num_train_timesteps`` (default 1000), ``scheduler_shift``
          (default 5.0), ``scheduler_sigma_min`` / ``scheduler_sigma_max``.
        - ``loss_weighting``: ``diffsynth`` (default) or ``uniform``.
        - ``num_transformers`` (1 or 2) and ``boundary_ratio`` for WAN 2.2.
        - ``stage``: ``full`` / ``high_noise`` / ``low_noise``. Narrowing the
          stage derives the timestep window and the weight subfolder from
          ``boundary_ratio`` so the two cannot disagree.
        - Architecture fields (``hidden_size``, ``num_attention_heads``,
          ``num_dit_layers``, ``ffn_hidden_size``, ``text_embed_dim``,
          ``text_seq_len``, ``patch_size_3d``, ``in_channels``,
          ``out_channels``, ``freq_dim``).
        - ``backbone_pretrained``, ``backbone_subfolder``,
          ``backbone_subfolder_2``.
        - ``attn_atomic_fp32`` (default true): let the attention backward use
          the fp32-atomic ASM kernels, on both the local spec (Primus-Turbo) and
          te_spec (TE / CK v3). Set false for a deterministic backward.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        params = self.backend_args

        self.num_train_timesteps = getattr(params, "num_train_timesteps", 1000)
        self.scheduler_shift = getattr(params, "scheduler_shift", 5.0)
        self.scheduler_sigma_min = getattr(params, "scheduler_sigma_min", 0.0)
        self.scheduler_sigma_max = getattr(params, "scheduler_sigma_max", 1.0)

        self.loss_weighting = getattr(params, "loss_weighting", "diffsynth")

        # Before the first forward, and re-read per call by both backends, so
        # the trainer constructor is early enough.
        self.attn_atomic_fp32 = bool(getattr(params, "attn_atomic_fp32", True))
        _apply_atomic_fp32_backward(self.attn_atomic_fp32)

        # The model config resolves stage -> window, so read the window back off
        # it rather than recomputing here and risking a disagreement.
        self.wan_config = self._build_wan_config_from_yaml()
        self.wan_config.validate()

        window = (
            self.wan_config.timestep_window_min,
            self.wan_config.timestep_window_max,
        )
        self.timestep_window = None if window == (0.0, 1.0) else window

        if self.wan_config.num_transformers == 2 and self.wan_config.boundary_ratio is not None:
            self.boundary_timestep = float(self.wan_config.boundary_ratio * self.num_train_timesteps)
        else:
            self.boundary_timestep = None

        log_rank_0(
            f"WAN trainer: shift={self.scheduler_shift}, "
            f"num_train_timesteps={self.num_train_timesteps}, "
            f"stage={self.wan_config.stage}, window={self.timestep_window}, "
            f"num_transformers={self.wan_config.num_transformers}, "
            f"boundary_ratio={self.wan_config.boundary_ratio}, "
            f"loss_weighting={self.loss_weighting}"
        )

    # ------------------------------------------------------------------
    # Forward step
    # ------------------------------------------------------------------

    def forward_step(self, data_iterator, model, return_schedule_plan=False):
        """Run one WAN forward step and return ``(noise_pred, loss_func)``.

        Overrides the Flux-flavored base step: WAN's loss needs the scheduler's
        target and per-timestep weight, which the base step does not produce.
        """
        from primus.backends.megatron.training.diffusion.loss_computation import (
            compute_weighted_flow_matching_loss,
        )
        from primus.backends.megatron.training.diffusion.wan_forward_step import (
            wan_forward_step_func,
        )

        if model.training:
            self._forward_step_count += 1

        (
            noise_pred,
            clean_latents,
            noise,
            target,
            weight,
            loss_mask,
            metrics,
            is_validation,
        ) = wan_forward_step_func(
            data_iterator,
            model,
            scheduler=self.scheduler,
            timestep_window=self.timestep_window,
            boundary_timestep=self.boundary_timestep,
            loss_weighting=self.loss_weighting,
        )

        self._last_clean_latents = clean_latents
        self._last_noise = noise
        self._last_target = target
        self._last_weight = weight
        self._last_loss_mask = loss_mask

        if hasattr(self, "runtime_state") and self.runtime_state:
            self.runtime_state.update_metrics(metrics)

        if is_validation:

            def val_loss_func(output_tensor, non_loss_data=False):
                if non_loss_data:
                    return output_tensor
                loss = compute_weighted_flow_matching_loss(
                    output_tensor,
                    self._last_target,
                    self._last_weight,
                    self._last_loss_mask,
                )
                sample_count = torch.tensor(output_tensor.shape[0], dtype=loss.dtype, device=loss.device)
                return loss, {"loss": (loss.detach(), sample_count.detach())}

            return noise_pred, val_loss_func

        def wan_loss_func(output_tensor, non_loss_data=False):
            if non_loss_data:
                return output_tensor

            loss = compute_weighted_flow_matching_loss(
                output_tensor,
                self._last_target,
                self._last_weight,
                self._last_loss_mask,
            )
            return loss, {"reduced_train_loss": loss.detach().clone()}

        return noise_pred, wan_loss_func

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def create_model(self, pre_process=True, post_process=True):
        """Build a Wan / Wan2_2 model from the YAML configuration."""
        from primus.backends.megatron.core.models.diffusion.wan.model import Wan, Wan2_2

        log_rank_0("=" * 80)
        log_rank_0("Creating WAN model from YAML config")

        config = self.wan_config

        if config.num_transformers == 2:
            model = Wan2_2(config=config)
            log_rank_0(
                "WAN 2.2 dual-expert model created "
                f"(boundary_ratio={config.boundary_ratio}, "
                f"boundary_timestep={self.boundary_timestep})"
            )
        else:
            model = Wan(config=config)
            log_rank_0("WAN single-transformer model created")

        total_params = sum(p.numel() for p in model.parameters())
        log_rank_0(f"Total parameters: {total_params / 1e9:.2f}B")
        log_rank_0("=" * 80)
        return model

    def _build_wan_config_from_yaml(self):
        """Build :class:`WanConfig` from ``backend_args``.

        Mirrors ``FluxPretrainTrainer._build_flux_config_from_yaml``.
        """
        from primus.backends.megatron.core.models.diffusion.wan.config import WanConfig

        params = self.backend_args

        cfg: dict = {
            "hidden_size": getattr(params, "hidden_size", 1536),
            "num_attention_heads": getattr(params, "num_attention_heads", 12),
            "num_dit_layers": getattr(params, "num_dit_layers", 30),
            "ffn_hidden_size": getattr(params, "ffn_hidden_size", 8960),
            "in_channels": getattr(params, "in_channels", 16),
            "out_channels": getattr(params, "out_channels", 16),
            "patch_size_3d": tuple(getattr(params, "patch_size_3d", (1, 2, 2))),
            "text_embed_dim": getattr(params, "text_embed_dim", 4096),
            "text_seq_len": getattr(params, "text_seq_len", 512),
            "freq_dim": getattr(params, "freq_dim", 256),
            "num_train_timesteps": getattr(params, "num_train_timesteps", 1000),
            "loss_weighting": getattr(params, "loss_weighting", "diffsynth"),
            "num_transformers": getattr(params, "num_transformers", 1),
            "boundary_ratio": getattr(params, "boundary_ratio", None),
            "stage": getattr(params, "stage", "full"),
            "transformer_impl": getattr(params, "transformer_impl", "transformer_engine"),
            # TransformerConfig defaults this to AttnBackend.auto, which TE
            # refuses to run under an image that pins NVTE_FLASH_ATTN /
            # NVTE_FUSED_ATTN, so the resolved value has to be threaded in.
            "attention_backend": getattr(params, "attention_backend", None),
            "layernorm_across_heads": getattr(params, "layernorm_across_heads", True),
            "adaln_zero_init": getattr(params, "adaln_zero_init", True),
            "backbone_pretrained": getattr(params, "backbone_pretrained", None),
            "init_method": nn.init.xavier_uniform_,
            "output_layer_init_method": nn.init.xavier_uniform_,
        }

        # Only forward an explicit window when the stage did not derive one;
        # WanConfig rejects setting both.
        if getattr(params, "stage", "full") == "full":
            cfg["timestep_window_min"] = getattr(params, "timestep_window_min", 0.0)
            cfg["timestep_window_max"] = getattr(params, "timestep_window_max", 1.0)
            cfg["backbone_subfolder"] = getattr(params, "backbone_subfolder", "transformer")
        cfg["backbone_subfolder_2"] = getattr(params, "backbone_subfolder_2", "transformer_2")

        if cfg["attention_backend"] is None:
            del cfg["attention_backend"]

        cfg.update(
            {
                "bf16": getattr(params, "bf16", True),
                "fp16": getattr(params, "fp16", False),
                "params_dtype": getattr(params, "params_dtype", torch.float32),
            }
        )

        cfg.update(
            {
                "recompute_granularity": getattr(params, "recompute_granularity", None),
                "recompute_method": getattr(params, "recompute_method", None),
                "recompute_num_layers": getattr(params, "recompute_num_layers", None),
                "recompute_modules": getattr(params, "recompute_modules", None),
            }
        )

        # FP4 / MXFP4, local transformer_impl only. ``fp4`` / ``fp4_recipe`` are
        # Megatron TransformerConfig fields; the mxfp4_* knobs are read by the
        # Primus-Turbo MXFP4 local linears.
        if getattr(params, "fp4", None):
            fp4_recipe = getattr(params, "fp4_recipe", None)
            if not fp4_recipe:
                raise ValueError(
                    "fp4_recipe must be set explicitly in YAML when fp4 is enabled "
                    "(use fp4_recipe: 'mxfp4' for AMD FP4 GEMM)."
                )
            cfg.update(
                {
                    "fp4": True,
                    "fp4_recipe": fp4_recipe,
                    "mxfp4_backward_precision": getattr(params, "mxfp4_backward_precision", "mxfp4"),
                    "mxfp4_gradient_stochastic_rounding": getattr(
                        params, "mxfp4_gradient_stochastic_rounding", False
                    ),
                    # The MXFP4 local linears subclass the native Megatron
                    # linears, which reject grad-accum fusion.
                    "gradient_accumulation_fusion": False,
                }
            )

        if getattr(params, "use_fsdp2_fp32_param_optimizer", False):
            cfg["params_dtype"] = torch.float32
            cfg["pipeline_dtype"] = torch.bfloat16

        cfg["tensor_model_parallel_size"] = getattr(params, "tensor_model_parallel_size", 1)
        cfg["pipeline_model_parallel_size"] = getattr(params, "pipeline_model_parallel_size", 1)
        cfg["context_parallel_size"] = getattr(params, "context_parallel_size", 1)

        return WanConfig(**cfg)

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def create_scheduler(self):
        """Create the WAN flow matching scheduler."""
        log_rank_0(
            f"Creating WanFlowMatchScheduler: num_train_timesteps={self.num_train_timesteps}, "
            f"shift={self.scheduler_shift}, "
            f"sigma_range=[{self.scheduler_sigma_min}, {self.scheduler_sigma_max}]"
        )

        return WanFlowMatchScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=self.scheduler_shift,
            sigma_min=self.scheduler_sigma_min,
            sigma_max=self.scheduler_sigma_max,
        )

    # ------------------------------------------------------------------
    # TaskEncoder
    # ------------------------------------------------------------------

    def get_task_encoder(self):
        """Return :class:`EncodedWanTaskEncoder` for the pre-encoded path."""
        from primus.backends.megatron.data.diffusion.task_encoders import (
            EncodedWanTaskEncoder,
        )

        encoder = EncodedWanTaskEncoder(worker_config=None)
        log_rank_0("Created EncodedWanTaskEncoder for pre-encoded WAN data")
        return encoder
