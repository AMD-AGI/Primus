# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Configuration for WAN (2.1 / 2.2) video diffusion models.

WAN is a 3D-video DiT family. ``WanConfig`` extends :class:`BaseDiffusionConfig`
(which itself extends Megatron-Core's ``TransformerConfig``) and adds the
video-specific and expert-routing fields consumed by ``Wan`` / ``Wan2_2``.

The precision fields (``fp8_*``, ``mxfp4_*``, ``sensitive_layers_*``) are
inherited from :class:`BaseDiffusionConfig` and shared with Flux; this module
only declares what is genuinely WAN-specific.

Strategy A constraints, enforced in :meth:`WanConfig.validate`:
    - ``tensor_model_parallel_size == 1``
    - ``pipeline_model_parallel_size == 1`` (also rejected by the base class)
    - DP / FSDP, and optionally CP, only.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch.nn as nn

from ..common.config import BaseDiffusionConfig

# Expert-training stages for WAN 2.2 A14B. ``full`` trains the whole schedule
# and is what every WAN 2.1 and TI2V-5B config uses.
WAN_STAGES = ("full", "high_noise", "low_noise")

# Timestep-weighting schemes for the flow-matching loss.
#   "diffsynth" -- the DiffSynth WAN template weight (bell curve over the
#                  shifted schedule, normalized to mean 1). This is what the
#                  reference WAN training recipes use.
#   "uniform"   -- unweighted MSE; every sample weighted 1.0.
WAN_LOSS_WEIGHTINGS = ("diffsynth", "uniform")


@dataclass
class WanConfig(BaseDiffusionConfig):
    """WAN-specific configuration.

    Covers WAN 2.1 (single transformer, all sizes) and WAN 2.2 (single-DiT
    TI2V-5B with ``num_transformers=1``, dual-expert A14B with
    ``num_transformers=2`` and a ``boundary_ratio``).
    """

    # Model identification.
    model_type: str = "wan"

    # Placeholder depth for TransformerConfig; the real depth is
    # ``num_dit_layers`` and is mirrored into ``num_layers`` in __post_init__.
    num_layers: int = 1

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------
    hidden_size: int = 1536
    num_attention_heads: int = 12
    num_dit_layers: int = 30
    ffn_hidden_size: int = 8960

    # Latent video channels.
    in_channels: int = 16
    out_channels: Optional[int] = 16

    # 3D patchification (temporal, height, width).
    patch_size_3d: Tuple[int, int, int] = (1, 2, 2)

    # Text conditioning (UMT5).
    text_embed_dim: int = 4096
    text_seq_len: int = 512

    # Timestep embedding sinusoidal dimension.
    freq_dim: int = 256

    # Dropout (WAN backbone attention).
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Normalization.
    layernorm_epsilon: float = 1e-6

    # ------------------------------------------------------------------
    # Backbone architecture (diffusers WanTransformer3DModel parity)
    # ------------------------------------------------------------------
    # Rotary position embedding table length (diffusers ``rope_max_seq_len``).
    rope_max_seq_len: int = 1024
    # Query/key normalization scheme; diffusers uses RMSNorm over the inner dim.
    qk_norm: str = "rms_norm_across_heads"
    # Affine LayerNorm before cross-attention (diffusers ``cross_attn_norm``).
    cross_attn_norm: bool = True
    # Apply q/k RMSNorm across the full inner dim (heads * head_dim) instead of
    # per head. WAN diffusers and Megatron-Bridge both normalize across heads,
    # so this must stay True for checkpoint parity.
    layernorm_across_heads: bool = True
    # AdaLN-zero init: identity blocks plus a zeroed output head.
    adaln_zero_init: bool = True
    # I2V-only extras; None for T2V.
    image_dim: Optional[int] = None
    added_kv_proj_dim: Optional[int] = None
    pos_embed_seq_len: Optional[int] = None

    # ------------------------------------------------------------------
    # Diffusion / scheduler
    # ------------------------------------------------------------------
    num_train_timesteps: int = 1000

    # Per-timestep weighting applied to the flow-matching loss.
    loss_weighting: str = "diffsynth"

    # ------------------------------------------------------------------
    # WAN 2.2 expert routing
    # ------------------------------------------------------------------
    num_transformers: int = 1
    boundary_ratio: Optional[float] = None

    # Which expert this job trains. ``full`` covers the whole schedule;
    # ``high_noise`` / ``low_noise`` derive the timestep window and the weight
    # subfolder from ``boundary_ratio`` so the two cannot disagree. This mirrors
    # the ``stage`` field of the NeMo AutoModel WAN 2.2 preset.
    stage: str = "full"

    # Explicit per-expert training window, as a fraction of the schedule.
    # Left at the full range unless ``stage`` narrows it; setting both is an
    # error, since the derived and explicit windows would silently disagree.
    timestep_window_min: float = 0.0
    timestep_window_max: float = 1.0

    # ------------------------------------------------------------------
    # Pretrained weights
    # ------------------------------------------------------------------
    backbone_pretrained: Optional[str] = None
    backbone_subfolder: str = "transformer"
    backbone_subfolder_2: str = "transformer_2"

    # ------------------------------------------------------------------
    # Attention / linear implementation
    # ------------------------------------------------------------------
    # Mirrors Flux's ``transformer_impl``:
    #   - "transformer_engine": TE linears + TENorm + TEDotProductAttention
    #     (fused THD kernel, mcore fused RoPE). Requires TransformerEngine.
    #   - "local": TE-free. Native Megatron parallel linears (or the
    #     Primus-Turbo MXFP4/FP8 subclasses when ``fp4``/``fp8`` is set),
    #     ``nn.RMSNorm`` q/k norm, and ``PrimusTurboLocalAttention`` with
    #     unfused interleaved RoPE. torch.compile-friendly; needs Primus-Turbo.
    transformer_impl: str = "transformer_engine"

    # Compute the attention core in FP32 with unfused torch ops. Only for
    # bit-exact parity investigations against a reference stack; there is no
    # FP32 fused-attention backend, so this is far slower.
    use_fp32_attention: bool = False

    # CPU init keeps parity with the Flux path.
    use_cpu_initialization: bool = True

    def __post_init__(self):
        """Post-initialization processing."""
        # Mirror the real depth into num_layers BEFORE the base class runs, so
        # BaseDiffusionConfig can validate the sensitive-layer counts against
        # the true depth rather than the placeholder.
        self.num_layers = self.num_dit_layers

        # Normalize patch size to a tuple (YAML gives a list).
        if isinstance(self.patch_size_3d, list):
            self.patch_size_3d = tuple(self.patch_size_3d)

        self._resolve_stage()

        super().__post_init__()

        # Xavier uniform initializers (parity with Flux defaults).
        self.init_method = nn.init.xavier_uniform_
        self.output_layer_init_method = nn.init.xavier_uniform_

    def _resolve_stage(self) -> None:
        """Derive the timestep window and weight subfolder from ``stage``."""
        if self.stage not in WAN_STAGES:
            raise ValueError(f"stage must be one of {WAN_STAGES}, got {self.stage!r}")

        if self.stage == "full":
            return

        if self.boundary_ratio is None:
            raise ValueError(
                f"stage={self.stage!r} needs boundary_ratio to derive the timestep "
                "window (WAN 2.2 A14B ships boundary_ratio=0.875)."
            )

        explicit_window = self.timestep_window_min != 0.0 or self.timestep_window_max != 1.0
        if explicit_window:
            raise ValueError(
                f"stage={self.stage!r} derives timestep_window_min/max from "
                "boundary_ratio; do not set them explicitly as well. Use "
                "stage='full' if you want a hand-picked window."
            )

        if self.stage == "high_noise":
            self.timestep_window_min = float(self.boundary_ratio)
            self.timestep_window_max = 1.0
            self.backbone_subfolder = "transformer"
        else:
            self.timestep_window_min = 0.0
            self.timestep_window_max = float(self.boundary_ratio)
            self.backbone_subfolder = "transformer_2"

    def validate(self):
        """Validate WAN configuration, including the Strategy A guards."""
        super().validate()

        if self.num_dit_layers <= 0:
            raise ValueError(f"num_dit_layers must be positive, got {self.num_dit_layers}")

        if len(self.patch_size_3d) != 3:
            raise ValueError(f"patch_size_3d must have 3 elements, got {self.patch_size_3d}")

        if self.num_transformers not in (1, 2):
            raise ValueError(f"num_transformers must be 1 or 2, got {self.num_transformers}")

        if self.num_transformers == 2 and self.boundary_ratio is None:
            raise ValueError("WAN 2.2 dual-expert (num_transformers=2) requires boundary_ratio")

        if self.boundary_ratio is not None and not (0.0 < self.boundary_ratio < 1.0):
            raise ValueError(f"boundary_ratio must be in (0, 1), got {self.boundary_ratio}")

        if not 0.0 <= self.timestep_window_min < self.timestep_window_max <= 1.0:
            raise ValueError(
                "timestep_window must satisfy 0 <= min < max <= 1, got "
                f"[{self.timestep_window_min}, {self.timestep_window_max}]"
            )

        if self.loss_weighting not in WAN_LOSS_WEIGHTINGS:
            raise ValueError(
                f"loss_weighting must be one of {WAN_LOSS_WEIGHTINGS}, got {self.loss_weighting!r}"
            )

        if self.transformer_impl not in ("transformer_engine", "local"):
            raise ValueError(
                "transformer_impl must be 'transformer_engine' or 'local', " f"got {self.transformer_impl!r}"
            )

        # MXFP4 and FP8 are only wired into the TE-free local linears.
        if getattr(self, "fp4", None) and self.transformer_impl != "local":
            raise ValueError("fp4 (MXFP4) on WAN requires transformer_impl='local'")

        if self.use_fp32_attention and self.transformer_impl != "transformer_engine":
            raise ValueError(
                "use_fp32_attention replaces the TE fused attention core and is "
                "only implemented for transformer_impl='transformer_engine'"
            )

        # Strategy A: no tensor parallelism for the WAN backbone. (PP is
        # rejected by BaseDiffusionConfig for every diffusion model.)
        if self.tensor_model_parallel_size != 1:
            raise ValueError(
                "WAN Strategy A requires tensor_model_parallel_size=1, "
                f"got {self.tensor_model_parallel_size}"
            )

    def get_num_layers(self) -> int:
        """Number of DiT blocks in one transformer."""
        return self.num_dit_layers

    @property
    def boundary_timestep(self) -> Optional[float]:
        """Routing boundary on the ``[0, num_train_timesteps]`` axis."""
        if self.boundary_ratio is None:
            return None
        return float(self.boundary_ratio) * float(self.num_train_timesteps)

    @property
    def timestep_window(self) -> Tuple[float, float]:
        """Training timestep window as a fraction of the schedule."""
        return (self.timestep_window_min, self.timestep_window_max)

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def wan2_1_t2v_1_3b(cls, **kwargs) -> "WanConfig":
        """WAN 2.1 T2V-1.3B (~1.11 B parameters)."""
        defaults = dict(
            hidden_size=1536,
            num_attention_heads=12,
            num_dit_layers=30,
            ffn_hidden_size=8960,
            in_channels=16,
            out_channels=16,
            num_transformers=1,
            boundary_ratio=None,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def wan2_1_t2v_14b(cls, **kwargs) -> "WanConfig":
        """WAN 2.1 T2V-14B."""
        defaults = dict(
            hidden_size=5120,
            num_attention_heads=40,
            num_dit_layers=40,
            ffn_hidden_size=13824,
            in_channels=16,
            out_channels=16,
            num_transformers=1,
            boundary_ratio=None,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def wan2_2_ti2v_5b(cls, **kwargs) -> "WanConfig":
        """WAN 2.2 TI2V-5B (single DiT, 48-channel VAE latents)."""
        defaults = dict(
            hidden_size=3072,
            num_attention_heads=24,
            num_dit_layers=30,
            ffn_hidden_size=14336,
            in_channels=48,
            out_channels=48,
            num_transformers=1,
            boundary_ratio=None,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def wan2_2_t2v_a14b(cls, **kwargs) -> "WanConfig":
        """WAN 2.2 T2V-A14B (dual expert, canonical boundary 0.875)."""
        defaults = dict(
            hidden_size=5120,
            num_attention_heads=40,
            num_dit_layers=40,
            ffn_hidden_size=13824,
            in_channels=16,
            out_channels=16,
            num_transformers=2,
            boundary_ratio=0.875,
        )
        defaults.update(kwargs)
        return cls(**defaults)


__all__ = ["WanConfig", "WAN_STAGES", "WAN_LOSS_WEIGHTINGS"]
