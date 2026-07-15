###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Flux Pretrain Trainer for Primus-Megatron.

This trainer implements Flux-specific training logic including:
    - Flow matching scheduler with dynamic shifting
    - Guidance embedding support
    - Custom forward step function
"""

import os

import numpy as np
import torch
import torch.nn as nn

from primus.backends.megatron.diffusion_trainer import DiffusionPretrainTrainer
from primus.backends.megatron.training.diffusion.schedulers import (
    FlowMatchEulerDiscreteScheduler,
)
from primus.backends.megatron.training.diffusion.timestep_sampling import (
    create_timestep_sampler,
)
from primus.core.utils.module_utils import log_rank_0


def _restore_chimera_rng_state(args) -> None:
    """Restore canonical RNG state after chimera model init.

    Calls Megatron's `_set_random_seed` to restore CPU, CUDA default, and
    model-parallel tracker generators to a canonical (per-rank-uniform) state.
    Falls back to a manual restore if Megatron's signature changes.

    Raises:
        May propagate exceptions other than ImportError/TypeError from
        _set_random_seed. The fallback handles ImportError (missing module)
        and TypeError (API signature changes) gracefully.
    """
    try:
        from megatron.training.initialize import _set_random_seed

        _set_random_seed(
            args.seed,
            args.data_parallel_random_init,
            args.te_rng_tracker,
            args.inference_rng_tracker,
            use_cudagraphable_rng=getattr(args, "enable_cuda_graph", False),
        )
    except (ImportError, TypeError) as e:
        import logging

        from megatron.core.tensor_parallel import random as tp_random

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        tp_random.model_parallel_cuda_manual_seed(args.seed)
        logging.warning(
            "[nemo_chimera_init] _set_random_seed API changed (%s). "
            "Restored all RNG generators manually. Verify convergence "
            "matches expected behavior.",
            e,
        )


class FluxPretrainTrainer(DiffusionPretrainTrainer):
    """
    Trainer for Flux diffusion model pre-training.

    Flux-specific features:
        - Flow matching with Euler discrete scheduler
        - Dynamic timestep shifting for variable resolution
        - Optional guidance embedding for CFG

    Config access via backend_args:
        - Megatron args (guidance_embed, guidance_scale, etc.) via backend_args
        - Primus-specific config (torch_compile) via backend_args (overrides section)
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize Flux pretrain trainer.

        Args:
            *args: Positional arguments passed to parent trainer
            **kwargs: Keyword arguments passed to parent trainer
                     (backend_args is extracted from kwargs by BaseTrainer.__init__())
        """
        super().__init__(*args, **kwargs)

        self._training_rng_seeded = False

        # backend_args is set by BaseTrainer.__init__()
        params = self.backend_args

        # Default False: per-step CUDA RNG reseed is an MLPerf-alignment
        # feature; flipping it on globally changes the observable RNG
        # sequence for every consumer of this trainer. MLPerf-aligned YAMLs
        # opt in explicitly via per_step_rng_reseed: true.
        self.per_step_rng_reseed = getattr(params, "per_step_rng_reseed", False)
        self.nemo_chimera_init = getattr(params, "nemo_chimera_init", False)

        # These come from the 'overrides' section in the YAML
        self.use_guidance_embed = getattr(params, "guidance_embed", False)
        self.guidance_scale = getattr(params, "guidance_scale", 3.5)

        # Scheduler config
        self.num_train_timesteps = getattr(params, "num_train_timesteps", 1000)
        self.scheduler_shift = getattr(params, "scheduler_shift", 1.0)
        self.use_dynamic_shifting = getattr(params, "use_dynamic_shifting", False)

        # Timestep sampling strategy (MLPerf uses "direct_uniform")
        timestep_strategy = getattr(params, "timestep_sampling_strategy", "logit_normal")
        self.timestep_sampler = create_timestep_sampler(timestep_strategy)
        log_rank_0(
            f"Timestep sampling strategy: {timestep_strategy} -> {type(self.timestep_sampler).__name__}"
        )

        # CFG dropout: replace text embeddings with fixed empty encodings at this probability
        self.cfg_dropout_prob = getattr(params, "cfg_dropout_prob", 0.0)
        self.empty_t5_encodings = None
        self.empty_clip_encodings = None

        if self.cfg_dropout_prob > 0.0:
            self._init_cfg_dropout(params)

        # VAE latent normalization (matches NVIDIA MLPerf v5.1)
        self.vae_scale = getattr(params, "vae_scale", None)
        self.vae_shift = getattr(params, "vae_shift", None)
        if self.vae_scale is not None:
            log_rank_0(f"VAE normalization: scale={self.vae_scale}, shift={self.vae_shift}")

        # VAE latent mode: "presampled" uses stored latents directly,
        # "resample" re-draws from (mean, logvar) each step
        self.vae_latent_mode = getattr(params, "vae_latent_mode", "presampled")
        if self.vae_latent_mode not in ("presampled", "resample"):
            raise ValueError(
                f"vae_latent_mode must be 'presampled' or 'resample', " f"got '{self.vae_latent_mode}'"
            )
        if self.vae_latent_mode == "resample":
            if self.vae_scale is None or self.vae_shift is None:
                raise ValueError(
                    "vae_latent_mode='resample' requires vae_scale and vae_shift "
                    "to be set (e.g. vae_scale: 0.3611, vae_shift: 0.1159)"
                )
            log_rank_0(f"VAE latent mode: resample (reparameterization from mean+logvar each step)")
        else:
            log_rank_0(f"VAE latent mode: presampled (stored latents used directly)")

        log_rank_0(f"Guidance embedding: {self.use_guidance_embed}")
        log_rank_0(f"Scheduler shift: {self.scheduler_shift}")
        log_rank_0(f"Dynamic shifting: {self.use_dynamic_shifting}")

    def _init_cfg_dropout(self, params):
        """
        Initialize CFG dropout with real empty encodings.

        Resolution order for empty encodings:
        1. Explicit ``empty_encodings_path`` from config
        2. ``{data_path}/empty_encodings/`` (generated by EncodedDatasetPipeline)
        3. ``{data_path}/../empty_encodings/`` (MLPerf convention)

        For mock_data runs, falls back to torch.randn().
        For real data, raises FileNotFoundError if no encodings are found.
        """
        tp_size = getattr(params, "tensor_model_parallel_size", 1)
        if tp_size != 1:
            raise ValueError(
                f"CFG dropout requires tensor_model_parallel_size=1 (got {tp_size}). "
                "Different TP ranks would generate different dropout masks, causing divergent forward passes."
            )

        context_dim = getattr(params, "context_dim", 4096)
        vec_in_dim = getattr(params, "vec_in_dim", 768)

        encodings_dir = self._discover_empty_encodings(params)

        if encodings_dir is not None:
            t5_path = os.path.join(encodings_dir, "t5_empty.npy")
            clip_path = os.path.join(encodings_dir, "clip_empty.npy")

            self.empty_t5_encodings = torch.from_numpy(np.load(t5_path))[0].unsqueeze(1)
            self.empty_clip_encodings = torch.from_numpy(np.load(clip_path))[0]

            log_rank_0(
                f"CFG dropout: loaded real empty encodings from {encodings_dir}, "
                f"t5={self.empty_t5_encodings.shape}, clip={self.empty_clip_encodings.shape}"
            )
        elif getattr(params, "mock_data", False):
            image_size = getattr(getattr(params, "mock_dataset", None), "params", None)
            image_size = getattr(image_size, "image_size", 256) if image_size is not None else 256
            image_tokens = (image_size // 8 // 2) ** 2
            total_seq = getattr(params, "seq_length", 512)
            t5_seq_len = total_seq - image_tokens

            self.empty_t5_encodings = torch.randn(t5_seq_len, 1, context_dim)
            self.empty_clip_encodings = torch.randn(vec_in_dim)
            log_rank_0("CFG dropout: using torch.randn() empty encodings (mock_data mode)")
        else:
            data_path = getattr(params, "data_path", "<not set>")
            if isinstance(data_path, list):
                data_path = data_path[0] if data_path else "<not set>"
            raise FileNotFoundError(
                f"CFG dropout requires empty T5/CLIP encodings but none were found.\n"
                f"Searched locations:\n"
                f"  1. empty_encodings_path config key (not set)\n"
                f"  2. {data_path}/empty_encodings/\n"
                f"  3. {os.path.dirname(str(data_path))}/empty_encodings/\n\n"
                f"To fix, either:\n"
                f"  - Re-run dataset preparation with 'primus data diffusion-encoded' (auto-generates them)\n"
                f"  - Run: python tools/generate_empty_encodings.py --output_dir <path>\n"
                f"    and set empty_encodings_path in your YAML config"
            )

        log_rank_0(f"CFG dropout prob: {self.cfg_dropout_prob}")

    @staticmethod
    def _discover_empty_encodings(params) -> "str | None":
        """Return the first valid empty_encodings directory, or None."""

        def _has_files(dirpath):
            return (
                os.path.isdir(dirpath)
                and os.path.isfile(os.path.join(dirpath, "t5_empty.npy"))
                and os.path.isfile(os.path.join(dirpath, "clip_empty.npy"))
            )

        explicit = getattr(params, "empty_encodings_path", None)
        if explicit and _has_files(str(explicit)):
            return str(explicit)

        data_path = getattr(params, "data_path", None)
        if data_path is not None:
            if isinstance(data_path, list):
                data_path = data_path[0] if data_path else None
            if data_path:
                data_path = str(data_path)
                inside = os.path.join(data_path, "empty_encodings")
                if _has_files(inside):
                    log_rank_0(f"CFG dropout: auto-discovered empty encodings at {inside}")
                    return inside
                alongside = os.path.join(os.path.dirname(data_path), "empty_encodings")
                if _has_files(alongside):
                    log_rank_0(f"CFG dropout: auto-discovered empty encodings at {alongside}")
                    return alongside

        return None

    def forward_step(self, data_iterator, model, return_schedule_plan=False):
        """
        Forward step for Flux diffusion training.

        Overrides base class to provide Flux forward step with scheduler and guidance config.
        The base class implementation handles the new return signature correctly, so we just
        call super() to use it.

        On the first call, sets a per-DP-rank random seed so that each data-parallel
        rank independently samples timesteps and noise (matching NeMo's approach).

        Args:
            data_iterator: Data iterator
            model: Diffusion model (Flux)
            return_schedule_plan: Whether to return schedule plan (for pipeline parallelism)

        Returns:
            Tuple of (output_tensor, loss_func_callable)
        """
        if not self._training_rng_seeded:
            from megatron.core import parallel_state
            from megatron.training import get_args

            seed = get_args().seed
            dp_rank = parallel_state.get_data_parallel_rank()
            per_rank_seed = seed + 100 * dp_rank
            torch.manual_seed(per_rank_seed)
            torch.cuda.manual_seed(per_rank_seed)
            self._training_rng_seeded = True
            log_rank_0(f"Per-DP-rank training seed: {per_rank_seed} " f"(base={seed}, dp_rank={dp_rank})")

        return super().forward_step(data_iterator, model, return_schedule_plan=return_schedule_plan)

    def create_model(self, pre_process=True, post_process=True):
        """
        Create Flux model from YAML configuration.

        Model architecture is loaded from YAML files:
        - flux_12b.yaml / flux_535m.yaml for layer counts
        - flux_base.yaml for common Flux architecture

        Optionally loads checkpoint after model creation if configured.

        Args:
            pre_process: Not used (kept for Megatron model_provider interface compatibility)
            post_process: Not used (kept for Megatron model_provider interface compatibility)

        Returns:
            Flux model instance
        """
        try:
            log_rank_0("=" * 80)
            log_rank_0(f"Creating Flux model from YAML config")

            from megatron.training import get_args

            from primus.backends.megatron.core.models.diffusion.flux.model import Flux

            config = self._build_flux_config_from_yaml()

            # get_args() is safe here since create_model() is called after setup() completes
            # setup() calls set_primus_global_variables() which initializes _GLOBAL_ARGS
            args = get_args()
            # Log complete FluxConfig at DEBUG level (matches Megatron's argument logging)
            self._log_flux_config(config, args)

            # Set torch_compile settings on args for trainer's apply_torch_compile_if_enabled method
            # FluxConfig always has these attributes (defined as dataclass fields)
            torch_compile_attrs = [
                "enable_torch_compile",
                "torch_compile_backend",
                "torch_compile_mode",
                "torch_compile_fullgraph",
                "torch_compile_optimizer",
                "torch_compile_optimizer_scope",
            ]
            for attr in torch_compile_attrs:
                if not hasattr(args, attr):
                    setattr(args, attr, getattr(config, attr))

            # Backend selection is handled automatically by Flux model based on config.transformer_impl
            # Pass backend=None to let get_flux_layer_spec() handle backend selection
            backend = None

            # Log which transformer implementation will be used
            if config.transformer_impl == "local":
                log_rank_0("Using local transformer implementation (NO TransformerEngine dependency)")
            else:
                log_rank_0("Using TransformerEngine implementation")
            log_rank_0(
                "Backend will be selected automatically by Flux model based on config.transformer_impl"
            )

            # Chimera init: replicate NeMo's per-rank seed contamination where
            # each DP rank initializes non-parallel weights with a different seed.
            # The distributed optimizer merges these into a chimera model.
            if self.nemo_chimera_init:
                import logging

                from megatron.core import parallel_state

                dp_rank = parallel_state.get_data_parallel_rank()
                per_rank_seed = args.seed + 100 * dp_rank
                torch.manual_seed(per_rank_seed)
                torch.cuda.manual_seed(per_rank_seed)
                logging.warning(
                    "[nemo_chimera_init] WARNING: DP weight invariant intentionally broken. "
                    "Each rank initializes non-parallel layers with a different seed. "
                    "This replicates NeMo's per-rank seed contamination for convergence "
                    "parity experiments only. dp_rank=%d, init_seed=%d (base=%d)",
                    dp_rank,
                    per_rank_seed,
                    args.seed,
                )

            # Create Flux model (backend=None lets model select based on config.transformer_impl)
            model = Flux(config=config, backend=backend)

            if self.nemo_chimera_init:
                _restore_chimera_rng_state(args)
                log_rank_0(
                    f"[nemo_chimera_init] Restored canonical RNG state (seed={args.seed}) after chimera model init"
                )

            # Calculate parameters for logging
            total_params = sum(p.numel() for p in model.parameters())

            log_rank_0(
                f"Flux model created: {config.num_joint_layers} joint + {config.num_single_layers} single layers"
            )
            log_rank_0(f"Total parameters: {total_params / 1e6:.1f}M")

            log_rank_0("=" * 80)

            return model
        except Exception as e:
            log_rank_0(f"[ERROR] create_model() failed: {type(e).__name__}: {e}")
            import traceback

            log_rank_0(f"[ERROR] Traceback:\n{traceback.format_exc()}")
            raise

    def _build_flux_config_from_yaml(self):
        """
        Build FluxConfig from YAML configuration + training args.

        All architectural parameters come from YAML files (merged into Megatron args):
        - flux_535m.yaml or flux_12b.yaml specifies layer counts
        - flux_base.yaml provides common architecture parameters

        Training precision and recomputation settings come from backend_args.
        Torch compile settings come from backend_args.torch_compile (Primus-specific, in overrides section, not in Megatron args).

        Returns:
            FluxConfig instance with all parameters
        """
        from functools import partial

        import torch.nn.functional as F

        from primus.backends.megatron.core.models.diffusion.flux.config import (
            FluxConfig,
            erf_gelu,
        )

        _openai_gelu_fused = partial(F.gelu, approximate="tanh")
        _openai_gelu_fused.__name__ = "openai_gelu_fused"

        # backend_args is set by BaseTrainer.__init__()
        params = self.backend_args

        # All architectural params from params (merged from YAML)
        config_params = {
            "num_joint_layers": getattr(params, "num_joint_layers", 19),
            "num_single_layers": getattr(params, "num_single_layers", 38),
            "hidden_size": getattr(params, "hidden_size", 3072),
            "num_attention_heads": getattr(params, "num_attention_heads", 24),
            "in_channels": getattr(params, "in_channels", 64),
            "context_dim": getattr(params, "context_dim", 4096),
            "vec_in_dim": getattr(params, "vec_in_dim", 768),
            "model_channels": getattr(params, "model_channels", 256),
            "guidance_embed": getattr(params, "guidance_embed", False),
            # RoPE configuration
            "apply_rope_fusion": getattr(params, "apply_rope_fusion", False),
            "rotary_interleaved": getattr(params, "rotary_interleaved", True),
            # Training hyperparameters stored on FluxConfig
            "timestep_sampling_strategy": getattr(params, "timestep_sampling_strategy", "logit_normal"),
            "cfg_dropout_prob": getattr(params, "cfg_dropout_prob", 0.0),
            # Weight init for Megatron-managed layers (ColumnParallelLinear, RowParallelLinear).
            # Xavier uniform matches NeMo's CustomFluxConfig (MLPerf v5.1).
            "init_method": nn.init.xavier_uniform_,
            "output_layer_init_method": nn.init.xavier_uniform_,
        }

        # Activation: YAML "openai_gelu" maps to fused F.gelu(approximate="tanh"); default in
        # FluxConfig is the non-JIT Python GELU (ROCm-safe). Both match the tanh GELU approx.
        activation_func_name = getattr(params, "activation_func", None)
        if activation_func_name is not None:
            activation_func_map = {
                "erf_gelu": erf_gelu,
                "openai_gelu": _openai_gelu_fused,
            }
            if activation_func_name not in activation_func_map:
                raise ValueError(
                    f"Unknown activation_func '{activation_func_name}'. "
                    f"Choose from: {list(activation_func_map.keys())}"
                )
            func = activation_func_map[activation_func_name]
            config_params["activation_func"] = func
            log_rank_0(f"Activation function: {activation_func_name} -> {func.__name__}")

        # Transformer implementation (standard TransformerConfig parameter)
        # Read from backend_args (overrides section)
        transformer_impl = getattr(params, "transformer_impl", "transformer_engine")
        config_params["transformer_impl"] = transformer_impl

        # Precision settings from params (not in model YAML)
        config_params.update(
            {
                "bf16": getattr(params, "bf16", True),
                "fp16": getattr(params, "fp16", False),
                "params_dtype": getattr(params, "params_dtype", torch.float32),
            }
        )

        # FP8 settings from params
        config_params.update(
            {
                "fp8": getattr(params, "fp8", None),
                "fp8_recipe": getattr(params, "fp8_recipe", "delayed"),
                "fp8_margin": getattr(params, "fp8_margin", 0),
                "fp8_amax_history_len": getattr(params, "fp8_amax_history_len", 1),
                "fp8_amax_compute_algo": getattr(params, "fp8_amax_compute_algo", "most_recent"),
                "fp8_wgrad": getattr(params, "fp8_wgrad", True),
                "fp8_dot_product_attention": getattr(params, "fp8_dot_product_attention", False),
                "fp8_multi_head_attention": getattr(params, "fp8_multi_head_attention", False),
                "fp8_scaling_strategy": getattr(params, "fp8_scaling_strategy", "dynamic"),
                "fp8_force_nt_layout": getattr(params, "fp8_force_nt_layout", False),
                "fp8_reduce_amax": getattr(params, "fp8_reduce_amax", False),
            }
        )

        # FP4/MXFP4 settings
        fp4_enabled = getattr(params, "fp4", None)
        fp4_recipe = getattr(params, "fp4_recipe", None)
        if fp4_enabled and not fp4_recipe:
            raise ValueError(
                "fp4_recipe must be explicitly set in YAML when fp4 is enabled. "
                "Use fp4_recipe: 'mxfp4' for AMD (native FP4 GEMM) or 'nvfp4' for NVIDIA."
            )
        config_params.update(
            {
                "fp4": fp4_enabled,
                "fp4_recipe": fp4_recipe,
                "mxfp4_backward_precision": getattr(params, "mxfp4_backward_precision", "mxfp4"),
            }
        )

        # Sensitive layer configuration
        config_params.update(
            {
                "sensitive_layers_enabled": getattr(params, "sensitive_layers_enabled", False),
                "sensitive_layers_start": getattr(params, "sensitive_layers_start", 0),
                "sensitive_layers_end": getattr(params, "sensitive_layers_end", 0),
                "sensitive_layer_precision": getattr(params, "sensitive_layer_precision", "bf16"),
                "mxfp4_gradient_stochastic_rounding": getattr(
                    params, "mxfp4_gradient_stochastic_rounding", False
                ),
            }
        )

        config_params["use_dual_fp8_output_projection"] = getattr(
            params,
            "use_dual_fp8_output_projection",
            False,
        )

        if (
            config_params["use_dual_fp8_output_projection"]
            and config_params.get("fp8_scaling_strategy") == "delayed"
        ):
            raise ValueError(
                "use_dual_fp8_output_projection=True is incompatible with "
                "fp8_scaling_strategy='delayed'. DualFP8LinearTensorwiseFunction "
                "bypasses delayed-scaling staged amax buffers, causing stale/zero "
                "values in the _DelayedScalingRegistry. Use "
                "fp8_scaling_strategy='dynamic' or set "
                "use_dual_fp8_output_projection=False."
            )

        config_params["use_triton_ops"] = getattr(
            params,
            "use_triton_ops",
            False,
        )
        config_params["adaln_plain_ops"] = getattr(
            params,
            "adaln_plain_ops",
            False,
        )
        config_params["adaln_always_jit_fuser"] = getattr(
            params,
            "adaln_always_jit_fuser",
            False,
        )
        config_params["optimizer_foreach"] = getattr(
            params,
            "optimizer_foreach",
            True,
        )
        config_params["overlap_grad_norm"] = getattr(
            params,
            "overlap_grad_norm",
            False,
        )
        config_params["use_cpp_fp8_quantize"] = getattr(
            params,
            "use_cpp_fp8_quantize",
            False,
        )

        # FSDP2 prefetch depth
        fsdp_prefetch = getattr(params, "fsdp_prefetch_depth", None)
        if fsdp_prefetch is not None:
            config_params["fsdp_prefetch_depth"] = int(fsdp_prefetch)

        # FSDP2 FP32 optimizer: initialize model in FP32, FSDP2 casts to BF16
        if getattr(params, "use_fsdp2_fp32_param_optimizer", False):
            config_params["params_dtype"] = torch.float32
            config_params["pipeline_dtype"] = torch.bfloat16
            log_rank_0("FSDP2 FP32 optimizer: params_dtype=FP32, pipeline=BF16")

        # Recomputation settings from params
        config_params.update(
            {
                "recompute_granularity": getattr(params, "recompute_granularity", None),
                "recompute_method": getattr(params, "recompute_method", None),
                "recompute_num_layers": getattr(params, "recompute_num_layers", None),
                "recompute_modules": getattr(params, "recompute_modules", None),
            }
        )

        # Torch compile settings from backend_args (overrides section)
        # torch_compile is in overrides section in YAML, accessible via backend_args.torch_compile
        torch_compile_config = getattr(self.backend_args, "torch_compile", None)

        if torch_compile_config is not None:
            compile_settings = {
                "enable_torch_compile": getattr(torch_compile_config, "enable", False),
                "torch_compile_backend": getattr(torch_compile_config, "backend", "inductor"),
                "torch_compile_mode": getattr(torch_compile_config, "mode", "default"),
                "torch_compile_fullgraph": getattr(torch_compile_config, "fullgraph", False),
                "torch_compile_optimizer": getattr(torch_compile_config, "compile_optimizer", False),
                "torch_compile_optimizer_scope": getattr(
                    torch_compile_config, "compile_optimizer_scope", "full"
                ),
                "torch_compile_strategy": getattr(torch_compile_config, "strategy", "per_block"),
                "torch_compile_replace_qk_rmsnorm": getattr(
                    torch_compile_config, "replace_qk_rmsnorm", False
                ),
                "torch_compile_disable_inductor_cudagraphs": getattr(
                    torch_compile_config, "disable_inductor_cudagraphs", True
                ),
                "torch_compile_emulate_precision_casts": getattr(
                    torch_compile_config, "emulate_precision_casts", True
                ),
                "torch_compile_fused_ln_modulate": getattr(torch_compile_config, "fused_ln_modulate", True),
            }
        else:
            # Default values if torch_compile section not present
            compile_settings = {
                "enable_torch_compile": False,
                "torch_compile_backend": "inductor",
                "torch_compile_mode": "default",
                "torch_compile_fullgraph": False,
                "torch_compile_optimizer": False,
                "torch_compile_optimizer_scope": "full",
                "torch_compile_strategy": "per_block",
                "torch_compile_replace_qk_rmsnorm": False,
                "torch_compile_disable_inductor_cudagraphs": True,
                "torch_compile_emulate_precision_casts": True,
                "torch_compile_fused_ln_modulate": True,
            }

        # Set on FluxConfig
        config_params.update(compile_settings)

        return FluxConfig(**config_params)

    def _log_flux_config(self, config, args):
        """Print FluxConfig fields (rank 0 only), similar to Megatron argument dumps."""
        import dataclasses

        # Only log on rank 0
        if args.rank != 0:
            return

        # Accumulate the full dump into one multi-line message so the banner
        # alignment is preserved (log_rank_0 stamps a caller prefix per call).
        lines = []
        lines.append("=" * 80)
        lines.append("FluxConfig (Model Configuration)")
        lines.append("=" * 80)

        # Organize config parameters by category for better readability
        categories = {
            "Model Architecture": [
                "model_type",
                "num_joint_layers",
                "num_single_layers",
                "num_layers",
                "hidden_size",
                "num_attention_heads",
                "ffn_hidden_size",
            ],
            "Input/Output Dimensions": [
                "in_channels",
                "out_channels",
                "patch_size",
                "context_dim",
                "vec_in_dim",
                "model_channels",
            ],
            "Position Embeddings (RoPE)": ["theta", "axes_dim", "rotary_interleaved", "apply_rope_fusion"],
            "Guidance & Diffusion": [
                "guidance_embed",
                "guidance_scale",
                "cfg_dropout_prob",
                "timestep_sampling_strategy",
            ],
            "Attention & Layers": [
                "add_qkv_bias",
                "single_block_bias",
                "attention_dropout",
                "hidden_dropout",
                "bias_dropout_fusion",
            ],
            "Normalization & Activation": ["activation_func", "layernorm_epsilon", "normalization"],
            "Precision & Optimization": [
                "bf16",
                "fp16",
                "fp32_residual_connection",
                "gradient_accumulation_fusion",
                "use_dual_fp8_output_projection",
                "params_dtype",
                "fp8",
                "fp8_recipe",
                "fp8_scaling_strategy",
                "fp8_force_nt_layout",
                "fp4",
                "fp4_recipe",
                "mxfp4_backward_precision",
                "mxfp4_gradient_stochastic_rounding",
                "sensitive_layers_enabled",
                "sensitive_layers_start",
                "sensitive_layers_end",
                "sensitive_layer_precision",
            ],
            "Recomputation": [
                "recompute_granularity",
                "recompute_method",
                "recompute_num_layers",
                "recompute_modules",
            ],
            "Torch Compile": [
                "enable_torch_compile",
                "torch_compile_backend",
                "torch_compile_mode",
                "torch_compile_fullgraph",
                "torch_compile_optimizer",
                "torch_compile_optimizer_scope",
                "torch_compile_strategy",
                "torch_compile_replace_qk_rmsnorm",
                "torch_compile_disable_inductor_cudagraphs",
                "torch_compile_emulate_precision_casts",
                "torch_compile_fused_ln_modulate",
            ],
            "Parallelism": [
                "tensor_model_parallel_size",
                "pipeline_model_parallel_size",
                "sequence_parallel",
                "expert_model_parallel_size",
            ],
            "CUDA Graph": ["enable_cuda_graph", "cuda_graph_scope", "cuda_graph_warmup_steps"],
            "Transformer Engine": ["use_te_rng_tracker"],
        }

        # Get all dataclass fields
        all_fields = {f.name for f in dataclasses.fields(config)}
        categorized_fields = set()

        # Print categorized fields
        for category, field_names in categories.items():
            # Check if any fields in this category exist
            existing_fields = [f for f in field_names if f in all_fields]
            if not existing_fields:
                continue

            lines.append(f"\n{category}:")
            for field_name in existing_fields:
                if hasattr(config, field_name):
                    value = getattr(config, field_name)
                    # Format value (handle callables specially)
                    if callable(value) and not isinstance(value, type):
                        value_str = getattr(value, "__name__", str(value))
                    else:
                        value_str = str(value)

                    # Create dots for alignment (match Megatron's style: 48 chars)
                    dots = "." * (48 - len(field_name))
                    lines.append(f"  {field_name} {dots} {value_str}")
                    categorized_fields.add(field_name)

        # Print uncategorized fields (any fields not in our category lists)
        uncategorized = all_fields - categorized_fields
        if uncategorized:
            lines.append("\nOther Configuration:")
            for field_name in sorted(uncategorized):
                if hasattr(config, field_name):
                    value = getattr(config, field_name)
                    if callable(value) and not isinstance(value, type):
                        value_str = getattr(value, "__name__", str(value))
                    else:
                        value_str = str(value)
                    dots = "." * (48 - len(field_name))
                    lines.append(f"  {field_name} {dots} {value_str}")

        # Add special note about apply_rope_fusion mismatch
        if hasattr(config, "apply_rope_fusion") and hasattr(args, "apply_rope_fusion"):
            if config.apply_rope_fusion != args.apply_rope_fusion:
                lines.append("\n" + "!" * 80)
                lines.append("IMPORTANT: FluxConfig vs Megatron Args Mismatch")
                lines.append("!" * 80)
                lines.append(f"  FluxConfig.apply_rope_fusion: {config.apply_rope_fusion}")
                lines.append(f"  Megatron args.apply_rope_fusion: {args.apply_rope_fusion}")
                lines.append(
                    f'  → Flux uses FluxConfig value: RoPE fusion is {"ENABLED" if config.apply_rope_fusion else "DISABLED"}'
                )
                lines.append("  → Megatron args value is set by position_embedding_type validation")
                lines.append(
                    '  → (Megatron sets apply_rope_fusion=False when position_embedding_type != "rope")'
                )
                lines.append("!" * 80)

        lines.append("=" * 80)
        lines.append("End of FluxConfig")
        lines.append("=" * 80)

        log_rank_0("\n".join(lines))

    def create_scheduler(self):
        """
        Create Flow Matching Euler Discrete Scheduler for Flux.

        Returns:
            FlowMatchEulerDiscreteScheduler instance
        """
        log_rank_0("Creating Flow Matching scheduler...")
        log_rank_0(f"  num_train_timesteps: {self.num_train_timesteps}")
        log_rank_0(f"  shift: {self.scheduler_shift}")
        log_rank_0(f"  use_dynamic_shifting: {self.use_dynamic_shifting}")

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=self.scheduler_shift,
            use_dynamic_shifting=self.use_dynamic_shifting,
            base_shift=0.5,  # Flux defaults
            max_shift=1.15,
            base_image_seq_len=256,
            max_image_seq_len=4096,
        )

        return scheduler

    def get_task_encoder(self):
        """
        Get Flux task encoder for Energon data pipeline.

        Returns:
            EncodedDiffusionTaskEncoder for pre-encoded data
        """
        from primus.backends.megatron.data.diffusion.task_encoders import (
            EncodedDiffusionTaskEncoder,
        )

        # EnergonDatasetProvider will pass WorkerConfig to Energon directly
        # TaskEncoder doesn't need WorkerConfig for pre-encoded data
        task_encoder = EncodedDiffusionTaskEncoder(worker_config=None)

        log_rank_0("Created EncodedDiffusionTaskEncoder for pre-encoded Flux data")
        return task_encoder
