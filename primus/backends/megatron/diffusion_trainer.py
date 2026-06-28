###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Base diffusion trainer for Megatron-LM.

This trainer provides a foundation for diffusion model training by overriding
the forward step and dataset provider methods from MegatronPretrainTrainer.
"""

from abc import abstractmethod

import torch
import torch.nn.functional as F

from primus.backends.megatron.megatron_pretrain_trainer import MegatronPretrainTrainer
from primus.modules.module_utils import log_rank_0


class DiffusionPretrainTrainer(MegatronPretrainTrainer):
    """
    Base trainer for diffusion models.

    This trainer inherits from the backend's MegatronPretrainTrainer (which uses
    MegatronBaseTrainer).

    It overrides the forward step and dataset provider methods to support
    diffusion-specific training. Subclasses should implement create_scheduler()
    and may override create_model() if needed.

    Configuration is accessed via backend_args (set by BaseTrainer.__init__()),
    not via module_config.params, to avoid BaseModule dependency.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize diffusion trainer.

        Args:
            *args: Positional arguments passed to parent trainer
            **kwargs: Keyword arguments passed to parent trainer
                     (backend_args is extracted from kwargs by BaseTrainer)
        """
        super().__init__(*args, **kwargs)

        self._scheduler = None
        self._compiled_loss_fn = None
        self._forward_step_count = 0
        self._forward_step_count_initialized = False

        # Composition pattern: avoids recreating the provider on each call
        use_mock_data = getattr(self.backend_args, "mock_data", False)

        if use_mock_data:
            from primus.backends.megatron.data.synthetic_dataset_provider import (
                SyntheticDatasetProvider,
            )

            # Get mock dataset configuration from YAML (if specified)
            mock_dataset_config_ns = getattr(self.backend_args, "mock_dataset", None)
            mock_dataset_config = self._convert_namespace_to_dict(mock_dataset_config_ns)

            # Determine model type for default dataset selection
            model_type_raw = getattr(self.backend_args, "model_type", None)
            model_type = (
                model_type_raw
                if (isinstance(model_type_raw, str) and model_type_raw != "diffusion_model")
                else "flux"
            )

            self.data_provider = SyntheticDatasetProvider(
                dataset_config=mock_dataset_config, model_type=model_type
            )
            log_rank_0("Created SyntheticDatasetProvider for mock data")
        else:
            from primus.backends.megatron.data.energon_dataset_provider import (
                EnergonDatasetProvider,
            )

            self.data_provider = EnergonDatasetProvider(task_encoder_factory=lambda: self.get_task_encoder())
            log_rank_0("Created EnergonDatasetProvider for real data")

        log_rank_0(f"{self.__class__.__name__} initialized")
        log_rank_0(f"Data provider: {type(self.data_provider).__name__}")

    def _convert_namespace_to_dict(self, ns_obj):
        """
        Convert SimpleNamespace (or nested SimpleNamespace) to dict.

        Handles nested namespaces by recursively converting them.

        Args:
            ns_obj: SimpleNamespace, dict, or None

        Returns:
            dict, original input (if already a dict or primitive), or None
        """
        if ns_obj is None:
            return None

        if not hasattr(ns_obj, "__dict__"):
            return ns_obj  # Already a dict or other type

        result = vars(ns_obj)

        # Recursively convert nested namespaces
        if "params" in result and hasattr(result["params"], "__dict__"):
            result["params"] = vars(result["params"])

        return result

    def setup(self):
        """
        Override setup() to inject diffusion model_provider.

        MegatronBaseTrainer.setup() handles Megatron path, global vars, and parse_args patching.
        We only need to set the model_provider here.
        """
        # Ensure data_parallel_size is set before parent setup() calls set_primus_global_variables()
        # This is required by set_primus_global_variables()
        if not hasattr(self.backend_args, "data_parallel_size"):
            world_size = getattr(self.backend_args, "world_size", 1)
            tensor_model_parallel_size = getattr(self.backend_args, "tensor_model_parallel_size", 1)
            pipeline_model_parallel_size = getattr(self.backend_args, "pipeline_model_parallel_size", 1)
            context_parallel_size = getattr(self.backend_args, "context_parallel_size", 1)
            data_parallel_size = world_size // (
                tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
            )
            setattr(self.backend_args, "data_parallel_size", data_parallel_size)
            log_rank_0(
                f"Computed data_parallel_size={data_parallel_size} from world_size={world_size}, tp={tensor_model_parallel_size}, pp={pipeline_model_parallel_size}, cp={context_parallel_size}"
            )

        # Create and set diffusion model provider
        def _diffusion_model_provider(
            pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None
        ):
            """
            Model provider wrapper for diffusion models.

            This matches Megatron's model_provider signature but calls
            our diffusion-specific create_model() instead of gpt_builder.

            Note: vp_stage, config, and pg_collection are accepted for interface
            compatibility with Megatron's model_provider signature but are not
            used by diffusion models.
            """
            return self.create_model(pre_process=pre_process, post_process=post_process)

        self.model_provider = _diffusion_model_provider
        log_rank_0("=" * 80)
        log_rank_0("Overridden model_provider to use diffusion model builder")
        log_rank_0("=" * 80)

        # Call parent's setup() which handles Megatron initialization
        super().setup()

    @abstractmethod
    def create_model(self, pre_process=True, post_process=True):
        """
        Create diffusion model instance.

        Returns:
            Model instance (e.g., Flux)

        Example:
            from primus.backends.megatron.core.models.diffusion.flux import Flux
            return Flux(config=self.model_config)
        """
        raise NotImplementedError("Subclasses must implement create_model()")

    def _get_loss_fn(self):
        """Return the loss function, optionally compiled.

        When torch.compile is enabled for the model, the loss function is also
        compiled as a standalone region.  This fuses the ~10 eager ATen ops
        (sub, float casts, mse_loss, mean) into 1-2 kernels and gives the
        autograd engine a single CompiledFunctionBackward node instead of
        multiple eager backward nodes.

        The compiled function is created lazily on first call and cached.
        """
        if self._compiled_loss_fn is not None:
            return self._compiled_loss_fn

        from primus.backends.megatron.training.diffusion.loss_computation import (
            compute_flow_matching_loss,
        )

        try:
            from megatron.training import get_args

            args = get_args()
            compile_enabled = getattr(args, "enable_torch_compile", False)
        except Exception:
            compile_enabled = False

        if compile_enabled:
            import torch

            self._compiled_loss_fn = torch.compile(
                compute_flow_matching_loss,
                backend="inductor",
                fullgraph=False,
            )
            log_rank_0("[DiffusionPretrainTrainer] Loss function compiled with torch.compile")
        else:
            self._compiled_loss_fn = compute_flow_matching_loss

        return self._compiled_loss_fn

    def forward_step(self, data_iterator, model, return_schedule_plan=False):
        """
        Forward training step for diffusion models.

        Args:
            data_iterator: Data iterator
            model: Diffusion model (Flux)
            return_schedule_plan: Whether to return schedule plan (for pipeline parallelism)

        Returns:
            Tuple of (noise_pred, loss_func_callable)
        """
        from primus.backends.megatron.training.diffusion.forward_step import (
            flux_forward_step_func,
        )

        # Skip counter advance in eval. Advancing _forward_step_count on
        # validation steps would shift the next training step's per-step seed
        # by eval_iters * num_microbatches per --eval-interval window,
        # defeating the goal of isolating training RNG from unrelated forward
        # passes. Eval forward passes reuse the most recent training counter
        # value, so the per-step CUDA reseed is a no-op replay during eval.
        if model.training:
            self._forward_step_count += 1

        # Megatron's pattern: forward_step returns model output, loss_func computes loss
        noise_pred, clean_latents, noise, loss_mask, metrics, is_validation = flux_forward_step_func(
            data_iterator,
            model,
            scheduler=self.scheduler,
            use_guidance_embed=getattr(self, "use_guidance_embed", False),
            guidance_scale=getattr(self, "guidance_scale", None),
            timestep_sampler=getattr(self, "timestep_sampler", None),
            cfg_dropout_prob=getattr(self, "cfg_dropout_prob", 0.0),
            empty_t5_encodings=getattr(self, "empty_t5_encodings", None),
            empty_clip_encodings=getattr(self, "empty_clip_encodings", None),
            vae_scale=getattr(self, "vae_scale", None),
            vae_shift=getattr(self, "vae_shift", None),
            vae_latent_mode=getattr(self, "vae_latent_mode", "presampled"),
            per_step_rng_reseed=getattr(self, "per_step_rng_reseed", False),
            step_count=self._forward_step_count,
        )

        # Store values needed for loss computation (will be used by loss function)
        self._last_clean_latents = clean_latents
        self._last_noise = noise
        self._last_loss_mask = loss_mask

        # Store metrics in runtime state
        if hasattr(self, "runtime_state") and self.runtime_state:
            self.runtime_state.update_metrics(metrics)
        else:
            log_rank_0("[DiffusionPretrainTrainer] WARNING: runtime_state not available, metrics not stored")

        if is_validation:

            def val_loss_func(output_tensor, non_loss_data=False):
                if non_loss_data:
                    return output_tensor
                target = self._last_noise - self._last_clean_latents
                loss = F.mse_loss(output_tensor.float(), target.float(), reduction="none")
                loss_per_sample = loss.mean(dim=tuple(range(1, loss.ndim)))
                loss_sum = loss_per_sample.sum()
                sample_count = torch.tensor(
                    loss_per_sample.numel(), dtype=loss_sum.dtype, device=loss_sum.device
                )
                return loss_sum, {"loss": (loss_sum.detach(), sample_count.detach())}

            return noise_pred, val_loss_func

        def diffusion_loss_func(output_tensor, non_loss_data=False):
            if non_loss_data:
                return output_tensor

            loss = self._get_loss_fn()(
                output_tensor, self._last_clean_latents, self._last_noise, self._last_loss_mask
            )

            reporting_metrics = {"reduced_train_loss": loss.detach().clone()}

            return loss, reporting_metrics

        return noise_pred, diffusion_loss_func

    def get_forward_step(self):
        """
        Return forward step function for diffusion models.

        Returns a function that wraps self.forward_step() to match
        the interface expected by Megatron's pretrain() function.

        On the first call, reconstructs the forward step counter from
        checkpoint state (args.iteration * num_microbatches) so that
        the per-step RNG seed sequence continues correctly after resume.
        """

        def diffusion_forward_step(data_iterator, model):
            if not self._forward_step_count_initialized:
                from megatron.core.num_microbatches_calculator import (
                    get_num_microbatches,
                )
                from megatron.training import get_args

                args = get_args()
                # Reconstruct counter from checkpoint iteration. Assumes no
                # iterations were skipped (iterations_to_skip is unused in
                # diffusion training).
                self._forward_step_count = args.iteration * get_num_microbatches()
                self._forward_step_count_initialized = True

            return self.forward_step(data_iterator, model, return_schedule_plan=False)

        return diffusion_forward_step

    def get_datasets_provider(self):
        """
        Return dataset provider function that delegates to self.data_provider.

        This uses the composition pattern: the provider function delegates
        to the injected data_provider instance, avoiding recreation on each call.
        """

        def diffusion_datasets_provider(train_val_test_num_samples, vp_stage=None):
            """Delegate to injected data provider."""
            from megatron.training import get_args

            args = get_args()

            return self.data_provider.create_dataloaders(
                trainer_config=args, train_val_test_num_samples=train_val_test_num_samples, vp_stage=vp_stage
            )

        # Mark as distributed (required by Megatron)
        diffusion_datasets_provider.is_distributed = self.data_provider.is_distributed

        # Set __module__ to help with debugging (point to this module, not pretrain_gpt)
        diffusion_datasets_provider.__module__ = __name__

        log_rank_0(
            f"[DiffusionPretrainTrainer] Created datasets provider: {type(self.data_provider).__name__}"
        )
        return diffusion_datasets_provider

    @property
    def scheduler(self):
        """Lazily-initialized diffusion scheduler."""
        if self._scheduler is None:
            self._scheduler = self.create_scheduler()
        return self._scheduler

    @abstractmethod
    def create_scheduler(self):
        """
        Create diffusion scheduler.

        Returns:
            Scheduler instance (e.g., FlowMatchEulerDiscreteScheduler)
        """
        raise NotImplementedError("Subclasses must implement create_scheduler()")

    @abstractmethod
    def get_task_encoder(self):
        """
        Get task encoder for Energon data pipeline.

        Returns:
            TaskEncoder instance (e.g., EncodedDiffusionTaskEncoder)

        This method is called by EnergonDatasetProvider to create the task encoder
        for processing WebDataset samples.
        """
        raise NotImplementedError("Subclasses must implement get_task_encoder()")
