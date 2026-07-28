###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.megatron.megatron_base_trainer import MegatronBaseTrainer
from primus.core.utils.module_utils import log_rank_0


class MegatronPretrainTrainer(MegatronBaseTrainer):
    """Trainer for Megatron-LM pre-training."""

    def setup_model_only(self):
        """Initialize Megatron and build the model WITHOUT running the training loop.

        A general, training-neutral capability: mirrors the front of
        ``megatron.training.pretrain`` (``initialize_megatron`` followed by
        ``setup_model_and_optimizer``) to construct a model identical to the real
        training path, but stops before datasets / the train loop. Useful for any
        "build the model only" scenario (offline profiling, layer benchmarking,
        model inspection); performance/memory projection is the current consumer.

        Prerequisites (handled by the runtime before calling this): ``setup()``
        has patched ``parse_args`` to return ``self.backend_args`` and set the
        Primus global vars, and the build_args/setup/before_train patch phases
        have been applied. Returns the built model (a list of model chunks, as
        produced by megatron's ``get_model``) and also stores it on ``self.model``.
        """
        log_rank_0("Setting up Megatron model only (no training loop)...")

        from megatron.core.enums import ModelType
        from megatron.training.initialize import initialize_megatron
        from megatron.training.training import setup_model_and_optimizer

        from primus.core.utils.import_utils import get_model_provider

        # Determine model type (gpt or mamba) from backend_args
        model_type = getattr(self.backend_args, "model_type", "gpt")
        log_rank_0(f"-detected model_type: {model_type}")

        # parse_args was patched in setup() to return backend_args, so
        # initialize_megatron consumes the Primus-configured arguments (same as
        # the front of megatron's pretrain()).
        initialize_megatron(args_defaults={"tokenizer_type": "GPT2BPETokenizer"})

        # Get model provider with correct model_type (reuse the core runtime helper)
        if model_type != "gpt":
            model_provider = get_model_provider(model_type=model_type)
        else:
            model_provider = get_model_provider()
        log_rank_0(f"-model_provider: {model_provider}")

        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            model_provider, ModelType.encoder_or_decoder, checkpointing_context={}
        )
        self.model = model
        self.optimizer = optimizer
        self.opt_param_scheduler = opt_param_scheduler

        log_rank_0("Megatron model-only setup completed.")
        return model

    def get_forward_step(self):
        """
        Return forward step function for training loop.

        Override this method in subclasses to provide custom forward step functions.
        Default implementation returns GPT forward step.

        Note: This method assumes Megatron-LM path is already set up by
        MegatronBaseTrainer._ensure_megatron_path() during setup().

        Returns:
            Callable: Forward step function compatible with Megatron's training loop.
        """
        # Path should already be set by _ensure_megatron_path() during setup()
        try:
            from pretrain_gpt import forward_step  # type: ignore
        except ImportError as e:
            log_rank_0(f"[MegatronPretrainTrainer] Failed to import forward_step from pretrain_gpt: {e}")
            log_rank_0(
                "[MegatronPretrainTrainer] This indicates a configuration issue. "
                "Ensure Megatron-LM is properly installed at third_party/Megatron-LM"
            )
            raise ImportError(
                "Could not import forward_step from pretrain_gpt. "
                "This indicates that Megatron-LM path setup failed during setup(). "
                "Ensure Megatron-LM is properly installed and _ensure_megatron_path() succeeded."
            ) from e
        return forward_step

    def get_datasets_provider(self):
        """
        Return dataset provider function for training.

        Override this method in subclasses to provide custom dataset providers.
        Default implementation returns GPT dataset provider.

        Note: This method assumes Megatron-LM path is already set up by
        MegatronBaseTrainer._ensure_megatron_path() during setup().

        Returns:
            Callable: Dataset provider function compatible with Megatron's training loop.
        """
        # Path should already be set by _ensure_megatron_path() during setup()
        try:
            from pretrain_gpt import train_valid_test_datasets_provider  # type: ignore
        except ImportError as e:
            log_rank_0(
                f"[MegatronPretrainTrainer] Failed to import train_valid_test_datasets_provider "
                f"from pretrain_gpt: {e}"
            )
            log_rank_0(
                "[MegatronPretrainTrainer] This indicates a configuration issue. "
                "Ensure Megatron-LM is properly installed at third_party/Megatron-LM"
            )
            raise ImportError(
                "Could not import train_valid_test_datasets_provider from pretrain_gpt. "
                "This indicates that Megatron-LM path setup failed during setup(). "
                "Ensure Megatron-LM is properly installed and _ensure_megatron_path() succeeded."
            ) from e

        provider = train_valid_test_datasets_provider
        provider.is_distributed = True  # Always True to match Megatron's behavior
        return provider

    def train(self):
        """Execute Megatron pre-training."""
        log_rank_0("Executing Megatron pretrain...")

        import inspect

        from megatron.core.enums import ModelType
        from megatron.training import pretrain  # type: ignore

        from primus.core.utils.import_utils import get_model_provider

        # Determine model type (gpt / mamba / deepseek_v4 / diffusion) from backend_args
        model_type = getattr(self.backend_args, "model_type", "gpt")
        log_rank_0(f"-detected model_type: {model_type}")

        # Import the appropriate training components based on model_type.
        # DeepSeek-V4 is causal-LM with the same data shape as GPT, so we
        # reuse pretrain_gpt's forward_step + dataset provider; only the
        # model_provider itself is V4-specific.
        if model_type == "mamba":
            from pretrain_mamba import (  # type: ignore
                forward_step,
                train_valid_test_datasets_provider,
            )

            log_rank_0("Using Mamba model provider and training components")
            # Upstream pretrain entrypoints set this in their __main__ blocks, but Primus imports the
            # provider directly and calls pretrain() programmatically. Without restoring this flag,
            # only TP rank 0 enters dataset construction while the core dataset builder still issues
            # distributed barriers, which deadlocks for TP>1.
            train_valid_test_datasets_provider.is_distributed = True
        elif model_type == "deepseek_v4":
            from pretrain_gpt import (  # type: ignore
                forward_step,
                train_valid_test_datasets_provider,
            )

            log_rank_0("Using DeepSeek-V4 model provider; reusing pretrain_gpt forward_step + datasets")
        else:
            # Use overridable methods so subclasses (e.g. diffusion/Flux) can plug in their own
            # forward_step / dataset_provider. Defaults pull from pretrain_gpt.
            forward_step = self.get_forward_step()
            train_valid_test_datasets_provider = self.get_datasets_provider()

        # Handle Megatron version differences (v0.12.0 vs newer with inprocess_restart)
        wrapped_pretrain = pretrain
        store = None
        try:
            from megatron.training import inprocess_restart  # type: ignore

            if hasattr(inprocess_restart, "maybe_wrap_for_inprocess_restart"):
                wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
        except Exception:
            pass

        sig = inspect.signature(wrapped_pretrain)
        kwargs = {}
        if "args_defaults" in sig.parameters:
            kwargs["args_defaults"] = {"tokenizer_type": "GPT2BPETokenizer"}
        if "extra_args_provider" in sig.parameters:
            kwargs["extra_args_provider"] = None
        if "store" in sig.parameters:
            kwargs["store"] = store

        # Resolve model_provider: prefer subclass-provided attribute (e.g. diffusion/Flux),
        # else fall back to registry-based lookup with model_type for mamba/gpt.
        model_provider = getattr(self, "model_provider", None)
        if model_provider is None:
            if model_type != "gpt":
                model_provider = get_model_provider(model_type=model_type)
            else:
                model_provider = get_model_provider()
        log_rank_0(f"-model_provider: {model_provider}")

        # Patch Megatron's get_forward_backward_func to support dump_pp_data
        import megatron.core.pipeline_parallel as mpp
        import megatron.training.training as mt_training

        orig_get_forward_backward_func = mpp.get_forward_backward_func

        def patched_get_forward_backward_func(*args, **kwargs):
            func = orig_get_forward_backward_func(*args, **kwargs)
            from megatron.training import get_args as get_megatron_args

            try:
                m_args = get_megatron_args()
                if getattr(m_args, "dump_pp_data", False):
                    from primus.backends.megatron.core.pipeline_parallel.pp_visualizer import (
                        schedule_wrapper,
                        set_dump_pp_data_patch,
                    )

                    set_dump_pp_data_patch()
                    return schedule_wrapper(func)
            except Exception as e:
                log_rank_0(f"[Primus] Warning: failed to apply dump_pp_data patch: {e}")
            return func

        mpp.get_forward_backward_func = patched_get_forward_backward_func
        if hasattr(mt_training, "get_forward_backward_func"):
            mt_training.get_forward_backward_func = patched_get_forward_backward_func

        try:
            wrapped_pretrain(
                train_valid_test_datasets_provider,
                model_provider,
                ModelType.encoder_or_decoder,
                forward_step,
                **kwargs,
            )
            log_rank_0("[MegatronPretrainTrainer] pretrain() completed successfully")
        except Exception as e:
            log_rank_0(f"[MegatronPretrainTrainer] ERROR in pretrain(): {type(e).__name__}: {e}")
            import traceback

            log_rank_0(f"[MegatronPretrainTrainer] Traceback: {traceback.format_exc()}")
            raise

        # Dump PP visualization data if enabled
        try:
            from megatron.training import get_args as get_megatron_args

            megatron_args = get_megatron_args()
            if getattr(megatron_args, "dump_pp_data", False):
                import os

                from megatron.core.num_microbatches_calculator import (
                    get_num_microbatches,
                )

                from primus.backends.megatron.core.pipeline_parallel.pp_visualizer import (
                    dump_pp_data,
                )

                pp_data_dir = os.environ.get("DUMP_PP_DIR", "output/pp_data")
                dump_pp_data(megatron_args, get_num_microbatches(), pp_data_dir)
                log_rank_0(f"PP schedule data dumped to {pp_data_dir}")
        except Exception as e:
            log_rank_0(f"Warning: Failed to dump PP data: {e}")

        log_rank_0("Megatron pretrain execution completed.")
