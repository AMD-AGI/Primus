###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.megatron.megatron_base_trainer import MegatronBaseTrainer
from primus.core.utils.module_utils import log_rank_0


class MegatronPretrainTrainer(MegatronBaseTrainer):
    """Trainer for Megatron-LM pre-training."""

    def build_model_for_benchmark(self):
        """Initialize Megatron and build the model WITHOUT running the training loop.

        This mirrors the front of ``megatron.training.pretrain`` (``initialize_megatron``
        followed by ``setup_model_and_optimizer``) so that Primus performance
        projection's layer benchmark can construct a model identical to the real
        training path and measure per-layer kernels, without datasets or a train
        loop.

        Prerequisites (handled by the runtime before calling this): ``setup()``
        has patched ``parse_args`` to return ``self.backend_args`` and set the
        Primus global vars, and the build_args/setup/before_train patch phases
        have been applied. Returns the built model (a list of model chunks, as
        produced by megatron's ``get_model``) and also stores it on ``self.model``.
        """
        log_rank_0("Building Megatron model for benchmark (no training loop)...")

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

        log_rank_0("Megatron model build for benchmark completed.")
        return model

    def train(self):
        """Execute Megatron pre-training."""
        log_rank_0("Executing Megatron pretrain...")

        import inspect

        from megatron.core.enums import ModelType
        from megatron.training import pretrain  # type: ignore

        from primus.core.utils.import_utils import get_model_provider

        # Determine model type (gpt or mamba) from backend_args
        model_type = getattr(self.backend_args, "model_type", "gpt")
        log_rank_0(f"-detected model_type: {model_type}")

        # Import the appropriate training components based on model_type
        if model_type == "mamba":
            from pretrain_mamba import (  # type: ignore
                forward_step,
                train_valid_test_datasets_provider,
            )

            log_rank_0("Using Mamba model provider and training components")
        else:
            from pretrain_gpt import (  # type: ignore
                forward_step,
                train_valid_test_datasets_provider,
            )

            log_rank_0("Using GPT model provider and training components")

        # Configure training components
        train_valid_test_datasets_provider.is_distributed = True

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

        # Get model provider with correct model_type
        # Only pass model_type if it's not the default to maintain compatibility
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

        wrapped_pretrain(
            train_valid_test_datasets_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            **kwargs,
        )

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
