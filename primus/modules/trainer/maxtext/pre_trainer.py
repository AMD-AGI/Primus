###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from typing import Any, Dict

from primus.core.utils import checker
from primus.modules.base_module import BaseModule
from primus.modules.module_utils import error_rank_0, log_rank_0, warning_rank_0


class MaxTextPretrainTrainer(BaseModule):
    def __init__(self, *args, **kwargs):
        extra_args = kwargs.pop("extra_args", None)
        super().__init__(*args, **kwargs)

        # important: make sure patch maxtext logger first
        self.patch_maxtext_logger()
        self.patch_max_utils()
        self.patch_checkpoint()
        self.patch_input_pipeline()
        self.patch_config_types()
        self.patch_layers()

        self.primus_cfg = kwargs.pop("primus_config", None)

        if self.primus_cfg is None:
            raise ValueError("primus_config is required")
        self.primus_cfg.export_module_config("pre_trainer")
        self.pre_trainer_cfg_path = self.primus_cfg.module_config_path("pre_trainer")

        self.patch_wandb()
        self.override_model_args = self.prepare_model_overrides(extra_args)

    def setup(self):
        log_rank_0(f"setup MaxText")

    def init(self, *init_args, **kwargs):
        argv = ["MaxText.train", self.pre_trainer_cfg_path]
        log_rank_0(f"init MaxText with argv {argv}")

        from primus.backends.maxtext.train import initialize

        self.train_config, self.recorder, self.diagnostic_config = initialize(
            argv, **self.override_model_args
        )

    def run(self, *args, **kwargs):
        log_rank_0(f"MaxText Pre-Trainer: begin training...")

        from primus.backends.maxtext.train import run

        run(self.train_config, self.recorder, self.diagnostic_config)
        log_rank_0("MaxText Pre-Trainer: after training is done")

    def prepare_model_overrides(self, override_args: Dict[str, Any]):
        """
        Monkey patch maxtext cli args to override model args dynamically.
        Supports nested overrides like:
            {"override_model": {"num_experts": 16, "base_num_decoder_layers": 4}}

        All override keys MUST be under the "model" key.
        """

        if not override_args:
            warning_rank_0("MaxText Pre-Trainer: No override_args provided, skip patch.")
            return {}

        warning_rank_0(f"MaxText Pre-Trainer: Applying override_args: {override_args}")

        # --- Step 1. Flatten any nested dict under 'override_model'
        flat_overrides = {}
        for k, v in override_args.items():
            if k != "override_model":
                raise ValueError(f"Only the 'override_model' key is supported for overrides, found: {k}")
            if not isinstance(v, dict):
                raise ValueError(
                    f"MaxText Pre-Trainer: The value for 'override_model' must be a dict, got {type(v).__name__}."
                )
            for subk, subv in v.items():
                if isinstance(subv, dict):
                    raise ValueError(
                        f"MaxText Pre-Trainer: Invalid override key-value detected: {k}.{subk}-{subv}"
                    )
                flat_overrides[subk] = subv
        return flat_overrides

    def patch_maxtext_logger(self):
        import logging

        from primus.core.utils.logger import _logger as primus_logger

        try:
            import MaxText.max_logging as maxtext_logging

            if hasattr(maxtext_logging, "log"):
                maxtext_logging.log = primus_logger.info
                warning_rank_0("MaxText Pre-Trainer: patch logger successfully.")
            else:
                error_rank_0("MaxText Pre-Trainer: logging module does not have a 'log' function.")
        except ImportError:
            error_rank_0("MaxText Pre-Trainer: failed to import MaxText Pre-Trainer's logging module.")

        level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

        stderr_sink_level = self.module_config.stderr_sink_level
        checker.check_true(stderr_sink_level in level_map)
        logging_level = level_map[stderr_sink_level]

        jax_loggers = [logging.getLogger("jax"), logging.getLogger("jaxlib")]
        for jax_logger in jax_loggers:
            jax_logger.setLevel(logging_level)

        warning_rank_0(
            f"jax.logging_level is deprecated, set logging_level={logging_level} [stderr_sink_level]"
        )

    def patch_max_utils(self):
        import functools

        import jax
        import MaxText.max_utils as orig_max_utils

        from primus.backends.maxtext.max_utils import (
            print_system_information,
            save_device_information,
        )

        # Wrap jax.distributed.initialize to inject heartbeat_timeout_seconds
        heartbeat_timeout = getattr(self.module_config, "jax_distributed_heartbeat_timeout_seconds", 100)
        _orig_jax_init = jax.distributed.initialize

        @functools.wraps(_orig_jax_init)
        def _jax_init_with_heartbeat(*args, **kwargs):
            kwargs.setdefault("heartbeat_timeout_seconds", heartbeat_timeout)
            return _orig_jax_init(*args, **kwargs)

        jax.distributed.initialize = _jax_init_with_heartbeat

        orig_max_utils.print_system_information = print_system_information
        orig_max_utils.save_device_information = save_device_information
        warning_rank_0("MaxText Pre-Trainer: patch max_utils successfully.")

    def patch_checkpoint(self):
        import MaxText.checkpointing as orig_checkpointing
        import MaxText.train_utils as orig_train_utils

        from primus.backends.maxtext.checkpointing import load_state_if_possible
        from primus.backends.maxtext.patches.checkpoint_patches import (
            _PRIMUS_DEFAULT_MAX_TO_KEEP,
            _make_opts_wrapper,
            _wrap_create_training_tools,
        )

        orig_opts = orig_checkpointing.CheckpointManagerOptions
        orig_checkpointing.CheckpointManagerOptions = _make_opts_wrapper(
            orig_opts, _PRIMUS_DEFAULT_MAX_TO_KEEP
        )

        orig_checkpointing.load_state_if_possible = load_state_if_possible
        orig_train_utils.create_training_tools = _wrap_create_training_tools(
            orig_train_utils.create_training_tools, orig_checkpointing, orig_opts
        )
        warning_rank_0("MaxText Pre-Trainer: patch checkpointing successfully.")

    def patch_wandb(self):
        def set_default_wandb_project(exp_meta_info):
            work_group = exp_meta_info["work_group"]
            user_name = exp_meta_info["user_name"]
            os.environ["WANDB_PROJECT"] = f"Primus-MaxText-Pretrain-{work_group}_{user_name}"

        set_default_wandb_project(self.primus_cfg.exp_meta_info)

        import MaxText.metric_logger as orig_metric_logger
        import MaxText.train as orig_train

        import primus.backends.maxtext.train as primus_train
        from primus.backends.maxtext.metric_logger import PrimusMetricLogger

        orig_metric_logger.MetricLogger = PrimusMetricLogger
        orig_train.MetricLogger = PrimusMetricLogger
        primus_train.MetricLogger = PrimusMetricLogger
        warning_rank_0("MaxText Pre-Trainer: patch wandb successfully.")

    def patch_input_pipeline(self):
        import MaxText.input_pipeline._hf_data_processing as orig_hf_data_processing
        import MaxText.input_pipeline.input_pipeline_interface as orig_input_pipeline_interface

        from primus.backends.maxtext.input_pipeline._hf_data_processing import (
            make_hf_eval_iterator,
            make_hf_train_iterator,
            preprocessing_pipeline,
        )

        orig_hf_data_processing.preprocessing_pipeline = preprocessing_pipeline
        orig_hf_data_processing.make_hf_train_iterator = make_hf_train_iterator
        orig_hf_data_processing.make_hf_eval_iterator = make_hf_eval_iterator

        orig_input_pipeline_interface.make_hf_train_iterator = make_hf_train_iterator
        orig_input_pipeline_interface.make_hf_eval_iterator = make_hf_eval_iterator

        warning_rank_0("MaxText Pre-Trainer: patch _hf_data_processing successfully.")

    def patch_config_types(self):
        import MaxText.configs.types as orig_config_types

        from primus.backends.maxtext.configs.types import PrimusMaxTextConfig

        orig_config_types.MaxTextConfig = PrimusMaxTextConfig
        warning_rank_0("MaxText Pre-Trainer: patch config types successfully.")

    def patch_layers(self):
        def patch_quantization():
            import MaxText.layers.quantizations as orig_quantizations

            from primus.backends.maxtext.layers.quantizations import (
                PrimusNANOOFp8Quantization,
            )

            orig_quantizations.NANOOFp8Quantization = PrimusNANOOFp8Quantization
            warning_rank_0("MaxText Pre-Trainer: patch NANOOFp8Quantization successfully.")

        patch_quantization()

        def patch_attn():
            import MaxText.layers.attention_op as orig_attention_op
            import MaxText.layers.attentions as orig_attentions

            from primus.backends.maxtext.layers.attention_op import PrimusAttentionOp

            orig_attention_op.AttentionOp = PrimusAttentionOp
            orig_attentions.AttentionOp = PrimusAttentionOp

            warning_rank_0("MaxText Pre-Trainer: patch Attention successfully.")

        patch_attn()

        def patch_moe():
            import MaxText.layers.moe as orig_moe

            from primus.backends.maxtext.layers.moe import PrimusRoutedMoE

            orig_moe.RoutedMoE = PrimusRoutedMoE
            warning_rank_0("MaxText Pre-Trainer: patch RoutedMoE successfully.")

        patch_moe()

        def patch_decoder_layer():
            import functools

            from MaxText.layers.gemma import GemmaDecoderLayer
            from MaxText.layers.gemma2 import Gemma2DecoderLayer
            from MaxText.layers.llama2 import LlamaDecoderLayer
            from MaxText.layers.mistral import MistralDecoderLayer
            from MaxText.layers.mixtral import MixtralDecoderLayer

            def _patch_init(cls, attention_attrs):
                _orig = cls.__init__

                @functools.wraps(_orig)
                def _new_init(self, *args, **kwargs):
                    _orig(self, *args, **kwargs)
                    scalar = self.config.head_dim**-0.5
                    for name in attention_attrs:
                        getattr(self, name).query_pre_attn_scalar = scalar

                cls.__init__ = _new_init

            _patch_init(GemmaDecoderLayer, ["self_attention"])
            _patch_init(Gemma2DecoderLayer, ["self_attention_local", "self_attention_global"])
            _patch_init(LlamaDecoderLayer, ["self_attention"])
            _patch_init(MistralDecoderLayer, ["self_attention"])
            _patch_init(MixtralDecoderLayer, ["self_attention"])
            warning_rank_0("MaxText Pre-Trainer: patch decoder layer successfully.")

        patch_decoder_layer()
