###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

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
        self.patch_layers()

        self.primus_cfg = kwargs.pop("primus_config", None)

        if self.primus_cfg is None:
            raise ValueError("primus_config is required")
        self.primus_cfg.export_module_config("pre_trainer")
        self.pre_trainer_cfg_path = self.primus_cfg.module_config_path("pre_trainer")

        self.override_model_args = self.patch_model_args(extra_args)

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
        log_rank_0(f"run MaxText...")

        from primus.backends.maxtext.train import run

        run(self.train_config, self.recorder, self.diagnostic_config)

    def patch_model_args(self, override_args: Dict[str, Any]):
        """
        Monkey patch maxtext cli args to override model args dynamically.
        Supports nested overrides like:
            {"model": {"num_experts": 16, "base_num_decoder_layers": 4}}

        All override keys MUST be under the "model" key.
        """

        if not override_args:
            warning_rank_0("MaxText Pre-Trainer: No override_args provided, skip patch.")
            return {}

        warning_rank_0(f"MaxText Pre-Trainer: Applying override_args: {override_args}")

        # --- Step 1. Flatten any nested dict under 'model'
        flat_overrides = {}
        for k, v in override_args.items():
            if k != "model" or not isinstance(v, dict):
                raise ValueError(
                    f"MaxText Pre-Trainer: Invalid override keys detected: {k}. "
                    "These parameters belong to the model configuration and must be specified "
                    "under the 'model' key"
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
        import MaxText.max_utils as orig_max_utils

        from primus.backends.maxtext.max_utils import (
            print_system_information,
            save_device_information,
        )

        orig_max_utils.print_system_information = print_system_information
        orig_max_utils.save_device_information = save_device_information
        warning_rank_0("MaxText Pre-Trainer: patch max_utils successfully.")

    def patch_checkpoint(self):
        import MaxText.checkpointing as orig_checkpointing

        from primus.backends.maxtext.checkpointing import (
            create_orbax_checkpoint_manager,
        )

        orig_checkpointing.create_orbax_checkpoint_manager = create_orbax_checkpoint_manager
        warning_rank_0("MaxText Pre-Trainer: patch checkpointing successfully.")

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
            import MaxText.layers.attention_mla as orig_attention_mla
            import MaxText.layers.attention_op as orig_attention_op
            import MaxText.layers.attentions as orig_attentions

            from primus.backends.maxtext.layers.attention_op import PrimusAttentionOp
            from primus.backends.maxtext.layers.attentions import PrimusAttention

            orig_attention_op.AttentionOp = PrimusAttentionOp
            orig_attentions.AttentionOp = PrimusAttentionOp

            orig_attentions.Attention = PrimusAttention
            orig_attention_mla.Attention = PrimusAttention
            warning_rank_0("MaxText Pre-Trainer: patch Attention successfully.")

        patch_attn()

        def patch_moe():
            import MaxText.layers.moe as orig_moe

            from primus.backends.maxtext.layers.moe import PrimusRoutedMoE

            orig_moe.RoutedMoE = PrimusRoutedMoE
            warning_rank_0("MaxText Pre-Trainer: patch RoutedMoE successfully.")

        patch_moe()
