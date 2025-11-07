###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.modules.base_module import BaseModule
from primus.modules.module_utils import error_rank_0, log_rank_0, warning_rank_0


class MaxTextPretrainTrainer(BaseModule):
    def __init__(self, *args, **kwargs):
        kwargs.pop("extra_args", None)
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

        self.pre_trainer_cfg_path = self.primus_cfg.module_config_path("pre_trainer")

    def setup(self):
        log_rank_0(f"setup MaxText")

    def init(self, *init_args, **kwargs):
        argv = ["MaxText.train", self.pre_trainer_cfg_path]
        log_rank_0(f"init MaxText with argv {argv}")

        from primus.backends.maxtext.train import initialize

        self.train_config, self.recorder, self.diagnostic_config = initialize(argv)

    def run(self, *args, **kwargs):
        log_rank_0(f"run MaxText")

        from primus.backends.maxtext.train import run

        run(self.train_config, self.recorder, self.diagnostic_config)

    def patch_maxtext_logger(self):
        from primus.core.utils.logger import _logger as primus_logger

        try:
            import MaxText.max_logging as maxtext_logging

            if hasattr(maxtext_logging, "log"):
                maxtext_logging.log = primus_logger.info
                log_rank_0("MaxText logger patched successfully.")
            else:
                warning_rank_0("MaxText logging module does not have a 'log' function.")
        except ImportError:
            error_rank_0("Failed to import MaxText's logging module.")

    def patch_max_utils(self):
        import MaxText.max_utils as orig_max_utils

        from primus.backends.maxtext.max_utils import (
            print_system_information,
            save_device_information,
        )

        orig_max_utils.print_system_information = print_system_information
        orig_max_utils.save_device_information = save_device_information
        log_rank_0("MaxText max_utils patched successfully.")

    def patch_checkpoint(self):
        import MaxText.checkpointing as orig_checkpointing

        from primus.backends.maxtext.checkpointing import (
            create_orbax_checkpoint_manager,
        )

        orig_checkpointing.create_orbax_checkpoint_manager = create_orbax_checkpoint_manager
        log_rank_0("MaxText checkpointing patched successfully.")

    def patch_input_pipeline(self):
        import MaxText.input_pipeline._hf_data_processing as orig_hf_data_processing

        from primus.backends.maxtext.input_pipeline._hf_data_processing import (
            make_hf_eval_iterator,
            make_hf_train_iterator,
            preprocessing_pipeline,
        )

        orig_hf_data_processing.preprocessing_pipeline = preprocessing_pipeline
        orig_hf_data_processing.make_hf_train_iterator = make_hf_train_iterator
        orig_hf_data_processing.make_hf_eval_iterator = make_hf_eval_iterator
        log_rank_0("MaxText _hf_data_processing patched successfully.")

    def patch_layers(self):
        import MaxText.layers.quantizations as orig_quantizations

        from primus.backends.maxtext.layers.quantizations import (
            PrimusNANOOFp8Quantization,
        )

        orig_quantizations.NANOOFp8Quantization = PrimusNANOOFp8Quantization
        log_rank_0("MaxText NANOOFp8Quantization patched successfully.")

        import MaxText.layers.attention_op as orig_attention_op

        from primus.backends.maxtext.layers.attention_op import PrimusAttentionOp

        orig_attention_op.AttentionOp = PrimusAttentionOp
        log_rank_0("MaxText AttentionOp patched successfully.")

        import MaxText.layers.attentions as orig_attentions

        from primus.backends.maxtext.layers.attentions import PrimusAttention

        orig_attentions.Attention = PrimusAttention
        log_rank_0("MaxText Attention patched successfully.")

        import MaxText.layers.moe as orig_moe

        from primus.backends.maxtext.layers.moe import PrimusRoutedMoE

        orig_moe.RoutedMoE = PrimusRoutedMoE
        log_rank_0("MaxText RoutedMoE patched successfully.")
