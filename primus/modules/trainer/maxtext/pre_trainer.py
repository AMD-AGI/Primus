###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.modules.base_module import BaseModule
from primus.modules.module_utils import log_rank_0
from primus.core.utils.logger import _logger as primus_logger


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

        self.pre_trainer_cfg_path = self.primus_cfg.module_config_path("pre_trainer")

    def setup(self):
        log_rank_0(f"setup MaxText")
        pass

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
        try:
            import MaxText.max_logging as maxtext_logging
        
            if hasattr(maxtext_logging, 'log'):
                maxtext_logging.log = primus_logger.info
                primus_logger.info("MaxText logger patched successfully.")
            else:
                primus_logger.warning("MaxText logging module does not have a 'log' function.")
        except ImportError:
            primus_logger.error("Failed to import MaxText's logging module.")

    def patch_max_utils(self):
        from primus.backends.maxtext.max_utils import print_system_information, save_device_information
        try:
            import MaxText.max_utils as orig_max_utils

            orig_max_utils.print_system_information = print_system_information
            orig_max_utils.save_device_information = save_device_information
            primus_logger.info("MaxText max_utils patched successfully.")
        except Exception as e:
            primus_logger.error("Failed to patch MaxText's max_utils because of {e}.")

    def patch_checkpoint(self):
        from primus.backends.maxtext.checkpointing import create_orbax_checkpoint_manager
        try:
            import MaxText.checkpointing as orig_checkpointing

            orig_checkpointing.create_orbax_checkpoint_manager = create_orbax_checkpoint_manager
            primus_logger.info("MaxText checkpointing patched successfully.")
        except Exception as e:
            primus_logger.error("Failed to patch MaxText's checkpointing because of {e}.")

    def patch_input_pipeline(self):
        from primus.backends.maxtext.input_pipeline._hf_data_processing import (
            preprocessing_pipeline,
            make_hf_train_iterator,
            make_hf_eval_iterator,
        )

        try:
            import MaxText.input_pipeline._hf_data_processing as orig_hf_data_processing

            orig_hf_data_processing.preprocessing_pipeline = preprocessing_pipeline
            orig_hf_data_processing.make_hf_train_iterator = make_hf_train_iterator
            orig_hf_data_processing.make_hf_eval_iterator = make_hf_eval_iterator
            primus_logger.info("MaxText _hf_data_processing patched successfully.")
        except Exception as e:
            primus_logger.error("Failed to patch MaxText's _hf_data_processing because of {e}.")

    def patch_layers(self):
        from primus.backends.maxtext.layers.quantizations import PrimusNANOOFp8Quantization
        try:
            import MaxText.layers.quantizations as orig_quantizations
            orig_quantizations.NANOOFp8Quantization = PrimusNANOOFp8Quantization
            primus_logger.info("MaxText NANOOFp8Quantization patched successfully.")
        except Exception as e:
            primus_logger.error("Failed to patch MaxText's NANOOFp8Quantization because of {e}.")

        from primus.backends.maxtext.layers.attention_op import PrimusAttentionOp
        try:
            import MaxText.layers.attention_op as orig_attention_op
            orig_attention_op.AttentionOp = PrimusAttentionOp
            primus_logger.info("MaxText AttentionOp patched successfully.")
        except Exception as e:
            primus_logger.error("Failed to patch MaxText's AttentionOp because of {e}.")

        from primus.backends.maxtext.layers.attentions import PrimusAttention
        try:
            import MaxText.layers.attentions as orig_attentions
            orig_attentions.Attention = PrimusAttention
            primus_logger.info("MaxText Attention patched successfully.")
        except Exception as e:
            primus_logger.error("Failed to patch MaxText's Attention because of {e}.")
        
        from primus.backends.maxtext.layers.moe import PrimusRoutedMoE
        try:
            import MaxText.layers.moe as orig_moe
            orig_moe.RoutedMoE = PrimusRoutedMoE
            primus_logger.info("MaxText RoutedMoE patched successfully.")
        except Exception as e:
            primus_logger.error("Failed to patch MaxText's RoutedMoE because of {e}.")
