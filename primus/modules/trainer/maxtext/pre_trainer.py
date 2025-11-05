###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.modules.base_module import BaseModule
from primus.modules.module_utils import log_rank_0


class MaxTextPretrainTrainer(BaseModule):
    def __init__(self, *args, **kwargs):
        extra_args = kwargs.pop("extra_args", None)
        super().__init__(*args, **kwargs)

        # important: make sure patch maxtext logger first
        self.patch_maxtext_logger()

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

        from MaxText.train import initialize
        self.train_config, self.recorder, self.diagnostic_config = initialize(argv)

    def run(self, *args, **kwargs):
        log_rank_0(f"run MaxText")
       
        from MaxText.train import run
        run(self.train_config, self.recorder, self.diagnostic_config)

    def patch_maxtext_logger(self):
        from primus.core.utils.logger import _logger as primus_logger
        try:
            import MaxText.max_logging as maxtext_logging
        
            if hasattr(maxtext_logging, 'log'):
                maxtext_logging.log = primus_logger.info
                primus_logger.info("MaxText logger patched successfully.")
            else:
                primus_logger.warning("MaxText logging module does not have a 'log' function.")
        except ImportError:
            primus_logger.error("Failed to import MaxText's logging module.")
