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

        self.primus_cfg = kwargs.pop("primus_config", None)

        if self.primus_cfg is None:
            raise ValueError("primus_config is required")
        self.primus_cfg.export_module_config("pre_trainer")
        self.pre_trainer_cfg_path = self.primus_cfg.module_config_path("pre_trainer")
        self.override_model_args = self.prepare_model_overrides(extra_args)

    def setup(self):
        log_rank_0(f"setup MaxText")

    def init(self, *init_args, **kwargs):
        import functools

        from MaxText import pyconfig
        from MaxText.train import initialize

        argv = ["MaxText.train", self.pre_trainer_cfg_path]
        log_rank_0(f"init MaxText with argv {argv}")

        if self.override_model_args:
            _orig = pyconfig.initialize
            pyconfig.initialize = functools.partial(_orig, **self.override_model_args)
            try:
                self.train_config, self.recorder, self.diagnostic_config = initialize(argv)
            finally:
                pyconfig.initialize = _orig
        else:
            self.train_config, self.recorder, self.diagnostic_config = initialize(argv)

    def run(self, *args, **kwargs):
        log_rank_0(f"MaxText Pre-Trainer: begin training...")

        from MaxText.train import run

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
