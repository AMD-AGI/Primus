###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronBridgePretrainTrainer: Primus wrapper for Megatron-Bridge pretraining with diffusion.

Uses ``megatron.bridge.training.pretrain`` with a diffusion forward step from
``primus.diffusion`` (FLUX, WAN, etc.).
"""

import traceback
from typing import Any

from primus.backends.megatron_bridge.config_utils import load_recipe_config
from primus.backends.megatron_bridge.diffusion_step_utils import load_diffusion_forward_step
from primus.backends.megatron_bridge.eval_result_logging import (
    install_eval_result_logging_patches,
    uninstall_eval_result_logging_patches,
)
from primus.backends.megatron_bridge.val_target_early_stop import (
    install_validation_target_early_stop,
    uninstall_validation_target_early_stop,
)
from primus.backends.megatron_bridge.megatron_bridge_base_trainer import (
    MegatronBridgeBaseTrainer,
)
from primus.modules.module_utils import log_dict_aligned, log_rank_0


class MegatronBridgePretrainTrainer(MegatronBridgeBaseTrainer):
    """Megatron-Bridge pretrain path for diffusion models (recipe_module + step_func)."""

    TASK_TYPE = "Pre-training (diffusion / FLUX)"

    def __init__(self, backend_args: Any):
        super().__init__(backend_args=backend_args)

    def setup(self):
        log_rank_0("MegatronBridgePretrainTrainer.setup()")

    def init(self):
        log_rank_0("Initializing Megatron-Bridge diffusion pretrain...")
        self.cfg_container = load_recipe_config(self.backend_args)
        log_rank_0("Diffusion pretrain initialization completed")

    def train(self):
        log_rank_0("Executing Megatron-Bridge diffusion pretrain...")
        try:
            from megatron.bridge.training.pretrain import pretrain

            forward_step = load_diffusion_forward_step(self.backend_args)
            log_dict_aligned("ConfigContainer", self.cfg_container.to_dict())
            _tc = self.cfg_container.train
            log_rank_0(
                "Megatron-Bridge validation: eval_iters=%r eval_interval=%r train_iters=%r "
                "(in-loop eval when do_valid and eval_interval is set and step %% eval_interval == 0; "
                "do_valid needs eval_iters > 0 and a built val dataloader)."
                % (_tc.eval_iters, _tc.eval_interval, _tc.train_iters)
            )
            install_validation_target_early_stop(self.cfg_container)
            install_eval_result_logging_patches()
            try:
                pretrain(self.cfg_container, forward_step_func=forward_step)
            finally:
                uninstall_eval_result_logging_patches()
                uninstall_validation_target_early_stop()
        except Exception as e:
            detail = str(e) or repr(e) or type(e).__name__
            log_rank_0(f"Error during diffusion pretrain ({type(e).__name__}): {detail}")
            log_rank_0(traceback.format_exc())
            raise

        log_rank_0("Megatron-Bridge diffusion pretrain completed.")
