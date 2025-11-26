###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.utils.import_utils import get_model_provider
from primus.modules.base_module import BaseModule
from primus.modules.module_utils import log_rank_0


class LightMegatronPretrainTrainer(BaseModule):

    def __init__(self, *args, **kwargs):
        extra_args = kwargs.pop("extra_args", None)
        if extra_args:
            raise ValueError(
                f"[LightMegatronPretrainTrainer] Unexpected extra_args detected: {extra_args}. "
                f"LightMegatron backend does not support unregistered config keys."
            )
        kwargs["module_name"] = "pre_trainer"
        super().__init__(*args, **kwargs)

    def setup(self):
        log_rank_0(f"setup light-megatron")

    def init(self, *init_args, **kwargs):
        log_rank_0("init light-megatron")
        from primus.modules.trainer.lightmegatron.launcher_adapter import (
            MegatronLauncherAdapter,
        )

        adapter = MegatronLauncherAdapter(self.module_config, self.exp_root_path, self.exp_meta_info)
        adapter.apply_all()

    def run(self, *args, **kwargs):
        log_rank_0("run light-megatron")

        from megatron.core.enums import ModelType
        from megatron.training import inprocess_restart, pretrain
        from pretrain_gpt import forward_step, train_valid_test_datasets_provider

        train_valid_test_datasets_provider.is_distributed = True
        wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

        wrapped_pretrain(
            train_valid_test_datasets_provider,
            get_model_provider(),
            ModelType.encoder_or_decoder,
            forward_step,
            store=store,
        )
