###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.megatron.training.global_vars import set_primus_global_variables
from primus.core.trainer.base_trainer import BaseTrainer
from primus.modules.module_utils import log_rank_0


class MegatronBaseTrainer(BaseTrainer):
    """Base trainer for Megatron-LM, handles parse_args patching."""

    def setup(self):
        """Setup Megatron runtime: set global vars and patch parse_args."""
        set_primus_global_variables(self.backend_args)
        self._patch_parse_args()

    def init(self):
        """Initialize Megatron training components."""
        log_rank_0("Initializing Megatron training...")
        # log_dict_aligned("Backend arguments", self.backend_args)

    def _patch_parse_args(self):
        """
        This function patches Megatron's parse_args to return pre-configured Primus arguments.
        It also validates the arguments on ROCM.
        """
        import megatron.training.arguments as megatron_args  # type: ignore
        import megatron.training.initialize as megatron_init  # type: ignore

        from primus.modules.trainer.megatron.utils import validate_args_on_rocm

        log_rank_0("Patching Megatron-LM parse_args()")

        patched_parse_args = lambda *args, **kwargs: (
            log_rank_0("parse_args() called; returning Primus arguments") or self.backend_args
        )

        original_validate_args = megatron_args.validate_args

        def patched_validate_args(*args, **kwargs):
            validated_args = original_validate_args(*args, **kwargs)
            parsed_args = args[0] if args else kwargs.get("args", None)
            if parsed_args is not None:
                log_rank_0("validate_args() called; validating on ROCM")
                validate_args_on_rocm(parsed_args)
            return validated_args

        megatron_args.parse_args = patched_parse_args
        megatron_init.parse_args = patched_parse_args

        megatron_args.validate_args = patched_validate_args
        megatron_init.validate_args = patched_validate_args

        log_rank_0(
            f"Patched parse_args()/validate_args(); Primus provided {len(vars(self.backend_args))} arguments"
        )
