###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

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

    def cleanup(self, on_error: bool = False):
        """Megatron cleanup: optional fast exit.

        * ``PRIMUS_EXIT_FAST=1`` — after ``super().cleanup()``, skip Python
          shutdown and call ``os._exit(0)`` directly. Saves ~20 s of Python
          interpreter shutdown / torchrun reaper lag at the cost of any
          ``atexit`` handlers below us. OFF by default.

        When both are off this method just delegates to ``super().cleanup()``,
        identical to the previous behavior.
        """
        super().cleanup(on_error=on_error)

        exit_fast = os.environ.get("PRIMUS_EXIT_FAST", "0") == "1"

        if exit_fast and not on_error:
            log_rank_0("[MegatronBaseTrainer] PRIMUS_EXIT_FAST=1 -> os._exit(0)")
            # Flush stdout/stderr so the final log lines are not lost.
            try:
                import sys

                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:  # pragma: no cover
                pass
            os._exit(0)

    def _patch_parse_args(self):
        """Patch Megatron's parse_args to return pre-configured Primus arguments."""
        import megatron.training.arguments as megatron_args  # type: ignore
        import megatron.training.initialize as megatron_init  # type: ignore

        log_rank_0("Patching Megatron-LM parse_args()")

        patched_parse_args = lambda *args, **kwargs: (
            log_rank_0("parse_args() called; returning Primus arguments") or self.backend_args
        )

        megatron_args.parse_args = patched_parse_args
        megatron_init.parse_args = patched_parse_args

        log_rank_0(f"Patched parse_args(); Primus provided {len(vars(self.backend_args))} arguments")
