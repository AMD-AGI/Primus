###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronBaseTrainer: Base class for all Megatron-LM trainers.

This base class handles Megatron-specific initialization logic that is
common across all Megatron training tasks (pretrain, sft, posttrain, etc.).

Inherits from BaseTrainer which provides:
    - Universal training workflow (run template method)
    - Universal patch management (via run_patches)
    - Consistent training lifecycle

This class implements:
    - Megatron runtime initialization (parse_args patching, etc.)
    - ROCm compatibility patches
"""

from typing import Any

from primus.backends.megatron.training.global_vars import set_primus_global_variables
from primus.core.trainer.base_trainer import BaseTrainer
from primus.modules.module_utils import log_rank_0


class MegatronBaseTrainer(BaseTrainer):
    """
    Base trainer class for all Megatron-LM training tasks.

    This class handles Megatron-specific concerns:
        - Argument injection into Megatron's runtime (parse_args patching)
        - ROCm compatibility patches
        - Megatron version detection

    All Megatron trainers (pretrain, sft, posttrain) should inherit from this class.

    Note:
        Patch management is handled by BaseTrainer via run_patches().
        This class only handles Megatron-specific initialization.
    """

    def __init__(self, backend_args: Any):
        """
        Initialize Megatron base trainer.

        Args:
            backend_args: Megatron-LM argument namespace (from MegatronArgBuilder)
        """
        log_rank_0("=" * 80)
        log_rank_0("Initializing MegatronBaseTrainer...")
        log_rank_0("=" * 80)

        # Initialize BaseTrainer
        super().__init__(backend_args=backend_args)
        set_primus_global_variables(self.backend_args)

        # Inject arguments into Megatron runtime by patching parse_args()
        self._patch_parse_args()

        log_rank_0("=" * 80)
        log_rank_0("MegatronBaseTrainer initialized successfully")
        log_rank_0("=" * 80)

    def _patch_parse_args(self):
        """
        Patch Megatron-LM's parse_args to return pre-configured Primus arguments.

        This ensures Megatron-LM uses the pre-configured Primus arguments.
        """
        import megatron.training.arguments as megatron_args  # type: ignore
        import megatron.training.initialize as megatron_init  # type: ignore

        log_rank_0("Patching Megatron-LM parse_args() to use pre-configured Primus arguments")

        # Create a lambda that always returns our prepared args
        patched_parse_args = lambda *args, **kwargs: (
            log_rank_0("parse_args() called; returning pre-configured Primus arguments") or self.backend_args
        )

        # Patch both locations where parse_args might be defined/called
        megatron_args.parse_args = patched_parse_args
        megatron_init.parse_args = patched_parse_args

        log_rank_0(
            f"Patched Megatron-LM parse_args(); Primus provided {len(vars(self.backend_args))} arguments"
        )
