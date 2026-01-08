###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
MaxTextPretrainTrainer: Primus wrapper for MaxText pre-training.

This trainer bridges Primus's configuration system with MaxText's training
loop, following the same pattern as ``TorchTitanPretrainTrainer`` does for
TorchTitan.

The trainer inherits from ``MaxTextBaseTrainer`` which handles:
    - Integration with the unified BaseTrainer workflow (run_patches)
    - Version detection and common logging

This class only needs to implement:
    - setup(): optional pre-initialization
    - init(): construct the underlying MaxText training components
    - run_train(): call into MaxText's training loop
"""

from typing import Any, Dict, Optional

from primus.backends.maxtext.maxtext_base_trainer import MaxTextBaseTrainer
from primus.modules.module_utils import log_rank_0, warning_rank_0


class MaxTextPretrainTrainer(MaxTextBaseTrainer):
    """
    Trainer class for MaxText pre-training.
    """

    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any):
        """
        Initialize MaxText pretrain trainer.

        Args:
            primus_config: Full Primus configuration
            module_config: Module-specific configuration
            backend_args: MaxText configuration (from MaxTextAdapter)
        """
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )

        # Store config path for MaxText
        self.primus_cfg = primus_config
        self.primus_cfg.export_module_config("pre_trainer")
        self.pre_trainer_cfg_path = self.primus_cfg.module_config_path("pre_trainer")

        # Training state
        self.train_config: Optional[Any] = None
        self.recorder: Optional[Any] = None
        self.diagnostic_config: Optional[Any] = None

        log_rank_0(f"Initialized MaxTextPretrainTrainer for model: {module_config.model or 'custom'}")

    # --------------------------------------------------------------------- #
    # Lifecycle hooks
    # --------------------------------------------------------------------- #

    def setup(self):
        """
        Optional setup phase (kept for API symmetry with other trainers).
        """
        log_rank_0("MaxTextPretrainTrainer.setup()")

    def init(self):
        """
        Construct the underlying MaxText training components.
        """
        log_rank_0("MaxTextPretrainTrainer.init() - initializing MaxText training")

        from primus.backends.maxtext.train import initialize

        # Prepare model overrides from extra_args if present
        override_model_args = self.prepare_model_overrides()

        # Build argv for MaxText initialization
        argv = ["MaxText.train", self.pre_trainer_cfg_path]
        log_rank_0(f"Initializing MaxText with argv: {argv}")

        # Initialize MaxText training components
        self.train_config, self.recorder, self.diagnostic_config = initialize(argv, **override_model_args)

        log_rank_0("MaxText training components initialized successfully")

    def prepare_model_overrides(self) -> Dict[str, Any]:
        """
        Prepare model overrides from extra_args.

        Supports nested overrides like:
            {"model": {"num_experts": 16, "base_num_decoder_layers": 4}}

        All override keys MUST be under the "model" key.

        Returns:
            Dictionary of flat overrides for MaxText
        """
        # Get extra_args from module_config if available
        override_args = getattr(self.module_config, "extra_args", None)

        if not override_args:
            return {}

        warning_rank_0(f"MaxText Pre-Trainer: Applying override_args: {override_args}")

        # Flatten any nested dict under 'model'
        flat_overrides = {}
        for k, v in override_args.items():
            if k != "model":
                raise ValueError(f"Only the 'model' key is supported for overrides, found: {k}")
            if not isinstance(v, dict):
                raise ValueError(
                    f"MaxText Pre-Trainer: The value for 'model' must be a dict, got {type(v).__name__}."
                )
            for subk, subv in v.items():
                if isinstance(subv, dict):
                    raise ValueError(
                        f"MaxText Pre-Trainer: Invalid override key-value detected: {k}.{subk}-{subv}"
                    )
                flat_overrides[subk] = subv

        return flat_overrides

    # --------------------------------------------------------------------- #
    # Training entrypoint
    # --------------------------------------------------------------------- #

    def run_train(self):
        """
        Execute MaxText pre-training.

        This method is called by BaseTrainer.run() after applying patches.
        """
        if self.train_config is None:
            raise RuntimeError("MaxTextPretrainTrainer.init() must be called before run_train().")

        log_rank_0("Executing MaxText pretrain...")

        from primus.backends.maxtext.train import run

        run(self.train_config, self.recorder, self.diagnostic_config)

        log_rank_0("MaxText pretrain execution completed.")
