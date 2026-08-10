###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
NemoAutomodelPretrainTrainer: Primus wrapper for NeMo AutoModel diffusion
pre-training. Wan 2.2 (T2V-A14B) is the model that is wired up and tested
here; AutoModel's diffusion recipe also targets other models (e.g. FLUX,
Qwen-Image) but those are not exercised by this backend yet.

Thin-wrapper pattern (same as ``MaxTextPretrainTrainer`` /
``TorchTitanPretrainTrainer``): AutoModel owns FSDP2, the dataloader, the
optimizer and checkpointing internally, so this trainer only

    backend_args (SimpleNamespace)
        -> cleaned dict
        -> temp YAML
        -> AutoModel ``parse_args_and_load_config`` (-> ConfigNode)
        -> ``TrainDiffusionRecipe``

and then delegates ``setup()`` / ``run_train_validation_loop()`` to the recipe.

By routing through AutoModel's own loader we inherit its config semantics
(``_target_``/``_fn`` resolution, the ``wandb.enable`` toggle, ...) and stay
agnostic to AutoModel internals.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from primus.core.trainer.base_trainer import BaseTrainer
from primus.modules.module_utils import error_rank_0, log_rank_0


class NemoAutomodelPretrainTrainer(BaseTrainer):
    """Trainer class for NeMo AutoModel diffusion pre-training."""

    def __init__(self, backend_args: Any):
        super().__init__(backend_args=backend_args)
        self._recipe: Optional[Any] = None
        log_rank_0("Initialized NemoAutomodelPretrainTrainer")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def setup(self):
        """Optional pre-init phase (kept for API symmetry)."""
        log_rank_0("NemoAutomodelPretrainTrainer.setup()")

    def init(self):
        """Build the AutoModel recipe from Primus params and set it up."""
        log_rank_0("NemoAutomodelPretrainTrainer.init() - building AutoModel recipe")

        from nemo_automodel.components.config._arg_parser import (
            parse_args_and_load_config,
        )
        from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe

        from primus.backends.nemo_automodel.argument_builder import (
            export_params_to_yaml,
            namespace_to_dict,
            strip_primus_keys,
        )

        params_dict = strip_primus_keys(namespace_to_dict(self.backend_args))

        # Delegate config materialization to AutoModel's own loader (argv=[] so
        # Primus's process argv is never re-parsed).
        yaml_path = export_params_to_yaml(params_dict)
        try:
            cfg = parse_args_and_load_config(yaml_path, argv=[])
        finally:
            try:
                os.unlink(yaml_path)
            except OSError as e:
                error_rank_0(f"NemoAutomodelPretrainTrainer: failed to delete temp YAML {yaml_path}: {e}")

        self._recipe = TrainDiffusionRecipe(cfg)
        self._recipe.setup()
        log_rank_0("AutoModel recipe initialized successfully")

    def train(self):
        """Execute the AutoModel train/validation loop."""
        if self._recipe is None:
            raise RuntimeError("NemoAutomodelPretrainTrainer.init() must be called before train().")
        log_rank_0("Executing AutoModel diffusion pretrain...")
        self._recipe.run_train_validation_loop()
        log_rank_0("AutoModel diffusion pretrain completed.")
