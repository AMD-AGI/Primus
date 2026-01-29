###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


# Trigger registration of all Megatron patches (args_patches, env_patches, etc.)
import primus.backends.megatron.patches  # noqa: F401
from primus.backends.megatron.argument_builder import MegatronArgBuilder
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.backend.backend_registry import BackendRegistry
from primus.modules.module_utils import log_rank_0


class MegatronAdapter(BackendAdapter):
    """
    The complete BackendAdapter implementation for Megatron-LM.

    This adapter is designed to:
        - Handle multi-version Megatron differences
        - Convert Primus config → Megatron args using ArgBuilder
        - Apply patches automatically (PR fixes, kernel bugs, attention fixes)
        - Load the appropriate Trainer class depending on Megatron version
    """

    def __init__(self, framework="megatron"):
        super().__init__(framework)

    # Backend Setup & Patches
    def prepare_backend(self, config):
        """
        Megatron-specific environment preparation.

        Steps:
            - Run Primus setup hooks
            - Set environment variables

        Note: setup patches are applied automatically by the base class
        before this method is called.
        """
        # Run setup hooks from BackendRegistry
        BackendRegistry.run_setup("megatron")

        log_rank_0("[Primus:MegatronAdapter] Backend prepared")

    # Override base class method for version detection
    def detect_backend_version(self) -> str:
        """
        Detect Megatron-LM version.

        Delegates to the trainer class's detect_version() classmethod to ensure
        consistency and proper separation of concerns.

        Returns:
            Version string (e.g., "0.15.0rc8")

        Raises:
            RuntimeError: If version cannot be detected
        """
        # Get trainer class and call its detect_version classmethod
        TrainerClass = self.load_trainer_class()
        return TrainerClass.detect_version()

    # Config → Megatron Args
    def convert_config(self, module_config):
        """
        Convert Primus ModuleConfig → final Megatron-LM argument Namespace.

        This layer:
            - Takes module_config.params (which already includes CLI overrides)
            - Fills missing fields using Megatron-LM defaults
            - Injects distributed environment variables (via builder)
            - Produces a Megatron-compatible argparse-like Namespace

        Note: build_args patches are applied automatically by the base class
        after this method returns.

        Args:
            module_config: ModuleConfig instance with params dict

        Returns:
            SimpleNamespace with Megatron args
        """
        # Instantiate the builder
        builder = MegatronArgBuilder()

        # Feed in config params (already merged with CLI overrides in train_launcher)
        # module_config.params is a flat dict of Megatron-recognized fields.
        builder.update(module_config.params)

        # Produce the final Megatron Namespace (with distributed env injected)
        megatron_args = builder.finalize()

        log_rank_0(f"[Primus:MegatronAdapter] Converted config → {len(vars(megatron_args))} Megatron args")

        return megatron_args

    # Load Trainer Class (Version Adaptive)
    def load_trainer_class(self, module_config=None):
        """
        Load Megatron trainer class registered via BackendRegistry.

        The stage is determined by (in order of priority):
            1. module_config.stage (explicit configuration)
            2. module_config.params.stage (from YAML overrides)
            3. Inferred from module name (pre_trainer → pretrain, sft_trainer → sft)
            4. Default stage for the framework

        Args:
            module_config: Module configuration (optional, for stage detection)

        Returns:
            Trainer class for the determined stage.
        """
        stage = self._resolve_stage(module_config)
        if stage:
            log_rank_0(f"[Primus:MegatronAdapter] Loading trainer for stage: {stage}")

        try:
            return BackendRegistry.get_trainer_class(self.framework, stage=stage)
        except ValueError as exc:
            available_stages = BackendRegistry.list_stages(self.framework)
            if available_stages:
                raise RuntimeError(
                    f"[Primus:MegatronAdapter] No trainer for stage '{stage}'.\n"
                    f"Available stages: {', '.join(available_stages)}\n"
                    f"Hint: Set stage in config or use module name like 'pre_trainer' or 'sft_trainer'"
                ) from exc
            else:
                raise RuntimeError(
                    "[Primus:MegatronAdapter] 'megatron' backend not registered. "
                    "Ensure primus.backends.megatron defines the trainer."
                ) from exc

    def _resolve_stage(self, module_config) -> str:
        """
        Resolve training stage from module configuration.

        Priority:
            1. module_config.stage (explicit)
            2. module_config.params.stage (from overrides)
            3. Inferred from module name
            4. Default stage (None - let registry decide)

        Args:
            module_config: Module configuration

        Returns:
            Stage name (e.g., "pretrain", "sft") or None for default
        """
        if module_config is None:
            return None

        # 1. Explicit stage in module_config
        if hasattr(module_config, "stage") and module_config.stage:
            return module_config.stage

        # 2. Stage in params (from YAML overrides)
        params = getattr(module_config, "params", None)
        if params and hasattr(params, "stage") and params.stage:
            return params.stage

        # 3. Infer from module name
        module_name = getattr(module_config, "name", "")
        return self._infer_stage_from_name(module_name)

    def _infer_stage_from_name(self, module_name: str) -> str:
        """
        Infer training stage from module name.

        Mapping:
            - pre_trainer, pretrain_trainer → pretrain
            - sft_trainer → sft
            - post_trainer, posttrain_trainer → sft

        Args:
            module_name: Module name from config

        Returns:
            Inferred stage name, or None if cannot infer
        """
        if not module_name:
            return None

        name_lower = module_name.lower()

        if "pre_train" in name_lower or name_lower == "pre_trainer":
            return "pretrain"
        elif "sft" in name_lower:
            return "sft"
        elif "post_train" in name_lower or name_lower == "post_trainer":
            return "sft"  # post_trainer typically means SFT

        return None
