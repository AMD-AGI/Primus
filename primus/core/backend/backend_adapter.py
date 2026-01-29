###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, Dict

from primus.core.utils.yaml_utils import merge_namespace, nested_namespace_to_dict
from primus.modules.module_utils import log_dict_aligned, log_rank_0


class BackendAdapter(ABC):
    """
    The unified interface for all backend frameworks.

    BackendAdapter is responsible for four main tasks:
        1. Backend initialization (env, patching, paths)
        2. Convert Primus TypedConfig → backend native config / args
        3. Build Trainer class or call backend launcher
        4. Optional hooks (patch, override behavior)
    """

    def __init__(self, framework: str):
        self.framework = framework

    # ============================================================================
    # Optional Methods (Can be overridden by subclasses)
    # ============================================================================

    def setup_sys_path(self, backend_path: str):
        """
        Customize sys.path for backend-specific requirements.

        This method is called after the backend path is added to sys.path.
        Subclasses can override this to add additional paths (e.g., src subdirectory).

        Args:
            backend_path: Resolved absolute path to the backend installation

        Example:
            # MegatronBridgeAdapter can override to add src directory:
            def setup_sys_path(self, backend_path: str):
                import os, sys
                src_path = os.path.join(backend_path, "src")
                if os.path.isdir(src_path) and src_path not in sys.path:
                    sys.path.insert(0, src_path)
        """
        # Default: do nothing

    # ============================================================================
    # Abstract Methods (Must be implemented by subclasses)
    # ============================================================================

    @abstractmethod
    def prepare_backend(self, config: Any):
        """
        Called before creating Trainer.
        Should:
            - setup sys.path for backend
            - run backend-specific setup hooks
            - initialize distributed detect if needed

        Note: setup patches are applied automatically by the base class
        before this method is called.
        """

    @abstractmethod
    def convert_config(self, config: Any) -> Dict[str, Any]:
        """
        Convert TypedConfig (PrimusConfig + ModuleConfig) to backend args.

        Example:
            Primus PretrainConfig -> Megatron args OR Titan args

        Note: build_args patches are applied automatically by the base class
        after this method returns.

        Return:
            A dict or namespace containing args for the backend trainer.
        """

    @abstractmethod
    def load_trainer_class(self, stage: str = "pretrain"):
        """
        Return backend Trainer class.

        Args:
            stage: Stage name (e.g., "pretrain", "sft").
                   The stage is resolved in the upper layer and passed down.

        Returns:
            Trainer class (Megatron → MegatronTrainer, etc.)
        """

    @abstractmethod
    def detect_backend_version(self) -> str:
        """
        Detect backend version for version-specific patches.

        Returns:
            Version string (e.g., "0.8.0", "commit:abc123")

        Raises:
            RuntimeError: If version cannot be detected

        Subclasses must implement this method and should fail fast
        if version detection is not possible.
        """

    # ============================================================================
    # Internal Methods (Do NOT override)
    # ============================================================================

    def _apply_setup_patches(self, primus_config, module_config):
        """
        Apply setup phase patches before backend preparation.

        This is called automatically by create_trainer() and should NOT
        be overridden by subclasses unless you have a very good reason.

        Note: backend_version is NOT passed here because setup patches
        must run BEFORE backend import, so we cannot detect version yet.
        Setup patches should be version-agnostic (e.g., set env vars).

        Args:
            module_config: ModuleConfig instance
        """
        from primus.core.patches import run_patches

        model_name = getattr(module_config, "model", None)

        run_patches(
            backend=self.framework,
            phase="setup",
            backend_version=None,  # No version yet - setup runs before import
            model_name=model_name,
            extra={
                "module_config": module_config,
                "primus_config": primus_config,
            },
        )

    def _apply_build_args_patches(self, primus_config, module_config, backend_args):
        """
        Apply build_args phase patches after config conversion.

        This is called automatically by create_trainer() and should NOT
        be overridden by subclasses unless you have a very good reason.

        Args:
            module_config: ModuleConfig instance
            backend_args: Backend-specific args (output of convert_config)
        """
        from primus.core.patches import run_patches

        backend_version = self.detect_backend_version()
        model_name = getattr(module_config, "model", None)

        run_patches(
            backend=self.framework,
            phase="build_args",
            backend_version=backend_version,
            model_name=model_name,
            extra={
                "primus_config": primus_config,
                "module_config": module_config,
                "backend_args": backend_args,
            },
        )

    def _resolve_stage(self, module_config) -> str | None:
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

        stage = getattr(module_config, "stage", None)
        if stage:
            return stage

        params = getattr(module_config, "params", None)
        if params and hasattr(params, "stage") and params.stage:
            return params.stage

        module_name = getattr(module_config, "name", "")
        return self._infer_stage_from_name(module_name)

    def _infer_stage_from_name(self, module_name: str) -> str | None:
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
        if "sft" in name_lower:
            return "sft"
        if "post_train" in name_lower or name_lower == "post_trainer":
            return "sft"

        return None
    # ============================================================================
    # Public API
    # ============================================================================

    def create_trainer(self, primus_config, module_config):
        """
        Create trainer instance with automatic patch application.

        This is the main entry point that orchestrates the entire trainer
        creation process:
            1. Apply setup patches
            2. Prepare backend environment
            3. Convert config to backend args
            4. Apply build_args patches
            5. Load and instantiate trainer

        Args:
            primus_config: Global Primus configuration
            module_config: Module-specific configuration

        Returns:
            Trainer instance ready to run
        """
        log_rank_0("=" * 80)
        log_rank_0(f"Creating {self.framework.upper()} trainer...")
        log_rank_0("=" * 80)

        # 1) apply setup patches (automatic for all backends)
        log_rank_0("[Step 1/5] Applying setup patches...")
        self._apply_setup_patches(primus_config, module_config)
        log_rank_0("Setup patches applied successfully")

        # 2) backend env/patch/detect
        log_rank_0("[Step 2/5] Preparing backend environment...")
        self.prepare_backend(module_config)
        log_rank_0("Backend environment prepared successfully")

        # 3) config translation
        log_rank_0("[Step 3/5] Converting Primus config to backend args...")
        backend_args = self.convert_config(module_config)
        log_rank_0("Config conversion completed successfully")

        # 4) apply build_args patches (automatic for all backends)
        log_rank_0("[Step 4/5] Applying build_args patches...")
        self._apply_build_args_patches(primus_config, module_config, backend_args)
        log_rank_0("Build_args patches applied successfully")

        # Log the final backend args in aligned format (after patches)
        log_dict_aligned("Final backend args (after patches)", backend_args)

        # Log parameters that were in module_config but not converted to backend_args.
        # These are likely Primus-specific parameters.
        params_dict = nested_namespace_to_dict(module_config.params)
        config_keys = set(params_dict.keys())
        backend_keys = set(vars(backend_args))
        primus_only_keys = config_keys - backend_keys

        if primus_only_keys:
            primus_only_params = {key: params_dict[key] for key in sorted(primus_only_keys)}
            log_dict_aligned("Primus-specific parameters", primus_only_params)

        # After logging the difference between module_config.params and backend_args,
        # merge module_config.params into backend_args so that:
        #   - backend_args contains both converted backend arguments and original
        #     Primus module parameters;
        #   - module_config.params is updated to the merged view, allowing
        #     later patches (e.g., before_train) that call get_args(ctx) to
        #     see backend-derived fields such as seq_length/world_size.
        params = getattr(module_config, "params", None)
        if isinstance(backend_args, SimpleNamespace) and isinstance(params, SimpleNamespace):
            backend_keys = set(vars(backend_args).keys())
            params_keys = set(vars(params).keys())
            # Avoid overriding or raising on keys that already exist in backend_args.
            excepts = list(backend_keys & params_keys)
            merge_namespace(backend_args, params, allow_override=False, excepts=excepts)
            module_config.params = backend_args

        # 5) load trainer class from backend (stage resolved in upper layer)
        log_rank_0("[Step 5/5] Loading trainer class...")
        stage = self._resolve_stage(module_config) or "pretrain"
        TrainerClass = self.load_trainer_class(stage=stage)
        log_rank_0(f"Trainer class loaded: {TrainerClass.__name__}")

        log_rank_0("=" * 80)
        log_rank_0("Trainer creation completed successfully")
        log_rank_0("=" * 80)

        return TrainerClass(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )
