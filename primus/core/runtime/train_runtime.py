###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# Primus Runtime Orchestrator for Training
#
# This module contains the high-level runtime orchestration logic for Primus
# training workflows. It is framework-agnostic and delegates framework-specific
# behavior to BackendAdapter and Trainer implementations.
###############################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from primus.core.backend.backend_registry import BackendRegistry
from primus.core.config.merge_utils import deep_merge
from primus.core.config.primus_config import (
    get_module_config,
    get_module_names,
    load_primus_config,
)
from primus.core.patches import run_patches
from primus.core.runtime.logging import init_worker_logger
from primus.core.utils.arg_utils import parse_cli_overrides
from primus.core.utils.env_setup import setup_training_env
from primus.core.utils.yaml_utils import (
    dict_to_nested_namespace,
    merge_namespace,
    nested_namespace_to_dict,
)
from primus.modules.module_utils import log_dict_aligned, log_rank_0, warning_rank_0

# ---------------------------------------------------------------------------
# Context & Hooks
# ---------------------------------------------------------------------------


@dataclass
class TrainContext:
    """Aggregated runtime context for a single training module."""

    # CLI & basic info
    config_path: Path
    data_path: Path
    module_name: str

    # Configs
    primus_config: Any
    module_config: Any
    framework: str

    # Runtime objects (lazy filled)
    adapter: Any = None
    trainer: Any = None
    backend_args: Any = None
    backend_version: Optional[str] = None

    # Distributed context
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_world_size: int = 1
    master_addr: str = ""
    master_port: int = 0

    # Cached distributed environment (to avoid redundant calls)
    _dist_env: Optional[Dict[str, Any]] = field(default=None)


# ---------------------------------------------------------------------------
# PrimusRuntime
# ---------------------------------------------------------------------------


class PrimusRuntime:
    """
    Orchestrator for Primus training workflows.

    Responsibilities:
      - Load and validate Primus configuration for a single module
      - Apply CLI overrides on top of module parameters
      - Initialize runtime environment (paths / distributed / logging)
      - Resolve backend adapter and construct the Trainer
      - Drive the Trainer lifecycle (setup → init → run → cleanup)

    Lifecycle:
        1. _initialize_configuration: Load/merge configs, apply CLI overrides
        2. _initialize_runtime_environment:
           a. _initialize_environment: Setup paths, cache directories
           b. _initialize_distributed_context: Read distributed env vars
           c. _initialize_logging: Initialize logger for this module
        3. _initialize_backend_and_execute:
           a. _initialize_adapter: Load backend adapter via registry
           b. _initialize_trainer: Convert config → backend args, create trainer
           c. _run_trainer_lifecycle: Execute trainer methods with patch phases

    Patch Phases (applied via _run_phase_patches):
        - setup: Before trainer initialization (backend setup, env fixes)
        - build_args: After backend args created, before trainer instantiation
        - before_train: After trainer.init(), before trainer.train()
        - after_train: After trainer.train(), before cleanup

    Error Handling:
        - Configuration errors: Raise immediately with context
        - Runtime errors: Clean up trainer, then wrap and re-raise
        - KeyboardInterrupt: Clean up gracefully, then propagate
    """

    def __init__(self, args: Any):
        self.args = args
        self.ctx: Optional[TrainContext] = None

    # ----------------------------- Helper Methods ------------------------- #

    def _ensure_context(self, require_adapter: bool = False, require_trainer: bool = False) -> TrainContext:
        """
        Validate that TrainContext is properly initialized.

        Args:
            require_adapter: If True, also check that adapter is initialized
            require_trainer: If True, also check that trainer is initialized

        Returns:
            TrainContext: The validated context

        Raises:
            RuntimeError: If context or required components are not initialized
        """
        if self.ctx is None:
            raise RuntimeError(
                "TrainContext not initialized. Ensure _initialize_configuration() is called first."
            )
        if require_adapter and self.ctx.adapter is None:
            raise RuntimeError(
                "Backend adapter not initialized. Ensure _initialize_adapter() is called first."
            )
        if require_trainer and self.ctx.trainer is None:
            raise RuntimeError("Trainer not initialized. Ensure _initialize_trainer() is called first.")
        return self.ctx

    # ----------------------------- Public API ----------------------------- #

    def run_train_module(self, module_name: str, overrides: Optional[List[str]] = None) -> None:
        """Top-level API used by CLI: run training for a single module."""
        overrides = overrides or []

        try:
            # 1) Initialize configuration (PrimusConfig + module_config + CLI overrides)
            self._initialize_configuration(module_name, overrides)
            # 2) Initialize runtime environment (paths, distributed, logging)
            self._initialize_runtime_environment()
            # 3) Initialize backend and execute trainer lifecycle
            self._initialize_backend_and_execute()
        except KeyboardInterrupt as e:
            log_rank_0("Interrupted by user (Ctrl+C)")
            self._safe_cleanup(error=e)
            raise
        except BaseException as e:
            # Best-effort cleanup; wrap into RuntimeError for caller.
            self._safe_cleanup(error=e)
            raise RuntimeError(f"Training execution failed: {e}") from e

    # --------------------------- Internal Steps --------------------------- #

    def _initialize_runtime_environment(self) -> None:
        """Initialize full runtime environment before creating backend/trainer."""
        self._initialize_environment()
        self._initialize_distributed_context()
        self._initialize_logging()

    def _initialize_backend_and_execute(self) -> None:
        """Load backend adapter, create trainer and execute its lifecycle."""
        self._initialize_adapter()
        self._initialize_trainer()
        self._run_trainer_lifecycle()

    def _get_backend_version(self) -> Optional[str]:
        """
        Get backend version, with caching to avoid repeated detection.

        Returns:
            Optional[str]: Backend version string, or None if detection fails
        """
        ctx = self._ensure_context(require_adapter=False)

        # Return cached version if available
        if ctx.backend_version is not None:
            return ctx.backend_version

        # Cannot detect without adapter
        if ctx.adapter is None:
            ctx.backend_version = None
            return None

        # Attempt detection with error handling
        try:
            ctx.backend_version = ctx.adapter.detect_backend_version()
        except Exception as e:
            warning_rank_0(f"[Runtime] Failed to detect backend version: {e}")
            ctx.backend_version = None

        return ctx.backend_version

    def _run_phase_patches(self, phase: str, backend_args: Any = None) -> None:
        """
        Apply a patch phase in a single, runtime-owned place.

        Runtime orchestrates all phases (setup/build_args/before_train/after_train)
        to keep phase placement consistent across backends.

        Phases:
            setup: Before trainer initialization (backend setup, env fixes)
            build_args: After backend args created, before trainer instantiation
            before_train: After trainer.init(), before trainer.train()
            after_train: After trainer.train(), before cleanup
        """
        ctx = self._ensure_context()
        backend_version = self._get_backend_version()

        log_rank_0(f"[Runtime] Applying {phase} patches...")
        run_patches(
            backend=ctx.framework,
            phase=phase,
            backend_version=backend_version,
            model_name=getattr(ctx.module_config, "model", None),
            module_name=ctx.module_name,
            extra={
                "backend_args": backend_args,
                "primus_config": ctx.primus_config,
                "module_config": ctx.module_config,
            },
        )

    def _initialize_environment(self) -> None:
        """Initialize training environment (paths, cache directories)."""
        ctx = self._ensure_context()
        data_path = ctx.data_path
        # Ensure data directory exists before environment setup.
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
        # setup_training_env expects a string path.
        setup_training_env(str(data_path), setup_hf=True)

    def _initialize_configuration(self, module_name: str, overrides: Optional[List[str]] = None) -> None:
        cfg_path = Path(self.args.config)
        assert cfg_path.exists(), f"[Primus:TrainRuntime] Config file not found: {cfg_path}"

        primus_cfg = load_primus_config(cfg_path, self.args)

        # Resolve module configuration via core helper.
        module_cfg = get_module_config(primus_cfg, module_name)
        available_modules = get_module_names(primus_cfg) or ["none"]
        assert module_cfg is not None, (
            f"Missing required module '{module_name}' in config file '{cfg_path}'.\n"
            f"Available modules: {', '.join(available_modules)}\n"
            f"Check your YAML and ensure 'module: {module_name}' is defined."
        )

        framework = module_cfg.framework
        if not framework:
            raise ValueError(f"[Primus:TrainRuntime] Module '{module_name}' missing 'framework'.")

        # Initialize TrainContext based on raw configuration (before CLI overrides).
        self.ctx = TrainContext(
            config_path=cfg_path,
            data_path=Path(getattr(self.args, "data_path", "./data")),
            module_name=module_name,
            primus_config=primus_cfg,
            module_config=module_cfg,
            framework=framework,
        )

        # Apply CLI overrides to module params as part of configuration initialization.
        self._apply_overrides(module_cfg, overrides)

    def _apply_overrides(self, module_cfg: Any, overrides: Optional[List[str]]):
        if not overrides:
            return

        override_dict: Dict[str, Any] = parse_cli_overrides(overrides)
        log_rank_0(f"[Runtime] Applying CLI overrides: {override_dict}")

        # module_cfg.params is a nested SimpleNamespace tree; convert to dict for merging,
        # apply deep_merge, then convert back to SimpleNamespace.
        base_params_dict = nested_namespace_to_dict(module_cfg.params)
        merged_params_dict = deep_merge(base_params_dict, override_dict)
        module_cfg.params = dict_to_nested_namespace(merged_params_dict)

    def _initialize_distributed_context(self) -> None:
        """Initialize distributed training context from environment variables."""
        ctx = self._ensure_context()

        from primus.core.utils.env import get_torchrun_env

        # Cache the distributed environment in the context to avoid redundant calls
        dist_env = get_torchrun_env()
        ctx._dist_env = dist_env

        ctx.rank = dist_env["rank"]
        ctx.world_size = dist_env["world_size"]
        ctx.local_rank = dist_env["local_rank"]
        ctx.local_world_size = dist_env["local_world_size"]
        ctx.master_addr = dist_env["master_addr"]
        ctx.master_port = dist_env["master_port"]

        log_rank_0(
            f"[Runtime] Distributed: rank={ctx.rank}, world_size={ctx.world_size}, "
            f"local_rank={ctx.local_rank}, master={ctx.master_addr}:{ctx.master_port}"
        )

    def _initialize_logging(self) -> None:
        """Initialize logging for the training module."""
        ctx = self._ensure_context()
        # Use legacy logger init if available; otherwise rely on module_utils logging.
        init_worker_logger(
            primus_config=ctx.primus_config,
            module_name=ctx.module_name,
            module_config=ctx.module_config,
        )

    def _initialize_adapter(self) -> None:
        """Resolve backend adapter instance via BackendRegistry."""
        ctx = self._ensure_context()
        backend_path = getattr(self.args, "backend_path", None)

        adapter = BackendRegistry.get_adapter(backend=ctx.framework, backend_path=backend_path)

        if adapter is None:
            available = BackendRegistry.list_available_backends()
            raise RuntimeError(
                f"Failed to resolve backend adapter for framework='{ctx.framework}' "
                f"with backend_path={backend_path!r}. "
                f"Available backends: {available or 'none'}"
            )

        # Ensure backend is importable before running setup phase patches.
        adapter.setup_backend_path(backend_path=backend_path)

        ctx.adapter = adapter

    def _initialize_trainer(self) -> None:
        """Create and initialize the trainer instance."""
        ctx = self._ensure_context(require_adapter=True)

        module_config = ctx.module_config
        adapter = ctx.adapter

        # Prepare backend environment
        adapter.prepare_backend(module_config)

        # Build backend args from Primus params
        backend_args = adapter.convert_config(module_config.params)
        ctx.backend_args = backend_args

        # Phase: build_args (after args creation, before trainer instantiation)
        self._run_phase_patches(phase="build_args", backend_args=backend_args)

        # Log final args after patches, then merge module_config.params into backend_args
        log_dict_aligned("Final backend args (after patches)", backend_args)

        # Log parameters that were in module_config but not converted to backend_args.
        # These are likely Primus-specific parameters.
        # Optimize: Use vars() directly instead of converting to dict
        config_keys = set(vars(module_config.params).keys())
        backend_keys = set(vars(backend_args))
        primus_only_keys = config_keys - backend_keys

        if primus_only_keys:
            primus_only_params = {key: getattr(module_config.params, key) for key in sorted(primus_only_keys)}
            log_dict_aligned("Primus-specific parameters", primus_only_params)

        # Merge backend_args into params (backend_args overrides params)
        merge_namespace(backend_args, module_config.params, allow_override=False, excepts=[])
        module_config.params = backend_args

        # Load trainer class and instantiate
        stage = getattr(module_config.params, "stage", "pretrain") or "pretrain"
        TrainerClass = adapter.load_trainer_class(stage=stage)

        # Pass cached distributed environment to avoid redundant get_torchrun_env() call
        trainer = TrainerClass(backend_args=backend_args)

        # If trainer supports dist_env parameter, we could optimize by passing it
        # For now, BaseTrainer will call get_torchrun_env() again but we've documented the issue

        if trainer is None:
            raise RuntimeError(
                f"Failed to create trainer instance for framework '{ctx.framework}'. "
                f"TrainerClass={TrainerClass}"
            )
        ctx.trainer = trainer

    def _run_trainer_lifecycle(self) -> None:
        """Execute the complete trainer lifecycle: setup → init → train → cleanup."""
        ctx = self._ensure_context(require_trainer=True)

        def _log_step(step_name: str, func):
            """Log step start/end and execute the function."""
            log_rank_0("=" * 80)
            log_rank_0(f"{step_name} started")
            log_rank_0("=" * 80)
            func()
            log_rank_0("=" * 80)
            log_rank_0(f"{step_name} completed")
            log_rank_0("=" * 80)

        trainer = ctx.trainer

        # 1) Optional setup phase
        self._run_phase_patches(phase="setup", backend_args=ctx.backend_args)
        _log_step("Setup", trainer.setup)

        # 2) Initialize training components
        _log_step("Init", trainer.init)

        # 3) Execute training
        self._run_phase_patches(phase="before_train", backend_args=ctx.backend_args)
        _log_step("Training", trainer.train)

        # 4) Cleanup and finalize
        self._run_phase_patches(phase="after_train", backend_args=ctx.backend_args)
        _log_step("Cleanup", trainer.cleanup)

    # --------------------------- Cleanup ---------------------------------- #

    def _safe_cleanup(self, error: Optional[BaseException]) -> None:
        ctx = self.ctx
        if ctx is None or ctx.trainer is None:
            return

        try:
            ctx.trainer.cleanup(on_error=error is not None)
        except Exception as e:
            # We are already in error path; log and continue instead of raising.
            warning_rank_0(f"Error during trainer.cleanup: {e}")
