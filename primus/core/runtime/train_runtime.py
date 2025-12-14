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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from primus.core.backend.backend_registry import BackendRegistry
from primus.core.config.merge_utils import deep_merge
from primus.core.config.primus_config import (
    get_module_config,
    get_module_names,
    load_primus_config,
)
from primus.core.runtime.logging import init_worker_logger
from primus.core.utils.arg_utils import parse_cli_overrides
from primus.core.utils.env_setup import setup_training_env
from primus.modules.module_utils import log_rank_0, warning_rank_0

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
    """

    def __init__(self, args: Any):
        self.args = args
        self.ctx: Optional[TrainContext] = None

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

    # --------------------------- Internal Steps --------------------------- #ide_str}")

    def _initialize_runtime_environment(self) -> None:
        """Initialize full runtime environment before creating backend/trainer."""
        self._initialize_environment()
        self._initialize_distributed_context()
        self._initialize_logging()

    def _initialize_backend_and_execute(self) -> None:
        """Load backend adapter, create trainer and execute its lifecycle."""
        self._initialize_backend()
        self._initialize_trainer()
        self._run_trainer_lifecycle()

    def _initialize_environment(self) -> None:
        assert self.ctx is not None, "TrainContext must be initialized before environment setup."
        data_path = self.ctx.data_path
        # Ensure data directory exists before environment setup.
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
        # setup_training_env expects a string path.
        setup_training_env(str(data_path), setup_hf=True)

    def _initialize_configuration(self, module_name: str, overrides: Optional[List[str]] = None) -> None:
        cfg_path = Path(self.args.config)
        assert cfg_path.exists(), f"[Primus:TrainRuntime] Config file not found: {cfg_path}"

        primus_cfg = load_primus_config(cfg_path, self.args)

        # For platform detection in distributed init.
        # set_global_variables(primus_cfg)

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
        self._apply_overrides(module_name, module_cfg, overrides)

    def _apply_overrides(self, module_name: str, module_cfg: Any, overrides: Optional[List[str]]):
        if not overrides:
            return

        override_dict: Dict[str, Any] = parse_cli_overrides(overrides)
        log_rank_0(
            f"[Primus:TrainRuntime] Applying CLI overrides for module "
            f"'{self.ctx.module_name}': {override_dict}"
        )
        module_cfg.params = deep_merge(module_cfg.params, override_dict)

    def _initialize_distributed_context(self) -> None:
        assert self.ctx is not None, "TrainContext must be initialized before distributed init."

        from primus.core.utils.env import get_torchrun_env

        dist_env = get_torchrun_env()
        self.ctx.rank = dist_env["rank"]
        self.ctx.world_size = dist_env["world_size"]
        self.ctx.local_rank = dist_env["local_rank"]
        self.ctx.local_world_size = dist_env["local_world_size"]
        self.ctx.master_addr = dist_env["master_addr"]
        self.ctx.master_port = dist_env["master_port"]

        print(
            f"[Primus:Env] rank: {self.ctx.rank}, world_size: {self.ctx.world_size}, local_rank: {self.ctx.local_rank}, local_world_size: {self.ctx.local_world_size}, master_addr: {self.ctx.master_addr}, master_port: {self.ctx.master_port}"
        )

    def _initialize_logging(self) -> None:
        assert self.ctx is not None, "TrainContext must be initialized before logger init."
        # Use legacy logger init if available; otherwise rely on module_utils logging.
        init_worker_logger(
            primus_config=self.ctx.primus_config,
            module_name=self.ctx.module_name,
            module_config=self.ctx.module_config,
        )

    def _initialize_backend(self) -> None:
        assert self.ctx is not None, "TrainContext must be initialized before backend adapter."
        backend_path = getattr(self.args, "backend_path", None)

        try:
            adapter = BackendRegistry.get_adapter(backend=self.ctx.framework, backend_path=backend_path)
        except Exception as e:
            # Preserve original error type but enrich with framework context for easier debugging.
            raise ValueError(
                f"[Primus:TrainRuntime] Requested framework '{self.ctx.framework}' "
                f"failed to resolve backend adapter (backend_path={backend_path}): {e}"
            ) from e

        assert (
            adapter is not None
        ), f"Failed to resolve backend adapter. framework: '{self.ctx.framework} backend_path: {backend_path}"

        self.ctx.adapter = adapter

    def _initialize_trainer(self) -> None:
        assert (
            self.ctx is not None and self.ctx.adapter is not None
        ), "Backend adapter must be loaded before creating trainer."

        try:
            trainer = self.ctx.adapter.create_trainer(
                primus_config=self.ctx.primus_config,
                module_config=self.ctx.module_config,
            )
        except Exception as e:
            # Wrap adapter errors into a clear RuntimeError with framework context.
            raise RuntimeError(
                f"Failed to create trainer instance for framework '{self.ctx.framework}': {e}"
            ) from e

        assert trainer is not None, f"Failed to create trainer instance for framework '{self.ctx.framework}'."

        self.ctx.trainer = trainer

    def _run_trainer_lifecycle(self) -> None:
        assert (
            self.ctx is not None and self.ctx.trainer is not None
        ), "Trainer must be created before executing lifecycle."

        trainer = self.ctx.trainer

        # 1) Optional setup phase
        trainer.setup()

        # 2) Initialize training components
        trainer.init()

        # 3) Execute training
        trainer.run()

        # 4) Cleanup and finalize
        trainer.cleanup()

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
