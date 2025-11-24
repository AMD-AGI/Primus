###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
from pathlib import Path

from primus.core.backend.backend_registry import BackendRegistry
from primus.core.config.merge_utils import deep_merge
from primus.core.config.primus_config import PrimusConfig
from primus.core.runtime import init_distributed_env, init_global_logger
from primus.core.utils.arg_utils import parse_cli_overrides
from primus.core.utils.env_setup import setup_training_env
from primus.core.utils.global_vars import set_global_variables


def add_train_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config file (alias: --exp)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Path to data directory [default: ./data]",
    )
    parser.add_argument(
        "--backend_path",
        nargs="?",
        default=None,
        help=(
            "Optional backend import path for Megatron or TorchTitan. "
            "If provided, it will be appended to PYTHONPATH dynamically."
        ),
    )
    parser.add_argument(
        "--export_config",
        type=str,
        help="Optional path to export the final merged config to a file.",
    )
    return parser


def launch_train(args, overrides, module: str):
    """
    Unified training entry point (framework-agnostic).

    Steps:
        1. Setup environment (cache paths, etc.)
        2. Load Primus config (needed for platform info)
        3. Initialize distributed environment (once)
        4. Initialize global logger (once)
        5. Resolve backend path
        6. Select backend adapter
        7. BackendAdapter.prepare_backend()   ← backend specific
        8. Create Trainer via adapter
        9. trainer.init()/run()
    """

    # 0 Validate config file
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"[Primus:Train] Config file not found: {cfg_path}")

    # 1 Environment setup (HuggingFace cache, etc.)
    setup_training_env(args.data_path, setup_hf=True)

    # 2 Load PrimusConfig (must come before distributed init as it needs platform info)
    primus_cfg = PrimusConfig.from_file(cfg_path, args)

    print(f"-------------------- {primus_cfg}")

    # 2.5 Set global variables (needed for platform detection in distributed init)
    set_global_variables(primus_cfg)

    # 3 Initialize distributed environment (one-time global initialization)
    init_distributed_env()

    # 4 Initialize global logger (one-time global initialization)
    init_global_logger(primus_cfg)

    # 5 Extract module-level train config
    try:
        train_cfg = primus_cfg.get_module_config(module)
    except ValueError as exc:
        available_modules = [f"{m.module} (name: {m.name})" for m in primus_cfg._modules.values()]
        raise RuntimeError(
            f"[Primus:Train] Config file missing required module '{module}'.\n"
            f"Available modules in config: {', '.join(available_modules)}\n"
            f"Please ensure your YAML defines a module with 'module: {module}'."
        ) from exc

    # 6 Apply CLI overrides to module params (deep merge to preserve nested structures)
    if overrides:
        override_dict = parse_cli_overrides(overrides)
        train_cfg.params = deep_merge(train_cfg.params, override_dict)

    from primus.core.utils.distributed_logging import log_rank_0

    log_rank_0(f"--------------------00000 {train_cfg.params}")

    # 7 Validate framework is specified
    framework = train_cfg.framework
    if not framework:
        raise ValueError(
            f"[Primus:Train] Module '{module}' missing 'framework' field.\n"
            f"Please specify framework (e.g., 'megatron', 'torchtitan') in your config."
        )

    # 8 Load backend adapter (automatically sets up path)
    try:
        adapter = BackendRegistry.get_adapter(framework, backend_path=args.backend_path)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        raise type(e)(
            f"{e}\n"
            f"Requested framework: '{framework}'\n"
            f"Check your config file's 'framework' field and backend installation."
        ) from e

    # 9 Create trainer (adapter handles backend setup, config conversion, and trainer loading)
    try:
        trainer = adapter.create_trainer(
            primus_config=primus_cfg,
            module_config=train_cfg,
        )
    except Exception as e:
        raise RuntimeError(f"[Primus:Train] Failed to create trainer for '{framework}': {e}") from e

    # 10 Execute training lifecycle
    try:
        trainer.init()
        trainer.run()
    except KeyboardInterrupt:
        print("\n[Primus:Train] Training interrupted by user (Ctrl+C)")
        raise
    except Exception as e:
        raise RuntimeError(f"[Primus:Train] Training execution failed: {e}") from e


# ------------------------------------------------------------------------------
# Standalone invocation
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Primus Train Launcher")
    add_train_parser(parser)

    args, unknown_args = parser.parse_known_args()

    # Use unknown_args as CLI overrides (e.g., batch_size=32, lr=0.001)
    launch_train(args, overrides=unknown_args, module="pretrain")
