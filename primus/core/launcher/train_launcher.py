###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os
from pathlib import Path

from primus.core.backend.backend_registry import BackendRegistry
from primus.core.config.primus_config import PrimusConfig
from primus.core.utils.arg_utils import parse_cli_overrides


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


def setup_env(data_path: str):
    """Setup HuggingFace cache path."""
    if "HF_HOME" not in os.environ:
        hf_home = os.path.join(data_path, "huggingface")
        os.environ["HF_HOME"] = hf_home
        print(f"[Primus] HF_HOME={hf_home}")
    else:
        print(f"[Primus] HF_HOME already set: {os.environ['HF_HOME']}")


# ------------------------------------------------------------------------------
# Main Training Flow
# ------------------------------------------------------------------------------


def launch_train(args, overrides, module: str):
    """
    Unified training entry point (framework-agnostic).

    Steps:
        1. Load Primus config
        2. Resolve backend path
        3. Select backend adapter
        4. BackendAdapter.prepare_backend()   ← backend specific
        5. Create Trainer via adapter
        6. trainer.init()/run()
    """

    # 0 Validate config file
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"[Primus:Train] Config file not found: {cfg_path}")

    # 1 Environment setup
    setup_env(args.data_path)

    # 2 Load PrimusConfig
    primus_cfg = PrimusConfig.from_file(cfg_path, args)

    # Extract module-level train config
    try:
        train_cfg = primus_cfg.get_module_config(module)
    except ValueError as exc:
        available_modules = [f"{m.module} (name: {m.name})" for m in primus_cfg._modules.values()]
        raise RuntimeError(
            f"[Primus:Train] Config file missing required module '{module}'.\n"
            f"Available modules in config: {', '.join(available_modules)}\n"
            f"Please ensure your YAML defines a module with 'module: {module}'."
        ) from exc

    # Apply CLI overrides to module params
    if overrides:
        override_dict = parse_cli_overrides(overrides)
        train_cfg.params.update(override_dict)

    # Validate framework is specified
    framework = train_cfg.framework
    if not framework:
        raise ValueError(
            f"[Primus:Train] Module '{module}' missing 'framework' field.\n"
            f"Please specify framework (e.g., 'megatron', 'torchtitan') in your config."
        )

    # 3 Load backend adapter (automatically sets up path)
    try:
        adapter = BackendRegistry.get_adapter(framework, backend_path=args.backend_path)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        raise type(e)(
            f"{e}\n"
            f"Requested framework: '{framework}'\n"
            f"Check your config file's 'framework' field and backend installation."
        ) from e

    # 4 Create trainer (adapter handles backend setup, config conversion, and trainer loading)
    try:
        trainer = adapter.create_trainer(
            primus_config=primus_cfg,
            module_config=train_cfg,
        )
    except Exception as e:
        raise RuntimeError(f"[Primus:Train] Failed to create trainer for '{framework}': {e}") from e

    # 5 Execute training lifecycle
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

    # Overrides reserved for future use (lr=xx style CLI overrides)
    overrides = []

    launch_train(args, overrides, module="pretrain")
