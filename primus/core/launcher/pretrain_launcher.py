###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os
import sys
from pathlib import Path

from primus.core.backend.backend_registry import BackendRegistry
from primus.core.trainer.pretrain_config import PretrainConfig


def add_pretrain_parser(parser: argparse.ArgumentParser):
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


# ------------------------------------------------------------------------------
# Framework-agnostic path handling
# ------------------------------------------------------------------------------


def _setup_backend_path(framework: str, backend_path=None, verbose=True):
    """
    Insert Python import path for backend modules.

    Priority:
        1. --backend-path
        2. BACKEND_PATH env var
        3. primus/third_party/<backend>
    """
    candidate_paths = []

    # 1) CLI
    if backend_path:
        if isinstance(backend_path, str):
            backend_path = [backend_path]
        candidate_paths.extend(backend_path)

    # 2) ENV
    env_path = os.getenv("BACKEND_PATH")
    if env_path:
        candidate_paths.append(env_path)

    # 3) fallback
    backend_name = BackendRegistry.get_path_name(framework)
    default_path = Path(__file__).resolve().parent.parent / "third_party" / backend_name
    candidate_paths.append(default_path)

    # Normalize
    normalized = list(dict.fromkeys(os.path.abspath(os.path.normpath(p)) for p in candidate_paths))

    # Insert first valid path
    for p in normalized:
        if os.path.exists(p):
            if p not in sys.path:
                sys.path.insert(0, p)
                if verbose:
                    print(f"[Primus] sys.path.insert → {p}")
            return p

    raise FileNotFoundError(f"[Primus] No valid backend path for '{framework}'. Tried: {normalized}")


# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------


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


def launch_pretrain(args, overrides):
    """
    Unified pretraining entry point (framework-agnostic).

    Steps:
        1. Load Primus config
        2. Resolve backend path
        3. Select backend adapter
        4. BackendAdapter.prepare_backend()   ← backend specific
        5. Create Trainer via adapter
        6. trainer.init()/run()
    """

    # 0 Validate config file ----------
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"[Primus:Train] Config file not found: {cfg_path}")

    # 1 Environment setup ----------
    setup_env(args.data_path)

    # 2 Load PrimusConfig ----------
    primus_cfg = PrimusConfig.from_file(cfg_path, args)

    # Extract module-level pretrain config
    try:
        pre_cfg = primus_cfg.get_module_config("pre_trainer")
    except ValueError as exc:
        raise RuntimeError(
            "[Primus:Train] Config file missing required module 'pre_trainer'. "
            "Please ensure your YAML defines a module entry with name/module 'pre_trainer'."
        ) from exc

    trainer_cfg = PretrainConfig.from_simple_namespace(pre_cfg)

    framework = trainer_cfg.framework

    # 3 Insert backend path (framework-agnostic) ----------
    _setup_backend_path(framework, backend_path=args.backend_path)

    # 4 Load backend adapter ----------
    adapter = BackendRegistry.get_adapter(framework)

    # 5 Backend-specific setup ----------
    # Includes: patch pipeline, version fix, env overrides, etc.
    adapter.prepare_backend(trainer_cfg)

    # 6 Create trainer (adapter ensures correct conversion/loader) ----------
    trainer = adapter.create_trainer(
        primus_config=primus_cfg,
        module_context=trainer_cfg,
    )

    # 7 Execute training lifecycle ----------
    trainer.init()
    trainer.run()


# ------------------------------------------------------------------------------
# Standalone invocation
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Primus Pretrain Launcher")
    add_pretrain_parser(parser)

    args, unknown_args = parser.parse_known_args()

    # Overrides reserved for future use (lr=xx style CLI overrides)
    overrides = []

    launch_pretrain(args, overrides)
