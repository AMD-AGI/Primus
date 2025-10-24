###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os
import sys
from pathlib import Path

from primus.core.launcher.config import PrimusConfig
from primus.core.launcher.parser import add_pretrain_parser, load_primus_config


# Lazy backend loader
def load_backend_trainer(framework: str):
    if framework == "megatron":
        from primus.modules.trainer.megatron.pre_trainer import MegatronPretrainTrainer

        return MegatronPretrainTrainer
    elif framework == "light-megatron":
        from primus.modules.trainer.lightmegatron.pre_trainer import (
            LightMegatronPretrainTrainer,
        )

        return LightMegatronPretrainTrainer
    elif framework == "torchtitan":
        from primus.modules.trainer.torchtitan.pre_trainer import (
            TorchTitanPretrainTrainer,
        )

        return TorchTitanPretrainTrainer
    elif framework == "hybrid_models":
        from primus.modules.trainer.hybrid_models.pre_trainer import HybridModelsPretrainTrainer

        return HybridModelsPretrainTrainer
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def setup_backend_path(framework: str, backend_path=None, verbose: bool = True):
    """
    Setup Python path for backend modules.

    Priority order:
    1. --backend-path from CLI
    2. BACKEND_PATH from environment
    3. Source tree fallback: <primus>/../../third_party/{framework}

    Returns:
        str: The first valid backend path inserted into sys.path.
    """
    candidate_paths = []

    # 1) From CLI
    if backend_path:
        if isinstance(backend_path, str):
            backend_path = [backend_path]
        candidate_paths.extend(backend_path)

    # 2) From environment variable
    env_path = os.getenv("BACKEND_PATH")
    if env_path:
        candidate_paths.append(env_path)

    # 3) Fallback to source tree under third_party
    fallback_name_map = {
        "megatron": "Megatron-LM",
        "light-megatron": "Megatron-LM",
        "torchtitan": "torchtitan",
        "hybrid_models": "AMD-Hybrid-Models",
    }
    mapped_name = fallback_name_map.get(framework, framework)
    default_path = Path(__file__).resolve().parent.parent / "third_party" / mapped_name
    candidate_paths.append(default_path)

    # Normalize & deduplicate
    candidate_paths = list(dict.fromkeys(os.path.normpath(os.path.abspath(p)) for p in candidate_paths))

    # Insert the first existing path into sys.path
    for path in candidate_paths:
        if os.path.exists(path):
            if path not in sys.path:
                sys.path.insert(0, path)
                if verbose:
                    print(f"[Primus] sys.path.insert: {path}")
            return path  # Return the first valid path

    # None of the candidate paths exist
    raise FileNotFoundError(
        f"[Primus] backend_path not found for framework '{framework}'. " f"Tried paths: {candidate_paths}"
    )


def setup_env(data_path: str):
    if "HF_HOME" not in os.environ:
        hf_home = os.path.join(data_path, "huggingface")
        os.environ["HF_HOME"] = hf_home
        print(f"[Primus CLI] HF_HOME={hf_home}")
    else:
        hf_home = os.environ["HF_HOME"]
        print(f"[Primus CLI] HF_HOME already set: {hf_home}")


def launch_pretrain_trainer(primus_cfg: PrimusConfig):
    """
    Launch the training using the Primus trainer.

    Args:
        primus_cfg (PrimusConfig): Parsed Primus configuration object.
    """
    # Get pre_trainer module configuration
    pre_trainer_cfg = primus_cfg.get_module_config("pre_trainer")
    framework = pre_trainer_cfg.framework

    # Lazy import backend trainer
    TrainerClass = load_backend_trainer(framework)

    # envs set by torchrun
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "29500"))

    # Initialize trainer
    trainer = TrainerClass(
        module_name="pre_trainer",
        primus_config=primus_cfg,
        module_rank=rank,
        module_world_size=world_size,
        module_master_addr=master_addr,
        module_master_port=master_port,
    )

    # Launch training
    trainer.init()
    trainer.run()


def launch_pretrain_from_cli(args, overrides):
    """
    Entry point for the 'train' subcommand.

     Steps:
        1. Load and parse the experiment YAML config
        2. Merge CLI overrides into the config
        3. Optionally export the merged config
        4. Setup backend path
        5. Launch the training
    """
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"[Primus:Train] Config file '{cfg_path}' not found.")

    setup_env(data_path=args.data_path)

    primus_cfg = load_primus_config(args, overrides)

    # Export merged config if requested
    if args.export_config:
        primus_cfg.export(export_path=args.export_config)

    # Setup backend path for dynamic import
    framework = primus_cfg.get_module_config("pre_trainer").framework
    setup_backend_path(framework=framework, backend_path=args.backend_path, verbose=True)

    launch_pretrain_trainer(primus_cfg=primus_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pretrain")
    add_pretrain_parser(parser)

    args, unknown_args = parser.parse_known_args()

    launch_pretrain_from_cli(args, unknown_args)
