# import _log_suppression  # noqa: F401  # must precede all other imports

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

PRIMUS_PATH = os.getenv("PRIMUS_PATH", "/workspace/Primus")
MEGATRON_PATH = os.path.join(PRIMUS_PATH, "third_party/Megatron-LM")

if PRIMUS_PATH not in sys.path:
    sys.path.insert(0, PRIMUS_PATH)
if MEGATRON_PATH not in sys.path:
    sys.path.insert(0, MEGATRON_PATH)

from primus.core.launcher.config import PrimusConfig
from primus.core.launcher.parser import load_primus_config, add_pretrain_parser
from primus_mllog import MLPerfMegatronPretrainTrainer

# _log_suppression.reapply_quiet_logger_levels()


def setup_environment(data_path: Optional[str] = None) -> None:
    if data_path and "HF_HOME" not in os.environ:
        hf_home = os.path.join(data_path, "huggingface")
        os.environ["HF_HOME"] = hf_home
        print(f"[MLPerf Train] HF_HOME={hf_home}")


def load_config(
    config_path: str, overrides: Optional[List[str]] = None
) -> Tuple[PrimusConfig, List[str]]:
    parser = argparse.ArgumentParser()
    add_pretrain_parser(parser)

    args = parser.parse_args([
        '--config', config_path,
        '--data_path', os.getenv('DATA_PATH', '/data'),
    ])

    primus_cfg, unknown_overrides = load_primus_config(args, overrides or [])

    print(f"[MLPerf Train] Loaded config from: {config_path}")
    print(f"[MLPerf Train] Framework: {primus_cfg.get_module_config('pre_trainer').framework}")

    return primus_cfg, unknown_overrides


def create_trainer(
    primus_cfg: PrimusConfig, extra_args: Optional[List[str]] = None
) -> MLPerfMegatronPretrainTrainer:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "29500"))

    trainer = MLPerfMegatronPretrainTrainer(
        module_name="pre_trainer",
        primus_config=primus_cfg,
        module_rank=rank,
        module_world_size=world_size,
        module_master_addr=master_addr,
        module_master_port=master_port,
        extra_args=extra_args,
    )
    return trainer


def main() -> None:
    config_path = os.environ.get("EXP", "/workspace/code/conf/llama3.1_8B-pretrain-fp4.yaml")

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    setup_environment(data_path=os.getenv('DATA_PATH', '/data'))
    primus_cfg, extra_args = load_config(config_path)

    trainer = create_trainer(primus_cfg, extra_args)
    trainer.init()
    trainer.run()


if __name__ == "__main__":
    main()