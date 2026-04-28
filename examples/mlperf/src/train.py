import os
import sys
from pathlib import Path
import argparse

PRIMUS_PATH = os.getenv("PRIMUS_PATH", "/home/vidgoyal/Primus")
MEGATRON_PATH = os.path.join(PRIMUS_PATH, "third_party/Megatron-LM")

if PRIMUS_PATH not in sys.path:
    sys.path.insert(0, PRIMUS_PATH)
if MEGATRON_PATH not in sys.path:
    sys.path.insert(0, MEGATRON_PATH)

from primus.core.launcher.config import PrimusConfig
from primus.core.launcher.parser import load_primus_config, add_pretrain_parser
from primus.modules.trainer.megatron.pre_trainer import MegatronPretrainTrainer


def setup_environment(data_path: str = None):
    if data_path and "HF_HOME" not in os.environ:
        hf_home = os.path.join(data_path, "huggingface")
        os.environ["HF_HOME"] = hf_home
        print(f"[MLPerf Train] HF_HOME={hf_home}")


def load_config(config_path: str, overrides: list = None) -> PrimusConfig:
    parser = argparse.ArgumentParser()
    add_pretrain_parser(parser)
    
    args = parser.parse_args([
        '--config', config_path,
        '--data_path', os.getenv('DATA_PATH', '/data'),
    ])
    
    primus_cfg = load_primus_config(args, overrides or [])
    
    print(f"[MLPerf Train] Loaded config from: {config_path}")
    print(f"[MLPerf Train] Framework: {primus_cfg[0].get_module_config('pre_trainer').framework}")
    
    return primus_cfg[0]


def create_trainer(primus_cfg: PrimusConfig) -> MegatronPretrainTrainer:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "29500"))

    trainer = MegatronPretrainTrainer(
        primus_config=primus_cfg,
        module_rank=rank,
        module_world_size=world_size,
        module_master_addr=master_addr,
        module_master_port=master_port,
    )

    return trainer


def main():
    config_path = os.environ.get("EXP", "/workspace/code/conf/llama3.1_8B-pretrain.yaml")
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    setup_environment(data_path=os.getenv('DATA_PATH', '/data'))
    primus_cfg = load_config(config_path)
    
    trainer = create_trainer(primus_cfg)
    trainer.init()

    if os.getenv("PROFILER") == "torchprof":
        import torch.profiler
        from prof_handler import trace_handler, TORCHPROF_OUTPUT_DIR
        import pathlib
        pathlib.Path(TORCHPROF_OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
        torch.profiler.tensorboard_trace_handler = lambda *a, **kw: trace_handler
        print(f"[Profiler] torchprof enabled, output dir: {TORCHPROF_OUTPUT_DIR}")

    trainer.run()


if __name__ == "__main__":
    main()

