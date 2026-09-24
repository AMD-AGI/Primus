###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import copy
import json
import logging
import os
import shutil
import subprocess

import yaml
from ckpt_report import get_ckpt_report

CURDIR = os.getcwd()
logger = logging.getLogger(__name__)


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Launch traning task and save checkpoint")
    parser.add_argument(
        "--yaml-config-path",
        type=str,
        required=True,
        help="Primus pretrainer config yaml path",
    )
    parser.add_argument(
        "--nnodes", type=int, default=1, help="Number of nodes to run training on (default: 1)"
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
    )
    parser.add_argument(
        "--expert-model-parallel-size",
        type=int,
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--ckpt-format",
        type=str,
    )
    parser.add_argument(
        "--async-save",
        action="store_true",
    )
    parser.add_argument(
        "--ckpt-fully-parallel-save",
        action="store_true",
    )
    parser.add_argument(
        "--ckpt-fully-parallel-load",
        action="store_true",
    )
    parser.add_argument("--no-remove-outputs", action="store_false", dest="remove_outputs")
    args = parser.parse_args()
    return args


def get_new_yaml_config(args, yaml_config):
    new_yaml_config = copy.deepcopy(yaml_config)
    config = new_yaml_config["modules"]["pre_trainer"]["overrides"]

    if args.tensor_model_parallel_size:
        config["tensor_model_parallel_size"] = args.tensor_model_parallel_size
    if args.pipeline_model_parallel_size:
        config["pipeline_model_parallel_size"] = args.pipeline_model_parallel_size
    if args.expert_model_parallel_size:
        config["expert_model_parallel_size"] = args.expert_model_parallel_size
    if args.train_iters:
        config["train_iters"] = args.train_iters
    if args.save_interval:
        config["save_interval"] = args.save_interval
    if args.ckpt_format:
        config["ckpt_format"] = args.ckpt_format
    if args.async_save:
        config["async_save"] = args.async_save
    if args.ckpt_fully_parallel_save:
        config["ckpt_fully_parallel_save"] = args.ckpt_fully_parallel_save
    if args.ckpt_fully_parallel_load:
        config["ckpt_fully_parallel_load"] = args.ckpt_fully_parallel_load

    config["no_save_rng"] = None
    config["no_save_optim"] = None
    config["disable_last_saving"] = True
    config["auto_continue_train"] = False
    config["finetune"] = False
    return new_yaml_config


def get_output_dir(yaml_config):
    def find_output_dir(root_dir: str) -> str:
        for dirpath, dirnames, _ in os.walk(root_dir):
            if "checkpoints" in dirnames:
                return os.path.abspath(dirpath)

    workspace_dir = os.path.join(CURDIR, yaml_config["workspace"])
    output_dir = find_output_dir(workspace_dir)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    logs_dir = os.path.join(output_dir, "logs", "pre_trainer")
    logger.debug(f"checkpoints_dir={checkpoints_dir}")
    logger.debug(f"logs_dir={logs_dir}")
    return (checkpoints_dir, logs_dir)


def train_with_yaml_config(args, yaml_config):
    def run_training_subprocess(shared_fs_path="."):
        NEW_YAML_FILE = "ckpt_training_config.yaml"
        new_yaml_config_path = os.path.join(shared_fs_path, NEW_YAML_FILE)
        logger.debug(f"new_yaml_config_path={new_yaml_config_path}")

        env = os.environ.copy()
        overwritten_env = {
            "EXP": new_yaml_config_path,
            "NNODES": f"{args.nnodes}",
            "BACKEND": "megatron",
        }
        env.update(overwritten_env)
        with open(new_yaml_config_path, "w") as f:
            yaml.dump(yaml_config, f)

        command = "bash runner/helpers/launch/slurm_pretrain.sh"
        result = subprocess.run(command, shell=True, capture_output=True, text=True, env=env)
        logger.debug(f"training subprocess stdout : {result.stdout}")
        logger.debug(f"training subprocess stderr : {result.stderr}")
        return result.returncode

    # launch training process for checkpoint save
    exit_code = run_training_subprocess()
    logger.info(f"checkpoint save, training process exit code : {exit_code}")

    # launch training process for checkpoint load
    checkpoint_dir, logs_dir = get_output_dir(yaml_config)
    yaml_config["modules"]["pre_trainer"]["overrides"]["load"] = checkpoint_dir
    exit_code = run_training_subprocess()
    logger.info(f"checkpoint load, training process exit code : {exit_code}")


def print_checkpoint_report(args, yaml_config):
    checkpoints_dir, logs_dir = get_output_dir(yaml_config)
    report_dict = get_ckpt_report(logs_dir, checkpoints_dir)
    report_json = json.dumps(report_dict, indent=4, ensure_ascii=False)
    logger.info(report_json)
    if args.remove_outputs:
        # may need root permission
        shutil.rmtree(checkpoints_dir, ignore_errors=True)
        shutil.rmtree(logs_dir, ignore_errors=True)


def main(args):
    logger.debug(args)
    with open(args.yaml_config_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    new_yaml_config = get_new_yaml_config(args, yaml_config)
    train_with_yaml_config(args, new_yaml_config)
    print_checkpoint_report(args, new_yaml_config)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s][%(levelname)s] - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    args = parse_cli_args()
    main(args)
