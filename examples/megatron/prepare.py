###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os
import subprocess
import time
from pathlib import Path
from time import sleep

import nltk
from datasets import load_dataset

from examples.scripts.utils import (
    get_env_case_insensitive,
    get_node_rank,
    log_error_and_exit,
    log_info,
    write_patch_args,
)
from primus.core.launcher.config import PrimusConfig
from primus.core.launcher.parser import load_primus_config
from primus.pretrain import setup_backend_path


# ---------- Helpers ----------
def check_dir_nonempty(path: Path, name: str):
    if not path.is_dir() or not any(path.iterdir()):
        log_error_and_exit(
            f"{name} ({path}) does not exist or is empty.\n"
            "Please ensure Primus is properly initialized.\n"
            "If not yet cloned, run:\n"
            "    git clone --recurse-submodules git@github.com:AMD-AIG-AIMA/Primus.git\n"
            "Or if already cloned, initialize submodules with:\n"
            "    git submodule update --init --recursive"
        )


def prepare_dataset(
    primus_path: Path,
    data_path: Path,
    tokenizer_type: str,
    tokenizer_model: str,
    tokenized_train_data_path: Path,
    tokenized_eval_data_path: Path,
    test_size: float = 0.005,
    seed: int = 42,
    env=None,
):
    dataset = "bookcorpus"
    dataset_path = data_path / dataset
    output_path = dataset_path / tokenizer_type
    hf_home = Path(os.environ.get("HF_HOME", data_path / "huggingface"))
    os.environ["HF_HOME"] = str(hf_home)

    train_tokenized_bin = tokenized_train_data_path.with_suffix(".bin")
    train_tokenized_idx = tokenized_train_data_path.with_suffix(".idx")
    train_files_exist = train_tokenized_bin.exists() and train_tokenized_idx.exists()

    eval_files_exist = False
    if tokenized_eval_data_path is not None:
        eval_tokenized_bin = tokenized_eval_data_path.with_suffix(".bin")
        eval_tokenized_idx = tokenized_eval_data_path.with_suffix(".idx")
        eval_files_exist = eval_tokenized_bin.exists() and eval_tokenized_idx.exists()

    # check if all required files exist
    if train_files_exist:
        if tokenized_eval_data_path is None or eval_files_exist:
            log_info(f"All required tokenized files exist. Skipping preprocessing.")
            return
        else:
            log_info(
                f"Train files exist, but evaluation files are missing. Will generate evaluation files only."
            )

    output_path.mkdir(parents=True, exist_ok=True)
    train_json = dataset_path / "bookcorpus_train.json"
    valid_json = dataset_path / "bookcorpus_valid.json"

    if train_json.exists() and valid_json.exists():
        log_info(
            f"Found train dataset file: {train_json} and valid dataset file: {valid_json}, skipping download."
        )
    else:
        log_info(
            f"Downloading and saving BookCorpus train dataset to {train_json} and valid dataset to {valid_json} ..."
        )
        nltk.download("punkt")
        dataset = load_dataset("bookcorpus", split="train", trust_remote_code=True)
        # split train / valid
        splits = dataset.train_test_split(test_size=test_size, seed=seed)
        train_ds = splits["train"]
        valid_ds = splits["test"]

        train_ds.to_json(str(train_json))
        valid_ds.to_json(str(valid_json))
        log_info("Download and save train and valid dataset completed.")

    log_info(f"Preprocessing dataset with tokenizer {tokenizer_type} / {tokenizer_model}")
    start = time.time()

    # process train dataset (only if not exists)
    if not train_files_exist:
        subprocess.run(
            [
                "python3",
                str(primus_path / "examples/megatron/preprocess_data.py"),
                "--input",
                str(train_json),
                "--tokenizer-type",
                tokenizer_type,
                "--tokenizer-model",
                tokenizer_model,
                "--output-prefix",
                str(output_path / "bookcorpus_train"),
                "--workers",
                str(os.cpu_count()),
                "--split-sentences",
                "--partitions",
                "2",
            ],
            check=True,
            env=env,
        )
        log_info(f"Train dataset preprocessing completed in {int(time.time() - start)} s")
    else:
        log_info(f"Train dataset files already exist, skipping train preprocessing.")

    if tokenized_eval_data_path is not None and not eval_files_exist:
        # process valid data
        subprocess.run(
            [
                "python3",
                str(primus_path / "examples/megatron/preprocess_data.py"),
                "--input",
                str(valid_json),
                "--tokenizer-type",
                tokenizer_type,
                "--tokenizer-model",
                tokenizer_model,
                "--output-prefix",
                str(output_path / "bookcorpus_eval"),
                "--workers",
                str(os.cpu_count()),
                "--split-sentences",
                "--partitions",
                "2",
            ],
            check=True,
            env=env,
        )
        log_info(f"Valid dataset preprocessing completed in {int(time.time() - start)} s")


def prepare_dataset_if_needed(
    primus_config: PrimusConfig,
    primus_path: Path,
    data_path: Path,
    patch_args: Path,
    test_size: float = 0.005,
    seed: int = 42,
    env=None,
):
    pre_trainer_cfg = primus_config.get_module_config("pre_trainer")
    if pre_trainer_cfg.train_data_path is not None:
        return

    tokenizer_type = pre_trainer_cfg.tokenizer_type
    if (
        pre_trainer_cfg.full_validation or pre_trainer_cfg.eval_iters > 0
    ) and pre_trainer_cfg.eval_interval > 0:
        default_eval_tokenized_path = (
            Path(data_path) / f"bookcorpus/{tokenizer_type}/bookcorpus_eval_text_sentence"
        )
        tokenized_eval_data_path = Path(
            os.environ.get("TOKENIZED_EVAL_DATA_PATH", str(default_eval_tokenized_path))
        )
    else:
        tokenized_eval_data_path = None

    default_train_tokenized_path = (
        Path(data_path) / f"bookcorpus/{tokenizer_type}/bookcorpus_train_text_sentence"
    )
    tokenized_train_data_path = Path(
        os.environ.get("TOKENIZED_TRAIN_DATA_PATH", str(default_train_tokenized_path))
    )

    if tokenized_eval_data_path is not None:
        done_flag = tokenized_eval_data_path.with_suffix(".done")
    else:
        done_flag = tokenized_train_data_path.with_suffix(".done")

    node_rank = get_node_rank()

    if node_rank == 0:
        if not done_flag.exists():
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                log_error_and_exit("Environment variable HF_TOKEN must be set.")

            tokenizer_type = primus_config.get_module_config("pre_trainer").tokenizer_type
            tokenizer_model = primus_config.get_module_config("pre_trainer").tokenizer_model

            log_info(
                f"TOKENIZED_TRAIN_DATA_PATH is {tokenized_train_data_path}, TOKENIZED_EVAL_DATA_PATH is {tokenized_eval_data_path}"
            )

            prepare_dataset(
                primus_path=primus_path,
                data_path=data_path,
                tokenizer_type=tokenizer_type,
                tokenizer_model=tokenizer_model,
                tokenized_train_data_path=tokenized_train_data_path,
                tokenized_eval_data_path=tokenized_eval_data_path,
                test_size=test_size,
                seed=seed,
                env=env,
            )
            done_flag.touch()
            log_info("Dataset preparation completed.")
    else:
        while not done_flag.exists():
            log_info("Waiting for dataset...")
            sleep(30)

    train_args = {
        "train_data_path": str(tokenized_train_data_path),
    }
    if tokenized_eval_data_path is not None:
        train_args["valid_data_path"] = str(tokenized_eval_data_path)
        train_args["test_data_path"] = str(tokenized_eval_data_path)
    write_patch_args(Path(patch_args), "train_args", train_args)


def build_megatron_helper(primus_path: Path, patch_args: Path, backend_path: str = None):
    """Build Megatron's helper C++ dataset library."""
    if backend_path:
        megatron_path = Path(backend_path).resolve()
        log_info(f"Using backend_path from argument: {megatron_path}")
    else:
        # megatron_path = primus_path / "third_party/megatron"
        # log_info(f"No backend_path provided, falling back to: {megatron_path}")
        env_backend = get_env_case_insensitive("MEGATRON_PATH")
        if env_backend:
            megatron_path = Path(env_backend).resolve()
            log_info(f"Using backend_path from environment: {megatron_path}")
        else:
            megatron_path = primus_path / "third_party/Megatron-LM"
            log_info(f"No backend_path provided, falling back to: {megatron_path}")
    write_patch_args(Path(patch_args), "train_args", {"backend_path": str(megatron_path)})

    check_dir_nonempty(megatron_path, "megatron")

    # build C++ helper
    dataset_cpp_dir = megatron_path / "megatron/core/datasets"
    log_info(f"Building Megatron dataset helper in {dataset_cpp_dir}")

    ret = subprocess.run(["make"], cwd=dataset_cpp_dir)
    if ret.returncode != 0:
        log_error_and_exit("Building Megatron C++ helper failed.")

    if os.environ.get("SKIP_EMERGING_OPTIMIZERS", "0") != "1":
        emerging_optimizers_path = primus_path / "third_party/Emerging-Optimizers"
        log_info(f"Building Emerging Optimizers in {emerging_optimizers_path}")
        ret = subprocess.run(["pip", "install", "-e", str(emerging_optimizers_path)], check=True)
        if ret.returncode != 0:
            log_error_and_exit("Building Emerging Optimizers failed.")
    else:
        log_info("Skipping Emerging Optimizers installation (SKIP_EMERGING_OPTIMIZERS=1)")


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Prepare Primus environment")
    parser.add_argument("--primus_path", type=str, required=True, help="Root path to the Primus project")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--test_size", type=float, default=0.005, help="Test size for dataset split")
    parser.add_argument("--seed", type=int, default=42, help="Seed for dataset split")
    parser.add_argument(
        "--patch_args",
        type=str,
        default="/tmp/primus_patch_args.txt",
        help="Path to write additional args (used during training phase)",
    )
    parser.add_argument(
        "--backend_path",
        type=str,
        default=None,
        help="Optional path to backend (e.g., Megatron), will be added to PYTHONPATH",
    )
    args, unknown = parser.parse_known_args()

    log_info(f"BACKEND_PATH {args.backend_path}")
    # primus_config = PrimusParser().parse(args)
    primus_config, _ = load_primus_config(args, unknown)

    primus_path = Path(args.primus_path).resolve()
    log_info(f"PRIMUS_PATH is set to: {primus_path}")

    data_path = Path(args.data_path).resolve()
    log_info(f"DATA_PATH is set to: {data_path}")

    exp_path = Path(args.config).resolve()
    if not exp_path.is_file():
        log_error_and_exit(f"The specified EXP file does not exist: {exp_path}")
    log_info(f"EXP is set to: {exp_path}")

    patch_args_file = Path(args.patch_args).resolve()
    log_info(f"PATCH-ARGS is set to: {patch_args_file}")

    used_backend_path = setup_backend_path(framework="megatron", backend_path=args.backend_path, verbose=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{used_backend_path}:{env.get('PYTHONPATH', '')}"

    mock_data = primus_config.get_module_config("pre_trainer").mock_data
    if mock_data:
        log_info(f"'mock_data: true', Skipping dataset preparation.")
    else:
        prepare_dataset_if_needed(
            primus_config=primus_config,
            primus_path=primus_path,
            data_path=data_path,
            patch_args=patch_args_file,
            test_size=args.test_size,
            seed=args.seed,
            env=env,
        )

    build_megatron_helper(primus_path=primus_path, backend_path=args.backend_path, patch_args=patch_args_file)


if __name__ == "__main__":
    log_info("========== Prepare Megatron dataset ==========")
    main()
