###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import subprocess
import sys
import time

from primus.core.utils import logger
from tests.utils import PrimusUT


def run_script(
    ut_name: str,
    tag: str,
    exp_path: str,
    env_override: dict = None,
    extra_args: list[str] = None,
):
    shell_entry = "./runner/primus-cli"
    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    env["EXP"] = exp_path

    ut_log_path = os.environ.get("UT_LOG_PATH", "ut_out")
    train_log_path = os.path.join(ut_log_path, f"log.test_torchtitan_trainer-{tag}.txt")
    env["TRAIN_LOG"] = train_log_path

    do_print_at_runtime = True
    run_stdout = subprocess.PIPE if not do_print_at_runtime else sys.stdout
    run_stderr = subprocess.PIPE if not do_print_at_runtime else sys.stderr

    cmd = [
        "bash",
        shell_entry,
        "direct",
        "--log_file",
        train_log_path,
        "--",
        "train",
        "pretrain",
        "--config",
        exp_path,
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        logger.info(f"[{tag}] Begin Titan run...")
        start = time.time()
        subprocess.run(
            cmd,
            check=True,
            stdout=run_stdout,
            stderr=run_stderr,
            text=True,
            env=env,
        )
        logger.info(f"[{tag}] End run, time={time.time() - start:.3f} s")

        stdout_output = ""
        if os.path.exists(train_log_path):
            with open(train_log_path, "r") as f:
                stdout_output = f.read()

        return stdout_output, ""

    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr or ""
        stdout_output = e.stdout or ""
        if os.path.exists(train_log_path):
            try:
                with open(train_log_path, "r") as f:
                    stdout_output = f.read()
            except Exception as log_err:
                logger.warning(f"[{tag}] Failed to read train log: {log_err}")

        if "Training completed." in stdout_output:
            logger.warning(f"[{tag}] Training likely succeeded despite return code != 0.")
            logger.warning(f"stderr excerpt:\n{stderr_output[:1000]}")
        else:
            raise AssertionError(f"Shell script failed: {stderr_output.strip()}")

    return stdout_output, stderr_output


class TestTorchTitanTrainer(PrimusUT):
    def test_llama3_1_8B_BF16(self):
        run_script(
            self.__class__.__name__,
            "llama3_8B-BF16",
            exp_path="examples/torchtitan/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml",
            extra_args=[
                "--model.n_layers",
                "4",
                "--training.steps",
                "3",
                "--training.mock_data",
                "True",
            ],
        )

    def test_llama3_1_8B_FP8(self):
        run_script(
            self.__class__.__name__,
            "llama3_8B-FP8",
            exp_path="examples/torchtitan/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml",
            extra_args=[
                "--model.n_layers",
                "4",
                "--training.steps",
                "3",
                "--training.mock_data",
                "True",
            ],
        )

    def test_llama3_1_405B_bf16(self):
        run_script(
            self.__class__.__name__,
            "llama3.1_405B_bf16",
            "examples/torchtitan/configs/MI300X/llama3.1_405B-BF16-pretrain.yaml",
            extra_args=[
                "--model.n_layers",
                "4",
                "--training.steps",
                "3",
                "--training.mock_data",
                "True",
            ],
        )

    def test_llama3_1_405B_fp8(self):
        run_script(
            self.__class__.__name__,
            "llama3.1_405B_fp8",
            "examples/torchtitan/configs/MI300X/llama3.1_405B-FP8-pretrain.yaml",
            extra_args=[
                "--model.n_layers",
                "4",
                "--training.steps",
                "3",
                "--training.mock_data",
                "True",
            ],
        )

    def test_llama3_1_70B_bf16(self):
        run_script(
            self.__class__.__name__,
            "llama3.1_70B_bf16",
            "examples/torchtitan/configs/MI300X/llama3.1_70B-BF16-pretrain.yaml",
            extra_args=[
                "--model.n_layers",
                "4",
                "--training.steps",
                "3",
                "--training.mock_data",
                "True",
            ],
        )

    def test_llama3_1_70B_fp8(self):
        run_script(
            self.__class__.__name__,
            "llama3.1_70B_fp8",
            "examples/torchtitan/configs/MI300X/llama3.1_70B-FP8-pretrain.yaml",
            extra_args=[
                "--model.n_layers",
                "4",
                "--training.steps",
                "3",
                "--training.mock_data",
                "True",
            ],
        )

    def test_qwen3_0_6B(self):
        run_script(
            self.__class__.__name__,
            "qwen3_0.6B",
            "examples/torchtitan/configs/MI300X/qwen3_0.6B-pretrain.yaml",
            extra_args=[
                "--model.n_layers",
                "4",
                "--training.steps",
                "3",
                "--training.mock_data",
                "True",
            ],
        )

    def test_qwen3_1_7B(self):
        run_script(
            self.__class__.__name__,
            "qwen3_1.7B",
            "examples/torchtitan/configs/MI300X/qwen3_1.7B-pretrain.yaml",
            extra_args=[
                "--model.n_layers",
                "4",
                "--training.steps",
                "3",
                "--training.mock_data",
                "True",
            ],
        )

    def test_qwen3_32B(self):
        run_script(
            self.__class__.__name__,
            "qwen3_32B",
            "examples/torchtitan/configs/MI300X/qwen3_32B-pretrain.yaml",
            extra_args=[
                "--model.n_layers",
                "4",
                "--training.steps",
                "3",
                "--training.mock_data",
                "True",
            ],
        )

    def test_deepseek_v3_16b(self):
        run_script(
            self.__class__.__name__,
            "deepseek_v3_16b",
            "examples/torchtitan/configs/MI300X/deepseek_v3_16b-BF16-pretrain.yaml",
            extra_args=[
                "--model.n_layers",
                "4",
                "--model.n_dense_layers",
                "1",
                "--training.steps",
                "3",
                "--training.mock_data",
                "True",
            ],
        )

    def test_deepseek_v3_16b_fp8(self):
        run_script(
            self.__class__.__name__,
            "deepseek_v3_16b_fp8",
            "examples/torchtitan/configs/MI300X/deepseek_v3_16b-FP8-pretrain.yaml",
            extra_args=[
                "--model.n_layers",
                "4",
                "--model.n_dense_layers",
                "1",
                "--training.steps",
                "3",
                "--training.mock_data",
                "True",
            ],
        )

    def test_deepseek_v3_671b(self):
        run_script(
            self.__class__.__name__,
            "deepseek_v3_671b",
            "examples/torchtitan/configs/MI300X/deepseek_v3_671b-pretrain.yaml",
            extra_args=[
                "--model.n_layers",
                "4",
                "--model.n_dense_layers",
                "1",
                "--training.steps",
                "3",
                "--training.mock_data",
                "True",
            ],
        )
