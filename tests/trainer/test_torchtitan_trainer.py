###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

import pytest

from tests.utils import PrimusUT, run_training_script


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

    return run_training_script(tag=tag, cmd=cmd, train_log_path=train_log_path, env=env)


class TestTorchTitanTrainer(PrimusUT):
    """One config per distinct code path; isomorphic flavors are not tested here.

    Every case is a full training launch, so a model whose code path a sibling
    already covers earns no coverage. What is kept and why:

    - llama3.1_8B BF16 + FP8: the dense llama path, both precisions.
    - qwen3_0.6B: the qwen3 dense path (activation_checkpoint off).
    - qwen3_32B: the dense config with activation_checkpoint.mode=full.
    - deepseek_v3_16b: MLA + MoE on classic attention, no float8.
    - deepseek_v3_16b_fp8: the only use_moe_fp8 path; also swaps classic
      attention for the turbo one.
    - deepseek_v3_671b: MLA + MoE under full recompute and
      use_turbo_float8_linear.
    - gpt_oss_20B BF16 + FP8: MoE + sink attention, selective checkpointing.

    llama3.1_70B / llama3.1_405B (both precisions) and qwen3_1.7B used to be
    here; they only scaled the dims of the above. Their recipes are still
    schema-checked by tests/unit_tests/configs/test_example_configs.py.
    """

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

    def test_gpt_oss_20B(self):
        # Default Primus-Turbo path: GPT-OSS sink attention (flash_attn_func).
        run_script(
            self.__class__.__name__,
            "gpt_oss_20B",
            "examples/torchtitan/configs/MI300X/gpt_oss_20B-BF16-pretrain.yaml",
            extra_args=[
                "--model.n_layers",
                "4",
                "--training.steps",
                "3",
                "--training.mock_data",
                "True",
            ],
        )

    def test_gpt_oss_20B_fp8(self):
        run_script(
            self.__class__.__name__,
            "gpt_oss_20B_fp8",
            "examples/torchtitan/configs/MI300X/gpt_oss_20B-FP8-pretrain.yaml",
            extra_args=[
                "--model.n_layers",
                "4",
                "--training.steps",
                "3",
                "--training.mock_data",
                "True",
            ],
        )

    # Llama 4 had no E2E in either backend. Distinct from the llama3 flavor above:
    # MoE with a shared expert, and the recipe's use_turbo_grouped_gemm path.
    @pytest.mark.weekly
    def test_llama4_17Bx16E(self):
        run_script(
            self.__class__.__name__,
            "llama4_17Bx16E",
            "examples/torchtitan/configs/MI300X/llama4_17Bx16E-BF16-pretrain.yaml",
            extra_args=[
                "--model.n_layers",
                "4",
                "--training.steps",
                "3",
                "--training.mock_data",
                "True",
            ],
        )
