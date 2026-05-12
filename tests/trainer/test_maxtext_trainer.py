###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

import pytest
from absl.testing import absltest

from tests.utils import PrimusUT, run_training_script

SKIP_TEST = os.getenv("JAX_SKIP_UT", "0") == "1"


def run_script(
    ut_name: str,
    tag: str,
    exp_path: str,
    env_override: dict = None,
    extra_args: list[str] = None,
):
    shell_entry = "examples/run_pretrain.sh"
    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    env["BACKEND"] = "MaxText"
    env["EXP"] = exp_path

    ut_log_path = os.environ.get("UT_LOG_PATH", "ut_out")
    train_log_path = os.path.join(ut_log_path, f"log.test_maxtext_trainer-{tag}.txt")
    env["TRAIN_LOG"] = train_log_path

    cmd = ["bash", shell_entry]
    if extra_args:
        cmd.extend(extra_args)

    return run_training_script(tag=tag, cmd=cmd, train_log_path=train_log_path, env=env)


class TestMaxTextTrainer(PrimusUT):
    def test_llama3_8B_BF16(self):
        run_script(
            self.__class__.__name__,
            "llama3_8B-BF16",
            exp_path="examples/maxtext/configs/MI300X/llama3_8B-pretrain.yaml",
            extra_args=[
                "--override_model.base_num_decoder_layers",
                "4",
                "--steps",
                "3",
            ],
        )

    def test_llama3_8B_FP8(self):
        run_script(
            self.__class__.__name__,
            "llama3_8B-FP8",
            exp_path="examples/maxtext/configs/MI300X/llama3_8B-pretrain.yaml",
            extra_args=[
                "--override_model.base_num_decoder_layers",
                "4",
                "--steps",
                "3",
                "--quantization",
                "nanoo_fp8",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST, reason="JAX_SKIP_UT=1, skipping")
    def test_llama3_70B_BF16(self):
        run_script(
            self.__class__.__name__,
            "llama3_70B-BF16",
            exp_path="examples/maxtext/configs/MI300X/llama3_70B-pretrain.yaml",
            extra_args=[
                "--override_model.base_num_decoder_layers",
                "4",
                "--steps",
                "3",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST, reason="JAX_SKIP_UT=1, skipping")
    def test_llama3_70B_FP8(self):
        run_script(
            self.__class__.__name__,
            "llama3_70B-FP8",
            exp_path="examples/maxtext/configs/MI300X/llama3_70B-pretrain.yaml",
            extra_args=[
                "--override_model.base_num_decoder_layers",
                "4",
                "--steps",
                "3",
                "--quantization",
                "nanoo_fp8",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST, reason="JAX_SKIP_UT=1, skipping")
    def test_llama3_3_70B_BF16(self):
        run_script(
            self.__class__.__name__,
            "llama3_3_70B-BF16",
            exp_path="examples/maxtext/configs/MI300X/llama3.3_70B-pretrain.yaml",
            extra_args=[
                "--override_model.base_num_decoder_layers",
                "4",
                "--steps",
                "3",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST, reason="JAX_SKIP_UT=1, skipping")
    def test_llama3_3_70B_FP8(self):
        run_script(
            self.__class__.__name__,
            "llama3_3_70B-FP8",
            exp_path="examples/maxtext/configs/MI300X/llama3.3_70B-pretrain.yaml",
            extra_args=[
                "--override_model.base_num_decoder_layers",
                "4",
                "--steps",
                "3",
                "--quantization",
                "nanoo_fp8",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST, reason="JAX_SKIP_UT=1, skipping")
    def test_llama2_7B_BF16(self):
        run_script(
            self.__class__.__name__,
            "llama2_7B-BF16",
            exp_path="examples/maxtext/configs/MI300X/llama2_7B-pretrain.yaml",
            extra_args=[
                "--override_model.base_num_decoder_layers",
                "4",
                "--steps",
                "3",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST, reason="JAX_SKIP_UT=1, skipping")
    def test_llama2_7B_FP8(self):
        run_script(
            self.__class__.__name__,
            "llama2_7B-FP8",
            exp_path="examples/maxtext/configs/MI300X/llama2_7B-pretrain.yaml",
            extra_args=[
                "--override_model.base_num_decoder_layers",
                "4",
                "--steps",
                "3",
                "--quantization",
                "nanoo_fp8",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST, reason="JAX_SKIP_UT=1, skipping")
    def test_llama2_70B_BF16(self):
        run_script(
            self.__class__.__name__,
            "llama2_70B-BF16",
            exp_path="examples/maxtext/configs/MI300X/llama2_70B-pretrain.yaml",
            extra_args=[
                "--override_model.base_num_decoder_layers",
                "4",
                "--steps",
                "3",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST, reason="JAX_SKIP_UT=1, skipping")
    def test_llama2_70B_FP8(self):
        run_script(
            self.__class__.__name__,
            "llama2_70B-FP8",
            exp_path="examples/maxtext/configs/MI300X/llama2_70B-pretrain.yaml",
            extra_args=[
                "--override_model.base_num_decoder_layers",
                "4",
                "--steps",
                "3",
                "--quantization",
                "nanoo_fp8",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST, reason="JAX_SKIP_UT=1, skipping")
    def test_mixtral_8x7B_BF16(self):
        run_script(
            self.__class__.__name__,
            "mixtral_8x7B-BF16",
            exp_path="examples/maxtext/configs/MI300X/mixtral_8x7B-pretrain.yaml",
            extra_args=[
                "--override_model.base_num_decoder_layers",
                "4",
                "--steps",
                "3",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST, reason="JAX_SKIP_UT=1, skipping")
    def test_mixtral_8x7B_FP8(self):
        run_script(
            self.__class__.__name__,
            "mixtral_8x7B-FP8",
            exp_path="examples/maxtext/configs/MI300X/mixtral_8x7B-pretrain.yaml",
            extra_args=[
                "--override_model.base_num_decoder_layers",
                "4",
                "--steps",
                "3",
                "--quantization",
                "nanoo_fp8",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST, reason="JAX_SKIP_UT=1, skipping")
    def test_grok1_BF16(self):
        run_script(
            self.__class__.__name__,
            "grok1-BF16",
            exp_path="examples/maxtext/configs/MI300X/grok1-pretrain.yaml",
            extra_args=[
                "--override_model.base_num_decoder_layers",
                "4",
                "--steps",
                "3",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST, reason="JAX_SKIP_UT=1, skipping")
    def test_grok1_FP8(self):
        run_script(
            self.__class__.__name__,
            "grok1-FP8",
            exp_path="examples/maxtext/configs/MI300X/grok1-pretrain.yaml",
            extra_args=[
                "--override_model.base_num_decoder_layers",
                "4",
                "--steps",
                "3",
                "--quantization",
                "nanoo_fp8",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST, reason="JAX_SKIP_UT=1, skipping")
    def test_dpsk_v2_16B_BF16(self):
        run_script(
            self.__class__.__name__,
            "dpsk_v2_16B-BF16",
            exp_path="examples/maxtext/configs/MI300X/deepseek_v2_16B-pretrain.yaml",
            extra_args=[
                "--override_model.base_num_decoder_layers",
                "4",
                "--steps",
                "3",
            ],
        )

    @pytest.mark.skipif(SKIP_TEST, reason="JAX_SKIP_UT=1, skipping")
    def test_dpsk_v2_16B_FP8(self):
        run_script(
            self.__class__.__name__,
            "dpsk_v2_16B-FP8",
            exp_path="examples/maxtext/configs/MI300X/deepseek_v2_16B-pretrain.yaml",
            extra_args=[
                "--override_model.base_num_decoder_layers",
                "4",
                "--steps",
                "3",
                "--quantization",
                "nanoo_fp8",
            ],
        )


if __name__ == "__main__":
    absltest.main()
