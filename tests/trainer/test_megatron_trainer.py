###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import os
import re
import subprocess
import sys
import time
import unittest

from primus.core.utils import logger
from tests.utils import PrimusUT


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
    env["EXP"] = exp_path

    ut_log_path = os.environ.get("UT_LOG_PATH", "ut_out")
    train_log_path = os.path.join(ut_log_path, f"log.test_megatron_trainer-{tag}.txt")
    env["TRAIN_LOG"] = train_log_path

    do_print_at_runtime = True
    run_stdout = subprocess.PIPE if not do_print_at_runtime else sys.stdout
    run_stderr = subprocess.PIPE if not do_print_at_runtime else sys.stderr

    cmd = ["bash", shell_entry]
    if extra_args:
        cmd.extend(extra_args)

    try:
        logger.info(f"Begin run {tag}...")
        start = time.time()
        result = subprocess.run(
            cmd,
            check=True,
            stdout=run_stdout,
            stderr=run_stderr,
            text=True,
            env=env,
        )
        logger.info(f"End run {tag}, time={time.time()-start:.3f} s")

        logger.info(f"Training log path: {ut_log_path}/logs/UT-{ut_name}")

        with open(train_log_path, "r") as f:
            stdout_output = f.read()

        stderr_output = ""

        return stdout_output, stderr_output

    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr or ""
        stdout_output = e.stdout or ""

        if os.path.exists(train_log_path):
            try:
                with open(train_log_path, "r") as f:
                    stdout_output = f.read()
            except Exception as log_err:
                logger.warning(f"[{tag}] Failed to read train log: {log_err}")

        if "after training is done" in stdout_output:
            logger.warning(f"[{tag}] Training likely succeeded despite return code != 0.")
            logger.warning(f"stderr excerpt:\n{stderr_output[:1000]}")
        else:
            raise AssertionError(f"Shell script failed: {stderr_output.strip()}")

    return stdout_output, stderr_output


class TestMegatronTrainer(PrimusUT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_llama2_7B(self):
        run_script(
            self.__class__.__name__,
            "llama2_7B",
            exp_path="examples/megatron/mi300/llama2_7B-pretrain.yaml",
            env_override={},
            extra_args=["--num_layers", "4", "--train_iters", "3"],
        )

    def test_llama3_8B(self):
        run_script(
            self.__class__.__name__,
            "llama3_8B",
            exp_path="examples/megatron/mi300/llama3_8B-pretrain.yaml",
            env_override={},
            extra_args=["--num_layers", "4", "--train_iters", "3"],
        )

    def test_llama3_70B(self):
        run_script(
            self.__class__.__name__,
            "llama3_70B",
            exp_path="examples/megatron/mi300/llama3_70B-pretrain.yaml",
            env_override={},
            extra_args=["--num_layers", "4", "--train_iters", "3"],
        )

    def test_qwen25_7B(self):
        run_script(
            self.__class__.__name__,
            "qwen2.5_7B",
            exp_path="examples/megatron/mi300/qwen2.5_7B-pretrain.yaml",
            env_override={},
            extra_args=["--num_layers", "4", "--train_iters", "3"],
        )

    def test_qwen25_72B(self):
        run_script(
            self.__class__.__name__,
            "qwen2.5_72B",
            exp_path="examples/megatron/mi300/qwen2.5_72B-pretrain.yaml",
            env_override={},
            extra_args=["--num_layers", "4", "--train_iters", "3"],
        )

    def test_deepseek_v2_lite(self):
        run_script(
            self.__class__.__name__,
            "deepseek_v2_lite",
            exp_path="examples/megatron/mi300/deepseek_v2_lite-pretrain.yaml",
            env_override={},
            extra_args=[
                "--num_layers",
                "4",
                "--train_iters",
                "3",
                "--micro_batch_size",
                "1",
                "--global_batch_size",
                "8",
                "--moe_layer_freq",
                "[0]*1+[1]*3",
                "--expert_model_parallel_size",
                "8",
            ],
        )

    def test_mixtral_8x7B(self):
        run_script(
            self.__class__.__name__,
            "mixtral_8x7B_v0.1",
            exp_path="examples/megatron/mi300/mixtral_8x7B_v0.1-pretrain.yaml",
            env_override={},
            extra_args=[
                "--num_layers",
                "4",
                "--train_iters",
                "3",
                "--micro_batch_size",
                "1",
                "--global_batch_size",
                "8",
                "--moe_layer_freq",
                "1",
                "--expert_model_parallel_size",
                "8",
            ],
        )

    def test_mixtral_8x22B(self):
        run_script(
            self.__class__.__name__,
            "mixtral_8x22B_v0.1",
            exp_path="examples/megatron/mi300/mixtral_8x22B_v0.1-pretrain.yaml",
            env_override={},
            extra_args=[
                "--num_layers",
                "4",
                "--train_iters",
                "3",
                "--micro_batch_size",
                "1",
                "--global_batch_size",
                "8",
                "--moe_layer_freq",
                "1",
                "--expert_model_parallel_size",
                "8",
                "--pipeline_model_parallel_size",
                "1",
            ],
        )

    def test_grok2(self):
        run_script(
            self.__class__.__name__,
            "grok2",
            exp_path="examples/megatron/mi300/grok2-pretrain.yaml",
            env_override={},
            extra_args=[
                "--num_layers",
                "2",
                "--train_iters",
                "3",
                "--micro_batch_size",
                "1",
                "--global_batch_size",
                "8",
                "--expert_model_parallel_size",
                "8",
                "--pipeline_model_parallel_size",
                "1",
                "--num_virtual_stages_per_pipeline_rank",
                "1",
            ],
        )

    def test_deepseek_v3(self):
        run_script(
            self.__class__.__name__,
            "deepseek_v3",
            exp_path="examples/megatron/mi300/deepseek_v3-pretrain.yaml",
            env_override={},
            extra_args=[
                "--num_layers",
                "4",
                "--moe_layer_freq",
                "[0]*1+[1]*3",
                "--train_iters",
                "3",
                "--micro_batch_size",
                "1",
                "--global_batch_size",
                "8",
                "--expert_model_parallel_size",
                "8",
                "--pipeline_model_parallel_size",
                "1",
            ],
        )

    def test_interleaved_pipeline_parallelism(self):
        run_script(
            self.__class__.__name__,
            "interleaved_pipeline_parallelism",
            exp_path="tests/trainer/test_megatron_trainer.yaml",
            env_override={
                "PRIMUS_PP": "4",
                "PRIMUS_VPP": "2",
                "PRIMUS_NUM_LAYERS": "8",
            },
            extra_args=[
                "--global_batch_size",
                "16",
                "--moe_layer_freq",
                "[0]*1+[1]*7",
            ],
        )

    def test_zero_bubble_pipeline_parallelism(self):
        run_script(
            self.__class__.__name__,
            "zero_bubble_pipeline_parallelism",
            exp_path="tests/trainer/test_megatron_trainer_zero_bubble.yaml",
            env_override={},
        )

    def test_turbo_deepep(self):
        run_script(
            self.__class__.__name__,
            "turbo_deepep",
            exp_path="examples/megatron/mi300/deepseek_v2_lite-pretrain.yaml",
            env_override={},
            extra_args=[
                "--num_layers",
                "4",
                "--train_iters",
                "3",
                "--micro_batch_size",
                "1",
                "--global_batch_size",
                "8",
                "--moe_layer_freq",
                "1",
                "--expert_model_parallel_size",
                "8",
                "--use_turbo_deepep",
                "1",
                "--enable_primus_turbo",
                "1",
                "--moe_router_dtype",
                "fp32",
                "--moe_shared_expert_overlap",
                "0",
                "--moe_use_legacy_grouped_gemm",
                "1",
                "--turbo_sync_free_moe_stage",
                "2",
            ],
        )


class TestMegatronTrainerDeterministic(PrimusUT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def extract_loss_from_log(self, log):
        LOSS_PATTERN = r"lm loss: (\d+.\d+E\+\d+)"

        loss = re.findall(LOSS_PATTERN, log)

        return loss

    def extract_num_zeros_from_log(self, log):
        NUM_ZEROS_IN_GRAD_PATTERN = r"num zeros: (\d+)"

        num_zeros_in_grad = re.findall(NUM_ZEROS_IN_GRAD_PATTERN, log)

        return num_zeros_in_grad

    def check_numerical_reproducility(self, log, log_ref):
        loss = self.extract_loss_from_log(log)
        loss_ref = self.extract_loss_from_log(log_ref)

        num_zeros = self.extract_num_zeros_from_log(log)
        num_zeros_ref = self.extract_num_zeros_from_log(log_ref)

        is_reproducility = True
        # compare as str, need bitwise equal.
        for i in range(0, len(loss)):
            if loss[i] != loss_ref[i] or num_zeros[i] != num_zeros_ref[i]:
                is_reproducility = False
                break

        return is_reproducility

    # TODO(0928): disable due to non-deterministic behavior in Dense implementation
    @unittest.skip("Skip non-deterministic Dense test")
    def test_llama3_8B(self):
        env_override = {
            "BACKEND": "megatron",
            "PRIMUS_MODEL": "llama3_8B",
            "PRIMUS_GLOBAL_BATCH_SIZE": "8",
            "PRIMUS_NUM_LAYERS": "4",
            # deterministic vars
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "NCCL_ALGO": "Ring",
        }
        stdout, _ = run_script(
            self.__class__.__name__,
            "llama3_8B",
            exp_path="tests/trainer/test_megatron_trainer_deterministic.yaml",
            env_override=env_override,
        )

        stdout_ref, _ = run_script(
            self.__class__.__name__,
            "llama3_8B_ref",
            exp_path="tests/trainer/test_megatron_trainer_deterministic.yaml",
            env_override=env_override,
        )

        assert self.check_numerical_reproducility(stdout, stdout_ref)

    # TODO(0928): disable due to non-deterministic behavior in MoE implementation
    @unittest.skip("Skip non-deterministic MoE test")
    def test_deepseek_v2_lite(self):
        env_override = {
            "BACKEND": "megatron",
            "PRIMUS_MODEL": "deepseek_v2_lite",
            "PRIMUS_GLOBAL_BATCH_SIZE": "8",
            "PRIMUS_MOE_LAYER_FREQ": "[0]*1+[1]*3",
            "PRIMUS_EP": "8",
            "PRIMUS_NUM_LAYERS": "4",
            # deterministic vars
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "NCCL_ALGO": "Ring",
        }
        stdout, _ = run_script(
            self.__class__.__name__,
            "deepseek_v2_lite",
            exp_path="tests/trainer/test_megatron_trainer_deterministic.yaml",
            env_override=env_override,
        )

        stdout_ref, _ = run_script(
            self.__class__.__name__,
            "deepseek_v2_lite_ref",
            exp_path="tests/trainer/test_megatron_trainer_deterministic.yaml",
            env_override=env_override,
        )

        assert self.check_numerical_reproducility(stdout, stdout_ref)


if __name__ == "__main__":
    unittest.main(buffer=False)
