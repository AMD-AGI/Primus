###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import os
import re
import unittest

from tests.utils import PrimusUT, run_training_script

_GFX_TO_PLATFORM = {
    "gfx942": "MI300X",
    "gfx950": "MI355X",
}


def detect_gpu_platform() -> str:
    """Map hardware GFX arch to platform config directory: gfx942 → MI300X, gfx950 → MI355X."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            props = torch.cuda.get_device_properties(0)
            arch_raw = getattr(props, "gcnArchName", "") or ""
            arch = arch_raw.split(":")[0].strip()
            if arch in _GFX_TO_PLATFORM:
                return _GFX_TO_PLATFORM[arch]
    except Exception:
        pass
    raise RuntimeError(f"Unable to detect GPU platform. Ensure ROCm GPU (gfx942/gfx950) is available.")


GPU_PLATFORM = detect_gpu_platform()


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
    train_log_path = os.path.join(ut_log_path, f"log.test_megatron_trainer-{tag}.txt")
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


class TestMegatronTrainer(PrimusUT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_llama3_8B(self):
        run_script(
            self.__class__.__name__,
            "llama3_8B",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/llama3_8B-BF16-pretrain.yaml",
            env_override={},
            extra_args=[
                "--num_layers",
                "4",
                "--train_iters",
                "3",
                "--enable_primus_turbo",
                "1",
                "--use_turbo_attention",
                "1",
            ],
        )

    def test_llama3_1_8B_tp2_distributed_dataset_regression(self):
        run_script(
            self.__class__.__name__,
            "llama3.1_8B_tp2_distributed_dataset_regression",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/llama3.1_8B-BF16-pretrain.yaml",
            env_override={
                "GPUS_PER_NODE": "2",
            },
            extra_args=[
                "--num_layers",
                "2",
                "--train_iters",
                "3",
                "--micro_batch_size",
                "1",
                "--global_batch_size",
                "2",
                "--tensor_model_parallel_size",
                "2",
                "--distributed_timeout_minutes",
                "3",
            ],
        )

    def test_llama3_70B(self):
        run_script(
            self.__class__.__name__,
            "llama3_70B",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/llama3_70B-BF16-pretrain.yaml",
            env_override={},
            extra_args=[
                "--num_layers",
                "4",
                "--train_iters",
                "3",
                "--enable_primus_turbo",
                "1",
                "--use_turbo_attention",
                "1",
            ],
        )

    def test_qwen3_30B_A3B(self):
        run_script(
            self.__class__.__name__,
            "qwen3_30B_A3B",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/qwen3_30B_A3B-BF16-pretrain.yaml",
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
                "--expert_model_parallel_size",
                "8",
                "--recompute_granularity",
                "full",
                "--recompute_method",
                "block",
                "--recompute_num_layers",
                "0",
                "--enable_primus_turbo",
                "1",
                "--use_turbo_attention",
                "1",
            ],
        )

    def test_qwen3_235B_A22B(self):
        run_script(
            self.__class__.__name__,
            "qwen3_235B_A22B",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/qwen3_235B_A22B-BF16-pretrain.yaml",
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
                "--recompute_granularity",
                "full",
                "--recompute_method",
                "block",
                "--recompute_num_layers",
                "0",
                "--enable_primus_turbo",
                "1",
                "--use_turbo_attention",
                "1",
            ],
        )

    def test_deepseek_v2_lite(self):
        run_script(
            self.__class__.__name__,
            "deepseek_v2_lite",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/deepseek_v2_lite-BF16-pretrain.yaml",
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
                "--enable_primus_turbo",
                "1",
                "--use_turbo_attention",
                "1",
            ],
        )

    def test_mixtral_8x7B(self):
        run_script(
            self.__class__.__name__,
            "mixtral_8x7B_v0.1",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/mixtral_8x7B_v0.1-BF16-pretrain.yaml",
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
                "--enable_primus_turbo",
                "1",
                "--use_turbo_attention",
                "1",
            ],
        )

    def test_mixtral_8x22B(self):
        run_script(
            self.__class__.__name__,
            "mixtral_8x22B_v0.1",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/mixtral_8x22B_v0.1-BF16-pretrain.yaml",
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
                "--enable_primus_turbo",
                "1",
                "--use_turbo_attention",
                "1",
            ],
        )

    def test_grok2(self):
        run_script(
            self.__class__.__name__,
            "grok2",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/grok2-BF16-pretrain.yaml",
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
                "--enable_primus_turbo",
                "1",
                "--use_turbo_attention",
                "1",
            ],
        )

    def test_deepseek_v3(self):
        run_script(
            self.__class__.__name__,
            "deepseek_v3",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/deepseek_v3-BF16-pretrain.yaml",
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
                "--enable_primus_turbo",
                "1",
                "--use_turbo_attention",
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
                "--enable_primus_turbo",
                "1",
                "--use_turbo_attention",
                "1",
            ],
        )

    # def test_zero_bubble_pipeline_parallelism(self):
    #     run_script(
    #         self.__class__.__name__,
    #         "zero_bubble_pipeline_parallelism",
    #         exp_path="tests/trainer/test_megatron_trainer_zero_bubble.yaml",
    #         env_override={},
    #     )

    def test_turbo_grouped_gemm(self):
        run_script(
            self.__class__.__name__,
            "deepseek_v2_lite_turbo_grouped_gemm",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/deepseek_v2_lite-BF16-pretrain.yaml",
            env_override={
                "PRIMUS_TURBO_AUTO_TUNE": "1",
            },
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
                "--enable_primus_turbo",
                "1",
                "--use_turbo_attention",
                "1",
                "--use_turbo_grouped_mlp",
                "1",
            ],
        )

    def test_turbo_fp8_grouped_gemm(self):
        run_script(
            self.__class__.__name__,
            "deepseek_v2_lite_turbo_fp8_grouped_gemm",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/deepseek_v2_lite-FP8-pretrain.yaml",
            env_override={
                "PRIMUS_TURBO_AUTO_TUNE": "1",
            },
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
                "--enable_primus_turbo",
                "1",
                "--use_turbo_attention",
                "1",
                "--use_turbo_grouped_mlp",
                "1",
                "--fp8",
                "e4m3",
                "--fp8_recipe",
                "tensorwise",
            ],
        )

    def test_turbo_deepep(self):
        stdout, _ = run_script(
            self.__class__.__name__,
            "turbo_deepep",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/deepseek_v2_lite-BF16-pretrain.yaml",
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
                "0",
                "--turbo_sync_free_moe_stage",
                "3",
                "--use_turbo_attention",
                "0",
                "--num_workers",
                "4",
                "--dataloader_mp_context",
                "forkserver",
            ],
        )
        # check dataloader_mp_context patch log
        Dataloader_mp_context_patch_log = "Setting DataLoader multiprocessing_context='forkserver'"
        assert (
            Dataloader_mp_context_patch_log in stdout
        ), "Expected dataloader_mp_context patch log not found in stdout"

    def test_deepseekv2_lite_uep(self):
        run_script(
            self.__class__.__name__,
            "deepseek_v2_lite_uep",
            exp_path="examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml",
            env_override={"USING_UEP": "1", "REBUILD_UEP": "1"},
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
                "0",
                "--turbo_sync_free_moe_stage",
                "3",
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

    def check_numerical_reproducility(self, log, log_ref):
        loss = self.extract_loss_from_log(log)
        loss_ref = self.extract_loss_from_log(log_ref)

        is_reproducility = True
        # compare as str, need bitwise equal.
        for i in range(0, len(loss)):
            if loss[i] != loss_ref[i]:
                is_reproducility = False
                break

        return is_reproducility

    def test_llama3_8B(self):
        env_override = {
            "BACKEND": "megatron",
            "PRIMUS_MODEL": "llama3_8B",
            "PRIMUS_GLOBAL_BATCH_SIZE": "8",
            "PRIMUS_NUM_LAYERS": "4",
            # deterministic vars
            "PRIMUS_DETERMINISTIC": "1",
            "NCCL_ALGO": "Ring",
            "TORCH_COMPILE_DISABLE": "1",
            "ROCBLAS_DEFAULT_ATOMICS_MODE": "0",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
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

    def test_deepseek_v2_lite(self):
        env_override = {
            "BACKEND": "megatron",
            "PRIMUS_MODEL": "deepseek_v2_lite",
            "PRIMUS_GLOBAL_BATCH_SIZE": "8",
            "PRIMUS_MOE_LAYER_FREQ": "[0]*1+[1]*3",
            "PRIMUS_EP": "8",
            "PRIMUS_NUM_LAYERS": "4",
            # deterministic vars
            "PRIMUS_DETERMINISTIC": "1",
            "NCCL_ALGO": "Ring",
            "TORCH_COMPILE_DISABLE": "1",
            "ROCBLAS_DEFAULT_ATOMICS_MODE": "0",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
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
