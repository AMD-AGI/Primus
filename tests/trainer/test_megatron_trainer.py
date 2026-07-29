###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import os
import re
import socket
import subprocess
import sys
import unittest

from tests.utils import PrimusUT, run_training_script


def _find_free_port() -> int:
    """Ask the kernel for a free TCP port (bind to 0); avoids EADDRINUSE from
    guessing a port that overlaps the ephemeral range or a not-yet-released one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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


def run_posttrain_script(
    ut_name: str,
    tag: str,
    exp_path: str,
    env_override: dict = None,
    extra_args: list[str] = None,
):
    """Like run_script, but for the "posttrain" suite (SFT/alignment).

    SFT experiment configs declare a `modules.post_trainer` section (instead
    of `modules.pre_trainer`), which the CLI only loads via `train posttrain`
    (see primus/cli/subcommands/train.py). The "Training completed." marker
    that run_training_script asserts on is emitted generically by
    PrimusRuntime._run_trainer_lifecycle for any module, so it applies here
    unchanged.
    """
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
        "posttrain",
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
                # Unset the config's interleaved pipeline_model_parallel_layout (it
                # requires PP>1, but this test runs with PP=1). "None" is coerced to
                # Python None by parse_cli_overrides, which clears the layout; an empty
                # string instead builds an empty layout that fails validation.
                "--pipeline_model_parallel_layout",
                "None",
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

    def test_qwen3_5_35B_A3B(self):
        run_script(
            self.__class__.__name__,
            "qwen3_5_35B_A3B",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/qwen3_5_35B_A3B-BF16-pretrain.yaml",
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

    def test_gpt_oss_20B_sink_attention(self):
        run_script(
            self.__class__.__name__,
            "gpt_oss_20B_sink_attention",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/gpt_oss_20B-BF16-pretrain.yaml",
            env_override={},
            extra_args=[
                "--num_layers",
                "4",
                "--train_iters",
                "3",
                "--micro_batch_size",
                "2",
                "--global_batch_size",
                "16",
                "--enable_primus_turbo",
                "1",
                "--use_turbo_attention",
                "1",
                "--use_sink_attention",
                "1",
                "--profile",
                "0",
                "--use_pytorch_profiler",
                "0",
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
                "--num_virtual_stages_per_pipeline_rank",
                "1",
                "--enable_primus_turbo",
                "1",
                "--use_turbo_attention",
                "1",
                "--pipeline_model_parallel_layout",
                "null",
                "--recompute_layer_ids",
                "null",
                "--recompute_granularity",
                "full",
                "--recompute_method",
                "block",
                "--recompute_num_layers",
                "0",
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
                "--use_turbo_grouped_gemm",
                "1",
                # use_turbo_grouped_gemm is incompatible with the config's default
                # moe_use_legacy_grouped_gemm=True, so disable the legacy path.
                "--moe_use_legacy_grouped_gemm",
                "0",
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
                "--use_turbo_grouped_gemm",
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
                # Sync-Free MoE stage 3 requires PrimusTurboGroupedLinear.
                "--use_turbo_grouped_gemm",
                "1",
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

    def test_sdma_allgather_fused_residual_norm(self):
        # Neither patch runs in any other case here: SDMA param all-gather
        # only activates with ENABLE_SDMA_ALLGATHER=1 (env var, not a --arg) +
        # a distributed optimizer; fused residual+RMSNorm needs
        # use_turbo_rms_norm=1 + PRIMUS_FUSED_RESIDUAL_NORM_V2=1. Confirmed via
        # coverage instrumentation that both patches install and execute.
        run_script(
            self.__class__.__name__,
            "sdma_allgather_fused_residual_norm",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/llama3_8B-BF16-pretrain.yaml",
            env_override={
                "ENABLE_SDMA_ALLGATHER": "1",
                "PRIMUS_FUSED_RESIDUAL_NORM_V2": "1",
            },
            extra_args=[
                "--num_layers",
                "4",
                "--train_iters",
                "3",
                "--enable_primus_turbo",
                "1",
                "--use_turbo_attention",
                "1",
                "--use_turbo_rms_norm",
                "1",
                "--use_distributed_optimizer",
                "1",
                "--overlap_param_gather",
                "1",
            ],
        )

    def test_mamba_370M(self):
        # Default `auto` attention backend exercises attention_backend_patches:
        # megatron-core's probe must respect the image's baked NVTE_FLASH_ATTN=0.
        run_script(
            self.__class__.__name__,
            "mamba_370M",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/mamba_370M-pretrain.yaml",
            env_override={},
            extra_args=[
                "--num_layers",
                "4",
                "--train_iters",
                "3",
                "--micro_batch_size",
                "2",
                "--global_batch_size",
                "16",
            ],
        )

    def test_zebra_llama_1B_hybrid(self):
        # Hybrid Mamba+MLA (HybridStack) path. num_layers=8 is the minimum that
        # keeps the default hybrid_attention_ratio=0.25 from allocating zero
        # attention layers (division by zero).
        run_script(
            self.__class__.__name__,
            "zebra_llama_1B_hybrid",
            exp_path=f"examples/megatron/configs/{GPU_PLATFORM}/zebra_llama_1B-pretrain.yaml",
            env_override={},
            extra_args=[
                "--num_layers",
                "8",
                "--train_iters",
                "3",
                "--micro_batch_size",
                "2",
                "--global_batch_size",
                "16",
            ],
        )

    def test_mamba_130M_bridge_pretrain(self):
        # Only E2E covering the megatron_bridge backend (mamba/zebra above use
        # the megatron backend). extra_args pin a tiny shape so the test doesn't
        # depend on the example yaml's sizes. Don't override seq_length: the
        # recipe feeds it to both model and dataset but a CLI override reaches
        # only the dataset, and Bridge asserts the two match.
        run_script(
            self.__class__.__name__,
            "mamba_130M_bridge_pretrain",
            exp_path=f"examples/megatron_bridge/configs/{GPU_PLATFORM}/mamba_130M_pretrain.yaml",
            env_override={},
            extra_args=[
                "--train_iters",
                "3",
                "--micro_batch_size",
                "1",
                "--global_batch_size",
                "8",
            ],
        )

    def test_qwen2_sft_lora(self):
        # Only E2E covering the "posttrain" suite (MegatronSFTTrainer) and
        # peft/*.py; LoRA is enabled in test_megatron_trainer_sft_lora.yaml.
        # The posttrain hook HF->Megatron-converts `tokenizer_model` into the
        # base checkpoint when pretrained_checkpoint/load are unset, so this
        # uses a tiny stand-in checkpoint, not real Qwen2.5-7B weights.
        run_posttrain_script(
            self.__class__.__name__,
            "qwen2_sft_lora",
            exp_path="tests/trainer/test_megatron_trainer_sft_lora.yaml",
            env_override={},
            # head_dim (2) is below the image's fused-attention CK kernel minimum,
            # so pin the unfused path; attention_backend_patches forces it over
            # the image's baked NVTE_FUSED_ATTN=1.
            extra_args=["--attention_backend", "unfused"],
        )

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

    def _run_deepseek_v2_lite_zbv_fp8_case(
        self,
        tag: str,
        extra_args: list[str] = None,
    ):
        base_env = {
            "BACKEND": "megatron",
        }
        base_extra_args = [
            "--num_layers",
            "8",
            "--moe_layer_freq",
            "[0]*1+[1]*7",
            "--global_batch_size",
            "16",
            "--pipeline_model_parallel_size",
            "4",
            "--num_virtual_stages_per_pipeline_rank",
            "2",
            "--expert_model_parallel_size",
            "2",
            "--pp_algorithm",
            "zbv-formatted",
            "--fp8",
            "hybrid",
            "--fp8_recipe",
            "delayed",
            "--enable_primus_turbo",
            "1",
            "--use_turbo_attention",
            "0",
            "--use_turbo_grouped_gemm",
            "0",
            "--use_turbo_gemm",
            "0",
            "--moe_use_legacy_grouped_gemm",
            "0",
        ]
        stdout, _ = run_script(
            self.__class__.__name__,
            tag,
            exp_path="tests/trainer/test_megatron_trainer_zbv_fp8.yaml",
            env_override=base_env,
            extra_args=base_extra_args + (extra_args or []),
        )
        self.assertIn("Training completed.", stdout)
        return stdout

    def test_deepseek_v2_lite_te_fp8_zbv_formatted(self):
        stdout = self._run_deepseek_v2_lite_zbv_fp8_case(
            "deepseek_v2_lite_te_fp8_zbv_formatted",
            extra_args=[
                "--enable_primus_turbo",
                "0",
            ],
        )
        self.assertIn("[Patch:megatron.pp.te_wgrad_split]", stdout)
        self.assertNotIn("[Patch:megatron.pp.legacy_grouped_mlp_wgrad_split]", stdout)

    def test_deepseek_v2_lite_turbo_bf16_zbv_formatted(self):
        stdout = self._run_deepseek_v2_lite_zbv_fp8_case(
            "deepseek_v2_lite_turbo_fp8_zbv_formatted",
            extra_args=[
                "--fp8",
                "false",
                "--use_turbo_attention",
                "1",
                "--use_turbo_grouped_gemm",
                "1",
                "--use_turbo_gemm",
                "1",
            ],
        )
        self.assertNotIn("[Patch:megatron.pp.legacy_grouped_mlp_wgrad_split]", stdout)

    def test_deepseek_v2_lite_bf16_lagacy_gg_zbv_formatted(self):
        stdout = self._run_deepseek_v2_lite_zbv_fp8_case(
            "deepseek_v2_lite_bf16_lagacy_gg_zbv_formatted",
            extra_args=[
                "--enable_primus_turbo",
                "0",
                "--fp8",
                "false",
                "--moe_use_legacy_grouped_gemm",
                "1",
            ],
        )
        self.assertIn("[Patch:megatron.pp.legacy_grouped_mlp_wgrad_split]", stdout)


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


class TestProjectionSimulate(PrimusUT):
    """Projection is an offline planner that training E2E never runs, so it gets no
    E2E coverage otherwise. Two paths are exercised so the coverage .pth records them:

    - simulate (no GPU): origami/SDPA backends, multinode scaling, CLI orchestration.
      A dense + a MoE config; the MoE one adds the expert-parallel / All-to-All /
      router / grouped-GEMM branches the dense config never reaches.
    - benchmark (real GPU layer): _run_layer_benchmark / benchmark_layer /
      memory_capture paths that simulate cannot reach, launched via primus-cli so
      torchrun sets up the distributed env the real layer bench requires.

    Projection auto-limits the stack to 1-2 layers, so all cases are quick."""

    _DENSE = "llama3.1_8B-BF16-pretrain.yaml"
    _MOE = "mixtral_8x7B_v0.1-BF16-pretrain.yaml"
    _BENCH = "qwen2.5_7B-BF16-pretrain.yaml"

    def _check(self, result) -> str:
        # returncode is the robust E2E-smoke contract: projection raises on any
        # real error (bad config/arch, missing data, gated model, ...) which the
        # CLI turns into a non-zero exit. We deliberately do NOT assert on result
        # text -- the result-line wording differs per path (simulate/benchmark,
        # dense/MoE/PP) and is brittle to reword.
        self.assertEqual(result.returncode, 0, msg=(result.stdout[-1500:] + result.stderr[-1500:]))
        return result.stdout

    def _run_simulate(self, suite: str, mode_flag: str, model: str, extra: list | None = None) -> str:
        config = f"examples/megatron/configs/{GPU_PLATFORM}/{model}"
        cmd = [
            sys.executable,
            "-m",
            "primus.cli.main",
            "projection",
            suite,
            "--config",
            config,
            mode_flag,
            "simulate",
            "--gpu-arch",
            GPU_PLATFORM.lower(),
            "--target-nodes",
            "4",
            *(extra or []),
        ]
        return self._check(subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy()))

    def _run_benchmark(self, suite: str, mode_flag: str) -> str:
        # Real GPU layer bench needs torchrun's distributed env -> go through
        # the primus-cli launcher (a plain subprocess won't set NNODES/NODE_RANK).
        config = f"examples/megatron/configs/{GPU_PLATFORM}/{self._BENCH}"
        cmd = [
            "bash",
            "./runner/primus-cli",
            "direct",
            "--",
            "projection",
            suite,
            "--config",
            config,
            mode_flag,
            "benchmark",
            "--benchmark-gpus",
            "8",
            "--target-nodes",
            "2",
        ]
        env = os.environ.copy()
        # Kernel-assigned free port so back-to-back GPU benches don't hit
        # EADDRINUSE (a fixed/random port can clash with a not-yet-released
        # previous run or the ephemeral range).
        env["MASTER_PORT"] = str(_find_free_port())
        return self._check(subprocess.run(cmd, capture_output=True, text=True, env=env))

    def test_performance_simulate_dense(self):
        self._run_simulate("performance", "--profiling-mode", self._DENSE)

    def test_memory_simulate_dense(self):
        self._run_simulate("memory", "--memory-mode", self._DENSE)

    def test_performance_simulate_moe(self):
        # --target-ep-size drives the expert-parallel projection path.
        self._run_simulate("performance", "--profiling-mode", self._MOE, extra=["--target-ep-size", "8"])

    def test_memory_simulate_moe(self):
        self._run_simulate("memory", "--memory-mode", self._MOE)

    def test_performance_simulate_pp(self):
        # PP>1 is the only thing that runs the pipeline-scheduler simulator
        # (simulator.py); the PP=1 configs all skip pipeline simulation.
        self._run_simulate(
            "performance", "--profiling-mode", self._DENSE, extra=["--pipeline_model_parallel_size", "4"]
        )

    def _require_2_gpus(self):
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("projection benchmark needs >=2 GPUs")

    def test_performance_benchmark(self):
        # Real GPU layer bench: covers _run_layer_benchmark / benchmark_layer /
        # utils.benchmark_layer that simulate can't reach.
        self._require_2_gpus()
        self._run_benchmark("performance", "--profiling-mode")

    def test_memory_benchmark(self):
        # Real GPU memory capture: covers memory_projection/benchmark.py +
        # memory_capture.py (HBM peak hooks) that simulate can't reach.
        self._require_2_gpus()
        self._run_benchmark("memory", "--memory-mode")


if __name__ == "__main__":
    unittest.main(buffer=False)
