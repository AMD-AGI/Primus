import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from logging import getLogger

import yaml

logger = getLogger("deterministic_testing")

DENSE_MODELS = [
    "llama2_7B",
    "llama3_8B",
    "llama3_70B",
    "llama3.1_8B",
    "qwen2.5_7B",
    "qwen2.5_72B",
    "qwen3_8B",
]

MOE_MODELS = [
    "deepseek_v2_lite",
    "llama4_17B16E",
    "mixtral_8x7B_v0.1",
    "qwen3_30B_A3B",
]

ALL_MODELS = DENSE_MODELS + MOE_MODELS

EXTRA_OPTIONS = {
    "enable_primus_turbo": False,
    "use_turbo_attention": False,
    "use_turbo_grouped_mlp": False,
    "use_flash_attn": False,
    "cross_entropy_loss_fusion": False,
    "deterministic_mode": True,
}

MOE_EXTRA_OPTIONS = {
    "moe_enable_deepep": False,
    "turbo_sync_free_moe_stage": 0,
}

MICRO_BATCH_SIZE = 1
GLOBAL_BATCH_SIZE = 8
ITERS = 1000

FP8 = [True, False]
EP = [1, 8]
TP = [1, 8]

BASE_CONFIG_PATH = "examples/megatron/configs/MI355X"
OUTPUT_DIR = "output"


@dataclass
class Config:
    model_name: str
    micro_batch_size: 1
    global_batch_size: 8
    iters: 1000
    # model type
    is_moe: True
    # precision
    fp8: True
    # parallel strategies
    TP: 1
    EP: 1
    # extra options
    extra_options: {}

    def __hash__(self) -> int:
        return hash(
            (
                self.micro_batch_size,
                self.global_batch_size,
                self.iters,
                self.is_moe,
                self.fp8,
                self.TP,
                self.EP,
                tuple(self.extra_options.items()),
            )
        )


def generate_yaml_config(config: Config):
    model_name = config.model_name
    with open(
        os.path.join(BASE_CONFIG_PATH, f"{model_name}-{'FP8' if config.fp8 else 'BF16'}-pretrain.yaml"), "r"
    ) as f:
        data = yaml.safe_load(f)

    overrides = data["modules"]["pre_trainer"]["overrides"]
    overrides["micro_batch_size"] = config.micro_batch_size
    overrides["global_batch_size"] = config.global_batch_size
    overrides["train_iters"] = config.iters

    overrides["tensor_model_parallel_size"] = config.TP
    overrides["expert_model_parallel_size"] = config.EP

    for key, value in config.extra_options.items():
        overrides[key] = value

    config_name = (
        f"{model_name}-{'FP8' if config.fp8 else 'BF16'}-TP{config.TP}-EP{config.EP}_deterministic.yaml"
    )
    config_path = os.path.join(BASE_CONFIG_PATH, config_name)
    with open(config_path, "w") as f:
        yaml.dump(data, f)

    return config_path


def generate_deterministic_testing_configs():
    # 组合所有列表
    all_combinations = list(product(ALL_MODELS, FP8, EP, TP))

    filter_configs = set()
    # 打印结果
    for combine in all_combinations:
        model_name, fp8, ep, tp = combine
        extra_options = EXTRA_OPTIONS
        if model_name in DENSE_MODELS:
            if ep != 1:
                continue
        if model_name in MOE_MODELS:
            if tp != 1:
                continue
            extra_options.update(MOE_EXTRA_OPTIONS)
        filter_configs.add(
            Config(
                model_name=model_name,
                micro_batch_size=MICRO_BATCH_SIZE,
                global_batch_size=GLOBAL_BATCH_SIZE,
                iters=ITERS,
                is_moe=model_name in MOE_MODELS,
                fp8=fp8,
                TP=tp,
                EP=ep,
                extra_options=extra_options,
            )
        )

    configs_path = []
    for config in filter_configs:
        configs_path.append(generate_yaml_config(config))

    return configs_path


def run_script(
    exp_path: str,
    log_path: str,
    env_override: dict = None,
    extra_args: list[str] = None,
):
    tag = os.path.basename(exp_path)
    shell_entry = "examples/run_pretrain.sh"
    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    env["EXP"] = exp_path

    env["TRAIN_LOG"] = log_path

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

        with open(log_path, "r") as f:
            stdout_output = f.read()

        stderr_output = ""

        return stdout_output, stderr_output

    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr or ""
        stdout_output = e.stdout or ""

        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    stdout_output = f.read()
            except Exception as log_err:
                logger.warning(f"[{tag}] Failed to read train log: {log_err}")

        if "after training is done" in stdout_output:
            logger.warning(f"[{tag}] Training likely succeeded despite return code != 0.")
            logger.warning(f"stderr excerpt:\n{stderr_output[:1000]}")
        else:
            raise AssertionError(f"Shell script failed: {stderr_output.strip()}")

    return stdout_output, stderr_output


def extract_loss_and_grad_norm_from_log(log):
    LOSS_PATTERN = r"lm loss: (\d+.\d+E\+\d+)"
    GRAD_NORM_PATTERN = r"grad norm:\s+([\d.E+-]+)"

    loss = re.findall(LOSS_PATTERN, log)
    grad_norm = re.findall(GRAD_NORM_PATTERN, log)

    return loss, grad_norm


def check_numerical_reproducibility(log, log_ref):
    loss, grad_norm = extract_loss_and_grad_norm_from_log(log)
    loss_ref, grad_norm_ref = extract_loss_and_grad_norm_from_log(log_ref)

    is_reproducility = True
    # compare as str, need bitwise equal.
    for i in range(0, len(loss)):
        if loss[i] != loss_ref[i]:
            is_reproducility = False
            break
        if grad_norm[i] != grad_norm_ref[i]:
            is_reproducility = False
            break

    return is_reproducility


def run_deterministic_testing_configs(configs_path: list[str]):

    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.mkdir(OUTPUT_DIR)

    success_cases = []
    failed_cases = []
    for config_path in configs_path:
        log_path = os.path.join(output_dir, f"{os.path.basename(config_path)}.0.txt")
        log, _ = run_script(
            config_path,
            log_path,
            env_override={
                "PRIMUS_DETERMINISTIC": "1",
                "BACKEND": "megatron",
            },
        )

        log_path = os.path.join(output_dir, f"{os.path.basename(config_path)}.1.txt")
        log_ref, _ = run_script(
            config_path,
            log_path,
            env_override={
                "PRIMUS_DETERMINISTIC": "1",
                "BACKEND": "megatron",
            },
        )

        if check_numerical_reproducibility(log, log_ref):
            logger.error(f"Numerical reproducibility check failed for {config_path}")
            failed_cases.append(config_path)
        else:
            logger.info(f"Numerical reproducibility check passed for {config_path}")
            success_cases.append(config_path)

    return success_cases, failed_cases


if __name__ == "__main__":
    configs_path = generate_deterministic_testing_configs()
    success_cases, failed_cases = run_deterministic_testing_configs(configs_path)
    print(f"Success cases: {success_cases}")
    print(f"Failed cases: {failed_cases}")
