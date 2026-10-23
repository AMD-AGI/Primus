###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for tools/perf/extract_results.py.

The parser has to cope with three different Megatron per-iteration layouts,
two legacy header styles and multi-rank logs. That variety is not academic:
the log format changed underneath this tool once already, which is what
forced the v2 rewrite. These tests pin each shape down with a tiny synthetic
log so the next format change fails here rather than silently producing empty
or wrong throughput numbers.

Pure CPU; no GPU, no yq, no Primus import required.
"""

from __future__ import annotations

import importlib.util
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "tools" / "perf" / "extract_results.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("perf_extract_results", MODULE_PATH)
    assert spec and spec.loader, f"cannot load {MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extract = _load_module()


BANNER_TEMPLATE = """\
################################################################################
# Primus Benchmark Run
################################################################################
# Timestamp        : 2026-09-12 20:00:00 UTC
# Hostname         : test-node-01
# Config Name      : {config_name}
# Config Hash      : abcd1234  (sha256, first 8 hex chars)
# Framework        : {framework}
# Micro Batch Size : 2
# Global Batch Size: 16
# Sequence Length  : 8192
# Train Steps/Iters: 10
# Repetition       : 1 / 3
{banner_extra}# Cluster          : NNODES={nnodes} GPUS_PER_NODE=8 NODE_RANK=0
# Docker image     : registry.example/ci:some-tag
# Image digest     : registry.example/ci@sha256:deadbeef
# Primus commit    : 137d6c256119
# Submodule pins   : 43f426fa
# GPU model        : AMD Instinct MI325X
# ROCm version     : 7.2.1
# Torch version    : 2.12.0+rocm10.0.0
# JAX version      : unknown
# World size       : {world_size}  (NNODES={nnodes} x GPUS_PER_NODE=8)
################################################################################

########################## Begin Config File Dump ##############################
{config_body}
########################### End Config File Dump ###############################

############################## Begin Run Output ################################
{body}{footer}"""

FOOTER_TEMPLATE = """\
############################### End Run Output #################################

################################################################################
# Run finished at : 2026-09-12 20:10:00 UTC
# Exit code       : {exit_code}
# Elapsed (sec)   : 600
{failure_line}################################################################################
"""


def write_log(
    tmp_path: Path,
    body: str,
    *,
    framework: str = "megatron",
    config_name: str = "llama3.1_8B-BF16-pretrain",
    nnodes: int = 1,
    world_size: int = 8,
    exit_code: int = 0,
    config_body: str = "modules:\n  pre_trainer:\n    framework: megatron\n",
    name: str = "run.log",
    failure_reason: str | None = None,
    with_footer: bool = True,
    banner_extra: str = "",
) -> Path:
    path = tmp_path / name
    failure_line = f"# Failure reason   : {failure_reason}\n" if failure_reason else ""
    footer = FOOTER_TEMPLATE.format(exit_code=exit_code, failure_line=failure_line) if with_footer else ""
    path.write_text(
        BANNER_TEMPLATE.format(
            config_name=config_name,
            framework=framework,
            nnodes=nnodes,
            world_size=world_size,
            config_body=config_body,
            body=body,
            footer=footer,
            banner_extra=banner_extra,
        )
    )
    return path


# --------------------------------------------------------------------------
# Precision / model-name splitting
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_name,expected",
    [
        # The hyphen-delimited form that always worked.
        ("llama3.1_8B-BF16-pretrain", ("llama3.1_8B", "BF16")),
        ("deepseek_v3-FP8-pretrain", ("deepseek_v3", "FP8")),
        # Underscore-delimited: previously left precision empty and glued the
        # token into the model name.
        ("gdn_1B_BF16-pretrain", ("gdn_1B", "BF16")),
        ("hylo_llama_mamba_8B_BF16-pretrain", ("hylo_llama_mamba_8B", "BF16")),
        # Precision trailing the suite verb.
        ("llama3.1_405B-pretrain-FP8", ("llama3.1_405B", "FP8")),
        ("mixtral_8x22B-pretrain-BF16", ("mixtral_8x22B", "BF16")),
        # Multi-word precision must not be truncated to its fp8 tail.
        ("llama3_8B-nanoo_fp8-pretrain", ("llama3_8B", "NANOO_FP8")),
        ("gpt_oss_20B-MXFP4-pretrain", ("gpt_oss_20B", "MXFP4")),
        ("llama3.1_8B-MXFP8-pretrain", ("llama3.1_8B", "MXFP8")),
        # No precision in the name at all.
        ("mamba_370M-pretrain", ("mamba_370M", None)),
        # The sft/lora marker distinguishes two workloads and must survive.
        ("qwen3_32b_sft_posttrain", ("qwen3_32b_sft", None)),
        ("qwen3_32b_lora_posttrain", ("qwen3_32b_lora", None)),
        # A full repo-relative path, which is what the runner now emits.
        ("examples/maxtext/configs/MI325X/llama3_8B-bf16-pretrain", ("llama3_8B", "BF16")),
    ],
)
def test_split_model_precision(config_name, expected):
    assert extract._split_model_precision(config_name) == expected


def test_precision_falls_back_to_config_body(tmp_path):
    """A name without a precision token still resolves via the config."""
    log = write_log(
        tmp_path,
        "iteration 4/10 | elapsed time per iteration (ms): 100.0 | "
        "compute per GPU (TFLOP/s/GPU): 500.0 | tokens/s/GPU inst/harmonic mean: 1000.0\n",
        config_name="mamba_370M-pretrain",
        config_body="modules:\n  pre_trainer:\n    overrides:\n      fp8: hybrid\n",
    )
    assert extract.parse_log_file(str(log))["precision"] == "FP8"


# --------------------------------------------------------------------------
# The three Megatron per-iteration layouts
# --------------------------------------------------------------------------

MEGATRON_LEGACY = """\
iteration 1/10 | elapsed time per iteration (ms): 2000.0 | throughput per GPU (TFLOP/s/GPU): 100.0 | tokens per GPU (tokens/s/GPU): 1000.0
iteration 2/10 | elapsed time per iteration (ms): 2000.0 | throughput per GPU (TFLOP/s/GPU): 200.0 | tokens per GPU (tokens/s/GPU): 2000.0
iteration 3/10 | elapsed time per iteration (ms): 2000.0 | throughput per GPU (TFLOP/s/GPU): 300.0 | tokens per GPU (tokens/s/GPU): 3000.0
iteration 4/10 | elapsed time per iteration (ms): 2000.0 | throughput per GPU (TFLOP/s/GPU): 400.0 | tokens per GPU (tokens/s/GPU): 4000.0
iteration 5/10 | elapsed time per iteration (ms): 2000.0 | throughput per GPU (TFLOP/s/GPU): 400.0 | tokens per GPU (tokens/s/GPU): 4000.0
"""

MEGATRON_SLASH = """\
iteration 1/10 | elapsed time per iteration (ms): 1997.8/2032.7 | throughput per GPU (TFLOP/s/GPU): 100.0/111.0 | tokens/s/GPU inst/harmonic mean: 1000.0/1111.0
iteration 2/10 | elapsed time per iteration (ms): 1997.8/2032.7 | throughput per GPU (TFLOP/s/GPU): 200.0/222.0 | tokens/s/GPU inst/harmonic mean: 2000.0/2222.0
iteration 3/10 | elapsed time per iteration (ms): 1997.8/2032.7 | throughput per GPU (TFLOP/s/GPU): 300.0/333.0 | tokens/s/GPU inst/harmonic mean: 3000.0/3333.0
iteration 4/10 | elapsed time per iteration (ms): 1997.8/2032.7 | throughput per GPU (TFLOP/s/GPU): 400.0/444.0 | tokens/s/GPU inst/harmonic mean: 4000.0/4444.0
iteration 5/10 | elapsed time per iteration (ms): 1997.8/2032.7 | throughput per GPU (TFLOP/s/GPU): 400.0/444.0 | tokens/s/GPU inst/harmonic mean: 4000.0/4444.0
"""

MEGATRON_AVG = """\
iteration 1/10 | elapsed time per iteration (ms): 10975.9/10975.9 | compute per GPU (TFLOP/s/GPU): 100.0 (avg 100.0) | tokens/s/GPU inst/harmonic mean: 1000.0/1000.0
iteration 2/10 | elapsed time per iteration (ms): 10975.9/10975.9 | compute per GPU (TFLOP/s/GPU): 200.0 (avg 150.0) | tokens/s/GPU inst/harmonic mean: 2000.0/1500.0
iteration 3/10 | elapsed time per iteration (ms): 10975.9/10975.9 | compute per GPU (TFLOP/s/GPU): 300.0 (avg 200.0) | tokens/s/GPU inst/harmonic mean: 3000.0/2000.0
iteration 4/10 | elapsed time per iteration (ms): 10975.9/10975.9 | compute per GPU (TFLOP/s/GPU): 400.0 (avg 250.0) | tokens/s/GPU inst/harmonic mean: 4000.0/2500.0
iteration 5/10 | elapsed time per iteration (ms): 10975.9/10975.9 | compute per GPU (TFLOP/s/GPU): 400.0 (avg 280.0) | tokens/s/GPU inst/harmonic mean: 4000.0/2800.0
"""


@pytest.mark.parametrize(
    "layout,body",
    [
        pytest.param("legacy_single_value", MEGATRON_LEGACY, id="legacy_single_value"),
        pytest.param("inst_harmonic_slash", MEGATRON_SLASH, id="inst_harmonic_slash"),
        pytest.param("compute_per_gpu_avg", MEGATRON_AVG, id="compute_per_gpu_avg"),
    ],
)
def test_megatron_layouts_all_parse(tmp_path, layout, body):
    """All three layouts yield the same instantaneous values.

    Each carries a different running-average spelling; the parser must take
    the leading (instantaneous) number in every case so it can compute its own
    harmonic mean over the post-warmup window.
    """
    result = extract.parse_log_file(str(write_log(tmp_path, body, name=f"{layout}.log")))

    assert result["backend"] == "megatron"
    assert result["num_iterations"] == 5
    # MIN_PERF_SKIP_STEPS drops the first three steps, leaving 400 and 400.
    assert result["num_post_warmup"] == 2
    assert result["hmean_tflops"] == pytest.approx(400.0)
    assert result["hmean_tps_per_gpu"] == pytest.approx(4000.0)


def test_warmup_floor_drops_compile_steps(tmp_path):
    """The first steps are compile/autotune and must not enter the mean."""
    result = extract.parse_log_file(str(write_log(tmp_path, MEGATRON_AVG)))
    # A naive mean over all five would be dragged down by the 100/200/300 ramp.
    assert result["hmean_tflops"] > 350.0


# --------------------------------------------------------------------------
# Multi-rank logs
# --------------------------------------------------------------------------


def test_multirank_keeps_a_single_rank(tmp_path):
    """Interleaved rank output must not inflate the iteration count.

    Primus-patched Megatron prints from the last rank rather than rank 0, so
    the parser picks the lowest rank that actually emits iteration lines.
    """
    lines = []
    for step in range(1, 6):
        for rank in (61, 63):
            lines.append(
                f"[20260912 20:00:00][rank-{rank}/64][INFO] iteration {step}/10 | "
                f"elapsed time per iteration (ms): 2000.0 | "
                f"compute per GPU (TFLOP/s/GPU): 400.0 (avg 400.0) | "
                f"tokens/s/GPU inst/harmonic mean: 4000.0/4000.0"
            )
    result = extract.parse_log_file(
        str(write_log(tmp_path, "\n".join(lines) + "\n", nnodes=8, world_size=64))
    )

    # 5 steps from one rank, not 10 from two.
    assert result["num_iterations"] == 5
    assert result["world_size"] == 64
    assert result["nnodes"] == 8


def test_rank0_preferred_when_present(tmp_path):
    lines = []
    for step in range(1, 6):
        for rank in (0, 1):
            tflops = 400.0 if rank == 0 else 999.0
            lines.append(
                f"[rank-{rank}/8][INFO] iteration {step}/10 | "
                f"elapsed time per iteration (ms): 2000.0 | "
                f"compute per GPU (TFLOP/s/GPU): {tflops} (avg {tflops}) | "
                f"tokens/s/GPU inst/harmonic mean: 4000.0/4000.0"
            )
    result = extract.parse_log_file(str(write_log(tmp_path, "\n".join(lines) + "\n")))

    assert result["num_iterations"] == 5
    assert result["hmean_tflops"] == pytest.approx(400.0)


# --------------------------------------------------------------------------
# Other backends
# --------------------------------------------------------------------------


def test_maxtext_parses_and_reports_samples_frames(tmp_path):
    body = "\n".join(
        f"completed step: {s}, seconds: 5.0, TFLOP/s/device: 100.0, "
        f"Tokens/s/device: 1000.0, Samples/s/device: 0.5, Frames/s/device: 2.0, loss: 3.0"
        for s in range(0, 6)
    )
    result = extract.parse_log_file(
        str(
            write_log(
                tmp_path,
                body + "\n",
                framework="maxtext",
                config_name="llama3_8B-bf16-pretrain",
            )
        )
    )
    assert result["backend"] == "maxtext"
    assert result["num_iterations"] == 6
    assert result["hmean_tflops"] == pytest.approx(100.0)
    assert result["hmean_samples_per_gpu"] == pytest.approx(0.5)
    assert result["hmean_frames_per_gpu"] == pytest.approx(2.0)


def test_torchtitan_parses_memory(tmp_path):
    body = "\n".join(
        f"[rank-0/8][INFO] step: {s}  loss:  7.0  grad_norm: 1.0  "
        f"memory: 50.00GiB(62.50%)  tps: 1,000  tflops: 400.00  mfu: 30.00%"
        for s in range(1, 6)
    )
    result = extract.parse_log_file(
        str(
            write_log(
                tmp_path,
                body + "\n",
                framework="torchtitan",
                config_name="llama3.1_8B-BF16-pretrain",
            )
        )
    )
    assert result["backend"] == "torchtitan"
    assert result["amean_memory_usage"] == pytest.approx(50.0)
    assert result["amean_memory_usage_pct"] == pytest.approx(62.5)
    assert result["hmean_tps_per_gpu"] == pytest.approx(1000.0)


def test_torchtitan_reads_numbers_with_thousands_separators(tmp_path):
    # TorchTitan's own step line (tflops:,.2f), colours included; a value of
    # 1,293.62 used to be read as 1.
    body = "\n".join(
        f"[\x1b[32m20261015 20:10:50\x1b[0m][\x1b[36mrank-0/16\x1b[0m][\x1b[1m[INFO]     \x1b[0m"
        f"\x1b[1m\x1b[31mstep: {s:2}  \x1b[32mloss: 27.72860  \x1b[38;2;180;60;0mgrad_norm: 130.8515  "
        f"\x1b[38;2;54;234;195mmemory: 281.23GiB(97.65%)  \x1b[34mtps: {tps}  "
        f"\x1b[36mtflops: {tflops}  \x1b[35mmfu: 51.74%\x1b[39m\x1b[0m"
        for s, tps, tflops in [
            (1, "218", "310.71"),
            (2, "1,909", "1,293.62"),
            (3, "1,923", "1,313.57"),
            (4, "1,921", "1,309.82"),
            (5, "1,921", "1,309.82"),
        ]
    )
    result = extract.parse_log_file(
        str(write_log(tmp_path, body + "\n", framework="torchtitan", config_name="llama3.1_8B-BF16-pretrain"))
    )
    # Steps 4 and 5 remain after the warmup skip.
    assert result["hmean_tflops"] == pytest.approx(1309.82)
    assert result["hmean_tps_per_gpu"] == pytest.approx(1921.0)
    assert result["tflops_min"] == pytest.approx(1309.82)


def bridge_log(tmp_path, *, module="post_trainer", resolved_packed=None, yaml_packed=None, name="bridge.log"):
    """A Megatron-Bridge run: rank 0 prints the MODEL_TFLOP estimate, the last
    rank the iteration line; GBS 8 on 8 GPUs."""
    lines = []
    if resolved_packed is not None:
        lines.append(
            "[20261016 13:53:30][rank-0/8][INFO]     [------------module_utils.py:256] :   "
            f"packed_sequence                      : {resolved_packed} (bool)"
        )
    for step, ms in enumerate([25241.4, 19615.4, 1976.7, 800.0, 1000.0], start=1):
        lines.append(
            "[20261016 13:55:59][rank-0/8][DEBUG]    [------------------------common_utils.py:179] : "
            f"Step Time : {ms / 1000:.2f}s GPU utilization: 900.0MODEL_TFLOP/s/GPU"
        )
        lines.append(
            "[20261016 13:55:59][rank-7/8][DEBUG]    [--------common_utils.py:198] :  [2026-10-16 13:55:59] "
            f"iteration {step:8d}/     200 | consumed samples: {8 * step:12d} | "
            f"elapsed time per iteration (ms): {ms} | learning rate: 3.000000E-07 | "
            "global batch size:     8 | lm loss: 3.752264E+00 |"
        )
    config = f"modules:\n  {module}:\n    framework: megatron_bridge\n    overrides:\n"
    if yaml_packed is not None:
        config += f"      packed_sequence: {yaml_packed}\n"
    suite = "posttrain" if module == "post_trainer" else "pretrain"
    return extract.parse_log_file(
        str(
            write_log(
                tmp_path,
                "\n".join(lines) + "\n",
                framework="megatron_bridge",
                config_name=f"qwen3_32b_sft_{suite}",
                config_body=config,
                name=name,
                banner_extra=f"# Train Module     : {module}  (CLI verb: train {suite})\n",
            )
        )
    )


@pytest.mark.parametrize(
    "resolved_packed,yaml_packed",
    [
        ("False", "false"),
        # Megatron-Bridge does not pack unless told to.
        (None, None),
        # The resolved value is what ran.
        ("False", "true"),
    ],
)
def test_bridge_post_training_without_packing_reports_samples_not_tflops(
    tmp_path, resolved_packed, yaml_packed
):
    result = bridge_log(tmp_path, resolved_packed=resolved_packed, yaml_packed=yaml_packed)
    assert result["hmean_tflops"] is None
    assert result["tflops_cv"] is None
    # The estimate is kept, under its own name.
    assert result["hmean_nominal_tflops"] == pytest.approx(900.0)
    assert result["throughput_note"].startswith("nominal TFLOP/s, not TFLOP/s")
    # Steps 4 and 5: 8 samples in 0.8 s and in 1.0 s, on 8 GPUs.
    assert result["num_post_warmup"] == 2
    assert result["hmean_samples_per_gpu"] == pytest.approx(2 / (1 / 1.25 + 1 / 1.0))
    assert "8192 tokens" in result["throughput_note"]
    assert result["status"] == "ok"


@pytest.mark.parametrize(
    "module,resolved_packed,yaml_packed",
    [
        ("post_trainer", "True", None),
        ("post_trainer", None, "true"),
        ("pre_trainer", None, None),
    ],
)
def test_bridge_tflops_estimate_is_kept_where_sequences_are_full(
    tmp_path, module, resolved_packed, yaml_packed
):
    result = bridge_log(tmp_path, module=module, resolved_packed=resolved_packed, yaml_packed=yaml_packed)
    assert result["hmean_tflops"] == pytest.approx(900.0)
    assert result["hmean_nominal_tflops"] is None
    assert result["hmean_samples_per_gpu"] is None
    assert result["throughput_note"] is None


# --------------------------------------------------------------------------
# Provenance plumbed through to the CSV row
# --------------------------------------------------------------------------


def test_provenance_fields_reach_the_result(tmp_path):
    result = extract.parse_log_file(str(write_log(tmp_path, MEGATRON_AVG)))

    assert result["docker_image"] == "registry.example/ci:some-tag"
    assert result["image_digest"] == "registry.example/ci@sha256:deadbeef"
    assert result["primus_commit"] == "137d6c256119"
    assert result["submodule_pins"] == "43f426fa"
    assert result["gpu_model"] == "AMD Instinct MI325X"
    assert result["rocm_version"] == "7.2.1"
    assert result["torch_version"] == "2.12.0+rocm10.0.0"
    assert result["world_size"] == 8
    assert result["exit_code"] == 0
    assert result["elapsed_sec"] == 600


def test_unknown_placeholders_become_empty(tmp_path):
    """`unknown (...)` reads fine in a log but is noise in a CSV cell."""
    result = extract.parse_log_file(str(write_log(tmp_path, MEGATRON_AVG)))
    assert result["jax_version"] is None


def test_every_result_key_is_a_csv_field(tmp_path):
    """The writer uses extrasaction='ignore', so a typo'd key vanishes."""
    result = extract.parse_log_file(str(write_log(tmp_path, MEGATRON_AVG)))
    missing = set(extract.CSV_FIELDS) - set(result)
    assert not missing, f"CSV_FIELDS not produced by the parser: {sorted(missing)}"


def test_failed_run_still_parses(tmp_path):
    result = extract.parse_log_file(str(write_log(tmp_path, "", exit_code=1)))
    assert result["exit_code"] == 1
    assert result["num_iterations"] == 0
    assert result["hmean_tflops"] is None


# --------------------------------------------------------------------------
# Legacy header formats
# --------------------------------------------------------------------------


def test_legacy_exp_header_still_parses(tmp_path):
    log = tmp_path / "legacy_exp.log"
    log.write_text(
        "starting benchmark\n"
        "some preamble\n"
        "config seq_len=4096\n"
        "EXP=examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml\n"
        "CMD: --micro_batch_size 2 --global_batch_size 16 --train_iters 10\n"
        "repeat: 2\n" + MEGATRON_LEGACY
    )
    result = extract.parse_log_file(str(log))
    assert result is not None
    assert result["backend"] == "megatron"
    assert result["model_name"] == "llama3.1_8B"
    assert result["precision"] == "BF16"
    assert result["repeat"] == 2
    # Provenance columns simply stay empty for logs that predate them.
    assert result["docker_image"] is None
    assert result["world_size"] is None


def test_non_primus_log_is_skipped(tmp_path):
    log = tmp_path / "random.log"
    log.write_text("\n".join(f"line {i}" for i in range(20)))
    assert extract.parse_log_file(str(log)) is None


# --------------------------------------------------------------------------
# Status and failure reasons
# --------------------------------------------------------------------------


def test_successful_run_is_ok_without_a_reason(tmp_path):
    result = extract.parse_log_file(str(write_log(tmp_path, MEGATRON_AVG, failure_reason="none")))
    assert result["status"] == "ok"
    assert result["failure_reason"] is None


@pytest.mark.parametrize(
    "body,exit_code,footer_reason,expected",
    [
        pytest.param(
            "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB. GPU 0 has a "
            "total capacity of 255.98 GiB of which 34.00 MiB is free. Of the allocated memory "
            "250.10 GiB is allocated by PyTorch\n",
            1,
            "exit: launcher exited with code 1",
            "oom",
            id="oom",
        ),
        pytest.param(
            "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB. GPU 0 has a "
            "total capacity of 255.98 GiB of which 34.00 MiB is free. Of the allocated memory "
            "10.23 GiB is allocated by PyTorch\n",
            1,
            "exit: launcher exited with code 1",
            "gpu_contended",
            id="oom-while-another-process-holds-the-gpu",
        ),
        pytest.param(
            "[rank-3/8] Signal 11 (SIGSEGV) received by PID 1234\nChildFailedError: boom\n",
            1,
            None,
            "segfault",
            id="segfault-from-old-footer",
        ),
        pytest.param(
            "RuntimeError: ncclRemoteError: remote process exited\n",
            1,
            "fail_pattern: failure text matching FAIL_PATTERN appeared in the output",
            "nccl_error",
            id="nccl-behind-a-fail-pattern-hit",
        ),
        pytest.param(
            "Last RCCL warning: 'Bootstrap : no socket interface found'\nRuntimeError: ncclInternalError\n",
            1,
            "exit: launcher exited with code 1",
            "nic_error",
            id="nic-beats-the-nccl-symptom",
        ),
        pytest.param(
            "Traceback (most recent call last):\nValueError: bad config value\n",
            1,
            "exit: launcher exited with code 1",
            "python_error",
            id="python-exception",
        ),
        pytest.param("training...\n", 137, None, "killed", id="sigkill-without-an-error"),
        pytest.param(
            "RuntimeError: ncclRemoteError\n",
            143,
            "idle: no new output for MAX_IDLE_SECONDS=1200s",
            "idle",
            id="runner-verdict-wins",
        ),
    ],
)
def test_failure_reason(tmp_path, body, exit_code, footer_reason, expected):
    result = extract.parse_log_file(
        str(write_log(tmp_path, body, exit_code=exit_code, failure_reason=footer_reason))
    )
    assert result["status"] == "failed"
    assert result["failure_reason"] == expected
    assert result["failure_detail"]


def test_old_footers_still_get_the_right_reason(tmp_path):
    # Older runners wrote no reason line: exit 130 was a Ctrl+C, 255 is not a signal.
    old_interrupt = extract.parse_log_file(str(write_log(tmp_path, "", exit_code=130, name="a.log")))
    assert (old_interrupt["status"], old_interrupt["failure_reason"]) == ("interrupted", "interrupted")
    exit_255 = extract.parse_log_file(str(write_log(tmp_path, "srun: error\n", exit_code=255, name="b.log")))
    assert exit_255["failure_reason"] == "exit"

    # The older multi-node runner flagged watchdog kills in its footer, even with exit 0.
    old_multinode = write_log(
        tmp_path,
        ">>> [12:00:00] TIMEOUT — attempt exceeded MAX_RUN_SECONDS\n",
        exit_code=0,
        name="m-abcd1234-MBS2-GBS16-rep1-failed-attempt1_20261001-000000.log",
    )
    old_multinode.write_text(
        old_multinode.read_text().replace(
            "# Elapsed (sec)   : 600\n", "# Elapsed (sec)   : 600\n# Failure pattern? : true\n"
        )
    )
    result = extract.parse_log_file(str(old_multinode))
    assert (result["status"], result["failure_reason"]) == ("failed", "timeout")


def test_contention_reads_every_unit_and_counts_reserved_memory(tmp_path):
    contended = (
        "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total "
        "capacity of 287.98 GiB of which 0 bytes is free. Of the allocated memory 0 bytes is "
        "allocated by PyTorch, and 0 bytes is reserved by PyTorch but unallocated.\n"
    )
    fragmented = (
        "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total "
        "capacity of 287.98 GiB of which 1.00 GiB is free. Including non-PyTorch memory, this "
        "process has 277.00 GiB memory in use. Of the allocated memory 100.00 GiB is allocated by "
        "PyTorch, and 170.00 GiB is reserved by PyTorch but unallocated.\n"
    )
    a = extract.parse_log_file(str(write_log(tmp_path, contended, exit_code=1, name="a.log")))
    b = extract.parse_log_file(str(write_log(tmp_path, fragmented, exit_code=1, name="b.log")))
    assert a["failure_reason"] == "gpu_contended"
    assert b["failure_reason"] == "oom"


@pytest.mark.parametrize(
    "body,expected",
    [
        pytest.param(
            "RuntimeError: HIP OUT OF MEMORY. GPU 0 has a total capacity of 287.98 gib of which 1 mib "
            "is free. Of the allocated memory 250.00 gib is allocated by PyTorch\n",
            "oom",
            id="upper-case-message-lower-case-units",
        ),
        pytest.param(
            "*** SIGSEGV (@0x0) received by PID 1234 (TID 0x7f) from PID 0\n", "segfault", id="glog-segv"
        ),
    ],
)
def test_failure_text_variants(tmp_path, body, expected):
    result = extract.parse_log_file(str(write_log(tmp_path, body, exit_code=1)))
    assert result["failure_reason"] == expected


def test_exit_codes_tell_no_logs_from_errors(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr("sys.argv", ["extract_results.py", "--quiet", str(empty)])
    assert extract.main() == 3

    broken = tmp_path / "broken"
    broken.mkdir()
    write_log(broken, MEGATRON_AVG)
    monkeypatch.setattr(extract, "parse_log_file", lambda path: 1 / 0)
    monkeypatch.setattr("sys.argv", ["extract_results.py", "--quiet", str(broken)])
    assert extract.main() == 1


def test_primus_patch_lines_and_benign_exceptions_are_not_the_cause(tmp_path):
    body = (
        "[rank-0/8][INFO] [Patch] Applying megatron.dataloader_mp_context: spawn to avoid SIGSEGV\n"
        "[rank-0/8][WARNING] ModuleNotFoundError: No module named 'apex'\n"
        "[Patch] Skipped: x (backend version mismatch: 1.0 vs 2.0)\n"
        "Traceback (most recent call last):\n"
        '  File "train.py", line 1, in <module>\n'
        "AssertionError: micro batch size must divide the global batch size\n"
    )
    result = extract.parse_log_file(str(write_log(tmp_path, body, exit_code=1)))
    assert result["failure_reason"] == "python_error"
    assert result["failure_detail"].startswith("AssertionError")
    assert result["version_warnings"] == []


def primus_record(level: str, where: str, message: str) -> str:
    """A record of Primus's console logger, colour codes included: a multi-line
    message runs on unprefixed lines up to the reset."""
    return (
        f"[\x1b[32m20261021 05:06:54\x1b[0m][\x1b[36mrank-0/8\x1b[0m]\x1b[1m{f'[{level}]':<11}\x1b[0m"
        f"\x1b[1m[-------{where}] : {message}\x1b[0m\n"
    )


# Trimmed from a v26.9 rc1 Megatron-Bridge failure: a patch skipped for a
# missing dependency logs an ImportError traceback first; the run then stops
# on a bare AssertionError, which Primus wraps and torchrun summarizes.
RC1_BRIDGE_FAILURE = (
    primus_record(
        "ERROR",
        "patch_runner.py:132",
        "[Patch] ✗ Patch 'megatron.turbo.te_spec_provider' failed due to missing dependency: "
        "cannot import name 'TEGroupedMLPSubmodules' from 'megatron.core.transformer.moe.experts'\n"
        "Traceback (most recent call last):\n"
        '  File "/primus/core/patches/patch_runner.py", line 127, in run_patches\n'
        "    patch.apply(ctx)\n"
        "ImportError: cannot import name 'TEGroupedMLPSubmodules' from 'megatron.core.transformer.moe.experts'\n",
    )
    + primus_record("INFO", "train_runtime.py:559", "Python GC re-enabled after training")
    + primus_record("ERROR", "traceback.py:1050", "Traceback (most recent call last):\n")
    + primus_record(
        "ERROR",
        "traceback.py:1050",
        '  File "/primus/core/runtime/train_runtime.py", line 108, in run_train_module\n'
        "    self._initialize_backend_and_execute()\n",
    )
    + primus_record(
        "ERROR",
        "traceback.py:1050",
        '  File "/Megatron-LM/megatron/core/optimizer_param_scheduler.py", line 74, in __init__\n'
        "    assert self.lr_warmup_steps < self.lr_decay_steps\n"
        "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
    )
    + primus_record("ERROR", "traceback.py:1050", "AssertionError\n")
    + primus_record(
        "ERROR",
        "traceback.py:1050",
        "\nThe above exception was the direct cause of the following exception:\n\n",
    )
    + primus_record("ERROR", "traceback.py:1050", "Traceback (most recent call last):\n")
    + primus_record(
        "ERROR",
        "traceback.py:1050",
        '  File "/primus/core/runtime/train_runtime.py", line 121, in run_train_module\n'
        '    raise RuntimeError(f"Training execution failed: {e}") from e\n',
    )
    + primus_record("ERROR", "traceback.py:1050", "RuntimeError: Training execution failed: \n")
    + primus_record(
        "ERROR",
        "__main__.py:220",
        "[Primus] Error: RuntimeError: Training execution failed:  (/primus/core/runtime/train_runtime.py:121)",
    )
    + "E1021 05:06:59.430000 1586 torch/distributed/elastic/multiprocessing/api.py:1007] failed (exitcode: 1) "
    "local_rank: 5 (pid: 1758) of binary: /opt/venv/bin/python3\n"
    "Traceback (most recent call last):\n"
    '  File "/opt/venv/bin/torchrun", line 10, in <module>\n'
    "    sys.exit(main())\n"
    "torch.distributed.elastic.multiprocessing.errors.ChildFailedError: \n"
    "============================================================\n"
    "primus/cli/main.py FAILED\n"
    "[2026-10-21 05:06:59] [NODE-0(node)] [ERROR] [direct] torchrun exited with code 1\n"
)


@pytest.mark.parametrize("colored", [True, False], ids=["colored", "no-colors"])
def test_the_exception_that_stopped_the_run_is_the_detail(tmp_path, colored):
    body = RC1_BRIDGE_FAILURE if colored else extract.strip_ansi(RC1_BRIDGE_FAILURE)
    result = extract.parse_log_file(
        str(write_log(tmp_path, body, exit_code=1, failure_reason="exit: launcher exited with code 1"))
    )
    assert (result["status"], result["failure_reason"]) == ("failed", "python_error")
    assert result["failure_detail"] == (
        "AssertionError: assert self.lr_warmup_steps < self.lr_decay_steps (optimizer_param_scheduler.py:74)"
    )
    assert result["failed_at"] == "setup"


def _traceback(*frames_then_exception: str) -> str:
    *frames, exception = frames_then_exception
    lines = ["Traceback (most recent call last):"]
    for frame in frames:
        lines += [f'  File "/src/{frame}.py", line 7, in {frame}', f"    {frame}()"]
    return "\n".join(lines + [exception]) + "\n"


CAUSE = "\nThe above exception was the direct cause of the following exception:\n\n"
CONTEXT = "\nDuring handling of the above exception, another exception occurred:\n\n"


@pytest.mark.parametrize(
    "body,reason,detail",
    [
        pytest.param(
            _traceback("load", "KeyError: 'lr'")
            + CAUSE
            + _traceback("setup", "ValueError: the config has no lr"),
            "python_error",
            "KeyError: 'lr'",
            id="raise-from-goes-to-the-cause",
        ),
        pytest.param(
            _traceback("load", "KeyError: 'lr'")
            + CONTEXT
            + _traceback("setup", "ValueError: the config has no lr"),
            "python_error",
            "ValueError: the config has no lr",
            id="an-error-while-handling-one-is-the-new-error",
        ),
        pytest.param(
            _traceback("probe", "ImportError: optional apex")
            + "training...\n"
            + _traceback("step", "ValueError: bad"),
            "python_error",
            "ValueError: bad",
            id="a-logged-traceback-before-the-fatal-one",
        ),
        pytest.param(
            _traceback("step", "torch.distributed.DistBackendError: NCCL error in: ProcessGroupNCCL.cpp:1234")
            + CAUSE
            + _traceback(
                "train", "RuntimeError: Training execution failed: NCCL error in: ProcessGroupNCCL.cpp:1234"
            ),
            "nccl_error",
            "torch.distributed.DistBackendError: NCCL error in: ProcessGroupNCCL.cpp:1234",
            id="an-nccl-root-cause",
        ),
        pytest.param(
            _traceback("step", "ValueError: bad") + _traceback("main", "ChildFailedError: "),
            "python_error",
            "ValueError: bad",
            id="torchrun-summary-is-not-the-cause",
        ),
        pytest.param(
            "".join(f"[rank3]: {line}\n" for line in _traceback("step", "ValueError: bad").splitlines()),
            "python_error",
            "ValueError: bad",
            id="torch-rank-prefix",
        ),
        pytest.param(
            _traceback("load", "KeyError: 'lr'\nsee the recipe's optimizer section")
            + CAUSE
            + _traceback("setup", "ValueError: the config has no lr"),
            "python_error",
            "KeyError: 'lr'",
            id="multi-line-message-before-the-link",
        ),
        pytest.param(
            "Root Cause (first observed failure):\n[0]:\n  rank      : 5 (local_rank: 5)\n"
            "  traceback : Traceback (most recent call last):\n"
            '    File "/src/sched.py", line 74, in __init__\n'
            "      assert self.lr_warmup_steps < self.lr_decay_steps\n"
            "  AssertionError\n"
            "============================================================\n",
            "python_error",
            "AssertionError: assert self.lr_warmup_steps < self.lr_decay_steps (sched.py:74)",
            id="torchrun-summary-with-an-error-file",
        ),
    ],
)
def test_traceback_chains(tmp_path, body, reason, detail):
    result = extract.parse_log_file(str(write_log(tmp_path, body, exit_code=1)))
    assert (result["failure_reason"], result["failure_detail"]) == (reason, detail)


def test_another_ranks_lines_inside_a_traceback_are_skipped(tmp_path):
    body = (
        primus_record("ERROR", "traceback.py:1050", "Traceback (most recent call last):\n")
        + primus_record("ERROR", "traceback.py:1050", '  File "/src/a.py", line 3, in f\n    g()\n')
        + primus_record("INFO", "training.py:9", "iteration 3/20").replace("rank-0/8", "rank-7/8")
        + "W1021 05:06:58.801000 1586 torch/distributed/elastic/multiprocessing/api.py:1033] Sending process 17 closing signal SIGTERM\n"
        + primus_record("ERROR", "traceback.py:1050", "ValueError: bad shard\n")
        + primus_record("ERROR", "__main__.py:220", "[Primus] Error: ValueError: bad shard (/src/a.py:3)")
    )
    result = extract.parse_log_file(str(write_log(tmp_path, body, exit_code=1)))
    assert (result["failure_reason"], result["failure_detail"]) == ("python_error", "ValueError: bad shard")


def test_an_uncoloured_patch_message_does_not_hide_what_follows(tmp_path):
    body = extract.strip_ansi(
        primus_record("INFO", "patch_runner.py:126", "[Patch] Applying maxdiffusion.te_preload: preload TF")
    ) + ("Fatal Python error: Segmentation fault\n" + _traceback("load", "RuntimeError: died"))
    result = extract.parse_log_file(str(write_log(tmp_path, body, exit_code=139)))
    assert result["failure_reason"] == "segfault"


def test_a_loguru_traceback_is_read(tmp_path):
    body = (
        "Traceback (most recent call last):\n"
        '> File "/src/train.py", line 12, in <module>\n'
        "    main()\n"
        '  File "/src/train.py", line 9, in main\n'
        "    step()\n"
        "ValueError: loss is NaN\n"
    )
    result = extract.parse_log_file(
        str(write_log(tmp_path, "ModuleNotFoundError: No module named 'apex'\n" + body, exit_code=1))
    )
    assert (result["failure_reason"], result["failure_detail"]) == ("python_error", "ValueError: loss is NaN")


def test_nccl_text_still_beats_a_generic_exception(tmp_path):
    body = "[Rank 0] Watchdog caught collective operation timeout: WorkNCCL(OpType=ALLREDUCE)\n" + _traceback(
        "step", "RuntimeError: Rank 0 received an exception from rank 7"
    )
    result = extract.parse_log_file(str(write_log(tmp_path, body, exit_code=1)))
    assert result["failure_reason"] == "nccl_error"
    assert result["failure_detail"].startswith("Watchdog caught collective operation timeout")


def test_a_bare_exception_other_than_assert_gets_its_location_only(tmp_path):
    result = extract.parse_log_file(
        str(write_log(tmp_path, _traceback("next_batch", "StopIteration"), exit_code=1))
    )
    assert result["failure_detail"] == "StopIteration (next_batch.py:7)"


@pytest.mark.parametrize("colored", [True, False], ids=["colored", "no-colors"])
def test_a_patch_messages_chained_traceback_is_noise_too(tmp_path, colored):
    patch = primus_record(
        "ERROR",
        "patch_runner.py:132",
        "[Patch] ✗ Patch 'x' failed due to missing dependency: need triton\n"
        + _traceback("probe", "ModuleNotFoundError: No module named 'triton'")
        + CAUSE
        + _traceback("patch", "ImportError: need triton"),
    )
    body = (patch if colored else extract.strip_ansi(patch)) + _traceback("step", "ValueError: bad shard")
    result = extract.parse_log_file(str(write_log(tmp_path, body, exit_code=1)))
    assert result["failure_detail"] == "ValueError: bad shard"


def test_a_link_line_only_joins_the_traceback_right_after_it(tmp_path):
    body = (
        _traceback("probe", "ValueError: caught and logged")
        + CAUSE
        + "training resumed\n"
        + _traceback("step", "RuntimeError: the real crash")
    )
    result = extract.parse_log_file(str(write_log(tmp_path, body, exit_code=1)))
    assert result["failure_detail"] == "RuntimeError: the real crash"


def test_a_c10_warning_inside_a_rank_traceback_is_skipped(tmp_path):
    lines = [
        "Traceback (most recent call last):",
        '  File "/src/sched.py", line 74, in __init__',
        "[W1021 05:04:29.512 ProcessGroupNCCL.cpp:1250] Warning: destroy_process_group() was not called",
        "    assert self.lr_warmup_steps < self.lr_decay_steps",
        "AssertionError",
    ]
    body = "".join(f"[rank3]: {line}\n" for line in lines)
    result = extract.parse_log_file(str(write_log(tmp_path, body, exit_code=1)))
    assert (
        result["failure_detail"]
        == "AssertionError: assert self.lr_warmup_steps < self.lr_decay_steps (sched.py:74)"
    )


def test_an_interrupt_with_a_stray_exception_line_keeps_the_signal(tmp_path):
    body = "[WARNING] ImportError: optional apex missing\n" + _traceback("train", "KeyboardInterrupt")
    result = extract.parse_log_file(
        str(
            write_log(
                tmp_path, body, exit_code=130 + 13, failure_reason="signal: killed by SIGTERM (exit 143)"
            )
        )
    )
    assert result["failure_reason"] == "signal"


def test_an_interrupt_traceback_leaves_the_runners_verdict(tmp_path):
    body = _traceback("train", "KeyboardInterrupt")
    result = extract.parse_log_file(
        str(write_log(tmp_path, body, exit_code=143, failure_reason="signal: killed by SIGTERM (exit 143)"))
    )
    assert result["failure_reason"] == "signal"


def test_exit_zero_without_a_training_step_is_no_metrics(tmp_path):
    result = extract.parse_log_file(str(write_log(tmp_path, "training finished\n", failure_reason="none")))
    assert (result["status"], result["failure_reason"], result["failed_at"]) == ("failed", "no_metrics", None)


def test_failed_at_tells_setup_from_training(tmp_path):
    crashed = extract.parse_log_file(
        str(write_log(tmp_path, MEGATRON_AVG + "RuntimeError: boom\n", exit_code=1, name="a.log"))
    )
    hung = extract.parse_log_file(
        str(
            write_log(
                tmp_path,
                "compiling...\n",
                exit_code=143,
                failure_reason="idle: no new output for MAX_IDLE_SECONDS=1200s",
                name="b.log",
            )
        )
    )
    ok = extract.parse_log_file(str(write_log(tmp_path, MEGATRON_AVG, failure_reason="none", name="c.log")))
    assert (crashed["failed_at"], hung["failed_at"], ok["failed_at"]) == ("after step 5", "setup", None)


# --------------------------------------------------------------------------
# The failing rank's own log
# --------------------------------------------------------------------------

failure = extract.failure


def torchrun_summary(
    rank: int, local_rank: int, exitcode: int, time: str, host: str = "node-1.cluster"
) -> str:
    return (
        "Root Cause (first observed failure):\n[0]:\n"
        f"  time      : {time}\n  host      : {host}\n"
        f"  rank      : {rank} (local_rank: {local_rank})\n  exitcode  : {exitcode} (pid: 1758)\n"
        "  error_file: <N/A>\n  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html\n"
        "============================================================\n"
    )


def test_failing_rank_is_the_earliest_root_cause():
    lines = (
        torchrun_summary(9, 1, -15, "2026-10-22_18:53:20", "node-2")
        + torchrun_summary(3, 3, 1, "2026-10-22_18:53:14", "node-1")
    ).splitlines(keepends=True)
    assert failure.failing_rank(lines) == {
        "rank": 3,
        "local_rank": 3,
        "host": "node-1",
        "exitcode": 1,
        "time": "2026-10-22_18:53:14",
    }
    assert failure.failing_rank(["training...\n"]) is None


def rank_record(time: str, level: str, message: str) -> str:
    """A record of Primus's per-rank log files (no colours)."""
    return f"[{time}][root/amd][pre_trainer][ip-localhost][rank-3/8]{f'[{level}]':<11}[-----traceback.py:1050] : {message}\n"


def failing_rank_run(tmp_path, with_rank_log: bool = True) -> Path:
    """A log whose rank 3 failed first, with rank 3's error.log in a workspace
    given through EXTRA_ENV; the log has no section yet."""
    workspace = tmp_path / "ws"
    rank_dir = workspace / "amd" / "root" / "exp1" / "logs" / "pre_trainer" / "rank-3"
    if with_rank_log:
        rank_dir.mkdir(parents=True)
        (rank_dir / "error.log").write_text(
            rank_record("20261021 09:00:00", "ERROR", "ValueError: an earlier run of this experiment")
            + rank_record(
                "20261022 18:52:05",
                "ERROR",
                "[Patch] ✗ Patch 'x' failed due to missing dependency: no triton",
            )
            + "Traceback (most recent call last):\n"
            + '  File "/primus/core/patches/patch_runner.py", line 127, in run_patches\n'
            + "ModuleNotFoundError: No module named 'triton'\n"
            + rank_record("20261022 18:52:27", "ERROR", "Traceback (most recent call last):")
            + rank_record("20261022 18:52:27", "ERROR", '  File "/Megatron-LM/x.py", line 74, in __init__')
            + "    assert self.lr_warmup_steps < self.lr_decay_steps\n"
            + rank_record("20261022 18:52:27", "ERROR", "AssertionError")
            + rank_record(
                "20261022 18:52:27",
                "ERROR",
                "[Primus] Error: RuntimeError: Training execution failed:  (x.py:121)",
            )
        )
    body = (
        primus_record("INFO", "train_runtime.py:90", "starting").replace(
            "20261021 05:06:54", "20261022 18:52:00"
        )
        + "torch.distributed.elastic.multiprocessing.errors.ChildFailedError: \n"
        + torchrun_summary(3, 3, 1, "2026-10-22_18:52:29")
    )
    log = write_log(
        tmp_path,
        body,
        exit_code=1,
        failure_reason="exit: launcher exited with code 1",
        with_footer=False,
        config_body="workspace: ${PRIMUS_WORKSPACE:./output}\nexp_name: ${PRIMUS_EXP_NAME:exp1}\nmodules:\n  pre_trainer:\n    framework: megatron\n",
        banner_extra=f"# Train Module     : pre_trainer  (CLI verb: train pretrain)\n# Extra env (--env): PRIMUS_WORKSPACE={workspace}\n",
    )
    return log


def finish_log(log: Path, section: list[str] | None) -> None:
    """End the log the way the runner does: section, then footer."""
    footer = FOOTER_TEMPLATE.format(
        exit_code=1, failure_line="# Failure reason   : exit: launcher exited with code 1\n"
    )
    head, _, tail = footer.partition("\n\n")
    log.write_text(log.read_text() + head + "\n\n" + ("\n".join(section) + "\n\n" if section else "") + tail)


def test_the_failing_ranks_own_records_are_the_cause(tmp_path):
    log = failing_rank_run(tmp_path)
    spec = failure.failing_rank_spec(str(log), str(tmp_path), since=0)
    assert (spec["rank"], spec["workspace"], spec["exp_name"], spec["module"]) == (
        3,
        str(tmp_path / "ws"),
        "exp1",
        "pre_trainer",
    )
    section = failure.failing_rank_section(spec)
    assert section[1] == "# Failing rank     : 3 (local rank 3) on node-1.cluster, exit code 1"
    assert "an earlier run" not in "\n".join(section)
    finish_log(log, section)

    result = extract.parse_log_file(str(log))
    assert (result["failing_rank"], result["failing_host"]) == (3, "node-1.cluster")
    assert (result["failure_reason"], result["failure_detail"]) == (
        "python_error",
        "AssertionError: assert self.lr_warmup_steps < self.lr_decay_steps (x.py:74)",
    )


def test_a_missing_rank_log_still_names_the_rank(tmp_path):
    log = failing_rank_run(tmp_path, with_rank_log=False)
    section = failure.failing_rank_section(failure.failing_rank_spec(str(log), str(tmp_path)))
    assert section[2].startswith("# Source           : no rank-3 log under")
    finish_log(log, section)
    result = extract.parse_log_file(str(log))
    assert result["failing_rank"] == 3
    assert result["failure_reason"] == "exit"


def test_a_back_to_back_run_of_the_same_experiment_is_left_out(tmp_path):
    log = failing_rank_run(tmp_path)
    error_log = tmp_path / "ws" / "amd" / "root" / "exp1" / "logs" / "pre_trainer" / "rank-3" / "error.log"
    previous = rank_record("20261022 18:51:50", "ERROR", "ValueError: the previous run's error")
    error_log.write_text(previous + error_log.read_text())
    # The run started at 18:51:58 UTC; its first Primus line is at 18:52:00.
    since = datetime(2026, 10, 22, 18, 51, 58, tzinfo=timezone.utc).timestamp()
    section = failure.failing_rank_section(failure.failing_rank_spec(str(log), str(tmp_path), since=since))
    assert "previous run" not in "\n".join(section)
    assert any("AssertionError" in line for line in section)


def test_a_specific_cause_in_rank_0s_output_beats_the_sections(tmp_path):
    section = [
        "#### Begin Failing Rank Output ####",
        "# Failing rank     : 3 (local rank 3) on node-1, exit code 1",
        f"# Source           : /x/rank-3/error.log ({failure.OWN_RECORDS})",
        rank_record(
            "20261022 18:52:27",
            "ERROR",
            "RuntimeError: NCCL error in: ProcessGroupNCCL.cpp:1 ncclSystemError",
        ).rstrip(),
        "#### End Failing Rank Output ####",
    ]
    log = write_log(
        tmp_path,
        "Last RCCL warning: 'Bootstrap : no socket interface found'\n",
        exit_code=1,
        with_footer=False,
    )
    finish_log(log, section)
    assert extract.parse_log_file(str(log))["failure_reason"] == "nic_error"


def test_the_failing_ranks_exception_beats_the_nccl_errors_it_left_behind(tmp_path):
    section = [
        "#### Begin Failing Rank Output ####",
        "# Failing rank     : 3 (local rank 3) on node-1, exit code 1",
        f"# Source           : /x/rank-3/error.log ({failure.OWN_RECORDS})",
        rank_record("20261022 18:52:27", "ERROR", "ValueError: bad shard").rstrip(),
        "#### End Failing Rank Output ####",
    ]
    log = write_log(
        tmp_path,
        "RuntimeError: NCCL error in: ProcessGroupNCCL.cpp:1 ncclRemoteError\n",
        exit_code=1,
        with_footer=False,
    )
    finish_log(log, section)
    result = extract.parse_log_file(str(log))
    assert (result["failure_reason"], result["failure_detail"]) == ("python_error", "ValueError: bad shard")


def nccl_error(name: str, last_error: str) -> str:
    """PyTorch's DistBackendError message, as v26.9 rc2 prints it."""
    return (
        "torch.distributed.DistBackendError: NCCL error in: /__w/pytorch/torch/csrc/distributed/c10d/"
        f"NCCLUtils.cpp:91, unhandled {name} error (run with NCCL_DEBUG=INFO for details), NCCL version 2.32.3\n"
        f"ncclUnhandledCudaError: Call to CUDA function failed.\nLast error:\n{last_error}"
    )


def test_rccls_failed_allocation_on_the_failing_rank_is_an_oom(tmp_path):
    # Trimmed from zaya1_8B-FP8 on v26.9 rc2: rank 7 could not get NCCL a
    # buffer for the evaluation all_reduce; rank 0 was only sent SIGTERM.
    records = [
        rank_record("20261022 18:52:27", "ERROR", "Traceback (most recent call last):"),
        rank_record(
            "20261022 18:52:27",
            "ERROR",
            '  File "/src/evaluator.py", line 168, in primus_evaluate\n    torch.distributed.all_reduce(',
        ),
        rank_record(
            "20261022 18:52:27", "ERROR", nccl_error("cuda", "Failed to CUDA calloc 536870912 bytes")
        ),
        rank_record("20261022 18:52:27", "ERROR", "[Primus] Error: Failed to CUDA calloc 536870912 bytes"),
    ]
    section = [
        "#### Begin Failing Rank Output ####",
        "# Failing rank     : 3 (local rank 3) on node-1, exit code 1",
        f"# Source           : /x/rank-3/error.log ({failure.OWN_RECORDS})",
        *"".join(records).splitlines(),
        "#### End Failing Rank Output ####",
    ]
    log = write_log(
        tmp_path, torchrun_summary(3, 3, 1, "2026-10-22_18:52:28"), exit_code=1, with_footer=False
    )
    finish_log(log, section)
    result = extract.parse_log_file(str(log))
    assert (result["failure_reason"], result["failure_detail"]) == (
        "oom",
        "Failed to CUDA calloc 536870912 bytes",
    )


SHORT_NCCL = "torch.distributed.DistBackendError: NCCL error: unhandled system error, NCCL version 2.32.3"
IB_TIMEOUT = "Call to ibv_modify_qp failed with 110 Connection timed out, on dev rdma4:1"


def torch_rank_lines(rank: int, text: str) -> str:
    return "".join(f"[rank{rank}]: {line}\n" for line in text.splitlines())


@pytest.mark.parametrize(
    "body,detail",
    [
        pytest.param(
            _traceback("step", nccl_error("system", IB_TIMEOUT)),
            f"{SHORT_NCCL} | Last error: {IB_TIMEOUT}",
            id="last-error",
        ),
        pytest.param(
            _traceback("step", nccl_error("system", ""))
            + CAUSE
            + _traceback("main", "RuntimeError: Training execution failed: NCCL error"),
            nccl_error("system", "").splitlines()[0],
            id="empty-last-error",
        ),
        pytest.param(
            torch_rank_lines(3, _traceback("step", nccl_error("system", IB_TIMEOUT).split("\nLast")[0]))
            + torch_rank_lines(5, "Last error:\nsocket timeout on rank 5")
            + torch_rank_lines(3, f"Last error:\n{IB_TIMEOUT}"),
            f"{SHORT_NCCL} | Last error: {IB_TIMEOUT}",
            id="interleaved-ranks",
        ),
        pytest.param(
            torch_rank_lines(3, nccl_error("system", IB_TIMEOUT)),
            f"{SHORT_NCCL} | Last error: {IB_TIMEOUT}",
            id="no-traceback",
        ),
        pytest.param(
            _traceback("step", nccl_error("system", "x" * 400)),
            (f"{SHORT_NCCL} | Last error: " + "x" * 400)[:300],
            id="long-last-error",
        ),
    ],
)
def test_an_nccl_errors_detail_keeps_nccls_last_error(tmp_path, body, detail):
    result = extract.parse_log_file(str(write_log(tmp_path, body, exit_code=1)))
    assert (result["failure_reason"], result["failure_detail"]) == ("nccl_error", detail)


@pytest.mark.parametrize(
    "first_line,last_error",
    [
        pytest.param("", "Failed to CUDA host alloc 1024 bytes", id="host-memory"),
        pytest.param(
            "Memory access fault by GPU node-2 (Agent handle: 0x1) on address 0x7f. Reason: Unknown.\n",
            "Failed to CUDA calloc 6291456 bytes",
            id="after-a-gpu-fault",
        ),
    ],
)
def test_a_failed_rccl_allocation_that_is_not_an_oom(tmp_path, first_line, last_error):
    body = first_line + _traceback("step", nccl_error("cuda", last_error))
    assert (
        extract.parse_log_file(str(write_log(tmp_path, body, exit_code=1)))["failure_reason"] == "nccl_error"
    )


def test_debug_log_lines_in_the_section_are_not_classified(tmp_path):
    section = [
        "#### Begin Failing Rank Output ####",
        "# Failing rank     : 3 (local rank 3) on node-1, exit code -11",
        "# Source           : /x/rank-3/error.log has no ERROR records from this run; the last lines of /x/rank-3/debug.log",
        rank_record("20261022 18:52:27", "DEBUG", "ValueError: handled and retried").rstrip(),
        "#### End Failing Rank Output ####",
    ]
    log = write_log(
        tmp_path, "[rank-3/8] Signal 11 (SIGSEGV) received by PID 1234\n", exit_code=1, with_footer=False
    )
    finish_log(log, section)
    result = extract.parse_log_file(str(log))
    assert (result["failure_reason"], result["failing_rank"]) == ("segfault", 3)


def test_another_experiments_rank_log_is_not_used(tmp_path):
    log = failing_rank_run(tmp_path, with_rank_log=False)
    other = tmp_path / "ws" / "amd" / "root" / "other_exp" / "logs" / "pre_trainer" / "rank-3"
    other.mkdir(parents=True)
    (other / "error.log").write_text(rank_record("20261022 18:52:30", "ERROR", "ValueError: not this run"))
    section = failure.failing_rank_section(failure.failing_rank_spec(str(log), str(tmp_path)))
    assert section[2].startswith("# Source           : no rank-3 log under")


def test_without_a_time_window_no_records_are_copied(tmp_path):
    log = failing_rank_run(tmp_path)
    log.write_text(log.read_text().replace("[\x1b[32m20261022 18:52:00\x1b[0m]", "[no timestamp]"))
    spec = failure.failing_rank_spec(str(log), str(tmp_path), since=0)
    assert spec["window_start"] is None
    section = failure.failing_rank_section(spec)
    assert len(section) == 4 and "no time to tell" in section[2]


def test_the_runner_writes_the_section_before_the_footer(tmp_path):
    log = failing_rank_run(tmp_path)
    script = (
        f"set -uo pipefail\nSCRIPT_DIR={MODULE_PATH.parent}\nPRIMUS_PATH={tmp_path}\nPERF_MULTINODE=0\n"
        f"source {MODULE_PATH.parent}/lib/common.sh\nsource {MODULE_PATH.parent}/lib/run.sh\n"
        f"source {MODULE_PATH.parent}/lib/control.sh\n"
        f'print_per_run_footer {log} 1 30 "exit: launcher exited with code 1" "$(perf_failing_rank_section {log} 0)"\n'
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    text = log.read_text()
    assert text.index("End Run Output") < text.index("Begin Failing Rank Output") < text.index("# Exit code")
    assert extract.parse_log_file(str(log))["failure_detail"].startswith("AssertionError: assert")


# --------------------------------------------------------------------------
# --verdict, and whether another attempt can help
# --------------------------------------------------------------------------


def test_verdict_prints_one_logs_outcome(tmp_path):
    log = write_log(tmp_path, _traceback("setup", "ValueError: bad config"), exit_code=1)
    out = subprocess.run(
        ["python3", str(MODULE_PATH), "--verdict", str(log)], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.rstrip("\n").split("\x1f") == [
        "failed",
        "python_error",
        "setup",
        "yes",
        "ValueError: bad config",
    ]


@pytest.mark.parametrize(
    "body,exit_code,footer,recurs",
    [
        pytest.param(_traceback("setup", "ValueError: bad config"), 1, None, "1", id="config-error-at-setup"),
        pytest.param(
            _traceback("setup", "omegaconf.errors.ConfigKeyError: Key 'x' is not in struct"),
            1,
            None,
            "1",
            id="omegaconf-error-at-setup",
        ),
        pytest.param(
            MEGATRON_AVG + _traceback("step", "ValueError: bad batch"), 1, None, "0", id="config-error-later"
        ),
        pytest.param(
            _traceback(
                "init",
                "torch.distributed.DistNetworkError: The server socket has failed to listen (EADDRINUSE)",
            ),
            1,
            None,
            "0",
            id="port-in-use-at-setup",
        ),
        pytest.param(
            _traceback("load", "OSError: [Errno 116] Stale file handle"),
            1,
            None,
            "0",
            id="stale-nfs-at-setup",
        ),
        pytest.param(
            _traceback("init", "RuntimeError: HIP error: hipErrorNoDevice"), 1, None, "0", id="runtime-error"
        ),
        pytest.param(
            "[WARNING] caught KeyError: 'foo' while probing, ignoring\n"
            "torch.distributed.DistStoreError: Timed out after 900 seconds waiting for clients\n",
            1,
            None,
            "0",
            id="a-warning-line-is-not-the-cause",
        ),
        pytest.param(
            _traceback(
                "load", "ValueError: Connection error, and we cannot find the requested files in the cache"
            ),
            1,
            None,
            "0",
            id="connection-error-as-a-value-error",
        ),
        pytest.param(
            _traceback("load", "ImportError: /nfs/x/_C.so: file too short"), 1, None, "0", id="import-error"
        ),
        pytest.param(
            _traceback("load", "ModuleNotFoundError: No module named 'flash_attn'"),
            1,
            None,
            "1",
            id="missing-module",
        ),
        pytest.param("torch.OutOfMemoryError: HIP out of memory.\n", 1, None, "0", id="one-oom"),
        pytest.param("Last RCCL warning: 'Bootstrap : no socket interface found'\n", 1, None, "1", id="nic"),
        pytest.param("RuntimeError: ncclRemoteError: remote process exited\n", 1, None, "0", id="nccl"),
        pytest.param(
            MEGATRON_AVG, 143, "timeout: still running after MAX_RUN_SECONDS=60s", "0", id="timeout"
        ),
    ],
)
def test_which_failures_another_attempt_would_repeat(tmp_path, body, exit_code, footer, recurs):
    log = write_log(tmp_path, body, exit_code=exit_code, failure_reason=footer)
    script = (
        f"set -uo pipefail\nSCRIPT_DIR={MODULE_PATH.parent}\nPERF_MULTINODE=1\n"
        f"source {MODULE_PATH.parent}/lib/common.sh\nsource {MODULE_PATH.parent}/lib/control.sh\n"
        f'perf_failure_verdict {log}; echo "$PERF_VERDICT_RECURS"; echo "$PERF_VERDICT"\n'
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    got, verdict = out.stdout.split("\n")[:2]
    assert got == recurs, verdict
    assert verdict


def test_without_a_verdict_the_attempt_is_retried(tmp_path):
    script = (
        f"set -uo pipefail\nSCRIPT_DIR={tmp_path}\nPERF_MULTINODE=1\n"
        f"source {MODULE_PATH.parent}/lib/common.sh\nsource {MODULE_PATH.parent}/lib/control.sh\n"
        f'perf_failure_verdict {tmp_path}/missing.log; echo "$PERF_VERDICT_RECURS|$PERF_VERDICT"\n'
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    assert (out.returncode, out.stdout.strip()) == (0, "0|")


# --------------------------------------------------------------------------
# TRAIN_STEPS against the learning-rate schedule
# --------------------------------------------------------------------------


def test_a_config_that_was_not_run_has_its_reason(tmp_path):
    why = "TRAIN_STEPS=10 is not more than lr_warmup_iters=50, and with lr_decay_iters unset ..."
    log = write_log(tmp_path, "", exit_code=2, failure_reason=f"not_run: {why}")
    result = extract.parse_log_file(str(log))
    assert (result["status"], result["failure_reason"], result["failure_detail"], result["failed_at"]) == (
        "failed",
        "not_run",
        why,
        None,
    )


def _mikefarah_yq() -> bool:
    import shutil

    if shutil.which("yq") is None:
        return False
    out = subprocess.run(["yq", "--version"], capture_output=True, text=True, check=False)
    return "mikefarah" in out.stdout + out.stderr


@pytest.mark.skipif(not _mikefarah_yq(), reason="needs mikefarah yq v4")
@pytest.mark.parametrize(
    "framework,overrides,train_steps,warmup",
    [
        pytest.param("megatron", {"lr_warmup_iters": 50}, 10, "50", id="warmup-longer"),
        pytest.param("megatron", {"lr_warmup_iters": 50}, 50, "50", id="warmup-as-long"),
        pytest.param("megatron", {}, 10, "40", id="module-default-warmup"),
        pytest.param("megatron", {"lr_warmup_iters": "${PRIMUS_WARMUP:50}"}, 10, "50", id="placeholder"),
        pytest.param("megatron", {"lr_warmup_iters": 50}, 51, None, id="longer-run"),
        pytest.param("megatron", {"lr_warmup_iters": 50, "lr_decay_iters": 1000}, 10, None, id="decay-set"),
        pytest.param("megatron", {"lr_warmup_iters": 50, "lr_warmup_fraction": 0.1}, 10, None, id="fraction"),
        pytest.param("megatron", {"lr_warmup_iters": 50, "train_samples": 1000}, 10, None, id="sample-based"),
        pytest.param("megatron_bridge", {"lr_warmup_iters": 50}, 10, None, id="ignored-backend"),
        pytest.param("torchtitan", {"lr_warmup_iters": 50}, 10, None, id="not-megatron"),
    ],
)
def test_train_steps_that_megatrons_schedule_rejects(tmp_path, framework, overrides, train_steps, warmup):
    lines = [
        f"modules:\n  pre_trainer:\n    framework: {framework}\n    config: pre_trainer.yaml\n    overrides:"
    ]
    lines += [
        f'      {key}: "{value}"' if isinstance(value, str) else f"      {key}: {value}"
        for key, value in overrides.items()
    ]
    config = tmp_path / "exp.yaml"
    config.write_text("\n".join(lines + ["      micro_batch_size: 1", ""]))
    script = (
        f"set -uo pipefail\nPRIMUS_PATH={REPO_ROOT}\nTRAIN_STEPS={train_steps}\n"
        f"source {MODULE_PATH.parent}/lib/common.sh\nsource {MODULE_PATH.parent}/lib/run.sh\n"
        f'train_steps_conflict {config} {framework} pre_trainer; echo "$?|$TRAIN_STEPS_CONFLICT"\n'
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    code, why = out.stdout.strip().split("|", 1)
    if warmup is None:
        assert (code, why) == ("0", "")
    else:
        assert code == "1"
        assert why.startswith(f"TRAIN_STEPS={train_steps} is not more than lr_warmup_iters={warmup},")


def _volume_yaml(volume):
    return f'\n      - "{volume}"' if volume else " []"


MOUNTED = "--volume /scratch/data:/scratch/data --env DATA_PATH=/scratch/data"
PASSED_ONLY = "--env DATA_PATH=/scratch/data"


@pytest.mark.parametrize(
    "data_path,mode,home_volume,runner_volume,extra_env,expected",
    [
        pytest.param("/scratch/data/", "container", None, None, "", (0, MOUNTED), id="mounted"),
        pytest.param(
            "{checkout}/data",
            "container",
            None,
            None,
            "",
            (0, "--env DATA_PATH={checkout}/data"),
            id="in-checkout",
        ),
        pytest.param(
            "{checkout}", "container", None, None, "", (0, "--env DATA_PATH={checkout}"), id="checkout"
        ),
        pytest.param(
            "/scratch/data",
            "container",
            "/nfs/d:/scratch/data/:ro",
            None,
            "",
            (0, PASSED_ONLY),
            id="home-volume",
        ),
        pytest.param(
            "/scratch/data", "container", None, "/scratch/data", "", (0, PASSED_ONLY), id="runner-volume"
        ),
        pytest.param(
            "/scratch/data",
            "container",
            "/other",
            "/scratch/data",
            "",
            (0, MOUNTED),
            id="runner-volume-not-loaded",
        ),
        pytest.param(
            "/scratch/data", "container", "/scratch/data2", None, "", (0, MOUNTED), id="other-volume"
        ),
        pytest.param("/scratch/data", "direct", None, None, "", (0, ""), id="direct-mode"),
        pytest.param("", "container", None, None, "", (0, ""), id="unset"),
        pytest.param("data", "container", None, None, "", (1, ""), id="relative"),
        pytest.param("/", "container", None, None, "", (1, ""), id="root"),
        pytest.param("/scratch/a:b", "container", None, None, "", (1, ""), id="colon"),
        pytest.param("", "container", None, None, "A=1 DATA_PATH=/x", (0, ""), id="only-in-extra-env"),
        pytest.param(
            "/scratch/data",
            "container",
            None,
            None,
            "DATA_PATH=/scratch/data/",
            (0, MOUNTED),
            id="both-agree",
        ),
        pytest.param("/scratch/data", "container", None, None, "DATA_PATH=/x", (1, ""), id="both-differ"),
        pytest.param("/scratch/data", "container", None, None, "DATA_PATH=", (1, ""), id="extra-env-empty"),
    ],
)
def test_data_path_reaches_the_container(
    tmp_path, data_path, mode, home_volume, runner_volume, extra_env, expected
):
    checkout = tmp_path / "checkout"
    (checkout / "runner").mkdir(parents=True)
    (checkout / "runner" / "lib").symlink_to(REPO_ROOT / "runner" / "lib")
    (checkout / "runner" / ".primus.yaml").write_text(
        f"container:\n  options:\n    volume:{_volume_yaml(runner_volume)}\n"
    )
    if home_volume:
        (tmp_path / ".primus.yaml").write_text(
            f"container:\n  options:\n    volume:{_volume_yaml(home_volume)}\n"
        )
    guards = "export __PRIMUS_COMMON_SOURCED=1 __PRIMUS_CONFIG_SOURCED=1\n"
    script = (
        f"set -uo pipefail\n{guards}HOME={tmp_path}\nPRIMUS_PATH={checkout}\nPRIMUS_MODE={mode}\n"
        f"DATA_PATH='{data_path.format(checkout=checkout)}'\nEXTRA_ENV='{extra_env}'\n"
        f"source {MODULE_PATH.parent}/lib/common.sh\nsource {MODULE_PATH.parent}/lib/control.sh\n"
        "perf_parse_extra_env || exit 9\n"
        'perf_prepare_data_path 2>/dev/null; echo "$?|${PERF_DATA_ARGS[*]}|$PERF_DATA_CHECK"\n'
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    code, args, check = out.stdout.strip().split("|", 2)
    assert (int(code), args) == (expected[0], expected[1].format(checkout=checkout))
    if extra_env and not data_path:
        assert check == "/x (from EXTRA_ENV; not mounted by the runner)"


def test_data_path_is_created_before_the_first_run(tmp_path):
    target = tmp_path / "new" / "data"
    (tmp_path / "file").write_text("")
    script = (
        f"set -uo pipefail\nHOME={tmp_path}\nPRIMUS_PATH={REPO_ROOT}\nPRIMUS_MODE=container\n"
        f"DATA_PATH={target}\nsource {MODULE_PATH.parent}/lib/common.sh\n"
        f"source {MODULE_PATH.parent}/lib/control.sh\n"
        'perf_prepare_data_path && perf_ensure_data_path >/dev/null; echo "rc=$? $PERF_DATA_CHECK"\n'
        f'PERF_DATA_PATH={tmp_path}/file/data; perf_ensure_data_path; echo "rc=$?"\n'
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    created, error, refused = out.stdout.strip().splitlines()
    assert target.is_dir()
    assert created.startswith(f"rc=0 {target} (") and created.endswith("; mounted by the runner)")
    assert error.startswith(f"[ERROR] DATA_PATH={tmp_path}/file/data cannot be created on every node: mkdir:")
    assert refused == "rc=1"


def test_legacy_log_without_footer_is_not_guessed(tmp_path):
    log = tmp_path / "legacy_exp.log"
    log.write_text(
        "starting benchmark\nsome preamble\nconfig seq_len=4096\n"
        "EXP=examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml\n"
        "CMD: --micro_batch_size 2\nrepeat: 1\n"
        "[WARNING] ModuleNotFoundError: No module named 'apex'\n" + MEGATRON_LEGACY
    )
    result = extract.parse_log_file(str(log))
    assert result["status"] == "unknown"
    assert result["failure_reason"] is None


def test_interrupted_and_incomplete_runs(tmp_path):
    interrupted = extract.parse_log_file(
        str(
            write_log(
                tmp_path,
                "",
                exit_code=130,
                failure_reason="interrupted: the batch was stopped by a signal",
                name="a.log",
            )
        )
    )
    assert interrupted["status"] == "interrupted"
    assert interrupted["failure_reason"] == "interrupted"

    incomplete = extract.parse_log_file(
        str(write_log(tmp_path, MEGATRON_AVG, with_footer=False, name="b.log"))
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["exit_code"] is None
    # The throughput up to the kill is still reported.
    assert incomplete["num_iterations"] == 5


def test_failed_attempt_is_flagged(tmp_path):
    result = extract.parse_log_file(
        str(
            write_log(
                tmp_path,
                "",
                exit_code=143,
                failure_reason="timeout: still running after MAX_RUN_SECONDS=3600s",
                name="megatron-x-abcd1234-MBS2-GBS16-rep1-failed-attempt2_20261014-093907.log",
                banner_extra="# Attempt          : 2 / 6 (incl. failed)\n",
            )
        )
    )
    assert result["status"] == "failed"
    assert result["failure_reason"] == "timeout"
    assert result["failed_attempt"] == 2
    assert result["attempt"] == 2


# --------------------------------------------------------------------------
# Fields added in parser version 2
# --------------------------------------------------------------------------


def test_banner_fields_reach_the_result(tmp_path):
    result = extract.parse_log_file(
        str(
            write_log(
                tmp_path,
                MEGATRON_AVG,
                config_name="examples/megatron/configs/MI355X/llama3.1_8B-BF16-pretrain",
                banner_extra=(
                    "# Batch Stamp      : 20261014-093907\n"
                    "# GPU check        : idle (8 GPUs, at most 283 MiB in use; threshold 2048 MiB)\n"
                ),
            )
        )
    )
    assert result["config_path"] == "examples/megatron/configs/MI355X/llama3.1_8B-BF16-pretrain"
    assert result["batch_stamp"] == "20261014-093907"
    assert result["num_reps"] == 3
    assert result["gpu_check"].startswith("idle (8 GPUs")
    assert result["parser_version"] == extract.PARSER_VERSION


def test_image_and_host_provenance_reach_the_result(tmp_path):
    result = extract.parse_log_file(
        str(
            write_log(
                tmp_path,
                MEGATRON_AVG,
                banner_extra=(
                    "# Image ID         : sha256:fe8e1551d200\n"
                    "# Container ROCm   : 10.1.0\n"
                    "# Container HIP    : 7.16.26385-0000000\n"
                    "# CPU governor     : schedutil x256\n"
                    "# NUMA balancing   : 0\n"
                    "# Transparent HP   : always\n"
                    "# IOMMU            : amd_iommu=on iommu=pt\n"
                    "# amdgpu driver    : 6.14.14\n"
                ),
            )
        )
    )
    assert result["image_id"] == "sha256:fe8e1551d200"
    assert result["container_rocm_version"] == "10.1.0"
    assert result["container_hip_version"] == "7.16.26385-0000000"
    assert result["cpu_governor"] == "schedutil x256"
    assert result["numa_balancing"] == "0"
    assert result["transparent_hugepages"] == "always"
    assert result["iommu"] == "amd_iommu=on iommu=pt"
    assert result["amdgpu_version"] == "6.14.14"


def test_platform_reaches_the_result(tmp_path):
    system = (
        "Dual AMD EPYC 9575F 64-Core Processor-based Supermicro AS -4126GS-NMR-LCC server with "
        "8x AMD Instinct MI355X (288GB HBM3E 1400W) GPUs, 1 NUMA node per socket, System BIOS 1.7, "
        "Ubuntu 22.04.5 LTS, Host GPU driver amdgpu 30.10.1 (ROCm 7.0.1)"
    )
    result = extract.parse_log_file(
        str(
            write_log(
                tmp_path,
                MEGATRON_AVG,
                banner_extra=(
                    f"# System           : {system}\n"
                    "# Host check       : same on all 2 nodes\n"
                    "# Server           : Supermicro AS -4126GS-NMR-LCC\n"
                    "# BIOS             : 1.7 (09/12/2025, American Megatrends International, LLC.)\n"
                    "# Kernel cmdline   : ro amd_iommu=on iommu=pt amdgpu.noretry=1\n"
                    "# CPU topology     : 2 sockets x 64 cores x 2 threads (256 CPUs), 2 NUMA nodes\n"
                    "# CPU idle states  : acpi_idle (menu): POLL C1 C2\n"
                    "# GPU power cap    : 1400 W x7, 1000 W x1\n"
                    "# GPU partition    : SPX/NPS1\n"
                    "# Host GPU driver  : amdgpu 30.10.1 (ROCm 7.0.1)\n"
                    "# RDMA NICs        : none\n"
                ),
            )
        )
    )
    assert result["system_description"] == system
    assert result["host_check"] == "same on all 2 nodes"
    assert result["server"] == "Supermicro AS -4126GS-NMR-LCC"
    assert result["bios"].startswith("1.7 (09/12/2025")
    # Only the first colon separates the key from the value.
    assert result["kernel_cmdline"] == "ro amd_iommu=on iommu=pt amdgpu.noretry=1"
    assert result["cpu_topology"].startswith("2 sockets x 64 cores")
    assert result["cpu_idle"] == "acpi_idle (menu): POLL C1 C2"
    assert result["gpu_power_cap"] == "1400 W x7, 1000 W x1"
    assert result["gpu_partition"] == "SPX/NPS1"
    assert result["gpu_driver"] == "amdgpu 30.10.1 (ROCm 7.0.1)"
    # "none" and "unknown" are left empty, like every other provenance field.
    assert result["rdma_nics"] is None
    assert result["gpu_vram"] is None

    csv_path, _ = extract.write_outputs([result], str(tmp_path))
    header = Path(csv_path).read_text().splitlines()[0].split(",")
    # Appended after the parser version 4 columns.
    v5 = header.index("step_log") + 1
    assert header[v5 : v5 + len(extract.PLATFORM_BANNER_KEYS)] == list(extract.PLATFORM_BANNER_KEYS)


def test_host_rocm_is_read_under_its_new_and_old_names(tmp_path):
    old = extract.parse_log_file(str(write_log(tmp_path, MEGATRON_AVG, name="old.log")))
    assert old["rocm_version"] == "7.2.1"
    new_log = tmp_path / "new.log"
    new_log.write_text((tmp_path / "old.log").read_text().replace("# ROCm version ", "# ROCm (host)  "))
    assert "# ROCm (host)" in new_log.read_text()
    assert extract.parse_log_file(str(new_log))["rocm_version"] == "7.2.1"


def test_older_banner_leaves_new_fields_empty(tmp_path):
    result = extract.parse_log_file(str(write_log(tmp_path, MEGATRON_AVG)))
    for field in (
        "image_id",
        "container_rocm_version",
        "cpu_governor",
        "gpu_check",
        "batch_stamp",
        "system_description",
        "gpu_driver",
    ):
        assert result[field] is None, field


def test_fractional_batch_size_is_kept(tmp_path):
    log = tmp_path / "frac.log"
    log.write_text(
        write_log(tmp_path, "", framework="maxtext", name="tmp.log")
        .read_text()
        .replace("# Micro Batch Size : 2", "# Micro Batch Size : 0.5")
    )
    assert extract.parse_log_file(str(log))["micro_batch_size"] == 0.5


def test_per_step_spread(tmp_path):
    result = extract.parse_log_file(str(write_log(tmp_path, MEGATRON_LEGACY)))
    # Post-warmup steps are 400 and 400.
    assert result["tflops_min"] == pytest.approx(400.0)
    assert result["tflops_max"] == pytest.approx(400.0)
    assert result["tflops_cv"] == pytest.approx(0.0)
    assert result["tps_per_gpu_stdev"] == pytest.approx(0.0)


def test_version_mismatch_warnings_are_collected(tmp_path):
    warning = (
        "[20261014 14:44:00][rank-0/8][WARNING] aiter version mismatch: "
        "installed=0.1.22.post1, expected=0.1.14.post1\n"
    )
    result = extract.parse_log_file(str(write_log(tmp_path, warning * 8 + MEGATRON_AVG)))
    assert result["version_warnings"] == [
        "aiter version mismatch: installed=0.1.22.post1, expected=0.1.14.post1"
    ]


def test_outputs_have_fixed_names_and_keep_old_column_order(tmp_path):
    import csv
    import json

    results = [extract.parse_log_file(str(write_log(tmp_path, MEGATRON_AVG)))]
    csv_path, json_path = extract.write_outputs(results, str(tmp_path))
    assert Path(csv_path).name == "results.csv"
    assert Path(json_path).name == "results.json"

    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    # The columns of parser version 1 keep their order; new ones come after.
    v1_columns = [
        "filename",
        "backend",
        "model_name",
        "precision",
        "seq_len",
        "micro_batch_size",
        "global_batch_size",
        "nnodes",
        "gpus_per_node",
        "world_size",
        "warmup_steps",
        "repeat",
        "total_steps",
        "num_iterations",
        "num_post_warmup",
        "hmean_tps_per_gpu",
        "hmean_tflops",
        "hmean_samples_per_gpu",
        "hmean_frames_per_gpu",
        "amean_memory_usage",
        "amean_memory_usage_pct",
        "exit_code",
        "elapsed_sec",
        "docker_image",
        "image_digest",
        "primus_commit",
        "submodule_pins",
        "gpu_model",
        "rocm_version",
        "torch_version",
        "jax_version",
        "config_hash",
        "host",
        "slurm_job_id",
        "nodelist",
        "timestamp",
    ]
    assert header[: len(v1_columns)] == v1_columns

    data = json.loads(Path(json_path).read_text())
    assert data["parser_version"] == extract.PARSER_VERSION
    assert data["runs"][0]["status"] == "ok"
    assert isinstance(data["runs"][0]["version_warnings"], list)
    # Written via a temporary file and renamed into place.
    assert not list(tmp_path.glob("*.tmp"))


def test_fractional_batch_size_is_written_plainly(tmp_path):
    import csv

    log = tmp_path / "frac.log"
    log.write_text(
        write_log(tmp_path, "", framework="maxtext", name="tmp.log")
        .read_text()
        .replace("# Micro Batch Size : 2", "# Micro Batch Size : 0.5")
    )
    (tmp_path / "tmp.log").unlink()
    csv_path, _ = extract.write_outputs([extract.parse_log_file(str(log))], str(tmp_path))
    with open(csv_path, newline="") as f:
        row = next(csv.DictReader(f))
    assert row["micro_batch_size"] == "0.5"


# --------------------------------------------------------------------------
# diffusion_dlrm runs (parser version 4)
# --------------------------------------------------------------------------

# Trimmed to the columns diffusion_bench/parse_results.py and the extractor read.
RUNS_SUMMARY = (
    "num_processes,num_iterations,avg_step_time_gpu,avg_fps_gpu,avg_tflops,status\n"
    "8,100,1.8466,8.6847,567.29,{status}\n"
)
DLRM_RESULTS = (
    "Time Stamp,Num. Nodes,Batch Size,Recommendations/s (mean),Recommendations/s (std/mean)\n"
    "2026:10:14_23:26:30,1,32768,552606.45,0.01\n"
)
STEP_EXIT = {"success": 0, "oom": 2, "failed": 3, "timeout": 124}
DD_RUN = "mi355_node_20261014_183231"


def make_diffusion_bench_run(root: Path, diffusion=(), dlrm=(), step_logs=None) -> Path:
    """A diffusion_bench results folder under ``root/diffusion_dlrm``.

    diffusion: ``(model, batch_size, step_status, runs_summary_status)``;
    dlrm: ``(precision, step_status, with_results_csv)``. A step_status of
    None leaves the step out of status.tsv, as for one still in flight.
    """
    run_dir = root / "diffusion_dlrm" / DD_RUN
    (run_dir / "logs").mkdir(parents=True)
    status = ["step\tstatus\texit_code\tduration_s\tlog", "setup\tsuccess\t0\t2\tlogs/setup.log"]
    for model, bs, step_status, summary_status in diffusion:
        out = run_dir / "diffusion" / f"{model}_bs{bs}"
        out.mkdir(parents=True)
        if summary_status:
            (out / "runs_summary.csv").write_text(RUNS_SUMMARY.format(status=summary_status))
        if step_status:
            step = f"diffusion_{model}_bs{bs}"
            status.append(f"{step}\t{step_status}\t{STEP_EXIT[step_status]}\t479\tlogs/{step}.log")
    for precision, step_status, with_results in dlrm:
        out = run_dir / "dlrm" / precision
        out.mkdir(parents=True)
        (out / "training_config.sh").write_text("export BATCH_SIZE=32768\n")
        if with_results:
            (out / "results.csv").write_text(DLRM_RESULTS)
        if step_status:
            step = f"dlrm_{precision}"
            status.append(f"{step}\t{step_status}\t{STEP_EXIT[step_status]}\t135\tlogs/{step}.log")
    (run_dir / "status.tsv").write_text("\n".join(status) + "\n")
    # Every recorded step has a log, as in a real run.
    for line in status[1:]:
        (run_dir / "logs" / f"{line.split(chr(9))[0]}.log").write_text("")
    for name, text in (step_logs or {}).items():
        (run_dir / "logs" / f"{name}.log").write_text(text)
    return run_dir


def write_dd_log(tmp_path: Path, results_line: str | None, plan=(), **kwargs) -> Path:
    """A diffusion_dlrm run log; ``plan`` holds diffusion_bench's announced
    steps, e.g. "Diffusion: flux-dev  batch size 16" or "DLRM     : TF32"."""
    body = "[diffusion_bench 2026-10-14 18:32:31] Config   : /repo/tools/perf/diffusion_bench/config.yaml\n"
    if results_line is not None:
        body += f"[diffusion_bench 2026-10-14 18:32:31] Results  : {results_line}\n"
    for step in plan:
        body += f"[diffusion_bench 2026-10-14 18:32:31] {step}\n"
    body += "[diffusion_bench 2026-10-14 18:41:30] All steps finished.\n"
    return write_log(
        tmp_path,
        body,
        framework="diffusion_dlrm",
        config_name="tools/perf/diffusion_bench/config",
        config_body="gpu: auto\n",
        name="diffusion_dlrm-config-222c8d07-rep1_20261014-214751.log",
        **kwargs,
    )


def test_diffusion_dlrm_log_becomes_one_row_per_model(tmp_path):
    run_dir = make_diffusion_bench_run(
        tmp_path, diffusion=[("flux-dev", 16, "success", "success")], dlrm=[("FP32", "success", True)]
    )
    log = write_dd_log(tmp_path, str(run_dir))
    flux, dlrm = extract.parse_log(str(log))

    assert flux["backend"] == "diffusion"
    assert flux["model_name"] == "flux-dev"
    assert flux["precision"] == "BF16"
    assert (flux["micro_batch_size"], flux["global_batch_size"], flux["world_size"]) == (16, 128, 8)
    assert flux["hmean_tflops"] == pytest.approx(567.29)
    assert flux["hmean_samples_per_gpu"] == pytest.approx(8.6847)
    assert (flux["status"], flux["failure_reason"]) == ("ok", None)
    assert (flux["exit_code"], flux["elapsed_sec"]) == (0, 479)
    assert flux["step_log"] == f"diffusion_dlrm/{DD_RUN}/logs/diffusion_flux-dev_bs16.log"
    # The iterations AMDiffusionBenchmark actually ran.
    assert flux["total_steps"] == 100

    assert dlrm["backend"] == "dlrm"
    assert (dlrm["model_name"], dlrm["precision"]) == ("dlrm-v2", "FP32")
    assert (dlrm["micro_batch_size"], dlrm["global_batch_size"]) == (32768, 262144)
    assert dlrm["rec_per_s"] == pytest.approx(552606.45)
    assert dlrm["rec_per_s_cv"] == pytest.approx(0.01)
    assert dlrm["total_steps"] is None
    assert dlrm["status"] == "ok"

    for row in (flux, dlrm):
        assert row["filename"] == log.name
        assert row["docker_image"] == "registry.example/ci:some-tag"
        assert row["config_hash"] == "abcd1234"


def test_diffusion_dlrm_step_failures_get_their_own_reason(tmp_path):
    run_dir = make_diffusion_bench_run(
        tmp_path,
        diffusion=[
            ("flux-dev", 16, "oom", None),
            ("stable-diffusion-xl", 12, "timeout", None),
            ("mochi-1", 4, "failed", None),
        ],
        dlrm=[("TF32", "success", False)],
        step_logs={
            "diffusion_flux-dev_bs16": "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB\n",
            "diffusion_mochi-1_bs4": (
                "Traceback (most recent call last):\n"
                '  File "launcher.py", line 1, in <module>\n'
                "KeyError: 'mochi'\n"
            ),
        },
    )
    log = write_dd_log(
        tmp_path,
        str(run_dir),
        exit_code=1,
        failure_reason="steps: 4 of 5 diffusion_bench steps not successful: ...",
    )
    rows = {row["model_name"]: row for row in extract.parse_log(str(log))}

    assert rows["flux-dev"]["status"] == "failed"
    assert rows["flux-dev"]["failure_reason"] == "oom"
    assert rows["flux-dev"]["exit_code"] == 2
    assert rows["stable-diffusion-xl"]["failure_reason"] == "timeout"
    assert rows["mochi-1"]["failure_reason"] == "python_error"
    assert rows["mochi-1"]["failure_detail"].startswith("KeyError")
    # Success without a throughput number is not a pass.
    assert (rows["dlrm-v2"]["status"], rows["dlrm-v2"]["failure_reason"]) == ("failed", "no_metrics")


def test_diffusion_dlrm_step_in_flight_takes_the_runs_outcome(tmp_path):
    # AMDiffusionBenchmark writes runs_summary.csv with not_run when it starts.
    run_dir = make_diffusion_bench_run(
        tmp_path, diffusion=[("flux-dev", 16, "success", "success"), ("hunyuan-video", 3, None, "not_run")]
    )
    log = write_dd_log(
        tmp_path, str(run_dir), exit_code=130, failure_reason="interrupted: the batch was stopped by a signal"
    )
    rows = {row["model_name"]: row for row in extract.parse_log(str(log))}
    assert rows["flux-dev"]["status"] == "ok"
    assert (rows["hunyuan-video"]["status"], rows["hunyuan-video"]["failure_reason"]) == (
        "interrupted",
        "interrupted",
    )


def test_diffusion_dlrm_results_are_found_after_the_folder_moved(tmp_path):
    make_diffusion_bench_run(tmp_path, diffusion=[("flux-dev", 16, "success", "success")])
    log = write_dd_log(tmp_path, f"/old/place/diffusion_dlrm/{DD_RUN}")
    (row,) = extract.parse_log(str(log))
    assert row["model_name"] == "flux-dev"
    assert row["hmean_tflops"] == pytest.approx(567.29)


def test_diffusion_dlrm_copy_reads_its_own_folder(tmp_path):
    original = tmp_path / "original"
    copy = tmp_path / "copy"
    original_run = make_diffusion_bench_run(original, diffusion=[("flux-dev", 16, "success", "success")])
    copy_run = make_diffusion_bench_run(copy, diffusion=[("flux-dev", 16, "oom", None)])
    # The copy's log still names the original's folder.
    log = write_dd_log(copy, str(original_run))
    (row,) = extract.parse_log(str(log))
    assert row["failure_reason"] == "oom"
    assert row["step_log"] == f"diffusion_dlrm/{copy_run.name}/logs/diffusion_flux-dev_bs16.log"


def test_diffusion_dlrm_steps_without_a_folder_still_get_a_row(tmp_path):
    # DLRM writes its folder only after training returns, so a DLRM step
    # stopped by step_timeout has a status.tsv entry and nothing else; SDXL
    # never started because the batch was stopped by a signal.
    run_dir = make_diffusion_bench_run(tmp_path, diffusion=[("flux-dev", 16, "success", "success")])
    with open(run_dir / "status.tsv", "a") as f:
        f.write("dlrm_TF32\ttimeout\t124\t14400\tlogs/dlrm_TF32.log\n")
    plan = (
        "Diffusion: flux-dev  batch size 16",
        "Diffusion: stable-diffusion-xl  batch size 12",
        "DLRM     : TF32",
    )
    log = write_dd_log(tmp_path, str(run_dir), plan=plan, exit_code=1, failure_reason="steps: 1 of 4 ...")
    rows = extract.parse_log(str(log))
    assert [(r["backend"], r["model_name"], r["precision"]) for r in rows] == [
        ("diffusion", "flux-dev", "BF16"),
        ("diffusion", "stable-diffusion-xl", "BF16"),
        ("dlrm", "dlrm-v2", "TF32"),
    ]
    flux, sdxl, dlrm = rows
    assert flux["status"] == "ok"
    assert (sdxl["status"], sdxl["failure_reason"]) == ("failed", "not_run")
    assert (sdxl["micro_batch_size"], sdxl["global_batch_size"]) == (12, 96)
    assert (dlrm["status"], dlrm["failure_reason"], dlrm["exit_code"]) == ("failed", "timeout", 124)

    # The same steps, had the batch been interrupted instead.
    log = write_dd_log(
        tmp_path,
        str(run_dir),
        plan=plan,
        exit_code=130,
        failure_reason="interrupted: the batch was stopped by a signal",
    )
    sdxl = extract.parse_log(str(log))[1]
    assert (sdxl["status"], sdxl["failure_reason"]) == ("interrupted", "interrupted")


def test_diffusion_dlrm_crashed_launcher_is_a_failure(tmp_path):
    # in_container.sh records not_run when runs_summary.csv still says so,
    # which outside a dry run means launcher.py died before training.
    run_dir = make_diffusion_bench_run(
        tmp_path,
        diffusion=[("wan2_1-i2v", 1, None, "not_run")],
        step_logs={
            "diffusion_wan2_1-i2v_bs1": (
                "Traceback (most recent call last):\n"
                '  File "launcher.py", line 1, in <module>\n'
                "RuntimeError: HIP error: invalid device function\n"
            )
        },
    )
    with open(run_dir / "status.tsv", "a") as f:
        f.write("diffusion_wan2_1-i2v_bs1\tnot_run\t4\t30\tlogs/diffusion_wan2_1-i2v_bs1.log\n")
    log = write_dd_log(tmp_path, str(run_dir), exit_code=1, failure_reason="steps: ...")
    (row,) = extract.parse_log(str(log))
    assert (row["status"], row["failure_reason"]) == ("failed", "python_error")
    assert row["failure_detail"].startswith("RuntimeError")


def test_diffusion_dlrm_without_steps_keeps_the_runs_row(tmp_path):
    # diffusion_bench stopped before creating its results folder.
    root = tmp_path / "a"
    root.mkdir()
    log = write_dd_log(root, None, exit_code=1, failure_reason="exit: launcher exited with code 1")
    (row,) = extract.parse_log(str(log))
    assert (row["backend"], row["status"]) == ("diffusion_dlrm", "failed")

    # It created the folder, but its setup step failed before any model ran.
    root = tmp_path / "b"
    root.mkdir()
    run_dir = make_diffusion_bench_run(
        root, step_logs={"setup": "sed: can't read Makefile: No such file or directory\n"}
    )
    status = run_dir / "status.tsv"
    status.write_text(status.read_text().replace("setup\tsuccess\t0", "setup\terror\t1"))
    log = write_dd_log(
        root,
        str(run_dir),
        plan=("Diffusion: flux-dev  batch size 16", "DLRM     : TF32"),
        exit_code=1,
        failure_reason="steps: 1 of 1 diffusion_bench steps not successful: setup error",
    )
    setup, flux, dlrm = extract.parse_log(str(log))
    assert (setup["backend"], setup["model_name"], setup["status"]) == ("diffusion_dlrm", "setup", "failed")
    assert setup["failure_detail"].startswith("sed: can't read Makefile")
    for row in (flux, dlrm):
        assert (row["status"], row["failure_reason"]) == ("failed", "not_run")
        assert row["failure_detail"] == "not run: diffusion_bench's setup step failed"
        # Nothing ran, so there is no log to point at.
        assert row["step_log"] is None


def test_diffusion_dlrm_failed_download_gets_a_row(tmp_path):
    # diffusion_bench carries on with the cached assets, and the run fails.
    run_dir = make_diffusion_bench_run(
        tmp_path,
        diffusion=[("flux-dev", 16, "success", "success")],
        step_logs={"download": "huggingface_hub.errors.HfHubHTTPError: 503 Server Error\n"},
    )
    with open(run_dir / "status.tsv", "a") as f:
        f.write("download\terror\t1\t60\tlogs/download.log\n")
    log = write_dd_log(tmp_path, str(run_dir), exit_code=1, failure_reason="steps: 1 of 3 ...")
    download, flux = extract.parse_log(str(log))
    assert (download["backend"], download["model_name"], download["status"]) == (
        "diffusion_dlrm",
        "download",
        "failed",
    )
    assert download["failure_reason"] == "python_error"
    assert flux["status"] == "ok"


def test_diffusion_dlrm_incomplete_run_blames_each_step_for_its_own_log_only(tmp_path):
    # The batch was killed (no footer) while SDXL ran, after flux hit an OOM.
    oom = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB\n"
    run_dir = make_diffusion_bench_run(
        tmp_path,
        diffusion=[("flux-dev", 16, "oom", None), ("stable-diffusion-xl", 12, None, "not_run")],
        step_logs={"diffusion_flux-dev_bs16": oom, "diffusion_stable-diffusion-xl_bs12": "Step 3 ...\n"},
    )
    plan = (
        "Diffusion: flux-dev  batch size 16",
        "Diffusion: stable-diffusion-xl  batch size 12",
        "DLRM     : TF32",
    )
    log = write_dd_log(tmp_path, str(run_dir), plan=plan, with_footer=False)
    log.write_text(log.read_text() + oom)
    flux, sdxl, dlrm = extract.parse_log(str(log))
    assert (flux["status"], flux["failure_reason"]) == ("failed", "oom")
    assert (sdxl["status"], sdxl["failure_reason"]) == ("incomplete", None)
    assert (dlrm["status"], dlrm["failure_reason"]) == ("incomplete", None)


@pytest.mark.parametrize(
    "status,text,reason,detail",
    [
        ("killed", "Step 1 ...\n", "killed", "SIGKILL before step_timeout"),
        (
            "failed",
            "[in_container 2026-10-15 03:15:46] Running: python launcher.py train_args=flux-dev\n"
            "Could not override 'train_args.no_such_option'.\n"
            "To append to your config use +train_args.no_such_option=1\n"
            "Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.\n"
            "[in_container 2026-10-15 03:15:47] launcher exit code=1, run status=missing\n",
            "config_error",
            "Could not override 'train_args.no_such_option'.",
        ),
        (
            "failed",
            "something went wrong\n[in_container 2026-10-15 03:15:47] launcher exit code=1, run status=failed\n",
            "exit",
            "something went wrong",
        ),
    ],
)
def test_diffusion_dlrm_failure_details(tmp_path, status, text, reason, detail):
    run_dir = make_diffusion_bench_run(tmp_path, diffusion=[("flux-dev", 16, None, None)])
    with open(run_dir / "status.tsv", "a") as f:
        f.write(f"diffusion_flux-dev_bs16\t{status}\t1\t30\tlogs/diffusion_flux-dev_bs16.log\n")
    (run_dir / "logs" / "diffusion_flux-dev_bs16.log").write_text(text)
    log = write_dd_log(tmp_path, str(run_dir), exit_code=1, failure_reason="steps: ...")
    (row,) = extract.parse_log(str(log))
    assert (row["status"], row["failure_reason"]) == ("failed", reason)
    assert row["failure_detail"].startswith(detail)


def test_diffusion_dlrm_separate_dlrm_image_is_reported(tmp_path):
    run_dir = make_diffusion_bench_run(
        tmp_path, diffusion=[("flux-dev", 16, "success", "success")], dlrm=[("TF32", "success", True)]
    )
    (run_dir / "run_info.txt").write_text("image: registry.example/ci:some-tag\ndlrm_image: rocm/dlrm:1.0\n")
    (run_dir / "env").mkdir()
    (run_dir / "env" / "image_id_dlrm.txt").write_text("sha256:0123abcd\n")
    log = write_dd_log(tmp_path, str(run_dir), banner_extra="# Image ID         : sha256:fe8e1551d200\n")
    flux, dlrm = extract.parse_log(str(log))
    assert (flux["docker_image"], flux["image_id"]) == ("registry.example/ci:some-tag", "sha256:fe8e1551d200")
    assert (dlrm["docker_image"], dlrm["image_id"]) == ("rocm/dlrm:1.0", "sha256:0123abcd")
    # The banner's container versions describe the main image only.
    assert dlrm["torch_version"] is None and dlrm["image_digest"] is None


def test_diffusion_dlrm_rows_reach_the_outputs(tmp_path, monkeypatch):
    import csv

    run_dir = make_diffusion_bench_run(
        tmp_path, diffusion=[("flux-dev", 16, "success", "success")], dlrm=[("TF32", "success", True)]
    )
    write_dd_log(tmp_path, str(run_dir))
    write_log(tmp_path, MEGATRON_AVG, name="megatron.log")
    monkeypatch.setattr("sys.argv", ["extract_results.py", "--quiet", str(tmp_path)])
    assert extract.main() == 0

    with open(tmp_path / "results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert sorted(r["backend"] for r in rows) == ["diffusion", "dlrm", "megatron"]
    with open(tmp_path / "results.csv", newline="") as f:
        header = next(csv.reader(f))
    # The parser version 4 columns follow version 3's last one.
    v4 = header.index("amdgpu_version") + 1
    assert header[v4 : v4 + 3] == ["rec_per_s", "rec_per_s_cv", "step_log"]
    dlrm = next(r for r in rows if r["backend"] == "dlrm")
    assert dlrm["rec_per_s"] == "552606.4500"
    assert dlrm["rec_per_s_cv"] == "0.01"
