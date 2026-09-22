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
# Cluster          : NNODES={nnodes} GPUS_PER_NODE=8 NODE_RANK=0
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
{body}
############################### End Run Output #################################

################################################################################
# Run finished at : 2026-09-12 20:10:00 UTC
# Exit code       : {exit_code}
# Elapsed (sec)   : 600
################################################################################
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
) -> Path:
    path = tmp_path / name
    path.write_text(
        BANNER_TEMPLATE.format(
            config_name=config_name,
            framework=framework,
            nnodes=nnodes,
            world_size=world_size,
            exit_code=exit_code,
            config_body=config_body,
            body=body,
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
