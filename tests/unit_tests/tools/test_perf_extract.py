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
        tmp_path, ">>> [12:00:00] TIMEOUT — attempt exceeded MAX_RUN_SECONDS\n", exit_code=0,
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
        pytest.param("*** SIGSEGV (@0x0) received by PID 1234 (TID 0x7f) from PID 0\n", "segfault", id="glog-segv"),
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

    incomplete = extract.parse_log_file(str(write_log(tmp_path, MEGATRON_AVG, with_footer=False, name="b.log")))
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


def test_host_rocm_is_read_under_its_new_and_old_names(tmp_path):
    old = extract.parse_log_file(str(write_log(tmp_path, MEGATRON_AVG, name="old.log")))
    assert old["rocm_version"] == "7.2.1"
    new_log = tmp_path / "new.log"
    new_log.write_text((tmp_path / "old.log").read_text().replace("# ROCm version ", "# ROCm (host)  "))
    assert "# ROCm (host)" in new_log.read_text()
    assert extract.parse_log_file(str(new_log))["rocm_version"] == "7.2.1"


def test_older_banner_leaves_new_fields_empty(tmp_path):
    result = extract.parse_log_file(str(write_log(tmp_path, MEGATRON_AVG)))
    for field in ("image_id", "container_rocm_version", "cpu_governor", "gpu_check", "batch_stamp"):
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

    header = next(csv.reader(open(csv_path)))
    # The columns of parser version 1 keep their order; new ones come after.
    v1_columns = [
        "filename", "backend", "model_name", "precision", "seq_len", "micro_batch_size",
        "global_batch_size", "nnodes", "gpus_per_node", "world_size", "warmup_steps", "repeat",
        "total_steps", "num_iterations", "num_post_warmup", "hmean_tps_per_gpu", "hmean_tflops",
        "hmean_samples_per_gpu", "hmean_frames_per_gpu", "amean_memory_usage",
        "amean_memory_usage_pct", "exit_code", "elapsed_sec", "docker_image", "image_digest",
        "primus_commit", "submodule_pins", "gpu_model", "rocm_version", "torch_version",
        "jax_version", "config_hash", "host", "slurm_job_id", "nodelist", "timestamp",
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
    assert next(csv.DictReader(open(csv_path)))["micro_batch_size"] == "0.5"
