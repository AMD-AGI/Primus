import sys
from pathlib import Path

# Import the standalone tool by its source path, not a third-party `tools` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/profile_trace"))
from analyze_flux_trace import render_markdown, summarize_steps, summarize_trace


def test_flux_trace_summary_uses_operator_context_and_overlap():
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::_scaled_mm",
                "ts": 0,
                "dur": 20,
                "args": {"External id": 1},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "hipblaslt_kernel",
                "ts": 0,
                "dur": 10,
                "args": {"External id": 1},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "aiter_flash_attn",
                "ts": 5,
                "dur": 10,
                "args": {"External id": 2},
            },
        ]
    }

    rows = {row["group"]: row for row in summarize_trace(trace)}

    assert rows["FP8 GEMM"]["work_ms"] == 0.01
    assert rows["FP8 GEMM"]["exposed_ms"] == 0.005
    assert rows["AITER attention"]["work_ms"] == 0.01
    assert rows["AITER attention"]["exposed_ms"] == 0.005
    assert "Kernel group" in render_markdown(list(rows.values()))


def test_flux_trace_summary_maps_rocm_correlation_and_clips_gpu_steps():
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "primus::flux_flydsl_natural_wgrad",
                "ts": 0,
                "dur": 10,
                "args": {"External id": 7},
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "hipModuleLaunchKernel",
                "ts": 1,
                "dur": 1,
                "args": {"External id": 7, "correlation": 99},
            },
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "ProfilerStep#1",
                "ts": 100,
                "dur": 100,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "kernel_dense_tn_wave4_0",
                "ts": 110,
                "dur": 20,
                "args": {"correlation": 99},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "carry_in_kernel",
                "ts": 50,
                "dur": 100,
                "args": {},
            },
        ]
    }

    rows = {row["group"]: row for row in summarize_trace(trace)}

    assert rows["FP8 GEMM"]["work_ms"] == 0.02
    assert "other GPU" not in rows


def test_flux_trace_step_summary_includes_memcpy_activity():
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "ProfilerStep#1",
                "ts": 100,
                "dur": 100,
                "args": {},
            },
            {"ph": "X", "cat": "gpu_memcpy", "name": "HtoD", "ts": 90, "dur": 20},
            {"ph": "X", "cat": "kernel", "name": "work", "ts": 120, "dur": 30},
        ]
    }

    steps = summarize_steps(trace)

    assert steps == [{"name": "ProfilerStep#1", "wall_ms": 0.1, "active_ms": 0.04, "idle_ms": 0.06}]
    assert "GPU active" in render_markdown([], steps)


def test_flux_trace_classification_requires_attention_or_fp8_evidence():
    from analyze_flux_trace import classify

    for operator in ("attn_fwd", "attn_bwd", "attn_fwd_split", "attn_bwd_split"):
        assert classify(f"opaque flux_flydsl::{operator}") == "FlyDSL attention"
    for name in ("flux_attn_fwd_bf16", "flux_attn_bwd_bf16"):
        assert classify(name) == "FlyDSL attention"
    for name in ("hipblaslt_kernel", "hipblas_gemm_bf16", "Cijk_bf16", "opaque aten::mm"):
        assert classify(name) in {"BF16/other GEMM", "other GPU"}
    assert classify("opaque flux_flydsl::unrelated") == "other GPU"
    for name in (
        "aten::_scaled_mm",
        "primus::flux_flydsl_scaled_mm",
        "primus::flux_flydsl_natural_wgrad",
        "fp8_gemm",
        "hipblaslt_gemm_fp8",
        "hipblas_gemm_f8",
        "Cijk_transA_f8",
    ):
        assert classify(name) == "FP8 GEMM"


def test_flux_trace_correlated_flydsl_attention_is_not_fp8_gemm():
    trace = {
        "traceEvents": [
            {"ph": "X", "cat": "cpu_op", "name": "flux_flydsl::attn_bwd_split", "args": {"External id": 8}},
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "hipLaunchKernel",
                "args": {"External id": 8, "correlation": 9},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "opaque_bf16_kernel",
                "ts": 0,
                "dur": 10,
                "args": {"correlation": 9},
            },
            {"ph": "X", "cat": "kernel", "name": "hipblaslt_kernel", "ts": 10, "dur": 5},
        ]
    }
    rows = {row["group"]: row for row in summarize_trace(trace)}
    assert rows["FlyDSL attention"]["work_ms"] == 0.01
    assert rows["BF16/other GEMM"]["work_ms"] == 0.005
    assert "FP8 GEMM" not in rows
