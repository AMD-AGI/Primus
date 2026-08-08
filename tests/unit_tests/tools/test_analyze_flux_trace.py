from tools.profile_trace.analyze_flux_trace import render_markdown, summarize_trace


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
