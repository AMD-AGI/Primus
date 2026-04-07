###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MoE MLP profiler: simulation emits grouped-GEMM / Origami diagnostics; GPU benchmark does not."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from primus.core.projection.module_profilers.moe_mlp import MoEMLPProfiler
from primus.core.projection.training_config import (
    ModelConfig,
    ModelParallelConfig,
    RuntimeConfig,
    TrainingConfig,
)


def _moe_training_config() -> TrainingConfig:
    mc = ModelConfig(
        num_layers=1,
        hidden_size=4096,
        padded_vocab_size=32000,
        ffn_hidden_size=14336,
        num_attention_heads=32,
        kv_channels=128,
        swiglu=True,
        num_experts=8,
        moe_ffn_hidden_size=None,
        moe_router_topk=2,
        moe_pattern=[1],
        moe_grouped_gemm=True,
        moe_use_legacy_grouped_gemm=False,
        enable_primus_turbo=True,
        use_turbo_grouped_mlp=False,
        moe_shared_expert_intermediate_size=None,
    )
    rc = RuntimeConfig(
        global_batch_size=256,
        micro_batch_size=4,
        sequence_length=4096,
        data_parallel_size=8,
    )
    mp = ModelParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=8,
        context_model_parallel_size=1,
    )
    return TrainingConfig(model_config=mc, runtime_config=rc, model_parallel_config=mp)


class _MockGemmBackend:
    """Minimal backend for :meth:`MoEMLPProfiler._get_simulated_results`."""

    hbm_bandwidth_gbps = 5300.0

    def simulate_gemm(self, m, n, k, dtype="bf16", batch=1, **kwargs):
        r = MagicMock()
        r.forward_time_ms = 0.01
        return r

    def simulate_mlp_gemms(self, **kwargs):
        r = MagicMock()
        r.forward_time_ms = 0.0
        r.backward_time_ms = 0.0
        return r


def test_gpu_benchmark_does_not_invoke_moe_grouped_gemm_logs():
    profiler = MoEMLPProfiler(_moe_training_config())
    profiler._gemm_backend = None
    profiler.module = MagicMock()

    with patch(
        "primus.core.projection.module_profilers.moe_mlp.benchmark_moe_layer_decomposed",
        return_value=(1.0, 2.0, 100, 0.0, 0.0),
    ) as bench, patch.object(
        profiler, "_log_grouped_gemm_context"
    ) as log_ctx, patch.object(
        profiler, "_print_origami_expert_grouped_gemm_simulation"
    ) as print_sim:
        profiler._get_benchmark_results(4, 4096)
        bench.assert_called_once()
        log_ctx.assert_not_called()
        print_sim.assert_not_called()


def test_simulation_invokes_moe_grouped_gemm_logs():
    profiler = MoEMLPProfiler(_moe_training_config())
    profiler._gemm_backend = _MockGemmBackend()

    with patch.object(profiler, "_log_grouped_gemm_context") as log_ctx, patch.object(
        profiler, "_print_origami_expert_grouped_gemm_simulation"
    ) as print_sim:
        profiler._get_simulated_results(4, 4096)
        log_ctx.assert_called_once()
        assert log_ctx.call_args[0][2] == "simulation"
        print_sim.assert_called_once()


def test_simulate_cli_stdout_contains_moe_origami_markers():
    """End-to-end: ``projection performance --profiling-mode simulate`` prints MoE Origami blocks."""
    repo = Path(__file__).resolve().parents[3]
    cfg = repo / "examples/megatron/configs/MI355X/mixtral_8x7B_v0.1-BF16-pretrain.yaml"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "primus.cli.main",
            "projection",
            "performance",
            "--config",
            str(cfg),
            "--profiling-mode",
            "simulate",
            "--target-nodes",
            "1",
            "--gpu-arch",
            "mi355x",
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 0, r.stderr[:2000] if r.stderr else r.stdout[-2000:]
    out = r.stdout + (r.stderr or "")
    assert "========== [MoE MLP] Origami expert grouped-GEMM FORWARD ==========" in out
    assert "========== [MoE MLP] Origami expert grouped-GEMM BACKWARD ==========" in out
    assert "[MoE MLP] Grouped GEMM (simulation):" in out


def _subprocess_output_suggests_gpu_oom(combined: str) -> bool:
    """True if Megatron/PyTorch reported GPU OOM (shared or fragmented GPU memory)."""
    s = combined.lower()
    return (
        "out of memory" in s
        or "hip out of memory" in s
        or "cuda out of memory" in s
        or "torch.outofmemoryerror" in s
    )


def test_benchmark_cli_has_no_moe_origami_diagnostic_banners():
    """On a CUDA machine, GPU benchmark must not print simulation-only MoE Origami blocks.

    Set ``PRIMUS_SKIP_GPU_BENCHMARK_CLI_TEST=1`` to skip (e.g. CI without GPU).

    If the subprocess exits non-zero due to **GPU OOM** (typical on a busy shared GPU),
    the test is **skipped** — the behaviour under test is already covered by
    ``test_gpu_benchmark_does_not_invoke_moe_grouped_gemm_logs``.  Use a quiet GPU or
    set ``PRIMUS_FORCE_GPU_BENCHMARK_CLI_TEST=1`` to fail instead of skip on OOM.
    """
    if os.environ.get("PRIMUS_SKIP_GPU_BENCHMARK_CLI_TEST", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        pytest.skip("PRIMUS_SKIP_GPU_BENCHMARK_CLI_TEST is set")

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for this smoke test")

    force = os.environ.get("PRIMUS_FORCE_GPU_BENCHMARK_CLI_TEST", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    repo = Path(__file__).resolve().parents[3]
    cfg = repo / "examples/megatron/configs/MI355X/mixtral_8x7B_v0.1-BF16-pretrain.yaml"
    # Align with ``_run_layer_benchmark`` / Megatron single-process expectations.
    env = {
        **os.environ,
        "NNODES": "1",
        "NODE_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "GPUS_PER_NODE": "1",
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    }
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "primus.cli.main",
            "projection",
            "performance",
            "--config",
            str(cfg),
            "--profiling-mode",
            "benchmark",
            "--benchmark-gpus",
            "1",
            "--target-nodes",
            "1",
            # Lower activation footprint when the GPU has little free memory.
            "--micro-batch-size",
            "1",
            "--global-batch-size",
            "8",
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    combined = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        if _subprocess_output_suggests_gpu_oom(combined) and not force:
            pytest.skip(
                "GPU benchmark subprocess OOM (insufficient free GPU memory for Mixtral "
                "8x7B on 1 GPU). MoE print behaviour is covered by "
                "test_gpu_benchmark_does_not_invoke_moe_grouped_gemm_logs."
            )
        assert r.returncode == 0, combined[:8000]
    out = combined
    assert "========== [MoE MLP] Origami expert grouped-GEMM FORWARD ==========" not in out
    assert "[MoE MLP] Grouped GEMM (simulation):" not in out
