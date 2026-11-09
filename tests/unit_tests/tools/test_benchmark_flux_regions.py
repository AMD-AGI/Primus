import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Import the standalone tool by source path, not a third-party `tools` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/profile_trace"))
import benchmark_flux_regions


@pytest.mark.parametrize(
    "options,fusion,max_size",
    [
        ([], False, None),
        (["--benchmark-fusion"], True, None),
        (["--max-fusion-size", "64"], False, 64),
        (["--benchmark-fusion", "--max-fusion-size", "64"], True, 64),
    ],
)
def test_main_defaults_and_inductor_options(options, fusion, max_size, monkeypatch, capsys):
    import torch._inductor.config as inductor_config

    benchmark = Mock(return_value={"region": "adaln", "latency_ms": 1.25})
    reset_memory = Mock()
    monkeypatch.setattr(sys, "argv", ["benchmark_flux_regions.py", "adaln", *options])
    monkeypatch.setattr(benchmark_flux_regions, "benchmark_adaln", benchmark)
    monkeypatch.setattr(benchmark_flux_regions.torch.cuda, "reset_peak_memory_stats", reset_memory)
    with inductor_config.patch(benchmark_fusion=False, max_fusion_size=32):
        benchmark_flux_regions.main()
        assert inductor_config.benchmark_fusion is fusion
        assert inductor_config.max_fusion_size == (max_size if max_size is not None else 32)
    reset_memory.assert_called_once_with()
    benchmark.assert_called_once()
    assert json.loads(capsys.readouterr().out) == {
        "region": "adaln",
        "latency_ms": 1.25,
        "benchmark_fusion": fusion,
        "max_fusion_size": max_size,
    }


@pytest.mark.parametrize("warmup", [0, 2])
@pytest.mark.parametrize("structured_pe", [False, True])
def test_timed_backward_clears_parameter_and_input_grads(warmup, structured_pe, monkeypatch):
    import torch

    from primus.backends.diffusion.models.flux.fused_norm_rope import RopeTables

    x = torch.ones(2, requires_grad=True)
    vec = torch.ones(2, requires_grad=True)
    pe = torch.zeros(1, 2, 1, 2, 2, 2)
    if structured_pe:
        pe = RopeTables(*(torch.zeros(2, 4) for _ in range(6)))
    module = torch.nn.Linear(2, 2)
    for parameter in module.parameters():
        parameter.grad = torch.ones_like(parameter)
    x.grad = torch.ones_like(x)
    vec.grad = torch.ones_like(vec)
    forwards = []

    def check_fresh_grads(_module, inputs):
        assert inputs == (x, vec, pe)
        assert all(parameter.grad is None for parameter in module.parameters())
        assert x.grad is None and vec.grad is None
        forwards.append(True)

    # Use a real module/backward while accepting the production structured PE.
    monkeypatch.setattr(
        module,
        "forward",
        lambda value, conditioning, embedding: torch.nn.functional.linear(
            value + conditioning, module.weight, module.bias
        ),
    )
    module.register_forward_pre_hook(check_fresh_grads)
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    assert benchmark_flux_regions.timed_backward(module, (x, vec, pe), warmup=warmup, iterations=3) >= 0
    assert len(forwards) == warmup + 3
    assert all(parameter.grad is None for parameter in module.parameters())
    assert x.grad is None and vec.grad is None


@pytest.mark.parametrize("fused", [False, True])
def test_single_stack_supplies_production_rope_type(fused, monkeypatch):
    from types import SimpleNamespace

    import torch

    from primus.backends.diffusion.models.flux import fused_norm_rope, model
    from primus.backends.diffusion.models.flux.fused_norm_rope import RopeTables

    # Simulate the imports performed when the fusion flag is on at startup.
    monkeypatch.setattr(model, "rope_tables", fused_norm_rope.rope_tables, raising=False)
    monkeypatch.setattr(model, "RopeTables", RopeTables, raising=False)
    args = SimpleNamespace(
        batch=1,
        sequence=4,
        hidden=8,
        heads=2,
        depth=1,
        fp8=False,
        strategy="eager",
        attention_backend="sdpa",
        warmup=0,
        iterations=1,
    )
    stack = Mock()
    stack.cuda.return_value.to.return_value = stack
    monkeypatch.setattr(benchmark_flux_regions, "SingleStack", Mock(return_value=stack))
    monkeypatch.setattr(benchmark_flux_regions.layers, "FUSED_NORM_ROPE", fused)
    monkeypatch.setattr(benchmark_flux_regions, "set_attention_backend", Mock())
    real_randn, real_arange = torch.randn, torch.arange
    monkeypatch.setattr(torch, "randn", lambda *a, **kw: real_randn(*a, **{**kw, "device": "cpu"}))
    monkeypatch.setattr(torch, "arange", lambda *a, **kw: real_arange(*a, **{**kw, "device": "cpu"}))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", Mock(return_value=0))

    def timed(fn, inputs, **kwargs):
        assert fn is stack
        x, vec, pe = inputs
        assert x.requires_grad and vec.requires_grad
        if fused:
            assert isinstance(pe, RopeTables)
            assert pe.all_cos.shape[0] == args.sequence
            assert pe.txt_cos.shape[0] == 0
        else:
            assert isinstance(pe, torch.Tensor)
        return 1.0

    monkeypatch.setattr(benchmark_flux_regions, "timed_backward", timed)
    assert benchmark_flux_regions.benchmark_single_stack(args)["latency_ms"] == 1.0
