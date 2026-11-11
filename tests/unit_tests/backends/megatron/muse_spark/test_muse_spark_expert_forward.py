###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``_PostExpertNormMixin.forward``: the expert forward of every Muse Spark expert class.

The mixin is put on a stub parent (the role of TEGroupedMLP / SequentialMLP), so each branch
runs on CPU: unit probs to the parent, the fused / eager gate, HF's rounding order (the norm
output is rounded to the model dtype, then multiplied by the fp32 prob), no norm, and the
bias error.
"""

import re
from types import SimpleNamespace

import pytest
import torch

# the GPU-property read of fla -> mamba_ssm on a CPU-only node: torch's "no GPU" messages
# and Triton's "0 active drivers" (no GPU backend). A narrow match, so an unrelated
# RuntimeError ("... on the same device", "device-side assert") fails, not skips
_GPU_PROBE_ERROR = re.compile(
    r"No (CUDA|HIP) GPUs are available|Found no NVIDIA driver|no (CUDA|HIP|ROCm)-capable device"
    r"|0 active drivers"
)

# skip only when the optional dependency (Megatron) is missing; a broken import inside
# moe.py itself (a renamed norms / post_expert_norm / moe_utils symbol) must fail, not skip
pytest.importorskip("megatron.core")
if not torch.cuda.is_available():
    # On a CPU-only node an ``import megatron.training`` can raise (fla -> mamba_ssm reads the
    # GPU properties at import time); a repeated import usually works. Skip if it never does.
    for _ in range(3):
        try:
            import megatron.training  # noqa: F401

            break
        except RuntimeError as e:
            # only the GPU-property read of that import chain is retried / skipped; any
            # other RuntimeError is a broken import and fails the module
            if not _GPU_PROBE_ERROR.search(str(e)):
                raise
            continue
    else:
        pytest.skip("megatron.training does not import on this CPU-only node", allow_module_level=True)
from primus.backends.megatron.core.transformer.muse_spark import moe  # noqa: E402
from primus.backends.megatron.core.transformer.muse_spark.norms import (  # noqa: E402
    MuseSparkInputScaledRMSNorm,
)

H, ROWS = 16, 6


class _Parent:
    """Records the probs it gets and returns a fixed expert output (and bias)."""

    def __init__(self, out, bias=None):
        self.out, self.bias, self.seen = out, bias, []

    def forward(self, hidden, tokens_per_expert, probs):
        self.seen.append(probs)
        return self.out.clone(), self.bias


class _Experts(moe._PostExpertNormMixin, _Parent):
    def __init__(self, out, fused, norm=True, bias=None):
        _Parent.__init__(self, out, bias)
        self._init_post_expert_norm()
        self.config = SimpleNamespace(muse_spark_fused_post_expert_norm=fused)
        if norm:
            n = MuseSparkInputScaledRMSNorm(SimpleNamespace(hidden_size=H), H, eps=1e-6)
            with torch.no_grad():
                n.weight.normal_(0, 0.3)
            object.__setattr__(self, "_muse_post_expert_norm", n)


def _inputs(dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = (torch.randn(ROWS, H, generator=g) * 3).to(dtype)
    probs = torch.rand(ROWS, generator=g, dtype=torch.float32) * 0.9 + 0.05
    return out, probs


def _hf(norm, out, probs):
    # HF MuseSparkTextExperts: norm (scale, round, normalize, round), then * fp32 prob, round
    s = (out.float() * (1 + norm.weight.float())).to(out.dtype).float()
    n = (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + norm.eps)).to(out.dtype)
    return (n.float() * probs.float().unsqueeze(-1)).to(out.dtype)


def _run(experts, probs):
    return experts.forward(torch.empty(ROWS, H), torch.tensor([ROWS]), probs)


@pytest.mark.parametrize("fused", [True, False])
def test_the_parent_gets_unit_probs(fused):
    out, probs = _inputs()
    e = _Experts(out, fused)
    _run(e, probs)
    (seen,) = e.seen
    assert seen.shape == probs.shape and seen.dtype == probs.dtype
    assert torch.equal(seen, torch.ones_like(probs))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_eager_path_is_hfs_rounding_order(dtype):
    out, probs = _inputs(dtype)
    e = _Experts(out, fused=False)
    got, bias = _run(e, probs)
    assert bias is None and got.dtype == dtype
    want = _hf(e._muse_post_expert_norm, out, probs)
    assert torch.equal(got, want)
    # the prob is applied after the norm: before it, the RMS norm would cancel it
    s = (out.float() * probs.unsqueeze(-1)).to(dtype)
    assert not torch.allclose(got.float(), e._muse_post_expert_norm(s).float(), atol=1e-2)


def test_the_fused_gate_calls_the_kernel_entry_point(monkeypatch):
    out, probs = _inputs()
    e = _Experts(out, fused=True)
    calls = []

    def spy(y, w, p, eps):
        calls.append((y, w, p, eps))
        return torch.full_like(y, 7.0)

    monkeypatch.setattr(moe, "post_expert_norm_probs", spy)
    got, bias = _run(e, probs)
    (call,) = calls
    norm = e._muse_post_expert_norm
    assert torch.equal(call[0], out) and call[1] is norm.weight and call[2] is probs and call[3] == norm.eps
    assert bias is None and torch.equal(got, torch.full_like(out, 7.0))


def test_the_eager_gate_does_not_call_the_kernel(monkeypatch):
    out, probs = _inputs()
    e = _Experts(out, fused=False)
    monkeypatch.setattr(moe, "post_expert_norm_probs", lambda *a: pytest.fail("fused path taken"))
    _run(e, probs)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_and_eager_paths_agree_on_the_cpu(dtype):
    # on CPU the fused entry point runs its eager reference: the same bits as the eager path
    out, probs = _inputs(dtype, seed=1)
    fused_e, eager_e = _Experts(out, fused=True), _Experts(out, fused=False)
    eager_e._muse_post_expert_norm.load_state_dict(fused_e._muse_post_expert_norm.state_dict())
    fused, _ = _run(fused_e, probs)
    eager, _ = _run(eager_e, probs)
    assert torch.equal(fused, eager)


@pytest.mark.parametrize("fused", [True, False])
def test_without_a_norm_only_the_prob_is_applied(fused):
    out, probs = _inputs()
    got, _ = _run(_Experts(out, fused, norm=False), probs)
    assert torch.equal(got, (out.float() * probs.unsqueeze(-1)).to(out.dtype))


def test_an_expert_bias_is_an_error():
    out, probs = _inputs()
    e = _Experts(out, fused=False, bias=torch.zeros(H))
    with pytest.raises(RuntimeError, match="no bias"):
        _run(e, probs)


@pytest.mark.parametrize("fused", [True, False])
def test_the_norm_forward_pre_hooks_run_on_both_paths(fused):
    # DDP's overlap_param_gather waits for the norm weight's all-gather in a forward
    # pre-hook on the norm module; the fused kernel never calls norm.forward, so the
    # mixin runs the hooks itself before it reads norm.weight
    out, probs = _inputs()
    e = _Experts(out, fused)
    calls = []
    e._muse_post_expert_norm.register_forward_pre_hook(lambda mod, args: calls.append(mod))
    # a with_kwargs hook (as an inspection harness registers) gets the kwargs dict, as in torch
    e._muse_post_expert_norm.register_forward_pre_hook(
        lambda mod, args, kwargs: calls.append(kwargs), with_kwargs=True
    )
    _run(e, probs)
    assert calls == [e._muse_post_expert_norm, {}]


@pytest.mark.parametrize("text, fused", [("false", False), ("0", False), ("true", True), ("on", True)])
def test_the_fused_flag_is_parsed_strictly(monkeypatch, text, fused):
    out, probs = _inputs()
    e = _Experts(out, text)
    calls = []
    monkeypatch.setattr(moe, "post_expert_norm_probs", lambda *a: calls.append(a) or a[0])
    _run(e, probs)
    assert bool(calls) == fused


def test_a_fused_flag_typo_raises():
    out, probs = _inputs()
    with pytest.raises(ValueError, match="muse_spark_fused_post_expert_norm='ture'"):
        _run(_Experts(out, "ture"), probs)
