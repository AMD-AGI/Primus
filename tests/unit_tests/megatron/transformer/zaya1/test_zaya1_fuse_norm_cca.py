###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU equivalence of the ZAYA_FUSE_NORM and ZAYA_FUSE_CCA paths against the
eager path, forward and backward, fp32 and bf16."""

import pytest
import torch
from test_zaya1_math import _cfg  # noqa: E402  (pytest puts this dir on sys.path)

from primus.backends.megatron.core.models.zaya1 import zaya1_modules as zm


def _perturb(module):
    """Move every parameter off its init so scales, biases and norm weights all matter."""
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.1 * torch.randn_like(p))


def _forward_backward(module, fn, hidden):
    module.zero_grad(set_to_none=True)
    h = hidden.detach().clone().requires_grad_(True)
    out = fn(h)
    outs = out if isinstance(out, tuple) else (out,)
    sum((o.float() * torch.linspace(-1, 1, o.numel()).view_as(o)).sum() for o in outs).backward()
    grads = {n: p.grad.detach().clone() for n, p in module.named_parameters() if p.grad is not None}
    return [o.detach() for o in outs] + [h.grad], grads


def _assert_close(got, ref, dtype, name=""):
    if dtype == torch.float32:
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4, msg=name)
        return
    # bf16 rounding differs between Inductor's fused kernels and eager, so
    # compare whole tensors instead of single elements.
    err = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
    assert err < 2e-2, f"{name}: relative error {err:.3e}"


def _compare(eager, fused, dtype):
    (out_e, grads_e), (out_f, grads_f) = eager, fused
    for i, (a, b) in enumerate(zip(out_f, out_e)):
        _assert_close(a, b, dtype, f"output {i}")
    assert grads_f.keys() == grads_e.keys()
    for name, ref in grads_e.items():
        _assert_close(grads_f[name], ref, dtype, name)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_norm_chain_matches_eager(monkeypatch, dtype):
    """Stage 0's scale + RMSNorm, every later stage's scale + merge + RMSNorm, and the final norm."""
    fused_res = zm._maybe_compile(zm._scale_merge_norm_res, True)
    fused_first = zm._maybe_compile(zm._scale_merge_norm_first, True)
    monkeypatch.setattr(zm, "_fused_smn_res", fused_res)
    monkeypatch.setattr(zm, "_fused_smn_first", fused_first)
    torch.manual_seed(0)
    stack = zm.ZayaStack(_cfg(num_layers=4)).to(dtype)
    _perturb(stack)
    stack.eval()  # keep the balancing biases fixed between the two runs
    hidden = torch.randn(6, 2, 32, dtype=dtype)
    runs = {}
    for flag in (False, True):
        monkeypatch.setattr(zm, "FUSE_NORM", flag)
        runs[flag] = _forward_backward(
            stack, lambda h: stack(h, position_ids=None, attention_mask=None), hidden
        )
    _compare(runs[False], runs[True], dtype)
    assert fused_res.compiled_ok() and fused_first.compiled_ok()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "clamp_temp, rotary_factor, positions",
    [
        (False, 0.5, "none"),
        (False, 0.5, "per_row"),
        (True, 0.5, "per_row"),
        (True, 0.0, "none"),
    ],
)
def test_fused_cca_qk_matches_eager(monkeypatch, dtype, clamp_temp, rotary_factor, positions):
    """Grouped means, qk-norm, temperature and partial RoPE (1-D and per-row positions, or none)."""
    fused = zm._maybe_compile(zm._cca_qk_post, True)
    monkeypatch.setattr(zm, "_fused_cca_qk", fused)
    torch.manual_seed(1)
    cca = zm.CCA(_cfg(clamp_temp=clamp_temp, partial_rotary_factor=rotary_factor)).to(dtype)
    _perturb(cca)
    with torch.no_grad():
        cca.temp.uniform_(0.2, 1.5)
    seq, batch = 7, 3
    position_ids = None
    if positions == "per_row":
        position_ids = torch.stack([torch.arange(seq) + 3 * row for row in range(batch)])
    hidden = torch.randn(seq, batch, 32, dtype=dtype)
    runs = {}
    for flag in (False, True):
        monkeypatch.setattr(zm, "FUSE_CCA", flag)
        runs[flag] = _forward_backward(cca, lambda h: cca.project(h, position_ids), hidden)
    _compare(runs[False], runs[True], dtype)
    assert fused.compiled_ok()
