###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ZAYA_MOE_GEMM=turbo and ZAYA_ROUTER_DEFER_AR=1 against the default paths.

The CPU tests swap primus_turbo's grouped GEMM for a reference with the same
contract (rows past group_offs[-1] come back zero), so they check the
dispatch, the MoD skip slot, empty experts, gradients and checkpoint keys of
the grouped path against the per-expert split path.
"""

import pytest
import torch
from test_zaya1_math import _cfg  # noqa: E402  (pytest puts this dir on sys.path)

from primus.backends.megatron.core.models.zaya1 import zaya1_modules as zm


def _reference_grouped_gemm(a, b, group_lens, group_offs):
    lens = group_lens.tolist()
    offs = group_offs.tolist()
    pieces = [a[offs[g] : offs[g] + n] @ b[g].t() for g, n in enumerate(lens)]
    tail = a.shape[0] - offs[-1]
    pieces.append(a.new_zeros(tail, b.shape[1]) * a[offs[-1] :, :1])
    return torch.cat(pieces, dim=0)


@pytest.fixture
def turbo_cpu(monkeypatch):
    monkeypatch.setattr(zm, "_turbo_grouped_gemm", _reference_grouped_gemm)


def _build_moe(mode, monkeypatch, seed=0, **cfg_overrides):
    monkeypatch.setattr(zm, "MOE_GEMM", mode)
    torch.manual_seed(seed)
    return zm.ZayaMoE(_cfg(**cfg_overrides))


def _skew_routes(moe, empty=(1,), mod_bias=0.3):
    # Push some experts out of every topk and give the MoD slot a share.
    with torch.no_grad():
        for e in empty:
            moe.router.balancing_biases[e] = -100.0
        moe.router.balancing_biases[-1] = mod_bias


def _expert_grads(moe, fc):
    if moe.use_turbo_gemm:
        g = getattr(moe, f"{fc}_weight").grad
        return [g[i] for i in range(moe.num_real_experts)]
    out = []
    for expert in moe.local_experts:
        w = getattr(expert, f"linear_{fc}").weight
        out.append(torch.zeros_like(w) if w.grad is None else w.grad)
    return out


@pytest.mark.parametrize("fuse_moe", [True, False])
@pytest.mark.parametrize("use_mod", [True, False])
def test_turbo_matches_split_forward_and_backward(turbo_cpu, monkeypatch, use_mod, fuse_moe):
    monkeypatch.setattr(zm, "FUSE_MOE", fuse_moe)
    split = _build_moe("split", monkeypatch, zaya_use_mod=use_mod)
    turbo = _build_moe("turbo", monkeypatch, zaya_use_mod=use_mod)
    assert split.local_experts is not None and turbo.use_turbo_gemm
    # Same init stream, same weights.
    for i, expert in enumerate(split.local_experts):
        torch.testing.assert_close(turbo.fc1_weight[i], expert.linear_fc1.weight, rtol=0, atol=0)
        torch.testing.assert_close(turbo.fc2_weight[i], expert.linear_fc2.weight, rtol=0, atol=0)
    for moe in (split, turbo):
        moe.eval()  # keep the balancing bias fixed between the two runs
        _skew_routes(moe, empty=(1, 2) if use_mod else (1,))

    torch.manual_seed(1)
    hidden = torch.randn(16, 3, 32)
    prev = torch.randn(16, 3, 8)
    outs = {}
    for name, moe in (("split", split), ("turbo", turbo)):
        h = hidden.clone().requires_grad_(True)
        p = prev.clone().requires_grad_(True)
        out, nxt = moe(h, p)
        (out * torch.linspace(-1, 1, out.numel()).view_as(out)).sum().add(nxt.square().sum()).backward()
        outs[name] = (out.detach(), h.grad, p.grad)

    _, idx, _ = split.router(hidden, prev)
    routed = set(idx.flatten().tolist())
    assert 1 not in routed  # an empty expert is exercised
    if use_mod:
        assert split.num_real_experts in routed  # the skip slot is exercised

    for a, b in zip(outs["split"], outs["turbo"]):
        torch.testing.assert_close(b, a)
    for fc in ("fc1", "fc2"):
        for gs, gt in zip(_expert_grads(split, fc), _expert_grads(turbo, fc)):
            torch.testing.assert_close(gt, gs)
    assert torch.count_nonzero(turbo.fc1_weight.grad[1]) == 0
    for (n_s, p_s), (n_t, p_t) in zip(split.router.named_parameters(), turbo.router.named_parameters()):
        assert n_s == n_t
        if p_s.grad is None:
            # Only the first-layer EDA scale (prev_router given here, so none).
            assert torch.count_nonzero(p_t.grad) == 0
        else:
            torch.testing.assert_close(p_t.grad, p_s.grad)


def test_turbo_keeps_per_expert_checkpoint_keys(turbo_cpu, monkeypatch):
    split = _build_moe("split", monkeypatch, seed=3)
    turbo = _build_moe("turbo", monkeypatch, seed=4)
    sd_split = split.state_dict()
    sd_turbo = turbo.state_dict()
    assert sorted(sd_split) == sorted(sd_turbo)
    assert "local_experts.0.linear_fc1.weight" in sd_turbo
    assert not any(k.startswith("fc1_weight") or k.startswith("fc2_weight") for k in sd_turbo)
    for k, v in sd_split.items():
        assert sd_turbo[k].shape == v.shape

    # split -> turbo
    turbo.load_state_dict(sd_split, strict=True)
    for i, expert in enumerate(split.local_experts):
        torch.testing.assert_close(turbo.fc1_weight[i], expert.linear_fc1.weight, rtol=0, atol=0)
        torch.testing.assert_close(turbo.fc2_weight[i], expert.linear_fc2.weight, rtol=0, atol=0)
    # turbo -> split, through a prefixed parent module
    with torch.no_grad():
        turbo.fc2_weight.add_(1.0)
    parent_t = torch.nn.ModuleDict({"moe": turbo})
    parent_s = torch.nn.ModuleDict({"moe": split})
    parent_s.load_state_dict(parent_t.state_dict(), strict=True)
    for i, expert in enumerate(split.local_experts):
        torch.testing.assert_close(expert.linear_fc2.weight, turbo.fc2_weight[i], rtol=0, atol=0)
    names = [n for n, _ in turbo.named_parameters()]
    assert "fc1_weight" in names and "fc2_weight" in names


def test_turbo_load_reports_missing_expert_keys(turbo_cpu, monkeypatch):
    split = _build_moe("split", monkeypatch, seed=3)
    partial = split.state_dict()
    del partial["local_experts.2.linear_fc1.weight"]
    turbo = _build_moe("turbo", monkeypatch, seed=4)
    with pytest.raises(RuntimeError, match=r"Missing key\(s\).*local_experts\.2\.linear_fc1\.weight"):
        turbo.load_state_dict(partial, strict=True)
    # Not strict: the other experts load, as separate per-expert weights would.
    turbo = _build_moe("turbo", monkeypatch, seed=4)
    kept = turbo.fc1_weight[2].detach().clone()
    result = turbo.load_state_dict(partial, strict=False)
    assert result.missing_keys == ["local_experts.2.linear_fc1.weight"]
    assert result.unexpected_keys == []
    for i, expert in enumerate(split.local_experts):
        fc1 = kept if i == 2 else expert.linear_fc1.weight
        torch.testing.assert_close(turbo.fc1_weight[i], fc1, rtol=0, atol=0)
        torch.testing.assert_close(turbo.fc2_weight[i], expert.linear_fc2.weight, rtol=0, atol=0)


def test_turbo_mod_skip_scales_the_hidden_state(turbo_cpu, monkeypatch):
    moe = _build_moe("turbo", monkeypatch)
    with torch.no_grad():
        for p in moe.router.parameters():
            p.zero_()
        moe.router.balancing_biases.zero_()
        moe.router.balancing_biases[-1] = 10.0
    hidden = torch.randn(4, 2, 32)
    out, _ = moe(hidden, prev_router=None)
    torch.testing.assert_close(out, hidden / (moe.num_real_experts + 1))


def test_turbo_falls_back_to_split_under_fp8(monkeypatch):
    monkeypatch.setattr(zm, "MOE_GEMM", "turbo")
    monkeypatch.setattr(zm, "_fp8_requested", lambda config: False)
    assert zm.ZayaMoE(_cfg()).use_turbo_gemm
    monkeypatch.setattr(zm, "_fp8_requested", lambda config: True)
    monkeypatch.setattr(zm, "_linear", lambda i, o, b, c, fp8=True: torch.nn.Linear(i, o, bias=b))
    assert not zm.ZayaMoE(_cfg()).use_turbo_gemm


def _build_stack(defer, monkeypatch):
    monkeypatch.setattr(zm, "ROUTER_DEFER_AR", defer)
    torch.manual_seed(7)
    stack = zm.ZayaStack(_cfg(num_layers=6))
    stack.train()
    return stack


def _biases(stack):
    return [r.balancing_biases.clone() for r in stack._routers()]


def test_deferred_router_updates_are_bit_exact(monkeypatch):
    eager = _build_stack(False, monkeypatch)
    deferred = _build_stack(True, monkeypatch)
    assert all(r.defer_bias_update for r in deferred._routers())
    reduces = []
    monkeypatch.setattr(zm, "_allreduce_dp", lambda t: reduces.append(t.numel()))
    n_routers = len(eager._routers())
    for step in range(3):
        torch.manual_seed(100 + step)
        hidden = torch.randn(8, 2, 32)
        outs = []
        for stack in (eager, deferred):
            h = hidden.clone().requires_grad_(True)
            out = stack(h, position_ids=None, attention_mask=None)
            out.square().sum().backward()
            outs.append(out.detach())
        # Outputs see identical biases, and after backward the biases match.
        torch.testing.assert_close(outs[1], outs[0], rtol=0, atol=0)
        for a, b in zip(_biases(eager), _biases(deferred)):
            torch.testing.assert_close(b, a, rtol=0, atol=0)
        assert all(r._pending_counts is None for r in deferred._routers())
        for a, b in zip(eager._routers(), deferred._routers()):
            assert int(a.balance_step) == int(b.balance_step) == step + 1
    # eager: one reduce per router per step; deferred: one per step.
    assert reduces.count(eager._routers()[0].num_scored) == 3 * n_routers
    assert reduces.count(n_routers * eager._routers()[0].num_scored) == 3


def test_deferred_router_flushes_on_next_forward_without_backward(monkeypatch):
    eager = _build_stack(False, monkeypatch)
    deferred = _build_stack(True, monkeypatch)
    hidden = torch.randn(8, 2, 32)
    with torch.no_grad():
        eager(hidden, None, None)
        deferred(hidden, None, None)
    assert all(r._pending_counts is not None for r in deferred._routers())
    eager.eval()
    deferred.eval()
    with torch.no_grad():
        a = eager(hidden, None, None)
        b = deferred(hidden, None, None)
    torch.testing.assert_close(b, a, rtol=0, atol=0)
    for x, y in zip(_biases(eager), _biases(deferred)):
        torch.testing.assert_close(y, x, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU and primus_turbo")
def test_turbo_gpu_matches_split_without_host_sync(monkeypatch):
    pytest.importorskip("primus_turbo.pytorch.ops")
    split = (
        _build_moe(
            "split",
            monkeypatch,
            hidden_size=256,
            moe_ffn_hidden_size=128,
            ffn_hidden_size=128,
            num_moe_experts=16,
        )
        .cuda()
        .bfloat16()
    )
    turbo = (
        _build_moe(
            "turbo",
            monkeypatch,
            hidden_size=256,
            moe_ffn_hidden_size=128,
            ffn_hidden_size=128,
            num_moe_experts=16,
        )
        .cuda()
        .bfloat16()
    )
    for moe in (split, turbo):
        moe.eval()
        _skew_routes(moe, empty=(3,))
    hidden = torch.randn(64, 8, 256, device="cuda", dtype=torch.bfloat16)
    ref, _ = split(hidden, None)
    turbo(hidden, None)  # warm up / autotune
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out, _ = turbo(hidden, None)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("fuse_moe", [True, False])
def test_turbo_every_param_gets_a_grad(turbo_cpu, monkeypatch, fuse_moe):
    """--overlap_grad_reduce needs a grad hook on every param every step."""
    monkeypatch.setattr(zm, "FUSE_MOE", fuse_moe)
    monkeypatch.setattr(zm, "MOE_GEMM", "split")
    torch.manual_seed(11)
    split = zm.ZayaStack(_cfg(num_layers=6))
    monkeypatch.setattr(zm, "MOE_GEMM", "turbo")
    torch.manual_seed(11)
    turbo = zm.ZayaStack(_cfg(num_layers=6))
    turbo.load_state_dict(split.state_dict(), strict=True)
    for stack in (split, turbo):
        stack.eval()  # fixed balancing biases
        for r in stack._routers():
            with torch.no_grad():
                r.balancing_biases[1] = -100.0  # expert 1 gets no tokens
    torch.manual_seed(12)
    hidden = torch.randn(8, 2, 32)
    outs = []
    for stack in (split, turbo):
        h = hidden.clone().requires_grad_(True)
        out = stack(h, position_ids=None, attention_mask=None)
        out.square().sum().backward()
        outs.append((out.detach(), h.grad))
    for a, b in zip(*outs):
        torch.testing.assert_close(b, a)
    split_missing = [n for n, p in split.named_parameters() if p.grad is None]
    assert any("router_states_scale" in n for n in split_missing)  # the bug being fixed
    assert any("local_experts.1." in n for n in split_missing)
    assert [n for n, p in turbo.named_parameters() if p.grad is None] == []
    sp = dict(split.named_parameters())
    for n, p in turbo.named_parameters():
        if n in sp:
            ref = sp[n].grad if sp[n].grad is not None else torch.zeros_like(p)
            torch.testing.assert_close(p.grad, ref)
