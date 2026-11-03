###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling MoE: routing / combine against an independent per-token reference, the
expert-parallel split (each EP rank's partial output sums to the full output), the
expert-bias load-balancing buffers and the refusal of unsupported MoE settings."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "megatron.core.transformer.module", reason="megatron.core is not importable in this environment"
)
if not torch.distributed.is_available():
    pytest.skip("torch.distributed is not available", allow_module_level=True)

import importlib.util  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import torch.nn.functional as F  # noqa: E402

from primus.backends.megatron.core.transformer.inkling.inkling_moe import (  # noqa: E402
    InklingMoE,
)

from ._helpers import parallel, small_config  # noqa: E402,F401


def _moe(cfg=None):
    moe = InklingMoE(cfg or small_config(), layer_number=1, inkling_layer_idx=0)
    with torch.no_grad():
        moe.gate.global_scale.fill_(1.3)
        moe.gate.expert_bias.normal_(std=0.1)
    return moe


def _reference(moe, x):
    """Token by token, written from the HF modeling description."""
    cfg = moe.config
    E, S, k = cfg.num_moe_experts, cfg.inkling_num_shared_experts, cfg.moe_router_topk
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        logits = moe.gate.weight @ x[t]
        routed, shared = logits[:E], logits[E:]
        idx = torch.topk(routed.sigmoid() + moe.gate.expert_bias, k).indices
        lp = F.logsigmoid(torch.cat([routed[idx], shared]))
        w = lp.softmax(-1) * cfg.inkling_route_scale * moe.gate.global_scale
        for j, e in enumerate(idx.tolist()):
            gate, up = (moe.gate_up_proj[e] @ x[t]).chunk(2)
            out[t] += w[j] * (moe.down_proj[e] @ (F.silu(gate) * up))
        for s in range(S):
            sgate, sup = (moe.shared_gate_up_proj[s] @ x[t]).chunk(2)
            h = F.silu(sgate) * sup
            out[t] += w[k + s] * (moe.shared_down_proj[s] @ h)
    return out


def test_moe_matches_reference(parallel):  # noqa: F811
    torch.manual_seed(0)
    moe = _moe().eval()
    x = torch.randn(5, 2, moe.config.hidden_size)
    with torch.no_grad():
        out, bias = moe(x)
        want = _reference(moe, x.reshape(-1, x.shape[-1])).view_as(x)
    assert bias is None
    torch.testing.assert_close(out, want, rtol=1e-4, atol=1e-5)


def _ep_worker(rank, world, tp, ep, store_path, out_dir):
    """One rank of a real TP x EP run (gloo): InklingMoE with its EP all-gather / reduce-scatter,
    on the tokens of this rank's data-parallel slice (the TP ranks of a slice share them)."""
    from megatron.core import parallel_state

    from ._helpers import init_parallel

    # gloo, CPU tensors: Megatron's gather / reduce-scatter allocate on current_device(), so pin it
    # to the CPU also on a GPU host (this is a spawned process: the patch dies with it)
    torch.cuda.current_device = lambda: "cpu"
    init_parallel(store_path, rank=rank, world_size=world, expert_parallel_size=ep, tensor_parallel_size=tp)
    assert parallel_state.get_expert_model_parallel_world_size() == ep
    dp_rank = parallel_state.get_data_parallel_rank()
    torch.manual_seed(0)  # _moe() draws the router's expert_bias: the same on every rank
    moe = _moe(small_config(tensor_model_parallel_size=tp))
    gen = torch.Generator().manual_seed(7)
    n, H = 5, moe.config.hidden_size
    blocks = world // tp
    x_all = torch.randn(blocks * n, H, generator=gen)
    g_all = torch.randn(blocks * n, H, generator=gen)
    rows = slice(dp_rank * n, (dp_rank + 1) * n)
    x = x_all[rows].clone().requires_grad_()
    out, _ = moe(x.view(n, 1, H))
    (out.view(n, H) * g_all[rows]).sum().backward()
    sd = moe.sharded_state_dict()
    torch.save(
        {
            "out": out.detach().view(n, H),
            "x_grad": x.grad,
            "rows": rows,
            "tp_rank": parallel_state.get_tensor_model_parallel_rank(),
            "e0": moe.e0,
            "n_local": moe.num_local_experts,
            "state": {k: v.detach() for k, v in moe.state_dict().items()},
            "grads": {k: p.grad for k, p in moe.named_parameters()},
            "replica_id": {k: sd[k].replica_id for k in ("gate_up_proj", "down_proj")},
            "global_offset": sd["gate_up_proj"].global_offset,
            "expt_dp_rank": parallel_state.get_expert_data_parallel_rank(),
            "x_all": x_all,
            "g_all": g_all,
        },
        f"{out_dir}/rank{rank}.pt",
    )
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    "world,tp,ep", [(2, 1, 2), (4, 1, 4), (4, 2, 2), (4, 1, 2)], ids=["ep2", "ep4", "tp2-ep2", "ep2-edp2"]
)
def test_expert_parallel_matches_single_rank(parallel, tmp_path, world, tp, ep):  # noqa: F811
    """A real TP x EP run (``world`` gloo processes, Megatron's groups, the layer's own
    all-gather / reduce-scatter; with EP = 4 some ranks get no token) against a single-rank
    module built independently (EP = 1, same seeds):

    * the EP shards hold the single-rank initial weights (layout-independent init);
    * outputs and input gradients agree;
    * every gradient, summed over the data-parallel slices as DDP does (expert weights over
      the expert-DP group, the replicated router / global_scale / shared experts over DP), is
      the single-rank gradient; the TP replicas hold identical gradients;
    * each expert shard is checkpointed at its global-expert offset with replica id
      (0, expert-TP rank, expert-DP rank)."""
    import torch.multiprocessing as mp

    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("gloo is not available")
    store_path = str(tmp_path / "store")
    mp.spawn(_ep_worker, args=(world, tp, ep, store_path, str(tmp_path)), nprocs=world, join=True)
    ranks = [torch.load(tmp_path / f"rank{r}.pt", weights_only=False) for r in range(world)]

    torch.manual_seed(0)
    ref = _moe()  # EP = TP = 1, built from the seeds, not from the shards
    for got in ranks:
        experts = slice(got["e0"], got["e0"] + got["n_local"])
        for k, v in got["state"].items():
            want = ref.state_dict()[k]
            torch.testing.assert_close(v, want[experts] if k in ("gate_up_proj", "down_proj") else want)
    x_all, g_all = ranks[0]["x_all"], ranks[0]["g_all"]
    x = x_all.clone().requires_grad_()
    out, _ = ref(x.view(x.shape[0], 1, -1))
    (out.view_as(x) * g_all).sum().backward()

    summed = {k: torch.zeros_like(p) for k, p in ref.named_parameters()}
    for r, got in enumerate(ranks):
        assert got["n_local"] * ep == ref.num_local_experts and got["e0"] % got["n_local"] == 0
        rows = got["rows"]
        torch.testing.assert_close(got["out"], out.detach().view_as(x)[rows], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(got["x_grad"], x.grad[rows], rtol=1e-5, atol=1e-6)
        experts = slice(got["e0"], got["e0"] + got["n_local"])
        if got["tp_rank"] == 0:
            for k, g in got["grads"].items():
                if k in ("gate_up_proj", "down_proj"):
                    summed[k][experts] += g
                else:
                    summed[k] += g
        else:  # a TP replica: the same tokens, the same gradients as its TP rank 0
            twin = next(o for o in ranks if o["tp_rank"] == 0 and o["rows"] == rows)
            for k, g in got["grads"].items():
                torch.testing.assert_close(g, twin["grads"][k])
        assert got["global_offset"][0] == got["e0"]
        for rid in got["replica_id"].values():
            assert rid == (0, got["tp_rank"], got["expt_dp_rank"])
    assert {o["expt_dp_rank"] for o in ranks} == set(range(world // (tp * ep)))
    for k, p in ref.named_parameters():
        torch.testing.assert_close(summed[k], p.grad, rtol=1e-5, atol=1e-6, msg=k)


def _reference_grouped_gemm(a, b, group_lens, trans_b=True):
    """What primus_turbo.pytorch.ops.grouped_gemm computes (row groups of ``a`` times ``b[g]``)."""
    assert trans_b
    return torch.cat([s @ w.t() for s, w in zip(a.split(group_lens.tolist()), b)], 0)


def test_grouped_gemm_path_matches_loop(parallel):  # noqa: F811
    torch.manual_seed(3)
    moe = _moe().eval()
    x = torch.randn(9, moe.config.hidden_size)
    with torch.no_grad():
        w, idx, _ = moe.gate(x)
        loop = moe._experts(x, idx, w)
        moe.grouped_gemm = _reference_grouped_gemm
        grouped = moe._experts(x, idx, w)
    torch.testing.assert_close(grouped, loop, rtol=1e-5, atol=1e-6)


def test_turbo_grouped_gemm_follows_the_turbo_switches(parallel, monkeypatch):  # noqa: F811
    """The Primus-Turbo grouped GEMM is used only when enable_primus_turbo and
    use_turbo_grouped_gemm are both set (as for the other Primus MoE layers); then a missing
    Primus-Turbo refuses the build (and names the fix) instead of silently running the loop.
    With either switch off the loop runs, with or without Primus-Turbo installed."""
    import megatron.training as mt

    from primus.backends.megatron.core.transformer.inkling import inkling_moe

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    for turbo, grouped in ((False, True), (True, False), (False, False)):
        args = types.SimpleNamespace(enable_primus_turbo=turbo, use_turbo_grouped_gemm=grouped)
        monkeypatch.setattr(mt, "get_args", lambda: args)
        assert _moe().grouped_gemm is None
    args = types.SimpleNamespace(enable_primus_turbo=True, use_turbo_grouped_gemm=True)
    monkeypatch.setattr(mt, "get_args", lambda: args)
    with pytest.raises(ImportError, match="use_turbo_grouped_gemm: false"):
        _moe()
    monkeypatch.setattr(inkling_moe, "_turbo_grouped_gemm", lambda: _reference_grouped_gemm)
    assert _moe().grouped_gemm is _reference_grouped_gemm


def test_turbo_probe_catches_only_the_not_initialized_case(monkeypatch):
    """Megatron not loaded, or loaded but its args not initialized: "not requested". Any other
    error from get_args propagates (no silent fallback to the per-expert loop)."""
    from primus.backends.megatron.core.transformer.inkling import inkling_moe

    monkeypatch.delitem(sys.modules, "megatron.training", raising=False)
    assert inkling_moe._turbo_grouped_gemm_requested() is False

    def not_initialized():
        raise AssertionError("args is not initialized.")

    def broken():
        raise RuntimeError("a broken Megatron")

    fake = types.ModuleType("megatron.training")
    monkeypatch.setitem(sys.modules, "megatron.training", fake)
    fake.get_args = not_initialized
    assert inkling_moe._turbo_grouped_gemm_requested() is False
    fake.get_args = broken
    with pytest.raises(RuntimeError, match="broken Megatron"):
        inkling_moe._turbo_grouped_gemm_requested()
    fake.get_args = lambda: types.SimpleNamespace(enable_primus_turbo=True, use_turbo_grouped_gemm=True)
    assert inkling_moe._turbo_grouped_gemm_requested() is True


def test_megatron_grouped_gemm_flag_does_not_select_turbo(parallel):  # noqa: F811
    """moe_grouped_gemm (needed by Megatron for moe_act recompute) does not select the kernel."""
    assert _moe(small_config(moe_grouped_gemm=True)).grouped_gemm is None


def test_rank_without_tokens_keeps_experts_in_graph(parallel):  # noqa: F811
    """An EP rank whose experts got no token still produces (zero) expert gradients."""
    moe = _moe()
    x = torch.randn(3, moe.config.hidden_size)
    w, idx, _ = moe.gate(x)
    x.requires_grad_()
    out = moe._experts(x, idx + moe.config.num_moe_experts, w)  # every slot routed off-rank
    assert torch.equal(out, torch.zeros_like(out))
    out.sum().backward()
    assert torch.equal(moe.gate_up_proj.grad, torch.zeros_like(moe.gate_up_proj))
    # the input stays in the graph, so the EP all-gather's backward (a collective) still runs
    assert x.grad is not None


def test_padding_mask_is_refused(parallel):  # noqa: F811
    """Megatron's MoE padding_mask is refused (not silently ignored) by the layer and the router."""
    moe = _moe()
    x = torch.randn(3, 1, moe.config.hidden_size)
    mask = torch.zeros(1, 3, dtype=torch.bool)
    with pytest.raises(NotImplementedError, match="padding_mask"):
        moe(x, padding_mask=mask)
    with pytest.raises(NotImplementedError, match="padding_mask"):
        moe.gate(x.view(3, -1), mask)
    moe(x, padding_mask=None)  # None is accepted


def test_expert_init_differs_per_layer(parallel):  # noqa: F811
    """The seeds include the layer (layout independence over EP: test_expert_parallel_matches_single_rank)."""
    moe = _moe()
    other_layer = InklingMoE(small_config(), layer_number=2, inkling_layer_idx=1)
    assert not torch.equal(other_layer.gate_up_proj, moe.gate_up_proj)


def test_unrouted_experts_get_zero_grad_and_nan_does_not_leak(parallel):  # noqa: F811
    torch.manual_seed(2)
    moe = _moe()
    x = torch.randn(1, 1, moe.config.hidden_size)
    _, idx, _ = moe.gate(x.view(1, -1))
    unused = sorted(set(range(moe.config.num_moe_experts)) - set(idx.view(-1).tolist()))[0]
    with torch.no_grad():
        moe.gate_up_proj[unused].fill_(float("nan"))
    out, _ = moe(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert moe.gate_up_proj.grad is not None
    assert torch.equal(moe.gate_up_proj.grad[unused], torch.zeros_like(moe.gate_up_proj.grad[unused]))


def test_expert_bias_buffers_for_aux_free_balancing(parallel, monkeypatch):  # noqa: F811
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")  # Megatron allocates them on CUDA
    cfg = small_config(moe_router_enable_expert_bias=True, moe_router_bias_update_rate=1e-3)
    moe = _moe(cfg).train()
    gate = moe.gate
    assert gate.expert_bias.dtype == torch.float32
    assert "expert_bias" in gate.state_dict() and "local_tokens_per_expert" not in gate.state_dict()
    x = torch.randn(4, 3, cfg.hidden_size)
    moe(x)  # grad-enabled training forward: counted
    assert gate.local_tokens_per_expert.sum().item() == 4 * 3 * cfg.moe_router_topk


@pytest.mark.parametrize(
    "override",
    [
        {"moe_router_load_balancing_type": "aux_loss", "moe_aux_loss_coeff": 0.01},
        {"moe_aux_loss_coeff": 0.01},
        {"moe_aux_loss_coeff": [0.0, 0.01]},
        {"moe_z_loss_coeff": 0.001},
        {"moe_input_jitter_eps": 0.01},
        {"moe_expert_capacity_factor": 1.0},
        {"moe_shared_expert_intermediate_size": 32},
        {"moe_router_num_groups": 2},
        {"moe_router_force_load_balancing": True},
        {"moe_latent_size": 16},
        {"moe_router_score_function": "softmax"},
        {"moe_router_pre_softmax": True},
        {"moe_router_topk_scaling_factor": 2.0},
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()),
)
def test_unsupported_moe_settings_are_refused(parallel, override):  # noqa: F811
    """Each MoE knob InklingMoE does not implement is refused by InklingMoE itself. The value is
    set on a valid config after it is built, so Megatron's own config validation (which may
    refuse some of them first, or require companion settings) cannot hide a missing check."""
    cfg = small_config()
    InklingMoE(cfg, layer_number=1, inkling_layer_idx=0)  # the valid config builds
    for key, value in override.items():
        setattr(cfg, key, value)
    with pytest.raises(ValueError, match="InklingMoE does not implement") as err:
        InklingMoE(cfg, layer_number=1, inkling_layer_idx=0)
    for key in override:
        assert key in str(err.value)


def test_turbo_that_fails_to_import_is_refused(monkeypatch):
    """An installed Primus-Turbo that raises on import (e.g. no visible GPU at build time, or a
    version mismatch) is refused with the reason, as a missing package is."""
    import builtins

    from primus.backends.megatron.core.transformer.inkling import inkling_moe

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: types.SimpleNamespace())
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name.startswith("primus_turbo"):
            raise RuntimeError("No HIP GPUs are available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(ImportError, match="No HIP GPUs are available"):
        inkling_moe._turbo_grouped_gemm()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_turbo_grouped_gemm_matches_loop_with_grads(parallel):  # noqa: F811
    """The real Primus-Turbo grouped GEMM (the path the example configs enable)
    matches the per-expert loop in outputs and in input / expert-weight gradients, also when some
    local experts get no token."""
    from primus.backends.megatron.core.transformer.inkling import inkling_moe

    torch.manual_seed(5)
    moe = _moe().eval().cuda()
    x = torch.randn(7, moe.config.hidden_size, device="cuda")
    with torch.no_grad():
        w, idx, _ = moe.gate(x)
        for p in (moe.gate_up_proj, moe.down_proj):  # the expert GEMMs run in bf16 in training
            p.data = p.data.to(torch.bfloat16)
    x = x.to(torch.bfloat16)
    idx = idx.clamp(max=moe.num_local_experts - 2)  # the last local experts get no token
    turbo = inkling_moe._turbo_grouped_gemm()  # the default path: fails (not skips) without Turbo
    outs, grads = [], []
    for gg in (None, turbo):
        moe.grouped_gemm = gg
        moe.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_()
        y = moe._experts(xi, idx, w)
        y.float().square().sum().backward()
        outs.append(y.float())
        grads.append([xi.grad.float(), moe.gate_up_proj.grad.float(), moe.down_proj.grad.float()])
    torch.testing.assert_close(outs[1], outs[0], rtol=2e-2, atol=2e-2)
    for a, b in zip(grads[1], grads[0]):
        torch.testing.assert_close(a, b, rtol=3e-2, atol=3e-2)


def test_recompute_does_not_double_count_expert_bias_tokens(parallel, monkeypatch):  # noqa: F811
    """With the ``moe`` recompute the layer runs twice per step (no-grad forward, then the
    grad-enabled recompute); only the grad-enabled run counts tokens, as Megatron's router."""
    from megatron.core.tensor_parallel import random as tp_random

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")  # Megatron allocates them on CUDA
    # with current_device patched the CUDA RNG cannot be addressed either: the MoE draws no random
    # numbers, so the checkpoint's CUDA RNG save / restore is a no-op here
    monkeypatch.setattr(tp_random, "_get_cuda_rng_state", lambda *a, **k: None)
    monkeypatch.setattr(tp_random, "_set_cuda_rng_state", lambda *a, **k: None)
    for kw in ({}, {"recompute_granularity": "selective", "recompute_modules": ["moe"]}):
        cfg = small_config(moe_router_enable_expert_bias=True, moe_router_bias_update_rate=1e-3, **kw)
        moe = _moe(cfg).train()
        x = torch.randn(4, 3, cfg.hidden_size, requires_grad=True)
        moe(x)[0].sum().backward()
        assert moe.gate.local_tokens_per_expert.sum().item() == 4 * 3 * cfg.moe_router_topk, kw
        with torch.no_grad():  # a no-grad (evaluation-style) forward counts nothing
            moe(x)
        assert moe.gate.local_tokens_per_expert.sum().item() == 4 * 3 * cfg.moe_router_topk, kw


def _cpu_only_rng(monkeypatch, tp_random):
    """Megatron's checkpoints save / restore the CUDA RNG too; on a host without a GPU only the
    CPU state matters (the MoE draws no random numbers), so make the CUDA part a no-op there."""
    if not torch.cuda.is_available():
        monkeypatch.setattr(tp_random, "_get_cuda_rng_state", lambda *a, **k: None)
        monkeypatch.setattr(tp_random, "_set_cuda_rng_state", lambda *a, **k: None)


@pytest.mark.parametrize(
    "modules", [["moe"], ["shared_experts"], ["moe_act"], ["moe", "shared_experts", "moe_act"]]
)
def test_moe_selective_recompute_matches(parallel, monkeypatch, modules):  # noqa: F811
    """Each MoE recompute module gives the same output and gradients as no recompute
    (``moe_act`` takes effect on the grouped-GEMM path, faked here by a per-expert loop)."""
    from megatron.core.tensor_parallel import random as tp_random

    from primus.backends.megatron.core.transformer.inkling import inkling_moe

    _cpu_only_rng(monkeypatch, tp_random)

    def fake_grouped_gemm(a, w, group_lens, trans_b):
        outs, start = [], 0
        for j, c in enumerate(group_lens.tolist()):
            outs.append(a[start : start + c] @ (w[j].t() if trans_b else w[j]))
            start += c
        return torch.cat(outs)

    monkeypatch.setattr(inkling_moe, "_turbo_grouped_gemm", lambda: fake_grouped_gemm)
    monkeypatch.setattr(inkling_moe, "_turbo_grouped_gemm_requested", lambda: True)

    # spy on the two recompute entry points, so each module is shown to take its own path
    calls = {"checkpoint": [], "act": 0}
    real_checkpoint = inkling_moe.tensor_parallel.checkpoint
    real_cwo = inkling_moe.tensor_parallel.CheckpointWithoutOutput

    def spy_checkpoint(fn, *args, **kwargs):
        calls["checkpoint"].append(getattr(fn, "__name__", repr(fn)))
        return real_checkpoint(fn, *args, **kwargs)

    class SpyCheckpointWithoutOutput(real_cwo):
        def __init__(self, *args, **kwargs):
            calls["act"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(inkling_moe.tensor_parallel, "checkpoint", spy_checkpoint)
    monkeypatch.setattr(inkling_moe.tensor_parallel, "CheckpointWithoutOutput", SpyCheckpointWithoutOutput)
    results = []
    for kw in ({}, {"recompute_granularity": "selective", "recompute_modules": modules}):
        torch.manual_seed(0)
        calls["checkpoint"].clear()
        calls["act"] = 0
        moe = _moe(small_config(moe_grouped_gemm=True, **kw)).train()
        assert moe.recompute_act == ("moe_act" in modules and bool(kw))
        x = torch.randn(5, 2, moe.config.hidden_size, generator=torch.Generator().manual_seed(1))
        x.requires_grad_()
        out = moe(x)[0]
        out.square().sum().backward()
        results.append((out.detach(), x.grad, {n: p.grad for n, p in moe.named_parameters()}))
        on = set(modules) if kw else set()
        # "moe" checkpoints the whole layer (_forward); "shared_experts" the shared experts
        # (_shared), only when "moe" does not already cover them; "moe_act" the routed SwiGLU
        want = ["_forward"] if "moe" in on else (["_shared"] if "shared_experts" in on else [])
        assert calls["checkpoint"] == want, (kw, calls)
        assert (calls["act"] > 0) == ("moe_act" in on), (kw, calls)
    torch.testing.assert_close(results[1][0], results[0][0])
    torch.testing.assert_close(results[1][1], results[0][1])
    for n, g in results[0][2].items():
        torch.testing.assert_close(results[1][2][n], g, msg=n)


def test_sharded_state_dict_keeps_parameter_order(parallel):  # noqa: F811
    """sharded_state_dict does not touch the module (parameter order and identity unchanged, so
    optimizer state that is matched by position stays valid), returns the expert keys as EP
    ShardedTensors and can be called again."""
    moe = _moe()
    before = [(n, id(p)) for n, p in moe.named_parameters()]
    first = moe.sharded_state_dict(prefix="mlp.")
    second = moe.sharded_state_dict(prefix="mlp.")
    assert [(n, id(p)) for n, p in moe.named_parameters()] == before
    assert set(first) == set(second)
    assert first["mlp.gate_up_proj"].data is moe.gate_up_proj
    assert first["mlp.gate_up_proj"].global_shape[0] == moe.config.num_moe_experts


def test_shared_experts_match_the_per_expert_sum_with_grads(parallel):  # noqa: F811
    """_shared (two GEMMs over the stacked shared experts) equals the per-expert sum of
    gamma_s * SwiGLU expert s, in values and gradients."""
    moe = _moe()
    torch.manual_seed(0)
    S = moe.shared_gate_up_proj.shape[0]
    x = torch.randn(5, moe.shared_gate_up_proj.shape[-1], requires_grad=True)
    gammas = torch.rand(5, S, requires_grad=True)

    def reference(x, gammas):
        out = 0
        for s in range(S):
            gate, up = (x @ moe.shared_gate_up_proj[s].t()).chunk(2, dim=-1)
            out = (
                out
                + gammas[:, s : s + 1] * (torch.nn.functional.silu(gate) * up) @ moe.shared_down_proj[s].t()
            )
        return out

    got, want = moe._shared(x, gammas), reference(x, gammas)
    torch.testing.assert_close(got, want)
    g = torch.randn_like(want)
    params = (x, gammas, moe.shared_gate_up_proj, moe.shared_down_proj)
    for a, b in zip(torch.autograd.grad(got, params, g), torch.autograd.grad(want, params, g)):
        torch.testing.assert_close(a, b)


def test_count_ids_matches_bincount():
    """The sync-free expert token count equals torch.bincount (ids in range, unused ids 0)."""
    from primus.backends.megatron.core.transformer.inkling.inkling_moe import _count_ids

    ids = torch.tensor([[3, 0, 3], [5, 3, 0]])
    got = _count_ids(ids, 7)
    assert got.dtype == torch.int64
    assert got.tolist() == torch.bincount(ids.reshape(-1), minlength=7).tolist()
