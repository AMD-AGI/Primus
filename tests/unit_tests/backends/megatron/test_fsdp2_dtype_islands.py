###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``_shard_dtype_islands``: opt-in fp32 islands as their own FSDP2 groups (fully_shard mocked, CPU)."""

import pytest
import torch

if not torch.distributed.is_available():
    pytest.skip("torch.distributed is not available in this build", allow_module_level=True)
pytest.importorskip("megatron.core")  # torch_fully_sharded_data_parallel imports Megatron

from torch import nn  # noqa: E402
from torch.distributed.fsdp import MixedPrecisionPolicy  # noqa: E402

from primus.backends.megatron.core.distributed import (  # noqa: E402
    torch_fully_sharded_data_parallel as fsdp_mod,
)

BF16_POLICY = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.bfloat16)
FP32_PARAM_BF16_POLICY = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)


@pytest.fixture
def sharded(monkeypatch):
    """Records (module, kwargs); a module counts as FSDP-managed once it was sharded."""
    calls = []
    monkeypatch.setattr(fsdp_mod, "fully_shard", lambda m, **kw: calls.append((m, kw)))
    monkeypatch.setattr(fsdp_mod, "_is_fsdp_module", lambda m: any(m is c for c, _ in calls))
    return calls


class _Norm(nn.Module):  # bf16, like MuseSparkInputScaledRMSNorm
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, dtype=dtype))


class _Weightless(nn.Module):  # like the attention post-norm: no params
    def forward(self, x):
        return x


class _Router(nn.Linear):  # like MuseSparkTopKRouter
    fsdp2_dtype_island = True


class _Gate(nn.Module):  # like MuseSparkGatedResidual: an fp32 gate + a norm child
    fsdp2_dtype_island = True

    def __init__(self, norm):
        super().__init__()
        self.post_norm = norm
        self.gate = nn.Parameter(torch.zeros(4))


class _Layer(nn.Module):
    def __init__(self, model_dtype=torch.bfloat16):
        super().__init__()
        self.proj = nn.Linear(4, 64).to(model_dtype)
        self.router = _Router(4, 4, bias=False)  # fp32
        self.attn_bda = _Gate(_Weightless())
        self.mlp_bda = _Gate(_Norm(model_dtype))


def test_no_opt_in_is_a_noop_for_any_dtype_mix(sharded):
    # other models: nothing checked, nothing split (fully_shard decides, as before)
    fp32_in_bf16 = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    mixed = nn.Sequential(nn.Linear(4, 4).bfloat16(), nn.Linear(4, 4))
    frozen = nn.Linear(4, 4).bfloat16()
    frozen.extra = nn.Parameter(torch.zeros(4), requires_grad=False)
    for m in (fp32_in_bf16, mixed, frozen):
        assert fsdp_mod._shard_dtype_islands(m, {"mp_policy": BF16_POLICY}) == []
    assert sharded == []


def test_groups_children_first_and_minimal(sharded):
    layer = _Layer()
    groups = fsdp_mod._shard_dtype_islands(layer, {"mp_policy": BF16_POLICY, "mesh": "M"})
    # the bf16 post_norm must leave mlp_bda's group (one dtype per group); nothing else is split
    assert set(groups) == {layer.router, layer.attn_bda, layer.mlp_bda.post_norm, layer.mlp_bda}
    order = [m for m, _ in sharded]
    assert order == groups and order.index(layer.mlp_bda.post_norm) < order.index(layer.mlp_bda)
    assert layer.proj not in groups and layer not in groups


@pytest.mark.parametrize(
    "model_dtype, policy",
    [
        (torch.bfloat16, BF16_POLICY),
        # params_dtype fp32 + bf16 compute: the islands must still not be cast / reduced in bf16
        (torch.float32, FP32_PARAM_BF16_POLICY),
    ],
)
def test_island_policy_keeps_fp32_for_every_model_policy(sharded, model_dtype, policy):
    layer = _Layer(model_dtype)
    fsdp_mod._shard_dtype_islands(layer, {"mp_policy": policy, "mesh": "M"})
    kw = {id(m): k for m, k in sharded}
    for fp32 in (layer.router, layer.attn_bda, layer.mlp_bda):
        pol = kw[id(fp32)]["mp_policy"]
        assert pol.reduce_dtype == torch.float32 and pol.param_dtype is None and kw[id(fp32)]["mesh"] == "M"
    # the split-off child keeps the model's own policy
    assert kw[id(layer.mlp_bda.post_norm)]["mp_policy"] is policy


def test_already_managed_islands_are_skipped(sharded):
    root = nn.Module()
    root.layer = _Layer()
    root.head = nn.Linear(4, 64).bfloat16()
    first = fsdp_mod._shard_dtype_islands(root.layer, {})
    fsdp_mod.fully_shard(root.layer)  # the caller shards the layer itself
    n = len(sharded)
    assert fsdp_mod._shard_dtype_islands(root, {}) == [] and len(sharded) == n and len(first) == 4


def test_root_island_is_sharded_from_the_root(sharded):
    root = nn.Module()
    root.body = nn.Linear(4, 4).bfloat16()
    root.router = _Router(4, 4)  # not under a wrapped layer: found by the root pass
    assert fsdp_mod._shard_dtype_islands(root, {}) == [root.router]


def test_island_with_mixed_trainable_params_raises(sharded):
    root = nn.Module()
    root.router = _Router(4, 4)
    root.router.extra = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="several dtypes"):
        fsdp_mod._shard_dtype_islands(root, {})
    assert sharded == []


def test_island_frozen_param_of_other_dtype_is_ignored(sharded):
    root = nn.Module()
    root.router = _Router(4, 4)
    root.router.frozen = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16), requires_grad=False)
    assert fsdp_mod._shard_dtype_islands(root, {}) == [root.router]
    assert sharded[0][1]["mp_policy"].reduce_dtype == torch.float32


def test_wrapped_module_cannot_be_the_island(sharded):
    with pytest.raises(ValueError, match="module being wrapped"):
        fsdp_mod._shard_dtype_islands(_Router(4, 4), {})


def test_fully_frozen_island_mixed_dtypes_gets_no_group(sharded, monkeypatch):
    # partial finetune with a frozen router: no false "trainable ... several dtypes" error
    root = nn.Module()
    root.router = _Router(4, 4)
    root.router.requires_grad_(False)
    root.router.b = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16), requires_grad=False)
    warned = []
    monkeypatch.setattr(fsdp_mod, "warning_rank_0", warned.append)
    assert fsdp_mod._shard_dtype_islands(root, {}) == []
    assert sharded == []
    assert len(warned) == 1 and "mixes dtypes" in warned[0] and "router" in warned[0]
    assert "next enclosing group" in warned[0] and "left to the parent" not in warned[0]
    # the wrapper calls this per wrapped submodule and again on the root: warn once
    outer = nn.Module()
    outer.inner = root
    assert fsdp_mod._shard_dtype_islands(root, {}) == []
    assert fsdp_mod._shard_dtype_islands(outer, {}) == []
    assert len(warned) == 1


def test_island_without_direct_params_warns(sharded, monkeypatch):
    class _EmptyIsland(nn.Module):
        fsdp2_dtype_island = True

        def __init__(self):
            super().__init__()
            self.inner = nn.Linear(4, 4)

    root = nn.Module()
    root.island = _EmptyIsland()
    warned = []
    monkeypatch.setattr(fsdp_mod, "warning_rank_0", warned.append)
    assert fsdp_mod._shard_dtype_islands(root, {}) == []
    assert len(warned) == 1 and "owns no unsharded parameters" in warned[0]
    assert fsdp_mod._shard_dtype_islands(root, {}) == []
    assert len(warned) == 1  # once per module


def test_fully_frozen_island_one_dtype_keeps_its_dtype(sharded):
    root = nn.Module()
    root.router = _Router(4, 4)
    root.router.requires_grad_(False)
    assert fsdp_mod._shard_dtype_islands(root, {}) == [root.router]
    assert sharded[0][1]["mp_policy"].reduce_dtype == torch.float32


def test_island_attr_is_one_shared_constant():
    """No GPU needed: the FSDP2 wrapper and the leaf module use the same ISLAND_ATTR."""
    from primus.backends.megatron.core.distributed import fsdp2_dtype_islands as leaf

    assert fsdp_mod.ISLAND_ATTR is leaf.ISLAND_ATTR == "fsdp2_dtype_island"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="primus_turbo queries the GPU at import")
def test_fp8_all_gather_wrap_skips_dtype_islands(monkeypatch):
    """_wrap_fp8_weights_for_all_gather wraps the bf16 FP8 weights and leaves an opted-in
    fp32 island (router / residual gate) in its own dtype, without an FP8 wrapper."""
    pytest.importorskip("triton")
    pytest.importorskip("primus_turbo")
    from primus.backends.megatron.core.distributed import fsdp2_fp8_all_gather as fp8

    class _Wrapped(torch.Tensor):  # stands in for WeightWithFP8AllGatherTensor (no GPU kernels)
        @staticmethod
        def __new__(cls, data, cfg):
            return torch.Tensor._make_subclass(cls, data)

    monkeypatch.setattr(fp8, "WeightWithFP8AllGatherTensor", _Wrapped)

    class _Fp8Linear(nn.Linear):
        _fp8_config = object()

    class _IslandRouter(_Fp8Linear):
        fsdp2_dtype_island = True

    model = nn.Sequential(_Fp8Linear(4, 4).to(torch.bfloat16), _IslandRouter(4, 2))
    lin_w, router_w = model[0].weight, model[1].weight
    assert fp8._wrap_fp8_weights_for_all_gather(model, fp8_config=object()) == 1
    assert model[0].weight is not lin_w  # the bf16 FP8 weight was re-wrapped
    assert model[1].weight is router_w and router_w.dtype == torch.float32
    assert not isinstance(model[1].weight, _Wrapped)


# --- PrimusTorchFullyShardedDataParallel wiring (real constructor, distributed calls stubbed) ---


class _Block(nn.Module):  # a wrapped unit (stands in for TransformerLayer)
    def __init__(self):
        super().__init__()
        self.layer = _Layer()


class _Model(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(n))
        self.head_router = _Router(4, 4, bias=False)  # an island outside every wrapped unit


def _build_fsdp(monkeypatch, model, recompute=False, wrap=(_Block,)):
    from types import SimpleNamespace

    calls = []

    def fake_fully_shard(m, **kw):
        calls.append((m, kw))
        m.fwd_prefetch, m.bwd_prefetch = [], []
        m.set_modules_to_forward_prefetch = lambda targets, m=m: m.fwd_prefetch.extend(targets)
        m.set_modules_to_backward_prefetch = lambda targets, m=m: m.bwd_prefetch.extend(targets)

    monkeypatch.setattr(fsdp_mod, "fully_shard", fake_fully_shard)
    monkeypatch.setattr(fsdp_mod, "_is_fsdp_module", lambda m: any(m is c for c, _ in calls))
    monkeypatch.setattr(fsdp_mod.parallel_state, "get_data_parallel_group", lambda **kw: object())
    monkeypatch.setattr(fsdp_mod.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(fsdp_mod.DeviceMesh, "from_group", staticmethod(lambda *a, **kw: "mesh"))
    import sys
    import types

    training = types.ModuleType("megatron.training")  # the wrapper reads get_args() from here
    training.get_args = lambda: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    config = SimpleNamespace(
        params_dtype=torch.bfloat16,
        bf16=True,
        transformer_impl="transformer_engine",
        fsdp_prefetch_depth=1,
        recompute_granularity="full" if recompute else None,
    )
    ddp_config = SimpleNamespace(reshard_after_forward=True)
    monkeypatch.setattr(
        fsdp_mod._BaseDataParallel,
        "__init__",
        lambda self, config, module: (
            nn.Module.__init__(self),
            setattr(self, "config", config),
            setattr(self, "module", module),
        )[0],
    )
    fsdp = fsdp_mod.PrimusTorchFullyShardedDataParallel(
        config=config, ddp_config=ddp_config, module=model, sub_modules_to_wrap=set(wrap)
    )
    return fsdp, calls


def test_wrapper_groups_islands_before_their_block_and_never_prefetches_them(monkeypatch):
    model = _Model()
    _, calls = _build_fsdp(monkeypatch, model)
    order = [m for m, _ in calls]
    pos = {id(m): i for i, m in enumerate(order)}
    for block in model.blocks:
        layer = block.layer
        # the islands (and the norm under a gate) are grouped before their block
        for m in (layer.router, layer.attn_bda, layer.mlp_bda, layer.mlp_bda.post_norm):
            assert pos[id(m)] < pos[id(block)], type(m).__name__
        assert calls[pos[id(layer.router)]][1]["mp_policy"].param_dtype is None  # fp32 kept
        assert calls[pos[id(layer.router)]][1]["mp_policy"].reduce_dtype == torch.float32
        assert "mp_policy" not in calls[pos[id(block)]][1]  # the block keeps the model kwargs
    # the island outside every block is grouped from the root, before the root
    assert pos[id(model.head_router)] < pos[id(model)] == len(order) - 1
    # only the wrapped blocks prefetch each other; islands are in no prefetch list
    b0, b1 = model.blocks
    assert b0.fwd_prefetch == [b1] and b1.bwd_prefetch == [b0]
    islands = [m for m in model.modules() if getattr(m, fsdp_mod.ISLAND_ATTR, False)]
    for m in order:
        for t in getattr(m, "fwd_prefetch", []) + getattr(m, "bwd_prefetch", []):
            assert t not in islands


def test_wrapper_without_islands_shards_only_the_blocks_and_the_root(monkeypatch):
    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(4, 4)) for _ in range(2))

    model = Plain()
    _, calls = _build_fsdp(monkeypatch, model, wrap=(nn.Sequential,))
    assert [m for m, _ in calls] == [*model.blocks, model]


def test_wrapper_with_recompute_sets_only_backward_prefetch(monkeypatch):
    model = _Model()
    _build_fsdp(monkeypatch, model, recompute=True)
    b0, b1 = model.blocks
    assert b0.fwd_prefetch == [] and b1.bwd_prefetch == [b0]
