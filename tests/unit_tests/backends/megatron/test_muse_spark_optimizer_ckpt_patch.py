###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The Muse Spark optimizer-checkpoint patches (muse_spark_optimizer_ckpt_patches), on CPU.

* fp32 params in a bf16 model: the optimizer state ids follow the optimizer's real order
  (upstream numbers them from ``float16_groups`` only and maps a state to the wrong param);
* a non-distributed optimizer checkpoint saved with another number of chained optimizers
  (expert parallel on / off) loads: single -> chain, chain -> single, else upstream.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")
from megatron.core.dist_checkpointing import ShardedTensor  # noqa: E402
from megatron.core.optimizer import optimizer as opt_module  # noqa: E402

from primus.backends.megatron.patches import (  # noqa: E402
    muse_spark_optimizer_ckpt_patches as patch,
)

FP16_CLS = opt_module.Float16OptimizerWithFloat16Params
CHAINED = opt_module.ChainedOptimizer


def _fp16_optimizer_with_an_fp32_param(step=True):
    """One group, the fp32 param first: optimizer order [gate (fp32), main copy of w].
    ``step=False``: no optimizer step, so the Adam state is empty (as before a load)."""
    w = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))  # bf16 model param
    gate = torch.nn.Parameter(torch.ones(2))  # fp32 model param (no main copy)
    main_w = torch.nn.Parameter(torch.ones(4))  # fp32 main copy of w
    inner = torch.optim.Adam([gate, main_w], lr=0.1)
    for p in (gate, main_w):
        p.grad = torch.ones_like(p)
    if step:
        inner.step()
    opt = SimpleNamespace(
        optimizer=inner,
        float16_groups=[[w]],
        fp32_from_float16_groups=[[main_w]],
        fp32_from_fp32_groups=[[gate]],
        config=None,
        init_state_fn=lambda *a: None,
        _extract_common_per_param_step=FP16_CLS._extract_common_per_param_step,
    )
    opt.state_dict = lambda: {
        "optimizer": inner.state_dict(),
        "fp32_from_fp16_params": opt.fp32_from_float16_groups,
    }
    model_sd = {
        "w": ShardedTensor.from_rank_offsets("w", w),
        "gate": ShardedTensor.from_rank_offsets("gate", gate),
    }
    return opt, w, gate, model_sd


def _two_group_fp16_optimizer():
    """Two groups (as Megatron's weight-decay / no-weight-decay split), each with an fp32
    param and an fp32 main copy of a bf16 param, in different orders, so a group-local
    position differs from torch's global state id:
    group 0 = [main of w0 (id 0), gate0 (id 1)], group 1 = [gate1 (id 2), main of w1 (id 3)]."""
    w0 = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    w1 = torch.nn.Parameter(torch.ones(5, dtype=torch.bfloat16))
    gate0 = torch.nn.Parameter(torch.ones(2))
    gate1 = torch.nn.Parameter(torch.ones(7))
    main0, main1 = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(5))
    inner = torch.optim.Adam(
        [{"params": [main0, gate0], "weight_decay": 0.1}, {"params": [gate1, main1], "weight_decay": 0.0}],
        lr=0.1,
    )
    for p in (main0, gate0, gate1, main1):
        p.grad = torch.ones_like(p)
    inner.step()
    opt = SimpleNamespace(
        optimizer=inner,
        float16_groups=[[w0], [w1]],
        fp32_from_float16_groups=[[main0], [main1]],
        fp32_from_fp32_groups=[[gate0], [gate1]],
        config=None,
        init_state_fn=lambda *a: None,
        _extract_common_per_param_step=FP16_CLS._extract_common_per_param_step,
    )
    opt.state_dict = lambda: {
        "optimizer": inner.state_dict(),
        "fp32_from_fp16_params": opt.fp32_from_float16_groups,
    }
    named = (("w0", w0), ("w1", w1), ("gate0", gate0), ("gate1", gate1))
    model_sd = {n: ShardedTensor.from_rank_offsets(n, p) for n, p in named}
    return opt, model_sd


def test_two_groups_map_state_ids_and_main_copies_per_group():
    # group-local positions (fp16_positions) must be translated to global ids through each
    # group's own params: main of w1 is at position 1 of group 1 but has global id 3
    opt, model_sd = _two_group_fp16_optimizer()
    params, fp16_positions = patch.optimizer_order_model_params(opt)
    assert fp16_positions == [[0], [1]]
    sd = patch.fp16_optimizer_sharded_state_dict(opt, model_sd)
    state = sd["optimizer"]["state"]
    for sid, name, shape in ((0, "w0", (3,)), (1, "gate0", (2,)), (2, "gate1", (7,)), (3, "w1", (5,))):
        assert state[sid]["exp_avg"].key == f"optimizer.state.exp_avg.{name}"
        assert state[sid]["exp_avg"].local_shape == shape
    (m0,), (m1,) = sd["fp32_from_fp16_params"]
    assert m0.key == "optimizer.state.fp32_param.w0" and m0.local_shape == (3,)
    assert m1.key == "optimizer.state.fp32_param.w1" and m1.local_shape == (5,)


def test_optimizer_order_maps_main_copies_back_to_model_params():
    opt, w, gate, _ = _fp16_optimizer_with_an_fp32_param()
    params, fp16_positions = patch.optimizer_order_model_params(opt)
    assert params[0] is gate and params[1] is w
    assert fp16_positions == [[1]]


def test_fp16_sharded_state_dict_ids_follow_the_optimizer_order():
    opt, _, _, model_sd = _fp16_optimizer_with_an_fp32_param()
    sd = patch.fp16_optimizer_sharded_state_dict(opt, model_sd)
    state = sd["optimizer"]["state"]
    # id 0 is the fp32 gate, id 1 the bf16 w (upstream would map id 0 to w: shape error)
    assert state[0]["exp_avg"].key == "optimizer.state.exp_avg.gate"
    assert state[0]["exp_avg"].local_shape == (2,)
    assert state[1]["exp_avg_sq"].key == "optimizer.state.exp_avg_sq.w"
    assert state[1]["exp_avg_sq"].local_shape == (4,)
    assert "step" not in state[0] and float(state["common_step"]) == 1.0
    (main,) = sd["fp32_from_fp16_params"][0]
    assert main.key == "optimizer.state.fp32_param.w" and main.dtype == torch.float32


def test_fp16_sharded_state_dict_for_loading_inits_the_state_first():
    # the is_loading branch fills the empty optimizer state before the mapping: the state is
    # empty (no step), and only the init (an Adam step, as Megatron's init_state_fn does)
    # creates the exp_avg entries the mapping reads
    opt, _, _, model_sd = _fp16_optimizer_with_an_fp32_param(step=False)
    assert not opt.optimizer.state
    calls = []
    opt.config = "cfg"

    def init_state(inner, config):
        calls.append((inner, config))
        inner.step()

    opt.init_state_fn = init_state
    sd = patch.fp16_optimizer_sharded_state_dict(opt, model_sd, is_loading=True)
    assert calls == [(opt.optimizer, "cfg")]
    state = sd["optimizer"]["state"]
    assert state[0]["exp_avg"].key == "optimizer.state.exp_avg.gate"
    assert state[1]["exp_avg"].key == "optimizer.state.exp_avg.w"
    calls.clear()
    patch.fp16_optimizer_sharded_state_dict(opt, model_sd)  # not loading: no init
    assert calls == []
    # without the init first, the empty state has nothing to map
    opt2, _, _, model_sd2 = _fp16_optimizer_with_an_fp32_param(step=False)
    sd2 = patch.fp16_optimizer_sharded_state_dict(opt2, model_sd2)
    assert 0 not in sd2["optimizer"]["state"]


def test_fp16_patch_takes_upstream_without_fp32_params(monkeypatch):
    seen = []

    def upstream(self, model_sd, is_loading=False, metadata=None):
        seen.append(self)
        return "upstream"

    monkeypatch.setattr(FP16_CLS, "sharded_state_dict", upstream)
    patch.patch_muse_spark_fp16_optimizer_ckpt(SimpleNamespace())
    patched = FP16_CLS.sharded_state_dict
    assert patched is not upstream and patched._primus_muse_spark
    plain = SimpleNamespace(fp32_from_fp32_groups=[[]])
    assert patched(plain, {}) == "upstream" and seen == [plain]
    stub = SimpleNamespace(is_stub_optimizer=True, fp32_from_fp32_groups=[[object()]])
    assert patched(stub, {}) == "upstream"
    # with an fp32 param the patched mapping runs
    opt, _, _, model_sd = _fp16_optimizer_with_an_fp32_param()
    assert patched(opt, model_sd)["optimizer"]["state"][0]["exp_avg"].key.endswith(".gate")
    patch.patch_muse_spark_fp16_optimizer_ckpt(SimpleNamespace())  # idempotent
    assert FP16_CLS.sharded_state_dict is patched


def _opt(n_params, **group):
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(n_params)]
    return SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"params": params, "lr": 0.1, **group}]))


def test_relayout_single_to_chain():
    # saved by one optimizer (no EP): tensors under 0, 1 and the common part at the top
    dense, experts = _opt(2, wd_mult=1.0), _opt(3, wd_mult=0.0)
    saved = {
        0: {"optimizer": {"state": {0: {"exp_avg": 1}}}},
        1: {"optimizer": {"state": {0: {"exp_avg": 2}}}},
        "optimizer": {"state": {"common_step": 5}, "param_groups": [{"params": [0, 1, 2, 3, 4]}]},
    }
    parts = patch.relayout_chained_optimizer_state(
        SimpleNamespace(chained_optimizers=[dense, experts]), saved
    )
    assert len(parts) == 2
    assert parts[0]["optimizer"]["param_groups"] == [{"lr": 0.1, "wd_mult": 1.0, "params": [0, 1]}]
    assert parts[1]["optimizer"]["param_groups"] == [{"lr": 0.1, "wd_mult": 0.0, "params": [0, 1, 2]}]
    assert parts[0]["optimizer"]["state"] == {0: {"exp_avg": 1}, "common_step": 5}
    assert parts[1]["optimizer"]["state"] == {0: {"exp_avg": 2}, "common_step": 5}
    assert "common_step" not in saved[0]["optimizer"]["state"]  # the input is not modified


def test_relayout_chain_to_single():
    # saved by a chain (EP): common parts under 0, 1; the tensors at the top level
    only = _opt(5)
    saved = {
        0: {"optimizer": {"state": {"common_step": 7}, "param_groups": [{"params": [0, 1]}]}},
        1: {"optimizer": {"state": {"common_step": 7}, "param_groups": [{"params": [0, 1, 2]}]}},
        # the part built for loading carries its own (stale) step: the saved one must win
        "optimizer": {"state": {3: {"exp_avg": 4}, "common_step": 0}},
    }
    (part,) = patch.relayout_chained_optimizer_state(SimpleNamespace(chained_optimizers=[only]), saved)
    assert part["optimizer"]["param_groups"] == [{"lr": 0.1, "params": [0, 1, 2, 3, 4]}]
    assert part["optimizer"]["state"] == {3: {"exp_avg": 4}, "common_step": 7}
    assert 0 not in part and 1 not in part


def test_the_saved_step_overwrites_a_step_built_for_loading():
    part = {"optimizer": {"state": {0: {"exp_avg": 1}, "common_step": 0}}}
    done = patch._complete_optimizer_state(part, _opt(1), 9)
    assert done["optimizer"]["state"]["common_step"] == 9
    assert part["optimizer"]["state"]["common_step"] == 0  # the input is not modified
    # no saved step: the loading side's own step stays
    assert patch._complete_optimizer_state(part, _opt(1), None)["optimizer"]["state"]["common_step"] == 0


def test_relayout_keeps_the_fp16_grad_scaler():
    # the fp16 dynamic loss scale must survive a load in the other layout
    scaler = {"scale": 1024.0, "growth_tracker": 3, "hysteresis_tracker": 2}
    dense, experts = _opt(2), _opt(3)
    saved = {
        0: {"optimizer": {"state": {}}},
        1: {"optimizer": {"state": {}}},
        "optimizer": {"state": {"common_step": 5}, "param_groups": [{"params": [0, 1, 2, 3, 4]}]},
        "grad_scaler": scaler,
    }
    parts = patch.relayout_chained_optimizer_state(
        SimpleNamespace(chained_optimizers=[dense, experts]), saved
    )
    assert len(parts) == 2 and all(p["grad_scaler"] == scaler for p in parts)
    chain = {
        0: {"optimizer": {"state": {}, "param_groups": [{"params": [0, 1]}]}, "grad_scaler": scaler},
        1: {"optimizer": {"state": {}, "param_groups": [{"params": [0, 1, 2]}]}, "grad_scaler": scaler},
        "optimizer": {"state": {}},
    }
    (part,) = patch.relayout_chained_optimizer_state(SimpleNamespace(chained_optimizers=[_opt(5)]), chain)
    assert part["grad_scaler"] == scaler
    # no saved scaler: none added
    del saved["grad_scaler"]
    parts = patch.relayout_chained_optimizer_state(
        SimpleNamespace(chained_optimizers=[dense, experts]), saved
    )
    assert len(parts) == 2 and all("grad_scaler" not in p for p in parts)


def test_relayout_same_layout_is_upstream():
    one, two = SimpleNamespace(chained_optimizers=[_opt(1)]), SimpleNamespace(
        chained_optimizers=[_opt(1), _opt(1)]
    )
    assert (
        patch.relayout_chained_optimizer_state(one, {"optimizer": {"state": {}, "param_groups": []}}) is None
    )
    assert patch.relayout_chained_optimizer_state(two, [{}, {}]) is None  # a chain's own list
    assert patch.relayout_chained_optimizer_state(two, {0: {}, 1: {}}) is None  # no top-level part
    assert patch.relayout_chained_optimizer_state(one, None) is None


def test_a_single_optimizers_own_checkpoint_with_int_keys_is_upstream():
    # int keys alone do not make a chain's checkpoint: a top-level optimizer.param_groups
    # means one optimizer saved it, and upstream loads it unchanged
    one = SimpleNamespace(chained_optimizers=[_opt(2)])
    saved = {
        0: {"extra": 1},
        "optimizer": {"state": {0: {"exp_avg": 1}}, "param_groups": [{"lr": 0.1, "params": [0, 1]}]},
    }
    assert patch.relayout_chained_optimizer_state(one, saved) is None
    # the same dict without the top-level param_groups is a chain's and is relaid out
    del saved["optimizer"]["param_groups"]
    saved[0] = {"optimizer": {"state": {}, "param_groups": [{"params": [0, 1]}]}}
    (part,) = patch.relayout_chained_optimizer_state(one, saved)
    # the optimizer's own param groups; the saved tensors kept; the int-keyed parts dropped
    assert part["optimizer"]["param_groups"] == [{"lr": 0.1, "params": [0, 1]}]
    assert part["optimizer"]["state"] == {0: {"exp_avg": 1}}
    assert set(part) == {"optimizer"}


def test_chained_load_patch_relayouts_or_calls_upstream(monkeypatch):
    calls = []
    monkeypatch.setattr(CHAINED, "load_state_dict", lambda self, sd: calls.append(("upstream", sd)))
    patch.patch_muse_spark_chained_optimizer_load(SimpleNamespace())
    load = CHAINED.load_state_dict
    assert load._primus_muse_spark

    def member(n):
        o = _opt(n)
        o.load_state_dict = lambda part, o=o: calls.append(
            ("member", part["optimizer"]["param_groups"], part["optimizer"]["state"])
        )
        return o

    chain = SimpleNamespace(
        chained_optimizers=[member(1), member(2)], _synchronize_steps=lambda: calls.append(("sync",))
    )
    load(chain, [{}, {}])
    assert calls == [("upstream", [{}, {}])]
    calls.clear()
    load(chain, {0: {}, 1: {}, "optimizer": {"state": {"common_step": 1}}})
    # each member gets its own param groups (in order, not swapped / shared) and the saved step
    assert calls == [
        ("member", patch.own_param_groups(chain.chained_optimizers[0]), {"common_step": 1}),
        ("member", patch.own_param_groups(chain.chained_optimizers[1]), {"common_step": 1}),
        ("sync",),
    ]
    assert calls[0][1] != calls[1][1]
    # one optimizer (EP off) loading an EP-saved checkpoint: one member call, no step sync
    calls.clear()
    single = SimpleNamespace(
        chained_optimizers=[member(3)], _synchronize_steps=lambda: calls.append(("sync",))
    )
    load(single, {0: {"optimizer": {"state": {"common_step": 2}}}, "optimizer": {"state": {}}})
    assert calls == [("member", patch.own_param_groups(single.chained_optimizers[0]), {"common_step": 2})]


def test_the_copy_calls_every_helper_upstream_calls():
    # the copy replaces upstream's method for models with fp32 params: every helper upstream's
    # sharded_state_dict calls must be called by the copy too, so a new upstream step (like
    # the GTP_remat backfill) cannot be dropped silently
    import ast
    import inspect
    import textwrap

    def called_names(fn):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None))
        return names - {None}

    method = FP16_CLS.sharded_state_dict
    if getattr(method, "_primus_muse_spark", False):
        pytest.skip("the class method is already patched in this process")
    upstream = called_names(method)
    ours = called_names(patch.fp16_optimizer_sharded_state_dict)
    if "backfill" in ours:  # called through getattr (absent in older Megatron)
        ours.add("_backfill_gtp_sharded_param_map")
    # helpers upstream calls that the copy does not need: the id map over float16 groups only
    replaced = {"chain", "from_iterable"}
    assert upstream - replaced <= ours, sorted(upstream - replaced - ours)


def test_the_copy_runs_the_gtp_backfill_with_the_optimizer_order(monkeypatch):
    seen = []
    monkeypatch.setattr(
        opt_module,
        "_backfill_gtp_sharded_param_map",
        lambda m, groups, sd: seen.append(groups),
        raising=False,
    )
    opt, _, _, model_sd = _fp16_optimizer_with_an_fp32_param()
    patch.fp16_optimizer_sharded_state_dict(opt, model_sd)
    params, _ = patch.optimizer_order_model_params(opt)
    assert seen == [[params]]
