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


def _fp16_optimizer_with_an_fp32_param():
    """One group, the fp32 param first: optimizer order [gate (fp32), main copy of w]."""
    w = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))  # bf16 model param
    gate = torch.nn.Parameter(torch.ones(2))  # fp32 model param (no main copy)
    main_w = torch.nn.Parameter(torch.ones(4))  # fp32 main copy of w
    inner = torch.optim.Adam([gate, main_w], lr=0.1)
    for p in (gate, main_w):
        p.grad = torch.ones_like(p)
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
        "optimizer": {"state": {3: {"exp_avg": 4}}},
    }
    (part,) = patch.relayout_chained_optimizer_state(SimpleNamespace(chained_optimizers=[only]), saved)
    assert part["optimizer"]["param_groups"] == [{"lr": 0.1, "params": [0, 1, 2, 3, 4]}]
    assert part["optimizer"]["state"] == {3: {"exp_avg": 4}, "common_step": 7}
    assert 0 not in part and 1 not in part


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
