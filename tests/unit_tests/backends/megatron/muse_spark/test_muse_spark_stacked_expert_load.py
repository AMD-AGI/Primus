###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""moe._sync_stacked_expert_weights: the load pre-hook of the Turbo grouped experts.

The Turbo grouped linears train one stacked ``weights`` [E, out, in]; the per-expert
``weight{i}`` are frozen views that exist only after the first forward / state_dict(). The hook
makes both checkpoint forms load strictly: torch_dist (``weights``) and TE / legacy
(``weight{i}``), with and without the views.
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

pytest.importorskip("megatron.core")

from primus.backends.megatron.core.transformer.muse_spark.moe import (  # noqa: E402
    _sync_stacked_expert_weights,
)


def _import_megatron_training():
    """Import megatron.training; skip only on a CPU-only node where its import chain reads
    the GPU properties (fla -> mamba_ssm) and fails. Any other error fails the test."""
    if torch.cuda.is_available():
        return
    for _ in range(3):
        try:
            import megatron.training  # noqa: F401

            return
        except RuntimeError as e:
            if not _GPU_PROBE_ERROR.search(str(e)):
                raise
    pytest.skip("megatron.training does not import on this CPU-only node")


E = 3
# fc1 [E, 4, 2] and fc2 [E, 2, 5]: different shapes, so a swapped or skipped linear fails
SHAPES = {"linear_fc1": (4, 2), "linear_fc2": (2, 5)}
LINEARS = tuple(SHAPES)


class _Linear(torch.nn.Module):
    def __init__(self, views, shape):
        super().__init__()
        self.num_gemms = E
        self.weights = torch.nn.Parameter(torch.zeros(E, *shape))
        if views:
            for i in range(E):
                self.register_parameter(f"weight{i}", torch.nn.Parameter(torch.zeros(*shape)))


def _module(views):
    return SimpleNamespace(**{n: _Linear(views, SHAPES[n]) for n in LINEARS})


def _stack(name):
    out, inp = SHAPES[name]
    base = 0 if name == "linear_fc1" else 1000  # distinct values per linear
    return (base + torch.arange(E * out * inp, dtype=torch.float32)).view(E, out, inp)


P = "decoder.layers.0.mlp.experts."


def test_stacked_weights_fill_the_per_expert_views():
    # (a) torch_dist: weights in the state dict, the views exist -> weight{i} = weights[i]
    sd = {P + f"{n}.weights": _stack(n) for n in LINEARS}
    _sync_stacked_expert_weights(_module(views=True), sd, P)
    for n in LINEARS:
        stacked = _stack(n)
        for i in range(E):
            assert torch.equal(sd[P + f"{n}.weight{i}"], stacked[i])  # stack order kept
        assert torch.equal(sd[P + f"{n}.weights"], stacked)


def test_stacked_weights_drop_the_per_expert_keys_without_views():
    # (b) weights present, no views yet: stray weight{i} keys are removed (strict load)
    sd = {}
    for n in LINEARS:
        sd[P + f"{n}.weights"] = _stack(n)
        sd.update({P + f"{n}.weight{i}": _stack(n)[i] for i in range(E)})
    _sync_stacked_expert_weights(_module(views=False), sd, P)
    assert set(sd) == {P + f"{n}.weights" for n in LINEARS}


@pytest.mark.parametrize("views", [True, False])
def test_per_expert_weights_are_stacked(views):
    # (c) TE / legacy: only weight{i} -> weights = stack(weight{i}) in expert order; the
    # per-expert keys stay when the views exist, else they are removed
    sd = {P + f"{n}.weight{i}": _stack(n)[i].clone() for n in LINEARS for i in range(E)}
    _sync_stacked_expert_weights(_module(views=views), sd, P)
    for n in LINEARS:
        assert torch.equal(sd[P + f"{n}.weights"], _stack(n))
        assert all((P + f"{n}.weight{i}" in sd) is views for i in range(E))


def test_an_incomplete_per_expert_set_is_left_for_strict_loading_to_report():
    sd = {P + "linear_fc1.weight0": torch.zeros(4, 2), P + "linear_fc2.weight1": torch.zeros(2, 5)}
    _sync_stacked_expert_weights(_module(views=False), sd, P)
    assert set(sd) == {P + "linear_fc1.weight0", P + "linear_fc2.weight1"}


def test_a_module_without_stacked_linears_is_untouched():
    sd = {P + "linear_fc1.weight0": torch.zeros(2, 2), P + "linear_fc2.weight0": torch.zeros(2, 2)}
    plain = SimpleNamespace(linear_fc1=torch.nn.Linear(2, 2), linear_fc2=torch.nn.Linear(2, 2))
    _sync_stacked_expert_weights(plain, sd, P)
    assert list(sd) == [P + "linear_fc1.weight0", P + "linear_fc2.weight0"]


@pytest.mark.parametrize("form", ["stacked", "per_expert"])
@pytest.mark.parametrize("views", [True, False])
def test_the_hook_makes_load_state_dict_strict_for_both_forms(views, form):
    # every case fails strict loading without the hook (see test_every_case_needs_the_hook)
    # through nn.Module.load_state_dict, as the Turbo experts register it
    class _Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_fc1 = _Linear(views, SHAPES["linear_fc1"])
            self.linear_fc2 = _Linear(views, SHAPES["linear_fc2"])
            self._register_load_state_dict_pre_hook(_sync_stacked_expert_weights, with_module=True)

    if form == "stacked":
        sd = {f"{n}.weights": _stack(n).clone() for n in LINEARS}
        if not views:
            # a checkpoint saved with the views also holds weight{i}; a module without the
            # views has no such keys, so strict loading needs the hook to drop them (without
            # them this case would load with no hook at all)
            sd.update({f"{n}.weight{i}": _stack(n)[i].clone() for n in LINEARS for i in range(E)})
    else:
        sd = {f"{n}.weight{i}": _stack(n)[i].clone() for n in LINEARS for i in range(E)}
    m = _Experts()
    m.load_state_dict(sd, strict=True)
    for n in LINEARS:
        linear = getattr(m, n)
        assert torch.equal(linear.weights.detach(), _stack(n))
        if views:  # the per-expert views hold expert i's rows, in expert order
            for i in range(E):
                assert torch.equal(getattr(linear, f"weight{i}").detach(), _stack(n)[i])


@pytest.mark.parametrize("form", ["stacked", "per_expert"])
def test_the_turbo_experts_register_the_hook_themselves(monkeypatch, form):
    # the real MuseSparkTurboGroupedMLP: its __init__ must register the pre-hook (no
    # registration here). PrimusGroupedMLP.__init__ (TE grouped linears, a GPU) is replaced
    # by one that only sets the two stacked linears, with their per-expert views registered:
    # then both forms need the hook (stacked: it fills weight{i}; per-expert: weights).
    _import_megatron_training()  # experts.py imports megatron.training (CPU-node guard)
    # a plain import: a broken experts.py (in this repo) must fail this test, not skip it
    from primus.backends.megatron.core.transformer import experts
    from primus.backends.megatron.core.transformer.muse_spark import moe

    def fake_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.linear_fc1 = _Linear(True, SHAPES["linear_fc1"])
        self.linear_fc2 = _Linear(True, SHAPES["linear_fc2"])

    monkeypatch.setattr(experts.PrimusGroupedMLP, "__init__", fake_init)
    m = moe.muse_spark_turbo_grouped_mlp_cls()()
    if form == "stacked":
        sd = {f"{n}.weights": _stack(n).clone() for n in LINEARS}
    else:
        sd = {f"{n}.weight{i}": _stack(n)[i].clone() for n in LINEARS for i in range(E)}
    m.load_state_dict(sd, strict=True)  # strict: fails without the hook for either form
    for n in LINEARS:
        assert torch.equal(getattr(m, n).weights.detach(), _stack(n))
        for i in range(E):
            assert torch.equal(getattr(getattr(m, n), f"weight{i}").detach(), _stack(n)[i])


@pytest.mark.parametrize("form", ["stacked", "per_expert"])
@pytest.mark.parametrize("views", [True, False])
def test_every_case_needs_the_hook(views, form):
    # the same modules and state dicts with no hook: strict loading fails in each case, so
    # test_the_hook_makes_load_state_dict_strict_for_both_forms checks the hook every time
    class _NoHook(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_fc1 = _Linear(views, SHAPES["linear_fc1"])
            self.linear_fc2 = _Linear(views, SHAPES["linear_fc2"])

    if form == "stacked":
        sd = {f"{n}.weights": _stack(n).clone() for n in LINEARS}
        if not views:
            sd.update({f"{n}.weight{i}": _stack(n)[i].clone() for n in LINEARS for i in range(E)})
    else:
        sd = {f"{n}.weight{i}": _stack(n)[i].clone() for n in LINEARS for i in range(E)}
    with pytest.raises(RuntimeError):
        _NoHook().load_state_dict(sd, strict=True)
