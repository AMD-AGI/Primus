###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests that the Primus parallelization strategies forward ``reshard_after_forward``.

WHY THIS TEST EXISTS:
  This is "Drop 2". Both strategies called ``apply_fsdp2_sharding_recursively`` with
  seven positional arguments and stopped one short of the eighth, which is
  ``reshard_after_forward``. The value arrived from upstream as a keyword, landed in
  ``**kwargs``, and was never read. Nothing failed; the setting was simply ignored and
  every transformer block reshared after forward anyway.

  Fixing the hook alone (Drop 1) changes nothing without this half, and vice versa,
  which is exactly why a test per half is worth having. This is the test that would have
  caught the original bug: it asserts the argument appears in the downstream call.

  The strategies are defined inside ``install()`` and close over the Automodel
  parallelizer module, so the module is stubbed and ``install()`` is run against it. No
  GPU and no real Automodel needed.
"""

import sys
import types

import pytest

CASES = [
    pytest.param(
        "primus.backends.nemo_automodel.models.ideogram4.parallelize",
        "PRIMUS_IDEOGRAM_REAL_AC",
        "Ideogram4Transformer2DModel",
        "layers",
        id="ideogram4",
    ),
    pytest.param(
        "primus.backends.nemo_automodel.models.flux.parallelize",
        "PRIMUS_FLUX_REAL_AC",
        "FluxTransformer2DModel",
        "transformer_blocks",
        id="flux",
    ),
]


class FakeMesh:
    mesh_dim_names = ()

    def __getitem__(self, _name):
        raise KeyError(_name)

    def size(self):
        return 8


def _install_stub_parallelizer(monkeypatch):
    """Stub ``nemo_automodel.components.distributed.parallelizer``.

    Only the handful of names the strategies touch at registration and parallelize time
    are provided; the sharding calls are recorded instead of performed.
    """
    calls = {"sharding": [], "root": []}
    path = "nemo_automodel.components.distributed.parallelizer"
    module = types.ModuleType(path)

    class ParallelizationStrategy:
        pass

    module.ParallelizationStrategy = ParallelizationStrategy
    module.PARALLELIZATION_STRATEGIES = {}

    def register_parallel_strategy(name):
        def _decorator(cls):
            module.PARALLELIZATION_STRATEGIES[name] = cls
            return cls

        return _decorator

    module.register_parallel_strategy = register_parallel_strategy
    module.get_fsdp_dp_mesh = lambda mesh, *_a, **_kw: mesh

    def apply_fsdp2_sharding_recursively(*args, **kwargs):
        calls["sharding"].append((args, kwargs))

    def fully_shard(model, **kwargs):
        calls["root"].append(kwargs)
        return model

    module.apply_fsdp2_sharding_recursively = apply_fsdp2_sharding_recursively
    module.fully_shard = fully_shard

    class MixedPrecisionPolicy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.MixedPrecisionPolicy = MixedPrecisionPolicy
    module.is_selective_activation_checkpointing = lambda value: value == "selective"
    module.apply_selective_checkpointing_to_layers = lambda *a, **kw: None
    module.checkpoint_wrapper = lambda block, **kw: block
    module.CheckpointImpl = types.SimpleNamespace(NO_REENTRANT="no_reentrant")

    for name in (
        "nemo_automodel",
        "nemo_automodel.components",
        "nemo_automodel.components.distributed",
        path,
    ):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, path, module)
    return calls


def _strategy(monkeypatch, module_path, env_var, model_name):
    import importlib

    monkeypatch.setenv(env_var, "1")
    calls = _install_stub_parallelizer(monkeypatch)
    module = importlib.import_module(module_path)
    stub = sys.modules["nemo_automodel.components.distributed.parallelizer"]
    monkeypatch.setattr(module, "P", stub, raising=False)

    assert module.install() is True
    return stub.PARALLELIZATION_STRATEGIES[model_name](), calls


@pytest.mark.parametrize("module_path,env_var,model_name,block_attr", CASES)
@pytest.mark.parametrize("value", [False, True, None])
def test_reshard_after_forward_reaches_the_sharding_helper(
    monkeypatch, module_path, env_var, model_name, block_attr, value
):
    """The assertion the original code was missing."""
    strategy, calls = _strategy(monkeypatch, module_path, env_var, model_name)
    model = types.SimpleNamespace(**{block_attr: []})

    strategy.parallelize(model=model, device_mesh=FakeMesh(), reshard_after_forward=value)

    assert calls["sharding"], "sharding helper was never called"
    _args, kwargs = calls["sharding"][-1]
    assert "reshard_after_forward" in kwargs, "value dropped before the sharding helper"
    assert kwargs["reshard_after_forward"] is value


@pytest.mark.parametrize("module_path,env_var,model_name,block_attr", CASES)
def test_prefetch_knobs_are_passed_by_keyword_too(
    monkeypatch, module_path, env_var, model_name, block_attr
):
    """Positional passing is what caused the bug; assert the call is keyword-based."""
    strategy, calls = _strategy(monkeypatch, module_path, env_var, model_name)
    model = types.SimpleNamespace(**{block_attr: []})

    strategy.parallelize(model=model, device_mesh=FakeMesh(), reshard_after_forward=False)

    args, kwargs = calls["sharding"][-1]
    # model, mesh, mp_policy, offload_policy stay positional; the rest must be keywords.
    assert len(args) == 4, f"expected 4 positional args, got {len(args)}"
    for name in (
        "enable_fsdp2_prefetch",
        "fsdp2_backward_prefetch_depth",
        "fsdp2_forward_prefetch_depth",
    ):
        assert name in kwargs


@pytest.mark.parametrize("module_path,env_var,model_name,block_attr", CASES)
def test_root_unit_still_keeps_its_params_unsharded(
    monkeypatch, module_path, env_var, model_name, block_attr
):
    """The root fully_shard keeps reshard_after_forward=False regardless of the override.

    The root unit holds embeddings, norms and heads - a small fraction of parameters -
    and was already correct before this fix. Changing it is not part of the fix, so
    pin it so a future edit does not quietly couple the two.
    """
    strategy, calls = _strategy(monkeypatch, module_path, env_var, model_name)
    model = types.SimpleNamespace(**{block_attr: []})

    strategy.parallelize(model=model, device_mesh=FakeMesh(), reshard_after_forward=True)

    assert calls["root"][-1]["reshard_after_forward"] is False
