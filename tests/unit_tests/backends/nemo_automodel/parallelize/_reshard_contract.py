###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared contract for "a parallelization strategy must forward reshard_after_forward".

WHY THIS IS SHARED:
  Every Primus parallelization strategy has the same obligation and the same way
  of failing it. ``apply_fsdp2_sharding_recursively`` takes the flag as its eighth
  parameter, and a strategy that passes seven positional arguments and stops
  drops the value: it arrives from upstream as a keyword, lands in ``**kwargs``,
  and is never read. Nothing raises. The config echo still shows the requested
  value, because the value *was* parsed -- it just was never applied.

  That is a per-model bug with one shape, so it gets one test body and one
  assertion per model rather than a copy each time. A model added later inherits
  the coverage by calling these; forgetting to is visible as an absent test file.

WHAT IS NOT SHARED:
  How each strategy gets installed. Some register through
  ``register_parallel_strategy``; others subclass an in-tree strategy and
  overwrite the registry entry. That is genuinely per-model, so each test file
  builds its own strategy and passes the result in here.

No GPU, no real AutoModel: the parallelizer module is stubbed and the sharding
calls are recorded rather than performed.
"""

import sys
import types

PARALLELIZER_PATH = "nemo_automodel.components.distributed.parallelizer"


class FakeMesh:
    """A device mesh that reports a size and refuses to be indexed.

    Strategies ask for named submeshes; raising KeyError makes them take their
    "no such mesh dimension" path, which is what a plain FSDP2 run looks like.
    """

    mesh_dim_names = ()

    def __getitem__(self, name):
        raise KeyError(name)

    def size(self):
        return 8


def install_stub_parallelizer(monkeypatch):
    """Stub the AutoModel parallelizer module and record what it is asked to do.

    Returns ``(module, calls)`` where ``calls["sharding"]`` collects
    ``(args, kwargs)`` per ``apply_fsdp2_sharding_recursively`` call and
    ``calls["root"]`` collects the kwargs of each root ``fully_shard``.

    Only the names strategies actually touch are provided. A strategy that starts
    using something new will fail with AttributeError here, which is the intended
    signal: this stub is a statement about the surface strategies depend on.
    """
    calls = {"sharding": [], "root": []}
    module = types.ModuleType(PARALLELIZER_PATH)

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

    _ensure_parent_packages(monkeypatch, PARALLELIZER_PATH)
    monkeypatch.setitem(sys.modules, PARALLELIZER_PATH, module)
    return module, calls


def _ensure_parent_packages(monkeypatch, dotted_path):
    """Create placeholder parent packages so the leaf can be imported."""
    parts = dotted_path.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))


def install_stub_module(monkeypatch, dotted_path, **attributes):
    """Install a stub module with the given attributes, creating parents as needed."""
    _ensure_parent_packages(monkeypatch, dotted_path)
    module = types.ModuleType(dotted_path)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, dotted_path, module)
    return module


def model_with_blocks(block_attr, blocks=()):
    """A stand-in model exposing the block list a strategy expects to checkpoint."""
    return types.SimpleNamespace(**{block_attr: list(blocks)})


# --- the contract itself -----------------------------------------------------


def assert_forwards_reshard(strategy, calls, block_attr, value):
    """The assertion the original code was missing: the flag reaches the helper.

    Call this for each of False, True and None. None matters as much as the
    others -- it is the "user said nothing, keep the heuristic" case, and a
    strategy that coerces it to a bool has changed the default for everyone.
    """
    del calls["sharding"][:]
    strategy.parallelize(
        model=model_with_blocks(block_attr),
        device_mesh=FakeMesh(),
        reshard_after_forward=value,
    )

    assert calls["sharding"], "the sharding helper was never called"
    _args, kwargs = calls["sharding"][-1]
    assert "reshard_after_forward" in kwargs, "value was dropped before the sharding helper"
    assert (
        kwargs["reshard_after_forward"] is value
    ), f"expected reshard_after_forward={value!r}, got {kwargs['reshard_after_forward']!r}"


def assert_root_unit_stays_unsharded(strategy, calls, block_attr):
    """The root ``fully_shard`` keeps ``reshard_after_forward=False`` regardless.

    The root unit holds embeddings, norms and heads -- a small fraction of the
    parameters -- and was already correct before any of this. Changing it is not
    part of the fix, so pin it, or a later edit will quietly couple the two.
    """
    del calls["root"][:]
    strategy.parallelize(
        model=model_with_blocks(block_attr),
        device_mesh=FakeMesh(),
        reshard_after_forward=True,
    )

    assert calls["root"], "the root fully_shard was never called"
    assert calls["root"][-1]["reshard_after_forward"] is False


def assert_sharding_args_are_keyword(strategy, calls, block_attr):
    """Only the first four arguments may be positional.

    Positional passing is the mechanism of the original bug: the eighth parameter
    was unreachable because the call site had run out of positions. Strategies
    Primus writes should pass by keyword so that adding a parameter upstream
    cannot silently shift meaning.

    Not part of the contract for strategies that delegate to an in-tree parent --
    that call site is not ours to change.
    """
    del calls["sharding"][:]
    strategy.parallelize(
        model=model_with_blocks(block_attr),
        device_mesh=FakeMesh(),
        reshard_after_forward=False,
    )

    args, kwargs = calls["sharding"][-1]
    assert len(args) == 4, f"expected 4 positional args (model, mesh, mp, offload), got {len(args)}"
    for name in (
        "enable_fsdp2_prefetch",
        "fsdp2_backward_prefetch_depth",
        "fsdp2_forward_prefetch_depth",
    ):
        assert name in kwargs, f"{name} should be passed by keyword"
