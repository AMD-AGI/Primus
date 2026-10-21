###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the monkeypatch rebind helper.

``rebind_everywhere`` exists because ``module.attr = new`` misses every caller
that did ``from module import attr`` before the patch ran.
"""

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest

import primus.backends.megatron.patches._rebind as rebind_mod
from primus.backends.megatron.patches._rebind import rebind_everywhere


@pytest.fixture
def defining_module():
    """A throwaway module owning the symbol, registered in sys.modules."""
    module = ModuleType("_primus_rebind_owner")
    module.target = lambda: "original"
    sys.modules[module.__name__] = module
    yield module
    sys.modules.pop(module.__name__, None)


@pytest.fixture
def importer(defining_module):
    """A second module that copied the symbol by name, as gpt_builders does."""
    module = ModuleType("_primus_rebind_importer")
    module.target = defining_module.target
    sys.modules[module.__name__] = module
    yield module
    sys.modules.pop(module.__name__, None)


def test_rebinds_owner_and_importer(defining_module, importer):
    replacement = lambda: "patched"

    rebound = rebind_everywhere(defining_module, "target", replacement)

    assert defining_module.target is replacement
    assert importer.target is replacement
    assert rebound[0] == "_primus_rebind_owner"
    assert "_primus_rebind_importer" in rebound


def test_leaves_unrelated_same_named_attributes_alone(defining_module):
    """Only a module holding *this* object is rebound -- matching by name alone
    would clobber every unrelated `target` in the process."""
    bystander = ModuleType("_primus_rebind_bystander")
    bystander.target = lambda: "someone else's"
    sys.modules[bystander.__name__] = bystander
    original_bystander_target = bystander.target

    try:
        rebound = rebind_everywhere(defining_module, "target", lambda: "patched")
    finally:
        sys.modules.pop(bystander.__name__, None)

    assert bystander.target is original_bystander_target
    assert "_primus_rebind_bystander" not in rebound


def test_warns_when_a_holder_refuses_setattr(defining_module, importer, monkeypatch):
    warnings = []
    monkeypatch.setattr(rebind_mod, "warning_rank_0", warnings.append)

    class ReadOnly(ModuleType):
        def __setattr__(self, name, value):
            raise RuntimeError("read-only module")

    read_only = ReadOnly("_primus_rebind_read_only")
    read_only.__dict__["target"] = defining_module.target
    monkeypatch.setitem(sys.modules, read_only.__name__, read_only)

    rebound = rebind_everywhere(defining_module, "target", lambda: "patched")

    assert importer.target() == "patched"
    assert "_primus_rebind_read_only" not in rebound
    assert len(warnings) == 1 and "_primus_rebind_read_only.target" in warnings[0]


class _HookedEntry:
    """A non-module sys.modules entry whose every attribute read is a hook: a lazy
    object proxy resolves its target there, and cffi's Lib reports ModuleType as
    its __class__."""

    def __init__(self, target):
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "reads", [])

    def __getattribute__(self, name):
        object.__getattribute__(self, "reads").append(name)
        return ModuleType if name == "__class__" else object.__getattribute__(self, name)


def test_leaves_non_module_entries_untouched(defining_module, importer, monkeypatch):
    entry = _HookedEntry(defining_module.target)
    monkeypatch.setitem(sys.modules, "_primus_rebind_hooked", entry)

    rebound = rebind_everywhere(defining_module, "target", lambda: "patched")

    assert object.__getattribute__(entry, "reads") == []
    assert "_primus_rebind_hooked" not in rebound
    assert importer.target() == "patched"


class _ModuleProxy(ModuleType):
    def __getattr__(self, name):
        self.__dict__.setdefault("probed", []).append(name)
        raise AttributeError(name)


class _ObjectProxy:
    """Shaped like apex's JIT op proxies (amp_C, ...): a plain object in
    sys.modules that builds the extension in __getattr__."""

    def __getattr__(self, name):
        self.__dict__.setdefault("probed", []).append(name)
        raise AttributeError(name)


@pytest.mark.parametrize(
    "make_proxy", [lambda: _ModuleProxy("_primus_rebind_proxy"), _ObjectProxy], ids=["module", "object"]
)
def test_does_not_call_module_getattr(defining_module, importer, make_proxy):
    proxy = make_proxy()
    sys.modules["_primus_rebind_proxy"] = proxy

    try:
        rebind_everywhere(defining_module, "target", lambda: "patched")
    finally:
        sys.modules.pop("_primus_rebind_proxy", None)

    assert "probed" not in proxy.__dict__
    assert importer.target() == "patched"


def test_does_not_load_lazy_modules(defining_module, importer, tmp_path, monkeypatch):
    executed = tmp_path / "executed"
    (tmp_path / "_primus_rebind_lazy.py").write_text(f"open({str(executed)!r}, 'w').close()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = importlib.util.find_spec("_primus_rebind_lazy")
    spec.loader = importlib.util.LazyLoader(spec.loader)
    lazy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lazy)
    monkeypatch.setitem(sys.modules, "_primus_rebind_lazy", lazy)

    rebind_everywhere(defining_module, "target", lambda: "patched")

    assert importer.target() == "patched"
    assert not executed.exists()


def test_does_not_read_a_dict_property(defining_module, importer, monkeypatch):
    """apipkg's ApiModule defines __dict__ as a property that imports every lazy
    export when read."""
    reads = []

    class EagerDict(ModuleType):
        @property
        def __dict__(self):
            reads.append(True)
            return ModuleType.__dict__["__dict__"].__get__(self)

    monkeypatch.setitem(sys.modules, "_primus_rebind_eager_dict", EagerDict("_primus_rebind_eager_dict"))

    rebind_everywhere(defining_module, "target", lambda: "patched")

    assert reads == []
    assert importer.target() == "patched"


def test_none_placeholder_is_not_rebound_everywhere(defining_module):
    """`except ImportError: target = None` placeholders are all the same None,
    so identity says nothing about who imported this one."""
    defining_module.target = None
    bystander = ModuleType("_primus_rebind_bystander")
    sys.modules[bystander.__name__] = bystander
    replacement = lambda: "patched"

    try:
        rebound = rebind_everywhere(defining_module, "target", replacement)
    finally:
        sys.modules.pop(bystander.__name__, None)

    assert defining_module.target is replacement
    assert rebound == ["_primus_rebind_owner"]
    assert not hasattr(bystander, "target")


def test_noop_when_already_pointing_at_the_replacement(defining_module, importer):
    replacement = defining_module.target

    rebound = rebind_everywhere(defining_module, "target", replacement)

    # Nothing to scan for: the old and new objects are the same, so the sweep
    # would rebind every holder to what it already has.
    assert rebound == ["_primus_rebind_owner"]
    assert importer.target is replacement


def test_ignores_none_entries_in_sys_modules(defining_module, importer):
    sys.modules["_primus_rebind_none"] = None

    try:
        rebind_everywhere(defining_module, "target", lambda: "patched")
    finally:
        sys.modules.pop("_primus_rebind_none", None)

    assert importer.target() == "patched"


def test_returns_owner_first(defining_module):
    holder = SimpleNamespace(target=defining_module.target)
    sys.modules["_primus_rebind_ns"] = holder

    try:
        rebound = rebind_everywhere(defining_module, "target", lambda: "patched")
    finally:
        sys.modules.pop("_primus_rebind_ns", None)

    assert rebound[0] == "_primus_rebind_owner"
