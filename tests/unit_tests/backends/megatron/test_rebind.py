###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the monkeypatch rebind helper.

``rebind_everywhere`` exists because ``module.attr = new`` misses every caller
that did ``from module import attr`` before the patch ran.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

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


def test_survives_modules_that_raise_on_getattr(defining_module, importer):
    class Hostile(ModuleType):
        def __getattr__(self, name):
            raise RuntimeError("lazy import boom")

    hostile = Hostile("_primus_rebind_hostile")
    sys.modules[hostile.__name__] = hostile

    try:
        rebind_everywhere(defining_module, "target", lambda: "patched")
    finally:
        sys.modules.pop(hostile.__name__, None)

    assert importer.target() == "patched"


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
