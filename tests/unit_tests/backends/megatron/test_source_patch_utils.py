###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the helpers that let source-rewrite patches survive upstream refactors."""

import pytest

from primus.backends.megatron.patches._source_patch_utils import (
    anchor_indent,
    find_source_owner,
    patch_method_source,
)


class _Base:
    def __init__(self):
        self.value = 1
        self.setup_done = True

    def forward(self):
        return self.value


class _ThinSubclass(_Base):
    """Stands in for Megatron-LM main's deprecated ``MambaModel(HybridModel)``."""

    def __init__(self):
        super().__init__()


def test_find_source_owner_walks_past_thin_subclass():
    assert find_source_owner(_ThinSubclass, "__init__", "self.setup_done = True") is _Base
    assert find_source_owner(_ThinSubclass, "forward", "return self.value") is _Base


def test_find_source_owner_prefers_the_defining_subclass():
    assert find_source_owner(_ThinSubclass, "__init__", "super().__init__()") is _ThinSubclass


def test_find_source_owner_raises_on_drift():
    with pytest.raises(RuntimeError, match="Anchor not found"):
        find_source_owner(_ThinSubclass, "__init__", "self.not_there = 0")


def test_patch_on_owner_reaches_subclass_instances():
    class Base:
        def forward(self):
            return 1

    class Thin(Base):
        pass

    owner = find_source_owner(Thin, "forward", "return 1")
    patch_method_source(owner, "forward", "return 1", "return 2")
    assert Thin().forward() == 2


def test_anchor_indent_reports_nesting_depth():
    def nested(flag):
        marker = 0
        if flag:
            for _ in range(1):
                marker = 1
        return marker

    assert anchor_indent(nested, "marker = 1") == " " * 16


def test_anchor_indent_raises_on_drift():
    with pytest.raises(RuntimeError, match="Anchor not found"):
        anchor_indent(_Base.forward, "self.not_there")
