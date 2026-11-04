###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the source-rewrite helper (``primus/backends/automodel/_source_patch.py``).

WHY THIS TEST EXISTS:
  The helper is what stops a source patch from quietly editing the wrong code:
  an anchor that no longer matches, or matches twice, has to raise and leave
  the class alone. It also rebuilds the method outside its class, so
  zero-argument ``super()`` and the original module's ``from __future__``
  flags have to survive the rebuild. The train-loop tests only ever drive it
  with anchors that match, through a method that calls neither.
"""

import importlib.util
import sys
import textwrap

import pytest

from primus.backends.automodel._source_patch import patch_method_source

STUB_SOURCE = textwrap.dedent(
    '''
    class Base:
        def greet(self):
            return "base"


    class Child(Base):
        def greet(self):
            """Child's greeting."""
            return "child+" + super().greet()

        def twice(self):
            value = 1
            value = 1
            return value

        def annotated(self, x: int) -> int:
            return x


    class GrandChild(Child):
        pass
    '''
)


def _load(tmp_path, monkeypatch, source):
    """Import ``source`` from a real file, which ``inspect.getsource`` needs."""
    path = tmp_path / "source_patch_stub.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("source_patch_stub", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "source_patch_stub", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stub(tmp_path, monkeypatch):
    return _load(tmp_path, monkeypatch, STUB_SOURCE)


def test_a_rewrite_is_applied_and_zero_argument_super_still_works(stub):
    patch_method_source(stub.Child, "greet", [('return "child+"', 'return "patched+"')])

    assert stub.Child().greet() == "patched+base"
    assert stub.GrandChild().greet() == "patched+base"
    assert stub.Child.greet.__qualname__ == "Child.greet"
    assert stub.Child.greet.__module__ == "source_patch_stub"
    assert stub.Child.greet.__doc__ == "Child's greeting."


@pytest.mark.parametrize("anchor,found", [("value = 2\n", 0), ("value = 1\n", 2)])
def test_an_anchor_that_does_not_match_exactly_once_raises_and_changes_nothing(stub, anchor, found):
    original = stub.Child.__dict__["twice"]

    with pytest.raises(RuntimeError, match=f"Anchor matched {found} times in Child.twice"):
        patch_method_source(stub.Child, "twice", [("return value", "return 0"), (anchor, "value = 3\n")])

    assert stub.Child.__dict__["twice"] is original
    assert stub.Child().twice() == 1


def test_a_second_patch_of_one_method_is_a_clear_error(stub):
    patch_method_source(stub.Child, "greet", [('"child+"', '"patched+"')])

    with pytest.raises(RuntimeError, match="source-patched only once"):
        patch_method_source(stub.Child, "greet", [('"patched+"', '"again+"')])


def test_a_method_the_class_only_inherits_is_a_clear_error(stub):
    with pytest.raises(RuntimeError, match="greet is not defined on GrandChild itself"):
        patch_method_source(stub.GrandChild, "greet", [('"child+"', '"patched+"')])


@pytest.mark.parametrize("future_import,annotation", [(False, int), (True, "int")])
def test_the_rewrite_keeps_the_original_modules_future_flags(
    tmp_path, monkeypatch, future_import, annotation
):
    """The rewritten method is compiled as the module it came from was, whatever the caller's flags."""
    header = "from __future__ import annotations\n" if future_import else ""
    stub = _load(tmp_path, monkeypatch, header + STUB_SOURCE)

    patch_method_source(stub.Child, "annotated", [("return x", "return x + 1")])

    assert stub.Child().annotated(1) == 2
    assert stub.Child.annotated.__annotations__ == {"x": annotation, "return": annotation}
