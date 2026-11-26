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
  flags have to survive the rebuild, and a traceback through it has to show
  the rewritten line where it sits upstream. The train-loop tests only ever
  drive it with anchors that match, through a method that calls neither.
"""

import importlib.util
import sys
import textwrap
import traceback

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
            total = 0
            total += 1
            total += 1
            return total

        def annotated(self, x: int) -> int:
            return x

        @staticmethod
        def static():
            return 1

        @classmethod
        def klass(cls):
            return 1


    class GrandChild(Child):
        pass


    def make_class(offset):
        class Local:
            def shifted(self, x):
                return x + offset

        return Local
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


@pytest.mark.parametrize("anchor,found", [("total += 2\n", 0), ("total += 1\n", 2)])
def test_an_anchor_that_does_not_match_exactly_once_raises_and_changes_nothing(stub, anchor, found):
    original = stub.Child.__dict__["twice"]

    with pytest.raises(RuntimeError, match=f"Anchor matched {found} times in Child.twice"):
        patch_method_source(stub.Child, "twice", [("return total", "return 0"), (anchor, "total += 3\n")])

    assert stub.Child.__dict__["twice"] is original
    assert stub.Child().twice() == 2


def test_a_second_patch_of_one_method_is_a_clear_error(stub):
    patch_method_source(stub.Child, "greet", [('"child+"', '"patched+"')])

    with pytest.raises(RuntimeError, match="source-patched only once"):
        patch_method_source(stub.Child, "greet", [('"patched+"', '"again+"')])


def test_a_method_the_class_only_inherits_is_a_clear_error(stub):
    with pytest.raises(RuntimeError, match="greet is not defined on GrandChild itself"):
        patch_method_source(stub.GrandChild, "greet", [('"child+"', '"patched+"')])


@pytest.mark.parametrize("name,kind", [("static", "staticmethod"), ("klass", "classmethod")])
def test_a_static_or_class_method_is_a_clear_error(stub, name, kind):
    with pytest.raises(RuntimeError, match=f"Child.{name} is a {kind}; only plain functions"):
        patch_method_source(stub.Child, name, [("return 1", "return 2")])


def test_a_method_reading_an_enclosing_functions_variables_is_a_clear_error(stub):
    local_cls = stub.make_class(1)

    with pytest.raises(RuntimeError, match=r"reads variables of an enclosing function \(offset\)"):
        patch_method_source(local_cls, "shifted", [("x + offset", "x - offset")])
    assert local_cls().shifted(1) == 2


def test_a_traceback_shows_the_rewritten_line_at_its_upstream_line_number(stub):
    original = stub.Child.__dict__["greet"]
    patch_method_source(
        stub.Child, "greet", [('return "child+" + super().greet()', 'raise ValueError("patched")')]
    )

    with pytest.raises(ValueError) as info:
        stub.Child().greet()

    frame = traceback.extract_tb(info.value.__traceback__)[-1]
    assert frame.name == "greet"
    assert original.__code__.co_filename in frame.filename
    assert frame.lineno == original.__code__.co_firstlineno + 2
    assert frame.line == 'raise ValueError("patched")'


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
