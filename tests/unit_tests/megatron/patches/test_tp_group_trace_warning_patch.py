###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the tp_group deprecation-warning trace patch.

The patch replaces the ``warnings`` module global inside ``megatron.core.utils``
so that the deprecation notice in ``get_tensor_model_parallel_group_if_none``
stops breaking compiled graphs. Three things have to hold: the warning is gone
while Dynamo traces, it is still emitted verbatim in eager, and the patch only
installs while the upstream defect is actually there.

Everything here runs on CPU without a process group: the Dynamo case compiles a
one-line function, and the Megatron modules are stubs, so the tests also pass in
an environment that has no Megatron-LM at all.
"""

from __future__ import annotations

import linecache
import sys
import warnings
from types import ModuleType

import pytest
import torch

from primus.backends.megatron.patches import tp_group_trace_warning_patches as mod
from primus.backends.megatron.patches.tp_group_trace_warning_patches import (
    ENV_VAR,
    PATCH_ID,
    _defect_present,
    _needs_patch,
    _WarningsSilentWhileTracing,
    patch_tp_group_trace_warning,
)

_MESSAGE = "tp_group is None, using default tensor model parallel group"


# ---------------------------------------------------------------------------
# Upstream stand-ins
#
# The shim is compiled into the fake module's namespace rather than defined
# here, because that is the whole point: the patch works by rebinding the
# module global that the shim's body resolves `warnings` through. A shim
# defined in this test file would read this file's global instead. The source
# is registered with linecache so inspect.getsource() can still read it back,
# which is what the defect detection does to the real shim.
# ---------------------------------------------------------------------------

_SHIM_THAT_WARNS = '''
def get_tensor_model_parallel_group_if_none(tp_group, is_expert=False, check_initialized=True):
    """Shaped like the upstream shim, warning branch included."""
    if tp_group is None:
        warnings.warn(MESSAGE, DeprecationWarning, stacklevel=2)
        return "default-group"
    return tp_group
'''

_SHIM_THAT_DOES_NOT_WARN = '''
def get_tensor_model_parallel_group_if_none(tp_group, is_expert=False, check_initialized=True):
    """Shaped like the shim after upstream fixes or drops the warning."""
    if tp_group is None:
        return "default-group"
    return tp_group
'''


def _install_fake_megatron(monkeypatch, shim_src=_SHIM_THAT_WARNS, version="0.16.0rc0"):
    """Put a minimal megatron.core.utils in sys.modules and return it."""
    utils = ModuleType("megatron.core.utils")
    utils.warnings = warnings
    utils.MESSAGE = _MESSAGE
    if shim_src is not None:
        filename = "<fake megatron/core/utils.py>"
        monkeypatch.setitem(
            linecache.cache, filename, (len(shim_src), None, shim_src.splitlines(True), filename)
        )
        exec(compile(shim_src, filename, "exec"), utils.__dict__)

    core = ModuleType("megatron.core")
    core.__version__ = version
    core.utils = utils
    core.__path__ = []

    root = ModuleType("megatron")
    root.core = core
    root.__path__ = []

    monkeypatch.setitem(sys.modules, "megatron", root)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.utils", utils)
    return utils


@pytest.fixture(autouse=True)
def _quiet_logging(monkeypatch):
    """The patch logs through the Primus logger, which needs no setup here."""
    monkeypatch.setattr(mod, "log_rank_0", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# The stand-in's own behaviour
# ---------------------------------------------------------------------------


def test_warning_is_skipped_while_tracing(monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    stub = _WarningsSilentWhileTracing(warnings)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stub.warn(_MESSAGE, DeprecationWarning, stacklevel=2)

    assert caught == []


def test_warning_still_fires_in_eager(monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    stub = _WarningsSilentWhileTracing(warnings)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stub.warn(_MESSAGE, DeprecationWarning, stacklevel=2)

    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    assert _MESSAGE in str(caught[0].message)


@pytest.mark.parametrize(
    "call, category, why",
    [
        (lambda w: w.warn("plain"), UserWarning, "default category comes from the builtin"),
        (lambda w: w.warn("cat is none", None), UserWarning, "None is normalised by the builtin"),
        (
            lambda w: w.warn(DeprecationWarning("instance")),
            DeprecationWarning,
            "a Warning instance carries its own category",
        ),
        (
            lambda w: w.warn("positional source", UserWarning, 2, object()),
            UserWarning,
            "source can be positional, so no fixed signature may reject it",
        ),
        pytest.param(
            lambda w: w.warn("kwonly", UserWarning, 1, None, skip_file_prefixes=()),
            UserWarning,
            "keyword-only arguments pass through",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 12),
                reason="skip_file_prefixes was added to warnings.warn in 3.12",
            ),
        ),
    ],
)
def test_eager_matches_the_builtin_calling_conventions(monkeypatch, call, category, why):
    """The stand-in declares no signature of its own; the builtin's rules apply.

    The builtin's own default is ``category=None``, not ``UserWarning``, and it
    accepts ``source`` positionally, so restating a signature here would change
    eager behaviour.
    """
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    stub = _WarningsSilentWhileTracing(warnings)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call(stub)

    assert len(caught) == 1 and caught[0].category is category, why


@pytest.mark.parametrize(
    "call, why",
    [
        (lambda w: w.warn("kw", DeprecationWarning, stacklevel=2), "stacklevel as a keyword"),
        (lambda w: w.warn("positional", DeprecationWarning, 2), "stacklevel positionally"),
    ],
)
def test_eager_points_at_the_caller(monkeypatch, call, why):
    """The extra frame this stand-in adds must not shift the reported line."""
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    seen = {}

    # Both warners are invoked from this one line, so a correct stacklevel
    # makes them report the same location.
    for label, warner in (("stand-in", _WarningsSilentWhileTracing(warnings)), ("builtin", warnings)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call(warner)
        seen[label] = caught[0].lineno

    assert seen["stand-in"] == seen["builtin"], why


def test_unknown_attributes_reach_the_real_module():
    stub = _WarningsSilentWhileTracing(warnings)
    assert stub.catch_warnings is warnings.catch_warnings
    assert stub.filters is warnings.filters


# ---------------------------------------------------------------------------
# Applicability: the patch installs only while the defect is present
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shim_src, expected, why",
    [
        (_SHIM_THAT_WARNS, True, "upstream still warns"),
        (_SHIM_THAT_DOES_NOT_WARN, False, "upstream dropped the warning"),
        (None, False, "upstream removed the shim"),
    ],
)
def test_defect_detection(monkeypatch, shim_src, expected, why):
    _install_fake_megatron(monkeypatch, shim_src=shim_src)
    assert _defect_present() is expected, why


@pytest.mark.parametrize(
    "shim_src, expected, why",
    [
        (_SHIM_THAT_WARNS, True, "bytecode still resolves warnings.warn"),
        (_SHIM_THAT_DOES_NOT_WARN, False, "nothing left to resolve"),
    ],
)
def test_defect_detection_without_source(monkeypatch, shim_src, expected, why):
    """A source-free install must not silently disable the patch."""

    def no_source(_):
        raise OSError("source not available")

    _install_fake_megatron(monkeypatch, shim_src=shim_src)
    monkeypatch.setattr(mod.inspect, "getsource", no_source)

    assert _defect_present() is expected, why


def test_no_megatron_is_not_an_error(monkeypatch):
    """A newer Megatron without the shim must not raise AttributeError."""
    for name in ("megatron", "megatron.core", "megatron.core.utils"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    # A None entry makes the import raise, which is the "not installed" path.
    monkeypatch.setitem(sys.modules, "megatron", None)
    assert _defect_present() is False


def test_env_var_overrides_an_applicable_patch(monkeypatch):
    _install_fake_megatron(monkeypatch)
    monkeypatch.setenv(ENV_VAR, "0")
    assert _needs_patch(None) is False

    monkeypatch.setenv(ENV_VAR, "1")
    assert _needs_patch(None) is True

    monkeypatch.delenv(ENV_VAR)
    assert _needs_patch(None) is True, "enabled by default"


def test_patch_id_is_namespaced():
    assert PATCH_ID.startswith("megatron.")


# ---------------------------------------------------------------------------
# Installing the patch
# ---------------------------------------------------------------------------


def test_install_swaps_the_global_and_keeps_the_shim(monkeypatch):
    utils = _install_fake_megatron(monkeypatch)
    shim = utils.get_tensor_model_parallel_group_if_none

    patch_tp_group_trace_warning(None)

    assert isinstance(utils.warnings, _WarningsSilentWhileTracing)
    assert utils.get_tensor_model_parallel_group_if_none is shim, "upstream logic stays authoritative"

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = (shim(None), shim("explicit-group"))
    assert resolved == ("default-group", "explicit-group"), "same group as upstream returns"
    assert caught == [], "warning gone while tracing"

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert (shim(None), shim("explicit-group")) == ("default-group", "explicit-group")
    assert len(caught) == 1, "eager still warns, and only on the None path"


def test_install_is_idempotent(monkeypatch):
    utils = _install_fake_megatron(monkeypatch)
    patch_tp_group_trace_warning(None)
    first = utils.warnings

    assert _defect_present() is False, "an installed patch is not a present defect"
    patch_tp_group_trace_warning(None)
    assert utils.warnings is first, "no stand-in wrapped around a stand-in"


def test_install_skips_a_renamed_global(monkeypatch):
    utils = _install_fake_megatron(monkeypatch)
    utils.warnings = "not a module"

    patch_tp_group_trace_warning(None)
    assert utils.warnings == "not a module", "left alone rather than raising"


# ---------------------------------------------------------------------------
# The reproducer: the graph break itself
# ---------------------------------------------------------------------------


def _region(warner):
    """A compiled region that warns, shaped like a row-parallel forward.

    The tensor work has to straddle the warning: a break on the first statement
    of a frame is elided into the graph prefix and never counted, while a break
    in the middle is the split this patch is about.
    """

    def fn(x):
        y = x * 2
        warner.warn(_MESSAGE, DeprecationWarning, stacklevel=2)
        return y + 1

    return fn


@pytest.mark.parametrize(
    "warner_factory, breaks, why",
    [
        (lambda: warnings, 1, "the C builtin warnings.warn cannot be traced"),
        (lambda: _WarningsSilentWhileTracing(warnings), 0, "a plain Python warn is inlined"),
    ],
)
def test_graph_break_reproducer(warner_factory, breaks, why):
    """Before/after the stand-in, counted by Dynamo itself."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(_region(warner_factory()))(torch.zeros(4))

    assert explanation.graph_break_count == breaks, why
    assert explanation.graph_count == breaks + 1, "one more graph than breaks"


@pytest.mark.parametrize(
    "warner_factory, compiles, why",
    [
        (lambda: warnings, False, "unpatched: fullgraph cannot capture the warn"),
        (lambda: _WarningsSilentWhileTracing(warnings), True, "patched: nothing left to break on"),
    ],
)
def test_fullgraph_capture(warner_factory, compiles, why):
    """The same result stated as capturability, which is what callers care about."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch._dynamo.reset()
        compiled = torch.compile(_region(warner_factory()), fullgraph=True)
        if compiles:
            assert torch.equal(compiled(torch.zeros(4)), torch.ones(4)), why
        else:
            with pytest.raises(torch._dynamo.exc.Unsupported, match="_warnings.warn"):
                compiled(torch.zeros(4))
