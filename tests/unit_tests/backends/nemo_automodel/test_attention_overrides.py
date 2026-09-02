###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the attention backend overrides and their shared registry.

WHAT THESE ARE DEFENDING:
  Both overrides rebind the same diffusers registry entries, and the original
  code checked only for its OWN marker before doing so. With both env gates set,
  the second override would therefore wrap the first one's wrapper: one kernel
  running inside the other's fallback path, with nothing in the logs to say so
  and no error. The shared marker and the refusal are the fix, and
  ``test_a_second_different_override_is_refused`` is the test that would have
  caught it.

  The other silent failure is the fallback set. Each override serves only the
  plain call and defers the rest to the original kernel; if a condition were
  dropped from that list the wrong kernel would run and produce plausible
  numbers. Each condition is pinned individually.

The registry is faked rather than mocked out of diffusers, because what is being
tested is this module's walk over it, not diffusers' own behaviour. Nothing here
needs torch or a GPU; the kernels themselves need both and are covered by the
V3 numerics stage.
"""

import sys
import types

import pytest

from primus.backends.nemo_automodel.attention import _backend_registry


class FakeBackendName:
    """Stands in for diffusers' AttentionBackendName enum members."""

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, FakeBackendName) and other.value == self.value


@pytest.fixture
def fake_diffusers(monkeypatch):
    """Install a fake ``diffusers.models.attention_dispatch`` for the walk to find.

    Returns the fake registry so a test can inspect what was rebound.
    """
    flash = FakeBackendName("flash")
    aiter = FakeBackendName("aiter")

    def original_flash(**kwargs):
        return ("original-flash", kwargs)

    def original_aiter(**kwargs):
        return ("original-aiter", kwargs)

    class FakeRegistry:
        _backends = {flash: original_flash, aiter: original_aiter}
        _supported_arg_names = {flash: {"query", "key", "value"}, aiter: {"query", "key", "value"}}
        _constraints = {flash: [], aiter: []}

    class FakeNames:
        FLASH = flash
        AITER = aiter

    module = types.ModuleType("diffusers.models.attention_dispatch")
    module.AttentionBackendName = FakeNames
    module._AttentionBackendRegistry = FakeRegistry

    parent = types.ModuleType("diffusers.models")
    root = types.ModuleType("diffusers")
    monkeypatch.setitem(sys.modules, "diffusers", root)
    monkeypatch.setitem(sys.modules, "diffusers.models", parent)
    monkeypatch.setitem(sys.modules, "diffusers.models.attention_dispatch", module)

    # The warn-once set is module state and would suppress warnings across tests.
    monkeypatch.setattr(_backend_registry, "_warned", set())

    FakeRegistry.originals = {flash: original_flash, aiter: original_aiter}
    return FakeRegistry


def spy_kernel(calls):
    def kernel(q, k, v, softmax_scale=None, causal=False):
        calls.append({"softmax_scale": softmax_scale, "causal": causal})
        return "kernel-output"

    return kernel


def install(name="test-override", kernel=None, log_prefix="[test]"):
    return _backend_registry.install_override(
        kernel=kernel or (lambda q, k, v, softmax_scale=None, causal=False: "kernel-output"),
        override_name=name,
        log_prefix=log_prefix,
        description="a test kernel",
        probe=lambda: None,
    )


class TestInstall:
    def test_it_rebinds_both_target_backends(self, fake_diffusers):
        assert install() is True
        for backend in fake_diffusers._backends:
            assert getattr(fake_diffusers._backends[backend], _backend_registry.MARKER) == ("test-override")

    def test_it_leaves_arg_names_and_constraints_alone(self, fake_diffusers):
        """Dispatch filters kwargs to the ORIGINAL supported set, and the wrapper
        accepts a superset, so touching these would only break things."""
        before_args = dict(fake_diffusers._supported_arg_names)
        before_constraints = dict(fake_diffusers._constraints)
        install()
        assert fake_diffusers._supported_arg_names == before_args
        assert fake_diffusers._constraints == before_constraints

    def test_it_keeps_the_original_reachable(self, fake_diffusers):
        install()
        for backend, original in fake_diffusers.originals.items():
            assert fake_diffusers._backends[backend]._primus_orig_fn is original

    def test_a_failing_probe_propagates(self, fake_diffusers):
        """A missing kernel library has to fail the run, not leave the original
        backend in place looking like the override worked."""

        def probe():
            raise ImportError("no kernel library")

        with pytest.raises(ImportError):
            _backend_registry.install_override(
                kernel=lambda *a, **k: None,
                override_name="x",
                log_prefix="[x]",
                description="d",
                probe=probe,
            )

    def test_reinstalling_the_same_override_is_idempotent(self, fake_diffusers):
        assert install() is True
        wrappers = dict(fake_diffusers._backends)
        assert install() is True
        assert fake_diffusers._backends == wrappers

    def test_a_second_different_override_is_refused(self, fake_diffusers, caplog):
        """The regression test for the shared marker. Checking only one's own
        marker means the second override wraps the first one's wrapper: one kernel
        inside the other's fallback path, silently."""
        assert install(name="first") is True
        with caplog.at_level("WARNING"):
            assert install(name="second") is False
        for backend in fake_diffusers._backends:
            owner = getattr(fake_diffusers._backends[backend], _backend_registry.MARKER)
            assert owner == "first"
        assert any("already owns it" in r.message for r in caplog.records)

    def test_the_refusal_names_the_owner(self, fake_diffusers, caplog):
        install(name="first")
        with caplog.at_level("WARNING"):
            install(name="second", log_prefix="[second]")
        # getMessage() interpolates args into msg. Reaching for `.message % .args`
        # instead double-formats, because caplog has already substituted them.
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "first" in joined and "[second]" in joined

    def test_an_unregistered_backend_is_skipped(self, fake_diffusers, caplog):
        fake_diffusers._backends.clear()
        with caplog.at_level("WARNING"):
            assert install() is False
        assert any("NOT active" in r.message for r in caplog.records)

    def test_uninstall_restores_the_originals(self, fake_diffusers):
        install(name="mine")
        assert _backend_registry.uninstall_override("mine") == 2
        for backend, original in fake_diffusers.originals.items():
            assert fake_diffusers._backends[backend] is original

    def test_uninstall_ignores_another_overrides_entries(self, fake_diffusers):
        install(name="mine")
        assert _backend_registry.uninstall_override("not-mine") == 0


class TestDispatch:
    """The plain path reaches the kernel; everything else reaches the original."""

    def _wrapper(self, fake_diffusers, calls):
        install(kernel=spy_kernel(calls))
        flash = next(b for b in fake_diffusers._backends if b.value == "flash")
        return fake_diffusers._backends[flash]

    def test_the_plain_call_reaches_the_kernel(self, fake_diffusers):
        calls = []
        wrapper = self._wrapper(fake_diffusers, calls)
        assert wrapper(query="q", key="k", value="v") == "kernel-output"
        assert len(calls) == 1

    def test_scale_and_causal_are_forwarded(self, fake_diffusers):
        """The kernel takes softmax_scale/causal while dispatch supplies
        scale/is_causal, so this rename is a real chance to drop a value."""
        calls = []
        wrapper = self._wrapper(fake_diffusers, calls)
        wrapper(query="q", key="k", value="v", scale=0.125, is_causal=True)
        assert calls[0] == {"softmax_scale": 0.125, "causal": True}

    @pytest.mark.parametrize(
        "kwargs,reason",
        [
            ({"_parallel_config": object()}, "context parallelism"),
            ({"return_lse": True}, "return_lse"),
            ({"attn_mask": "mask"}, "attn_mask"),
            ({"dropout_p": 0.1}, "dropout_p"),
            ({"window_size": (256, 256)}, "window_size"),
        ],
    )
    def test_unsupported_calls_reach_the_original(self, fake_diffusers, kwargs, reason):
        calls = []
        wrapper = self._wrapper(fake_diffusers, calls)
        result = wrapper(query="q", key="k", value="v", **kwargs)
        assert result[0] == "original-flash"
        assert calls == [], f"{reason} should not have reached the override kernel"

    def test_the_fallback_warns_once_per_reason(self, fake_diffusers, caplog):
        wrapper = self._wrapper(fake_diffusers, [])
        with caplog.at_level("WARNING"):
            wrapper(query="q", key="k", value="v", return_lse=True)
            wrapper(query="q", key="k", value="v", return_lse=True)
        assert sum("falling back" in r.message for r in caplog.records) == 1

    def test_distinct_reasons_each_warn(self, fake_diffusers, caplog):
        wrapper = self._wrapper(fake_diffusers, [])
        with caplog.at_level("WARNING"):
            wrapper(query="q", key="k", value="v", return_lse=True)
            wrapper(query="q", key="k", value="v", dropout_p=0.5)
        assert sum("falling back" in r.message for r in caplog.records) == 2

    def test_window_size_is_not_passed_to_a_backend_that_rejects_it(self, fake_diffusers):
        """AITER's original does not accept window_size; passing it would be a
        TypeError that only surfaces once a fallback actually fires."""
        install()
        aiter = next(b for b in fake_diffusers._backends if b.value == "aiter")
        result = fake_diffusers._backends[aiter](query="q", key="k", value="v", return_lse=True)
        assert result[0] == "original-aiter"
        assert "window_size" not in result[1]

    def test_window_size_is_passed_to_a_backend_that_accepts_it(self, fake_diffusers):
        install()
        flash = next(b for b in fake_diffusers._backends if b.value == "flash")
        result = fake_diffusers._backends[flash](query="q", key="k", value="v", return_lse=True)
        assert result[1]["window_size"] == (-1, -1)


class TestUnsupportedReason:
    def test_the_plain_call_is_supported(self):
        assert _backend_registry.unsupported_reason(None, 0.0, (-1, -1), False, None) is None

    def test_a_none_window_size_is_supported(self):
        assert _backend_registry.unsupported_reason(None, 0.0, None, False, None) is None

    def test_each_condition_is_reported(self):
        assert "context parallelism" in _backend_registry.unsupported_reason(
            None, 0.0, (-1, -1), False, object()
        )
        assert "return_lse" in _backend_registry.unsupported_reason(None, 0.0, (-1, -1), True, None)
        assert "attn_mask" in _backend_registry.unsupported_reason("m", 0.0, (-1, -1), False, None)
        assert "dropout_p" in _backend_registry.unsupported_reason(None, 0.3, (-1, -1), False, None)
        assert "window_size" in _backend_registry.unsupported_reason(None, 0.0, (8, 8), False, None)

    def test_context_parallelism_is_checked_first(self):
        """A CP run with a mask should report CP, since that is the condition that
        makes the whole path inapplicable."""
        reason = _backend_registry.unsupported_reason("m", 0.5, (8, 8), True, object())
        assert "context parallelism" in reason


class TestEnvGates:
    def test_fp8_attn_is_off_by_default(self, monkeypatch):
        from primus.backends.nemo_automodel.attention import fp8

        monkeypatch.delenv("PRIMUS_TURBO_FP8_ATTN", raising=False)
        assert fp8.is_enabled() is False
        monkeypatch.setenv("PRIMUS_TURBO_FP8_ATTN", "1")
        assert fp8.is_enabled() is True

    def test_nondeterministic_is_off_by_default(self, monkeypatch):
        from primus.backends.nemo_automodel.attention import nondeterministic

        monkeypatch.delenv("PRIMUS_ATTN_NONDETERMINISTIC", raising=False)
        assert nondeterministic.is_enabled() is False
        monkeypatch.setenv("PRIMUS_ATTN_NONDETERMINISTIC", "1")
        assert nondeterministic.is_enabled() is True

    def test_the_two_overrides_have_distinct_names(self):
        """They share a marker namespace, so a collision would make each look
        already-installed to the other."""
        from primus.backends.nemo_automodel.attention import fp8, nondeterministic

        assert fp8.OVERRIDE_NAME != nondeterministic.OVERRIDE_NAME


class TestPatchRegistration:
    def _patches(self):
        import primus.backends.nemo_automodel.patches  # noqa: F401
        from primus.core.patches.patch_registry import PatchRegistry

        return {p.id: p for p in PatchRegistry.iter_patches(backend="nemo_automodel", phase="before_train")}

    def test_both_overrides_are_registered(self):
        by_id = self._patches()
        assert "nemo_automodel.attention.fp8" in by_id
        assert "nemo_automodel.attention.nondeterministic" in by_id

    def test_fp8_attention_takes_precedence(self):
        """Whichever installs first owns the registry entries, so the priority is
        what decides the winner."""
        by_id = self._patches()
        assert (
            by_id["nemo_automodel.attention.fp8"].priority
            < by_id["nemo_automodel.attention.nondeterministic"].priority
        )

    def test_the_overrides_precede_the_model_strategies(self):
        """They must be in place before the transformer's first forward."""
        by_id = self._patches()
        for strategy in (
            "nemo_automodel.models.wan.parallelize",
            "nemo_automodel.models.flux.parallelize",
        ):
            assert by_id["nemo_automodel.attention.fp8"].priority < by_id[strategy].priority
            assert by_id["nemo_automodel.attention.nondeterministic"].priority < by_id[strategy].priority

    @pytest.mark.parametrize(
        "patch_id,env",
        [
            ("nemo_automodel.attention.fp8", "PRIMUS_TURBO_FP8_ATTN"),
            ("nemo_automodel.attention.nondeterministic", "PRIMUS_ATTN_NONDETERMINISTIC"),
        ],
    )
    def test_each_patch_is_gated_by_its_env_var(self, monkeypatch, patch_id, env):
        patch = self._patches()[patch_id]
        monkeypatch.delenv(env, raising=False)
        assert patch.condition(None) is False
        monkeypatch.setenv(env, "1")
        assert patch.condition(None) is True


class TestPadToBlock:
    """The pad-to-64 wrapper. Needs torch for the tensor path, so the arithmetic
    is checked separately from the padding itself."""

    def test_the_block_size_matches_the_kernel(self):
        from primus.backends.nemo_automodel.attention import fp8

        assert fp8.ATTN_BLOCK == 64

    @pytest.mark.parametrize(
        "seqlen,expected",
        [(64, 64), (128, 128), (1, 64), (63, 64), (65, 128), (4608, 4608), (256, 256)],
    )
    def test_the_padded_length_is_the_next_multiple(self, seqlen, expected):
        assert ((seqlen + 63) // 64) * 64 == expected


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@pytest.mark.skipif(torch is None, reason="the padding tests need torch")
class TestPadding:
    def test_a_conforming_sequence_is_returned_untouched(self):
        from primus.backends.nemo_automodel.attention import fp8

        t = torch.zeros(2, 128, 4, 64)
        padded, real = fp8.pad_to_block(t)
        assert padded is t, "the common case should allocate nothing"
        assert real == 128

    def test_a_nonconforming_sequence_is_padded(self):
        from primus.backends.nemo_automodel.attention import fp8

        t = torch.ones(2, 100, 4, 64)
        padded, real = fp8.pad_to_block(t)
        assert padded.shape[1] == 128
        assert real == 100

    def test_only_the_sequence_dim_is_padded(self):
        from primus.backends.nemo_automodel.attention import fp8

        t = torch.ones(2, 100, 4, 64)
        padded, _ = fp8.pad_to_block(t)
        assert padded.shape[0] == 2 and padded.shape[2] == 4 and padded.shape[3] == 64

    def test_the_padding_is_zeros_and_the_data_survives(self):
        """Nonzero padding would contribute real weight to the softmax rather than
        the bounded exp(0) dilution the docstring argues is acceptable."""
        from primus.backends.nemo_automodel.attention import fp8

        t = torch.ones(1, 100, 1, 8)
        padded, real = fp8.pad_to_block(t)
        assert torch.all(padded[:, :real] == 1)
        assert torch.all(padded[:, real:] == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
