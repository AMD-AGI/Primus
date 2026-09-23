###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the low-precision backend selector and the FP8 linear swap.

WHY THE SELECTOR IS TESTED HARDEST:
  Exactly one precision can own AutoModel's ``_replace_linear_with_transformer
  _engine`` symbol, and more than one can be requested at once. The failure mode
  is someone setting two env vars, getting a run, and believing it used the
  narrower format. Nothing raises and the logs from the winning backend look
  entirely normal, so the only defence is pinning the precedence order and
  asserting the loser is announced rather than dropped in silence.

  The selector is also the C9 remedy -- the alternative was an if-chain in a
  shared install() that every precision added later would have to edit -- so the
  tests here register synthetic backends to check the ordering holds for
  precisions that do not exist yet.

The selector and env-parsing tests deliberately need no torch, so they still run
where it is absent; only the tensor-level swap tests require it.
"""

import os

import pytest

from primus.backends.nemo_automodel.quantization import _common, fp8_linear

# Guarded rather than importorskip, which would skip the whole module and take
# the torch-free selector tests down with it.
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="the tensor-level swap tests need torch")

if torch is not None:

    class _Marked(torch.nn.Linear):
        """Stand-in for the real low-precision Linear, so the shared walk can be
        tested without primus_turbo."""

else:  # pragma: no cover
    _Marked = None


@pytest.fixture
def clean_registry(monkeypatch):
    """Isolate the module-level registry so tests do not leak into each other."""
    monkeypatch.setattr(_common, "_BACKENDS", {}, raising=False)
    return _common


def register(reg, name, precedence, requested):
    reg.register_backend(name, precedence=precedence, is_requested=lambda: requested, description=name)


class TestSelector:
    def test_nothing_requested_means_no_active_backend(self, clean_registry):
        register(clean_registry, "a", 10, False)
        assert clean_registry.active_backend() is None
        assert clean_registry.is_active("a") is False

    def test_a_single_request_wins(self, clean_registry):
        register(clean_registry, "a", 10, True)
        assert clean_registry.active_backend().name == "a"
        assert clean_registry.is_active("a") is True

    def test_highest_precedence_wins(self, clean_registry):
        register(clean_registry, "fp8", 10, True)
        register(clean_registry, "fp4", 20, True)
        register(clean_registry, "te_fp4", 30, True)
        assert clean_registry.active_backend().name == "te_fp4"
        assert clean_registry.is_active("fp8") is False
        assert clean_registry.is_active("fp4") is False

    def test_precedence_not_registration_order(self, clean_registry):
        """Registration is import order, which is discovery order, which is not
        something anyone should have to reason about."""
        register(clean_registry, "high", 30, True)
        register(clean_registry, "low", 10, True)
        assert clean_registry.active_backend().name == "high"

    def test_an_unrequested_higher_precedence_backend_does_not_win(self, clean_registry):
        register(clean_registry, "fp8", 10, True)
        register(clean_registry, "fp4", 20, False)
        assert clean_registry.active_backend().name == "fp8"

    def test_a_losing_request_is_announced(self, clean_registry, caplog):
        """Silently ignoring it leaves someone looking at a run they believe is
        FP4 and is not."""
        register(clean_registry, "fp8", 10, True)
        register(clean_registry, "fp4", 20, True)
        with caplog.at_level("WARNING"):
            clean_registry.active_backend()
        assert any("takes precedence" in r.message for r in caplog.records)
        assert any("fp8" in r.message for r in caplog.records)

    def test_reregistering_with_a_different_precedence_is_refused(self, clean_registry):
        """Two files disagreeing about the order is a bug worth failing on, not
        resolving by import order."""
        register(clean_registry, "a", 10, True)
        with pytest.raises(ValueError, match="already registered"):
            register(clean_registry, "a", 99, True)

    def test_reregistering_identically_is_allowed(self, clean_registry):
        register(clean_registry, "a", 10, True)
        register(clean_registry, "a", 10, True)  # module re-imported; not an error

    def test_backends_are_listed_highest_first(self, clean_registry):
        register(clean_registry, "low", 10, True)
        register(clean_registry, "high", 30, True)
        assert [e.name for e in clean_registry.registered_backends()] == ["high", "low"]


class TestDiscovery:
    """Registration is a side effect of importing a module, so the selector has to
    import every precision before it can decide which one wins.

    Without that, each patch condition imports only its own backend, so the first
    one evaluated sees a registry containing nothing but itself, concludes it is
    the highest-precedence request and installs -- and the next one does the same
    and overwrites it. Two swaps get installed, the last wins by evaluation order
    rather than precedence, and the warning line announces that a backend will not
    be applied immediately before applying it.
    """

    def test_every_precision_is_registered_without_importing_it_first(self):
        """In a fresh interpreter that has touched nothing but the selector."""
        import subprocess
        import sys

        code = (
            "from primus.backends.nemo_automodel.quantization import _common\n"
            "print(sorted(e.name for e in _common.registered_backends()))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "turbo_fp8" in result.stdout

    def test_exactly_one_patch_is_active_when_two_are_requested(self, monkeypatch):
        """The regression this guards: the bug was invisible to every test that
        asked the selector directly, because asking it imported it."""
        import subprocess
        import sys

        code = (
            "import primus.backends.nemo_automodel.patches\n"
            "from primus.core.patches.patch_registry import PatchRegistry\n"
            "patches = [p for p in PatchRegistry.iter_patches("
            "backend='nemo_automodel', phase='before_train') if 'quantization' in p.id]\n"
            "print([p.id for p in patches if p.condition(None)])"
        )
        env = dict(os.environ, PRIMUS_TURBO_FP8="1", PRIMUS_TURBO_FP4="1")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr
        active = eval(result.stdout.strip().splitlines()[-1])
        assert len(active) == 1, f"more than one swap would install: {active}"

    def test_a_missing_library_does_not_break_the_selector(self, monkeypatch):
        """A precision whose library is absent cannot win, so it should drop out
        of the registry rather than stop a run that wanted a different one."""
        import importlib

        real_import = importlib.import_module

        def failing_import(name, *args, **kwargs):
            if name.endswith("te_mxfp4_linear"):
                raise ImportError("no transformer_engine in this image")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", failing_import)
        monkeypatch.setattr(_common, "_discovered", False)
        names = [e.name for e in _common.registered_backends()]
        assert "turbo_fp8" in names

    def test_discovery_only_walks_the_package_once(self, monkeypatch):
        """It is called from every selector query, so a repeated package walk
        would put an import scan on a hot path."""
        import importlib

        walks = []
        real_import = importlib.import_module

        def counting_import(name, *args, **kwargs):
            if name == _common.__package__:
                walks.append(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", counting_import)
        monkeypatch.setattr(_common, "_discovered", False)
        for _ in range(3):
            _common.registered_backends()
        assert walks == [_common.__package__]

    def test_a_raising_module_is_not_retried(self, monkeypatch):
        """The flag is set before importing, so a module that raises does not put
        a failing import on every subsequent selector query."""
        import importlib

        attempts = []
        real_import = importlib.import_module

        def failing_import(name, *args, **kwargs):
            if name.endswith("fp8_linear"):
                attempts.append(name)
                raise ImportError("boom")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", failing_import)
        monkeypatch.setattr(_common, "_discovered", False)
        for _ in range(3):
            _common.registered_backends()
        assert len(attempts) == 1


class TestFp8Registration:
    def test_fp8_registers_itself_on_import(self):
        names = [e.name for e in _common.registered_backends()]
        assert fp8_linear.BACKEND_NAME in names

    def test_fp8_is_the_lowest_precedence(self):
        """It is the fallback: a narrower format asked for at the same time wins."""
        entry = next(e for e in _common.registered_backends() if e.name == fp8_linear.BACKEND_NAME)
        assert entry.precedence == 10

    def test_requested_follows_the_env_var(self, monkeypatch):
        monkeypatch.delenv("PRIMUS_TURBO_FP8", raising=False)
        assert fp8_linear.is_enabled() is False
        monkeypatch.setenv("PRIMUS_TURBO_FP8", "1")
        assert fp8_linear.is_enabled() is True

    def test_the_patch_is_registered_and_gated(self, monkeypatch):
        import primus.backends.nemo_automodel.patches  # noqa: F401
        from primus.core.patches.patch_registry import PatchRegistry

        patch = next(
            (
                p
                for p in PatchRegistry.iter_patches(backend="nemo_automodel", phase="before_train")
                if p.id == "nemo_automodel.quantization.fp8_linear"
            ),
            None,
        )
        assert patch is not None
        monkeypatch.delenv("PRIMUS_TURBO_FP8", raising=False)
        assert patch.condition(None) is False
        monkeypatch.setenv("PRIMUS_TURBO_FP8", "1")
        assert patch.condition(None) is True

    def test_the_swap_runs_before_the_strategies(self):
        """The transformer has to be swapped before it is built and sharded."""
        import primus.backends.nemo_automodel.patches  # noqa: F401
        from primus.core.patches.patch_registry import PatchRegistry

        by_id = {p.id: p for p in PatchRegistry.iter_patches(backend="nemo_automodel", phase="before_train")}
        assert (
            by_id["nemo_automodel.quantization.fp8_linear"].priority
            < by_id["nemo_automodel.models.wan.parallelize"].priority
        )


class TestConfigResolution:
    def test_an_invalid_granularity_raises(self, monkeypatch):
        """A typo that silently trains in a different format is worse than a
        failed launch, and is otherwise invisible."""
        monkeypatch.setenv("PRIMUS_TURBO_FP8_GRANULARITY", "TENSORWIZE")
        with pytest.raises(ValueError, match="PRIMUS_TURBO_FP8_GRANULARITY"):
            fp8_linear.resolve_config()

    def test_an_invalid_format_raises(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_FP8_FORMAT", "E4M4")
        with pytest.raises(ValueError, match="PRIMUS_TURBO_FP8_FORMAT"):
            fp8_linear.resolve_config()

    @pytest.mark.parametrize("gran", ["TENSORWISE", "ROWWISE", "BLOCKWISE", "MX_BLOCKWISE"])
    def test_every_advertised_granularity_actually_builds(self, gran, monkeypatch):
        """Listing a granularity as valid and then failing to construct it is the
        worst of both: the name check passes, so the error surfaces from inside
        primus_turbo as a bare assertion about a field the user was never asked for.
        The two blockwise ones need a block size; MX additionally pins the scale
        dtype."""
        pytest.importorskip("primus_turbo")
        monkeypatch.setenv("PRIMUS_TURBO_FP8_GRANULARITY", gran)
        cfg = fp8_linear.resolve_config()
        assert cfg.granularity.name == gran
        if gran == "BLOCKWISE":
            assert cfg.block_size == 128
        elif gran == "MX_BLOCKWISE":
            assert cfg.block_size == 32
            assert cfg.scale_dtype.name == "E8M0"

    def test_the_blockwise_block_size_is_settable(self, monkeypatch):
        pytest.importorskip("primus_turbo")
        monkeypatch.setenv("PRIMUS_TURBO_FP8_GRANULARITY", "BLOCKWISE")
        monkeypatch.setenv("PRIMUS_TURBO_FP8_BLOCK_SIZE", "64")
        assert fp8_linear.resolve_config().block_size == 64


@requires_torch
class TestReplaceLinears:
    """The shared walk. Uses plain nn.Linear as its own replacement type, so this
    exercises the traversal and parameter copying without primus_turbo."""

    def _model(self):
        return torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU()),
            torch.nn.LayerNorm(16),
        )

    def _factory(self, linear):
        return _Marked(linear.in_features, linear.out_features, bias=linear.bias is not None)

    def test_it_converts_nested_linears(self):
        model = self._model()
        converted, skipped = _common.replace_linears(
            model,
            "m",
            factory=self._factory,
            should_convert=lambda fqn, lin: True,
            already_converted=(_Marked,),
            log_prefix="[t]",
        )
        assert (converted, skipped) == (2, 0)
        assert isinstance(model[0], _Marked)
        assert isinstance(model[1][0], _Marked)

    def test_it_leaves_non_linear_modules_alone(self):
        model = self._model()
        _common.replace_linears(
            model,
            "m",
            factory=self._factory,
            should_convert=lambda fqn, lin: True,
            already_converted=(_Marked,),
            log_prefix="[t]",
        )
        assert isinstance(model[2], torch.nn.LayerNorm)

    def test_the_skip_predicate_is_honoured(self):
        model = self._model()
        converted, skipped = _common.replace_linears(
            model,
            "m",
            factory=self._factory,
            # Skip the one with mismatched features, as the real skip-lists do.
            should_convert=lambda fqn, lin: lin.in_features == lin.out_features,
            already_converted=(_Marked,),
            log_prefix="[t]",
        )
        assert (converted, skipped) == (1, 1)
        assert isinstance(model[1][0], torch.nn.Linear)
        assert not isinstance(model[1][0], _Marked)

    def test_a_second_pass_is_a_no_op(self):
        """Without the already-converted check, a re-entrant install wraps a
        wrapper and the numerics change silently."""
        model = self._model()
        kwargs = dict(
            factory=self._factory,
            should_convert=lambda fqn, lin: True,
            already_converted=(_Marked,),
            log_prefix="[t]",
        )
        _common.replace_linears(model, "m", **kwargs)
        converted, _ = _common.replace_linears(model, "m", **kwargs)
        assert converted == 0

    def test_weights_and_bias_are_copied(self):
        src = torch.nn.Linear(8, 8)
        with torch.no_grad():
            src.weight.fill_(0.5)
            src.bias.fill_(-0.25)
        dst = torch.nn.Linear(8, 8)
        _common.copy_linear_params(dst, src)
        assert torch.allclose(dst.weight, src.weight)
        assert torch.allclose(dst.bias, src.bias)

    def test_requires_grad_is_preserved(self):
        """A swap that silently unfreezes a layer changes what the optimizer sees
        with nothing visible in the config."""
        src = torch.nn.Linear(8, 8)
        src.weight.requires_grad_(False)
        dst = torch.nn.Linear(8, 8)
        _common.copy_linear_params(dst, src)
        assert dst.weight.requires_grad is False

    def test_the_training_flag_is_preserved(self):
        src = torch.nn.Linear(8, 8).eval()
        dst = torch.nn.Linear(8, 8).train()
        _common.copy_linear_params(dst, src)
        assert dst.training is False

    def test_a_bias_free_source_is_handled(self):
        src = torch.nn.Linear(8, 8, bias=False)
        dst = torch.nn.Linear(8, 8, bias=False)
        _common.copy_linear_params(dst, src)
        assert dst.bias is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
