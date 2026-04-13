###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for ``validate_args_patches.py``.

Verifies:
  1. Base wrapper installs correctly and invokes both original + ROCm validation.
  2. Source-modification patches (pipeline_split, fp4) rewrite validate_args as expected.
  3. Patches compose: base wrapper + source-mod work together end-to-end.
"""

import linecache
import types
from types import SimpleNamespace

import pytest

from primus.core.patches.context import PatchContext

# ---------------------------------------------------------------------------
# Fake megatron modules
# ---------------------------------------------------------------------------

_FAKE_VALIDATE_FILE = "<fake_validate_args>"
_FAKE_VALIDATE_SOURCE = '''\
def validate_args(args, defaults={}):
    """Minimal fake validate_args with the two code paths we patch."""
    if args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None:
        args._entered_pipeline_split_block = True
    else:
        args._entered_pipeline_split_block = False

    if getattr(args, "fp4", False):
        raise ValueError("--fp4-format requires Transformer Engine >= 2.7.0.dev0 for NVFP4BlockScaling support.")

    args._validate_args_called = True
'''

# Register with linecache so inspect.getsource works on exec'd functions.
linecache.cache[_FAKE_VALIDATE_FILE] = (
    len(_FAKE_VALIDATE_SOURCE),
    None,
    _FAKE_VALIDATE_SOURCE.splitlines(True),
    _FAKE_VALIDATE_FILE,
)


def _install_fake_megatron(monkeypatch: pytest.MonkeyPatch):
    """Create fake ``megatron.*`` modules sufficient for validate_args patches.

    Includes ``megatron.core.parallel_state`` and ``megatron.training.global_vars``
    so that transitive imports from ``primus.modules.trainer.megatron.utils`` succeed.
    """
    import sys

    megatron_mod = types.ModuleType("megatron")
    megatron_mod.__path__ = []

    training_pkg = types.ModuleType("megatron.training")
    training_pkg.__path__ = []
    args_mod = types.ModuleType("megatron.training.arguments")
    init_mod = types.ModuleType("megatron.training.initialize")
    global_vars_mod = types.ModuleType("megatron.training.global_vars")
    global_vars_mod.get_args = lambda: None

    core_pkg = types.ModuleType("megatron.core")
    core_pkg.__path__ = []
    parallel_state_mod = types.ModuleType("megatron.core.parallel_state")

    ns = {}
    exec(compile(_FAKE_VALIDATE_SOURCE, _FAKE_VALIDATE_FILE, "exec"), ns)
    fake_validate = ns["validate_args"]
    fake_validate.__globals__.update({"__builtins__": __builtins__})

    args_mod.validate_args = fake_validate
    init_mod.validate_args = fake_validate

    training_pkg.arguments = args_mod
    training_pkg.initialize = init_mod
    training_pkg.global_vars = global_vars_mod
    megatron_mod.training = training_pkg
    megatron_mod.core = core_pkg
    core_pkg.parallel_state = parallel_state_mod

    fake_modules = {
        "megatron": megatron_mod,
        "megatron.training": training_pkg,
        "megatron.training.arguments": args_mod,
        "megatron.training.initialize": init_mod,
        "megatron.training.global_vars": global_vars_mod,
        "megatron.core": core_pkg,
        "megatron.core.parallel_state": parallel_state_mod,
    }
    for name, mod in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, mod)

    return args_mod, init_mod


def _silence_logging(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "primus.backends.megatron.patches.args.validate_args_patches.log_rank_0",
        lambda *a, **k: None,
    )


def _make_ctx():
    return PatchContext(backend="megatron", phase="before_train")


def _make_args(**overrides):
    """Build a minimal args namespace that satisfies both validate_args and validate_args_on_rocm."""
    defaults = dict(
        deterministic_mode=False,
        fp8=False,
        fp4=False,
        fp8_recipe=None,
        fp4_recipe=None,
        use_turbo_parallel_linear=False,
        dump_pp_data=False,
        pipeline_model_parallel_size=1,
        turbo_sync_free_moe_stage=0,
        enable_primus_turbo=False,
        moe_use_legacy_grouped_gemm=False,
        use_turbo_deepep=False,
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
        decoder_pipeline_manual_split_list=None,
        _validate_args_called=False,
        _entered_pipeline_split_block=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaseValidateArgsPatch:
    """Tests for the unconditional ``patch_validate_args`` wrapper."""

    def test_wraps_validate_args_on_both_modules(self, monkeypatch):
        args_mod, init_mod = _install_fake_megatron(monkeypatch)
        _silence_logging(monkeypatch)
        monkeypatch.setattr(
            "primus.modules.trainer.megatron.utils.validate_args_on_rocm",
            lambda args: setattr(args, "_rocm_validated", True),
        )

        from primus.backends.megatron.patches.args.validate_args_patches import (
            patch_validate_args,
        )

        original = args_mod.validate_args
        patch_validate_args(_make_ctx())

        assert args_mod.validate_args is not original
        assert init_mod.validate_args is args_mod.validate_args

    def test_calls_original_and_rocm_validation(self, monkeypatch):
        _install_fake_megatron(monkeypatch)
        _silence_logging(monkeypatch)
        monkeypatch.setattr(
            "primus.modules.trainer.megatron.utils.validate_args_on_rocm",
            lambda args: setattr(args, "_rocm_validated", True),
        )

        import megatron.training.arguments as megatron_args

        from primus.backends.megatron.patches.args.validate_args_patches import (
            patch_validate_args,
        )

        patch_validate_args(_make_ctx())

        args = _make_args()
        result = megatron_args.validate_args(args)

        assert args._validate_args_called is True
        assert args._rocm_validated is True
        assert result is args

    def test_stores_original_on_module(self, monkeypatch):
        args_mod, _ = _install_fake_megatron(monkeypatch)
        _silence_logging(monkeypatch)
        monkeypatch.setattr(
            "primus.modules.trainer.megatron.utils.validate_args_on_rocm",
            lambda args: None,
        )

        original = args_mod.validate_args

        from primus.backends.megatron.patches.args.validate_args_patches import (
            patch_validate_args,
        )

        patch_validate_args(_make_ctx())

        assert args_mod._primus_original_validate_args is original


class TestPipelineSplitPatch:
    """Tests for ``patch_validate_args_pipeline_split``."""

    def _apply_base_and_split(self, monkeypatch):
        _install_fake_megatron(monkeypatch)
        _silence_logging(monkeypatch)
        monkeypatch.setattr(
            "primus.modules.trainer.megatron.utils.validate_args_on_rocm",
            lambda args: None,
        )

        from primus.backends.megatron.patches.args.validate_args_patches import (
            patch_validate_args,
            patch_validate_args_pipeline_split,
        )

        ctx = _make_ctx()
        patch_validate_args(ctx)
        patch_validate_args_pipeline_split(ctx)

    def test_skips_pipeline_block_when_manual_split_set(self, monkeypatch):
        self._apply_base_and_split(monkeypatch)
        import megatron.training.arguments as megatron_args

        args = _make_args(decoder_pipeline_manual_split_list=[4, 4])
        megatron_args.validate_args(args)

        assert args._entered_pipeline_split_block is False
        assert args._validate_args_called is True

    def test_enters_pipeline_block_when_manual_split_none(self, monkeypatch):
        self._apply_base_and_split(monkeypatch)
        import megatron.training.arguments as megatron_args

        args = _make_args(decoder_pipeline_manual_split_list=None)
        megatron_args.validate_args(args)

        assert args._entered_pipeline_split_block is True


class TestFp4Patch:
    """Tests for ``patch_validate_args_fp4``."""

    def _apply_base_and_fp4(self, monkeypatch):
        _install_fake_megatron(monkeypatch)
        _silence_logging(monkeypatch)
        monkeypatch.setattr(
            "primus.modules.trainer.megatron.utils.validate_args_on_rocm",
            lambda args: None,
        )

        from primus.backends.megatron.patches.args.validate_args_patches import (
            patch_validate_args,
            patch_validate_args_fp4,
        )

        ctx = _make_ctx()
        patch_validate_args(ctx)
        patch_validate_args_fp4(ctx)

    def test_fp4_no_longer_raises(self, monkeypatch):
        self._apply_base_and_fp4(monkeypatch)
        import megatron.training.arguments as megatron_args

        args = _make_args(fp4=True)
        megatron_args.validate_args(args)

        assert args._validate_args_called is True

    def test_fp4_false_still_works(self, monkeypatch):
        self._apply_base_and_fp4(monkeypatch)
        import megatron.training.arguments as megatron_args

        args = _make_args(fp4=False)
        megatron_args.validate_args(args)

        assert args._validate_args_called is True

    def test_without_patch_fp4_raises(self, monkeypatch):
        """Baseline: without the fp4 patch, fp4=True raises ValueError."""
        _install_fake_megatron(monkeypatch)
        _silence_logging(monkeypatch)
        monkeypatch.setattr(
            "primus.modules.trainer.megatron.utils.validate_args_on_rocm",
            lambda args: None,
        )

        from primus.backends.megatron.patches.args.validate_args_patches import (
            patch_validate_args,
        )

        patch_validate_args(_make_ctx())

        import megatron.training.arguments as megatron_args

        args = _make_args(fp4=True)
        with pytest.raises(ValueError, match="--fp4-format requires Transformer Engine"):
            megatron_args.validate_args(args)


class TestEndToEnd:
    """Verify that base + source-mod patches compose correctly."""

    def test_pipeline_split_and_rocm_validation(self, monkeypatch):
        _install_fake_megatron(monkeypatch)
        _silence_logging(monkeypatch)

        rocm_calls = []
        monkeypatch.setattr(
            "primus.modules.trainer.megatron.utils.validate_args_on_rocm",
            lambda args: rocm_calls.append(True),
        )

        from primus.backends.megatron.patches.args.validate_args_patches import (
            patch_validate_args,
            patch_validate_args_pipeline_split,
        )

        ctx = _make_ctx()
        patch_validate_args(ctx)
        patch_validate_args_pipeline_split(ctx)

        import megatron.training.arguments as megatron_args

        args = _make_args(decoder_pipeline_manual_split_list=[4, 4])
        result = megatron_args.validate_args(args)

        assert args._entered_pipeline_split_block is False
        assert args._validate_args_called is True
        assert len(rocm_calls) == 1
        assert result is args
