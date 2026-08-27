###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the build_args patch that derives eval_iters from eval_samples.

The case these exist for: ``eval_samples`` and ``val_num_workers`` are not
Megatron arguments, and MegatronArgBuilder drops every key Megatron's parser
does not define. The runtime does merge the leftover Primus-only params into
backend_args, but only *after* the build_args phase has run. So a patch in this
phase that reads them straight off args sees them as unset and silently leaves
eval_iters alone -- the exact failure these tests pin, since it turns the
coverage fix into a no-op while still looking like a working config.
"""

from types import SimpleNamespace
from unittest.mock import patch as mock_patch

import pytest

from primus.backends.megatron.patches.args import eval_samples_patches
from primus.backends.megatron.patches.args.eval_samples_patches import (
    patch_eval_samples,
)
from primus.backends.megatron.training.eval_budget import EvalCoverageError
from primus.core.patches import PatchContext

MLPERF_EVAL_SAMPLES = 29696


@pytest.fixture(autouse=True)
def _silence_patch_logging():
    """The Primus logger is not initialised outside a real run."""
    with mock_patch.object(eval_samples_patches, "log_kv_rank_0", lambda *a, **k: None):
        yield


def _megatron_args(**overrides):
    """backend_args as MegatronArgBuilder produces them: no Primus-only keys."""
    base = dict(
        data_parallel_size=8,
        micro_batch_size=64,
        global_batch_size=512,
        eval_iters=32,
        full_validation=False,
        data_path="/does/not/exist",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _ctx(args, **primus_only):
    module_config = SimpleNamespace(params=SimpleNamespace(**primus_only))
    return PatchContext(
        backend="megatron",
        phase="build_args",
        extra={"backend_args": args, "module_config": module_config},
    )


class TestPrimusOnlyKeysReachThePatch:
    def test_eval_samples_from_module_config_derives_eval_iters(self):
        """The regression: eval_samples lives only on the module config here."""
        args = _megatron_args()

        patch_eval_samples(_ctx(args, eval_samples=MLPERF_EVAL_SAMPLES, val_num_workers=0))

        assert args.eval_iters == 58, "eval_samples on the module config must still derive eval_iters"

    def test_val_num_workers_from_module_config_is_enforced(self):
        """A worker count that cannot read the whole set must still be caught."""
        args = _megatron_args()

        with pytest.raises(EvalCoverageError):
            patch_eval_samples(_ctx(args, eval_samples=MLPERF_EVAL_SAMPLES, val_num_workers=8))

    def test_args_value_wins_over_module_config(self):
        """If a key already reached args, do not overwrite it from the config."""
        args = _megatron_args(eval_samples=MLPERF_EVAL_SAMPLES, val_num_workers=0)

        patch_eval_samples(_ctx(args, eval_samples=512, val_num_workers=0))

        assert args.eval_samples == MLPERF_EVAL_SAMPLES
        assert args.eval_iters == 58

    def test_missing_module_config_does_not_crash(self):
        args = _megatron_args(eval_iters=58)
        ctx = PatchContext(
            backend="megatron",
            phase="build_args",
            extra={"backend_args": args},
        )

        patch_eval_samples(ctx)

        assert args.eval_iters == 58

    def test_no_backend_args_is_a_no_op(self):
        patch_eval_samples(PatchContext(backend="megatron", phase="build_args", extra={}))


class TestEvalItersPath:
    def test_plain_eval_iters_still_gets_a_coverage_check(self):
        """Configs that never opt into eval_samples are still covered."""
        args = _megatron_args(eval_iters=58)

        with pytest.raises(EvalCoverageError):
            patch_eval_samples(_ctx(args, val_num_workers=8))

    def test_reachable_eval_iters_shape_is_left_alone(self):
        args = _megatron_args(eval_iters=58)

        patch_eval_samples(_ctx(args, val_num_workers=0))

        assert args.eval_iters == 58

    def test_zero_eval_iters_is_not_checked(self):
        """eval_iters 0 means "no validation"; there is nothing to cover."""
        args = _megatron_args(eval_iters=0)

        patch_eval_samples(_ctx(args, val_num_workers=8))

        assert args.eval_iters == 0


class TestFullValidation:
    def test_full_validation_reads_the_split_size(self):
        args = _megatron_args(full_validation=True)

        with mock_patch.object(
            eval_samples_patches,
            "read_energon_split_sample_count",
            return_value=MLPERF_EVAL_SAMPLES,
        ):
            patch_eval_samples(_ctx(args, val_num_workers=0))

        assert args.eval_samples == MLPERF_EVAL_SAMPLES
        assert args.eval_iters == 58

    def test_full_validation_without_a_readable_index_errors(self):
        args = _megatron_args(full_validation=True)

        with mock_patch.object(eval_samples_patches, "read_energon_split_sample_count", return_value=None):
            with pytest.raises(ValueError, match="could not be read"):
                patch_eval_samples(_ctx(args, val_num_workers=0))

    def test_explicit_eval_samples_takes_precedence(self):
        """full_validation must not re-read the index when a count is given."""
        args = _megatron_args(full_validation=True)

        with mock_patch.object(eval_samples_patches, "read_energon_split_sample_count") as reader:
            patch_eval_samples(_ctx(args, eval_samples=MLPERF_EVAL_SAMPLES, val_num_workers=0))

        reader.assert_not_called()
        assert args.eval_iters == 58
