# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests for the shared evaluation budget derivation.

The numbers here are the MLPerf Flux shape: 29,696 validation samples,
global batch 512, micro batch 64, DP 8. Both shipped recipes under-read that
set today (27,776 at num_workers=16, 28,928 at num_workers=8), so the
divisibility rule is asserted against those exact configurations.
"""

import json
from types import SimpleNamespace

import pytest

from primus.backends.megatron.training.eval_budget import (
    DEFAULT_VAL_NUM_WORKERS,
    EvalCoverageError,
    assert_val_worker_divisibility,
    get_eval_num_microbatches,
    get_val_num_workers,
    read_energon_split_sample_count,
    resolve_eval_iters,
)

MLPERF_EVAL_SAMPLES = 29696


def _args(**overrides):
    base = dict(
        data_parallel_size=8,
        micro_batch_size=64,
        global_batch_size=512,
        eval_iters=58,
        eval_samples=None,
        val_num_workers=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestEvalNumMicrobatches:
    def test_mlperf_shape(self):
        assert get_eval_num_microbatches(_args()) == 1

    def test_multiple_microbatches(self):
        assert get_eval_num_microbatches(_args(global_batch_size=2048)) == 4

    def test_zero_microbatches_is_an_error_not_a_silent_noop(self):
        """global_batch < one microbatch per rank floored to 0 and evaluated nothing."""
        with pytest.raises(EvalCoverageError, match="smaller than one microbatch"):
            get_eval_num_microbatches(_args(global_batch_size=256))

    def test_indivisible_global_batch_is_an_error(self):
        with pytest.raises(EvalCoverageError, match="not divisible"):
            get_eval_num_microbatches(_args(global_batch_size=768))


class TestValNumWorkers:
    def test_defaults_to_zero_not_to_training_num_workers(self):
        args = _args(val_num_workers=None)
        args.num_workers = 16
        assert get_val_num_workers(args) == DEFAULT_VAL_NUM_WORKERS == 0

    def test_explicit_value_is_used(self):
        assert get_val_num_workers(_args(val_num_workers=2)) == 2

    def test_negative_rejected(self):
        with pytest.raises(EvalCoverageError, match="must be >= 0"):
            get_val_num_workers(_args(val_num_workers=-1))


class TestWorkerDivisibility:
    @pytest.mark.parametrize("workers", [0, 1, 2, 29, 58])
    def test_accepts_the_shapes_that_cover_the_set(self, workers):
        assert_val_worker_divisibility(_args(val_num_workers=workers), MLPERF_EVAL_SAMPLES)

    @pytest.mark.parametrize(
        "workers, observed",
        [
            (16, 27776),  # local_spec recipe
            (8, 28928),  # te_spec recipe and the MXFP6 convergence config
        ],
    )
    def test_rejects_the_shipped_recipe_worker_counts(self, workers, observed):
        with pytest.raises(EvalCoverageError, match="silently read fewer") as excinfo:
            assert_val_worker_divisibility(_args(val_num_workers=workers), MLPERF_EVAL_SAMPLES)

        message = str(excinfo.value)
        assert f"val_num_workers       = {workers}" in message
        # The error must offer a way out, not just refuse.
        assert "Valid val_num_workers for this shape: [0, 1, 2, 29, 58]" in message

    def test_default_worker_count_does_not_divide_by_zero(self):
        """max(1, val_num_workers) is what keeps the default of 0 usable."""
        assert_val_worker_divisibility(_args(val_num_workers=0), MLPERF_EVAL_SAMPLES)


class TestResolveEvalIters:
    def test_null_eval_samples_leaves_eval_iters_alone(self):
        assert resolve_eval_iters(_args(eval_samples=None)) is None

    def test_derives_from_samples(self):
        args = _args(eval_samples=MLPERF_EVAL_SAMPLES, eval_iters=0)
        assert resolve_eval_iters(args) == 58

    def test_agreeing_eval_iters_is_accepted(self):
        args = _args(eval_samples=MLPERF_EVAL_SAMPLES, eval_iters=58)
        assert resolve_eval_iters(args) == 58

    def test_eval_samples_overrides_an_inherited_eval_iters(self):
        """eval_samples wins rather than conflicting.

        trainer_base.yaml gives every module an eval_iters, and Megatron's
        parser defaults it too, so a recipe opting into eval_samples always
        carries some inherited eval_iters as well. Treating that as a conflict
        would reject every such recipe at startup.
        """
        args = _args(eval_samples=MLPERF_EVAL_SAMPLES, eval_iters=10)
        assert resolve_eval_iters(args) == 58

    def test_override_holds_for_the_trainer_base_default(self):
        args = _args(eval_samples=MLPERF_EVAL_SAMPLES, eval_iters=32)
        assert resolve_eval_iters(args) == 58

    def test_indivisible_sample_count_reports_nearest_reachable(self):
        args = _args(eval_samples=MLPERF_EVAL_SAMPLES, eval_iters=0, global_batch_size=768)
        with pytest.raises(EvalCoverageError, match="not divisible by global_batch_size") as excinfo:
            resolve_eval_iters(args)
        assert "29184" in str(excinfo.value) and "29952" in str(excinfo.value)

    def test_non_positive_sample_count_rejected(self):
        with pytest.raises(EvalCoverageError, match="must be positive"):
            resolve_eval_iters(_args(eval_samples=0, eval_iters=0))

    def test_derivation_also_enforces_coverage(self):
        """Deriving a correct iteration count is not enough if workers under-read."""
        args = _args(eval_samples=MLPERF_EVAL_SAMPLES, eval_iters=0, val_num_workers=8)
        with pytest.raises(EvalCoverageError, match="silently read fewer"):
            resolve_eval_iters(args)

    def test_derivation_tracks_global_batch_size(self):
        """The point of eval_samples: the same coverage at a different batch size."""
        args = _args(eval_samples=MLPERF_EVAL_SAMPLES, eval_iters=0, global_batch_size=1024)
        assert resolve_eval_iters(args) == 29


class TestReadEnergonSplitSampleCount:
    """Reading the split size from the dataset's own index.

    This is what lets full_validation mean "all of it" without the count being
    hand-written into a recipe, and gives the evaluation loop an independent
    third number to check against.
    """

    @staticmethod
    def _dataset(tmp_path, shard_counts):
        meta = tmp_path / ".nv-meta"
        meta.mkdir()
        (meta / ".info.json").write_text(
            json.dumps({"energon_version": "7.3.2", "shard_counts": shard_counts})
        )
        return tmp_path

    def test_sums_only_the_requested_split(self, tmp_path):
        path = self._dataset(
            tmp_path,
            {
                "train/shard_000000.tar": 231,
                "train/shard_000001.tar": 230,
                "val/shard_000000.tar": 231,
                "val/shard_000001.tar": 230,
            },
        )

        assert read_energon_split_sample_count(path, "val") == 461
        assert read_energon_split_sample_count(path, "train") == 461

    def test_mlperf_shape(self, tmp_path):
        """26 shards of 231 and 103 of 230 is the published val split."""
        counts = {f"val/shard_{i:06d}.tar": (231 if i < 26 else 230) for i in range(129)}
        path = self._dataset(tmp_path, counts)

        assert read_energon_split_sample_count(path, "val") == MLPERF_EVAL_SAMPLES

    def test_accepts_list_form_data_path(self, tmp_path):
        path = self._dataset(tmp_path, {"val/shard_000000.tar": 7})
        assert read_energon_split_sample_count([str(path)]) == 7

    @pytest.mark.parametrize("value", [None, [], ""])
    def test_empty_inputs_return_none(self, value):
        assert read_energon_split_sample_count(value) is None

    def test_missing_dataset_returns_none_rather_than_raising(self, tmp_path):
        """Mock data and unprepared directories must fall back, not fail."""
        assert read_energon_split_sample_count(tmp_path / "nope") is None

    def test_malformed_index_returns_none(self, tmp_path):
        meta = tmp_path / ".nv-meta"
        meta.mkdir()
        (meta / ".info.json").write_text("not json")

        assert read_energon_split_sample_count(tmp_path) is None

    def test_absent_split_returns_none(self, tmp_path):
        path = self._dataset(tmp_path, {"train/shard_000000.tar": 231})
        assert read_energon_split_sample_count(path, "val") is None
