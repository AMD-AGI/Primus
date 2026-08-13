from __future__ import annotations

from types import SimpleNamespace

import torch

from primus.backends.diffusion.trainers.base import (
    BaseWanTrainer,
    ContiguousDistributedSampler,
)


class RecordingLogger:
    def __init__(self):
        self.records = []

    def event(self, **kwargs):
        self.records.append(("event", kwargs))

    def start(self, **kwargs):
        self.records.append(("start", kwargs))

    def end(self, **kwargs):
        self.records.append(("end", kwargs))


def _trainer() -> BaseWanTrainer:
    trainer = BaseWanTrainer.__new__(BaseWanTrainer)
    trainer.mlperf_enabled = True
    trainer.rank = 0
    trainer.mlperf_constants = SimpleNamespace(
        CACHE_CLEAR="cache_clear",
        SUBMISSION_BENCHMARK="submission_benchmark",
        SUBMISSION_DIVISION="submission_division",
        SUBMISSION_ORG="submission_org",
        SUBMISSION_PLATFORM="submission_platform",
        SUBMISSION_STATUS="submission_status",
        TRAIN_SAMPLES="train_samples",
        EVAL_SAMPLES="eval_samples",
        SEED="seed",
        GLOBAL_BATCH_SIZE="global_batch_size",
        GRADIENT_ACCUMULATION_STEPS="gradient_accumulation_steps",
        OPT_NAME="opt_name",
        ADAMW="adamw",
        OPT_LR_WARMUP_STEPS="opt_learning_rate_warmup_steps",
        OPT_ADAMW_BETA_1="opt_adamw_beta_1",
        OPT_ADAMW_BETA_2="opt_adamw_beta_2",
        OPT_ADAMW_EPSILON="opt_adamw_epsilon",
        OPT_ADAMW_WEIGHT_DECAY="opt_adamw_weight_decay",
        OPT_BASE_LR="opt_base_learning_rate",
        OPT_GRADIENT_CLIP_NORM="opt_gradient_clip_norm",
        INIT_START="init_start",
        INIT_STOP="init_stop",
        RUN_START="run_start",
        BLOCK_START="block_start",
        BLOCK_STOP="block_stop",
        EVAL_START="eval_start",
        EVAL_ACCURACY="eval_accuracy",
        EVAL_STOP="eval_stop",
        RUN_STOP="run_stop",
        SAMPLES_COUNT="samples_count",
        STATUS="status",
        SUCCESS="success",
        ABORTED="aborted",
    )
    trainer.mlperf_logger = RecordingLogger()
    trainer.args = {
        "learning_rate": 2.0e-4,
        "warmup_steps": 1600,
        "seed": 10007,
        "mlperf_train_samples": 1099776,
        "mlperf_eval_total_samples": 29696,
    }
    trainer.optimizer = SimpleNamespace(
        param_groups=[
            {
                "lr": 0.0,
                "betas": (0.9, 0.95),
                "eps": 1.0e-8,
                "weight_decay": 0.1,
            }
        ]
    )
    trainer.max_grad_norm = 1.0
    trainer.mlperf_target_eval_loss = 0.586
    trainer.mlperf_eval_samples = 262144
    trainer.per_device_train_batch_size = 64
    trainer.grad_accum_steps = 1
    trainer.data_parallel_world_size = 8
    trainer.logging_steps = 10
    trainer.global_step = 512
    trainer.mlperf_run_success = False
    return trainer


def test_mlperf_logs_configured_base_lr_before_warmup():
    trainer = _trainer()
    trainer._mlperf_log_run_start()

    base_lr = [
        record["value"]
        for kind, record in trainer.mlperf_logger.records
        if kind == "event" and record["key"] == "opt_base_learning_rate"
    ]
    assert base_lr == [2.0e-4]
    assert trainer.mlperf_logger.records[-1] == ("start", {"key": "init_start"})


def test_mlperf_cache_clear_matches_launcher(monkeypatch):
    monkeypatch.setenv("MLPERF_CLEAR_CACHES", "false")
    trainer = _trainer()
    trainer._mlperf_log_run_start()

    cache_clear = next(
        record
        for kind, record in trainer.mlperf_logger.records
        if kind == "event" and record["key"] == "cache_clear"
    )
    assert cache_clear["value"] is False


def test_mlperf_eval_events_bracket_validation():
    trainer = _trainer()
    trainer._mlperf_log_eval_start()
    trainer._mlperf_log_eval_stop(0.585)

    assert [record[1]["key"] for record in trainer.mlperf_logger.records] == [
        "eval_start",
        "eval_accuracy",
        "eval_stop",
    ]
    assert [record[0] for record in trainer.mlperf_logger.records] == ["event", "event", "end"]
    assert all(record[1]["metadata"]["samples_count"] == 262144 for record in trainer.mlperf_logger.records)


def test_mlperf_training_blocks_are_paired_at_reference_frequency():
    trainer = _trainer()
    trainer._mlperf_log_block_start(1)
    trainer._mlperf_log_block_stop(1)
    trainer._mlperf_log_block_start(2)
    trainer._mlperf_log_block_stop(2)

    assert [record[1]["key"] for record in trainer.mlperf_logger.records] == [
        "block_start",
        "block_stop",
    ]


def test_rank_offset_rng_is_reproducible_and_distinct():
    from primus.backends.diffusion.utils.train_utils import set_seed

    set_seed(10007)
    rank_zero_first = torch.rand(4)
    set_seed(10008)
    rank_one = torch.rand(4)
    set_seed(10007)
    rank_zero_second = torch.rand(4)

    torch.testing.assert_close(rank_zero_first, rank_zero_second)
    assert not torch.equal(rank_zero_first, rank_one)


def test_mlperf_sampler_matches_torchtitan_contiguous_shards():
    dataset = list(range(16))
    rank_zero = ContiguousDistributedSampler(dataset, num_replicas=4, rank=0)
    rank_two = ContiguousDistributedSampler(dataset, num_replicas=4, rank=2)

    assert list(rank_zero) == [0, 1, 2, 3]
    assert list(rank_two) == [8, 9, 10, 11]
