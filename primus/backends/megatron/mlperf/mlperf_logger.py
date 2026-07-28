###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MLPERF Logging support for Primus Megatron backend."""

import os
import time
from typing import Any, Dict, Optional


class MLPerfLogger:
    """Wrapper around mllog library with rank-aware logging."""

    def __init__(self):
        from mlperf_logging import mllog

        self.mllogger = mllog.get_mllogger()
        self._configured = False
        self._rank = None
        self._save_to_file = os.getenv("MLLOG_SAVE_TO_FILE", "1").lower() not in ("0", "false", "no")

    def configure(self, filepath: str, args) -> Dict[str, Any]:
        from mlperf_logging import mllog

        if self._save_to_file:
            mllog.config(filename=filepath, default_stack_offset=3)
        else:
            mllog.config(default_stack_offset=3)
        self._configured = True
        self._rank = self._get_rank()
        return self.extract_mlperf_configs(args)

    def _get_rank(self) -> int:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return int(os.environ.get("RANK", 0))

    def log_event_all_ranks(self, key: str, value: Any, metadata: Optional[Dict] = None):
        self.mllogger.event(key=key, value=value, metadata=metadata or {})

    def log_event(self, key: str, value: Any, metadata: Optional[Dict] = None, stack_offset: int = 2):
        if self._rank == 0:
            self.mllogger.event(key=key, value=value, metadata=metadata or {}, stack_offset=stack_offset)

    def log_start(self, key: str, metadata: Optional[Dict] = None, stack_offset: int = 2):
        if self._rank == 0:
            self.mllogger.start(key=key, metadata=metadata or {}, stack_offset=stack_offset)

    def log_end(self, key: str, metadata: Optional[Dict] = None, stack_offset: int = 2):
        if self._rank == 0:
            self.mllogger.end(key=key, metadata=metadata or {}, stack_offset=stack_offset)

    def extract_mlperf_configs(self, args) -> Dict[str, Any]:
        """Extract MLPERF config parameters from Megatron args."""
        from mlperf_logging.mllog import constants

        data_parallel_size = getattr(args, "data_parallel_size", 1)
        if data_parallel_size == 0:
            data_parallel_size = 1
        micro_batches = args.global_batch_size // (args.micro_batch_size * data_parallel_size)

        eval_samples = args.eval_iters * args.global_batch_size if args.eval_iters > 0 else 1024

        train_samples = getattr(args, "train_samples", None)
        if not train_samples:
            train_samples = args.train_iters * args.global_batch_size

        optimizer_name = getattr(args, "optimizer", "adam")
        if optimizer_name.lower() == "adam":
            optimizer_name = "adamw"
        else:
            optimizer_name = optimizer_name.lower()

        configs = {
            constants.GLOBAL_BATCH_SIZE: args.global_batch_size,
            constants.GRADIENT_ACCUMULATION_STEPS: micro_batches,
            "max_sequence_length": args.seq_length,
            constants.TRAIN_SAMPLES: train_samples,
            constants.EVAL_SAMPLES: eval_samples,
            constants.SEED: args.seed,
            "init_checkpoint_step": args.iteration,
            constants.OPT_NAME: optimizer_name,
            constants.OPT_BASE_LR: args.lr,
            constants.OPT_ADAMW_BETA_1: getattr(args, "adam_beta1", 0.9),
            constants.OPT_ADAMW_BETA_2: getattr(args, "adam_beta2", 0.999),
            constants.OPT_ADAMW_EPSILON: getattr(args, "adam_eps", 1e-8),
            constants.OPT_ADAMW_WEIGHT_DECAY: args.weight_decay,
            "opt_gradient_clip_norm": getattr(args, "clip_grad", 1.0),
            "opt_end_learning_rate": args.min_lr,
            "opt_learning_rate_warmup_steps": getattr(args, "lr_warmup_iters", 0),
            "opt_learning_rate_decay_steps": args.lr_decay_iters if args.lr_decay_iters else args.train_iters,
            "opt_learning_rate_decay_schedule": self._get_lr_schedule_name(args.lr_decay_style),
            "max_steps": args.train_iters,
            constants.SUBMISSION_BENCHMARK: os.getenv("MLLOG_SUBMISSION_BENCHMARK", ""),
            constants.SUBMISSION_DIVISION: os.getenv("MLLOG_SUBMISSION_DIVISION", ""),
            constants.SUBMISSION_STATUS: os.getenv("MLLOG_SUBMISSION_STATUS", ""),
            constants.SUBMISSION_ORG: os.getenv("MLLOG_SUBMISSION_ORG", ""),
            constants.SUBMISSION_PLATFORM: os.getenv("MLLOG_SUBMISSION_PLATFORM", ""),
            constants.TENSOR_PARALLELISM: int(os.getenv("MLLOG_TENSOR_PARALLELISM", 1)),
            constants.PIPELINE_PARALLELISM: int(os.getenv("MLLOG_PIPELINE_PARALLELISM", 1)),
            constants.CONTEXT_PARALLELISM: int(os.getenv("MLLOG_CONTEXT_PARALLELISM", 1)),
            constants.EXPERT_PARALLELISM: int(os.getenv("MLLOG_EXPERT_PARALLELISM", 1)),
            constants.MICRO_BATCH_SIZE: int(os.getenv("MLLOG_MICRO_BATCH_SIZE", 1)),
            constants.CONFIG_FILENAME: os.getenv("MLLOG_CONFIG_FILENAME", ""),
            "lowest_numerical_precision_linear": os.getenv("MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR", ""),
        }
        return configs

    def _get_lr_schedule_name(self, style: str) -> str:
        mapping = {
            "cosine": "cosine with linear warmup",
            "linear": "linear",
            "constant": "constant",
        }
        return mapping.get(style, style)


class ThroughputTimer:
    """Per-step accumulation timer for training throughput (samples/sec).

    Measures only the time inside each training step, excluding eval time
    and inter-step overhead (dataloader, callbacks, Python gaps).  This
    matches the approach used by nemo's Timer class.
    """

    def __init__(self, global_batch_size: int):
        self.gbs = global_batch_size
        self._step_start_time = None
        self._accumulated_time = 0.0
        self._accumulated_samples = 0

    def step_start(self):
        """Mark the beginning of a training step."""
        self._step_start_time = time.time()

    def step_stop(self):
        """Mark the end of a training step and accumulate elapsed time."""
        if self._step_start_time is not None:
            self._accumulated_time += time.time() - self._step_start_time
            self._accumulated_samples += self.gbs
            self._step_start_time = None

    def get_throughput(self) -> Optional[float]:
        """Return throughput for the accumulated steps and reset counters.

        Returns:
            Samples per second, or None if no steps were accumulated.
        """
        if self._accumulated_time <= 0:
            return None
        throughput = self._accumulated_samples / self._accumulated_time
        self._accumulated_time = 0.0
        self._accumulated_samples = 0
        return throughput
