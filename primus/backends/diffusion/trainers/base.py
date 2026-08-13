###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Base trainer with shared logic for Wan PyTorch trainers.

This module holds functionality shared across FSDP-style Wan trainers:
config parsing, optimizer creation, LR scheduling, training loop, logging,
and W&B integration. Concrete trainers (e.g. FSDP2Trainer) subclass it.
"""

from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager

import torch
from torch.utils.data import Sampler

from primus.backends.diffusion.optim.adamw_fp32_state import AdamWFP32State
from primus.backends.diffusion.schedulers.flow_match import FlowMatchScheduler
from primus.backends.diffusion.utils.log import logger
from primus.backends.diffusion.utils.train_utils import (
    get_memory,
    resolve_dtype,
    set_seed,
)

try:
    import wandb
except ImportError:
    wandb = None


def create_lr_scheduler(optimizer, scheduler_type, warmup_steps, total_steps):
    """
    Create LR scheduler with warmup support.

    Supports: constant, constant_with_warmup, linear, cosine, polynomial.
    Shared between FSDP and FSDP2 trainers for consistency.
    """
    if total_steps <= 0:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    warmup_steps = int(warmup_steps)
    if warmup_steps > total_steps:
        logger.warning(
            f"Warmup steps ({warmup_steps}) exceed total steps ({total_steps}). "
            f"Adjusting warmup steps to {total_steps}."
        )
        warmup_steps = total_steps

    def linear_warmup(step):
        if warmup_steps == 0:
            return 1.0
        return min(1.0, float(step) / float(max(1, warmup_steps)))

    def linear_decay(step):
        if step <= warmup_steps:
            return linear_warmup(step)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    def cosine_decay(step):
        if step <= warmup_steps:
            return linear_warmup(step)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def constant_with_warmup(step):
        return linear_warmup(step)

    def polynomial_decay(step, power=1.0):
        if step <= warmup_steps:
            return linear_warmup(step)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, (1.0 - progress) ** power)

    scheduler_type = (scheduler_type or "constant").lower()
    lambdas = {
        "constant": lambda step: 1.0,
        "constant_with_warmup": constant_with_warmup,
        "linear": linear_decay,
        "cosine": cosine_decay,
        "cosine_with_restarts": cosine_decay,
        "polynomial": polynomial_decay,
    }
    lr_lambda = lambdas.get(scheduler_type)
    if lr_lambda is None:
        logger.warning(
            f"Unknown lr_scheduler_type={scheduler_type}, falling back to constant."
        )

        def lr_lambda(step):
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ContiguousDistributedSampler(Sampler[int]):
    """TorchTitan-compatible contiguous, non-padding map-style dataset shard."""

    def __init__(self, dataset, *, num_replicas: int, rank: int):
        if len(dataset) % num_replicas != 0:
            raise ValueError(
                f"Dataset size {len(dataset)} must be divisible by DP world size {num_replicas} "
                "for exact contiguous MLPerf sharding."
            )
        self.start = rank * (len(dataset) // num_replicas)
        self.stop = self.start + len(dataset) // num_replicas
        self.offset = 0

    def __iter__(self):
        return iter(range(self.start + self.offset, self.stop))

    def __len__(self) -> int:
        return self.stop - self.start - self.offset

    def set_epoch(self, epoch: int) -> None:
        del epoch

    def set_offset(self, offset: int) -> None:
        if not 0 <= offset <= self.stop - self.start:
            raise ValueError(f"Sampler offset out of range: {offset}")
        self.offset = offset


class BaseWanTrainer:
    """
    Shared base class for Wan PyTorch FSDP-style trainers.

    Subclasses must implement:
      - _apply_parallelism(): set up distributed wrapping / sharding
      - save_model(): save final model

    Subclasses may override:
      - _grad_sync_context(is_update_step): gradient sync during accumulation
      - _clip_grad_norm(): gradient clipping strategy
      - _save_checkpoint(): periodic checkpoint saving
    """

    def __init__(
        self,
        model: torch.nn.Module,
        args: dict,
        train_dataset,
        data_collator,
        processing_class,
        rank: int,
        world_size: int,
        local_rank: int,
        eval_dataset=None,
        eval_processor=None,
    ):
        self.model = model
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")

        # --- Config extraction ---
        self.output_dir = self.args.get("output_dir", "./output")
        self.logging_steps = int(self.args.get("logging_steps", 1))
        self.save_steps = int(self.args.get("save_steps", 0))
        self.max_steps = int(
            self.args.get("max_steps", -1)
            if self.args.get("max_steps") is not None
            else -1
        )
        self.grad_accum_steps = int(self.args.get("gradient_accumulation_steps", 1))
        self.max_grad_norm = float(self.args.get("max_grad_norm", 1.0))
        self.num_train_epochs = int(self.args.get("num_train_epochs", 1))

        if self.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

        # --- Seeding ---
        seed = self.args.get("seed")
        self.base_seed = int(seed) if seed is not None else None
        if seed is not None:
            set_seed(self.base_seed)

        # --- Gradient Checkpointing ---
        if self.args.get("gradient_checkpointing", False):
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable(
                    {"ratio": float(self.args.get("gradient_checkpointing_ratio", 1.0))}
                )
            elif hasattr(self.model, "dit") and hasattr(
                self.model.dit, "gradient_checkpointing"
            ):
                self.model.dit.gradient_checkpointing = True
            if self.rank == 0:
                logger.info(
                    "Gradient checkpointing enabled "
                    f"(ratio={float(self.args.get('gradient_checkpointing_ratio', 1.0)):.3f})"
                )

        # --- W&B ---
        self._setup_wandb()

        # --- Parallelism (subclass hook) ---
        # Subclass sets self.sp_group (Ulysses SP group) if SP is enabled.
        self.sp_group = None
        self._apply_parallelism()

        # --- DataLoader ---
        self.train_dataset = train_dataset
        self.processing_class = processing_class
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.eval_processor = eval_processor or processing_class

        # When SP is enabled, all ranks in the same SP group process the same sample.
        # DistributedSampler should use DP-only rank/size so SP peers get identical data.
        self.sp_size = 1
        dp_world_size = world_size
        dp_rank = rank
        if self.sp_group is not None:
            import torch.distributed as dist

            self.sp_size = dist.get_world_size(self.sp_group)
            dp_world_size = world_size // self.sp_size
            dp_rank = rank // self.sp_size

        self.data_parallel_world_size = dp_world_size
        self.data_parallel_rank = dp_rank
        if self.base_seed is not None:
            set_seed(self.base_seed + dp_rank)
            if self.rank == 0:
                logger.info(
                    f"Training RNG: base_seed={self.base_seed} with distinct DP-rank offsets"
                )
        self.per_device_train_batch_size = int(
            self.args.get("per_device_train_batch_size", 1)
        )
        self.per_device_eval_batch_size = int(
            self.args.get(
                "per_device_eval_batch_size", self.per_device_train_batch_size
            )
        )

        mlperf_mode = bool(self.args.get("mlperf_enable", False))
        if mlperf_mode:
            self.sampler = ContiguousDistributedSampler(
                train_dataset,
                num_replicas=dp_world_size,
                rank=dp_rank,
            )
        else:
            self.sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=dp_world_size,
                rank=dp_rank,
                shuffle=self.args.get("shuffle", True),
            )

        num_workers = int(self.args.get("dataloader_num_workers", 4) or 0)
        self.dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.per_device_train_batch_size,
            sampler=self.sampler,
            num_workers=num_workers,
            collate_fn=data_collator,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        self.eval_dataloader = None
        if self.eval_dataset is not None:
            if mlperf_mode:
                self.eval_sampler = ContiguousDistributedSampler(
                    self.eval_dataset,
                    num_replicas=dp_world_size,
                    rank=dp_rank,
                )
            else:
                self.eval_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.eval_dataset,
                    num_replicas=dp_world_size,
                    rank=dp_rank,
                    shuffle=False,
                )
            self.eval_dataloader = torch.utils.data.DataLoader(
                self.eval_dataset,
                batch_size=self.per_device_eval_batch_size,
                sampler=self.eval_sampler,
                num_workers=num_workers,
                collate_fn=self.eval_dataset.get_collator(),
                pin_memory=True,
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None,
            )

        self.mlperf_enabled = bool(self.args.get("mlperf_enable", False))
        self.mlperf_target_eval_loss = float(
            self.args.get("mlperf_target_eval_loss", 0.586)
        )
        self.mlperf_eval_samples = int(self.args.get("mlperf_eval_samples", 262144))
        self.mlperf_run_success = False
        self.mlperf_logger = None
        self.mlperf_constants = None
        self.mlperf_train_start_time = None
        global_batch_size = (
            self.per_device_train_batch_size * self.grad_accum_steps * dp_world_size
        )
        self.mlperf_eval_freq_steps = max(
            1, math.ceil(self.mlperf_eval_samples / global_batch_size)
        )
        if self.mlperf_enabled and self.eval_dataloader is None:
            raise ValueError("MLPerf Flux training requires `data.eval_dataset_path`.")
        if self.mlperf_enabled:
            expected_train_samples = int(self.args.get("mlperf_train_samples", 1099776))
            actual_train_samples = len(self.train_dataset)
            if actual_train_samples != expected_train_samples:
                raise ValueError(
                    "MLPerf FLUX training requires exactly "
                    f"{expected_train_samples} samples, found {actual_train_samples}."
                )
            expected_eval_samples = int(
                self.args.get("mlperf_eval_total_samples", 29696)
            )
            actual_eval_samples = len(self.eval_dataset)
            if actual_eval_samples != expected_eval_samples:
                raise ValueError(
                    "MLPerf FLUX evaluation requires exactly "
                    f"{expected_eval_samples} samples, found {actual_eval_samples}."
                )
            if actual_eval_samples % dp_world_size != 0:
                raise ValueError(
                    "MLPerf FLUX evaluation dataset size must be divisible by the DP world size "
                    "to avoid DistributedSampler padding and duplicate samples."
                )

        # --- Optimizer ---
        self.optimizer = self._create_optimizer()

        # --- LR Scheduler ---
        steps_per_epoch = math.ceil(
            len(self.dataloader) / max(1, self.grad_accum_steps)
        )
        self.total_steps = (
            self.max_steps
            if self.max_steps > 0
            else self.num_train_epochs * steps_per_epoch
        )
        self.lr_scheduler = create_lr_scheduler(
            self.optimizer,
            self.args.get("lr_scheduler_type", "constant"),
            int(self.args.get("warmup_steps", 0)),
            self.total_steps,
        )
        logger.info("Rank %d initialized LR scheduler", self.rank)

        # --- Diffusion Scheduler (configurable from YAML) ---
        scheduler_cfg = self.args.get("flow_match_scheduler", {}) or {}
        self.scheduler = FlowMatchScheduler(
            shift=float(scheduler_cfg.get("shift", 5)),
            sigma_min=float(scheduler_cfg.get("sigma_min", 0.0)),
            extra_one_step=bool(scheduler_cfg.get("extra_one_step", True)),
        )
        self.scheduler.set_timesteps(
            int(scheduler_cfg.get("num_train_timesteps", 1000)),
            training=True,
        )
        logger.info("Rank %d initialized diffusion scheduler", self.rank)

        self.global_step = 0
        logger.info("Rank %d entering MLPerf logger setup", self.rank)
        self._setup_mlperf()
        logger.info("Rank %d completed MLPerf logger setup", self.rank)

    # ------------------------------------------------------------------ #
    #                       Subclass hooks                                 #
    # ------------------------------------------------------------------ #

    def _apply_parallelism(self):
        """Set up distributed parallelism. Called during __init__."""
        raise NotImplementedError

    @contextmanager
    def _grad_sync_context(self, is_update_step: bool):
        """Context manager for gradient sync control during accumulation."""
        yield

    def _clip_grad_norm(self) -> float | torch.Tensor:
        """Clip gradient norm. Returns the total norm value."""
        if self.max_grad_norm <= 0:
            return 0.0

        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0

        grads = [p.grad for p in parameters]
        norm = torch.nn.utils.get_total_norm(grads, norm_type=2.0, foreach=True)
        dtensor_cls = None
        try:
            from torch.distributed.tensor import DTensor

            dtensor_cls = DTensor
        except (ImportError, RuntimeError) as exc:
            logger.debug(f"Skipping DTensor grad-norm conversion: {exc}")
        if dtensor_cls is not None and isinstance(norm, dtensor_cls):
            norm = norm.full_tensor()

        torch.nn.utils.clip_grads_with_norm_(
            parameters, self.max_grad_norm, norm, foreach=True
        )
        return norm

    def _save_checkpoint(self):
        """Save checkpoint at save_steps intervals. Override for custom strategies."""

    def _start_profiler(self) -> None:
        self._profiler = None
        if not self.args.get("profile", False):
            return
        profile_rank = int(self.args.get("profile_rank", 0))
        if self.rank != profile_rank:
            return

        output_dir = str(
            self.args.get("profile_output_dir") or os.path.join(self.output_dir, "torch_profile")
        )
        os.makedirs(output_dir, exist_ok=True)
        self._profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=int(self.args.get("profile_wait_steps", 10)),
                warmup=int(self.args.get("profile_warmup_steps", 2)),
                active=int(self.args.get("profile_active_steps", 10)),
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                output_dir,
                worker_name=f"rank{self.rank}",
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=bool(self.args.get("profile_with_stack", False)),
        )
        self._profiler.start()
        logger.info("Torch profiler enabled on rank %d: %s", self.rank, output_dir)

    def _step_profiler(self) -> None:
        if self._profiler is not None:
            self._profiler.step()

    def _stop_profiler(self) -> None:
        if self._profiler is not None:
            self._profiler.stop()
            self._profiler = None

    # ------------------------------------------------------------------ #
    #                       Common methods                                 #
    # ------------------------------------------------------------------ #

    def _setup_wandb(self):
        self.use_wandb = False
        if self.rank != 0:
            return
        if str(self.args.get("report_to", "")).lower() != "wandb":
            return
        if self.args.get("use_wandb") is False:
            return
        if wandb is None:
            logger.warning("W&B requested but wandb is not installed.")
            return

        def env(name):
            value = os.environ.get(name, "").strip()
            return value or None

        kwargs = {
            "project": self.args.get("wandb_project")
            or env("WANDB_PROJECT")
            or "mlperf-flux1",
            "name": self.args.get("wandb_name")
            or env("WANDB_RUN_NAME")
            or self.args.get("run_name"),
            "dir": self.args.get("wandb_dir")
            or env("WANDB_SAVE_DIR")
            or env("WANDB_DIR"),
            "entity": env("WANDB_ENTITY"),
            "group": env("WANDB_GROUP"),
            "job_type": env("WANDB_JOB_TYPE"),
            "id": env("WANDB_RUN_ID"),
            "config": self.args,
        }
        tags = env("WANDB_TAGS")
        if tags:
            kwargs["tags"] = [tag.strip() for tag in tags.split(",") if tag.strip()]
        if str(os.environ.get("WANDB_OFFLINE", "0")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            kwargs["mode"] = "offline"
        if kwargs["id"]:
            kwargs["resume"] = os.environ.get("WANDB_RESUME", "allow")
        wandb.init(**{key: value for key, value in kwargs.items() if value is not None})
        wandb.define_metric("validation_metrics/samples_count")
        wandb.define_metric(
            "validation_metrics/loss_vs_samples",
            step_metric="validation_metrics/samples_count",
        )
        self.use_wandb = True

    def _setup_mlperf(self):
        if not self.mlperf_enabled:
            return
        try:
            from mlperf_logging import mllog
            from mlperf_logging.mllog import constants
        except ImportError as exc:
            raise ImportError(
                "Diffusion MLPerf mode requires `mlperf_logging`. "
                "Install Primus MLPerf dependencies before enabling `mlperf.enable`."
            ) from exc

        self.mlperf_logger = mllog.get_mllogger()
        self.mlperf_constants = constants
        if self.rank == 0:
            output_file = (
                self.args.get("mlperf_output_file")
                or os.environ.get("MLLOG_OUTPUT_FILE")
                or os.path.join(self.output_dir, "mlperf_compliance.log")
            )
            mllog.config(filename=output_file, default_stack_offset=3)

    def _global_batch_size(self) -> int:
        return (
            self.per_device_train_batch_size
            * self.grad_accum_steps
            * self.data_parallel_world_size
        )

    def _mlperf_log_run_start(self):
        if not self.mlperf_enabled:
            return
        if self.rank != 0:
            return
        c = self.mlperf_constants
        opt = self.optimizer.param_groups[0]
        cache_cleared = os.getenv("MLPERF_CLEAR_CACHES", "true").strip().lower() == "true"
        self.mlperf_logger.event(key=c.CACHE_CLEAR, value=cache_cleared)
        self.mlperf_logger.event(key=c.SUBMISSION_BENCHMARK, value="flux1")
        self.mlperf_logger.event(
            key=c.SUBMISSION_DIVISION,
            value=os.getenv("MLLOG_SUBMISSION_DIVISION", "closed"),
        )
        self.mlperf_logger.event(
            key=c.SUBMISSION_ORG, value=os.getenv("MLLOG_SUBMISSION_ORG", "reference")
        )
        self.mlperf_logger.event(
            key=c.SUBMISSION_PLATFORM,
            value=os.getenv("MLLOG_SUBMISSION_PLATFORM", "reference"),
        )
        poc_name_key = getattr(c, "SUBMISSION_POC_NAME", None)
        poc_email_key = getattr(c, "SUBMISSION_POC_EMAIL", None)
        if poc_name_key is not None:
            self.mlperf_logger.event(
                key=poc_name_key,
                value=os.getenv("MLLOG_SUBMISSION_POC_NAME", "reference"),
            )
        if poc_email_key is not None:
            self.mlperf_logger.event(
                key=poc_email_key,
                value=os.getenv("MLLOG_SUBMISSION_POC_EMAIL", "reference"),
            )
        self.mlperf_logger.event(
            key=c.SUBMISSION_STATUS,
            value=os.getenv("MLLOG_SUBMISSION_STATUS", "onprem"),
        )
        self.mlperf_logger.event(
            key=c.TRAIN_SAMPLES,
            value=int(self.args.get("mlperf_train_samples", 1099776)),
        )
        self.mlperf_logger.event(
            key=c.EVAL_SAMPLES,
            value=int(self.args.get("mlperf_eval_total_samples", 29696)),
        )
        self.mlperf_logger.event(
            key="target_accuracy", value=self.mlperf_target_eval_loss
        )
        self.mlperf_logger.event(key=c.SEED, value=self.args.get("seed"))
        self.mlperf_logger.event(
            key=c.GLOBAL_BATCH_SIZE, value=self._global_batch_size()
        )
        self.mlperf_logger.event(
            key=c.GRADIENT_ACCUMULATION_STEPS, value=self.grad_accum_steps
        )
        self.mlperf_logger.event(key=c.OPT_NAME, value=c.ADAMW)
        self.mlperf_logger.event(
            key=c.OPT_LR_WARMUP_STEPS, value=int(self.args.get("warmup_steps", 0))
        )
        self.mlperf_logger.event(key=c.OPT_ADAMW_BETA_1, value=opt["betas"][0])
        self.mlperf_logger.event(key=c.OPT_ADAMW_BETA_2, value=opt["betas"][1])
        self.mlperf_logger.event(key=c.OPT_ADAMW_EPSILON, value=opt["eps"])
        self.mlperf_logger.event(
            key=c.OPT_ADAMW_WEIGHT_DECAY, value=opt["weight_decay"]
        )
        self.mlperf_logger.event(
            key=c.OPT_BASE_LR, value=float(self.args["learning_rate"])
        )
        self.mlperf_logger.event(key=c.OPT_GRADIENT_CLIP_NORM, value=self.max_grad_norm)
        self.mlperf_logger.event(
            key="evaluation_frequency", value=self.mlperf_eval_samples
        )
        self.mlperf_logger.start(key=c.INIT_START)

    def _mlperf_log_train_start(self):
        if not self.mlperf_enabled:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.mlperf_train_start_time = time.time()
        if self.rank == 0:
            c = self.mlperf_constants
            self.mlperf_logger.end(key=c.INIT_STOP)
            self.mlperf_logger.start(key=c.RUN_START)

    def _mlperf_log_block_start(self, step: int):
        if not self.mlperf_enabled or self.rank != 0:
            return
        if (step - 1) % self.logging_steps != 0:
            return
        c = self.mlperf_constants
        self.mlperf_logger.start(
            key=c.BLOCK_START,
            value="training_step",
            metadata={c.SAMPLES_COUNT: (step - 1) * self._global_batch_size()},
        )

    def _mlperf_log_block_stop(self, step: int):
        if not self.mlperf_enabled or self.rank != 0:
            return
        if (step - 1) % self.logging_steps != 0:
            return
        c = self.mlperf_constants
        self.mlperf_logger.end(
            key=c.BLOCK_STOP,
            value="training_step",
            metadata={c.SAMPLES_COUNT: (step - 1) * self._global_batch_size()},
        )

    def _mlperf_log_eval_start(self):
        if not self.mlperf_enabled or self.rank != 0:
            return
        c = self.mlperf_constants
        metadata = {c.SAMPLES_COUNT: self.global_step * self._global_batch_size()}
        self.mlperf_logger.event(key=c.EVAL_START, metadata=metadata)

    def _mlperf_log_eval_stop(self, loss: float):
        if not self.mlperf_enabled or self.rank != 0:
            return
        c = self.mlperf_constants
        samples = self.global_step * self._global_batch_size()
        metadata = {c.SAMPLES_COUNT: samples}
        self.mlperf_logger.event(key=c.EVAL_ACCURACY, value=loss, metadata=metadata)
        self.mlperf_logger.end(key=c.EVAL_STOP, value=loss, metadata=metadata)

    def _mlperf_log_run_stop(self):
        if not self.mlperf_enabled or self.rank != 0:
            return
        c = self.mlperf_constants
        samples = self.global_step * self._global_batch_size()
        status = c.SUCCESS if self.mlperf_run_success else c.ABORTED
        self.mlperf_logger.end(
            key=c.RUN_STOP, metadata={c.SAMPLES_COUNT: samples, c.STATUS: status}
        )

    def _resolve_dtype(self) -> torch.dtype:
        return resolve_dtype(self.args)

    def _create_optimizer(self):
        lr = float(self.args.get("learning_rate", 1e-4))
        wd = float(self.args.get("weight_decay", 0.01))
        betas = (
            float(self.args.get("adam_beta1", 0.9)),
            float(self.args.get("adam_beta2", 0.999)),
        )
        eps = float(self.args.get("adam_epsilon", 1e-8))

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer_kwargs = {
            "params": params,
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": wd,
        }

        optimizer = None
        try:
            optimizer = torch.optim.AdamW(**optimizer_kwargs, fused=True)
            if self.rank == 0:
                logger.info("Optimizer: torch.optim.AdamW(fused=True)")
        except (TypeError, RuntimeError):
            try:
                optimizer = torch.optim.AdamW(**optimizer_kwargs, foreach=True)
                if self.rank == 0:
                    logger.info("Optimizer: torch.optim.AdamW(foreach=True)")
            except (TypeError, RuntimeError):
                optimizer = torch.optim.AdamW(**optimizer_kwargs)
                if self.rank == 0:
                    logger.info("Optimizer: torch.optim.AdamW(default)")

        use_custom_master_weights = (
            self.args.get("bf16", False) or self.args.get("fp16", False)
        ) and os.getenv("FP32_MASTER_WEIGHTS", "0") == "1"
        if getattr(self, "mlperf_enabled", False) and use_custom_master_weights:
            raise ValueError(
                "MLPerf FLUX uses TorchTitan-aligned FP32 FSDP parameters with ordinary AdamW; "
                "unset FP32_MASTER_WEIGHTS."
            )
        if use_custom_master_weights:
            if self.rank == 0:
                logger.info(
                    "FP32_MASTER_WEIGHTS=1: using AdamWFP32State (fp32 master weights + fp32 moments)."
                )
            optimizer = AdamWFP32State(
                optimizer.param_groups,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=wd,
            )

        return optimizer

    def _prepare_batch(self, batch, processor):
        prepare_batch = getattr(processor, "prepare_batch", None)
        if callable(prepare_batch):
            batch = prepare_batch(
                batch=batch,
                device=self.device,
                dtype=self._resolve_dtype(),
            )
        # Ensure all tensors are on the correct device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)

        # Pass SP group so model can shard sequences across SP ranks
        if self.sp_group is not None:
            batch["sp_group"] = self.sp_group
        return batch

    def compute_loss(self, batch, processor=None):
        """Prepare batch and compute training loss."""
        batch = self._prepare_batch(batch, processor or self.processing_class)
        # Use explicit training entry point if available (GenAIModel interface)
        forward_train = getattr(self.model, "forward_train", None)
        if callable(forward_train):
            outputs = forward_train(batch, scheduler=self.scheduler)
        else:
            outputs = self.model(batch, self.scheduler)
        return outputs["loss"]

    def validate_loss(self) -> float:
        if self.eval_dataloader is None:
            raise RuntimeError("validate_loss() requires an eval dataloader.")

        was_training = self.model.training
        self.model.eval()
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        count_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        max_steps = int(self.args.get("mlperf_eval_steps", -1))
        if self.mlperf_enabled and max_steps > 0:
            raise ValueError(
                "MLPerf FLUX validation must consume all 29,696 evaluation samples."
            )

        with torch.no_grad():
            for step, batch in enumerate(self.eval_dataloader):
                if max_steps > 0 and step >= max_steps:
                    break
                local_count = self._infer_local_batch_size(batch)
                loss = (
                    self.compute_loss(batch, processor=self.eval_processor)
                    .detach()
                    .float()
                )
                loss_sum += loss * float(local_count)
                count_sum += float(local_count)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_sum)
            torch.distributed.all_reduce(count_sum)

        if was_training:
            self.model.train()
        if count_sum.item() <= 0:
            raise RuntimeError("Validation did not consume any samples.")
        if self.mlperf_enabled:
            expected_samples = int(self.args.get("mlperf_eval_total_samples", 29696))
            actual_samples = int(count_sum.item())
            if actual_samples != expected_samples:
                raise RuntimeError(
                    f"MLPerf FLUX validation consumed {actual_samples} samples; expected {expected_samples}."
                )
        return (loss_sum / count_sum).item()

    def _infer_batch_size_from_tensors(self, value) -> int | None:
        if isinstance(value, torch.Tensor):
            return int(value.shape[0]) if value.ndim > 0 else 1
        if isinstance(value, dict):
            for item in value.values():
                batch_size = self._infer_batch_size_from_tensors(item)
                if batch_size is not None:
                    return batch_size
        if isinstance(value, (list, tuple)):
            for item in value:
                batch_size = self._infer_batch_size_from_tensors(item)
                if batch_size is not None:
                    return batch_size
        return None

    def _infer_batch_size_from_sequences(self, value) -> int | None:
        if isinstance(value, dict):
            for item in value.values():
                batch_size = self._infer_batch_size_from_sequences(item)
                if batch_size is not None:
                    return batch_size
            return None
        if isinstance(value, (list, tuple)):
            if not value:
                return 0
            first = value[0]
            if isinstance(first, (dict, list, tuple, torch.Tensor)):
                for item in value:
                    batch_size = self._infer_batch_size_from_sequences(item)
                    if batch_size is not None:
                        return batch_size
                return None
            return len(value)
        return None

    def _infer_local_batch_size(self, batch) -> int:
        if isinstance(batch, (list, tuple)):
            return len(batch)

        tensor_batch_size = self._infer_batch_size_from_tensors(batch)
        if tensor_batch_size is not None:
            return tensor_batch_size

        sequence_batch_size = self._infer_batch_size_from_sequences(batch)
        if sequence_batch_size is not None:
            return sequence_batch_size

        return self.per_device_train_batch_size

    def _compute_samples_per_gpu_per_second(
        self,
        local_samples: int,
        interval_seconds: float | None,
    ) -> float | None:
        if (
            interval_seconds is None
            or interval_seconds <= 0
            or local_samples <= 0
            or self.world_size <= 0
        ):
            return None

        global_samples = float(local_samples) * float(self.data_parallel_world_size)
        return global_samples / float(self.world_size) / float(interval_seconds)

    def _log_step(
        self,
        loss_value: float | torch.Tensor,
        grad_norm: float | torch.Tensor = 0.0,
        step_time: float | None = None,
        elapsed: float | None = None,
        eta_seconds: float | None = None,
        throughput_samples_per_gpu_s: float | None = None,
    ):
        """Log training metrics. Format matches test regex expectations."""
        if self.rank != 0:
            return
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.item()
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()
        alloc, res, max_mem = get_memory()
        lr = self.optimizer.param_groups[0]["lr"]

        # NOTE: The "step=... loss=... mem=.../...GB" line format is relied on by
        # downstream log parsers; keep it stable when editing.
        msg = (
            f"step={self.global_step} loss={loss_value:.4f} "
            f"mem={alloc:.2f}/{res:.2f}GB peak_mem={max_mem:.2f}GB "
            f"gnorm={grad_norm:.4f}"
        )
        if step_time is not None:
            msg += f" step_time={step_time:.2f}s"
        if throughput_samples_per_gpu_s is not None:
            msg += f" throughput={throughput_samples_per_gpu_s:.4f}samples/gpu/s"
        if elapsed is not None:
            msg += f" elapsed={elapsed / 60:.2f}m"
        if eta_seconds is not None:
            msg += f" eta={eta_seconds / 60:.2f}m"
        logger.info(msg)

        if self.use_wandb:
            global_batch_size = (
                self.per_device_train_batch_size
                * self.grad_accum_steps
                * self.data_parallel_world_size
            )
            payload = {
                "train/loss": loss_value,
                "train/step": self.global_step,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "train/samples_count": self.global_step * global_batch_size,
                "loss_metrics/global_avg_loss": loss_value,
                "lr": lr,
                "mem/allocated_gb": alloc,
                "mem/reserved_gb": res,
                "mem/max_alloc_gb": max_mem,
            }
            if step_time is not None:
                payload["time/step_s"] = step_time
            if throughput_samples_per_gpu_s is not None:
                payload["perf/samples_per_gpu_s"] = throughput_samples_per_gpu_s
                payload["performance/throughput"] = (
                    throughput_samples_per_gpu_s * self.world_size
                )
                payload["throughput(global_samples/s)"] = (
                    throughput_samples_per_gpu_s * self.world_size
                )
            if elapsed is not None:
                payload["time/elapsed_s"] = elapsed
            if eta_seconds is not None:
                payload["time/eta_s"] = eta_seconds
            wandb.log(payload, step=self.global_step)

    def train(self):
        if self.rank == 0:
            logger.info("Starting training...")
        self._mlperf_log_run_start()

        # Ensure frozen state (idempotent)
        core = getattr(self.model, "module", self.model)
        if hasattr(core, "freeze_except"):
            core.freeze_except()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        self._start_profiler()

        start_time = time.time()
        last_log_time = start_time
        local_samples_in_update = 0
        local_samples_since_log = 0
        update_steps_since_log = 0
        update_loss_sum = None
        update_loss_count = 0
        mlperf_train_started = False

        steps_per_epoch = math.ceil(
            len(self.dataloader) / max(1, self.grad_accum_steps)
        )
        start_epoch = self.global_step // steps_per_epoch
        resume_batch_offset = (
            self.global_step % steps_per_epoch
        ) * self.grad_accum_steps

        for epoch in range(start_epoch, self.num_train_epochs):
            self.sampler.set_epoch(epoch)
            if isinstance(self.sampler, ContiguousDistributedSampler):
                sample_offset = (
                    resume_batch_offset * self.per_device_train_batch_size
                    if epoch == start_epoch
                    else 0
                )
                self.sampler.set_offset(sample_offset)

            for batch_idx, batch in enumerate(self.dataloader):
                if self.rank == 0 and self.global_step == 0 and batch_idx == 0:
                    logger.info("First training batch loaded; entering forward pass")
                if self.mlperf_enabled and not mlperf_train_started:
                    self._mlperf_log_train_start()
                    mlperf_train_started = True
                is_update_step = ((batch_idx + 1) % max(1, self.grad_accum_steps)) == 0
                local_samples_in_update += self._infer_local_batch_size(batch)
                if is_update_step:
                    self._mlperf_log_block_start(self.global_step + 1)

                with self._grad_sync_context(is_update_step):
                    try:
                        raw_loss = self.compute_loss(batch)
                    except BaseException as exc:
                        logger.exception(
                            "Training forward failed at step %d, batch %d: %r",
                            self.global_step,
                            batch_idx,
                            exc,
                        )
                        raise
                    if self.rank == 0 and self.global_step == 0 and batch_idx == 0:
                        logger.info(
                            "First training forward completed; entering backward pass"
                        )
                    detached_loss = raw_loss.detach().float()
                    update_loss_sum = (
                        detached_loss
                        if update_loss_sum is None
                        else update_loss_sum + detached_loss
                    )
                    update_loss_count += 1
                    loss = raw_loss / max(1, self.grad_accum_steps)
                    loss.backward()
                    if self.rank == 0 and self.global_step == 0 and batch_idx == 0:
                        logger.info("First training backward completed")

                if is_update_step:
                    loss_val = update_loss_sum / max(1, update_loss_count)
                    update_loss_sum = None
                    update_loss_count = 0
                    grad_norm = self._clip_grad_norm()

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    self._step_profiler()
                    self._mlperf_log_block_stop(self.global_step)
                    update_steps_since_log += 1
                    local_samples_since_log += local_samples_in_update
                    local_samples_in_update = 0

                    # Logging
                    if self.global_step % self.logging_steps == 0:
                        now = time.time()
                        log_interval = now - last_log_time
                        step_time = log_interval / max(1, update_steps_since_log)
                        last_log_time = now
                        elapsed = now - start_time
                        steps_left = max(0, self.total_steps - self.global_step)
                        eta_seconds = step_time * steps_left
                        throughput_samples_per_gpu_s = (
                            self._compute_samples_per_gpu_per_second(
                                local_samples=local_samples_since_log,
                                interval_seconds=log_interval,
                            )
                        )
                        self._log_step(
                            loss_val,
                            grad_norm=grad_norm,
                            step_time=step_time,
                            elapsed=elapsed,
                            eta_seconds=eta_seconds,
                            throughput_samples_per_gpu_s=throughput_samples_per_gpu_s,
                        )
                        local_samples_since_log = 0
                        update_steps_since_log = 0

                    # Periodic save
                    if self.save_steps > 0 and self.global_step % self.save_steps == 0:
                        self._save_checkpoint()

                    if (
                        self.mlperf_enabled
                        and self.global_step % self.mlperf_eval_freq_steps == 0
                    ):
                        self._mlperf_log_eval_start()
                        val_loss = self.validate_loss()
                        self._mlperf_log_eval_stop(val_loss)
                        if self.rank == 0:
                            logger.info(
                                f"mlperf_validation step={self.global_step} "
                                f"loss={val_loss:.6f} target={self.mlperf_target_eval_loss:.6f}"
                            )
                            if self.use_wandb:
                                payload = {
                                    "val/loss": val_loss,
                                    "validation_metrics/loss": val_loss,
                                    "validation_metrics/loss_vs_samples": val_loss,
                                    "validation_metrics/samples_count": (
                                        self.global_step * self._global_batch_size()
                                    ),
                                }
                                if (
                                    val_loss <= self.mlperf_target_eval_loss
                                    and self.mlperf_train_start_time
                                ):
                                    payload["time_metrics/time_to_converge(s)"] = (
                                        time.time() - self.mlperf_train_start_time
                                    )
                                wandb.log(payload, step=self.global_step)
                        if val_loss <= self.mlperf_target_eval_loss:
                            self.mlperf_run_success = True
                            if self.rank == 0 and self.mlperf_train_start_time:
                                time_to_converge = (
                                    time.time() - self.mlperf_train_start_time
                                )
                                logger.info(
                                    "MLPerf target reached: "
                                    f"validation_loss={val_loss:.6f}, "
                                    f"time_to_converge_s={time_to_converge:.2f}"
                                )
                                if self.mlperf_logger is not None:
                                    self.mlperf_logger.event(
                                        key="time_metrics/time_to_converge(s)",
                                        value=time_to_converge,
                                    )
                            self._mlperf_log_run_stop()
                            self._stop_profiler()
                            return

                    # Early termination
                    if self.max_steps > 0 and self.global_step >= self.max_steps:
                        self._mlperf_log_run_stop()
                        self._stop_profiler()
                        return

            if self.max_steps > 0 and self.global_step >= self.max_steps:
                break

        if self.rank == 0:
            elapsed = time.time() - start_time
            logger.info(f"Training finished in {elapsed / 60:.2f} min")
        self._mlperf_log_run_stop()
        self._stop_profiler()

    def save_model(self):
        """Save final model. Override in subclass."""
        raise NotImplementedError
