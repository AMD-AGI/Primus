###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MLPerf Logging Patches for Flux Training.

Installs MLPerf-compliant logging into Megatron's training loop by wrapping:
  - training_log: emit INIT_STOP, RUN_START, tracked_stats, train_loss
  - evaluate_and_print_results: emit EVAL events, convergence check
  - print_rank_last / get_tensorboard_writer / get_wandb_writer: suppress

Uses mlperf_logging.mllog library for structured event output.
"""

import logging
import os
import sys
import time

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

logger = logging.getLogger(__name__)


def _mlperf_logging_enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return args is not None and getattr(args, "mlperf_mode", False)


class ThroughputTimer:
    """Wall-clock throughput tracker with eval pause/resume."""

    def __init__(self, gbs: int):
        self.gbs = gbs
        self.training_start_time: float | None = None
        self.eval_cumulative_secs: float = 0.0
        self._eval_enter_time: float | None = None
        self.consumed_samples: int = 0

    def mark_training_start(self):
        if self.training_start_time is None:
            self.training_start_time = time.time()

    def update_samples(self, iteration: int):
        self.consumed_samples = iteration * self.gbs

    def pause_for_eval(self):
        self._eval_enter_time = time.time()

    def resume_after_eval(self):
        if self._eval_enter_time is not None:
            self.eval_cumulative_secs += time.time() - self._eval_enter_time
            self._eval_enter_time = None

    def compute_throughput(self):
        if self.training_start_time is None:
            return 0.0
        wall = time.time() - self.training_start_time
        training_secs = wall - self.eval_cumulative_secs
        if training_secs <= 0:
            return 0.0
        return self.consumed_samples / training_secs

    def compute_combined_throughput(self):
        if self.training_start_time is None:
            return 0.0
        wall = time.time() - self.training_start_time
        if wall <= 0:
            return 0.0
        return self.consumed_samples / wall


class FluxMLPerfLogger:
    """MLPerf logger using mlperf_logging.mllog directly."""

    def __init__(
        self,
        global_batch_size: int,
        micro_batch_size: int,
        target_val_loss: float = 0.586,
        log_every_n_steps: int = 10,
    ):
        from mlperf_logging import mllog

        self._mllogger = mllog.get_mllogger()
        self._constants = mllog.constants
        self.gbs = global_batch_size
        self.mbs = micro_batch_size
        self.target_val_loss = target_val_loss
        self.log_every_n_steps = log_every_n_steps
        self.timer = ThroughputTimer(global_batch_size)
        self._converged = False

        self.profiler = os.getenv("PROFILER", "")
        self.profiler_warmup_steps = int(os.getenv("PROF_WARMUP_STEPS", "0"))
        self.profiler_active_steps = int(os.getenv("PROF_ACTIVE_STEPS", "0"))
        self.rpd = None
        self.rpd_running = False

        if self.profiler == "rpd":
            try:
                from rpdTracerControl import rpdTracerControl

                rpdTracerControl.setFilename("trace.rpd", append=True)
                self.rpd = rpdTracerControl()
                logger.info("RPD profiler initialized")
            except ImportError:
                logger.warning("rpdTracerControl not available")

    def _event(self, key, value=None, metadata=None):
        self._mllogger.event(key=key, value=value, metadata=metadata)

    def _start(self, key, value=None, metadata=None):
        self._mllogger.start(key=key, value=value, metadata=metadata)

    def _end(self, key, value=None, metadata=None):
        self._mllogger.end(key=key, value=value, metadata=metadata)

    def log_init(self, seed: int):
        if int(os.environ.get("RANK", "0")) == 0:
            self._start(key=self._constants.INIT_START)
            self._event(key=self._constants.SUBMISSION_BENCHMARK, value="flux1")
            self._event(
                key=self._constants.SUBMISSION_ORG,
                value=os.environ.get("MLLOG_SUBMISSION_ORG", "AMD"),
            )
            self._event(
                key=self._constants.SUBMISSION_DIVISION,
                value=os.environ.get("MLLOG_SUBMISSION_DIVISION", "closed"),
            )
            self._event(
                key=self._constants.SUBMISSION_PLATFORM,
                value=os.environ.get("MLLOG_SUBMISSION_PLATFORM", "MI355X"),
            )
            self._event(key=self._constants.SUBMISSION_STATUS, value="onprem")
            self._event(key="target_accuracy", value=self.target_val_loss)
            self._event(key=self._constants.SEED, value=seed)

    def log_hyperparams(self, args):
        if int(os.environ.get("RANK", "0")) != 0:
            return
        self._event(key=self._constants.GLOBAL_BATCH_SIZE, value=self.gbs)
        self._event(
            key=self._constants.TRAIN_SAMPLES,
            value=getattr(args, "train_samples", 1099776),
        )
        self._event(
            key=self._constants.EVAL_SAMPLES,
            value=getattr(args, "eval_samples", 29696),
        )
        gas = max(self.gbs // self.mbs, 1)
        self._event(key=self._constants.GRADIENT_ACCUMULATION_STEPS, value=gas)
        self._event(key=self._constants.OPT_NAME, value="adamw")
        self._event(
            key=self._constants.OPT_BASE_LR,
            value=getattr(args, "lr", 2e-4),
        )
        self._event(
            key="opt_adamw_beta_1",
            value=getattr(args, "adam_beta1", 0.9),
        )
        self._event(
            key="opt_adamw_beta_2",
            value=getattr(args, "adam_beta2", 0.95),
        )
        self._event(
            key="opt_adamw_epsilon",
            value=getattr(args, "adam_eps", 1e-8),
        )
        self._event(
            key="opt_adamw_weight_decay",
            value=getattr(args, "weight_decay", 0.1),
        )

    def log_init_stop_run_start(self):
        if int(os.environ.get("RANK", "0")) == 0:
            self._end(key=self._constants.INIT_STOP)
            self._start(key=self._constants.RUN_START)
            self._start(key=self._constants.EPOCH_START, metadata={"epoch_num": 0})
            self._start(key=self._constants.BLOCK_START, metadata={"first_epoch_num": 0})

    def on_train_batch_end(self, global_step: int, loss: float, lr: float):
        self.timer.mark_training_start()
        self.timer.update_samples(global_step)

        self._handle_profiler(global_step)

        if int(os.environ.get("RANK", "0")) != 0:
            return
        if global_step % self.log_every_n_steps == 0:
            self._event(
                key="tracked_stats",
                value={"train_loss": loss},
                metadata={
                    "samples_count": global_step * self.gbs,
                    "lr": lr,
                    "step": global_step,
                },
            )

    def on_validation_start(self, global_step: int):
        self.timer.update_samples(global_step)
        self.timer.pause_for_eval()

        if int(os.environ.get("RANK", "0")) == 0:
            if global_step > 0:
                throughput = self.timer.compute_throughput()
                self._event(
                    key="throughput",
                    value=throughput,
                    metadata={
                        "samples_count": global_step * self.gbs,
                        "step": global_step,
                    },
                )
            self._end(
                key=self._constants.BLOCK_STOP,
                metadata={"first_epoch_num": 0},
            )
            self._start(key=self._constants.EVAL_START, metadata={"epoch_num": 0})

    def on_validation_end(self, global_step: int, val_loss: float):
        self.timer.resume_after_eval()

        if int(os.environ.get("RANK", "0")) == 0:
            self._event(
                key=self._constants.EVAL_ACCURACY,
                value=val_loss,
                metadata={
                    "samples_count": global_step * self.gbs,
                    "step": global_step,
                },
            )
            self._end(key=self._constants.EVAL_STOP, metadata={"epoch_num": 0})
            combined_throughput = self.timer.compute_combined_throughput()
            self._event(
                key="combined_throughput",
                value=combined_throughput,
                metadata={
                    "samples_count": global_step * self.gbs,
                    "step": global_step,
                },
            )

    def _handle_profiler(self, global_step: int):
        if self.profiler != "rpd":
            return
        if self.rpd and not self.rpd_running and global_step >= self.profiler_warmup_steps:
            logger.info("Starting RPD profiler")
            self.rpd.start()
            self.rpd.rangePush("python", "Training", "")
            self.rpd_running = True
        if self.rpd_running and global_step > self.profiler_warmup_steps + self.profiler_active_steps:
            logger.info("Stopping RPD profiler")
            self.rpd.rangePop()
            self.rpd.stop()
            self.rpd = None
            self.rpd_running = False

    @property
    def converged(self):
        return self._converged

    def log_run_stop(self, success: bool, global_step: int):
        if success:
            self._converged = True
        if int(os.environ.get("RANK", "0")) == 0:
            status = "success" if success else "aborted"
            self._end(
                key=self._constants.RUN_STOP,
                value=status,
                metadata={
                    "samples_count": global_step * self.gbs,
                    "step": global_step,
                    "status": status,
                },
            )

    def teardown(self):
        if self.rpd_running and self.rpd:
            self.rpd.rangePop()
            self.rpd.stop()
            self.rpd = None
            self.rpd_running = False


def _extract_val_loss(loss_dict):
    """Extract scalar validation loss from captured total_loss_dict."""
    if not loss_dict or not isinstance(loss_dict, dict):
        return None
    for key in ("loss", "lm loss"):
        if key in loss_dict:
            val = loss_dict[key]
            return val.item() if hasattr(val, "item") else float(val)
    if loss_dict:
        val = next(iter(loss_dict.values()))
        return val.item() if hasattr(val, "item") else float(val)
    return None


@register_patch(
    "megatron.training.mlperf_logging",
    backend="megatron",
    phase="before_train",
    description="Install MLPerf logging wrappers for Flux training",
    condition=_mlperf_logging_enabled,
    priority=15,
)
def patch_mlperf_logging(ctx: PatchContext):
    """Install MLPerf logging: suppress Megatron output, wrap training_log and eval."""
    import megatron.training.training as megatron_training

    if getattr(megatron_training, "_primus_mlperf_logging_installed", False):
        return

    args = get_args(ctx)
    seed = getattr(args, "seed", 42)
    gbs = getattr(args, "global_batch_size", 512)
    mbs = getattr(args, "micro_batch_size", 64)
    target_val_loss = getattr(args, "target_val_loss", 0.586)
    log_interval = getattr(args, "log_interval", 10)

    mlperf_logger = FluxMLPerfLogger(
        global_batch_size=gbs,
        micro_batch_size=mbs,
        target_val_loss=target_val_loss,
        log_every_n_steps=log_interval,
    )

    mlperf_logger.log_init(seed=seed)
    mlperf_logger.log_hyperparams(args)

    # --- Suppress Megatron's built-in logging ---
    megatron_training.print_rank_last = lambda *a, **k: None

    for writer_fn in ("get_tensorboard_writer", "get_wandb_writer"):
        if hasattr(megatron_training, writer_fn):
            setattr(megatron_training, writer_fn, lambda: None)

    # --- Wrap training_log ---
    _orig_training_log = megatron_training.training_log
    _first_training_log_call = [True]

    def _mlperf_training_log(*args_tl, **kwargs_tl):
        if _first_training_log_call[0]:
            _first_training_log_call[0] = False
            mlperf_logger.log_init_stop_run_start()

        result = _orig_training_log(*args_tl, **kwargs_tl)

        try:
            loss_dict = args_tl[0] if len(args_tl) > 0 else kwargs_tl.get("loss_dict", {})
            learning_rate = args_tl[2] if len(args_tl) > 2 else kwargs_tl.get("learning_rate", 0.0)
            # Upstream Megatron: training_log(loss_dict, total_loss_dict, learning_rate, iteration, ...)
            iteration = args_tl[3] if len(args_tl) > 3 else kwargs_tl.get("iteration", 0)

            if loss_dict:
                loss_val = next(iter(loss_dict.values()))
                if hasattr(loss_val, "item"):
                    loss_val = loss_val.item()
                mlperf_logger.on_train_batch_end(iteration, loss_val, learning_rate)

                if iteration % log_interval == 0:
                    lr_str = f"{learning_rate:.2e}" if learning_rate else "N/A"
                    sys.stdout.write(
                        f"step {iteration} | loss: {loss_val:.4f} | lr: {lr_str}"
                        f" | samples: {iteration * gbs}\n"
                    )
                    sys.stdout.flush()
        except Exception as e:
            logger.debug("MLPerf training_log hook: %s", e)

        return result

    _mlperf_training_log._primus_mlperf_logging_wrapper = True
    megatron_training.training_log = _mlperf_training_log

    # --- Wrap evaluate_and_print_results ---
    _orig_eval = megatron_training.evaluate_and_print_results

    def _mlperf_evaluate_and_print_results(*eval_args, **eval_kwargs):
        # evaluate_and_print_results(prefix, fwd, data, model, iteration[4], ...)
        iteration = eval_kwargs.get("iteration", eval_args[4] if len(eval_args) > 4 else 0)

        mlperf_logger.on_validation_start(iteration)

        # Temporarily wrap whatever `evaluate` is at call time (e.g.
        # primus_evaluate installed by the evaluate patch) so we can
        # capture the total_loss_dict it returns.  This avoids relying
        # on a persistent hook that later patches can overwrite.
        _loss_capture = {}
        _current_eval = megatron_training.evaluate

        def _capture_wrapper(*a, **kw):
            res = _current_eval(*a, **kw)
            td = res[0] if isinstance(res, tuple) else res
            if isinstance(td, dict):
                _loss_capture.update(td)
            return res

        megatron_training.evaluate = _capture_wrapper
        try:
            import gc

            result = _orig_eval(*eval_args, **eval_kwargs)
            gc.collect()
        finally:
            megatron_training.evaluate = _current_eval

        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

        val_loss = _extract_val_loss(_loss_capture)
        if val_loss is not None:
            # Megatron's `evaluate()` (training.py:3178-3180) divides the
            # per-rank accumulated loss locally and does NOT all-reduce
            # across the data-parallel group — the result is intended for
            # `print_rank_last` / TensorBoard which only read on a single
            # rank.  We must reduce here so every rank evaluates the same
            # global validation loss against `target_val_loss`; otherwise
            # ranks can disagree on the early-stop branch and the
            # divergent `args.train_iters` mutation below desyncs
            # collective ordering at the next training step, producing a
            # NCCL watchdog deadlock (observed on FLUX 12B MLPerf at
            # step 2560 when val_loss landed near target).
            import torch
            import torch.distributed as dist

            if dist.is_initialized():
                try:
                    from megatron.core import parallel_state as mpu

                    dp_group = mpu.get_data_parallel_group()
                except Exception:
                    dp_group = None
                _vl = torch.tensor(val_loss, dtype=torch.float64, device="cuda")
                dist.all_reduce(_vl, op=dist.ReduceOp.AVG, group=dp_group)
                val_loss = _vl.item()

            mlperf_logger.on_validation_end(iteration, val_loss)
            log_rank_0(
                f"[MLPerf] Validation loss at step {iteration}: {val_loss:.6f} "
                f"(target: {target_val_loss:.6f})"
            )

            if val_loss <= target_val_loss:
                log_rank_0(
                    f"[MLPerf] Convergence reached! val_loss={val_loss:.6f} "
                    f"<= target={target_val_loss:.6f}"
                )
                mlperf_logger.log_run_stop(success=True, global_step=iteration)
                try:
                    from megatron.training import get_args as megatron_get_args

                    megatron_get_args().train_iters = iteration
                except Exception:
                    logger.warning("Could not set args.train_iters for early stop")
            else:
                if int(os.environ.get("RANK", "0")) == 0:
                    mlperf_logger._start(
                        key=mlperf_logger._constants.BLOCK_START,
                        metadata={"first_epoch_num": 0},
                    )
        else:
            logger.warning("Could not extract validation loss from evaluate result")

        return result

    _mlperf_evaluate_and_print_results._primus_mlperf_eval_wrapper = True
    megatron_training.evaluate_and_print_results = _mlperf_evaluate_and_print_results

    megatron_training._primus_mlperf_logging_installed = True

    log_rank_0(
        f"[Patch:mlperf_logging] Installed MLPerf logging (gbs={gbs}, "
        f"target_val_loss={target_val_loss}, log_interval={log_interval})"
    )
