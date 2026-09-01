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

_PRECISION_DISCLOSURE_ENV = {
    "lowest_numerical_precision_in_linear": "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR",
    "lowest_numerical_precision_in_attn": "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN",
    "lowest_numerical_precision_in_comm": "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM",
}

# The compliance checker rejects any lowest_numerical_precision_* value outside
# this set (training_6.0.0/common.yaml). mxfp6 is deliberately absent: adding it
# is the Training WG request tracked separately, and until it lands a run that
# discloses mxfp6 produces a structurally valid log that the checker refuses.
# Emitting anything else would misdescribe the run, so the value is passed
# through and the mismatch is surfaced loudly rather than silently corrected.
_CHECKER_PRECISION_VALUES = frozenset(
    {
        "fp64",
        "fp32",
        "tf32",
        "fp16",
        "fp8",
        "nvfp4",
        "mxfp4",
        "bfloat16",
        "Graphcore FLOAT 16.16",
        "int8",
        "uint8",
        "int4",
        "uint4",
    }
)

# Identity records that decide which division the log is judged in. None may
# fall back to a built-in guess.
_SUBMISSION_IDENTITY_ENV = {
    "submission_org": "MLLOG_SUBMISSION_ORG",
    "submission_division": "MLLOG_SUBMISSION_DIVISION",
    "submission_platform": "MLLOG_SUBMISSION_PLATFORM",
}


def _is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _require_env(name: str, purpose: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"MLPerf mode requires {name} to be set explicitly ({purpose}).")
    return value


def _precision_disclosures_from_env() -> dict[str, str]:
    """Return the mandatory precision disclosures without guessing policy names."""
    values = {
        key: os.environ.get(environment_name, "").strip()
        for key, environment_name in _PRECISION_DISCLOSURE_ENV.items()
    }
    missing = [_PRECISION_DISCLOSURE_ENV[key] for key, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "MLPerf mode requires explicit precision disclosures; missing " + ", ".join(missing)
        )
    unaccepted = sorted({value for value in values.values() if value not in _CHECKER_PRECISION_VALUES})
    if unaccepted:
        logger.warning(
            "Precision disclosure(s) %s are not in the compliance checker's accepted set; "
            "the resulting log will be rejected until the format is approved upstream.",
            ", ".join(unaccepted),
        )
    return values


def _submission_identity_from_env() -> dict[str, str]:
    """Return org/division/platform, refusing to default any of them."""
    values = {
        key: _require_env(environment_name, f"MLLOG {key}")
        for key, environment_name in _SUBMISSION_IDENTITY_ENV.items()
    }
    division = values["submission_division"]
    if division not in ("closed", "open"):
        raise RuntimeError(f"MLLOG_SUBMISSION_DIVISION must be 'closed' or 'open', got {division!r}.")
    return values


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
        self._run_started = False
        self._run_stopped = False

        # A submission result is a file, not console scrollback: stdout is
        # interleaved with every other rank's output and with framework noise
        # that the parser then has to be trusted to ignore. Rank zero writes the
        # log itself so the artifact the checker reads is the artifact produced.
        if _is_rank_zero():
            mllog.config(
                filename=_require_env("MLLOG_OUTPUT_FILE", "the path this run's result_*.txt is written to"),
                # Frames from mllogger.event() back to the FluxMLPerfLogger
                # method that called it, so every record reports a stable
                # origin line. The seed checker compares those across runs.
                default_stack_offset=int(os.environ.get("MLLOG_STACK_OFFSET", "3")),
            )

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
        if not _is_rank_zero():
            return
        identity = _submission_identity_from_env()
        # The launcher clears the page cache before it starts the container and
        # reports what it did; defaulting this to false would let a run that
        # never cleared claim a cold start.
        clear_caches = (
            _require_env("MLPERF_CLEAR_CACHES", "whether the launcher dropped caches before this run").lower()
            == "true"
        )
        self._event(key="cache_clear", value=clear_caches)
        self._start(key=self._constants.INIT_START)
        self._event(key=self._constants.SUBMISSION_BENCHMARK, value="flux1")
        self._event(key=self._constants.SUBMISSION_ORG, value=identity["submission_org"])
        self._event(key=self._constants.SUBMISSION_DIVISION, value=identity["submission_division"])
        self._event(key=self._constants.SUBMISSION_PLATFORM, value=identity["submission_platform"])
        self._event(
            key=self._constants.SUBMISSION_STATUS,
            value=os.environ.get("MLLOG_SUBMISSION_STATUS", "onprem"),
        )
        self._event(key="target_accuracy", value=self.target_val_loss)
        self._event(key=self._constants.SEED, value=seed)

    def log_hyperparams(self, args):
        if not _is_rank_zero():
            return
        self._event(key=self._constants.GLOBAL_BATCH_SIZE, value=self.gbs)
        for key, value in _precision_disclosures_from_env().items():
            self._event(key=key, value=value)
        for key, value in (
            ("tensor_parallelism", getattr(args, "tensor_model_parallel_size", 1)),
            ("pipeline_parallelism", getattr(args, "pipeline_model_parallel_size", 1)),
            ("context_parallelism", getattr(args, "context_parallel_size", 1)),
            ("expert_parallelism", getattr(args, "expert_model_parallel_size", 1)),
            ("micro_batch_size", self.mbs),
            # Names the recipe a reviewer has to be able to find in the
            # submission's code/ directory, so "unknown" is not an answer.
            ("config_filename", _require_env("EXP", "the recipe this run was launched from")),
        ):
            self._event(key=key, value=value)
        self._event(
            key=self._constants.TRAIN_SAMPLES,
            value=getattr(args, "train_samples", None) or 1099776,
        )
        # EVAL_SAMPLES is emitted before any evaluation has run, so it can only
        # ever state the configured budget. The check that the budget was
        # actually read lives in the evaluation loop
        # (evaluator._record_consumed_valid_samples), which raises rather than
        # letting this number stand in for an unverified one.
        eval_iters = getattr(args, "eval_iters", 0) or 0
        self._event(
            key=self._constants.EVAL_SAMPLES,
            value=getattr(args, "eval_samples", None) or eval_iters * self.gbs,
        )
        # How often evaluation runs, in samples. EXACTLY_ONE in
        # closed_flux1.yaml, and spelled as a literal because no
        # mlperf_logging release defines a constant for it: the name appears
        # only in the checker's own rulesets. A getattr against the constants
        # module would therefore always take its fallback.
        self._event(
            key="evaluation_frequency",
            value=getattr(args, "eval_interval", 0) * self.gbs,
        )
        data_parallel_size = getattr(args, "data_parallel_size", 1) or 1
        gas = max(self.gbs // (self.mbs * data_parallel_size), 1)
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
        self._event(
            key="opt_learning_rate_warmup_steps",
            value=getattr(args, "lr_warmup_iters", 0),
        )
        self._event(
            key="opt_gradient_clip_norm",
            value=getattr(args, "clip_grad", 1.0),
        )

    def log_init_stop_run_start(self):
        if self._run_started:
            return
        self._run_started = True
        if _is_rank_zero():
            self._end(key=self._constants.INIT_STOP)
            self._start(key=self._constants.RUN_START)
            self._start(key=self._constants.EPOCH_START, metadata={"epoch_num": 0})
            self.log_block_start(0)

    def log_block_start(self, global_step: int):
        if _is_rank_zero():
            self._start(
                key=self._constants.BLOCK_START,
                metadata={"samples_count": global_step * self.gbs},
            )

    def on_train_batch_end(self, global_step: int, loss: float, lr: float):
        self.timer.mark_training_start()
        self.timer.update_samples(global_step)

        self._handle_profiler(global_step)

        if not _is_rank_zero():
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
        self.log_init_stop_run_start()
        self.timer.update_samples(global_step)
        self.timer.pause_for_eval()

        if _is_rank_zero():
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
                metadata={"samples_count": global_step * self.gbs},
            )
            self._start(
                key=self._constants.EVAL_START,
                metadata={"samples_count": global_step * self.gbs},
            )

    def on_validation_end(self, global_step: int, val_loss: float):
        self.timer.resume_after_eval()

        if _is_rank_zero():
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

    @property
    def run_stopped(self):
        """Whether run_stop has been emitted, however the run ended.

        Distinct from ``converged``, which is only the success path. Anything
        guarding against records past the end of the run wants this one.
        """
        return self._run_stopped

    def log_run_stop(self, success: bool, global_step: int):
        # run_stop is EXACTLY_ONE in the ruleset: a converged run that also hits
        # the end-of-training path must not emit a second, contradictory record.
        if self._run_stopped:
            return
        if success:
            self._converged = True
        self._run_stopped = True
        if _is_rank_zero():
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
    eval_purge_memory = getattr(args, "eval_purge_memory", False)

    mlperf_logger = FluxMLPerfLogger(
        global_batch_size=gbs,
        micro_batch_size=mbs,
        target_val_loss=target_val_loss,
        log_every_n_steps=log_interval,
    )

    mlperf_logger.log_init(seed=seed)
    mlperf_logger.log_hyperparams(args)

    # The clock has to start before Megatron opens the dataset, which happens
    # inside pretrain() after this phase has already run. mlperf_boundary
    # creates that seam; the call below is what it fires there. The
    # first-training_log path further down stays as a backstop and turns into a
    # no-op once this has run.
    from primus.backends.megatron.patches import mlperf_boundary

    mlperf_boundary.set_transition(mlperf_logger.log_init_stop_run_start)
    mlperf_boundary.install()

    # Reachable from the after_train phase, which closes out runs that finish
    # their step budget without converging.
    megatron_training._primus_mlperf_logger = mlperf_logger

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

        # The run is over and run_stop is EXACTLY_ONE, so an evaluation reaching
        # here is outside the measured region and contributes nothing to the
        # submission. Clearing do_valid on convergence means nothing should get
        # this far; this makes "the log ends at run_stop" hold structurally
        # rather than by call ordering, for whatever call site comes next.
        if mlperf_logger.run_stopped:
            return _orig_eval(*eval_args, **eval_kwargs)

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
            result = _orig_eval(*eval_args, **eval_kwargs)
        finally:
            megatron_training.evaluate = _current_eval

        # Reclaiming memory after every eval costs a full GC pause plus an
        # allocator flush inside the measured window, and the allocator has to
        # re-grow its pools on the next training step. Off by default; set
        # eval_purge_memory to re-enable if a run proves it needs the headroom.
        if eval_purge_memory:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()

        val_loss = _extract_val_loss(_loss_capture)
        if val_loss is not None:
            # primus_evaluate already reduces over the data-parallel group and
            # returns a value identical on every rank, which is what the
            # early-stop comparison below requires: if ranks disagree near the
            # target, one can exit train() alone while the others keep training,
            # desyncing collectives into an NCCL watchdog deadlock (observed on
            # FLUX 12B MLPerf at step 2560). A further ReduceOp.AVG here would
            # average identical values -- a no-op costing one collective and one
            # host sync -- so it is deliberately absent.
            mlperf_logger.on_validation_end(iteration, val_loss)
            log_rank_0(
                f"[MLPerf] Validation loss at step {iteration}: {val_loss:.6f} "
                f"(target: {target_val_loss:.6f})"
            )

            converged = val_loss <= target_val_loss
            if converged:
                log_rank_0(
                    f"[MLPerf] Convergence reached! val_loss={val_loss:.6f} "
                    f"<= target={target_val_loss:.6f}"
                )
                mlperf_logger.log_run_stop(success=True, global_step=iteration)
                try:
                    from megatron.training import get_args as megatron_get_args

                    megatron_args = megatron_get_args()
                    megatron_args.train_iters = iteration
                    # Breaking the loop returns into pretrain(), which runs one
                    # more validation whenever do_valid is set. That evaluation
                    # is past run_stop, and it draws fresh VAE epsilon and
                    # flow-matching noise, so it reports a different loss and
                    # can land above target -- contradicting the evaluation
                    # that just ended the run.
                    megatron_args.do_valid = False
                except Exception:
                    logger.warning("Could not set args.train_iters/do_valid for early stop")
        else:
            converged = False
            logger.warning("Could not extract validation loss from evaluate result")

        # Training resumes unless this eval ended the run, so the block that
        # on_validation_start closed has to be reopened -- including on the
        # path where the loss could not be read and training carries on.
        if not converged:
            mlperf_logger.log_block_start(iteration)

        return result

    _mlperf_evaluate_and_print_results._primus_mlperf_eval_wrapper = True
    megatron_training.evaluate_and_print_results = _mlperf_evaluate_and_print_results

    megatron_training._primus_mlperf_logging_installed = True

    log_rank_0(
        f"[Patch:mlperf_logging] Installed MLPerf logging (gbs={gbs}, "
        f"target_val_loss={target_val_loss}, log_interval={log_interval})"
    )


@register_patch(
    "megatron.training.mlperf_run_stop",
    backend="megatron",
    phase="after_train",
    description="Close out a run that ended without reaching the quality target",
    condition=_mlperf_logging_enabled,
    priority=15,
)
def patch_mlperf_terminal_run_stop(ctx: PatchContext):
    """Emit run_stop for a run that exhausted its step budget.

    Only convergence emitted run_stop before, so a run that never hit the
    target produced a log with run_start and no run_stop. The ruleset requires
    exactly one, and a non-converging run is still evidence -- it belongs in
    the RCP comparison as a run that did not make it, not as an unparseable
    file. log_run_stop is idempotent, so converged runs fall through here.
    """
    import megatron.training.training as megatron_training

    mlperf_logger = getattr(megatron_training, "_primus_mlperf_logger", None)
    if mlperf_logger is None or mlperf_logger.converged:
        return

    iteration = 0
    try:
        from megatron.training import get_args as megatron_get_args

        megatron_args = megatron_get_args()
        iteration = getattr(megatron_args, "curr_iteration", 0) or getattr(megatron_args, "train_iters", 0)
    except Exception:
        logger.warning("Could not read the final iteration for the terminal run_stop")

    mlperf_logger.log_run_stop(success=False, global_step=iteration)
    log_rank_0(f"[Patch:mlperf_run_stop] Run ended without converging at iteration {iteration}")
