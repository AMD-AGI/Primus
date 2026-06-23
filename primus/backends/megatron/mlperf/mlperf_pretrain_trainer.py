###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MLPerf-enabled Megatron pretrain trainer for the Primus new architecture.

Unlike the legacy ``primus_mllog.MLPerfMegatronPretrainTrainer`` (which
subclassed the *old* Primus trainer and overrode Primus-owned ``init`` /
``run`` / ``train`` / ``train_step`` methods), the new architecture delegates
the entire training loop to upstream ``megatron.training.pretrain()`` inside
:meth:`MegatronPretrainTrainer.train`.  There are therefore no Primus-owned
loop methods to override.

This trainer maps the MLPerf logging onto the new ``BaseTrainer`` lifecycle
(``setup -> init -> train -> cleanup``) by **monkey-patching the upstream
Megatron functions** that the loop calls:

    * ``megatron.training.training.train``                 (loop entry/exit)
    * ``megatron.training.training.train_step``            (per-step hooks)
    * ``megatron.training.training.evaluate``              (capture val loss)
    * ``megatron.training.training.evaluate_and_print_results`` (eval markers)

All ``mlperf_logging`` imports are lazy so importing this module never breaks
non-MLPerf runs.
"""

import os
import time

from primus.backends.megatron.megatron_pretrain_trainer import MegatronPretrainTrainer
from primus.backends.megatron.mlperf.mlperf_logger import MLPerfLogger, ThroughputTimer

try:
    from rpdTracerControl import rpdTracerControl
except ImportError:
    rpdTracerControl = None


def _get_arg(args, kwargs, index, name):
    """Fetch a positional-or-keyword argument from a wrapped call.

    Upstream Megatron calls ``train`` / ``train_step`` positionally, but we
    accept either form so the patch is robust across signature tweaks.
    """
    if name in kwargs:
        return kwargs[name]
    if index < len(args):
        return args[index]
    return None


class MLPerfMegatronPretrainTrainer(MegatronPretrainTrainer):
    """MegatronPretrainTrainer with MLPerf (mllog) logging."""

    def __init__(self, backend_args):
        super().__init__(backend_args)

        self.mllogger = MLPerfLogger()
        self.throughput_timer = None
        self.train_start_time = None
        self.train_stop_time = None
        self.last_validation_loss = None
        self.train_loss_log_freq = int(os.getenv("MLLOG_TRAIN_LOSS_LOG_FREQ", "1"))
        self.block_tput_log = os.getenv("MLLOG_BLOCK_TPUT_LOG", "0") == "1"
        self.target_eval_loss = float(os.getenv("MLLOG_TARGET_EVAL_LOSS", "0.0"))
        self.is_target_reached = False
        self._warmup_active = False

        # RPD profiler (optional). Mirrors the legacy wrapper.
        profiler = os.getenv("PROFILER", "")
        self.profile_segment = os.getenv("PROFILE_SEGMENT", "train")
        self.rpd = None
        if profiler == "rpd" and rpdTracerControl is not None:
            rpdTracerControl.setFilename(name="trace.rpd", append=True)
            self.rpd = rpdTracerControl()
            enable_python_trace = os.getenv("ENABLE_PYTHON_TRACE", "false").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            self.rpd.setPythonTrace(doTrace=enable_python_trace)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self):
        """Megatron setup + MLPerf CACHE_CLEAR / INIT_START.

        ``setup()`` runs before ``init()`` and before ``train()`` (which
        calls upstream ``pretrain()`` that performs the heavy init), so this
        is the correct place to bracket the start of initialisation.
        """
        super().setup()

        # Primus reconfigures stdlib ``logging`` via loguru during setup,
        # which resets the per-logger levels installed by the log-suppression
        # module at import time. Re-apply the quiet levels (no-op unless
        # PRIMUS_LOG_SUPPRESSION=1 and not verbose).
        try:
            from primus.mlperf_log_suppression import reapply_quiet_logger_levels

            reapply_quiet_logger_levels()
        except Exception:
            pass

        from mlperf_logging.mllog import constants

        self.mllogger.log_event_all_ranks(key=constants.CACHE_CLEAR, value=True)
        self.mllogger.log_event_all_ranks(key=constants.INIT_START, value=None)

    def train(self):
        """Install MLPerf hooks on upstream Megatron, then run pretrain()."""
        import megatron.training.training as mt

        orig_train = mt.train
        orig_train_step = mt.train_step
        orig_evaluate = mt.evaluate
        orig_eval_and_print = mt.evaluate_and_print_results

        wrapped_train = self._make_wrapped_train(mt, orig_train)
        wrapped_train_step = self._make_wrapped_train_step(orig_train_step)
        wrapped_evaluate = self._make_wrapped_evaluate(orig_evaluate)
        wrapped_eval_and_print = self._make_wrapped_eval_and_print(orig_eval_and_print)

        mt.train = wrapped_train
        mt.train_step = wrapped_train_step
        mt.evaluate = wrapped_evaluate
        mt.evaluate_and_print_results = wrapped_eval_and_print

        try:
            return super().train()
        finally:
            mt.train = orig_train
            mt.train_step = orig_train_step
            mt.evaluate = orig_evaluate
            mt.evaluate_and_print_results = orig_eval_and_print

    # ------------------------------------------------------------------
    # Patch builders
    # ------------------------------------------------------------------

    def _make_wrapped_train(self, mt, orig_train):
        def wrapped_train(*args, **kwargs):
            from megatron.training import get_args
            from mlperf_logging.mllog import constants

            from primus.backends.megatron.mlperf.warmup import run_synthetic_warmup

            megatron_args = get_args()

            # Configure mllog + log hyper-parameters + INIT_STOP. ``get_args``
            # is valid here because upstream pretrain() has finished init by
            # the time it calls train().
            output_file = os.getenv("MLLOG_OUTPUT_FILE", "/results/mlperf_output.log")
            configs = self.mllogger.configure(output_file, megatron_args)
            for key, value in configs.items():
                self.mllogger.log_event(key=key, value=value)
            self.mllogger.log_end(key=constants.INIT_STOP)

            self.throughput_timer = ThroughputTimer(megatron_args.global_batch_size)

            forward_step_func = _get_arg(args, kwargs, 0, "forward_step_func")
            model = _get_arg(args, kwargs, 1, "model")
            optimizer = _get_arg(args, kwargs, 2, "optimizer")
            opt_param_scheduler = _get_arg(args, kwargs, 3, "opt_param_scheduler")
            config = _get_arg(args, kwargs, 7, "config")

            # Synthetic warmup runs BEFORE RUN_START so its time is excluded
            # from the timed run. It calls the (patched) module-level
            # train_step; _warmup_active suppresses per-step loss logging.
            self._warmup_active = True
            try:
                run_synthetic_warmup(
                    mt.train_step,
                    forward_step_func,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    config,
                    megatron_args,
                )
            finally:
                self._warmup_active = False
            # Discard warmup step timings.
            self.throughput_timer.get_throughput()

            self.mllogger.log_start(key=constants.RUN_START)
            self.train_start_time = time.time()
            self.mllogger.log_start(key=constants.EPOCH_START, metadata={constants.SAMPLES_COUNT: 0})
            self.mllogger.log_start(key=constants.BLOCK_START, metadata={constants.SAMPLES_COUNT: 0})

            if self.rpd and self.profile_segment == "train":
                self.rpd.start()

            result = orig_train(*args, **kwargs)

            if self.rpd and self.profile_segment == "train":
                self.rpd.stop()
                self.rpd = None

            final_args = get_args()
            consumed_samples = final_args.consumed_train_samples
            self.mllogger.log_end(
                key=constants.EPOCH_STOP, metadata={constants.SAMPLES_COUNT: consumed_samples}
            )

            if self.is_target_reached:
                self.mllogger.log_end(
                    key=constants.RUN_STOP,
                    metadata={
                        constants.SAMPLES_COUNT: consumed_samples,
                        constants.STATUS: constants.SUCCESS,
                    },
                )
            else:
                self.train_stop_time = time.time()
                self.mllogger.log_end(
                    key=constants.RUN_STOP,
                    metadata={
                        constants.SAMPLES_COUNT: consumed_samples,
                        constants.STATUS: constants.ABORTED,
                    },
                )

            # Overall run summary (rank-0 only; informational keys).
            duration = (self.train_stop_time or time.time()) - (self.train_start_time or time.time())
            duration_minutes = duration / 60.0
            overall_throughput = consumed_samples / duration if duration > 0 else 0.0
            self.mllogger.log_event(
                key="run_duration",
                value=f"{round(duration, 2)}s -> {round(duration_minutes, 2)} minutes",
                metadata={"samples": consumed_samples},
            )
            self.mllogger.log_event(
                key="overall_throughput",
                value=round(overall_throughput, 2),
                metadata={"samples": consumed_samples},
            )

            return result

        return wrapped_train

    def _make_wrapped_train_step(self, orig_train_step):
        def wrapped_train_step(*args, **kwargs):
            if self.rpd and self.mllogger._get_rank() == 0 and self.profile_segment == "train_step":
                self.rpd.start()

            if self.throughput_timer is not None:
                self.throughput_timer.step_start()
            result = orig_train_step(*args, **kwargs)
            if self.throughput_timer is not None:
                self.throughput_timer.step_stop()

            # Upstream train_step returns an 8-tuple:
            # (loss_dict, skipped_iter, should_checkpoint, should_exit,
            #  exit_code, grad_norm, num_zeros_in_grad, log_max_attention_logit)
            loss_dict = result[0]
            skipped_iter = result[1]

            if not skipped_iter and loss_dict and not self._warmup_active:
                optimizer = _get_arg(args, kwargs, 3, "optimizer")
                self._on_train_step_end(loss_dict, optimizer)

            if self.rpd and self.mllogger._get_rank() == 0 and self.profile_segment == "train_step":
                self.rpd.stop()
                self.rpd = None

            return result

        return wrapped_train_step

    def _make_wrapped_evaluate(self, orig_evaluate):
        def wrapped_evaluate(*args, **kwargs):
            if self.rpd and self.mllogger._get_rank() == 0 and self.profile_segment == "eval":
                self.rpd.start()

            result = orig_evaluate(*args, **kwargs)

            if self.rpd and self.mllogger._get_rank() == 0 and self.profile_segment == "eval":
                self.rpd.stop()
                self.rpd = None

            total_loss_dict = result[0]
            if total_loss_dict and "lm loss" in total_loss_dict:
                val_loss = total_loss_dict["lm loss"]
                if hasattr(val_loss, "item"):
                    self.last_validation_loss = val_loss.item()
                else:
                    self.last_validation_loss = float(val_loss)

            return result

        return wrapped_evaluate

    def _make_wrapped_eval_and_print(self, orig_eval_and_print):
        def wrapped_eval_and_print(*args, **kwargs):
            from megatron.training import get_args
            from mlperf_logging.mllog import constants

            eval_args = get_args()
            consumed_samples = eval_args.consumed_train_samples

            self._log_tracked_stats(eval_args.iteration, consumed_samples)

            self.mllogger.log_end(
                key=constants.BLOCK_STOP, metadata={constants.SAMPLES_COUNT: consumed_samples}
            )
            self.mllogger.log_start(
                key=constants.EVAL_START, metadata={constants.SAMPLES_COUNT: consumed_samples}
            )

            if self.rpd and self.mllogger._get_rank() == 0 and self.profile_segment == "eval":
                self.rpd.start()

            result = orig_eval_and_print(*args, **kwargs)

            if self.rpd and self.mllogger._get_rank() == 0 and self.profile_segment == "eval":
                self.rpd.stop()
                self.rpd = None

            validation_loss = self._get_validation_loss()
            if validation_loss is not None:
                self.mllogger.log_event(
                    key=constants.EVAL_ACCURACY,
                    value=validation_loss,
                    metadata={constants.SAMPLES_COUNT: consumed_samples},
                )

                if self.target_eval_loss > 0.0 and validation_loss <= self.target_eval_loss:
                    if not self.is_target_reached:
                        self.is_target_reached = True
                        self.train_stop_time = time.time()
                        if self.mllogger._get_rank() == 0:
                            print(
                                f"[MLPERF] Target eval loss {self.target_eval_loss} reached "
                                f"with validation loss {validation_loss}"
                            )
                        eval_args.train_iters = eval_args.iteration
                        eval_args.do_valid = False
                        eval_args.do_test = False

            self.mllogger.log_end(
                key=constants.EVAL_STOP, metadata={constants.SAMPLES_COUNT: consumed_samples}
            )

            if not self.is_target_reached:
                self.mllogger.log_start(
                    key=constants.BLOCK_START, metadata={constants.SAMPLES_COUNT: consumed_samples}
                )

            return result

        return wrapped_eval_and_print

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _on_train_step_end(self, loss_dict, optimizer):
        from megatron.training import get_args
        from mlperf_logging.mllog import constants

        args = get_args()
        iteration = args.iteration
        consumed_samples = args.consumed_train_samples

        if self.train_loss_log_freq <= 0 or iteration % self.train_loss_log_freq != 0:
            return

        loss_value = loss_dict.get("lm loss")
        if loss_value is None:
            return

        if isinstance(loss_value, (tuple, list)):
            loss_value = loss_value[0]
        if hasattr(loss_value, "item"):
            loss_value = loss_value.item()

        learning_rate = None
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                if not param_group.get("is_decoupled_lr", False):
                    learning_rate = param_group["lr"]
                    break

        self.mllogger.log_event(
            key="train_loss",
            value=loss_value,
            metadata={constants.SAMPLES_COUNT: consumed_samples, "lr": learning_rate},
        )

    def _log_tracked_stats(self, iteration, consumed_samples):
        if self.throughput_timer is None:
            return
        throughput = self.throughput_timer.get_throughput()
        if throughput is not None and self.block_tput_log:
            self.mllogger.log_event(
                key="tracked_stats",
                value={"throughput": throughput},
                metadata={"step": consumed_samples},
            )

    def _get_validation_loss(self):
        return getattr(self, "last_validation_loss", None)
