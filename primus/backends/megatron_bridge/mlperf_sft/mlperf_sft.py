###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MLPerf logging support for Primus SFT (Supervised Fine-Tuning) workloads.

Standalone lifecycle hooks for Megatron-Bridge training loops (e.g.
``llama2_custom.py``).  Pretrain workloads use
:class:`~primus.backends.megatron.mlperf.mlperf_pretrain_trainer.MLPerfMegatronPretrainTrainer`.
"""

import os
import time
from typing import Any, Dict, Optional

import numpy as np

from .mlperf_logger import MLPerfLogger


class MLPerfSFTLogger:
    """Standalone MLPerf logger for SFT / fine-tuning workloads.

    Composes :class:`MLPerfLogger` (rank-aware mllog wrapper) with an inline
    wall-clock throughput timer.  All ``:::MLLOG`` emission is handled by
    the lifecycle hooks listed below.

    Typical call order from the training code::

        logger = MLPerfSFTLogger(global_batch_size=8, micro_batch_size=1)
        logger.log_cache_clear_and_init_start()
        logger.log_init_params(config_dict)
        # ... Megatron-Bridge init completes ...
        logger.log_init_stop_run_start()
        for step in training_loop:
            logger.on_train_step(step, loss_dict, lr, consumed_samples)
            if eval_step:
                logger.on_eval_start(consumed_samples)
                val_loss = run_eval()
                should_stop = logger.on_eval_end(consumed_samples, val_loss)
        logger.log_run_stop(consumed_samples)
    """

    def __init__(
        self,
        global_batch_size: int,
        micro_batch_size: int,
        target_eval_loss: Optional[float] = None,
        train_loss_log_freq: Optional[int] = None,
    ):
        self._logger = MLPerfLogger()
        self._gbs = global_batch_size
        self._mbs = micro_batch_size
        self._target_eval_loss = target_eval_loss or float(os.getenv("MLLOG_TARGET_EVAL_LOSS", "0.925"))
        self._train_loss_log_freq = train_loss_log_freq or int(os.getenv("MLLOG_TRAIN_LOSS_LOG_FREQ", "10"))
        self._block_start_time: Optional[float] = None
        self._block_start_samples: int = 0
        self._is_target_reached = False
        self._run_start_time: Optional[float] = None

    @property
    def is_target_reached(self) -> bool:
        return self._is_target_reached

    def log_cache_clear_and_init_start(self) -> None:
        """Emit ``CACHE_CLEAR`` (all ranks) and ``INIT_START`` (all ranks)."""
        from mlperf_logging.mllog import constants

        self._logger.log_event_all_ranks(key=constants.CACHE_CLEAR, value=True)
        self._logger.log_event_all_ranks(key=constants.INIT_START, value=None)

    def log_init_params(self, config: Dict[str, Any]) -> None:
        """Log submission metadata and hyper-parameters before training starts."""
        from mlperf_logging.mllog import constants

        output_file = os.getenv("MLLOG_OUTPUT_FILE", "/results/mlperf_output.log")
        from mlperf_logging import mllog

        save_to_file = os.getenv("MLLOG_SAVE_TO_FILE", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        if save_to_file:
            out_dir = os.path.dirname(output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            mllog.config(filename=output_file, default_stack_offset=3)
        else:
            mllog.config(default_stack_offset=3)
        self._logger._configured = True
        self._logger._rank = self._logger._get_rank()

        self._logger.log_event(
            key=constants.SUBMISSION_BENCHMARK,
            value=os.getenv("MLLOG_SUBMISSION_BENCHMARK", "llama2_70b_lora"),
        )
        self._logger.log_event(
            key=constants.SUBMISSION_ORG,
            value=os.getenv("MLLOG_SUBMISSION_ORG", "AMD"),
        )
        self._logger.log_event(
            key=constants.SUBMISSION_DIVISION,
            value=os.getenv("MLLOG_SUBMISSION_DIVISION", "closed"),
        )
        self._logger.log_event(
            key=constants.SUBMISSION_PLATFORM,
            value=os.getenv("MLLOG_SUBMISSION_PLATFORM", "MI355X"),
        )
        self._logger.log_event(key=constants.SUBMISSION_STATUS, value="onprem")
        self._logger.log_event(key="target_accuracy", value=self._target_eval_loss)

        _KEY_MAP = {
            "global_batch_size": constants.GLOBAL_BATCH_SIZE,
            "train_samples": constants.TRAIN_SAMPLES,
            "eval_samples": constants.EVAL_SAMPLES,
            "gradient_accumulation_steps": constants.GRADIENT_ACCUMULATION_STEPS,
            "seed": constants.SEED,
            "opt_name": constants.OPT_NAME,
            "opt_base_lr": constants.OPT_BASE_LR,
            "opt_adamw_beta_1": constants.OPT_ADAMW_BETA_1,
            "opt_adamw_beta_2": constants.OPT_ADAMW_BETA_2,
            "opt_adamw_epsilon": constants.OPT_ADAMW_EPSILON,
            "opt_weight_decay": constants.OPT_ADAMW_WEIGHT_DECAY,
            "opt_gradient_clip_norm": "opt_gradient_clip_norm",
            "opt_lr_warmup_factor": constants.OPT_LR_WARMUP_FACTOR,
            "opt_lr_training_steps": constants.OPT_LR_TRAINING_STEPS,
            "max_sequence_length": "max_sequence_length",
            "max_steps": "max_steps",
            "lora_rank": "lora_rank",
            "lora_alpha": "lora_alpha",
            "tensor_parallelism": constants.TENSOR_PARALLELISM,
            "pipeline_parallelism": constants.PIPELINE_PARALLELISM,
            "context_parallelism": constants.CONTEXT_PARALLELISM,
            "expert_parallelism": constants.EXPERT_PARALLELISM,
            "micro_batch_size": constants.MICRO_BATCH_SIZE,
            "config_filename": constants.CONFIG_FILENAME,
            "lowest_numerical_precision_linear": "lowest_numerical_precision_linear",
        }
        for cfg_key, mllog_key in _KEY_MAP.items():
            if cfg_key in config and config[cfg_key] is not None and config[cfg_key] != "":
                self._logger.log_event(key=mllog_key, value=config[cfg_key])

    def log_init_stop_run_start(self) -> None:
        """Emit ``INIT_STOP``, ``RUN_START``, ``EPOCH_START``, ``BLOCK_START``."""
        from mlperf_logging.mllog import constants

        self._logger.log_end(key=constants.INIT_STOP)
        self._logger.log_start(key=constants.RUN_START)
        self._run_start_time = time.time()
        self._logger.log_start(key=constants.EPOCH_START, metadata={constants.SAMPLES_COUNT: 0})
        self._logger.log_start(key=constants.BLOCK_START, metadata={constants.SAMPLES_COUNT: 0})
        self._block_start_time = time.time()
        self._block_start_samples = 0

    def on_train_step(
        self,
        step: int,
        loss_dict: Dict[str, Any],
        lr: Optional[float],
        consumed_samples: int,
    ) -> None:
        if self._train_loss_log_freq == 0 or step % self._train_loss_log_freq != 0:
            return

        from mlperf_logging.mllog import constants

        loss_value = loss_dict.get("lm loss")
        if loss_value is None:
            return
        if isinstance(loss_value, (tuple, list)):
            loss_value = loss_value[0]
        if hasattr(loss_value, "item"):
            loss_value = loss_value.item()

        self._logger.log_event(
            key="train_loss",
            value=loss_value,
            metadata={constants.SAMPLES_COUNT: consumed_samples, "lr": lr},
        )

    def on_eval_start(self, consumed_samples: int) -> None:
        from mlperf_logging.mllog import constants

        self._logger.log_end(
            key=constants.BLOCK_STOP,
            metadata={constants.SAMPLES_COUNT: consumed_samples},
        )
        self._logger.log_start(
            key=constants.EVAL_START,
            metadata={constants.SAMPLES_COUNT: consumed_samples},
        )

    def on_eval_end(self, consumed_samples: int, eval_loss: float) -> bool:
        from mlperf_logging.mllog import constants

        self._logger.log_event(
            key=constants.EVAL_ACCURACY,
            value=eval_loss,
            metadata={constants.SAMPLES_COUNT: consumed_samples},
        )

        if (
            self._target_eval_loss > 0.0
            and eval_loss <= self._target_eval_loss
            and not self._is_target_reached
        ):
            self._is_target_reached = True

        self._logger.log_end(
            key=constants.EVAL_STOP,
            metadata={constants.SAMPLES_COUNT: consumed_samples},
        )

        if not self._is_target_reached:
            self._logger.log_start(
                key=constants.BLOCK_START,
                metadata={constants.SAMPLES_COUNT: consumed_samples},
            )
            self._block_start_time = time.time()
            self._block_start_samples = consumed_samples

        return self._is_target_reached

    def log_run_stop(self, consumed_samples: int) -> None:
        from mlperf_logging.mllog import constants

        self._logger.log_end(
            key=constants.EPOCH_STOP,
            metadata={constants.SAMPLES_COUNT: consumed_samples},
        )

        if not self._is_target_reached:
            self._logger.log_end(
                key=constants.RUN_STOP,
                metadata={
                    constants.SAMPLES_COUNT: consumed_samples,
                    constants.STATUS: constants.ABORTED,
                },
            )
        else:
            self._logger.log_end(
                key=constants.RUN_STOP,
                metadata={
                    constants.SAMPLES_COUNT: consumed_samples,
                    constants.STATUS: constants.SUCCESS,
                },
            )

        if self._run_start_time is not None:
            duration = time.time() - self._run_start_time
            duration_minutes = duration / 60.0
            overall_throughput = consumed_samples / duration if duration > 0 else 0.0
            self._logger.log_event(
                key="run_duration",
                value=f"{round(duration, 2)}s -> {round(duration_minutes, 2)} minutes",
                metadata={"samples": consumed_samples},
            )
            self._logger.log_event(
                key="throughput",
                value=round(overall_throughput, 2),
                metadata={"samples": consumed_samples},
            )

    @staticmethod
    def extract_sft_configs(
        train_gbs: int,
        train_mbs: int,
        train_iters: int,
        eval_iters: int,
        seq_length: int,
        seed: int,
        lr: float,
        weight_decay: float,
        clip_grad: float,
        lr_warmup_iters: int,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        data_root: str = "/data",
        data_parallel_size: Optional[int] = None,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        config_filename: str = "",
        lowest_numerical_precision_linear: str = "",
    ) -> Dict[str, Any]:
        train_samples = int(np.load(os.path.join(data_root, "train.npy"), allow_pickle=True).shape[0])
        eval_samples = int(np.load(os.path.join(data_root, "validation.npy"), allow_pickle=True).shape[0])

        if data_parallel_size is not None and data_parallel_size > 0:
            dp_size = data_parallel_size
        else:
            dp_size = 1
            try:
                import torch

                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    from megatron.core import parallel_state

                    dp_size = parallel_state.get_data_parallel_world_size()
            except Exception:
                pass
            if dp_size == 0:
                dp_size = 1
        grad_accum_steps = train_gbs // (train_mbs * dp_size)

        warmup_factor = lr_warmup_iters / train_iters if train_iters > 0 else 0.0

        return {
            "global_batch_size": train_gbs,
            "train_samples": train_samples,
            "eval_samples": eval_samples,
            "gradient_accumulation_steps": grad_accum_steps,
            "seed": seed,
            "max_sequence_length": seq_length,
            "max_steps": train_iters,
            "opt_name": "adamw",
            "opt_base_lr": lr,
            "opt_weight_decay": weight_decay,
            "opt_gradient_clip_norm": clip_grad,
            "opt_lr_warmup_factor": warmup_factor,
            "opt_lr_training_steps": train_iters,
            "opt_adamw_beta_1": adam_beta1,
            "opt_adamw_beta_2": adam_beta2,
            "opt_adamw_epsilon": adam_eps,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "tensor_parallelism": tensor_model_parallel_size,
            "pipeline_parallelism": pipeline_model_parallel_size,
            "context_parallelism": context_parallel_size,
            "expert_parallelism": 1,
            "micro_batch_size": train_mbs,
            "config_filename": config_filename,
            "lowest_numerical_precision_linear": lowest_numerical_precision_linear,
        }
