###############################################################################
# Copyright 2023–2025 Google LLC. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unified PrimusMetricLogger that works with both the Dec-version MaxText
(which exposes the ``MetadataKey`` enum) and the Aug-version (which uses
plain string keys for metadata).
"""

import os

import jax
import numpy as np
from MaxText import max_logging, max_utils, maxtext_utils
from MaxText.metric_logger import MetricLogger

try:
    from MaxText.metric_logger import MetadataKey

    _TFLOPS_KEY = MetadataKey.PER_DEVICE_TFLOPS
    _TOKENS_KEY = MetadataKey.PER_DEVICE_TOKENS
except ImportError:
    _TFLOPS_KEY = "per_device_tflops"
    _TOKENS_KEY = "per_device_tokens"


# ---------------------------------------------------------------------------
# WandB helpers (moved here from max_utils – only used by PrimusMetricLogger)
# ---------------------------------------------------------------------------


def _safe_get_config(config, key, default=None):
    try:
        return getattr(config, key)
    except (KeyError, AttributeError):
        return default


def initialize_wandb_writer(config):
    if jax.process_index() != 0 or not config.enable_wandb:
        return None

    import wandb

    wandb_save_dir = _safe_get_config(config, "wandb_save_dir")
    if not wandb_save_dir:
        wandb_save_dir = os.path.join(config.base_output_directory, "wandb")

    wandb_project = _safe_get_config(config, "wandb_project")
    if not wandb_project:
        wandb_project = os.getenv("WANDB_PROJECT", "Primus-MaxText-Pretrain")

    wandb_exp_name = _safe_get_config(config, "wandb_exp_name")
    if not wandb_exp_name:
        wandb_exp_name = config.run_name

    if "WANDB_API_KEY" not in os.environ:
        max_logging.log(
            "The environment variable WANDB_API_KEY is not set. "
            "Please set it or login wandb before proceeding"
        )
        return None

    os.makedirs(wandb_save_dir, exist_ok=True)

    wandb.init(
        project=wandb_project,
        name=wandb_exp_name,
        dir=wandb_save_dir,
        config=dict(config.get_keys()),
    )
    max_logging.log(f"WandB logging enabled: {wandb_save_dir=}, {wandb_project=}, {wandb_exp_name=}")
    return wandb


def close_wandb_writer(wandb_writer):
    if jax.process_index() == 0 and wandb_writer is not None:
        wandb_writer.finish()


# ---------------------------------------------------------------------------
# PrimusMetricLogger
# ---------------------------------------------------------------------------


class PrimusMetricLogger(MetricLogger):
    """
    Logger for saving metrics to a local file, GCS and TensorBoard/WandB.
    """

    def __init__(self, config, learning_rate_schedule):
        super().__init__(config, learning_rate_schedule)
        self.wandb_writer = None
        if self.config.enable_wandb:
            self.wandb_writer = initialize_wandb_writer(config)

    def write_metrics(self, metrics, step, is_training=True):
        """Entry point for all metrics writing in Train's Main."""
        super().write_metrics(metrics, step, is_training)
        if self.config.enable_wandb:
            self.write_metrics_to_wandb(metrics, step, is_training)

    def write_metrics_to_wandb(self, metrics, step, is_training):
        """Writes metrics to WandB."""
        if jax.process_index() != 0 or self.wandb_writer is None:
            return

        log_dict = {}

        for name in metrics.get("scalar", []):
            log_dict[name] = float(np.array(metrics["scalar"][name]))

        for name in metrics.get("scalars", []):
            for k, v in metrics["scalars"][name].items():
                log_dict[f"{name}/{k}"] = float(v)

        self.wandb_writer.log(log_dict, step=step)

    def write_setup_info_to_tensorboard(self, params):
        """Writes setup information like train config params, num model params, and XLA flags to TensorBoard."""
        num_model_parameters = max_utils.calculate_num_params_from_pytree(params)
        self.metadata[_TFLOPS_KEY], _, _ = maxtext_utils.calculate_tflops_training_per_device(self.config)
        self.metadata[_TOKENS_KEY] = maxtext_utils.calculate_tokens_training_per_device(self.config)
        max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")
        max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), self.writer)
        max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], self.writer)
        maxtext_utils.add_config_to_summary_writer(self.config, self.writer)

        if self.wandb_writer is not None:
            self.wandb_writer.log({"num_model_parameters": num_model_parameters})

    def flush_metrics_and_cleanup(self):
        super().flush_metrics_and_cleanup()
        if self.config.enable_wandb:
            close_wandb_writer(self.wandb_writer)
