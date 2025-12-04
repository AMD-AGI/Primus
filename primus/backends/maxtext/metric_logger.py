###############################################################################
# Copyright 2023–2025 Google LLC. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

import jax
import numpy as np
from MaxText import max_logging, max_utils, maxtext_utils
from MaxText.metric_logger import MetricLogger

from .max_utils import close_wandb_writer, initialize_wandb_writer


class PrimusMetricLogger(MetricLogger):
    """
    Logger for saving metrics to a local file, GCS and TensorBoard/WandB.
    """

    def __init__(self, config, learning_rate_schedule):
        super().__init__(config, learning_rate_schedule)
        self.wandb_writer = initialize_wandb_writer(config)

    def write_metrics(self, metrics, step, is_training=True):
        """Entry point for all metrics writing in Train's Main."""
        super().write_metrics(metrics, step, is_training)
        if self.config.enable_wandb:
            self.write_metrics_to_wandb(metrics, step, is_training)

    def write_metrics_to_wandb(self, metrics, step, is_training):
        """Writes metrics to WandB."""
        if jax.process_index() != 0:
            return

        log_dict = {}

        for name in metrics.get("scalar", []):
            log_dict[name] = float(np.array(metrics["scalar"][name]))

        for name in metrics.get("scalars", []):
            # multi scalers flatten
            for k, v in metrics["scalars"][name].items():
                log_dict[f"{name}/{k}"] = float(v)

        self.wandb_writer.log(log_dict, step=step)

    def write_setup_info_to_tensorboard(self, params):
        """Writes setup information like train config params, num model params, and XLA flags to TensorBoard."""
        num_model_parameters = max_utils.calculate_num_params_from_pytree(params)
        self.metadata["per_device_tflops"], _, _ = maxtext_utils.calculate_tflops_training_per_device(
            self.config
        )
        self.metadata["per_device_tokens"] = maxtext_utils.calculate_tokens_training_per_device(self.config)
        max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")
        max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), self.writer)
        max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], self.writer)
        maxtext_utils.add_config_to_summary_writer(self.config, self.writer)

        if self.config.enable_wandb:
            self.wandb_writer.log({"num_model_parameters": str(num_model_parameters)})

    def flush_metrics_and_cleanup(self):
        super().flush_metrics_and_cleanup()
        close_wandb_writer(self.wandb_writer)
