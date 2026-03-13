###############################################################################
# Copyright 2023–2025 Google LLC. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import datetime
import os
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pathwaysutils  # pylint: disable=unused-import
import tensorflow as tf
from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import (
    debug_configuration,
    diagnostic_configuration,
    stack_trace_configuration,
)
from flax.linen import partitioning as nn_partitioning
from MaxText import (
    checkpointing,
    exceptions,
    max_logging,
    max_utils,
    maxtext_utils,
    profiler,
    pyconfig,
    train_utils,
)
from MaxText.data_loader import DataLoader
from MaxText.metric_logger import MetricLogger
from MaxText.train import (
    _merge_dpo_state,
    _split_dpo_state,
    eval_step,
    get_first_step,
    setup_train_loop,
    train_step,
)
from MaxText.utils import gcs_utils
from MaxText.utils.goodput_utils import (
    GoodputEvent,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
)
from MaxText.vertex_tensorboard import VertexTensorboardManager


def validate_train_config(config):
    """Validates the configuration is set correctly for 'train.py'."""

    assert config.run_name, "Erroring out, need a real run_name"
    if config.dataset_path and not config.dataset_path.startswith("gs://"):
        max_logging.log("WARNING: 'dataset_path' might be pointing your local file system")
    if not config.base_output_directory.startswith("gs://"):
        max_logging.log("WARNING: 'base_output_directory' might be pointing your local file system")
    assert config.steps > 0, "You must set steps or learning_rate_schedule_steps to a positive integer."

    if config.quantization in ("fp8", "nanoo_fp8"):
        # pylint: disable=line-too-long
        assert config.gradient_accumulation_steps == 1, (
            "fp8 can't be used with gradient_accumulation_steps right now. Please use other quantization or set "
            "gradient_accumulation_steps to 1"
        )

    # Check if GPU Flash Attention is being used with sequence packing
    # if config.attention == "cudnn_flash_te" and config.packing and config.dataset_type != "synthetic":
    #  raise ValueError(
    #      "cudnn_flash_te only supports BSHD format. The THD (seq packing) support is going to be available in "
    #      "Transformer Engine 2.0 release. "
    #      "Please disable sequence packing (set packing=False) or use a different attention mechanism. "
    #      "With synthetic data, the format is not important as packing is not applied."
    #  )


def train_loop(config, recorder, state=None):
    """Main Training loop."""
    (
        init_rng,
        checkpoint_manager,
        state_mesh_shardings,
        model,
        mesh,
        learning_rate_schedule,
        data_iterator,
        eval_data_iterator,
        state,
    ) = setup_train_loop(config, recorder)

    if config.use_dpo:
        if "reference_params" not in state.params:
            reference_params = jax.tree.map(jnp.copy, state.params["params"])
            state = _merge_dpo_state(state, reference_params)
        state_mesh_shardings = _merge_dpo_state(state_mesh_shardings, state_mesh_shardings.params["params"])

    p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
        config, model, mesh, state, state_mesh_shardings, train_step, eval_step, eval_data_iterator
    )

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        shaped_batch = maxtext_utils.get_shaped_batch(config)
        compiled = p_train_step.lower(state, shaped_batch, init_rng).compile()
        compiled_stats = compiled.memory_analysis()
        max_utils.print_compiled_memory_stats(compiled_stats)

    start_step = get_first_step(state)  # this is the start_step for training
    prof = profiler.Profiler(config, offset_step=start_step)
    data_loader = DataLoader(config, mesh, data_iterator, recorder)
    metric_logger = MetricLogger(config=config, learning_rate_schedule=learning_rate_schedule)

    # Write train config params, num model params, and XLA flags to tensorboard
    metric_logger.write_setup_info_to_tensorboard(state.params)

    try:
        last_step_completion = datetime.datetime.now()
        for step in np.arange(start_step, config.steps):
            prof.maybe_activate_profiler(step, state)

            with jax.profiler.StepTraceAnnotation("train", step_num=step):
                example_batch = data_loader.load_next_batch()
                # pylint: disable=not-callable
                nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
                with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
                    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
                        state, metrics = p_train_step(state, example_batch, nextrng)
                    jax.block_until_ready(state)

            step_time_delta = datetime.datetime.now() - last_step_completion
            last_step_completion = datetime.datetime.now()

            state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
            checkpointing.maybe_save_checkpoint(
                checkpoint_manager, state_to_save, config, data_iterator, step
            )

            if config.dump_hlo and step == (config.dump_step if config.dump_step >= 0 else start_step):
                jax.block_until_ready(state)  # Ensure compilation has finished.
                gcs_utils.upload_dump(
                    config.dump_hlo_local_dir,
                    config.dump_hlo_gcs_dir,
                    module_name=config.dump_hlo_module_name,
                    delete_local_after=config.dump_hlo_delete_local_after,
                    all_host_upload=config.dump_hlo_upload_all,
                )

            if config.eval_interval > 0 and step > start_step and (step + 1) % config.eval_interval == 0:
                assert eval_data_iterator

                # Explicitly reset the eval counters before starting the eval loop
                metric_logger.reset_eval_metrics()

                eval_step_count = 0
                # pylint: disable=not-callable
                for eval_batch in eval_data_iterator:
                    if config.eval_steps > 0 and eval_step_count >= config.eval_steps:
                        break
                    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
                        eval_metrics = p_eval_step(state, eval_batch, nextrng)
                    metric_logger.record_eval_metrics(step, metrics=eval_metrics)
                    max_logging.log(f"Completed eval step {eval_step_count}")
                    eval_step_count += 1
                metric_logger.record_eval_metrics(step, eval_step_count=eval_step_count)
                if (
                    metric_logger.cumulative_eval_metrics["scalar"]["eval/avg_loss"]
                    <= config.target_eval_loss
                ):
                    prof.deactivate()
                    raise exceptions.StopTraining(f"Target loss {config.target_eval_loss=} is achieved.")

            prof.maybe_deactivate_profiler(step, state)

            if step == start_step:
                max_utils.print_mem_stats("After params initialized")

            metric_logger.buffer_and_write_train_metrics(metrics, step, step_time_delta)

        state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
        checkpointing.maybe_save_checkpoint(checkpoint_manager, state_to_save, config, data_iterator)
    except exceptions.StopTraining as e:
        max_logging.log(f"Training stopped: {str(e)}")
    finally:
        metric_logger.flush_metrics_and_cleanup()

    return state


def initialize(argv: Sequence[str], **kwargs) -> tuple[pyconfig.HyperParameters, Any, Any]:
    """Initialization of hyperparameters and utilities"""
    pathwaysutils.initialize()
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    # TF allocates extraneous GPU memory when using TFDS data
    # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
    tf.config.set_visible_devices([], "GPU")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
        os.environ["LIBTPU_INIT_ARGS"] = (
            os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
        )
    # TODO: mazumdera@ : ensure missing mandatory fields in base.yml are filled in in argv,
    # or fill in here
    config = pyconfig.initialize(argv, **kwargs)
    jax.config.update("jax_use_shardy_partitioner", config.shardy)
    max_utils.print_system_information()
    validate_train_config(config)
    max_utils.save_device_information(config)
    os.environ["TFDS_DATA_DIR"] = config.dataset_path or ""
    vertex_tensorboard_manager = VertexTensorboardManager()
    if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
        vertex_tensorboard_manager.configure_vertex_tensorboard(config)

    # Goodput configurations
    maybe_monitor_goodput(config)
    recorder = create_goodput_recorder(config)

    # Stack traces configurations
    debug_config = debug_configuration.DebugConfig(
        stack_trace_config=stack_trace_configuration.StackTraceConfig(
            collect_stack_trace=config.collect_stack_trace,
            stack_trace_to_cloud=config.stack_trace_to_cloud,
            stack_trace_interval_seconds=config.stack_trace_interval_seconds,
        )
    )
    diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
    return config, recorder, diagnostic_config


def run(config, recorder, diagnostic_config):
    """Run the job given hyperparameters and utilities"""
    with diagnostic.diagnose(diagnostic_config):
        with maybe_record_goodput(recorder, GoodputEvent.JOB):
            train_loop(config, recorder)
