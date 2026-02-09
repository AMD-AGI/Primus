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
    sharding,
    train_utils,
)
from MaxText.common_types import ShardMode
from MaxText.metric_logger import MetricLogger
from MaxText.train import (
    _merge_dpo_state,
    _split_dpo_state,
    eval_step,
    get_first_step,
    train_step,
)
from MaxText.train_utils import validate_train_config
from MaxText.utils import gcs_utils
from MaxText.utils.goodput_utils import (
    GoodputEvent,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
)
from MaxText.vertex_tensorboard import VertexTensorboardManager


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
      data_loader,
      rampup_manager,
      eval_data_iterator,
      state,
  ) = train_utils.setup_train_loop(config, recorder)

  if config.use_dpo:
    if "reference_params" not in state.params:
      reference_params = jax.tree.map(jnp.copy, state.params["params"])
      state = _merge_dpo_state(state, reference_params)
    state_mesh_shardings = _merge_dpo_state(state_mesh_shardings, state_mesh_shardings.params["params"])

  params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)

  p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
      config,
      model,
      mesh,
      state,
      state_mesh_shardings,
      train_step,
      eval_step,
      eval_data_iterator,
      params_shardings,
  )

  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    shaped_batch = maxtext_utils.get_shaped_batch(config)
    if config.shard_optimizer_over_data:
      state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
    if config.compiled_trainstep_file == "":  # compile only when there is no pre-compiled file loaded
      compiled = p_train_step.lower(state, shaped_batch, init_rng).compile()
      compiled_stats = compiled.memory_analysis()
      max_utils.print_compiled_memory_stats(compiled_stats)

  start_step = get_first_step(state)  # this is the start_step for training
  prof = profiler.Profiler(config, offset_step=start_step)
  metric_logger = MetricLogger(config=config, learning_rate_schedule=learning_rate_schedule)

  # Write train config params, num model params, and XLA flags to tensorboard
  metric_logger.write_setup_info_to_tensorboard(state.params)

  # Synchronize all hosts before entering the training loop.
  # Without this barrier, timing variance during initialization (JIT compilation,
  # profiler/logger setup, etc.) causes hosts to enter the training loop at different
  # times. The first collective operation (data sharding in load_next_batch) then
  # times out waiting for straggler hosts, resulting in "collective operation timeout"
  # or "stop sending heartbeats" errors.
  max_logging.log("====== BARRIER: Synchronizing hosts before training loop ======")
  jax.experimental.multihost_utils.sync_global_devices("sync_before_training_loop")
  max_logging.log("====== BARRIER PASSED: Starting training loop ======")

  try:
    last_step_completion = datetime.datetime.now()
    for step in np.arange(start_step, config.steps):
      prof.maybe_activate_profiler(step, state)

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        example_batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
        # Reshard data from loaded sharding to performant activation sharding
        example_batch = sharding.maybe_shard_with_name(
            example_batch,
            sharding.get_input_data_sharding(config, mesh),
            shard_mode=config.shard_mode,
        )
        # pylint: disable=not-callable
        nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
        with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
          with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            if config.shard_optimizer_over_data:
              state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
            state, metrics = p_train_step(state, example_batch, nextrng)
          jax.block_until_ready(state)

      step_time_delta = datetime.datetime.now() - last_step_completion
      last_step_completion = datetime.datetime.now()

      state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
      checkpointing.maybe_save_checkpoint(checkpoint_manager, state_to_save, config, data_iterator, step)

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
        # Explicitly reset the eval iterator and counters before starting the eval loop
        eval_data_iterator.reset()
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
        if metric_logger.cumulative_eval_metrics["scalar"]["eval/avg_loss"] <= config.target_eval_loss:
          prof.deactivate()
          raise exceptions.StopTraining(f"Target loss {config.target_eval_loss=} is achieved.")

      prof.maybe_deactivate_profiler(step, state)

      if step == start_step:
        max_utils.print_mem_stats("After params initialized")

      metric_logger.buffer_and_write_train_metrics(metrics, step, step_time_delta)

    if config.save_checkpoint_on_completion:
      state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
      checkpointing.maybe_save_checkpoint(checkpoint_manager, state_to_save, config, data_iterator)
    if checkpoint_manager is not None:
      # in case the last checkpoint_period checkpoint is still in progress
      checkpoint_manager.wait_until_finished()
  except exceptions.StopTraining as e:
    max_logging.log(f"Training stopped: {str(e)}")
  finally:
    metric_logger.flush_metrics_and_cleanup()

  return state


def initialize(argv: Sequence[str]) -> tuple[pyconfig.HyperParameters, Any, Any]:
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
  config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  validate_train_config(config)
  max_utils.save_device_information(config)
  jax.config.update("jax_use_shardy_partitioner", config.shardy)
  # update explicit sharding-supported config
  if config.shard_mode == ShardMode.EXPLICIT:
    jax.config.update("jax_remove_size_one_mesh_axis_from_type", True)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path or ""
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)

  # Create the Goodput recorder
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
  with (
    diagnostic.diagnose(diagnostic_config),
    maybe_record_goodput(recorder, GoodputEvent.JOB),
    max_utils.maybe_get_transformer_engine_context(config),
    maybe_monitor_goodput(config),
  ):
    train_loop(config, recorder)
