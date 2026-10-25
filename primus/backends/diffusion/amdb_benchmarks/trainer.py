###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""AMDiffusionBenchmark trainer.

FPS, TFLOPS, and the BF16 FSDP2 policy stay in this package so the shared
Wan trainer is unchanged.
"""

from __future__ import annotations

import math
import os
import time
from importlib import import_module

import torch
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.flop_counter import FlopCounterMode

from primus.backends.diffusion.trainers.base import (
    ContiguousDistributedSampler,
    benchmark_step_rates,
)
from primus.backends.diffusion.trainers.fsdp2 import FSDP2Trainer
from primus.backends.diffusion.utils.log import logger
from primus.backends.diffusion.utils.train_utils import get_memory

try:
    import wandb
except ImportError:
    wandb = None


class AMDBBenchmarkTrainer(FSDP2Trainer):
    def _prepare_host_batch(self, batch):
        """Decode raw samples on the host before the step timer starts."""
        processor_hook = getattr(self.processing_class, "prepare_host_batch", None)
        if callable(processor_hook):
            return processor_hook(batch)
        core = getattr(self.model, "module", self.model)
        hook = getattr(core, "prepare_host_batch", None)
        if not callable(hook):
            return batch
        if not isinstance(batch, dict):
            prepare = getattr(self.processing_class, "prepare_batch", None)
            if not callable(prepare):
                return batch
            batch = prepare(batch=batch, device=self.device, dtype=self._resolve_dtype())
        return hook(batch)

    def _apply_fsdp2(self):
        """Apply torch.distributed._composable.fsdp.fully_shard to the model."""
        mp_dtype = self._resolve_dtype()
        # AMDiffusionBenchmark moves every frozen submodule to the training dtype.
        cast_frozen = getattr(self.model, "cast_frozen_modules", None)
        if callable(cast_frozen) and mp_dtype != torch.float32:
            cast_frozen(mp_dtype)
            if self.rank == 0:
                logger.info(f"Cast frozen encoder modules to {mp_dtype}")
        # AMDiffusionBenchmark casts the FSDP-wrapped denoiser to the training dtype, so
        # parameters, gradients, reductions, and AdamW states all stay in BF16.
        pure_low_precision = bool(self.args.get("fsdp2_pure_low_precision", False)) and (
            mp_dtype != torch.float32
        )
        default_reduce = "bf16" if pure_low_precision else "fp32"
        reduce_dtype_name = os.getenv("FSDP2_REDUCE_DTYPE", default_reduce).lower()
        if reduce_dtype_name not in {"fp32", "bf16"}:
            raise ValueError(f"Unsupported FSDP2_REDUCE_DTYPE={reduce_dtype_name!r}")
        reduce_dtype = torch.float32 if reduce_dtype_name == "fp32" else torch.bfloat16

        # Keep FP32 parameter/optimizer storage while allowing an explicit
        # reduction-dtype experiment around the FP32-qualified default.
        if mp_dtype != torch.float32:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=mp_dtype,
                reduce_dtype=reduce_dtype,
            )
        else:
            mp_policy = None

        # When SP is enabled, FSDP2 shards across dp_shard_sp (DP + SP combined).
        # This reduces per-rank parameter memory.
        fsdp_mesh = self.mesh
        if self.mesh is not None and self.sp_group is not None:
            try:
                fsdp_mesh = self.mesh["dp_shard_sp"]
            except KeyError:
                pass  # fallback to full mesh

        wrap_target = str(self.args.get("fsdp2_wrap_target", "") or "").strip()
        wrap_root = (
            self._get_module_by_path(self.model, wrap_target, option_name="fsdp2_wrap_target")
            if wrap_target
            else self.model
        )

        non_fp32_params = [
            name for name, param in wrap_root.named_parameters() if param.dtype != torch.float32
        ]
        if non_fp32_params:
            raise ValueError(
                "TorchTitan-aligned FSDP2 requires FP32 parameter storage before wrapping; "
                f"found non-FP32 parameters including {non_fp32_params[:3]}."
            )
        if pure_low_precision:
            wrap_root.to(dtype=mp_dtype)
            if self.rank == 0:
                logger.info(
                    f"FSDP2: cast '{wrap_target or '<model>'}' parameters to {mp_dtype} before sharding"
                )

        if self.world_size == 1:
            if self.rank == 0:
                logger.info("FSDP2: world_size=1; skipping composable FSDP wrapping.")
            return

        if bool(self.args.get("compile_transformer_blocks", False)):
            self._compile_transformer_blocks(wrap_root)

        reshard_after_forward = bool(self.args.get("fsdp2_reshard_after_forward", True))
        no_reshard_paths = {
            item.strip()
            for item in str(self.args.get("fsdp_module_paths_no_reshard", "") or "").split(",")
            if item.strip()
        }

        module_paths_spec = str(self.args.get("fsdp_module_paths_to_wrap", "") or "")
        module_paths = [item.strip() for item in module_paths_spec.split(",") if item.strip()]
        wrapped_paths = 0
        for module_path in module_paths:
            module = self._get_module_by_path(wrap_root, module_path, option_name="fsdp_module_paths_to_wrap")
            fully_shard(
                module,
                mesh=fsdp_mesh,
                reshard_after_forward=False if module_path in no_reshard_paths else reshard_after_forward,
                mp_policy=mp_policy,
            )
            wrapped_paths += 1
        if self.rank == 0 and wrapped_paths:
            logger.info(
                "FSDP2: wrapped explicit module paths under " f"'{wrap_target or '<model>'}': {module_paths}"
            )

        # Wrap transformer blocks first for optimal memory management
        layer_cls_spec = self.args.get("fsdp_transformer_layer_cls_to_wrap")
        if layer_cls_spec:
            if isinstance(layer_cls_spec, str):
                layer_cls_items = [x.strip() for x in layer_cls_spec.split(",") if x.strip()]
            else:
                layer_cls_items = [str(x).strip() for x in layer_cls_spec if str(x).strip()]

            cls_objs = set()
            cls_names = set()
            for item in layer_cls_items:
                if "." in item:
                    mod_path, cls_name = item.rsplit(".", 1)
                    cls_objs.add(getattr(import_module(mod_path), cls_name))
                else:
                    cls_names.add(item)

            wrapped_count = 0
            seen = set()
            for _, module in wrap_root.named_modules():
                if id(module) in seen:
                    continue
                seen.add(id(module))
                if module is wrap_root:
                    continue
                checkpointed_module = getattr(module, "_checkpoint_wrapped_module", None)
                target_module = checkpointed_module if checkpointed_module is not None else module
                if (cls_objs and isinstance(target_module, tuple(cls_objs))) or (
                    cls_names and target_module.__class__.__name__ in cls_names
                ):
                    fully_shard(
                        module,
                        mesh=fsdp_mesh,
                        reshard_after_forward=reshard_after_forward,
                        mp_policy=mp_policy,
                    )
                    if checkpointed_module is not None:
                        seen.add(id(checkpointed_module))
                    wrapped_count += 1

            if self.rank == 0:
                logger.info(f"FSDP2: wrapped {wrapped_count} submodules under '{wrap_target or '<model>'}'")

        fully_shard(
            wrap_root,
            mesh=fsdp_mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
        )
        if self.rank == 0:
            logger.info(
                f"FSDP2: applied fully_shard to '{wrap_target or '<model>'}' "
                f"with mp={mp_dtype}, reduce={reduce_dtype}"
            )

    def _log_step(
        self,
        loss_value: float | torch.Tensor,
        step_loss_value: float | torch.Tensor | None = None,
        grad_norm: float | torch.Tensor = 0.0,
        step_time: float | None = None,
        step_time_gpu: float | None = None,
        elapsed: float | None = None,
        eta_seconds: float | None = None,
        throughput_samples_per_gpu_s: float | None = None,
        fps_gpu: float | None = None,
        tflops_per_gpu: float | None = None,
    ):
        """Log training metrics. Format matches test regex expectations."""
        if self.rank != 0:
            return
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.item()
        if isinstance(step_loss_value, torch.Tensor):
            step_loss_value = step_loss_value.item()
        if step_loss_value is None:
            step_loss_value = loss_value
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()
        alloc, res, max_mem = get_memory()
        lr = self.optimizer.param_groups[0]["lr"]

        # AMDiffusionBenchmark prints the local rank loss as step_loss and the
        # all-reduced mean as train_loss. Those are different values.
        msg = (
            f"step={self.global_step} "
            f"step_loss={step_loss_value:.6f} train_loss={loss_value:.6f} "
            f"mem={alloc:.2f}/{res:.2f}GB peak_mem={max_mem:.2f}GB "
            f"gnorm={grad_norm:.4f}"
        )
        if step_time is not None:
            msg += f" step_time_total={step_time:.6f}s"
        if step_time_gpu is not None:
            msg += f" step_time_gpu={step_time_gpu:.6f}s"
        if fps_gpu is not None:
            msg += f" fps_gpu={fps_gpu:.4f}"
        if tflops_per_gpu is not None:
            msg += f" tflops/s={tflops_per_gpu:.4f}"
        if elapsed is not None:
            msg += f" elapsed={elapsed / 60:.2f}m"
        if eta_seconds is not None:
            msg += f" eta={eta_seconds / 60:.2f}m"
        logger.info(msg)

        if self.use_wandb:
            global_batch_size = (
                self.per_device_train_batch_size * self.grad_accum_steps * self.data_parallel_world_size
            )
            payload = {
                "train/loss": loss_value,
                "step_loss": step_loss_value,
                "train_loss": loss_value,
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
                payload["step_time_total"] = step_time
            if step_time_gpu is not None:
                payload["step_time_gpu"] = step_time_gpu
            if throughput_samples_per_gpu_s is not None:
                payload["perf/samples_per_gpu_s"] = throughput_samples_per_gpu_s
                payload["performance/throughput"] = throughput_samples_per_gpu_s * self.world_size
                payload["throughput(global_samples/s)"] = throughput_samples_per_gpu_s * self.world_size
            if fps_gpu is not None:
                payload["perf/fps_gpu"] = fps_gpu
            if tflops_per_gpu is not None:
                payload["perf/tflops_per_gpu"] = tflops_per_gpu
                payload["perf/tflops/s"] = tflops_per_gpu
                payload["tflops/s"] = tflops_per_gpu
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

        start_time = time.time()
        local_samples_in_update = 0
        update_loss_sum = None
        update_loss_count = 0
        mlperf_train_started = False
        count_flops = not bool(self.args.get("compile_transformer_blocks", False))
        flop_counter = FlopCounterMode(display=False, depth=None) if count_flops else None
        total_flops: float | None = None
        flops_captured = flop_counter is None
        flop_capture_open = False
        if flop_counter is None and self.rank == 0:
            logger.warning(
                "Disabling TFLOPS/GPU calculation because compile_transformer_blocks is enabled, "
                "matching AMDiffusionBenchmark."
            )
        # AMDiffusionBenchmark synchronizes all ranks before both timing events so each
        # step time excludes skew carried over from the previous step.
        barrier_around_timing = (
            not self.mlperf_enabled
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and self.world_size > 1
        )
        start_event_gpu = torch.cuda.Event(enable_timing=True)
        end_event_gpu = torch.cuda.Event(enable_timing=True)
        steps_per_epoch = math.ceil(len(self.dataloader) / max(1, self.grad_accum_steps))
        start_epoch = self.global_step // steps_per_epoch
        resume_batch_offset = (self.global_step % steps_per_epoch) * self.grad_accum_steps

        for epoch in range(start_epoch, self.num_train_epochs):
            self.sampler.set_epoch(epoch)
            if isinstance(self.sampler, ContiguousDistributedSampler):
                sample_offset = (
                    resume_batch_offset * self.per_device_train_batch_size if epoch == start_epoch else 0
                )
                self.sampler.set_offset(sample_offset)

            start_time_step = time.time()
            for batch_idx, batch in enumerate(self.dataloader):
                batch = self._prepare_host_batch(batch)
                if self.mlperf_enabled and not mlperf_train_started:
                    self._mlperf_log_train_start()
                    mlperf_train_started = True
                is_update_step = ((batch_idx + 1) % max(1, self.grad_accum_steps)) == 0
                starts_update = (batch_idx % max(1, self.grad_accum_steps)) == 0
                local_samples_in_update += self._infer_local_batch_size(batch)
                if starts_update:
                    if barrier_around_timing:
                        torch.distributed.barrier()
                    start_event_gpu.record()
                    if flop_counter is not None and not flops_captured:
                        flop_counter.__enter__()
                        flop_capture_open = True
                if is_update_step:
                    self._mlperf_log_block_start(self.global_step + 1)

                with self._grad_sync_context(is_update_step):
                    try:
                        raw_loss = self.compute_loss(batch)
                    except BaseException as exc:
                        if flop_capture_open:
                            flop_counter.__exit__(type(exc), exc, exc.__traceback__)
                            flop_capture_open = False
                            flops_captured = True
                        logger.exception(
                            "Training forward failed at step %d, batch %d: %r",
                            self.global_step,
                            batch_idx,
                            exc,
                        )
                        raise
                    detached_loss = raw_loss.detach().float()
                    update_loss_sum = (
                        detached_loss if update_loss_sum is None else update_loss_sum + detached_loss
                    )
                    update_loss_count += 1
                    loss = raw_loss / max(1, self.grad_accum_steps)
                    loss.backward()

                if is_update_step:
                    try:
                        loss_val = update_loss_sum / max(1, update_loss_count)
                        update_loss_sum = None
                        update_loss_count = 0
                        grad_norm = self._clip_grad_norm()
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        if flop_capture_open:
                            flop_counter.__exit__(None, None, None)
                            flop_capture_open = False
                            total_flops = float(flop_counter.get_total_flops())
                            flops_captured = True
                        if barrier_around_timing:
                            torch.distributed.barrier()
                        end_event_gpu.record()
                        torch.cuda.synchronize()
                        step_time_gpu = start_event_gpu.elapsed_time(end_event_gpu) / 1000.0
                        if step_time_gpu > 0 and local_samples_in_update > 0:
                            step_fps, step_tflops = benchmark_step_rates(
                                local_batch_size=local_samples_in_update,
                                step_time_gpu_s=step_time_gpu,
                                total_flops=0.0 if total_flops is None else total_flops,
                            )
                        else:
                            step_fps = None
                            step_tflops = None
                        # AMDiffusionBenchmark gathers the detached per-rank loss
                        # after CUDA-event timing; do likewise so metric reduction
                        # is excluded from ``step_time_gpu``.
                        global_loss = loss_val.detach().clone()
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            torch.distributed.all_reduce(global_loss, op=torch.distributed.ReduceOp.SUM)
                            global_loss.div_(self.world_size)
                        self.global_step += 1
                        self._mlperf_log_block_stop(self.global_step)
                        local_samples_in_update = 0

                        # AMDiffusionBenchmark emits one metric record per update.
                        step_time_total = time.time() - start_time_step
                        elapsed = time.time() - start_time
                        steps_left = max(0, self.total_steps - self.global_step)
                        eta_seconds = step_time_total * steps_left
                        self._log_step(
                            global_loss,
                            step_loss_value=loss_val,
                            grad_norm=grad_norm,
                            step_time=step_time_total,
                            step_time_gpu=step_time_gpu,
                            elapsed=elapsed,
                            eta_seconds=eta_seconds,
                            throughput_samples_per_gpu_s=step_fps,
                            fps_gpu=step_fps,
                            tflops_per_gpu=step_tflops if total_flops is not None else None,
                        )
                        start_time_step = time.time()
                    except BaseException as exc:
                        logger.exception(
                            "Training update failed after backward at step %s batch %s: %s",
                            self.global_step,
                            batch_idx,
                            type(exc).__name__,
                        )
                        raise

                    # Periodic save
                    if self.save_steps > 0 and self.global_step % self.save_steps == 0:
                        self._save_checkpoint()

                    if self.mlperf_enabled and self.global_step % self.mlperf_eval_freq_steps == 0:
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
                                if val_loss <= self.mlperf_target_eval_loss and self.mlperf_train_start_time:
                                    payload["time_metrics/time_to_converge(s)"] = (
                                        time.time() - self.mlperf_train_start_time
                                    )
                                wandb.log(payload, step=self.global_step)
                        if val_loss <= self.mlperf_target_eval_loss:
                            self.mlperf_run_success = True
                            if self.rank == 0 and self.mlperf_train_start_time:
                                time_to_converge = time.time() - self.mlperf_train_start_time
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
                            return

                    # Early termination
                    if self.max_steps > 0 and self.global_step >= self.max_steps:
                        self._mlperf_log_run_stop()
                        return

            if self.max_steps > 0 and self.global_step >= self.max_steps:
                break

        if self.rank == 0:
            elapsed = time.time() - start_time
            logger.info(f"Training finished in {elapsed / 60:.2f} min")
        self._mlperf_log_run_stop()


def build_amdb_benchmark_trainer(
    *, model, dataset, processor, trainer_args: dict, eval_dataset=None, eval_processor=None
):
    from primus.backends.diffusion.distributed import setup_distributed

    rank, world_size, local_rank = setup_distributed()
    return AMDBBenchmarkTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset,
        data_collator=dataset.get_collator(),
        processing_class=processor,
        eval_dataset=eval_dataset,
        eval_processor=eval_processor,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
