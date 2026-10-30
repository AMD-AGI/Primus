###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""AMDiffusionBenchmark trainer.

FPS and TFLOPS stay in this trainer. The BF16 FSDP2 policy is applied
through hooks on ``FSDP2Trainer._apply_fsdp2`` so wrapping fixes stay shared.
"""

from __future__ import annotations

import math
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from primus.backends.diffusion.trainers.base import (
    ContiguousDistributedSampler,
    EvenSizeBatchSampler,
    benchmark_step_rates,
)
from primus.backends.diffusion.trainers.fsdp2 import FSDP2Trainer
from primus.backends.diffusion.utils.log import logger
from primus.backends.diffusion.utils.train_utils import get_memory

try:
    import wandb
except ImportError:
    wandb = None


def _collate_encoded(samples: list[dict]) -> dict:
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in samples[0] if key != "prompt"}
    if "prompt" in samples[0]:
        batch["prompts"] = [sample["prompt"] for sample in samples]
    return batch


class AMDBBenchmarkTrainer(FSDP2Trainer):
    _encoded_cache = False

    def _cache_encoded_batches(self) -> None:
        """Run the frozen encoders once over this rank's shard, as AMDiffusionBenchmark does.

        The timed steps then see only the denoiser, so FPS and TFLOPS cover the same
        work as AMDiffusionBenchmark's cached-latent runs.
        """
        core = getattr(self.model, "module", self.model)
        encode = getattr(core, "encode_batch", None)
        if not callable(encode):
            return
        cache_start = time.time()
        self.sampler.set_epoch(0)
        samples: list[dict] = []
        with torch.no_grad():
            for batch in self.dataloader:
                batch = self._prepare_batch(self._prepare_host_batch(batch), self.processing_class)
                prompts = list(batch.get("prompts") or [])
                if (
                    isinstance(batch, dict)
                    and "image" not in batch
                    and (isinstance(batch.get("latents"), torch.Tensor) or "mean" in batch)
                ):
                    if self.rank == 0:
                        logger.info(
                            "Skipping latent caching; samples are already encoded. "
                            "Precomputed latents stay resampled on each step."
                        )
                    return
                encoded = encode(batch)
                if encoded is None:
                    return
                count = next(iter(encoded.values())).shape[0]
                for index in range(count):
                    sample = {key: value[index].detach().cpu() for key, value in encoded.items()}
                    if index < len(prompts):
                        sample["prompt"] = prompts[index]
                    samples.append(sample)
        if not samples:
            raise RuntimeError("Latent caching produced no samples for this rank.")

        offload = getattr(core, "offload_frozen_modules", None)
        if callable(offload):
            offload()
        torch.cuda.empty_cache()

        self.sampler = torch.utils.data.distributed.DistributedSampler(
            samples,
            num_replicas=1,
            rank=0,
            shuffle=self.args.get("shuffle", True),
        )
        loader_kwargs = {
            "collate_fn": _collate_encoded,
            "pin_memory": True,
        }
        if bool(self.args.get("dataloader_even_batches", False)):
            self.dataloader = torch.utils.data.DataLoader(
                samples,
                batch_sampler=EvenSizeBatchSampler(self.sampler, self.per_device_train_batch_size),
                **loader_kwargs,
            )
        else:
            self.dataloader = torch.utils.data.DataLoader(
                samples,
                batch_size=self.per_device_train_batch_size,
                sampler=self.sampler,
                drop_last=bool(self.args.get("dataloader_drop_last", False)),
                **loader_kwargs,
            )
        self._encoded_cache = True
        if self.rank == 0:
            logger.info(
                f"Cached {len(samples)} encoded samples per rank in {time.time() - cache_start:.1f}s; "
                "frozen encoders offloaded to CPU."
            )

    def _prepare_batch(self, batch, processor):
        if self._encoded_cache:
            processor = None
        return super()._prepare_batch(batch, processor)

    def _prepare_host_batch(self, batch):
        """Decode raw samples on the host before the step timer starts."""
        if self._encoded_cache:
            return batch
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

    def _pure_low_precision(self, mp_dtype: torch.dtype) -> bool:
        return bool(self.args.get("fsdp2_pure_low_precision", False)) and mp_dtype != torch.float32

    def _prepare_fsdp2_modules(self, mp_dtype: torch.dtype) -> None:
        # AMDiffusionBenchmark moves every frozen submodule to the training dtype.
        cast_frozen = getattr(self.model, "cast_frozen_modules", None)
        if callable(cast_frozen) and mp_dtype != torch.float32:
            cast_frozen(mp_dtype)
            if self.rank == 0:
                logger.info(f"Cast frozen encoder modules to {mp_dtype}")

    def _fsdp2_reduce_dtype_name(self, mp_dtype: torch.dtype) -> str:
        # AMDiffusionBenchmark keeps parameters, gradients, reductions, and AdamW in BF16.
        return "bf16" if self._pure_low_precision(mp_dtype) else "fp32"

    def _prepare_fsdp2_root(self, wrap_root: torch.nn.Module, mp_dtype: torch.dtype) -> None:
        if not self._pure_low_precision(mp_dtype):
            return
        wrap_root.to(dtype=mp_dtype)
        if self.rank == 0:
            target = str(self.args.get("fsdp2_wrap_target", "") or "").strip() or "<model>"
            logger.info(f"FSDP2: cast '{target}' parameters to {mp_dtype} before sharding")

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
        if throughput_samples_per_gpu_s is None:
            throughput_samples_per_gpu_s = fps_gpu
        alloc, res, max_mem = get_memory()
        lr = self.optimizer.param_groups[0]["lr"]

        # AMDiffusionBenchmark prints the local rank loss as step_loss and the
        # all-reduced mean as train_loss. Those are different values.
        msg = (
            f"step={self.global_step} "
            f"step_loss={step_loss_value:.6f} train_loss={loss_value:.6f} "
            f"mem={alloc:.2f}/{res:.2f}GB peak_mem={max_mem:.2f}GB "
            f"grad_norm={grad_norm:.4f}"
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
            global_batch_size = self._global_batch_size()
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
                global_throughput = throughput_samples_per_gpu_s * self.world_size
                payload["perf/fps_gpu"] = throughput_samples_per_gpu_s
                payload["throughput(global_samples/s)"] = global_throughput
            if tflops_per_gpu is not None:
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

        if bool(self.args.get("cache_latents", True)) and not self.mlperf_enabled:
            self._cache_encoded_batches()

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
        accum = max(1, self.grad_accum_steps)
        steps_per_epoch = math.ceil(len(self.dataloader) / accum)
        start_epoch = self.global_step // steps_per_epoch
        resume_batch_offset = (self.global_step % steps_per_epoch) * accum

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
                is_update_step = ((batch_idx + 1) % accum) == 0
                starts_update = (batch_idx % accum) == 0
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
                    loss = raw_loss / accum
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
                        # The CUDA window covers every micro-batch in the update, so
                        # FPS divides that window by the samples actually consumed.
                        if step_time_gpu > 0 and local_samples_in_update > 0:
                            step_fps, step_tflops = benchmark_step_rates(
                                local_batch_size=local_samples_in_update,
                                step_time_gpu_s=step_time_gpu,
                                total_flops=total_flops,
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
                            fps_gpu=step_fps,
                            tflops_per_gpu=step_tflops,
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
