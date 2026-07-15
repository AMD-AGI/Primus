###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Runtime Megatron-Bridge patches for MLPerf Llama2-70B LoRA.

Replaces the former git patches under ``third_party/Megatron-Bridge`` without
writing to ``third_party/Megatron-Bridge``.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Optional

import torch

from primus.backends.megatron_bridge.patches.mlperf_llama2_70b.conditions import is_llama2_70b_mlperf
from primus.backends.megatron_bridge.patches.mlperf_llama2_70b.resettable_data_iterator import (
    ResettableDataIterator,
)
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCHED_ATTR = "_primus_mlperf_llama2_70b_patched"


def _mark_patched(obj: Any) -> None:
    setattr(obj, _PATCHED_ATTR, True)


def _already_patched(obj: Any) -> bool:
    return bool(getattr(obj, _PATCHED_ATTR, False))


@register_patch(
    "mlperf_llama2_70b.bridge.data_patches",
    backend="megatron_bridge",
    phase="before_train",
    condition=is_llama2_70b_mlperf,
    description="MLPerf data/eval/sampler overrides for Megatron-Bridge",
)
def patch_bridge_data(ctx: PatchContext) -> None:
    import megatron.bridge.data.loaders as loaders
    import megatron.bridge.data.samplers as samplers
    import megatron.bridge.training.eval as bridge_eval
    from megatron.core.rerun_state_machine import RerunDataIterator

    loaders.ResettableDataIterator = ResettableDataIterator

    import megatron.bridge.peft.lora as bridge_lora
    from primus.backends.megatron_bridge.patches.mlperf_llama2_70b.lora import LoRA, VLMLoRA

    bridge_lora.LoRA = LoRA
    bridge_lora.VLMLoRA = VLMLoRA

    if not _already_patched(loaders.get_train_valid_test_num_samples):
        orig_num_samples = loaders.get_train_valid_test_num_samples

        @functools.wraps(orig_num_samples)
        def _mlperf_get_train_valid_test_num_samples(cfg):
            if cfg.train.train_samples is not None:
                train_samples = cfg.train.train_samples
            else:
                train_samples = cfg.train.train_iters * cfg.train.global_batch_size
            eval_iters = cfg.train.eval_iters
            test_iters = cfg.train.eval_iters
            return (
                train_samples,
                eval_iters * cfg.train.global_batch_size,
                test_iters * cfg.train.global_batch_size,
            )

        loaders.get_train_valid_test_num_samples = _mlperf_get_train_valid_test_num_samples
        _mark_patched(_mlperf_get_train_valid_test_num_samples)

    if not _already_patched(samplers.build_pretraining_data_loader):
        orig_build_loader = samplers.build_pretraining_data_loader

        @functools.wraps(orig_build_loader)
        def _mlperf_build_pretraining_data_loader(
            dataset,
            consumed_samples,
            dataloader_type,
            micro_batch_size,
            num_workers,
            data_sharding,
            worker_init_fn=None,
            collate_fn=None,
            pin_memory=True,
            persistent_workers=False,
            data_parallel_rank=0,
            data_parallel_size=1,
            drop_last=True,
            global_batch_size=None,
            eval_iters: Optional[int] = None,
            name: str = "",
        ):
            if dataset is None:
                return None

            if name == "validation":
                if eval_iters is None or global_batch_size is None:
                    raise RuntimeError(
                        "eval_iters and global_batch_size must be provided when creating "
                        "a validation dataloader for MLPerf Llama2."
                    )
                eval_samples = eval_iters * global_batch_size
                total_samples = min(len(dataset), eval_samples)
                batch_sampler = samplers.MegatronPretrainingSampler(
                    total_samples=total_samples,
                    consumed_samples=0,
                    micro_batch_size=micro_batch_size,
                    data_parallel_rank=data_parallel_rank,
                    data_parallel_size=data_parallel_size,
                )
            elif dataloader_type == "single":
                batch_sampler = samplers.MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=micro_batch_size,
                    data_parallel_rank=data_parallel_rank,
                    data_parallel_size=data_parallel_size,
                    drop_last=drop_last,
                )
            elif dataloader_type == "cyclic":
                batch_sampler = samplers.MegatronPretrainingRandomSampler(
                    dataset,
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=micro_batch_size,
                    data_parallel_rank=data_parallel_rank,
                    data_parallel_size=data_parallel_size,
                    data_sharding=data_sharding,
                )
            elif dataloader_type == "batch":
                if global_batch_size is None:
                    raise RuntimeError(
                        "global_batch_size must be provided when using dataloader_type='batch'."
                    )
                batch_sampler = samplers.MegatronPretrainingBatchSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=micro_batch_size,
                    global_batch_size=global_batch_size,
                    data_parallel_rank=data_parallel_rank,
                    data_parallel_size=data_parallel_size,
                    drop_last=drop_last,
                    pad_samples_to_global_batch_size=not drop_last,
                )
            elif dataloader_type == "external":
                return dataset
            else:
                raise Exception(f"Unsupported dataloader_type: {dataloader_type}")

            return torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                worker_init_fn=worker_init_fn,
                collate_fn=collate_fn,
            )

        samplers.build_pretraining_data_loader = _mlperf_build_pretraining_data_loader
        _mark_patched(_mlperf_build_pretraining_data_loader)

    if not _already_patched(loaders.build_train_valid_test_data_loaders):
        orig_build_loaders = loaders.build_train_valid_test_data_loaders

        @functools.wraps(orig_build_loaders)
        def _mlperf_build_train_valid_test_data_loaders(
            cfg, train_state, build_train_valid_test_datasets_provider
        ):
            (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)
            loaders.print_rank_0("> building train, validation, and test datasets ...")
            train_ds, valid_ds, test_ds = loaders.build_train_valid_test_datasets(
                cfg=cfg,
                build_train_valid_test_datasets_provider=build_train_valid_test_datasets_provider,
            )
            exit_signal = cfg.train.exit_signal

            from megatron.bridge.training.utils.sig_utils import DistributedSignalHandler

            def worker_init_fn(_):
                DistributedSignalHandler(exit_signal).__enter__()

            maybe_worker_init_fn = (
                worker_init_fn if cfg.train.exit_signal_handler_for_dataloader else None
            )

            train_dataloader = samplers.build_pretraining_data_loader(
                train_ds,
                train_state.consumed_train_samples,
                cfg.dataset.dataloader_type,
                cfg.train.micro_batch_size,
                cfg.dataset.num_workers,
                cfg.dataset.data_sharding,
                worker_init_fn=maybe_worker_init_fn,
                collate_fn=train_ds.collate_fn if hasattr(train_ds, "collate_fn") else None,
                pin_memory=cfg.dataset.pin_memory,
                persistent_workers=cfg.dataset.persistent_workers,
                data_parallel_rank=loaders.mpu.get_data_parallel_rank(),
                data_parallel_size=loaders.mpu.get_data_parallel_world_size(),
                global_batch_size=cfg.train.global_batch_size,
                name="train",
            )
            if cfg.train.skip_train and cfg.train.eval_iters > 0:
                valid_dataloader = samplers.build_pretraining_data_loader(
                    valid_ds,
                    0,
                    cfg.dataset.dataloader_type,
                    cfg.train.micro_batch_size,
                    cfg.dataset.num_workers,
                    cfg.dataset.data_sharding,
                    worker_init_fn=maybe_worker_init_fn,
                    collate_fn=valid_ds.collate_fn if hasattr(valid_ds, "collate_fn") else None,
                    pin_memory=cfg.dataset.pin_memory,
                    persistent_workers=cfg.dataset.persistent_workers,
                    data_parallel_rank=loaders.mpu.get_data_parallel_rank(),
                    data_parallel_size=loaders.mpu.get_data_parallel_world_size(),
                    global_batch_size=cfg.train.global_batch_size,
                    name="validation",
                    eval_iters=cfg.train.eval_iters,
                )
            elif cfg.train.eval_iters > 0:
                val_dataloader_type = (
                    "cyclic"
                    if isinstance(cfg.dataset, loaders.GPTDatasetConfig)
                    else cfg.dataset.dataloader_type
                )
                valid_dataloader = samplers.build_pretraining_data_loader(
                    valid_ds,
                    train_state.consumed_valid_samples,
                    val_dataloader_type,
                    cfg.train.micro_batch_size,
                    cfg.dataset.num_workers,
                    cfg.dataset.data_sharding,
                    worker_init_fn=maybe_worker_init_fn,
                    collate_fn=valid_ds.collate_fn if hasattr(valid_ds, "collate_fn") else None,
                    pin_memory=cfg.dataset.pin_memory,
                    persistent_workers=cfg.dataset.persistent_workers,
                    data_parallel_rank=loaders.mpu.get_data_parallel_rank(),
                    data_parallel_size=loaders.mpu.get_data_parallel_world_size(),
                    global_batch_size=cfg.train.global_batch_size,
                    name="validation",
                    eval_iters=cfg.train.eval_iters,
                )

            if cfg.train.eval_iters > 0:
                test_dataloader = samplers.build_pretraining_data_loader(
                    test_ds,
                    0,
                    cfg.dataset.dataloader_type,
                    cfg.train.micro_batch_size,
                    cfg.dataset.num_workers,
                    cfg.dataset.data_sharding,
                    worker_init_fn=maybe_worker_init_fn,
                    collate_fn=test_ds.collate_fn if hasattr(test_ds, "collate_fn") else None,
                    pin_memory=cfg.dataset.pin_memory,
                    persistent_workers=cfg.dataset.persistent_workers,
                    data_parallel_rank=loaders.mpu.get_data_parallel_rank(),
                    data_parallel_size=loaders.mpu.get_data_parallel_world_size(),
                    global_batch_size=cfg.train.global_batch_size,
                    name="test",
                )

            do_train = train_dataloader is not None and cfg.train.train_iters > 0
            do_valid = valid_dataloader is not None and cfg.train.eval_iters > 0
            do_test = test_dataloader is not None and cfg.train.eval_iters > 0
            flags = torch.tensor(
                [int(do_train), int(do_valid), int(do_test)], dtype=torch.long, device="cuda"
            )
            torch.distributed.broadcast(flags, 0)
            train_state.do_train = flags[0].item()
            train_state.do_valid = flags[1].item()
            train_state.do_test = flags[2].item()
            return train_dataloader, valid_dataloader, test_dataloader

        loaders.build_train_valid_test_data_loaders = _mlperf_build_train_valid_test_data_loaders
        _mark_patched(_mlperf_build_train_valid_test_data_loaders)

    if not _already_patched(loaders.build_train_valid_test_data_iterators):
        orig_build_iterators = loaders.build_train_valid_test_data_iterators

        @functools.wraps(orig_build_iterators)
        def _mlperf_build_train_valid_test_data_iterators(
            cfg, train_state, build_train_valid_test_datasets_provider
        ):
            train_dataloader, valid_dataloader, test_dataloader = (
                loaders.build_train_valid_test_data_loaders(
                    cfg=cfg,
                    train_state=train_state,
                    build_train_valid_test_datasets_provider=build_train_valid_test_datasets_provider,
                )
            )
            dl_type = cfg.dataset.dataloader_type
            assert dl_type in ["single", "cyclic", "batch", "external"]

            def _get_iterator(dataloader_type, dataloader):
                if dataloader_type == "single":
                    return RerunDataIterator(iter(dataloader))
                if dataloader_type in ("cyclic", "batch"):
                    return RerunDataIterator(iter(loaders.cyclic_iter(dataloader)))
                if dataloader_type == "external":
                    if isinstance(dataloader, list):
                        return [RerunDataIterator(d) for d in dataloader]
                    return RerunDataIterator(dataloader)
                raise RuntimeError("unexpected dataloader type")

            train_data_iterator = (
                _get_iterator(dl_type, train_dataloader) if train_dataloader is not None else None
            )
            if valid_dataloader is not None:
                valid_data_iterator = RerunDataIterator(ResettableDataIterator(valid_dataloader))
            else:
                valid_data_iterator = None
            test_data_iterator = (
                _get_iterator(dl_type, test_dataloader) if test_dataloader is not None else None
            )
            return train_data_iterator, valid_data_iterator, test_data_iterator

        loaders.build_train_valid_test_data_iterators = _mlperf_build_train_valid_test_data_iterators
        _mark_patched(_mlperf_build_train_valid_test_data_iterators)

    def _reset_data_iterator(data_iterator):
        if data_iterator is None:
            return
        if isinstance(data_iterator, list):
            for it in data_iterator:
                _reset_data_iterator(it)
            return
        if isinstance(data_iterator, RerunDataIterator):
            inner = data_iterator.iterable
            if isinstance(inner, ResettableDataIterator):
                inner.reset()
                data_iterator.saved_microbatches.clear()
                data_iterator.replaying = False
                data_iterator.replay_pos = 0
        elif isinstance(data_iterator, ResettableDataIterator):
            data_iterator.reset()

    if not _already_patched(bridge_eval.evaluate):
        orig_evaluate = bridge_eval.evaluate

        @functools.wraps(orig_evaluate)
        def _mlperf_evaluate(*args, **kwargs):
            if len(args) >= 3:
                _reset_data_iterator(args[2])
            else:
                _reset_data_iterator(kwargs.get("data_iterator"))
            return orig_evaluate(*args, **kwargs)

        bridge_eval.evaluate = _mlperf_evaluate
        _mark_patched(_mlperf_evaluate)

    log_rank_0("[Patch:mlperf_llama2_70b.bridge.data_patches] Megatron-Bridge data patches applied")


@register_patch(
    "mlperf_llama2_70b.bridge.sft_attention_mask",
    backend="megatron_bridge",
    phase="before_train",
    condition=is_llama2_70b_mlperf,
    description="Cache causal attention masks in GPTSFTDataset for MLPerf steady-state SFT",
)
def patch_sft_attention_mask(ctx: PatchContext) -> None:
    from megatron.bridge.data.datasets import sft as sft_mod

    if _already_patched(sft_mod.GPTSFTDataset._create_attention_mask):
        return

    orig_create_mask = sft_mod.GPTSFTDataset._create_attention_mask

    @functools.wraps(orig_create_mask)
    def _cached_create_attention_mask(self, max_length):
        cache = getattr(self, "_attention_mask_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_attention_mask_cache", cache)
        cached = cache.get(max_length)
        if cached is not None:
            return cached
        attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
        attention_mask = attention_mask < 0.5
        cache[max_length] = attention_mask
        return attention_mask

    sft_mod.GPTSFTDataset._create_attention_mask = _cached_create_attention_mask
    _mark_patched(_cached_create_attention_mask)
    log_rank_0("[Patch:mlperf_llama2_70b.bridge.sft_attention_mask] SFT attention-mask cache applied")


@register_patch(
    "mlperf_llama2_70b.bridge.training_log_nemo",
    backend="megatron_bridge",
    phase="before_train",
    condition=is_llama2_70b_mlperf,
    description="NeMo-style train_step timing support in Megatron-Bridge training_log",
)
def patch_training_log_nemo(ctx: PatchContext) -> None:
    import megatron.bridge.training.utils.train_utils as train_utils
    from megatron.bridge.training.utils import flop_utils
    from megatron.bridge.utils.common_utils import get_world_size_safe, print_rank_0

    original_training_log = train_utils.training_log
    if original_training_log is None or _already_patched(original_training_log):
        return

    sig = inspect.signature(original_training_log)
    if "nemo_elapsed_time_per_iter_sec" in sig.parameters:
        log_rank_0(
            "[Patch:mlperf_llama2_70b.bridge.training_log_nemo] "
            "training_log already supports NeMo timing; skipping wrapper"
        )
        return

    @functools.wraps(original_training_log)
    def _training_log_with_nemo(
        loss_dict,
        total_loss_dict,
        learning_rate,
        decoupled_learning_rate,
        loss_scale,
        report_memory_flag,
        skipped_iter,
        grad_norm,
        params_norm,
        num_zeros_in_grad,
        config,
        global_state,
        history_wct,
        model,
        log_max_attention_logit=None,
        nemo_elapsed_time_per_iter_sec=None,
    ):
        result = original_training_log(
            loss_dict,
            total_loss_dict,
            learning_rate,
            decoupled_learning_rate,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
            config,
            global_state,
            history_wct,
            model,
            log_max_attention_logit,
        )

        if (
            nemo_elapsed_time_per_iter_sec is not None
            and global_state.train_state.step % config.logger.log_interval == 0
        ):
            batch_size = config.train.global_batch_size
            num_flops = flop_utils.num_floating_point_operations(config, batch_size)
            per_gpu_tf = (
                num_flops / nemo_elapsed_time_per_iter_sec / get_world_size_safe() / 1e12
            )
            elapsed = global_state.timers("interval-time").elapsed(barrier=True)
            interval_ms = (elapsed / max(config.logger.log_interval, 1)) * 1000.0
            print_rank_0(
                f"[TFLOP/s basis: NeMo-style train_step wall clock] "
                f"Step time: {nemo_elapsed_time_per_iter_sec:.2f}s | "
                f"model TFLOP/s/GPU: {per_gpu_tf:.1f}"
            )
            print_rank_0(
                f" NeMo-style time/iter (ms): {nemo_elapsed_time_per_iter_sec * 1000.0:.1f} |"
                f" Megatron-Bridge interval-time/iter (ms): {interval_ms:.1f} |"
            )

        return result

    train_utils.training_log = _training_log_with_nemo
    _mark_patched(_training_log_with_nemo)
    log_rank_0("[Patch:mlperf_llama2_70b.bridge.training_log_nemo] NeMo timing wrapper applied")
