###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DDP and FSDP2 training loop for vision models.

The model owns its forward and loss. This loop owns the device, the data
parallel wrapper, the optimizer, the schedule, the checkpoint, and the log.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from primus.backends.vision.trainers.schedule import lr_factor
from primus.core.utils.module_utils import log_rank_0


def _log(message: str) -> None:
    try:
        log_rank_0(message)
    except Exception:
        if int(os.environ.get("RANK", "0")) == 0:
            print(message)


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _plain_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Full tensors on CPU. Every rank calls this; DTensor gathers are collective."""
    state = module.export_state_dict()
    plain = {}
    for key, value in state.items():
        tensor = value.full_tensor() if hasattr(value, "full_tensor") else value
        plain[key] = tensor.detach().to("cpu").contiguous().clone()
        return plain


class VisionFSDP2Trainer:
    """Train one vision model with DDP or FSDP2."""

    def __init__(
        self,
        *,
        model,
        dataset,
        processor,
        trainer_args: dict,
        eval_dataset=None,
        eval_processor=None,
    ):
        self.model = model
        self.dataset = dataset
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.eval_processor = eval_processor
        self.args = trainer_args
        self.steps = 0
        self.skipped_steps = 0
        self.optimizer = None
        self.scheduler = None

    def train(self) -> None:
        args = self.args
        device = self._device()
        seed = int(args.get("seed", 42))
        torch.manual_seed(seed + _rank())
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + _rank())
            torch.backends.cuda.matmul.allow_tf32 = True

        model = self.model.to(device)
        if bool(args.get("gradient_checkpointing", False)):
            self._checkpoint_blocks(model)
        if bool(args.get("compile_transformer_blocks", False)):
            self._compile_blocks(model)
        model.param_groups()
        model = self._parallelize(model, device)
        self.model = model
        self.optimizer = self._optimizer(_unwrap(model))
        self.scheduler = self._scheduler(self.optimizer)
        resume = str(args.get("resume_from_checkpoint") or "").strip()
        if resume:
            self._load_checkpoint(resume)

        output_dir = Path(args["output_dir"])
        if _rank() == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()

        max_steps = int(args.get("max_steps", 100))
        max_epochs = int(args.get("num_train_epochs", 1000))
        accum_steps = max(int(args.get("gradient_accumulation_steps", 1)), 1)
        log_every = max(int(args.get("logging_steps", 1)), 1)
        save_every = int(args.get("save_steps", 0) or 0)
        use_autocast = bool(args.get("bf16", False)) and device.type == "cuda"
        profiler = self._start_profiler(output_dir)

        _log(
            "[Primus:Vision] "
            f"mode={args.get('parallelism_mode', 'ddp')} steps={max_steps} "
            f"accum={accum_steps} bf16={use_autocast} device={device}"
        )
        model.train()
        self.optimizer.zero_grad(set_to_none=True)
        epoch = 0
        pending = 0
        while self.steps < max_steps and epoch < max_epochs:
            loader = self._loader(epoch)
            saw_batch = False
            for batch in loader:
                saw_batch = True
                sync = pending + 1 >= accum_steps
                loss_value = self._forward_backward(model, batch, device, use_autocast, sync, accum_steps)
                if loss_value is None:
                    self.optimizer.zero_grad(set_to_none=True)
                    pending = 0
                    self.skipped_steps += 1
                    self.steps += 1
                    if self.steps >= max_steps:
                        break
                    continue
                pending += 1
                if pending < accum_steps:
                    continue
                pending = 0
                self._clip()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.steps += 1
                if profiler is not None:
                    profiler.step()
                if self.steps % log_every == 0:
                    self._log_step(loss_value, batch, max_steps)
                if save_every > 0 and self.steps % save_every == 0:
                    self._save_checkpoint(output_dir / f"step_{self.steps}.pt")
                if self.steps >= max_steps:
                    break
            if not saw_batch:
                break
            epoch += 1

        if profiler is not None:
            profiler.stop()
        if self.steps == 0:
            raise RuntimeError("Vision trainer ran zero steps. The dataloader produced no batches.")
        self._save_checkpoint(output_dir / "last.pt")
        self._export(output_dir / "model.safetensors")
        _log(f"[Primus:Vision] finished steps={self.steps} skipped={self.skipped_steps}")

    def _device(self) -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device("cuda", local_rank)

    def _parallelize(self, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if world <= 1:
            return model
        mode = str(self.args.get("parallelism_mode", "ddp")).lower()
        if mode == "fsdp2":
            return self._fsdp2(model)
        local_rank = device.index if device.type == "cuda" else None
        find_unused = bool(getattr(model, "find_unused_parameters", False))
        return DDP(
            model,
            device_ids=None if local_rank is None else [local_rank],
            find_unused_parameters=find_unused,
        )

    def _fsdp2(self, model: torch.nn.Module) -> torch.nn.Module:
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

        reduce = torch.float32
        param_dtype = torch.bfloat16 if bool(self.args.get("bf16", False)) else torch.float32
        policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce)
        reshard = bool(self.args.get("fsdp2_reshard_after_forward", True))
        block_types = tuple(model.block_classes())
        if block_types:
            for module in model.modules():
                if isinstance(module, block_types):
                    fully_shard(module, mp_policy=policy, reshard_after_forward=reshard)
        fully_shard(model, mp_policy=policy, reshard_after_forward=reshard)
        return model

    def _optimizer(self, module: torch.nn.Module) -> torch.optim.Optimizer:
        learning_rate = float(self.args.get("learning_rate", 1.0e-4))
        weight_decay = float(self.args.get("weight_decay", 0.05))
        groups = []
        for group in module.param_groups():
            params = [param for param in group.get("params", []) if param.requires_grad]
            if not params:
                continue
            built = dict(group)
            built["params"] = params
            built.setdefault("lr", learning_rate)
            built.setdefault("weight_decay", weight_decay)
            groups.append(built)
        if not groups:
            raise RuntimeError("Vision model produced no optimizer parameter groups.")
        return torch.optim.AdamW(
            groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(float(self.args.get("adam_beta1", 0.9)), float(self.args.get("adam_beta2", 0.95))),
            eps=float(self.args.get("adam_epsilon", 1.0e-8)),
        )

    def _scheduler(self, optimizer: torch.optim.Optimizer):
        args = self.args

        def factor(step: int) -> float:
            return lr_factor(
                step,
                scheduler_type=str(args.get("lr_scheduler_type", "cosine")),
                warmup_steps=int(args.get("warmup_steps", 0)),
                total_steps=int(args.get("max_steps", 100)),
                eta_min_ratio=float(args.get("eta_min_ratio", 0.1)),
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    def _loader(self, epoch: int):
        source = self.dataset
        if hasattr(source, "get_loader"):
            return source.get_loader(epoch)
        dataset = getattr(source, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        for sampler in (getattr(source, "batch_sampler", None), getattr(source, "sampler", None)):
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        return source

    def _forward_backward(self, model, batch, device, use_autocast: bool, sync: bool, accum_steps: int):
        sync_context = self._sync_context(model, sync)
        try:
            with sync_context:
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
                    loss = model(batch)
                if not torch.isfinite(loss.detach()):
                    _log("[Primus:Vision] skipping a non-finite loss")
                    return None
                (loss / accum_steps).backward()
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        return float(loss.detach().float().item())

    @staticmethod
    def _sync_context(model, sync: bool):
        if sync:
            return contextlib.nullcontext()
        if isinstance(model, DDP):
            return model.no_sync()
        setter = getattr(model, "set_requires_gradient_sync", None)
        if setter is None:
            return contextlib.nullcontext()

        class _FSDPSync:
            def __enter__(self_nonlocal):
                setter(False)

            def __exit__(self_nonlocal, exc_type, exc, tb):
                setter(True)
                return False

        return _FSDPSync()

    def _clip(self) -> None:
        max_norm = float(self.args.get("max_grad_norm", 1.0))
        if max_norm <= 0:
            return
        clip = getattr(self.model, "clip_grad_norm_", None)
        if clip is not None and not isinstance(self.model, DDP):
            clip(max_norm)
            return
        torch.nn.utils.clip_grad_norm_(_unwrap(self.model).parameters(), max_norm)

    def _log_step(self, loss_value: float, batch: Any, max_steps: int) -> None:
        module = _unwrap(self.model)
        metrics = {}
        for key, value in getattr(module, "last_metrics", {}).items():
            if torch.is_tensor(value):
                if value.numel() == 1:
                    metrics[key] = float(value.detach().float().cpu())
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)
        learning_rate = self.optimizer.param_groups[0]["lr"]
        parts = [
            f"step={self.steps}/{max_steps}",
            f"loss={loss_value:.6f}",
            f"lr={learning_rate:.3e}",
            f"skipped={self.skipped_steps}",
        ]
        if "num_images" in metrics:
            parts.append(f"images={int(metrics['num_images'])}")
        for key in sorted(metrics):
            if key in {"loss", "num_images", "objective"}:
                continue
            parts.append(f"{key}={metrics[key]:.4f}")
        _log("[Primus:Vision] " + " ".join(parts))

    def _checkpoint_blocks(self, model: torch.nn.Module) -> None:
        from torch.utils.checkpoint import checkpoint

        block_types = tuple(model.block_classes())
        if not block_types:
            return
        for module in model.modules():
            if not isinstance(module, block_types) or getattr(module, "_vision_ckpt", False):
                continue
            original = module.forward

            def wrapped(*args, _original=original, **kwargs):
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)

            module.forward = wrapped
            module._vision_ckpt = True

    def _compile_blocks(self, model: torch.nn.Module) -> None:
        block_types = tuple(model.block_classes())
        for module in model.modules():
            if isinstance(module, block_types):
                module.forward = torch.compile(module.forward)

    def _start_profiler(self, output_dir: Path):
        if not bool(self.args.get("profile", False)):
            return None
        from torch.profiler import ProfilerActivity, profile, schedule

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        trace_dir = output_dir / "profiler"
        if _rank() == 0:
            trace_dir.mkdir(parents=True, exist_ok=True)
        profiler = profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
        )
        profiler.start()
        return profiler

    def _save_checkpoint(self, path: Path) -> None:
        module = _unwrap(self.model)
        model_state = _plain_state(module)
        if _rank() == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": self.steps,
                    "model": model_state,
                    "optimizer": self.optimizer.state_dict(),
                    "skipped_steps": self.skipped_steps,
                },
                path,
            )
            limit = int(self.args.get("save_total_limit", 3) or 0)
            if limit > 0 and path.name.startswith("step_"):
                files = sorted(path.parent.glob("step_*.pt"), key=lambda item: item.stat().st_mtime)
                while len(files) > limit:
                    files.pop(0).unlink(missing_ok=True)
        self._barrier()

    def _load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        module = _unwrap(self.model)
        incoming = checkpoint.get("model", {})
        current = module.export_state_dict()
        matched = {
            key: value
            for key, value in incoming.items()
            if key in current and tuple(current[key].shape) == tuple(value.shape)
        }
        inner = getattr(module, "vggt", None) or getattr(module, "wm", None) or module
        inner.load_state_dict(matched, strict=False)
        if "optimizer" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = int(checkpoint.get("step", 0))
        self.skipped_steps = int(checkpoint.get("skipped_steps", 0))
        _log(f"[Primus:Vision] resumed {path} at step {self.steps} ({len(matched)} tensors)")

    def _export(self, path: Path) -> None:
        state = _plain_state(_unwrap(self.model))
        if _rank() != 0:
            self._barrier()
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from safetensors.torch import save_file

            save_file(state, str(path))
        except Exception:
            torch.save(state, path.with_suffix(".pt"))
            _log(f"[Primus:Vision] safetensors export failed, wrote {path.with_suffix('.pt')}")
            self._barrier()
            return
        _log(f"[Primus:Vision] wrote {path}")
        self._barrier()

    @staticmethod
    def _barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
