###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Wan PyTorch FSDP2 trainer
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from importlib import import_module

import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

from primus.backends.diffusion.distributed import (
    create_device_mesh,
    load_checkpoint_dtcp,
    save_checkpoint_dtcp,
    setup_distributed,
)
from primus.backends.diffusion.utils.log import logger

from .base import BaseWanTrainer


class FSDP2Trainer(BaseWanTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        args: dict,
        train_dataset,
        data_collator,
        processing_class,
        rank: int,
        world_size: int,
        local_rank: int,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            processing_class=processing_class,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )

        # FSDP2-specific: checkpoint strategy
        # - "dit_only": save `dit_model.safetensors` (default for T2V training)
        # - "dtcp_full": save model + optimizer via DTCP
        # - "dtcp_model_only": save only model via DTCP
        # - "dtcp_trainable": DTCP save only trainable params + optimizer
        self.save_strategy = str(self.args.get("save_strategy", "dit_only")).lower()

        # Checkpoint loading (after optimizer & scheduler are fully initialized)
        resume_from = self.args.get("resume_from_checkpoint")
        if resume_from:
            self._load_checkpoint(resume_from)

    # ------------------------------------------------------------------ #
    #                       Parallelism                                    #
    # ------------------------------------------------------------------ #

    def _apply_parallelism(self):
        """Set up FSDP2 composable sharding, optionally with Ulysses SP."""
        sp_size = int(self.args.get("sp_size", 1))
        dp_replicate = int(self.args.get("dp_replicate", 1))

        self.mesh = create_device_mesh(self.world_size, sp_size=sp_size, dp_replicate=dp_replicate)
        self.sp_group = self.mesh.get_group("ulysses") if (self.mesh is not None and sp_size > 1) else None
        self.model.to(self.device)

        # Freeze non-trainable params BEFORE FSDP and optimizer creation
        if hasattr(self.model, "freeze_except"):
            self.model.freeze_except()
            if self.rank == 0:
                logger.info("FSDP2: Applied freeze_except (frozen non-trainable params)")

        self._apply_fsdp2()

    def _apply_fsdp2(self):
        """Apply torch.distributed._composable.fsdp.fully_shard to the model."""
        mp_dtype = self._resolve_dtype()

        # ---- Mixed-precision policy (aligned with DeepSpeed bf16) ----
        #   1. Pre-cast params to bf16 BEFORE FSDP wrapping  (bf16 storage)
        #   2. reduce_dtype = bf16  (gradient reduce matches DeepSpeed bf16 all-reduce)
        #   3. AdamWFP32State maintains fp32 master weights and writes back to bf16
        if mp_dtype != torch.float32:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=mp_dtype,
                reduce_dtype=mp_dtype,  # bf16 reduce (matches DeepSpeed bf16)
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
        wrap_root = self._get_module_by_path(self.model, wrap_target) if wrap_target else self.model

        # Pre-cast to bf16 before FSDP wrapping so parameters are stored in bf16,
        # matching DiffSynth/DeepSpeed bf16 behavior.
        if mp_dtype != torch.float32:
            wrap_root.to(dtype=mp_dtype)
            if self.rank == 0:
                logger.info(f"FSDP2: pre-cast '{wrap_target or '<model>'}' to {mp_dtype} (DiffSynth-aligned)")

        if self.world_size == 1:
            if self.rank == 0:
                logger.info("FSDP2: world_size=1; skipping composable FSDP wrapping.")
            return

        reshard_after_forward = bool(self.args.get("fsdp2_reshard_after_forward", True))

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
                if (cls_objs and isinstance(module, tuple(cls_objs))) or (
                    cls_names and module.__class__.__name__ in cls_names
                ):
                    fully_shard(
                        module,
                        mesh=fsdp_mesh,
                        reshard_after_forward=reshard_after_forward,
                        mp_policy=mp_policy,
                    )
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
            logger.info(f"FSDP2: applied fully_shard to '{wrap_target or '<model>'}' with mp={mp_dtype}")

    @staticmethod
    def _get_module_by_path(root: torch.nn.Module, path: str) -> torch.nn.Module:
        """Resolve a dot-separated attribute path on a module."""
        cur = root
        if not path:
            return cur
        for part in path.split("."):
            if not hasattr(cur, part):
                raise ValueError(f"fsdp2_wrap_target='{path}' is invalid: missing attribute '{part}'")
            cur = getattr(cur, part)
        if not isinstance(cur, torch.nn.Module):
            raise ValueError(f"fsdp2_wrap_target='{path}' did not resolve to a torch.nn.Module")
        return cur

    # ------------------------------------------------------------------ #
    #                       Gradient sync                                  #
    # ------------------------------------------------------------------ #

    def _set_requires_gradient_sync(self, enabled: bool) -> None:
        """
        Best-effort gradient sync control for composable FSDP2.
        `fully_shard()` turns modules into FSDPM which exposes
        `set_requires_gradient_sync`. For grad accumulation, we disable
        sync on non-update micro-steps.
        """
        seen = set()
        for _, m in self.model.named_modules():
            if id(m) in seen:
                continue
            seen.add(id(m))
            setter = getattr(m, "set_requires_gradient_sync", None)
            if callable(setter):
                setter(bool(enabled))

    @contextmanager
    def _grad_sync_context(self, is_update_step: bool):
        if self.world_size > 1 and self.grad_accum_steps > 1:
            self._set_requires_gradient_sync(is_update_step)
        yield
        if is_update_step and self.world_size > 1 and self.grad_accum_steps > 1:
            self._set_requires_gradient_sync(True)

    # ------------------------------------------------------------------ #
    #                       Checkpointing                                  #
    # ------------------------------------------------------------------ #

    def _dtcp_save_args(self):
        """Compute DTCP save parameters based on save_strategy."""
        if self.save_strategy == "dtcp_model_only":
            return None, None
        if self.save_strategy == "dtcp_trainable":
            opts = StateDictOptions(full_state_dict=False, ignore_frozen_params=True)
            return self.optimizer, opts
        # dtcp_full (default non-dit strategy)
        return self.optimizer, None

    def _save_dtcp(self, path):
        """Save checkpoint via Distributed Tensor Checkpointing."""
        optimizer, opts = self._dtcp_save_args()
        kwargs = {}
        if opts is not None:
            kwargs["model_state_options"] = opts
            kwargs["optim_state_options"] = opts
        save_checkpoint_dtcp(
            self.model,
            optimizer,
            path,
            epoch=0,
            step=self.global_step,
            additional_data={
                "lr_scheduler": (self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None)
            },
            **kwargs,
        )

    def _save_checkpoint(self):
        path = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
        if self.save_strategy == "dit_only":
            self._save_dit(os.path.join(path, "dit_model.safetensors"))
        else:
            self._save_dtcp(path)

    def _load_checkpoint(self, path):
        if self.rank == 0:
            logger.info(f"Loading checkpoint from {path}")

        if self.save_strategy == "dit_only":
            core_model = self.model
            if not hasattr(core_model, "dit"):
                raise ValueError("save_strategy=dit_only requires model.dit to exist for checkpoint loading")
            candidate = path
            if os.path.isdir(path):
                candidate = os.path.join(path, "dit_model.safetensors")
            if not os.path.exists(candidate):
                raise FileNotFoundError(f"dit_only checkpoint not found at: {candidate}")
            state = safe_load_file(candidate)
            missing, unexpected = core_model.dit.load_state_dict(state, strict=False)
            if self.rank == 0:
                logger.info(
                    f"Loaded DiT weights from {candidate}. "
                    f"Missing keys: {len(missing)}, "
                    f"Unexpected keys: {len(unexpected)}"
                )
            return

        if self.save_strategy == "dtcp_trainable":
            opts = StateDictOptions(full_state_dict=False, ignore_frozen_params=True)
            meta = load_checkpoint_dtcp(
                self.model,
                self.optimizer,
                path,
                model_state_options=opts,
                optim_state_options=opts,
            )
        elif self.save_strategy == "dtcp_model_only":
            meta = load_checkpoint_dtcp(self.model, None, path)
        else:
            meta = load_checkpoint_dtcp(self.model, self.optimizer, path)

        if "step" in meta:
            self.global_step = meta["step"]
            if self.rank == 0:
                logger.info(f"Resumed from step {self.global_step}")
        if self.lr_scheduler is not None and isinstance(meta, dict) and meta.get("lr_scheduler") is not None:
            try:
                self.lr_scheduler.load_state_dict(meta["lr_scheduler"])
                if self.rank == 0:
                    logger.info("Resumed lr_scheduler state from checkpoint meta")
            except Exception as exc:
                if self.rank == 0:
                    logger.warning(f"Failed to restore lr_scheduler state: {exc}")

    def _save_dit(self, save_path: str) -> None:
        """
        Save `dit` weights as a single safetensors file.

        Notes:
        - On composable FSDP (world_size>1), ``get_model_state_dict`` with
          ``full_state_dict=True`` returns the full dict **only on rank 0**;
          other ranks receive an empty dict.  This is expected – do NOT
          early-return on the empty-dict check, or non-rank-0 processes will
          race ahead and desynchronize from rank 0 (which still needs to
          write to disk).
        - A ``dist.barrier()`` at the end keeps all ranks in lockstep so
          that back-to-back saves (e.g. periodic checkpoint + final save)
          never overlap.
        """
        import torch.distributed as dist

        core_model = self.model
        if not hasattr(core_model, "dit"):
            logger.warning("save_model: model has no `dit` attribute; skipping save.")
            return

        if self.world_size > 1:
            # FSDP-wrapped: use get_model_state_dict to unshard DTensors
            full_state = get_model_state_dict(
                core_model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            dit_state_dict = {k[len("dit.") :]: v for k, v in full_state.items() if k.startswith("dit.")}
        else:
            # Single GPU (no FSDP wrapping): direct state_dict
            dit_state_dict = core_model.dit.state_dict()
            if not dit_state_dict:
                logger.warning("save_model: DiT state dict is empty; skipping save.")
                return

        # Only rank 0 has the full dict in multi-GPU; write from rank 0 only.
        if self.rank == 0:
            if not dit_state_dict:
                logger.warning("save_model: DiT state dict is empty on rank 0; skipping save.")
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                safe_save_file(dit_state_dict, save_path)
                logger.info(f"Saved DiT weights to {save_path}")

            if hasattr(core_model, "config"):
                try:
                    core_model.config.save_pretrained(os.path.dirname(save_path))
                except Exception as exc:
                    logger.warning(f"save_model: failed to save config: {exc}")

        # Barrier: ensure all ranks wait for rank 0 to finish writing before
        # any rank proceeds to the next operation (e.g. another save or exit).
        if self.world_size > 1:
            dist.barrier()

    def save_model(self):
        """Save final model using the configured strategy."""
        if self.save_strategy in ("none", "skip", "disabled"):
            if self.rank == 0:
                logger.info(f"Skipping final checkpoint save (strategy={self.save_strategy})")
            return
        if self.save_strategy == "dit_only":
            save_path = os.path.join(self.output_dir, "dit_model.safetensors")
            self._save_dit(save_path)
            return

        path = os.path.join(self.output_dir, "checkpoint-final")
        if self.rank == 0:
            logger.info(f"Saving final checkpoint to {path} (strategy={self.save_strategy})")
        self._save_dtcp(path)


def build_fsdp2_trainer(*, model, dataset, processor, trainer_args: dict):
    rank, world_size, local_rank = setup_distributed()
    return FSDP2Trainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset,
        data_collator=dataset.get_collator(),
        processing_class=processor,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
