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
import random
import re
import shutil
from contextlib import contextmanager
from importlib import import_module

import torch
import torch.distributed as dist
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
        eval_dataset=None,
        eval_processor=None,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            processing_class=processing_class,
            eval_dataset=eval_dataset,
            eval_processor=eval_processor,
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
        self.save_total_limit = int(self.args.get("save_total_limit", 3))

        # Checkpoint loading (after optimizer & scheduler are fully initialized)
        resume_from = self.args.get("resume_from_checkpoint")
        if resume_from:
            path = (
                self._latest_checkpoint() if str(resume_from).lower() in {"auto", "latest"} else resume_from
            )
            if path:
                self._load_checkpoint(path)
            elif str(resume_from).lower() not in {"auto", "latest"}:
                raise FileNotFoundError(f"Checkpoint not found: {resume_from}")

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
        if hasattr(self.model, "compute_dtype"):
            self.model.compute_dtype = self._resolve_dtype()

        # Freeze non-trainable params BEFORE FSDP and optimizer creation
        if hasattr(self.model, "freeze_except"):
            self.model.freeze_except()
            if self.rank == 0:
                logger.info("FSDP2: Applied freeze_except (frozen non-trainable params)")

        self._apply_fsdp2()

    def _apply_fsdp2(self):
        """Apply torch.distributed._composable.fsdp.fully_shard to the model."""
        mp_dtype = self._resolve_dtype()
        reduce_dtype_name = os.getenv("FSDP2_REDUCE_DTYPE", "fp32").lower()
        if reduce_dtype_name not in {"fp32", "bf16"}:
            raise ValueError(f"Unsupported FSDP2_REDUCE_DTYPE={reduce_dtype_name!r}")
        reduce_dtype = torch.float32 if reduce_dtype_name == "fp32" else torch.bfloat16

        fp8_all_reduce = os.getenv("FSDP2_HSDP_FP8_ALL_REDUCE", "").lower()
        if fp8_all_reduce:
            fp8_dtypes = {
                "e4m3": torch.float8_e4m3fn,
                "e5m2": torch.float8_e5m2,
            }
            if fp8_all_reduce not in fp8_dtypes:
                raise ValueError(
                    f"Unsupported FSDP2_HSDP_FP8_ALL_REDUCE={fp8_all_reduce!r}; "
                    "expected e4m3 or e5m2"
                )
            block_size = int(os.getenv("FSDP2_HSDP_FP8_BLOCK_SIZE", "0"))
            if block_size < 0:
                raise ValueError("FSDP2_HSDP_FP8_BLOCK_SIZE must be non-negative")
            from types import SimpleNamespace

            from torch.distributed.fsdp._fully_shard import _fsdp_collectives

            if not hasattr(_fsdp_collectives, "_primus_original_dist"):
                original_dist = _fsdp_collectives.dist
                fp8_dtype = fp8_dtypes[fp8_all_reduce]
                fp8_limit = torch.finfo(fp8_dtype).max

                def fp8_gradient_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
                    if tensor.dtype != torch.bfloat16 or tensor.numel() < 1 << 20:
                        return original_dist.all_reduce(tensor, op=op, group=group, async_op=async_op)
                    if async_op or op not in (dist.ReduceOp.SUM, dist.ReduceOp.AVG):
                        raise ValueError("FP8 HSDP gradient AllReduce only supports synchronous SUM or AVG")
                    world_size = group.size()
                    flat = tensor.flatten()
                    if block_size:
                        padding = (-flat.numel()) % block_size
                        scaled_input = torch.nn.functional.pad(flat, (0, padding)) if padding else flat
                        blocks = scaled_input.view(-1, block_size)
                        absmax = blocks.abs().amax(dim=1).float()
                    else:
                        blocks = flat
                        absmax = flat.abs().amax().float()
                    original_dist.all_reduce(absmax, op=dist.ReduceOp.MAX, group=group)
                    scale = (fp8_limit / world_size) / absmax.clamp_min(torch.finfo(torch.float32).eps)
                    if block_size:
                        # E8M0-style power-of-two scales make scaling exact.
                        scale = torch.exp2(torch.floor(torch.log2(scale)).clamp(-126, 127))[:, None]
                    quantized = (blocks * scale).clamp(
                        -fp8_limit / world_size, fp8_limit / world_size
                    ).to(fp8_dtype)
                    original_dist.all_reduce(quantized, op=dist.ReduceOp.SUM, group=group)
                    reduced = (quantized.to(tensor.dtype) / scale).flatten()[: tensor.numel()]
                    tensor.copy_(reduced.view_as(tensor))
                    if op == dist.ReduceOp.AVG:
                        tensor.div_(world_size)

                fp8_dist = SimpleNamespace(**vars(original_dist))
                fp8_dist.all_reduce = fp8_gradient_all_reduce
                _fsdp_collectives._primus_original_dist = original_dist
                _fsdp_collectives.dist = fp8_dist
            if self.rank == 0:
                scaling = f"block-{block_size}" if block_size else "tensor-wise"
                logger.info(f"FSDP2: HSDP gradient AllReduce uses FP8 {fp8_all_reduce}, {scaling}")

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

        compile_transformer_blocks = bool(self.args.get("compile_transformer_blocks", False))
        compile_strategy = str(self.args.get("compile_strategy", "per_block")).strip().lower()
        if compile_transformer_blocks:
            self._compile_transformer_blocks(wrap_root)
        if bool(self.args.get("compile_output_head", False)) and not (
            compile_transformer_blocks and compile_strategy == "full_dit"
        ):
            self._compile_output_head(wrap_root)

        if self.world_size == 1:
            if self.rank == 0:
                logger.info("FSDP2: world_size=1; skipping composable FSDP wrapping.")
            return

        reshard_after_forward = bool(self.args.get("fsdp2_reshard_after_forward", True))
        no_reshard_paths = {
            item.strip()
            for item in str(self.args.get("fsdp_module_paths_no_reshard", "") or "").split(",")
            if item.strip()
        }

        module_paths_spec = str(self.args.get("fsdp_module_paths_to_wrap", "") or "")
        module_paths = [item.strip() for item in module_paths_spec.split(",") if item.strip()]
        if compile_transformer_blocks and compile_strategy == "full_dit":
            module_paths = [path for path in module_paths if path != "final_layer"]
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
            # FSDP hooks are Dynamo-disabled, so compiled stacks must remain
            # root-managed to keep those hooks outside the full graph.
            root_managed_stacks = {
                "single_stack": {"single_blocks"},
                "double_stack": {"double_blocks"},
                "stack": {"double_blocks", "single_blocks"},
                "full_dit": {"double_blocks", "single_blocks"},
            }.get(compile_strategy, set())
            for module_name, module in wrap_root.named_modules():
                if id(module) in seen:
                    continue
                seen.add(id(module))
                if module is wrap_root:
                    continue
                if compile_transformer_blocks and module_name.split(".", 1)[0] in root_managed_stacks:
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

    def _compile_transformer_blocks(self, root: torch.nn.Module) -> None:
        strategy = str(self.args.get("compile_strategy", "per_block")).strip().lower()
        stack_regions = {
            "single_stack": ("_run_single_blocks",),
            "double_stack": ("_run_double_blocks",),
            "stack": ("_run_double_blocks", "_run_single_blocks"),
            "full_dit": ("_run_dit",),
        }
        if strategy != "per_block" and strategy not in stack_regions:
            valid = ["per_block", *stack_regions]
            raise ValueError(f"Unsupported compile_strategy={strategy!r}; expected one of {valid}")

        backend = str(self.args.get("compile_backend", "inductor")).strip() or "inductor"
        fullgraph = bool(self.args.get("compile_fullgraph", True))
        dynamic = bool(self.args.get("compile_dynamic", False))
        compile_mode = os.getenv("TORCH_COMPILE_MODE", "").strip() or None
        compile_kwargs = {
            "backend": backend,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "mode": compile_mode,
        }

        compiled_regions = []
        if strategy == "per_block":
            for attr in ("double_blocks", "single_blocks"):
                blocks = getattr(root, attr, None)
                if blocks is None:
                    continue
                for index, block in enumerate(blocks):
                    block.compile(**compile_kwargs)
                    compiled_regions.append(f"{attr}[{index}]")
        else:
            for name in stack_regions[strategy]:
                region = getattr(root, name, None)
                if region is None:
                    raise ValueError(f"compile_strategy={strategy!r} requires model method {name}")
                setattr(root, name, torch.compile(region, **compile_kwargs))
                compiled_regions.append(name)

        if self.rank == 0:
            logger.info(
                f"FSDP2: compiled FLUX regions {compiled_regions} with strategy={strategy}, "
                f"backend={backend}, fullgraph={fullgraph}, dynamic={dynamic}, "
                f"mode={compile_mode or 'default'}"
            )

    def _compile_output_head(self, root: torch.nn.Module) -> None:
        final_layer = getattr(root, "final_layer", None)
        if final_layer is None:
            raise ValueError("compile_output_head requires a FLUX model with final_layer")
        backend = str(self.args.get("compile_backend", "inductor")).strip() or "inductor"
        fullgraph = bool(self.args.get("compile_fullgraph", True))
        dynamic = bool(self.args.get("compile_dynamic", False))
        compile_mode = os.getenv("TORCH_COMPILE_MODE", "").strip() or None
        final_layer.compile(
            backend=backend,
            fullgraph=fullgraph,
            dynamic=dynamic,
            mode=compile_mode,
        )
        if self.rank == 0:
            logger.info(
                f"FSDP2: compiled FLUX output head with backend={backend}, "
                f"fullgraph={fullgraph}, dynamic={dynamic}, mode={compile_mode or 'default'}"
            )

    @staticmethod
    def _get_module_by_path(root: torch.nn.Module, path: str, *, option_name: str) -> torch.nn.Module:
        """Resolve a dot-separated attribute path on a module."""
        cur = root
        if not path:
            return cur
        for part in path.split("."):
            if not hasattr(cur, part):
                raise ValueError(f"{option_name}='{path}' is invalid: missing attribute '{part}'")
            cur = getattr(cur, part)
        if not isinstance(cur, torch.nn.Module):
            raise ValueError(f"{option_name}='{path}' did not resolve to a torch.nn.Module")
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
        self._finalize_checkpoint(path)

    def _save_checkpoint(self):
        path = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
        if self.save_strategy == "dit_only":
            self._save_dit(os.path.join(path, "dit_model.safetensors"))
            self._finalize_checkpoint(path)
        else:
            self._save_dtcp(path)
        self._prune_checkpoints()

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
        self._restore_rng_state(path)

    def _latest_checkpoint(self):
        if not os.path.isdir(self.output_dir):
            return None
        checkpoints = []
        for name in os.listdir(self.output_dir):
            match = re.fullmatch(r"checkpoint-(\d+)", name)
            path = os.path.join(self.output_dir, name)
            if match and os.path.isfile(os.path.join(path, ".complete")):
                checkpoints.append((int(match.group(1)), path))
        return max(checkpoints, default=(None, None))[1]

    def _finalize_checkpoint(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(self.device),
            },
            os.path.join(path, f"rng-rank-{self.rank}.pt"),
        )
        if self.world_size > 1:
            dist.barrier()
        if self.rank == 0:
            open(os.path.join(path, ".complete"), "a", encoding="utf-8").close()
        if self.world_size > 1:
            dist.barrier()

    def _restore_rng_state(self, path):
        rng_path = os.path.join(path, f"rng-rank-{self.rank}.pt")
        if not os.path.isfile(rng_path):
            if self.rank == 0:
                logger.warning("Checkpoint has no RNG state; resumed trajectory may differ")
            return
        state = torch.load(rng_path, map_location="cpu", weights_only=False)
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state(state["cuda"], self.device)

    def _prune_checkpoints(self):
        if self.rank != 0 or self.save_total_limit <= 0:
            return
        checkpoints = []
        for name in os.listdir(self.output_dir):
            match = re.fullmatch(r"checkpoint-(\d+)", name)
            path = os.path.join(self.output_dir, name)
            if match and os.path.isfile(os.path.join(path, ".complete")):
                checkpoints.append((int(match.group(1)), path))
        for _, path in sorted(checkpoints)[: -self.save_total_limit]:
            shutil.rmtree(path)
            logger.info(f"Removed old checkpoint: {path}")

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


def build_fsdp2_trainer(
    *, model, dataset, processor, trainer_args: dict, eval_dataset=None, eval_processor=None
):
    rank, world_size, local_rank = setup_distributed()
    return FSDP2Trainer(
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
