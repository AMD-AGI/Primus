###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
PrimusVeScaleRaggedShardFSDP
============================

FSDP implementation using veScale's **RaggedShard** DTensor for zero-copy
batched parameter communication.

Background
----------
PyTorch FSDP2 stores each parameter as a ``Shard(0)`` DTensor.  When multiple
parameters are batched into a single all-gather collective (one per FSDP unit /
module), they must be padded or interleaved-copied into a temporary buffer
because individual parameter sizes rarely align to the device count.

``RaggedShard`` eliminates this overhead:
  1. All parameters in one FSDP unit are **flattened into a single contiguous
     flat buffer** (flat_param).
  2. The flat buffer is sharded with ``RaggedShard(dims=(0,), local_units=...)``.
     Each rank holds a *contiguous slice* of the flat buffer — no padding or
     interleaving needed.
  3. All-gather  = ``flat_dtensor.redistribute([Replicate()])``
     → single all-gather on the flat buffer, zero-copy.
  4. Reduce-scatter = pack gradients → redistribute ``[Replicate()] →
     [RaggedShard(...)]``  → single reduce-scatter, zero-copy.

Comparison with PyTorch FSDP2 (Shard(0)):
  - FSDP2: N separate ``Shard(0)`` tensors → batched all-gather needs
    interleaved copy because tensor shapes differ.
  - veScale: 1 ``RaggedShard`` flat buffer → single zero-copy all-gather for
    the entire FSDP unit.

Lifecycle
---------
::

    init
      └─ flatten params → flat_full → distribute_tensor(RaggedShard)
                        → flat_dtensor (local shard per rank)

    forward (pre-hook)
      └─ flat_dtensor.redistribute([Replicate()]) → flat_full
         for each param: param.data = flat_full[offset:offset+numel].view(shape)

    backward (post-hook via register_full_backward_hook)
      └─ pack param.grad → grad_full
         DTensor.from_local(grad_full, [Replicate()]).redistribute([RaggedShard])
         → shard grad stored on flat_dtensor._local_tensor.grad

    optimizer step (external)
      └─ optimizer sees flat_shard_param (local shard, with .grad set)

Activation
----------
Enable via environment variable (opt-in, does not affect default FSDP2 path):

.. code-block:: bash

    export PRIMUS_VESCALE_RAGGED_SHARD_FSDP=1

and ensure ``use_torch_fsdp2: true`` in the trainer YAML config.

References
----------
- veScale-FSDP paper: https://arxiv.org/abs/2602.22437
- RaggedShard docs:   third_party/veScale/docs/texts/raggedshard.md
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Type

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


def _log_rank0(msg: str) -> None:
    """Log a message only from rank 0 (uses standard logging + Primus when available)."""
    try:
        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        rank = 0
    if rank != 0:
        return
    # Try Primus log_rank_0 first; fall back to standard logging
    try:
        from primus.modules.module_utils import log_rank_0

        log_rank_0(msg)
    except Exception:
        logger.info(msg)


def _warn_rank0(msg: str) -> None:
    """Warn only from rank 0."""
    try:
        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        rank = 0
    if rank != 0:
        return
    try:
        from primus.modules.module_utils import warning_rank_0

        warning_rank_0(msg)
    except Exception:
        logger.warning(msg)

# ---------------------------------------------------------------------------
# Lazy veScale import guard
# ---------------------------------------------------------------------------
_VESCALE_CHECKED: Optional[bool] = None


def _check_vescale() -> None:
    """Raise ImportError if veScale is not available."""
    global _VESCALE_CHECKED
    if _VESCALE_CHECKED is None:
        try:
            import vescale  # noqa: F401

            _VESCALE_CHECKED = True
        except ImportError:
            _VESCALE_CHECKED = False
    if not _VESCALE_CHECKED:
        raise ImportError(
            "veScale is required for PrimusVeScaleRaggedShardFSDP. "
            "Install it with:\n"
            "  pip install --ignore-requires-python -e third_party/veScale/"
        )


def is_vescale_available() -> bool:
    """Return ``True`` if ``vescale`` package can be imported."""
    try:
        _check_vescale()
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Parameter metadata
# ---------------------------------------------------------------------------
@dataclass
class _ParamMeta:
    """Metadata to reconstruct a single parameter from the flat buffer."""

    fqn: str  # Fully qualified name (relative to the FSDP unit root)
    shape: torch.Size
    dtype: torch.dtype
    numel: int
    offset: int  # Element offset inside the flat buffer


# ---------------------------------------------------------------------------
# One FSDP unit: wraps a single nn.Module
# ---------------------------------------------------------------------------
class _RaggedShardFSDPUnit:
    """
    Manages the flat-param / RaggedShard sharding for **one** FSDP module.

    Responsibilities
    ~~~~~~~~~~~~~~~~
    * ``__init__``        — flatten params → flat_full → distribute(RaggedShard)
    * ``all_gather``      — redistribute RaggedShard → Replicate (before fwd)
    * ``install_params``  — set param.data to slices of the all-gathered tensor
    * ``uninstall_params``— restore param.data to contiguous local storage
    * ``reduce_scatter_grads`` — pack param.grads → reduce-scatter to shard
    """

    def __init__(
        self,
        module: nn.Module,
        param_list: List[Tuple[str, nn.Parameter]],
        device_mesh,  # torch.distributed.DeviceMesh
        flat_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        _check_vescale()
        from vescale.dtensor import distribute_tensor
        from vescale.dtensor.placement_types import RaggedShard

        self.module = module
        self.device_mesh = device_mesh
        self.flat_dtype = flat_dtype
        self._flat_full: Optional[torch.Tensor] = None  # set during all_gather

        world_size: int = device_mesh.size()

        # ---- Build param metadata ----------------------------------------
        self.param_metas: List[_ParamMeta] = []
        offset = 0
        for fqn, param in param_list:
            meta = _ParamMeta(
                fqn=fqn,
                shape=param.shape,
                dtype=param.dtype,
                numel=param.numel(),
                offset=offset,
            )
            self.param_metas.append(meta)
            offset += param.numel()
        self.total_numel: int = offset

        if self.total_numel == 0:
            self.flat_dtensor = None
            self.ragged_shard = None
            log_rank_0(f"[RaggedShardFSDP] module {type(module).__name__} has no params, skipping.")
            return

        # ---- Build the flat global buffer from current param data ----------
        flat_global = torch.empty(self.total_numel, dtype=flat_dtype, device="cuda")
        for meta in self.param_metas:
            param = self._get_param(meta.fqn)
            flat_global[meta.offset : meta.offset + meta.numel].copy_(param.data.view(-1).to(flat_dtype))

        # ---- Compute RaggedShard local_units (contiguous equal-ratio split) -
        #
        # Equal split: local_units = (1, 1, ..., 1) means each rank gets
        # total_numel // world_size elements.  If total_numel % world_size != 0,
        # the last rank gets the remainder.
        base = self.total_numel // world_size
        remainder = self.total_numel % world_size
        raw_units: Tuple[int, ...] = tuple(
            base + (1 if i == world_size - 1 and remainder > 0 else 0) for i in range(world_size)
        )
        gcd = math.gcd(*raw_units) if len(raw_units) > 1 else raw_units[0]
        local_units: Tuple[int, ...] = tuple(u // gcd for u in raw_units)

        self.ragged_shard = RaggedShard(dims=(0,), local_units=local_units)

        # ---- Distribute flat buffer with RaggedShard -----------------------
        # distribute_tensor scatters data from rank-0 to all ranks.
        self.flat_dtensor = distribute_tensor(
            flat_global,
            device_mesh,
            [self.ragged_shard],
        )

        # ---- Replace param data with local shard slices -------------------
        self._store_local_shard_in_params()

        num_params = len(self.param_metas)
        local_numel = self.flat_dtensor.to_local().numel()
        log_rank_0(
            f"[RaggedShardFSDP] Wrapped {type(module).__name__}: "
            f"{num_params} params, total_numel={self.total_numel}, "
            f"local_shard_numel={local_numel}, "
            f"local_units={local_units}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_param(self, fqn: str) -> nn.Parameter:
        """Retrieve a parameter from the module by its fully qualified name."""
        parts = fqn.split(".")
        obj = self.module
        for part in parts:
            obj = getattr(obj, part)
        return obj  # type: ignore[return-value]

    def _store_local_shard_in_params(self) -> None:
        """
        After initial distribution, replace each param's .data with a slice of
        the **local shard**.

        The authoritative storage for training is ``flat_dtensor.to_local()``.
        Parameters are restored from the flat buffer on each all_gather call.
        """
        local_shard = self.flat_dtensor.to_local()  # 1D tensor (local numel)
        rank = self.device_mesh.get_local_rank()

        # Determine the global offset range covered by this rank's shard.
        lu = self.ragged_shard.local_units
        total_units = sum(lu)
        rank_start = sum(lu[:rank]) * self.total_numel // total_units
        rank_end = sum(lu[: rank + 1]) * self.total_numel // total_units

        for meta in self.param_metas:
            param = self._get_param(meta.fqn)
            p_start = meta.offset
            p_end = meta.offset + meta.numel

            overlap_start = max(p_start, rank_start)
            overlap_end = min(p_end, rank_end)

            if overlap_start < overlap_end:
                # This rank has some of this param — keep current data.
                pass
            param.data = param.data.detach()

    # ------------------------------------------------------------------
    # All-gather: RaggedShard → Replicate
    # ------------------------------------------------------------------

    def all_gather(self) -> None:
        """
        All-gather the flat buffer (RaggedShard → Replicate).

        This is a SINGLE collective for ALL parameters in this FSDP unit
        (as opposed to PyTorch FSDP2's per-parameter collectives).

        After this call:
          - ``self._flat_full`` holds the replicated full flat tensor.
          - Each parameter's ``.data`` is set to its corresponding slice.
        """
        if self.flat_dtensor is None:
            return

        from vescale.dtensor.placement_types import Replicate

        # ONE redistribute call = ONE all-gather for ALL params in this unit.
        full_dtensor = self.flat_dtensor.redistribute(
            placements=[Replicate()],
            async_op=False,
        )
        self._flat_full = full_dtensor.to_local()  # (total_numel,) on device

        self.install_params(self._flat_full)

    def install_params(self, flat_full: torch.Tensor) -> None:
        """
        Set each parameter's ``.data`` to its slice of *flat_full*.

        Parameters become views of the all-gathered flat tensor; gradients will
        flow into those views during backward.
        """
        for meta in self.param_metas:
            param = self._get_param(meta.fqn)
            chunk = flat_full[meta.offset : meta.offset + meta.numel]
            with torch.no_grad():
                param.data = chunk.view(meta.shape).to(meta.dtype)

    def uninstall_params(self) -> None:
        """
        Free the all-gathered flat tensor and restore parameter data to
        independent (contiguous) storage.

        Called after backward to release GPU memory.
        """
        if self._flat_full is None:
            return
        for meta in self.param_metas:
            param = self._get_param(meta.fqn)
            with torch.no_grad():
                param.data = param.data.clone()
        del self._flat_full
        self._flat_full = None

    # ------------------------------------------------------------------
    # Reduce-scatter gradients: Replicate → RaggedShard
    # ------------------------------------------------------------------

    def reduce_scatter_grads(self) -> None:
        """
        Reduce-scatter gradients from all ranks back to the local shard.

        Steps:
          1. Pack per-param gradients into a single ``grad_full`` tensor.
          2. Wrap as a ``DTensor`` with ``[Replicate()]`` placement.
          3. Redistribute to ``[RaggedShard]`` — triggers ONE reduce-scatter
             for ALL parameters in this FSDP unit.
          4. Store the resulting shard gradient on flat_dtensor._local_tensor.grad.
        """
        if self.flat_dtensor is None:
            return

        from vescale.dtensor import DTensor
        from vescale.dtensor.placement_types import Replicate

        # 1. Pack param.grad → grad_full
        grad_full = torch.zeros(self.total_numel, dtype=self.flat_dtype, device="cuda")
        has_any_grad = False
        for meta in self.param_metas:
            param = self._get_param(meta.fqn)
            if param.grad is not None:
                grad_full[meta.offset : meta.offset + meta.numel].add_(
                    param.grad.view(-1).to(self.flat_dtype)
                )
                has_any_grad = True
                param.grad = None  # Clear to avoid memory leak

        if not has_any_grad:
            return

        # 2. Wrap as Partial (= local contribution needing all-reduce / reduce-scatter)
        #    Each rank has computed its own local gradient; we need to SUM across
        #    ranks and SCATTER to give each rank its shard → Partial → RaggedShard.
        from vescale.dtensor.placement_types import Partial as _Partial

        grad_dtensor = DTensor.from_local(
            grad_full,
            self.device_mesh,
            [_Partial()],
            run_check=False,
        )

        # 3. ONE reduce-scatter for ALL gradients in this unit
        #    Partial → RaggedShard triggers sum+scatter (correct data-parallel reduce-scatter)
        shard_grad_dtensor = grad_dtensor.redistribute(
            placements=[self.ragged_shard],
            async_op=False,
        )
        shard_grad = shard_grad_dtensor.to_local()

        # 4. Store shard gradient on the local tensor of flat_dtensor
        local = self.flat_dtensor._local_tensor
        if local.grad is None:
            local.grad = shard_grad.clone()
        else:
            local.grad.add_(shard_grad)

    def zero_shard_grad(self) -> None:
        """Zero the local shard gradient (called before each training step)."""
        if self.flat_dtensor is not None:
            local = self.flat_dtensor._local_tensor
            if local.grad is not None:
                local.grad.zero_()


# ---------------------------------------------------------------------------
# Main FSDP class
# ---------------------------------------------------------------------------
class PrimusVeScaleRaggedShardFSDP:
    """
    FSDP-like wrapper using veScale ``RaggedShard`` DTensor for zero-copy
    batched parameter communication.

    Key advantage over PyTorch FSDP2
    ---------------------------------
    PyTorch FSDP2 issues one collective per parameter (or per small group).
    This class issues **ONE collective per FSDP unit** regardless of how many
    parameters are in it.  For a TransformerLayer with ~100M parameters split
    into dozens of tensors, this means 1 all-gather instead of N all-gathers.

    Activation
    ----------
    Set environment variable before launching::

        export PRIMUS_VESCALE_RAGGED_SHARD_FSDP=1

    Constructor signature mirrors ``PrimusTorchFullyShardedDataParallel``
    so the patch can swap them transparently.
    """

    def __init__(
        self,
        config,  # TransformerConfig
        ddp_config,  # DistributedDataParallelConfig
        module: nn.Module,
        sub_modules_to_wrap: Optional[List[Type[nn.Module]]] = None,
        process_group=None,
        flat_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> None:
        _check_vescale()

        from megatron.core import parallel_state, tensor_parallel
        from megatron.core.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )
        from megatron.core.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )
        from megatron.core.transformer.transformer_layer import TransformerLayer
        from torch.distributed.device_mesh import DeviceMesh

        if kwargs:
            warning_rank_0(f"[PrimusVeScaleRaggedShardFSDP] ignoring unknown kwargs: {kwargs}")

        self.module = module
        self.config = config
        self.ddp_config = ddp_config
        self.flat_dtype = flat_dtype

        if sub_modules_to_wrap is None:
            sub_modules_to_wrap = [
                TransformerLayer,
                LanguageModelEmbedding,
                RotaryEmbedding,
                tensor_parallel.ColumnParallelLinear,
            ]
        # Exclude ColumnParallelLinear for local transformer impl
        if getattr(config, "transformer_impl", "transformer_engine") == "local":
            sub_modules_to_wrap = [
                m for m in sub_modules_to_wrap if m is not tensor_parallel.ColumnParallelLinear
            ]

        # ---- Build process group / device mesh ----------------------------
        if process_group is None:
            process_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        self.process_group = process_group
        self.device_mesh = DeviceMesh.from_group(process_group, "cuda")

        # ---- Identify and wrap FSDP units ---------------------------------
        self._fsdp_units: List[_RaggedShardFSDPUnit] = []
        claimed_param_ids: Set[int] = set()

        for submod in module.modules():
            if not any(isinstance(submod, cls) for cls in sub_modules_to_wrap):
                continue
            extra: List[Type] = list(getattr(submod, "_fsdp_modules", []))
            is_target = any(isinstance(submod, cls) for cls in sub_modules_to_wrap + extra)
            if not is_target:
                continue

            param_list: List[Tuple[str, nn.Parameter]] = [
                (fqn, p)
                for fqn, p in submod.named_parameters()
                if id(p) not in claimed_param_ids and p.requires_grad
            ]
            if not param_list:
                continue

            unit = _RaggedShardFSDPUnit(
                module=submod,
                param_list=param_list,
                device_mesh=self.device_mesh,
                flat_dtype=flat_dtype,
            )
            self._fsdp_units.append(unit)
            for _, p in param_list:
                claimed_param_ids.add(id(p))

            self._register_hooks(submod, unit)

        # ---- Handle root-module params not claimed by any child unit ------
        root_params: List[Tuple[str, nn.Parameter]] = [
            (fqn, p)
            for fqn, p in module.named_parameters()
            if id(p) not in claimed_param_ids and p.requires_grad
        ]
        if root_params:
            root_unit = _RaggedShardFSDPUnit(
                module=module,
                param_list=root_params,
                device_mesh=self.device_mesh,
                flat_dtype=flat_dtype,
            )
            self._fsdp_units.append(root_unit)
            self._register_hooks(module, root_unit)

        log_rank_0(
            f"[PrimusVeScaleRaggedShardFSDP] Initialized with "
            f"{len(self._fsdp_units)} FSDP units, "
            f"device_mesh={self.device_mesh}. "
            f"Each FSDP unit uses 1 all-gather + 1 reduce-scatter "
            f"(vs N collectives in vanilla FSDP2)."
        )

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self, submod: nn.Module, unit: _RaggedShardFSDPUnit) -> None:
        """Register pre-forward and post-backward hooks for an FSDP unit."""

        def _pre_forward_hook(mod, args):
            """All-gather: RaggedShard → Replicate before forward."""
            unit.all_gather()

        def _post_forward_hook(mod, args, output):
            """Post-forward: keep flat_full alive until after backward."""
            pass  # flat_full released in _post_backward_hook

        def _post_backward_hook(mod, grad_input, grad_output):
            """After backward: reduce-scatter gradients + release flat_full."""
            unit.reduce_scatter_grads()
            unit.uninstall_params()

        submod.register_forward_pre_hook(_pre_forward_hook)
        submod.register_forward_hook(_post_forward_hook)
        submod.register_full_backward_hook(_post_backward_hook)

    # ------------------------------------------------------------------
    # Public interface matching _BaseDataParallel
    # ------------------------------------------------------------------

    def forward(self, *inputs, **kwargs):
        """Delegates to the wrapped module."""
        return self.module(*inputs, **kwargs)

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.__dict__["module"], name)

    def state_dict(self, prefix="", keep_vars=False, **kwargs):
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars, **kwargs)

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        """No-op: tensors already loaded in-place (same as FSDP2 behavior)."""

    def start_grad_sync(self, *unused) -> None:
        """
        Trigger reduce-scatter for all FSDP units.

        Called by the training loop to initiate gradient synchronization.
        In the non-overlapped path, this is called after the full backward pass.
        """
        for unit in self._fsdp_units:
            unit.reduce_scatter_grads()

    def finish_grad_sync(self) -> None:
        """Wait for async reduce-scatter ops to complete (no-op for sync path)."""

    def zero_grad_buffer(self) -> None:
        """Zero gradient buffers at the start of each training step."""
        for unit in self._fsdp_units:
            unit.zero_shard_grad()

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale all gradient shards (e.g. for gradient accumulation)."""
        for unit in self._fsdp_units:
            if unit.flat_dtensor is not None:
                local = unit.flat_dtensor._local_tensor
                if local.grad is not None:
                    local.grad.mul_(scaling_factor)

    def broadcast_params(self) -> None:
        """Broadcast parameters from rank 0 to all ranks (after checkpoint load)."""
        for unit in self._fsdp_units:
            if unit.flat_dtensor is None:
                continue
            local = unit.flat_dtensor._local_tensor
            torch.distributed.broadcast(local, src=0, group=self.process_group)

    def no_sync(self):
        """Context manager that disables gradient sync (compatible stub)."""
        import contextlib
        return contextlib.nullcontext()

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse)

    def named_modules(self, *args, **kwargs):
        return self.module.named_modules(*args, **kwargs)

    def modules(self):
        return self.module.modules()
