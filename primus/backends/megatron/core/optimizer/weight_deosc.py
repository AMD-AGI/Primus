###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Weight de-oscillation for Primus-Turbo MXFP4 training on the Megatron backend.

Background
----------
When a weight is trained in MXFP4, the forward GEMM re-quantizes the (bf16)
weight every step. If an element sits near a quantization-bin boundary, a tiny
fp32-master movement can make the *quantized* value flip back and forth between
adjacent bins while the master barely moves. This oscillation hurts convergence.

The mitigation (ported from ALTO's ``han/weight-deosc`` branch) is a
``DistRatio`` detector evaluated over a fixed ``period`` window of optimizer
steps. For each element we accumulate::

    dist_w     = sum_t |w_t     - w_{t-1}|          # fp32 master movement
    dist_w_qdq = sum_t |Q(w_t)  - Q(w_{t-1})|       # quantized-value movement

At the end of a period, any element whose ``dist_w_qdq / dist_w`` exceeds a
threshold is "snapped" to its current quantization-bin center ``Q(w)`` so that
future small gradients no longer keep flipping it.

Why this lives in the optimizer instead of a per-tensor op
----------------------------------------------------------
``Q(w) = dequant(quant(w))`` must use the *same* quantization grid as the
forward GEMM. The Primus-Turbo MXFP4 weight path uses 2D (32x32) block scaling
(``ScalingRecipe(use_2d_block=True)``, ``axis=-1``, ``block_size=32``), so
``Q(w)[i, j]`` depends on the entire 32x32 tile that contains ``(i, j)`` -- i.e.
the original 2D coordinates of that element.

Megatron's distributed optimizer, however, flattens every parameter and splits
it into contiguous 1D slices across DP ranks. A single rank's fp32-master shard
can start/stop in the middle of a row (or tile). To avoid a synchronous
all-gather on every optimizer step, this module reconstructs only the
tile-aligned 2D regions touched by the local shard. Missing elements in the two
cross-rank boundary tile rows are zero-filled. Interior 32x32 tiles therefore
match the forward quantizer exactly; only boundary tiles use an approximation.

The local fp32 master is cast to the model dtype (normally bf16) before QDQ so
the quantizer sees the same input precision as the forward GEMM. Snapshots of
the master and of Q(w) are kept in BF16; DistRatio window sums and the snap
write-back into the fp32 master stay FP32.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from primus.core.utils.module_utils import log_rank_0, warning_rank_0

# Primus-Turbo quantization primitives (mirror the forward weight path in
# primus.backends.megatron.core.extensions.primus_turbo).
try:
    from primus_turbo.pytorch.core import QuantizedTensor as _PrimusTurboQuantizedTensor
except (ImportError, ModuleNotFoundError):
    _PrimusTurboQuantizedTensor = None

try:
    from primus_turbo.pytorch.core.low_precision import ScalingRecipe as _ScalingRecipe
except (ImportError, ModuleNotFoundError):
    try:
        from primus_turbo.pytorch.core.low_precision import (
            MXScalingRecipe as _ScalingRecipe,
        )
    except (ImportError, ModuleNotFoundError):
        _ScalingRecipe = None

try:
    from primus_turbo.pytorch.core.low_precision import (
        ScalingGranularity as _ScalingGranularity,
    )
    from primus_turbo.pytorch.core.low_precision import (
        float4_e2m1fn_x2 as _float4_e2m1fn_x2,
    )
except (ImportError, ModuleNotFoundError):
    _ScalingGranularity = None
    _float4_e2m1fn_x2 = None

# Block size used by the Primus-Turbo MXFP4 weight path (== 32).
try:
    from primus.backends.megatron.core.fp4_utils import MXFP4_SCALING_BLOCK_SIZE
except (ImportError, ModuleNotFoundError):
    MXFP4_SCALING_BLOCK_SIZE = 32

# prev / prev_q (and the per-step w, q views used to update them) live in BF16.
# dist_w / dist_w_qdq and the snap write-back into the fp32 master stay FP32.
_SNAP_DTYPE = torch.bfloat16


def _as_snap(t: torch.Tensor) -> torch.Tensor:
    """Detach ``t`` as a BF16 snapshot view/copy without an extra fp32 clone."""
    x = t.detach()
    return x if x.dtype == _SNAP_DTYPE else x.to(dtype=_SNAP_DTYPE)


@dataclass
class WeightDeOscConfig:
    """Configuration for MXFP4 weight de-oscillation.

    Attributes:
        enable: Master switch.
        period: Number of optimizer steps per observe/reset window.
        ratio_threshold: DistRatio threshold above which an element is snapped.
        start_step: Global optimizer step at which tracking begins.
        log_freq: Log a summary every ``log_freq`` periods (0 disables logging).
    """

    enable: bool = False
    period: int = 200
    ratio_threshold: float = 4.0
    start_step: int = 0
    log_freq: int = 0

    def validate(self) -> None:
        if not self.enable:
            return
        if self.period <= 0:
            raise ValueError(f"weight_deosc_period must be > 0, got {self.period}")
        if self.ratio_threshold <= 0:
            raise ValueError(f"weight_deosc_ratio must be > 0, got {self.ratio_threshold}")
        if self.start_step < 0:
            raise ValueError(f"weight_deosc_start_step must be >= 0, got {self.start_step}")
        if self.log_freq < 0:
            raise ValueError(f"weight_deosc_log_freq must be >= 0, got {self.log_freq}")


def deosc_dependencies_available() -> Tuple[bool, str]:
    """Return whether the Primus-Turbo MXFP4 QDQ primitives are importable."""
    if _PrimusTurboQuantizedTensor is None:
        return False, "primus_turbo.pytorch.core.QuantizedTensor is unavailable"
    if _ScalingRecipe is None:
        return False, "primus_turbo ScalingRecipe / MXScalingRecipe is unavailable"
    if _ScalingGranularity is None or _float4_e2m1fn_x2 is None:
        return False, "primus_turbo low_precision MXFP4 symbols are unavailable"
    return True, ""


@torch.no_grad()
def qdq_mxfp4(weight: torch.Tensor) -> torch.Tensor:
    """Quantize-dequantize ``weight`` exactly as the Primus-Turbo forward weight path.

    Mirrors ``PrimusTurbo*Linear.forward_internal`` (the FP4 branch):
    ``MX_BLOCKWISE`` granularity, ``float4_e2m1fn_x2``, ``block_size=32``,
    ``ScalingRecipe(use_2d_block=True)``, quantized along ``axis=-1``.

    Supports 2D dense weights ``[out, in]`` and 3D grouped expert weights
    ``[num_experts, out, in]``. The grouped case is handled per-expert because
    the single-direction MXFP4 kernel only accepts 2D input; this reproduces the
    Primus-Turbo grouped MXFP4 forward weight operand (PR #398), which quantizes
    each expert row-wise along the K (in) axis with ``use_2d_block=True``.
    Whether a grouped weight is actually de-osc eligible still depends on the
    grouped FP4 forward setting ``quantized_weight_buffer`` on its module.
    """
    recipe = _ScalingRecipe(use_2d_block=True)

    def _qdq_2d(w2d: torch.Tensor) -> torch.Tensor:
        qt = _PrimusTurboQuantizedTensor.quantize(
            w2d,
            dest_dtype=_float4_e2m1fn_x2,
            granularity=_ScalingGranularity.MX_BLOCKWISE,
            block_size=MXFP4_SCALING_BLOCK_SIZE,
            scaling_recipe=recipe,
            axis=-1,
        )
        out = qt.dequantize()
        # dequantize() only un-pads the last dim; defensively restore the exact
        # original 2D shape so the flat slice mapping below stays aligned.
        if out.shape != w2d.shape:
            out = out[tuple(slice(0, s) for s in w2d.shape)].contiguous()
        return out.to(w2d.dtype)

    if weight.ndim == 2:
        return _qdq_2d(weight)
    if weight.ndim == 3:
        return torch.stack([_qdq_2d(weight[g]) for g in range(weight.shape[0])], dim=0)
    raise ValueError(f"qdq_mxfp4 expects a 2D or 3D weight, got {weight.ndim}D")


@torch.no_grad()
def qdq_mxfp4_local_shard(
    shard_main_param: torch.Tensor,
    full_shape: Tuple[int, ...],
    start: int,
    end: int,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """QDQ a flattened local shard in its original 32x32 tile coordinates.

    ``start:end`` describes the shard in the flattened full 2D dense weight or
    3D grouped weight. Each grouped expert is an independent 2D matrix. Regions
    outside the local shard in a touched boundary tile row are represented by
    zeros, avoiding communication while preserving exact coordinates for all
    locally complete tiles.
    """
    shape = tuple(int(dim) for dim in full_shape)
    if len(shape) not in (2, 3):
        raise ValueError(f"local MXFP4 QDQ expects a 2D or 3D full shape, got {shape}")
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"local MXFP4 QDQ requires positive dimensions, got {shape}")

    total_numel = 1
    for dim in shape:
        total_numel *= dim
    if not (0 <= start <= end <= total_numel):
        raise ValueError(f"invalid local shard range [{start}, {end}) for shape {shape}")

    local_fp32 = shard_main_param.detach().reshape(-1)
    if local_fp32.numel() != end - start:
        raise ValueError(
            f"local shard has {local_fp32.numel()} elements, expected {end - start} "
            f"for range [{start}, {end})"
        )
    if local_fp32.numel() == 0:
        return local_fp32.to(dtype=model_dtype)

    # Forward quantizes bf16 model weights, not fp32 masters. Casting here
    # reproduces the copy-to-model rounding without gathering the model param.
    local_model = local_fp32.to(dtype=model_dtype)
    q_local = torch.empty_like(local_model)

    rows, cols = shape[-2:]
    matrix_numel = rows * cols
    first_matrix = start // matrix_numel
    last_matrix = (end - 1) // matrix_numel
    block = MXFP4_SCALING_BLOCK_SIZE

    for matrix_idx in range(first_matrix, last_matrix + 1):
        matrix_base = matrix_idx * matrix_numel
        local_begin = max(start, matrix_base)
        local_end = min(end, matrix_base + matrix_numel)
        begin_in_matrix = local_begin - matrix_base
        end_in_matrix = local_end - matrix_base

        first_row = begin_in_matrix // cols
        last_row = (end_in_matrix - 1) // cols
        tile_row_begin = (first_row // block) * block
        tile_row_end = ((last_row // block) + 1) * block

        tile = torch.zeros(
            (tile_row_end - tile_row_begin, cols),
            device=local_model.device,
            dtype=model_dtype,
        )
        tile_begin = begin_in_matrix - tile_row_begin * cols
        tile_end = end_in_matrix - tile_row_begin * cols
        shard_begin = local_begin - start
        shard_end = local_end - start
        tile.reshape(-1)[tile_begin:tile_end].copy_(local_model[shard_begin:shard_end])

        q_tile = qdq_mxfp4(tile)
        q_local[shard_begin:shard_end].copy_(q_tile.reshape(-1)[tile_begin:tile_end])

    return q_local


class _ParamDeOscState:
    """Per-(local-shard) tracking buffers, all sized to the local shard."""

    __slots__ = ("prev", "prev_q", "dist_w", "dist_w_qdq", "step")

    def __init__(self, w_local: torch.Tensor, q_local: torch.Tensor):
        self.prev = _as_snap(w_local).contiguous()
        self.prev_q = _as_snap(q_local).contiguous()
        self.dist_w = torch.zeros(self.prev.shape, device=self.prev.device, dtype=torch.float32)
        self.dist_w_qdq = torch.zeros_like(self.dist_w)
        self.step = 0

    def to_serializable(self) -> dict:
        return {
            "prev": self.prev.detach().cpu(),
            "prev_q": self.prev_q.detach().cpu(),
            "dist_w": self.dist_w.detach().cpu(),
            "dist_w_qdq": self.dist_w_qdq.detach().cpu(),
            "step": int(self.step),
        }

    @classmethod
    def from_serializable(cls, blob: dict, device, like: torch.Tensor) -> "_ParamDeOscState":
        """Rebuild from a checkpoint blob, only if it matches the current shard.

        Returns ``None`` when the saved shape does not match ``like`` (e.g. the
        checkpoint was taken under a different parallel layout); the caller then
        falls back to re-seeding, which is harmless for a window accumulator.
        """
        if tuple(blob["prev"].shape) != tuple(like.shape):
            return None
        obj = cls.__new__(cls)
        obj.prev = blob["prev"].to(device=device, dtype=_SNAP_DTYPE)
        obj.prev_q = blob["prev_q"].to(device=device, dtype=_SNAP_DTYPE)
        obj.dist_w = blob["dist_w"].to(device=device, dtype=torch.float32)
        obj.dist_w_qdq = blob["dist_w_qdq"].to(device=device, dtype=torch.float32)
        obj.step = int(blob["step"])
        return obj


class WeightDeOscRunner:
    """Drives MXFP4 weight de-oscillation for a single ``DistributedOptimizer``.

    Call :meth:`run` once per optimizer step, after the optimizer has updated
    the local fp32 master.
    """

    _EPS = 1e-12

    def __init__(self, config: WeightDeOscConfig):
        config.validate()
        self.config = config
        self._global_step = 0
        self._period_index = 0
        # Keyed by a stable structural key ("<param_name>|<start>:<end>") so the
        # state round-trips across checkpoint save/load under the same parallel
        # layout. Snapshots are BF16; DistRatio sums and the snap target stay FP32.
        self._state: Dict[str, _ParamDeOscState] = {}
        # id(model_param) -> stable param name, cached.
        self._param_name_cache: Dict[int, str] = {}
        # State loaded from a checkpoint, consumed lazily on first observation.
        self._loaded_params: Dict[str, dict] = {}
        # Lazily-built set of id(model_param) for weights actually quantized in
        # the FP4 forward (auto-excludes bf16 first/last layers and any layer
        # whose FP4 path never ran, e.g. grouped experts).
        self._eligible_ids: Optional[set] = None

    # ------------------------------------------------------------------
    # Stable keys (for in-memory tracking + checkpoint round-trip)
    # ------------------------------------------------------------------
    def _stable_key(self, dist_opt, model_param, start: int, end: int) -> str:
        name = self._param_name_cache.get(id(model_param))
        if name is None:
            try:
                name = dist_opt._param_name(model_param)
            except Exception:
                name = f"param@{id(model_param)}"
            self._param_name_cache[id(model_param)] = name
        return f"{name}|{start}:{end}"

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------
    def _build_eligible_ids(self, dist_opt) -> set:
        """Collect weights of modules whose FP4 forward actually quantized them.

        A Primus-Turbo linear registers a ``quantized_weight_buffer`` that stays
        ``None`` unless its FP4 forward ran. This is a precise runtime signal of
        "this weight is re-quantized in the forward GEMM", so de-osc snaps only
        match weights the forward actually quantizes.
        """
        eligible: set = set()
        n_dense = 0
        n_grouped = 0
        model_chunks = getattr(dist_opt, "model_chunks", None)
        if not model_chunks:
            return eligible
        for chunk in model_chunks:
            modules = chunk.modules() if hasattr(chunk, "modules") else []
            for module in modules:
                if getattr(module, "quantized_weight_buffer", None) is None:
                    continue
                weight = getattr(module, "_parameters", {}).get("weight", None)
                if weight is None:
                    weight = getattr(module, "weight", None)
                if isinstance(weight, torch.Tensor):
                    if id(weight) not in eligible:
                        n_dense += 1
                    eligible.add(id(weight))
                # Grouped-linear consolidates experts into a 3D ``weights``
                # (G, out, in). The Primus-Turbo grouped MXFP4 forward
                # (PR #398, ``_quant_weight_dual``) quantizes it row-wise along
                # axis=-1 with use_2d_block=True, which is exactly what
                # ``qdq_mxfp4``'s per-expert 2D path reproduces. Eligible only
                # once the grouped FP4 forward actually sets quantized_weight_buffer.
                weights = getattr(module, "weights", None)
                if isinstance(weights, torch.Tensor):
                    if id(weights) not in eligible:
                        n_grouped += 1
                    eligible.add(id(weights))
        if eligible:
            # One-time summary so MoE (grouped) coverage is easy to confirm in logs.
            log_rank_0(
                f"[WeightDeOsc] eligible FP4 weights: {len(eligible)} "
                f"(dense 2D={n_dense}, grouped/MoE 3D={n_grouped})"
            )
        return eligible

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run(self, dist_opt) -> None:
        if not self.config.enable:
            return

        self._global_step += 1
        if self._global_step < self.config.start_step:
            return

        if self._eligible_ids is None or len(self._eligible_ids) == 0:
            self._eligible_ids = self._build_eligible_ids(dist_opt)
            if len(self._eligible_ids) == 0:
                # FP4 forward has not populated any quantized weight buffers yet;
                # retry on a later step.
                return

        shard_groups = getattr(dist_opt, "shard_fp32_from_float16_groups", None)
        model_groups = getattr(dist_opt, "model_float16_groups", None)
        if shard_groups is None or model_groups is None:
            return

        total_reset = 0
        total_elems = 0
        period_closed = False

        for shard_group, model_group in zip(shard_groups, model_groups):
            for shard_main_param, model_param in zip(shard_group, model_group):
                if shard_main_param is None or model_param is None:
                    continue
                if id(model_param) not in self._eligible_ids:
                    continue

                rng = dist_opt._get_model_param_range_map(model_param)["param"]
                start, end = rng.start, rng.end
                if end <= start:
                    continue

                # QDQ the local fp32 master after bf16 rounding, preserving the
                # original 32x32 tile coordinates and zero-filling only missing
                # cross-rank boundary values.
                w_local = shard_main_param.detach()
                q_local = qdq_mxfp4_local_shard(
                    w_local,
                    tuple(model_param.shape),
                    start,
                    end,
                    model_param.dtype,
                )

                key = self._stable_key(dist_opt, model_param, start, end)
                reset, elems, closed = self._track_and_snap(key, shard_main_param, w_local, q_local)
                total_reset += reset
                total_elems += elems
                period_closed = period_closed or closed

        if period_closed:
            self._period_index += 1
            if (
                self.config.log_freq > 0
                and self._period_index % self.config.log_freq == 0
                and total_elems > 0
            ):
                frac = 100.0 * total_reset / max(total_elems, 1)
                log_rank_0(
                    f"[WeightDeOsc] step={self._global_step} period={self._period_index} "
                    f"snapped {total_reset}/{total_elems} elems ({frac:.3f}%)"
                )

    # ------------------------------------------------------------------
    # Per-parameter tracking / reset
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _track_and_snap(
        self,
        key: str,
        shard_main_param: torch.Tensor,
        w_local: torch.Tensor,
        q_local: torch.Tensor,
    ) -> Tuple[int, int, bool]:
        state = self._state.get(key)
        w_snap = _as_snap(w_local)
        q_snap = _as_snap(q_local)
        n_elem = w_snap.numel()

        if state is None:
            # Restore from a loaded checkpoint if the shard matches, else seed.
            loaded = self._loaded_params.pop(key, None)
            if loaded is not None:
                state = _ParamDeOscState.from_serializable(loaded, w_snap.device, w_snap)
            if state is None:
                # First observation: seed snapshots, do not track this step.
                self._state[key] = _ParamDeOscState(w_snap, q_snap)
                return 0, n_elem, False
            self._state[key] = state
            # fall through to track this step using the restored snapshots

        # Promote only the BF16 delta into the FP32 window accumulators.
        state.dist_w += (w_snap - state.prev).abs()
        state.dist_w_qdq += (q_snap - state.prev_q).abs()
        state.prev.copy_(w_snap)
        state.prev_q.copy_(q_snap)
        state.step += 1

        if state.step < self.config.period:
            return 0, n_elem, False

        # End of period: snap oscillating elements to the current bin center.
        ratio = state.dist_w_qdq / state.dist_w.clamp(min=self._EPS)
        reset_mask = (state.dist_w > 0) & (ratio >= self.config.ratio_threshold)

        reset_count = 0
        if reset_mask.any():
            reset_count = int(reset_mask.sum().item())
            q_flat = q_snap.reshape(-1)
            # Snap target is the dequantized bin center, written back in FP32.
            shard_main_param.data.view(-1)[reset_mask] = q_flat[reset_mask].to(dtype=shard_main_param.dtype)
            # Refresh master snapshot so the snap is not counted as a large
            # movement on the next period's first step. prev_q already equals
            # Q(snapped) because the snapped values are dequantized bin centers
            # (QDQ is idempotent on them).
            state.prev.copy_(w_snap)
            state.prev.view(-1)[reset_mask] = q_flat[reset_mask]

        state.dist_w.zero_()
        state.dist_w_qdq.zero_()
        state.step = 0
        return reset_count, n_elem, True

    # ------------------------------------------------------------------
    # Checkpoint persistence (per-rank; correct for same parallel layout)
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "global_step": int(self._global_step),
            "period_index": int(self._period_index),
            "params": {k: v.to_serializable() for k, v in self._state.items()},
        }

    def load_state_dict(self, sd: Optional[dict]) -> None:
        if not sd:
            return
        self._global_step = int(sd.get("global_step", 0))
        self._period_index = int(sd.get("period_index", 0))
        # Consumed lazily on each param's next observation (shape-checked there).
        self._loaded_params = dict(sd.get("params", {}))


def iter_deosc_runners(optimizer) -> List[WeightDeOscRunner]:
    """Return every :class:`WeightDeOscRunner` attached to ``optimizer``."""
    candidates = getattr(optimizer, "chained_optimizers", None) or [optimizer]
    runners: List[WeightDeOscRunner] = []
    for opt in candidates:
        runner = getattr(opt, "_primus_weight_deosc_runner", None)
        if runner is not None:
            runners.append(runner)
    return runners


def _current_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _sidecar_path(ckpt_dir: str, rank: int) -> str:
    return os.path.join(ckpt_dir, "weight_deosc", f"rank_{rank}.pt")


def save_deosc_sidecars(optimizer, ckpt_dir: Optional[str]) -> None:
    """Write this rank's de-oscillation state next to the checkpoint.

    Format-agnostic (works for both legacy and torch_dist checkpoints): each
    rank writes its own ``<ckpt_dir>/weight_deosc/rank_<R>.pt``. Correct for
    resume under the same parallel layout; mismatched shards are dropped on load
    and simply re-seeded.
    """
    runners = iter_deosc_runners(optimizer)
    if not runners or not ckpt_dir:
        return
    sub = os.path.join(ckpt_dir, "weight_deosc")
    os.makedirs(sub, exist_ok=True)
    blob = {str(i): runner.state_dict() for i, runner in enumerate(runners)}
    torch.save(blob, _sidecar_path(ckpt_dir, _current_rank()))


def load_deosc_sidecars(optimizer, ckpt_dir: Optional[str]) -> None:
    """Restore this rank's de-oscillation state from a checkpoint, if present."""
    runners = iter_deosc_runners(optimizer)
    if not runners or not ckpt_dir:
        return
    path = _sidecar_path(ckpt_dir, _current_rank())
    if not os.path.isfile(path):
        return
    try:
        blob = torch.load(path, map_location="cpu")
    except Exception as exc:
        warning_rank_0(f"[WeightDeOsc] failed to load sidecar {path}: {exc}")
        return
    for i, runner in enumerate(runners):
        runner.load_state_dict(blob.get(str(i)))


def _uses_precision_aware_main_params(opt) -> bool:
    """True if the optimizer holds bf16 main params inside FusedAdam.

    With ``use_precision_aware_optimizer`` the distributed optimizer does not
    keep a separate fp32 master shard: ``shard_fp32_from_float16_groups`` is
    filled with ``None`` and the main params live inside FusedAdam. De-osc reads
    those shards for dist_w and as the snap target, so this mode is unsupported
    and must be detected explicitly (otherwise de-osc would silently no-op).
    """
    cfg = getattr(opt, "config", None)
    if cfg is not None and (
        getattr(cfg, "use_precision_aware_optimizer", False)
        or getattr(cfg, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
    ):
        return True
    # Structural fallback: float16 params exist but every main shard is None.
    saw_slot = False
    for group in getattr(opt, "shard_fp32_from_float16_groups", None) or []:
        for shard_main_param in group:
            saw_slot = True
            if shard_main_param is not None:
                return False
    return saw_slot


def install_weight_deosc(optimizer, config: WeightDeOscConfig) -> int:
    """Attach a :class:`WeightDeOscRunner` to every distributed optimizer instance.

    Wraps each ``DistributedOptimizer.step_with_ready_grads`` so de-oscillation
    runs right after the inner step on the local fp32 master. Returns the number
    of distributed optimizer instances instrumented.
    """
    if not config.enable:
        return 0

    ok, reason = deosc_dependencies_available()
    if not ok:
        warning_rank_0(f"[WeightDeOsc] disabled: {reason}")
        return 0

    # Unwrap ChainedOptimizer if present.
    candidates = getattr(optimizer, "chained_optimizers", None)
    if candidates is None:
        candidates = [optimizer]

    instrumented = 0
    skipped_precision_aware = 0
    for opt in candidates:
        # Duck-type a DistributedOptimizer (avoid hard import / version coupling).
        if not (
            hasattr(opt, "shard_fp32_from_float16_groups")
            and hasattr(opt, "model_float16_groups")
            and hasattr(opt, "_get_model_param_range_map")
            and hasattr(opt, "step_with_ready_grads")
        ):
            continue
        if getattr(opt, "_primus_weight_deosc_installed", False):
            instrumented += 1
            continue

        # bf16 main params (use_precision_aware_optimizer) have no fp32 master
        # shard to track/snap -> de-osc cannot run. Skip with a clear warning
        # instead of silently doing nothing.
        if _uses_precision_aware_main_params(opt):
            skipped_precision_aware += 1
            warning_rank_0(
                "[WeightDeOsc] use_precision_aware_optimizer detected (bf16 main params held "
                "inside FusedAdam; no fp32 master shard). Weight de-oscillation is NOT supported "
                "in this mode and is skipped. Disable use_precision_aware_optimizer to use it."
            )
            continue

        runner = WeightDeOscRunner(config)
        original_step = opt.step_with_ready_grads

        def _make_wrapped(orig, run, bound_opt):
            def _wrapped(*args, **kwargs):
                ok_update = orig(*args, **kwargs)
                try:
                    run.run(bound_opt)
                except Exception as exc:  # never let de-osc crash training
                    warning_rank_0(f"[WeightDeOsc] skipped this step due to error: {exc}")
                return ok_update

            return _wrapped

        opt.step_with_ready_grads = _make_wrapped(original_step, runner, opt)
        opt._primus_weight_deosc_runner = runner
        opt._primus_weight_deosc_installed = True
        instrumented += 1

    if instrumented > 0:
        log_rank_0(
            f"[WeightDeOsc] enabled on {instrumented} distributed optimizer instance(s): "
            f"period={config.period}, ratio={config.ratio_threshold}, "
            f"start_step={config.start_step}"
        )
    elif skipped_precision_aware > 0:
        # Already warned per instance above; avoid the misleading "no instance" message.
        pass
    else:
        warning_rank_0(
            "[WeightDeOsc] no DistributedOptimizer instance found; de-oscillation not installed "
            "(requires use_distributed_optimizer=true)."
        )
    return instrumented
