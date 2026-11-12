"""MXFP4 weight de-oscillation for FLUX (FLUX_FP4_DEOSC).

A weight near the boundary between two MXFP4 bins can flip across it every step while the
underlying fp32 value barely moves. The GEMM then sees a parameter swinging by a full bin
while the optimizer thinks it is converging, and Adam's moments are driven by the noise. The
fix, from the GPT-OSS-20B MXFP4 recipe, is to detect those elements and park them on a bin
centre so they stop flipping:

  * at the start of a window, snapshot the master weight and its quantized image;
  * over the window, accumulate how far EACH moves, element-wise;
  * at the end, an element whose quantized image moved ``ratio`` times further than its master
    is oscillating -- real learning moves both together, a bin flip moves only the quantized
    one;
  * for those elements, write the dequantized value back into the master.

Ported from Primus ``primus/backends/megatron/core/optimizer/weight_deosc.py``. The algorithm is
theirs; the plumbing is not, because that version drives Megatron's DistributedOptimizer over
flat per-shard buffers and FLUX runs FSDP2.

Two things make the FLUX port simpler than the Megatron one, and both are worth stating because
they are the reason this file has no shard arithmetic in it:

  * MLPerf FLUX keeps TorchTitan-aligned FP32 FSDP parameters and an ordinary AdamW -- the
    trainer refuses FP32_MASTER_WEIGHTS outright. So the parameter IS the master. There is no
    optimizer-state master to locate and no risk of snapping a copy that the next step
    overwrites.
  * FSDP2 shards a Linear weight ``[out, in]`` as DTensor ``Shard(0)``, i.e. whole rows. MXFP4
    scales 32 contiguous elements along the last dim, so a block never straddles a rank and the
    local quantization equals the global one. ``_local_rows`` asserts this rather than assuming
    it; a future mesh that shards the contraction dim must not silently produce a wrong grid.

Two traps inherited from the reference, both of which it learned the hard way:

  * The QDQ here MUST use the same E8M0 scale rounding mode as the forward. Primus shipped a
    version that snapped at mode 0 while the forward ran mode 2 (AMD-AGI/Primus#1184), so the
    snap targets were not points of the grid the weights actually live on. We read the same
    FLUX_FP4_SCALE_ROUNDING the packs read.
  * The snap uses ``torch.where``, not boolean-mask indexing. Mask indexing needs the mask's
    population on the host, which synchronizes the device once per parameter per window -- 228
    syncs. Nothing in this file calls ``.item()`` except the logging path, and only on the steps
    that log.

KNOWN DIVERGENCE from FLUX's forward, stated plainly because it is not obviously harmless: the
forward packs through the fused H16 kernels, which quantize in a Hadamard-rotated basis, while
this file quantizes unrotated, exactly as the reference does. The detector is therefore measuring
movement on a related grid rather than the identical one, and a snapped value is a bin centre of
the unrotated grid. The reference recipe also runs RHT and still converges, so this is the
behaviour that has evidence behind it; doing it in the rotated basis would need the inverse
rotation to write a master weight back and has none.

Installed as ``primus/backends/diffusion/optim/flux_weight_deosc.py`` by flux_weight_deosc.py.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

MXFP4_BLOCK_SIZE = 32
_SNAP_DTYPE = torch.bfloat16
_EPS = 1e-12

# Six distinct Linear shapes at hidden 3072, so the compiled accumulator specializes six times.
# The default limit of 8 would survive that, but only just, and a silent fall back to eager would
# cost 50% of the accumulation without saying so.
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)


@torch.compile(dynamic=False)
def _accumulate(w_snap, q_snap, prev, prev_q, dist_w, dist_q):
    """Element-wise movement of master and quantized image, fused into one pass.

    MEASURED on m11-17 at FLUX's real shapes (228 parameters, 1.08e9 local elements): 8.70 ms
    against 13.05 ms for the obvious eager form, a 33% saving on what is two thirds of the whole
    de-oscillation cost. The win is bandwidth, not launch count -- batching the same eager ops
    with torch._foreach_* made it WORSE (15.95 ms), because multi-tensor-apply adds a copy.

    In the training loop that is 390.0 ms -> 380.0 ms median step time against a 370.0 ms
    no-de-osc baseline, i.e. the overhead halves from 5.4% to 2.7%. Both fused runs read 380.0,
    so the step-time win reproduces even though the snapped counts below do not.

    Two constraints, both of which produced a wrong answer before they were understood:

      * Every operand must ALREADY be bf16. Passing the fp32 master and casting inside looks
        tidier and is what the first version did, but inductor declines to materialize the
        downcast and does the subtract in fp32, so |w - bf16(w)| -- half a bf16 ULP, ~2e-3 at
        these magnitudes -- enters dist_w as movement that never happened. Against a real
        per-step movement of order 1e-4 that is not a rounding difference, it is a wrong
        number: it accumulated to 1.05e-2 against a 7.99e-4 signal.
      * Nothing this function reads may also be written here. Refreshing `prev` inside the
        fused region let the write land before the read.

    Not bit-identical to eager even so, because inductor keeps the subtract's intermediate in
    fp32, and the real run does NOT reproduce the synthetic test's clean result. Both numbers,
    because the synthetic one on its own would be misleading:

      * synthetic, a 100-step window on 3.5e6 elements with a coarser quantized grid: both
        forms snapped exactly the same 651,914 elements, zero decision flips, 5.4e-4 relative
        difference on dist_w and 6.2e-3 on dist_q.
      * in the training loop at window 1, where nothing has been snapped yet so the two are
        still comparable: eager 80,524,791 against fused 81,219,081, a 0.86% difference. Two
        runs of the SAME fused code differ by 0.24% (81,219,081 and 81,414,249), so the spread
        is roughly 3.6x run-to-run noise and is probably real rather than nondeterminism. It
        also has a direction: both fused runs snap MORE than eager.

    Accepted anyway, and the reason is a judgement rather than a measurement: 0.86% more of a
    population that is itself ~7.5% of the shard is a smaller perturbation than moving
    FLUX_FP4_DEOSC_RATIO from 4.00 to about 3.98, and that threshold is an untuned heuristic
    inherited from a recipe for a different model. Do not quote "numerically equivalent" --
    it is not; it is a sub-threshold-noise shift in a knob nobody has swept.
    """
    dist_w += (w_snap - prev).abs()
    dist_q += (q_snap - prev_q).abs()


def _env_int(name: str, default: int) -> int:
    # `or default`, not a getenv default: the launcher forwards every name the config exports and
    # materialises its value from the live environment, so an unset knob arrives as "".
    return int(os.getenv(name) or default)


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name) or default)


class _ParamState:
    """Per-parameter window state, all sized to the LOCAL shard."""

    __slots__ = ("prev", "prev_q", "dist_w", "dist_q", "steps")

    def __init__(self, w: torch.Tensor, q: torch.Tensor):
        # Snapshots in bf16 and accumulators in fp32, as the reference does. The snapshots only
        # ever feed a difference that is then accumulated in fp32, so bf16 costs nothing real and
        # halves the resident bytes; the accumulators are summed over a whole window and must not
        # lose the small per-step movements that are the entire signal.
        self.prev = w.to(_SNAP_DTYPE).contiguous()
        self.prev_q = q.to(_SNAP_DTYPE).contiguous()
        self.dist_w = torch.zeros_like(self.prev, dtype=torch.float32)
        self.dist_q = torch.zeros_like(self.dist_w)
        self.steps = 0


class WeightDeOsc:
    """Drives MXFP4 weight de-oscillation over a model's MXFP4 Linear weights.

    Call :meth:`step` once per optimizer step, AFTER ``optimizer.step()`` -- the window measures
    the movement the optimizer just applied, and the snap has to land on the value the next
    forward will quantize.
    """

    def __init__(self, model, scale_rounding_mode: int = 0):
        self.period = _env_int("FLUX_FP4_DEOSC_PERIOD", 100)
        self.ratio = _env_float("FLUX_FP4_DEOSC_RATIO", 4.0)
        # Default to the end of warmup, and expect to sweep it. Do NOT copy the 768 from the
        # GPT-OSS recipe: that number was found by trial and error there, and it does not mean
        # what it looks like. Their lr_warmup_iters is 128, so 768 is SIX times their warmup --
        # and also exactly their eval_interval -- roughly 10% into a ~7300-step run.
        #
        # Neither reading transfers. FLUX's warmup is 1600 of the ~8800 steps it takes to reach
        # the target, i.e. 18% of the run against their 1.8%, so scaling their 10% gives ~920,
        # which is inside our warmup. The two readings disagree here, and the only defensible
        # floor left is the end of warmup: before that the weights are travelling fast enough
        # that the ratio test has nothing to find, and a snap applied to a weight still in flight
        # just discards optimizer progress.
        #
        # Read from WARMUP_STEPS rather than pinned, because the pinned number drifts: it is 800
        # on one node and 1600 at four, and it is set in config_MI355X_4x8x32.sh AFTER
        # config_common.sh has run, so a default written there cannot see it.
        warmup = _env_int("WARMUP_STEPS", 0)
        self.start_step = _env_int("FLUX_FP4_DEOSC_START_STEP", 0) or warmup
        if warmup and self.start_step < warmup:
            raise ValueError(
                f"FLUX_FP4_DEOSC_START_STEP={self.start_step} is inside the {warmup}-step LR "
                "warmup. Snapping a weight that is still travelling discards optimizer progress; "
                "if this is deliberate, say so by setting WARMUP_STEPS too."
            )
        self.log_freq = _env_int("FLUX_FP4_DEOSC_LOG_FREQ", 1)
        self.scale_rounding_mode = int(scale_rounding_mode)
        if self.period <= 0:
            raise ValueError(f"FLUX_FP4_DEOSC_PERIOD must be > 0, got {self.period}")
        if self.ratio <= 0:
            raise ValueError(f"FLUX_FP4_DEOSC_RATIO must be > 0, got {self.ratio}")
        if self.start_step < 0:
            raise ValueError(f"FLUX_FP4_DEOSC_START_STEP must be >= 0, got {self.start_step}")

        self._params = self._eligible(model)
        self._state: dict[int, _ParamState] = {}
        self._windows = 0
        # Reset counts live on the device and are only read on a logging step, so the common step
        # does no host synchronization at all.
        self._reset_count = None
        # LOCAL elements, via the same shard view the window actually tracks. A plain p.numel() on
        # an FSDP2 parameter returns the GLOBAL count, which would be dp_shard times too large and
        # would divide the snapped fraction below by the wrong denominator -- the counts are
        # per-rank, so the percentage would read 8x low at one node and 32x low at four.
        self._tracked = sum(self._local_rows(p.data).numel() for _, p in self._params)
        logger.info(
            f"FLUX_FP4_DEOSC: {len(self._params)} MXFP4 linears, {self._tracked} local elements, "
            f"start_step={self.start_step} period={self.period} ratio={self.ratio} "
            f"scale_rounding_mode={self.scale_rounding_mode}"
        )
        if not self._params:
            raise ValueError(
                "FLUX_FP4_DEOSC=1 but no MXFP4 Linear has an mxfp4 fprop -- de-oscillation would "
                "track nothing and the run would quietly measure the baseline. Check "
                "FLUX_FP4_PASSES / FLUX_FP4_FORWARD."
            )

    @staticmethod
    def _eligible(model):
        """MXFP4 Linears whose FPROP weight is actually quantized.

        Only the fprop weight operand matters: an oscillating weight hurts because the forward
        sees it flip. A recipe with an FP8 or bf16 forward and MXFP4 only in the backward has no
        oscillating forward weight to suppress, and must not be tracked -- that is the same
        ``cfg.fprop`` gate the all-gather eligibility helpers use.
        """
        from primus.backends.diffusion.models.quantization.mxfp4_linear import (
            MXFP4Linear,
        )

        out = []
        for name, module in model.named_modules():
            if not isinstance(module, MXFP4Linear):
                continue
            if getattr(module.config, "fprop", None) != "mxfp4":
                continue
            weight = getattr(module, "weight", None)
            if weight is None or not weight.requires_grad:
                continue
            out.append((f"{name}.weight", weight))
        return out

    @staticmethod
    def _local_rows(param: torch.Tensor) -> torch.Tensor:
        """The local shard, asserted to hold whole rows.

        MXFP4 scales 32 contiguous elements of the last dim together. If a rank held a partial
        row, its blocks would differ from the ones the forward builds on the gathered weight and
        every ratio computed here would be against the wrong grid -- silently, since the shapes
        still work. FSDP2's Shard(0) gives whole rows; this refuses anything else.
        """
        local = param.to_local() if hasattr(param, "to_local") else param
        if local.ndim != 2 or local.shape[-1] != param.shape[-1]:
            raise ValueError(
                "FLUX_FP4_DEOSC needs a row-sharded 2D weight so MXFP4 blocks stay rank-local; "
                f"got local {tuple(local.shape)} from global {tuple(param.shape)}"
            )
        if local.shape[-1] % MXFP4_BLOCK_SIZE:
            raise ValueError(
                f"MXFP4 block size {MXFP4_BLOCK_SIZE} does not divide the contraction dim "
                f"{local.shape[-1]}"
            )
        return local

    @torch.no_grad()
    def _qdq(self, w: torch.Tensor) -> torch.Tensor:
        """Quantize-dequantize onto the MXFP4 grid the forward uses.

        Cast to bf16 first: FSDP2 runs the forward at bf16 param dtype, so the value that
        actually reaches the packs is the bf16 image of this fp32 master, and quantizing the
        fp32 value directly would measure a grid no forward ever sees.
        """
        from primus_turbo.pytorch.core import QuantizedTensor
        from primus_turbo.pytorch.core.low_precision import (
            ScalingGranularity,
            ScalingRecipe,
            float4_e2m1fn_x2,
        )

        qt = QuantizedTensor.quantize(
            w.to(torch.bfloat16),
            dest_dtype=float4_e2m1fn_x2,
            granularity=ScalingGranularity.MX_BLOCKWISE,
            block_size=MXFP4_BLOCK_SIZE,
            scaling_recipe=ScalingRecipe(use_2d_block=False),
            scale_rounding_mode=self.scale_rounding_mode,
            axis=-1,
        )
        out = qt.dequantize()
        if out.shape != w.shape:
            # dequantize() only un-pads the last dim; restore the exact shape so the element-wise
            # accumulators below stay aligned.
            out = out[tuple(slice(0, s) for s in w.shape)].contiguous()
        return out

    @torch.no_grad()
    def step(self, global_step: int) -> None:
        if global_step < self.start_step:
            return

        closed = False
        for key, param in self._params:
            w = self._local_rows(param.data)
            q = self._qdq(w)
            state = self._state.get(id(param))
            if state is None:
                # Seed the window; there is no movement to measure yet.
                self._state[id(param)] = _ParamState(w, q)
                continue

            # Cast before the fused region, not inside it, and refresh the snapshots after it.
            # Both are load-bearing -- see _accumulate.
            w_snap = w.to(_SNAP_DTYPE)
            q_snap = q.to(_SNAP_DTYPE)
            _accumulate(w_snap, q_snap, state.prev, state.prev_q, state.dist_w, state.dist_q)
            state.prev.copy_(w_snap)
            state.prev_q.copy_(q_snap)
            state.steps += 1
            if state.steps < self.period:
                continue

            # dist_w in the denominator, clamped: an element the optimizer never moved has no
            # ratio to speak of, and `dist_w > 0` excludes it rather than letting the clamp
            # manufacture a huge one.
            ratio = state.dist_q / state.dist_w.clamp(min=_EPS)
            mask = (state.dist_w > 0) & (ratio >= self.ratio)
            w.copy_(torch.where(mask, q.to(w.dtype), w))
            # Re-snapshot AFTER the snap, so the jump we just applied is not counted as movement
            # at the start of the next window. prev_q needs no update: QDQ is idempotent on bin
            # centres, so the quantized image of a snapped element is already what it was.
            state.prev.copy_(w.to(_SNAP_DTYPE))
            state.dist_w.zero_()
            state.dist_q.zero_()
            state.steps = 0
            closed = True
            count = mask.sum()
            self._reset_count = count if self._reset_count is None else self._reset_count + count

        if closed:
            self._windows += 1
            if self.log_freq and self._windows % self.log_freq == 0 and self._reset_count is not None:
                n = int(self._reset_count.item())
                logger.info(
                    f"FLUX_FP4_DEOSC window {self._windows} @ step {global_step}: snapped {n} of "
                    f"{self._tracked} local elements ({100.0 * n / max(self._tracked, 1):.3f}%)"
                )
            self._reset_count = None


def build_weight_deosc(model):
    """Return a :class:`WeightDeOsc` when FLUX_FP4_DEOSC=1, else ``None``."""
    if os.getenv("FLUX_FP4_DEOSC", "0") != "1":
        return None
    return WeightDeOsc(model, scale_rounding_mode=_env_int("FLUX_FP4_SCALE_ROUNDING", 0))
