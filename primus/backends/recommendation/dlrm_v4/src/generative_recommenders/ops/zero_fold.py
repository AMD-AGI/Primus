# pyre-ignore-all-errors
"""zero-fills (ROCm 10.2 port): per-step zero fills folded into their producers.

torch 2.16 runs a contiguous ``zero_()`` / ``torch.zeros`` as ``hipMemsetAsync`` (a rocclr
``fillBufferUnAligned`` blit on the compute stream; torch 2.12 ran a ``FillFunctor`` kernel), and rocPRIM's
host code under ``torch.sort`` issues 1 + 2 ``hipMemsetAsync`` per radix pass. Each fill is one more dispatch
(5-21 us) on the compute stream. The trainer's per-step fills that a kernel it launches anyway can absorb:

* ``dlrm_v4/train/ec_gpu_plan.py`` LPT EC segment reduce (``_CastExpandFp32Bwd`` x3 and
  ``GatherSegReduceBwd`` per step, [~54k, 512] fp32): ``torch.empty`` output; ``_plan_counts_z`` stores +0.0
  into the hot rows (the only rows whose first write is an atomic add), the cold section stores every other
  row (an empty segment stores +0.0). 4 fills / step.
* ``ops/preproc_dedup.py`` ``build_unique``: the stable sort of the int64 triple keys (< U^3, ~2.4M per step)
  runs on the onesweep extension (``ops/onesweep_sort.sort64_pairs``: rocPRIM's own gfx950 <int64, int64>
  onesweep kernels, over bit_length(U^3 - 1) bits = 6 passes instead of 8, ONE fill instead of 17, no iota
  kernel). The stable permutation is unique, so (keys, order) are torch.sort's bits; the first sort of every key
  width also runs torch.sort and compares raw bits (fail closed to torch.sort). 16 fills / step.
* ``ops/triton/triton_position.py`` ``_posenc_concat_backward``: the pos / ts fp32 accumulators are two views
  of ONE zero-filled buffer. 1 fill / step.
* ``dlrm_v4/host_sync_free.py`` GAUC patch: one ``asynchronous_complete_cumsum`` (a fill + a scan) for the three
  ``gauc.to_3d`` calls on the same ``num_candidates`` (first call raw-bit checked vs ``gauc.to_3d``). 2 fills / step.
* Not in this tree: the zero-fills lane's cache of the FP8 out-proj wgrad's saved y dequant scale in
  ``ops/triton/triton_hstu_linear.py`` (2 FillFunctor / step). It edits ``HSTUComputeOutputFunction.forward``,
  whose AST digest and every decorator line before it ``ops/tests/l11_do8_mx6_seam_test.py`` pins.
* ``modules/multitask_module.py``: the ``ones_like`` default supervision weight is built only when a task lacks one.

Every site writes the values a zero fill (or torch.sort) gave, bit for bit. Kill switch:
``DLRM_HSTU_FP8_ZERO_FOLD=0`` (read once, at import) restores the zero fills and torch.sort.
"""
import os

_ON: bool = os.environ.get("DLRM_HSTU_FP8_ZERO_FOLD", "1").strip().lower() not in (
    "0", "false", "off", "no")


def enabled() -> bool:
    return _ON
