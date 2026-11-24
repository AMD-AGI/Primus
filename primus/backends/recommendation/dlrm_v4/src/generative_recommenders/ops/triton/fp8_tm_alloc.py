# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Token-major FP8 operands from a private, non-expandable caching-allocator pool.

DLRM_HSTU_FP8_TM_NONEXP=1 (default off; Arbor r14 iteration 37; read once at import).

What: the K = tokens FP8 weight-gradient GEMMs read one TOKEN-MAJOR operand,
[N, Mp] e4m3 with the tokens contiguous and rows Mp bytes apart:
  - the LN store_t copy xT of normed_x (uvqk / vk fp8t wgrad, B operand);
  - the out-proj dout dual-cast copy dout_t (out-proj F8SS NN wgrad, B operand).
Under the production allocator (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
launch_slurm.sh) plus the shipped pre-grow ("main=rest"), those tensors live in
the main pool's expandable segment. The SAME hipBLASLt wgrad kernel on the SAME
bytes runs 8-9% faster when only that operand comes from an ordinary hipMalloc'd
segment (uvqk wgrad, M = 2.65M, sustained soak at the 1000 W cap: 4.24 ->
3.86-3.91 ms per call; output bitwise identical; r14 probes/iter37/tm). Each
stream-K K-step touches one row per output row of the tile, i.e. hundreds of
pages Mp bytes apart; the hypothesis is GPU TLB reach (the expandable segment
is mapped in hipMemCreate chunks, an ordinary segment is one physically larger
allocation). Row-major operands read in contiguous K-steps are not affected.

How: `empty_tm` allocates such a tensor under torch.cuda.use_mem_pool(<private
pool of this device>) with the allocator setting switched to
expandable_segments:False for that one allocation (PyTorch then skips expandable
blocks and hipMallocs a plain segment). The private pool keeps those blocks for
these tensors only, so a freed block is reused by the next step's token-major
operand and never handed to an unrelated allocation. Only the allocation
changes: shapes, strides and every computed value are unchanged (bitwise by
construction). Inert (plain torch.empty) unless the process runs with
expandable segments on; any failure falls back to torch.empty, latched, logged.

Size classes: plain segments cannot merge, so exact-size requests would add
segments every time the token count reaches a new maximum (40 simulated steps
with M in [1.95M, 2.95M] grew the pool to 25 segments / 32 GiB). Each tensor is
therefore a view of a whole block of a size class -- the request rounded up to
512 MiB, at least 2 GiB (512 x 4.19M tokens, above any per-GPU token count seen
at 1n/2n/4n, so one class in practice) -- so a freed block is reused whole by
the next request and the pool holds one block per concurrently live token-major
tensor: three xT (L0, L1, L2, live forward -> wgrad) plus one dout_t, i.e.
~8-10 GiB (prewarm() maps five 2 GiB blocks before RUN_START, from the margin). (With a 1.5 GiB floor, 60 simulated steps with M in [1.95M, 3.3M]
held 5 blocks, then 10 once M passed 3.14M; 0 allocation retries.)

Memory: the pool is hipMalloc'd outside the pre-grown segment, i.e. from the
pre-grow margin (56 GiB by default; prewarm() maps it after the pre-grow).

xT placement, DLRM_HSTU_FP8_TM_NONEXP_XT (Arbor r14 iteration 38; read once at
import; only matters with TM_NONEXP active): "1" = the LN xT (`empty_xt`: store_t /
recompute_fp8t, the uvqk / vk fp8t wgrad operand) comes from the private pool as
above; "0" = plain torch.empty (the expandable main pool) for xT only, the dual-cast
dout_t stays in the private pool. Unset: "1" (the private pool, as before this knob),
also with the custom wgrad DLRM_HSTU_FP8_BWD_WGRAD_CK=1. Why not "0" for the custom
wgrad: it reads xT faster from the expandable pool at M >= 2.1M (same GPU, M = 2.65M,
W1 W1 W2 chain: 8.20 ms expandable vs 9.44 ms private; hipBLASLt 10.35 ms private,
11.22 ms expandable; r14 iter38 ck G4pe), but an M sweep (r14 iter38 probes/msweep,
2 GPUs, sustained) shows its expandable placement no faster than hipBLASLt-private at
M = 1.91M-1.99M (W1 1.05x / 0.97-1.01x), while the private placement is faster than
hipBLASLt-private at every M tested (W1 0.78-0.96x, W2 0.85-0.92x, M = 1.91M-3.1M). Without
xT the pool holds one dout_t, so the default prewarm drops to two blocks
(`default_blocks`). Only the allocation moves: every computed value is unchanged (the
producers zero-fill the xT tail either way).

Step-tiled operands, DLRM_HSTU_FP8_TM_NONEXP_TILED (Arbor r14 iteration 40; read once at
import; only matters with TM_NONEXP active): "1" (default, unchanged) = a 3-D request (the
step-tiled [KT, 512, 128] yT of DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED and dout_t of
_CK_W3_TILED; only the tiled producers ask for 3-D) comes from the private pool like the
row-major operands; "0" = a 3-D request is a plain torch.empty (the ordinary caching
allocator), and alloc_pregrow skips the prewarm when every operand the pool would serve is
step-tiled (`pooled_all_tiled`). Row-major operands (hipBLASLt-only configuration, a fallback
after a latch, the untiled copies of `empty_copy`) still use the pool; an unprewarmed pool is
mapped on demand (logged once per device). Only the allocation moves: bitwise by construction.

Untiled copies (`empty_copy`, iteration 40): the row-major copy of a step-tiled operand that
a hipBLASLt reader needs (the first-call guard's reference, a decline or latch fallback) comes
from the pool when it is active and already mapped on that device (normally the prewarmed spare
block: no new device memory), else from torch.empty (a pool the prewarm skipped is not created
for a transient copy). Out of memory there propagates WITHOUT latching the pool off (`is_oom`),
so the caller falls back for that call only.
"""

import logging
import os
import threading
from typing import Dict, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

_TM_NONEXP: bool = os.environ.get("DLRM_HSTU_FP8_TM_NONEXP", "0") == "1"
# DLRM_HSTU_FP8_TM_NONEXP_XT (see the module docstring): xT from the private pool?
# Unset: the private pool (see the module docstring for why also with WGRAD_CK=1).
_XT_ENV: str = os.environ.get("DLRM_HSTU_FP8_TM_NONEXP_XT", "").strip()
_XT_POOL: bool = (
    _XT_ENV == "1"
    if _XT_ENV in ("0", "1")
    else True
)
# DLRM_HSTU_FP8_TM_NONEXP_TILED (see the module docstring): step-tiled (3-D) operands from
# the private pool? Default "1" (the pool, as before this knob); only "0" moves them out.
_TILED_POOL: bool = os.environ.get("DLRM_HSTU_FP8_TM_NONEXP_TILED", "1").strip() != "0"
_tiled_pool_override: Optional[bool] = None
# Size classes (see the module docstring): round up to _GRANULE, at least _FLOOR.
_GRANULE: int = 512 << 20
_FLOOR: int = 4 * _GRANULE

_CONF_ENV: Optional[str] = None
_CONF: str = ""
# The settings string torch itself parsed (torch 2.12 reads PYTORCH_CUDA_ALLOC_CONF,
# then PYTORCH_HIP_ALLOC_CONF, then PYTORCH_ALLOC_CONF), so restoring it cannot
# swap in another variable's content. Reading it creates only the allocator-config
# singleton (no device init). Fallback: the same environment order.
_getter = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
if _getter is not None:
    try:
        _CONF = _getter() or ""
        _CONF_ENV = "torch._C._accelerator_getAllocatorSettings()"
    except Exception:  # pragma: no cover - defensive
        _CONF = ""
if not _CONF:
    for _name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        if os.environ.get(_name):
            _CONF_ENV, _CONF = _name, os.environ[_name]
            break
_CONF_ITEMS = [s.strip() for s in _CONF.split(",") if s.strip()]
_EXPANDABLE: bool = any(
    s.replace(" ", "").lower() == "expandable_segments:true" for s in _CONF_ITEMS
)
# The same settings string with only expandable_segments switched off; applied
# for the one allocation, then the original string is re-applied in full, so the
# result does not depend on whether a partial string resets other keys.
_CONF_OFF: str = ",".join(
    "expandable_segments:False"
    if s.replace(" ", "").lower() == "expandable_segments:true"
    else s
    for s in _CONF_ITEMS
)

_lock = threading.Lock()
_pools: Dict[int, "torch.cuda.MemPool"] = {}
# Per device: data_ptr of every block the pool handed out, and whether prewarm()
# ran; a block not seen before after prewarm means a segment was hipMalloc'd in
# the timed region (logged; e.g. a 6th concurrent tensor or a request above the
# floor class).
_block_ptrs: Dict[int, set] = {}
_prewarmed: set = set()
_armed_logged: set = set()
_lazy_logged: set = set()  # devices whose unprewarmed pool mapped its first block (logged)
_broken: Optional[str] = None


def _print(message: str) -> None:
    print(f"[hstu-fp8-tm] {message}", flush=True)


def _set_settings(conf: str) -> None:
    setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if setter is None:
        setter = torch.cuda.memory._set_allocator_settings
    setter(conf)


def requested() -> bool:
    return _TM_NONEXP


def active() -> bool:
    """True when empty_tm takes the private non-expandable pool path."""
    return (
        _TM_NONEXP
        and _EXPANDABLE
        and _broken is None
        and hasattr(torch.cuda, "MemPool")
        and hasattr(torch.cuda, "use_mem_pool")
    )


if _TM_NONEXP:
    if not _EXPANDABLE:
        _print(
            "requested: tm_nonexp=1 -> INERT (allocator conf "
            f"{_CONF_ENV}={_CONF!r} has no expandable_segments:True; token-major "
            "operands already come from plain segments)"
        )
    elif not (hasattr(torch.cuda, "MemPool") and hasattr(torch.cuda, "use_mem_pool")):
        _print("requested: tm_nonexp=1 -> INERT (torch.cuda.MemPool unavailable)")
    elif _XT_POOL:
        _print(
            "requested: tm_nonexp=1 (token-major FP8 operands: store_t xT, dual-cast "
            f"dout_t from a private non-expandable pool; conf {_CONF_ENV}={_CONF!r})"
        )
    else:
        _print(
            "requested: tm_nonexp=1 (token-major FP8 operands: dual-cast dout_t from a "
            "private non-expandable pool; LN xT from the main pool, tm_nonexp_xt=0 "
            f"(DLRM_HSTU_FP8_TM_NONEXP_XT=0); "
            f"conf {_CONF_ENV}={_CONF!r})"
        )
if (
    _TM_NONEXP
    and _EXPANDABLE
    and hasattr(torch.cuda, "MemPool")
    and hasattr(torch.cuda, "use_mem_pool")
):
    # DLRM_HSTU_FP8_TM_NONEXP_TILED: where the step-tiled (3-D) operands come from.
    _print(
        "tiled operands: B11 pool"
        if _TILED_POOL
        else "tiled operands: ordinary allocator (DLRM_HSTU_FP8_TM_NONEXP_TILED=0)"
    )


def _pool(device: torch.device) -> "torch.cuda.MemPool":
    index = device.index if device.index is not None else torch.cuda.current_device()
    pool = _pools.get(index)
    if pool is None:
        with torch.cuda.device(index):
            pool = torch.cuda.MemPool()
        _pools[index] = pool
    return pool


def _alloc_block(cap: int, device: torch.device, prewarming: bool = False) -> torch.Tensor:
    """One uint8 block of `cap` bytes from this device's private pool, with the
    allocator switched to expandable_segments:False for this allocation only.

    The switch is process-wide (one allocator config). That is safe while device
    memory is allocated by one thread at a time, as in this trainer: the main
    thread in forward, the autograd device thread in backward (the main thread
    blocks in .backward()); the window-prep and cold-prefetch threads never touch
    the device. A MemPool with its own pluggable allocator would need no switch.
    The first block of a pool that prewarm() did not map (e.g. the prewarm skipped
    under DLRM_HSTU_FP8_TM_NONEXP_TILED=0) is logged once per device.
    """
    with _lock:
        index = device.index if device.index is not None else torch.cuda.current_device()
        pool = _pool(device)
        with torch.cuda.use_mem_pool(pool, device=device):
            try:
                # Inside the try: parseArgs resets unspecified keys before it
                # parses, so the full original string is re-applied even when the
                # OFF string fails.
                _set_settings(_CONF_OFF)
                block = torch.empty(cap, dtype=torch.uint8, device=device)
            finally:
                _set_settings(_CONF)
        seen = _block_ptrs.setdefault(index, set())
        ptr = block.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            if index in _prewarmed:
                _print(
                    f"pool GREW on cuda:{index} after prewarm: new {cap / (1 << 30):.2f} GiB "
                    f"block (#{len(seen)}) mapped in the timed region"
                )
            elif not prewarming and index not in _lazy_logged:
                _lazy_logged.add(index)
                _print(
                    f"pool created lazily on cuda:{index} (not prewarmed): first "
                    f"{cap / (1 << 30):.2f} GiB block mapped on demand"
                )
        return block


def prewarm(nblocks: int = 5, device: Optional[torch.device] = None) -> int:
    """Map `nblocks` floor-class blocks of the private pool now (allocate them
    all, then free them; they stay cached in the pool) and return the bytes.

    Called by dlrm_v4/alloc_pregrow.py before RUN_START, AFTER the main pool took
    "the rest": the pool is mapped outside the timed region from the pre-grow
    margin, and the main pool keeps its full pre-grown size (its peak demand drops
    by the tensors moved here). Touches no data. The production peak is
    three xT (L0, L1, L2) plus one dout_t; the default five blocks leave one
    spare (DLRM_HSTU_FP8_TM_NONEXP_BLOCKS), so no block is mapped in the timed
    region. A sixth concurrent tensor would still be mapped on demand.
    """
    global _broken
    if not active() or nblocks <= 0:
        return 0
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    try:
        blocks = [_alloc_block(_FLOOR, device, prewarming=True) for _ in range(nblocks)]
        total = sum(b.numel() for b in blocks)
        del blocks
        torch.cuda.synchronize(device)
    except Exception as error:  # pragma: no cover - defensive fallback
        _broken = f"{type(error).__name__}: {error}"
        logger.warning("[hstu-fp8-tm] tm_nonexp DISABLED at prewarm: %s", _broken)
        _print(f"tm_nonexp DISABLED at prewarm (plain torch.empty from now on): {_broken}")
        return 0
    _prewarmed.add(device.index if device.index is not None else torch.cuda.current_device())
    _print(
        f"prewarm before run_start on {device}: {nblocks} x "
        f"{_FLOOR / (1 << 30):.2f} GiB blocks mapped in the private pool"
    )
    return total


def _pooled(
    shape: Sequence[int], dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, int]:
    """(a view of a size-class block of the private pool with this shape, the class
    bytes). Raises on failure (the callers decide whether that latches)."""
    numel = 1
    for dim in shape:
        numel *= int(dim)
    nbytes = numel * dtype.itemsize
    cap = max(_FLOOR, -(-nbytes // _GRANULE) * _GRANULE)
    block = _alloc_block(cap, device)
    # A view of the block's prefix: contiguous, storage offset 0, the
    # requested shape; the block returns to the pool whole when the view
    # dies.
    return block[:nbytes].view(dtype).view(tuple(int(d) for d in shape)), cap


def _latch(error: BaseException) -> None:
    global _broken
    _broken = f"{type(error).__name__}: {error}"
    logger.warning("[hstu-fp8-tm] tm_nonexp DISABLED: %s", _broken)
    _print(f"tm_nonexp DISABLED (plain torch.empty from now on): {_broken}")


def _mapped(device: torch.device) -> bool:
    """This device's private pool holds mapped blocks (prewarmed or allocated)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return index in _prewarmed or bool(_block_ptrs.get(index))


def is_oom(error: BaseException) -> bool:
    """A device out-of-memory error: torch.OutOfMemoryError, or a RuntimeError whose
    text says "out of memory" (HIP / extension paths that do not raise the subclass)."""
    oom_type = getattr(torch, "OutOfMemoryError", None) or getattr(
        torch.cuda, "OutOfMemoryError", None
    )
    if oom_type is not None and isinstance(error, oom_type):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def empty_copy(
    shape: Sequence[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """A transient row-major copy of a step-tiled operand for a hipBLASLt reader (the
    first-call guard's reference, a decline / latch fallback; wgrad_ck and
    wgrad_ck_w3 `untile_to_row_major`): from this device's private pool when it is
    active and already mapped (prewarmed, or holding blocks of earlier operands;
    normally the prewarmed spare block, so no new device memory), else a plain
    torch.empty (a pool the prewarm skipped, DLRM_HSTU_FP8_TM_NONEXP_TILED=0, is not
    created for a transient copy). Out of memory propagates (`is_oom`) and does NOT
    latch the pool off: the caller falls back for that call only. Any other pool
    failure latches as in `empty_tm` and returns a plain torch.empty."""
    device = torch.device(device)
    if (
        device.type != "cuda"
        or not active()
        or any(int(d) == 0 for d in shape)
        or not _mapped(device)
    ):
        return torch.empty(shape, dtype=dtype, device=device)
    try:
        return _pooled(shape, dtype, device)[0]
    except Exception as error:
        if is_oom(error):
            raise
        _latch(error)  # pragma: no cover - defensive fallback
    return torch.empty(shape, dtype=dtype, device=device)  # pragma: no cover


def tiled_pool() -> bool:
    """DLRM_HSTU_FP8_TM_NONEXP_TILED (or the test-only override): 3-D (step-tiled)
    requests come from the private pool."""
    return _TILED_POOL if _tiled_pool_override is None else _tiled_pool_override


def set_tiled_pool_for_testing(value: Optional[bool]) -> None:
    """Test-only: override the import-time DLRM_HSTU_FP8_TM_NONEXP_TILED (None restores it)."""
    global _tiled_pool_override
    _tiled_pool_override = value


def pooled_all_tiled() -> bool:
    """DLRM_HSTU_FP8_TM_NONEXP_TILED=0 and every operand the pool would serve is
    requested step-tiled, so the pool serves none in steady state and alloc_pregrow
    skips the prewarm: the LN xT (pooled unless DLRM_HSTU_FP8_TM_NONEXP_XT=0) is
    tiled with DLRM_HSTU_FP8_BWD_WGRAD_CK=1 + _CK_XT_TILED=1, the dual-cast dout_t
    with _CK_W3=1 + _CK_W3_TILED=1 (the knobs as their modules resolved them)."""
    if tiled_pool():
        return False
    from generative_recommenders.ops.triton import wgrad_ck, wgrad_ck_w3

    xt_tiled = wgrad_ck.requested() and wgrad_ck.tiled_requested()
    return (xt_tiled or not _XT_POOL) and wgrad_ck_w3.tiled_requested()


def empty_tm(
    shape: Sequence[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """torch.empty(shape, dtype, device) for a token-major FP8 GEMM operand.

    With DLRM_HSTU_FP8_TM_NONEXP=1 under expandable segments: from this device's
    private non-expandable pool (see the module docstring). Otherwise, and on
    any failure (latched), a plain torch.empty. A 3-D (step-tiled) request with
    DLRM_HSTU_FP8_TM_NONEXP_TILED=0: a plain torch.empty.
    """
    device = torch.device(device)
    if (
        device.type != "cuda"
        or not active()
        or any(int(d) == 0 for d in shape)
        or (len(shape) == 3 and not tiled_pool())
    ):
        return torch.empty(shape, dtype=dtype, device=device)
    try:
        out, cap = _pooled(shape, dtype, device)
    except Exception as error:  # pragma: no cover - defensive fallback
        _latch(error)
        return torch.empty(shape, dtype=dtype, device=device)
    key = str(device)
    if key not in _armed_logged:
        _armed_logged.add(key)
        _print(
            f"ARMED tm_nonexp on {key}: token-major operand {list(shape)} {dtype} "
            f"from the private non-expandable pool (size class {cap / (1 << 30):.2f} GiB)"
        )
    return out


def xt_in_pool() -> bool:
    """DLRM_HSTU_FP8_TM_NONEXP_XT: `empty_xt` uses the private pool (when active)."""
    return _XT_POOL


def default_blocks() -> int:
    """prewarm() blocks when DLRM_HSTU_FP8_TM_NONEXP_BLOCKS is unset: three xT + one
    dout_t + one spare (5), or one dout_t + one spare (2) with xT out of the pool."""
    return 5 if _XT_POOL else 2


def empty_xt(
    shape: Sequence[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """The LN token-major xT (uvqk / vk fp8t wgrad operand): `empty_tm`, or a plain
    torch.empty when DLRM_HSTU_FP8_TM_NONEXP_XT is 0 (default: 1, the private pool)."""
    if _XT_POOL:
        return empty_tm(shape, dtype, device)
    return torch.empty(shape, dtype=dtype, device=device)


def reset_for_testing() -> None:
    global _broken
    _broken = None
    _armed_logged.clear()
    _lazy_logged.clear()
