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

# pyre-unsafe

"""LN*U*dropout keep-mask NIB layout (r16 "NIB", gin item (94), knob DLRM_HSTU_FP8_LNMD_MASK_NIB, default OFF).

The saved LN*U*dropout keep mask of the two full-length STU layers is int8 [N, 512] today: column c's byte holds the
keep bits of the three dropout planes (bit 0 y, bit 1 x, bit 2 u); bits 3-7 are always 0 (the writers store
c32 = (pk >> base) & 7, MGPK masks its dwords with 0x07070707). NIB stores the same bits as uint8 [N, 256]
("layout A"): the lane l that owns columns 8l..8l+7 (the writers and the L11 / DO8 reader use the same lane map)
keeps ONE dword at byte 4l of the row, whose byte k (k = 0..3) holds column 8l+k's keep bits in bits 0-2 and column
8l+4+k's in bits 4-6; bits 3 and 7 are 0. Same decisions, same Philox counters, half the bytes.

This module is the shared contract of the NIB writers (ops/triton/triton_hstu_linear.py, module end), the DO8 NIB
reader build (ops/hip/l11_do8.py + l11_do8.cu under -DL11_MKB_NIB=1) and L11's dispatch gate
(ops/hip/l11_outproj_lnmd.py). It imports torch only (no Triton, no HIP): every helper is host-side metadata or an
exact torch reference. (The unpack line's printer, ops.utils.arm_print_all_ranks, is imported lazily at the first
unpack.)

Fail-closed rule: a NIB mask reaches ONLY the DO8 NIB build. Every other consumer of the keep mask (L11's own E5 /
bf16-du launches, B1a v3, the Triton v2 / v1 backward) receives `as_int8(mask, where)`: the exact unpack (counted;
one line per (where) per process). With the knob off no NIB mask is ever written, so every helper is a no-op.
"""

import os
import threading
from typing import Dict, Optional

import torch

KNOB: str = "DLRM_HSTU_FP8_LNMD_MASK_NIB"
FAMILY: str = "LN*U*dropout keep mask NIB"
D: int = 512  # columns of the int8 keep mask (the LN*U*dropout width)
ROW_BYTES: int = 256  # bytes per row of the NIB mask (D // 2)
_INT8_ATTR: str = "_lnmd_mask_nib_int8"  # cached exact unpack on a NIB tensor (fallback consumers only)

_lock = threading.Lock()
_COUNTS: Dict[str, int] = {"unpack": 0}
_printed: set = set()


def requested() -> bool:
    """The knob, read per call (apply_env_bootstrap exports the gin binding; a pre-set env value wins): exactly "1"."""
    return os.environ.get(KNOB, "").strip() == "1"


def is_nib(t: Optional[torch.Tensor]) -> bool:
    """True for a NIB-layout keep mask (uint8 [N, 256]); False for the int8 [N, 512] mask, None and placeholders."""
    return (
        isinstance(t, torch.Tensor)
        and t.dtype == torch.uint8
        and t.dim() == 2
        and int(t.shape[1]) == ROW_BYTES
    )


def layout_ok(t: Optional[torch.Tensor], m: int) -> bool:
    """A NIB mask the DO8 NIB reader can take for m rows: uint8 [m, 256], row stride 256, 16-byte aligned base."""
    return (
        is_nib(t)
        and int(t.shape[0]) == int(m)
        and tuple(t.stride()) == (ROW_BYTES, 1)
        and t.data_ptr() % 16 == 0
    )


def pack_ref(mask: torch.Tensor) -> torch.Tensor:
    """Exact torch reference: int8 [N, 512] keep mask (bytes 0..7) -> NIB uint8 [N, 256] (layout A above).
    The inverse of `unpack` on every mask whose bytes are <= 7 (every mask the writers produce)."""
    n = int(mask.shape[0])
    v = mask.reshape(n, D // 8, 8).view(torch.uint8)
    return ((v[:, :, 0:4] & 7) | ((v[:, :, 4:8] & 7) << 4)).reshape(n, ROW_BYTES).contiguous()


def unpack(packed: torch.Tensor) -> torch.Tensor:
    """Exact torch unpack: NIB uint8 [N, 256] -> int8 [N, 512] (fresh contiguous tensor, row stride 512).
    Both nibble planes are written straight into the strided halves of the preallocated result (no temporaries:
    the peak is the int8 result plus the NIB input; r16 iteration 54 review R1-F6). Column 8l+k = byte (4l+k) & 7,
    column 8l+4+k = (byte (4l+k) >> 4) & 7, on every byte value."""
    n = int(packed.shape[0])
    p = packed.reshape(n, D // 8, 4)
    out = torch.empty((n, D), dtype=torch.uint8, device=packed.device)
    o = out.view(n, D // 8, 8)
    torch.bitwise_and(p, 7, out=o[:, :, 0:4])
    hi = o[:, :, 4:8]
    torch.bitwise_right_shift(p, 4, out=hi)
    hi.bitwise_and_(7)
    return out.view(torch.int8)


def counters() -> Dict[str, int]:
    with _lock:
        return dict(_COUNTS)


def reset_for_testing() -> None:
    with _lock:
        for k in _COUNTS:
            _COUNTS[k] = 0
        _printed.clear()


def as_int8(mask: Optional[torch.Tensor], where: str) -> Optional[torch.Tensor]:
    """The int8 [N, 512] keep mask for a consumer that cannot read NIB. Non-NIB input (None, int8, placeholders) is
    returned unchanged (no work, no count). A NIB mask is unpacked exactly ONCE per tensor (cached on it, so every
    fallback consumer of one backward shares the unpack; it dies with the saved mask) and counted; the first unpack
    per `where` prints one line per process (steady-state training should print none: the in-model census treats
    unpack > 0 as VOID)."""
    if not is_nib(mask):
        return mask
    cached = getattr(mask, _INT8_ATTR, None)
    if cached is not None:
        return cached
    out = unpack(mask)
    try:
        setattr(mask, _INT8_ATTR, out)
    except Exception:  # noqa: BLE001 -- caching is an optimization only
        pass
    with _lock:
        _COUNTS["unpack"] += 1
        count = _COUNTS["unpack"]
        first = where not in _printed
        _printed.add(where)
    if first:
        line = (
            f"[lnmd-mask-nib] {FAMILY}: exact unpack to int8 for {where} (a non-DO8-NIB consumer read a NIB mask; "
            f"counted, unpack={count})"
        )
        # Every rank, tagged [rank N], one atomic write (ops.utils.arm_print_all_ranks: a plain per-rank print can be
        # lost to the launcher's chatter filter, and this line is a must-be-absent marker). Imported lazily, so this
        # module still imports torch only; if that import fails, a plain print keeps the line.
        try:
            from generative_recommenders.ops.utils import arm_print_all_ranks
        except Exception:  # noqa: BLE001
            print(f"{line} [rank {os.environ.get('RANK', '?')}]", flush=True)
        else:
            arm_print_all_ranks(line)
    return out
