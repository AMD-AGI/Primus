"""GPU offsets for the existing EC caller's already-valid integer row IDs.

The caller must establish 0 <= ID < num_rows (the EC forward gather does).
This is deliberately not a general invalid-input-compatible bincount API.
The incumbent argsort permutation, including duplicate tie order, is retained
unchanged (``argsort_ids`` may compute it on int32 keys: same permutation).
"""
import logging
import os

import torch
import triton
import triton.language as tl

logger: logging.Logger = logging.getLogger(__name__)


def _env_on(name: str) -> bool:
    """Default-off gate: on only for an explicit 1/true/on/yes (gin binding)."""
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "on", "yes")


# EC bundle (b) SORT32 -- int32-key segment argsort. Default OFF in code; the
# gin binding apply_env_bootstrap.DLRM_HSTU_FP8_EC_SORT32 turns it on, and
# DLRM_HSTU_FP8_EC_SORT32=0 (kill switch, wins over gin) restores
# torch.argsort(idx) on the int64 ids.
# torch.argsort of the int64 EC row ids runs a rocprim radix sort over 64-bit
# keys (8 onesweep digit passes; ship10 1n trace, one grouped EC backward:
# ~0.75 ms at T=4.74M and ~0.53 ms at T=2.37M) although every id is < num_rows.
# Above _SMALL_SORT_MAX elements torch sorts a 1-D CUDA tensor with its
# STABLE radix sort whatever ``stable`` says (ATen cuda/Sort.cpp:
# should_use_small_sort is size <= 4096, larger sorts take
# launch_stable_sort_kernel), so the incumbent permutation is the stable one:
# equal ids keep ascending token order. The stable sorting permutation is
# unique and int64 -> int32 is order-preserving on [0, 2^31), so a stable
# argsort of the same ids narrowed to int32 returns that identical
# permutation with half the key passes and key bytes. Bitwise: same order ->
# same offsets -> same fp32 add order. Below the cutoff the incumbent call is
# kept as is.
_SORT32: bool = _env_on("DLRM_HSTU_FP8_EC_SORT32")
_SMALL_SORT_MAX: int = 4096  # torch's small-sort cutoff; incumbent call below it
_SORT32_LOGGED: bool = False


def argsort_ids(idx, num_rows):
    """The incumbent ``torch.argsort(idx)`` permutation of valid ids in [0, num_rows)."""
    global _SORT32_LOGGED
    if (
        _SORT32
        and idx.is_cuda
        and idx.dim() == 1
        and idx.dtype == torch.int64
        and idx.numel() > _SMALL_SORT_MAX
        and num_rows <= 2**31 - 1
    ):
        if not _SORT32_LOGGED:
            _SORT32_LOGGED = True
            logger.info(
                "DLRMV4 armed OK: ec sort32 -- EC segment argsort on int32 "
                "keys, stable, same permutation as the int64 argsort "
                "(T=%d, U=%d; kill switch DLRM_HSTU_FP8_EC_SORT32=0)",
                idx.numel(), num_rows,
            )
        return torch.argsort(idx.to(torch.int32), stable=True)
    return torch.argsort(idx)


@triton.jit
def _search(IDX, ORDER, query, active, T, STRIDE):
    left = tl.full(query.shape, 0, tl.int64)
    right = tl.full(query.shape, T, tl.int64)
    remaining = T.to(tl.int64)
    while remaining > 0:
        live = active & (left < right)
        middle = left + ((right - left) >> 1)
        source = tl.load(ORDER + middle, live, other=0).to(tl.int64)
        value = tl.load(IDX + source * STRIDE.to(tl.int64), live, other=0).to(tl.int64)
        advance = live & (value < query)
        left = tl.where(advance, middle + 1, left)
        right = tl.where(live & ~advance, middle, right)
        remaining = remaining >> 1
    return left


@triton.jit(do_not_specialize=["T", "U", "STRIDE"])
def _offsets(IDX, ORDER, OUT, T, U, STRIDE, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    tokens = T.to(tl.int64)
    rows = U.to(tl.int64)
    location = _search(IDX, ORDER, row, (row > 0) & (row < rows), tokens, STRIDE)
    # The valid-ID precondition proves both endpoints, including T=U=0.
    location = tl.where(row == rows, tokens, location)
    tl.store(OUT + row, location, row <= rows)


def _metadata(reverse_indices, num_rows):
    if not isinstance(num_rows, int) or isinstance(num_rows, bool) or num_rows < 0:
        raise ValueError("num_rows must be a nonnegative Python integer")
    if reverse_indices.device.type != 'cuda' or reverse_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("valid EC IDs must be CUDA int32 or int64")
    idx = reverse_indices.reshape(-1)
    if num_rows == 0 and idx.numel():
        raise ValueError("nonempty IDs cannot be valid with num_rows=0")
    return idx


def offsets_from_order_valid_ids(reverse_indices, order, num_rows, out=None):
    """Preserve a supplied incumbent permutation; optional out is for redzones."""
    idx = _metadata(reverse_indices, num_rows)
    if order.dtype != torch.int64 or order.device != idx.device or order.dim() != 1 or not order.is_contiguous() or order.numel() != idx.numel():
        raise ValueError("order must be the contiguous int64 argsort of the flattened IDs")
    if out is None:
        out = torch.empty(num_rows + 1, dtype=torch.int64, device=idx.device)
    elif out.dtype != torch.int64 or out.device != idx.device or out.shape != (num_rows + 1,) or not out.is_contiguous():
        raise ValueError("offsets output must be contiguous int64 of length num_rows+1")
    _offsets[(triton.cdiv(num_rows + 1, 128),)](
        idx, order, out, idx.numel(), num_rows, idx.stride(0), 128, num_warps=2,
    )
    return out


def build_segments_valid_ids(reverse_indices, num_rows, order=None):
    """Return exact incumbent order and fixed-size offsets for valid EC IDs."""
    idx = _metadata(reverse_indices, num_rows)
    if order is None:
        order = argsort_ids(idx, num_rows)
    return order, offsets_from_order_valid_ids(idx, order, num_rows)


# EC JOINT_SORT -- ONE stable argsort for the two hoisted members of the
# grouped EC backward (ec_bf16_output._ECExpandGroup, DLRM_HSTU_FP8_EC_GROUP_BWD)
# instead of one per member. Default OFF in code (_env_on above: on only for
# an explicit 1/true/on/yes); the gin binding
# apply_env_bootstrap.DLRM_HSTU_FP8_EC_JOINT_SORT turns it on and
# DLRM_HSTU_FP8_EC_JOINT_SORT=0 is the kill switch (each member then keeps
# build_segments_valid_ids above). Reducer-agnostic: it only supplies each
# member's (order, offsets), which ec_gpu_plan.segment_reduce_rows hands to
# whichever reducer takes the call (DLRM_HSTU_FP8_EC_REDUCE_LPT's single
# launch, or the cold/hot two-launch reducer). Keys: member 0's ids, then
# member 1's ids + num_rows, in one int32 array. With valid ids (0 <= id <
# num_rows, the precondition of this whole module) every member-0 key is below
# every member-1 key, and a stable sort orders by (key, position), so the
# joint stable permutation is [stable_argsort(ids0), T0 + stable_argsort(ids1)]
# (the +num_rows key shift and the +T0 position shift preserve member 1's
# order). Above _SMALL_SORT_MAX elements each member's incumbent permutation IS
# the stable one (SORT32 note above), so both members get exactly their
# incumbent order, hence the same offsets and the same fp32 add order
# (bitwise). It is the SORT32 argsort shared by both members, so it also
# requires SORT32 (DLRM_HSTU_FP8_EC_SORT32=0 leaves no int32 sort anywhere).
# Saves one radix sort's fixed cost: its histogram launches, 9 lookback-state
# fills, the key cast / iota / copy launches and the ramp/tail of 4 onesweep
# passes (ship10 TB1: member 1's sort alone 0.24 ms for 2.37M keys). Adds one
# [T1] int64 subtract (member 1's positions) and a +num_rows key add.
_JOINT_SORT: bool = _env_on("DLRM_HSTU_FP8_EC_JOINT_SORT")
_JOINT_LOGGED: bool = False


def joint_sort_engages(ids, num_rows) -> bool:
    """Whether build_segments_joint_valid_ids applies to these two id tensors."""
    if not (_JOINT_SORT and _SORT32) or len(ids) != 2:
        return False
    if not isinstance(num_rows, int) or isinstance(num_rows, bool):
        return False
    a, b = ids
    return (
        0 < num_rows
        and 2 * num_rows <= 2**31 - 1
        and all(
            t.is_cuda and t.dtype in (torch.int32, torch.int64)
            and t.numel() > _SMALL_SORT_MAX
            for t in (a, b)
        )
        and a.device == b.device
        and a.numel() + b.numel() <= 2**31 - 1
    )


# EC JOINT_ONESWEEP -- yambda_5b.gin item (49)(c). Default OFF in code (_env_on: on only for an explicit
# 1/true/on/yes); the gin binding apply_env_bootstrap.DLRM_HSTU_FP8_EC_JOINT_ONESWEEP turns it on and
# DLRM_HSTU_FP8_EC_JOINT_ONESWEEP=0 is the kill switch (the JOINT_SORT chain below). Refines JOINT_SORT and is
# inert without it. The same joint stable sort of the same keys ([ids_a ; ids_b + num_rows], all below
# 2*num_rows by the valid-id precondition) runs on ops/onesweep_sort.py: rocPRIM's onesweep device code over
# bits [0, bit_length(2*num_rows - 1)) in 9-bit digit passes (2 passes for num_rows <= 2^17 instead of 4),
# keys read straight from the two int64 id tensors (no int32 key build), member-relative positions as values
# (no "joint[ta:] - ta" kernel, no iota), one fill instead of 9, and both members' offsets from the sorted keys
# (one streaming kernel over the run boundaries + one kernel for long runs of absent keys) instead of the two
# _offsets binary searches (integer counts: the same values).
# Bitwise: an LSD radix sort is stable, the stable sorting permutation is unique, so each member gets exactly
# its incumbent order and offsets (int64, contiguous, 16-byte aligned like the incumbent's tensors). The first
# _ONESWEEP_CHECKS builds of the process also run the incumbent chain and compare raw bits; a mismatch, a
# raising call or an unavailable extension returns the incumbent result and turns this off for the process
# with one log line. int64 ids only (int32 ids keep the incumbent chain).
_JOINT_ONESWEEP: bool = _env_on("DLRM_HSTU_FP8_EC_JOINT_ONESWEEP")
_ONESWEEP_CHECKS: int = 2
_ONESWEEP_LOGGED: bool = False


def _raw_eq(x, y) -> bool:
    return (
        x.dtype == y.dtype == torch.int64
        and x.shape == y.shape
        and x.is_contiguous()
        and x.data_ptr() % 16 == y.data_ptr() % 16 == 0
        and torch.equal(x, y)
    )


def _joint_onesweep(a, b, num_rows):
    """Item (49)(c): the incumbent [(order, offsets)] x 2 on the onesweep extension, or None (incumbent chain)."""
    global _JOINT_ONESWEEP, _ONESWEEP_CHECKS, _ONESWEEP_LOGGED
    try:
        from generative_recommenders.ops import onesweep_sort as osort

        segs = osort.ec_joint_segments(a, b, num_rows)
        if segs is None and osort._FAILED is not None:
            raise RuntimeError("onesweep sort extension unavailable: " + osort._FAILED)
    except Exception as exc:  # fail closed: incumbent chain from now on
        _JOINT_ONESWEEP = False
        logger.warning(
            "DLRMV4 ec joint onesweep FAILED CLOSED -> joint torch.argsort for the rest of the run (%s: %s)",
            type(exc).__name__, str(exc)[:300],
        )
        return None
    if segs is None:
        return None
    if _ONESWEEP_CHECKS > 0:
        ref = _joint_incumbent(a, b, num_rows)
        bad = [
            f"member {m} {what}"
            for m, (got_m, ref_m) in enumerate(zip(segs, ref))
            for what, x, y in (("order", got_m[0], ref_m[0]), ("offsets", got_m[1], ref_m[1]))
            if not _raw_eq(x, y)
        ]
        if bad:
            _JOINT_ONESWEEP = False
            logger.error(
                "DLRMV4 ec joint onesweep SELF-CHECK FAILED (T=%d+%d, U=%d): %s differ -- incumbent result used, "
                "joint torch.argsort for the rest of the run", a.numel(), b.numel(), num_rows, ", ".join(bad),
            )
            return ref
        _ONESWEEP_CHECKS -= 1
    if not _ONESWEEP_LOGGED:
        _ONESWEEP_LOGGED = True
        logger.warning(
            "DLRMV4 armed OK: ec joint onesweep -- joint EC segments on the key-bound onesweep sort (9-bit digits, "
            "%d key bits, fused key read, offsets from sorted keys; T=%d+%d, U=%d; first builds == incumbent; "
            "kill switch DLRM_HSTU_FP8_EC_JOINT_ONESWEEP=0)",
            int(2 * num_rows - 1).bit_length(), a.numel(), b.numel(), num_rows,
        )
    return segs


def build_segments_joint_valid_ids(ids, num_rows):
    """[(order, offsets)] of two members == [build_segments_valid_ids(i, num_rows)].

    Caller checks joint_sort_engages(ids, num_rows) first.
    """
    global _JOINT_LOGGED
    a = _metadata(ids[0], num_rows)
    b = _metadata(ids[1], num_rows)
    if not _JOINT_LOGGED:
        _JOINT_LOGGED = True
        logger.warning(
            "DLRMV4 armed OK: ec joint sort -- one stable int32 argsort for the "
            "two hoisted EC members (T=%d+%d, U=%d; same per-member permutation "
            "as two argsorts; kill switch DLRM_HSTU_FP8_EC_JOINT_SORT=0)",
            a.numel(), b.numel(), num_rows,
        )
    if _JOINT_ONESWEEP and a.dtype == torch.int64 and b.dtype == torch.int64:
        segs = _joint_onesweep(a, b, num_rows)
        if segs is not None:
            return segs
    return _joint_incumbent(a, b, num_rows)


def _joint_incumbent(a, b, num_rows):
    """The JOINT_SORT chain: int32 key build, one stable torch.argsort, two _offsets searches."""
    ta, tb = a.numel(), b.numel()
    keys = torch.empty(ta + tb, dtype=torch.int32, device=a.device)
    keys[:ta].copy_(a)
    torch.add(b, num_rows, out=keys[ta:])
    joint = torch.argsort(keys, stable=True)
    del keys
    order_a = joint[:ta]
    order_b = joint[ta:] - ta
    return [
        (order_a, offsets_from_order_valid_ids(a, order_a, num_rows)),
        (order_b, offsets_from_order_valid_ids(b, order_b, num_rows)),
    ]
