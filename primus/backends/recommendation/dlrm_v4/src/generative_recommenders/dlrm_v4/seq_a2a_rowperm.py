"""Sequence-embedding all-to-all permutes at row granularity, in the wire dtype.

TorchRec's ``All2All_Seq_Req`` reorders the looked-up sequence embeddings
from (source rank, feature) order into (feature, source rank) order before
the forward all-to-all, and undoes that reordering after the backward
all-to-all. It does this with ``fbgemm.permute_2D_sparse_data`` on the fp32
values, with lengths ``[local_T * world_size, B] * D``. On yambda-5b at 4 nodes
those lengths have 458,752 segments for about 137.7k rows, which is fewer than
one embedding row per segment. ``permute_2D_data_kernel_vec<false,long,float>``
spends most of its time on per-segment overhead: 2 x 655 us/step at about
0.66 TB/s in the ship4 4-node trace (gap item N3).

Each row of the 2-D lengths covers one contiguous block of whole embedding
rows (all B samples of one feature from one source rank). This module
therefore:

* builds a row index ``fidx`` with ``permute_2D_sparse_data`` on per-block row
  counts ``[local_T * world_size, 1]`` and ``values = arange(rows)``. That is a
  few hundred segments over a small int64 array, instead of 458k segments over
  ``rows * D`` floats;
* forward: ``wire = encode(x)[fidx]``. The codec's encode is elementwise, so
  encoding before the permutation instead of after it gives bit-identical
  results, and the gather runs on the 2-byte wire dtype;
* backward: ``bidx`` comes from ``backward_recat_tensor`` and the permuted
  per-block counts in the same way. Then ``grad = decode(wire[bidx])`` and
  ``div_`` as in stock.

Every output element is copied from exactly one input element. The outputs
are bit-identical to stock, including NaN payloads, -0.0 and the fp16 clamp.
``a2ai.permuted_lengths_after_sparse_data_all2all`` keeps its stock value: the
D-scaled ``[local_T * world_size, B]`` lengths, permuted by rows. Code in the
backward pass and elsewhere reads it.

The stock path still runs, with the original methods, in these cases:
variable batch size, ``local_T == 0``, ``forward_recat_tensor is None``,
``numel % D != 0``, a non-contiguous input, or a codec that is not an
elementwise cast. Elementwise casts are fbgemm ``QuantizedCommCodec`` with
FP32/FP16/BF16 precision (optionally loss-scaled) and torchrec
``NoOpQuantizedCommCodec``. FP8/INT8/MX4 are row-wise and are never re-ordered.

Trace attribution: the index build plus the gather run under the stock
``## All2All_Seq_fwd_permute ##`` / ``## All2All_Seq_bwd_permute ##``
record_function labels. Keep those labels so traces stay attributable.

Env (the ``DLRM_HSTU_FP8_`` prefix is forwarded by ``launch_slurm.sh``):
  DLRM_HSTU_FP8_SEQ_A2A_ROWPERM=1          enable (absent = off in code; the
                                           yambda_5b.gin apply_env_bootstrap
                                           binding turns it on; =0 opts out).
                                           Read once, at install() time.
  DLRM_HSTU_FP8_SEQ_A2A_ROWPERM_CHECK=N    first N forward calls check
                                           sum(block rows) == rows (one host
                                           sync each, default 1); on failure the
                                           module falls back to stock permanently
  DLRM_HSTU_FP8_SEQ_A2A_ROWPERM_VERIFY=N   first N forward/backward calls also
                                           run the stock math and compare bits
                                           (default 0; costs an extra fp32 copy).
                                           On a mismatch they use the stock
                                           result and disable the module.

Fused send path (``seq_a2a_fused.py``, Triton; needs the rowperm switch above):
the forward's row index, fp16 encode, clamp and row gather become ONE kernel
(K1, which also writes the permuted D-scaled lengths), and the backward's row
index, row gather, fp16 decode and ``GRADIENT_DIVISION`` scale become ONE
kernel (K2). The prologue is the existing ``aten::sum`` of the block row counts
(``blk``); K2 reuses the forward's ``blk``, so the backward has no index launch.
Outputs are bit-identical to stock (wire, permuted lengths, gradient); the
argument is in ``seq_a2a_fused.py``. It engages only when the rowperm path is
eligible AND both codecs are exactly fbgemm ``QuantizedCommCodec`` FP16 with no
loss scale, the encode is fbgemm's ``fp32_to_fp16_with_clamp`` or the
``train/utils.py`` lowmem cast-then-``clamp_`` patch (same bits), the decode is
fbgemm's ``fp16_to_fp32``, the input is fp32, ``D % 8 == 0``, the world size is
a power of two, n_blocks <= ``seq_a2a_fused.MAX_BLOCKS`` and Triton can launch.
Otherwise the rowperm path above runs unchanged.
  DLRM_HSTU_FP8_SEQ_A2A_FUSED=1            master switch (absent = off in code;
                                           the yambda_5b.gin binding is True,
                                           which _env_bootstrap turns into 1
                                           at every node count, item (64)(a);
                                           at one node the table-/column-wise
                                           sets run it; env wins over gin:
                                           =1 on, =0 off). Read once.
  DLRM_HSTU_FP8_SEQ_A2A_FUSED_SEND=0       opt out of the send part (K1 + K2);
                                           default on with the master switch.
  DLRM_HSTU_FP8_SEQ_A2A_FUSED_FWD=0 / _BWD=0
                                           per-direction opt out (attribution);
                                           default on with the send switch.
  DLRM_HSTU_FP8_SEQ_A2A_FUSED_VERIFY=N     first N fused forward/backward calls
                                           compare integer views against the
                                           stock math (default: the ROWPERM_VERIFY
                                           value, i.e. 0; a host sync per call).
                                           On a mismatch they use the stock
                                           result and turn the fused path off
                                           (the rowperm path continues).
Arming: "DLRMV4 armed OK: seq a2a fused send (fwd cast+clamp+gather, bwd
gather+decode+div)", logged once at the first fused call (after it succeeded).
Hardening: an exception from a fused send kernel (e.g. a Triton compile or
launch failure) is caught; that call runs the rowperm path, one warning is
logged and the fused send path stays off for the rest of the run. The static
part of the eligibility (Triton, device type, D, world size and the exact-codec
check) is cached per long-lived codecs object; ``All2AllSequenceInfo`` itself is
rebuilt by ``alltoall_sequence`` on every call, so it cannot be the key. The
per-call checks (dtypes, contiguity, shapes) are host-side attribute reads.

Fused receive path (receive tier 1; ``seq_a2a_fused.py`` K3'/K4): this module
also patches torchrec ``All2All_Seq_Req_Wait.forward/backward`` (only when the
master switch is on). Forward: wait, then ONE kernel decodes the fp16 wire and
applies the row-wise unbucketize permutation (K3'), instead of the codec decode
plus the later ``index_select`` of ``unbucketize_gather._unbucketize``.
Backward: ONE kernel scatters the gradient rows through the permutation and
encodes them (K4), instead of the ``_PermuteRows`` backward plus the codec
encode; then the stock backward all-to-all (same record_function label, same
stream, same request bookkeeping). The permutation reaches the wait through the
``unbucketize_gather`` hand-off (``recv_handoff``): only for a
``comm_ops.Request`` whose ``wait_function`` is the patched
``All2All_Seq_Req_Wait`` (never a ``NoWait``), and only while the fused receive
path is on. The patched wait applies a handed-over permutation on EVERY branch
(fused, stock fallback, VERIFY mismatch, no-grad/eval, disabled mid-run) and
marks it consumed; the awaitable raises if it was not. It reproduces
``_unbucketize``'s decision exactly (same checks, same order, the same
first-calls bijectivity budget): K3'/K4 only where ``_unbucketize`` would use
``_PermuteRows`` (int64 permutation, autograd recording the call, bijectivity
check passed, module not disabled); otherwise the stock decode +
``torch.index_select``, whose backward (``zeros + index_add_``, which turns -0.0
into +0.0) is reproduced op for op. Without a permutation (table-/column-wise
sets) K3' is the plain decode and K4 the plain encode (one pass instead of
``.half()`` + ``clamp_``). Engages only with exact fbgemm FP16 codecs both ways
(the send path's codec check), D % 8 == 0, an fp16 contiguous wire and a
contiguous fp32 gradient; otherwise the stock ops run for that call.
  DLRM_HSTU_FP8_SEQ_A2A_FUSED_RECV=0       opt out of the receive part (K3' + K4);
                                           default on with the master switch.
                                           Read once, at install() time.
  DLRM_HSTU_FP8_SEQ_A2A_FUSED_VERIFY=N     also covers the first N fused receive
                                           forward/backward calls: compared
                                           against the stock decode +
                                           index_select / _PermuteRows backward +
                                           encode; on a mismatch the stock result
                                           is used and the fused receive path is
                                           turned off.
Arming: "DLRMV4 armed OK: seq a2a fused recv (fwd decode+unbucketize gather, bwd
unbucketize scatter+encode)", logged once per process at the first fused call.
Fallback: "DLRMV4 seq a2a fused recv: ... DISABLED ..." (WARNING) on a kernel
exception or a VERIFY mismatch.
"""

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_ENV = "DLRM_HSTU_FP8_SEQ_A2A_ROWPERM"
_CHECK_ENV = "DLRM_HSTU_FP8_SEQ_A2A_ROWPERM_CHECK"
_VERIFY_ENV = "DLRM_HSTU_FP8_SEQ_A2A_ROWPERM_VERIFY"
_FUSED_ENV = "DLRM_HSTU_FP8_SEQ_A2A_FUSED"
_FUSED_SEND_ENV = "DLRM_HSTU_FP8_SEQ_A2A_FUSED_SEND"
_FUSED_FWD_ENV = "DLRM_HSTU_FP8_SEQ_A2A_FUSED_FWD"
_FUSED_BWD_ENV = "DLRM_HSTU_FP8_SEQ_A2A_FUSED_BWD"
_FUSED_VERIFY_ENV = "DLRM_HSTU_FP8_SEQ_A2A_FUSED_VERIFY"
_FUSED_RECV_ENV = "DLRM_HSTU_FP8_SEQ_A2A_FUSED_RECV"
_ARMED_FUSED_MSG = (
    "DLRMV4 armed OK: seq a2a fused send (fwd cast+clamp+gather, bwd gather+decode+div)"
)
_ARMED_RECV_MSG = (
    "DLRMV4 armed OK: seq a2a fused recv (fwd decode+unbucketize gather, "
    "bwd unbucketize scatter+encode)"
)
# Per-path proof lines, logged once per process at the first SUCCESSFUL launch
# of the backward kernels (the two lines above are logged at the first forward,
# where the backward is only planned).
_ARMED_SEND_BWD_MSG = "DLRMV4 armed OK: seq a2a fused send bwd ran (K2 gather+decode+div)"
_ARMED_RECV_BWD_MSG = "DLRMV4 armed OK: seq a2a fused recv bwd ran (K4 scatter+encode)"

_installed = False
_recv_installed = False
_orig: Dict[str, Any] = {}
_state: Dict[str, Any] = {
    "disabled": False,
    "announced": False,
    "check_left": None,
    "verify_fwd_left": None,
    "verify_bwd_left": None,
    # fused send path
    "fused_cfg": None,  # (fwd, bwd), read once
    "fused_disabled": False,
    "fused_announced": False,
    "fused_skip_logged": set(),
    "fused_mod": None,
    "fail_logged": set(),
    # static eligibility, per long-lived codecs object: key -> (codecs, reason)
    "static_cache": {},
    # fused receive path
    "recv_cfg": None,  # bool, read once
    "recv_disabled": False,
    "recv_announced": False,
    "recv_skip_logged": set(),
    "recv_verify_fwd_left": None,
    "recv_verify_bwd_left": None,
    # per-path proof lines (backward kernels) and refused hand-offs, logged once
    "send_bwd_announced": False,
    "recv_bwd_announced": False,
    "handoff_refused_logged": set(),
}

# Grow-only constant 0..n-1 buffers per (device, dtype); slicing == arange(n).
_arange_cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}


def enabled() -> bool:
    return os.environ.get(_ENV, "0").strip().lower() in ("1", "true", "on", "yes")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or 0)
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "on", "yes")


def fused_config() -> Tuple[bool, bool]:
    """(fused forward, fused backward) switches, read once from the env."""
    cfg = _state["fused_cfg"]
    if cfg is None:
        send = _env_flag(_FUSED_ENV, False) and _env_flag(_FUSED_SEND_ENV, True)
        cfg = (
            send and _env_flag(_FUSED_FWD_ENV, True),
            send and _env_flag(_FUSED_BWD_ENV, True),
        )
        _state["fused_cfg"] = cfg
    return cfg


def _verify_count() -> int:
    if any(fused_config()) and os.environ.get(_FUSED_VERIFY_ENV, "").strip():
        return _env_int(_FUSED_VERIFY_ENV, 0)
    return _env_int(_VERIFY_ENV, 0)


def _fused_module() -> Any:
    mod = _state["fused_mod"]
    if mod is None:
        try:
            from generative_recommenders.dlrm_v4 import seq_a2a_fused as mod
        except Exception as e:  # noqa: BLE001
            logger.warning("DLRMV4 seq a2a fused send: import failed (%s: %s)", type(e).__name__, e)
            mod = False
        _state["fused_mod"] = mod
    return mod or None


def _arange(n: int, device: torch.device, dtype: torch.dtype = torch.int64) -> torch.Tensor:
    key = (device, dtype)
    buf = _arange_cache.get(key)
    if buf is None or buf.numel() < n:
        cap = max(n, 2 * buf.numel() if buf is not None else 0)
        buf = torch.arange(cap, device=device, dtype=dtype)
        _arange_cache[key] = buf
    return buf[:n]


# --------------------------------------------------------------------------
# Pure math (no process group); unit-testable / benchmarkable directly.
# --------------------------------------------------------------------------
def block_rows(tok_lengths: torch.Tensor, n_blocks: int) -> torch.Tensor:
    """Rows per contiguous block: token lengths [n_blocks, B] summed -> [n_blocks, 1]."""
    return tok_lengths.view(n_blocks, -1).sum(dim=1, keepdim=True)


def row_index(
    recat: torch.Tensor, blk: torch.Tensor, rows: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(permuted block rows [n_blocks, 1], row index [rows]) for a block permute."""
    pblk, idx, _ = torch.ops.fbgemm.permute_2D_sparse_data(
        recat, blk, _arange(rows, blk.device), None, rows
    )
    return pblk, idx


def fwd_rowperm(
    recat: torch.Tensor,
    tok_lengths: torch.Tensor,
    lengths_d: torch.Tensor,
    emb: torch.Tensor,
    n_blocks: int,
    D: int,
    encode: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (permuted D-scaled lengths [n_blocks, B], wire 1-D, permuted blk).

    Equals stock ``permute_2D_sparse_data(recat, lengths_d.view(n_blocks,-1),
    emb.view(-1), None, numel)`` followed by ``encode`` (bitwise), for any
    elementwise ``encode``.
    """
    permuted_lengths_d, wire, pblk, _ = _fwd_rowperm_impl(
        recat, tok_lengths, lengths_d, emb, n_blocks, D, encode
    )
    return permuted_lengths_d, wire, pblk


def _fwd_rowperm_impl(
    recat: torch.Tensor,
    tok_lengths: torch.Tensor,
    lengths_d: torch.Tensor,
    emb: torch.Tensor,
    n_blocks: int,
    D: int,
    encode: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``fwd_rowperm`` plus the block row counts ``blk`` (int64 [n_blocks])."""
    rows = emb.numel() // D
    permuted_lengths_d = torch.index_select(
        lengths_d.view(n_blocks, -1), 0, recat
    )
    blk = block_rows(tok_lengths, n_blocks)
    pblk, fidx = row_index(recat, blk, rows)
    x = emb.view(-1)
    if encode is not None:
        x = encode(x)
    wire = torch.index_select(x.view(rows, D), 0, fidx).view(-1)
    return permuted_lengths_d, wire, pblk, blk.view(-1)


def bwd_rowperm(
    bwd_recat: torch.Tensor,
    pblk: torch.Tensor,
    wire: torch.Tensor,
    D: int,
    decode: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    """Equals stock ``decode`` then ``permute_2D_sparse_data(bwd_recat, ...)`` (1-D)."""
    rows = wire.numel() // D
    _, bidx = row_index(bwd_recat, pblk, rows)
    g = torch.index_select(wire.view(rows, D), 0, bidx).view(-1)
    if decode is not None:
        g = decode(g)
    return g


def stock_fwd(recat, lengths_d, emb, n_blocks, encode):
    """Reference: exactly the stock All2All_Seq_Req forward permute + encode."""
    pl, v, _ = torch.ops.fbgemm.permute_2D_sparse_data(
        recat, lengths_d.view(n_blocks, -1), emb.view(-1), None, emb.numel()
    )
    if encode is not None:
        v = encode(v)
    return pl, v


def stock_bwd(bwd_recat, permuted_lengths_d, wire, decode):
    """Reference: exactly the stock All2All_Seq_Req backward decode + permute."""
    g = decode(wire) if decode is not None else wire
    _, g, _ = torch.ops.fbgemm.permute_2D_sparse_data(
        bwd_recat, permuted_lengths_d, g, None, g.numel()
    )
    return g


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.is_floating_point:
        it = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return bool(torch.equal(a.view(it), b.view(it)))
    return bool(torch.equal(a, b))


# --------------------------------------------------------------------------
# Codec eligibility
# --------------------------------------------------------------------------
def _codec_elementwise(codec: Any) -> bool:
    name = type(codec).__name__
    if name == "NoOpQuantizedCommCodec":
        return True
    if name != "QuantizedCommCodec":
        return False
    try:
        from fbgemm_gpu.split_embedding_configs import SparseType
    except Exception:  # noqa: BLE001
        return False
    prec = getattr(codec, "_comm_precision", None)
    return prec in (SparseType.FP32, SparseType.FP16, SparseType.BF16)


def _codecs_ok(codecs: Any) -> bool:
    if codecs is None:
        return True
    return _codec_elementwise(codecs.forward) and _codec_elementwise(codecs.backward)


def _is_stock_fp16_encode(fn: Any) -> bool:
    """fbgemm's ``torch.clamp(t, TORCH_HALF_MIN, TORCH_HALF_MAX).half()``."""
    return (
        getattr(fn, "__module__", None) == "fbgemm_gpu.quantize_utils"
        and getattr(fn, "__qualname__", None) == "fp32_to_fp16_with_clamp"
    )


def _is_lowmem_fp16_encode(fn: Any) -> bool:
    """train/utils.py ``_lowmem_fp16``: ``t.half().clamp_(_HMIN, _HMAX)``, +-65504."""
    if getattr(fn, "__qualname__", None) != (
        "_patch_fbgemm_lowmem_clamp_cast.<locals>._lowmem_fp16"
    ):
        return False
    if not str(getattr(fn, "__module__", "")).endswith("dlrm_v4.train.utils"):
        return False
    code = getattr(fn, "__code__", None)
    if code is None or tuple(code.co_names) != ("half", "clamp_"):
        return False
    try:
        cells = {
            name: cell.cell_contents
            for name, cell in zip(code.co_freevars, fn.__closure__ or ())
        }
        return float(cells["_HMIN"]) == -65504.0 and float(cells["_HMAX"]) == 65504.0
    except Exception:  # noqa: BLE001
        return False


def _fp16_codecs_exact(codecs: Any) -> Optional[str]:
    """None if both codecs are exactly fbgemm's FP16 cast codec, else a reason.

    The fused kernels hard-code the FP16 encode (RNE cast + clamp to +-65504,
    NaN passing) and decode (exact upcast), so anything else (another precision,
    a loss scale, a patched codec or quantize function) is rejected.
    """
    if codecs is None:
        return "no codec"
    try:
        from fbgemm_gpu import quantize_comm as qc, quantize_utils as qu
        from fbgemm_gpu.split_embedding_configs import SparseType
    except Exception:  # noqa: BLE001
        return "fbgemm import"
    why: Optional[str] = None
    for side in ("forward", "backward"):
        c = getattr(codecs, side, None)
        if type(c) is not qc.QuantizedCommCodec:
            why = "%s codec %s" % (side, type(c).__name__)
        elif getattr(c, "_comm_precision", None) != SparseType.FP16:
            why = "%s codec precision %s" % (side, getattr(c, "_comm_precision", None))
        elif getattr(c, "_loss_scale", None) is not None:
            why = "%s codec loss_scale" % side
        elif "encode" in vars(c) or "decode" in vars(c):
            why = "%s codec instance override" % side
        if why:
            break
    if why is None:
        for name in ("encode", "decode"):
            f = getattr(qc.QuantizedCommCodec, name, None)
            if getattr(f, "__module__", None) != qc.__name__:
                why = "QuantizedCommCodec.%s patched" % name
        for name in ("_quantize_tensor", "_dequantize_tensor"):
            f = getattr(qc, name, None)
            if getattr(f, "__module__", None) != qc.__name__:
                why = "quantize_comm.%s patched" % name
    if why is None:
        dec = getattr(qc, "fp16_to_fp32", None)
        if not (
            getattr(dec, "__module__", None) == "fbgemm_gpu.quantize_utils"
            and getattr(dec, "__qualname__", None) == "fp16_to_fp32"
        ):
            why = "fp16 decode %s" % getattr(dec, "__qualname__", dec)
    if why is None:
        enc = getattr(qc, "fp32_to_fp16_with_clamp", None)
        if _is_stock_fp16_encode(enc):
            if not (float(qu.TORCH_HALF_MIN) == -65504.0 and float(qu.TORCH_HALF_MAX) == 65504.0):
                why = "fbgemm half limits"
        elif not _is_lowmem_fp16_encode(enc):
            why = "fp16 encode %s.%s" % (
                getattr(enc, "__module__", "?"),
                getattr(enc, "__qualname__", "?"),
            )
    return why


def _static_reason(
    kind: str, codecs: Any, D: int, world_size: int, device: torch.device
) -> Optional[str]:
    """The call-invariant part of the fused eligibility, cached.

    Key: the identity of the codecs object (owned by the sharded module and
    reused by every call; the cache holds a strong reference, so the id cannot
    be recycled while cached), D, the world size and the device type. The
    torchrec ``All2AllSequenceInfo`` is rebuilt on every call and is not a usable
    key. Valid while the fbgemm codec functions are not re-patched mid-run
    (``train/utils.py`` patches them once, at shard time, before training).
    """
    cache = _state["static_cache"]
    key = (kind, id(codecs), int(D), int(world_size), device.type)
    hit = cache.get(key)
    if hit is not None and hit[0] is codecs:
        return hit[1]
    saf = _fused_module()
    why: Optional[str] = None
    if saf is None or not saf.available():
        why = "triton unavailable"
    elif not (device.type == "cuda" or (device.type == "cpu" and saf.interpret_mode())):
        why = "device %s" % device
    elif D % 8 != 0:
        why = "D=%d not a multiple of 8" % D
    elif kind == "send" and not saf.is_pow2(world_size):
        why = "world_size=%d not a power of two" % world_size
    else:
        why = _fp16_codecs_exact(codecs)
    cache[key] = (codecs, why)
    return why


def _fused_reason(
    a2ai: Any,
    emb: torch.Tensor,
    tok_lengths: torch.Tensor,
    n_blocks: int,
    world_size: int,
    rows: int,
    D: int,
) -> Optional[str]:
    """None if the fused send path may run for this call, else why not."""
    why = _static_reason("send", a2ai.codecs, D, world_size, emb.device)
    if why is not None:
        return why
    saf = _fused_module()
    if emb.dtype != torch.float32:
        return "input dtype %s" % emb.dtype
    if not 1 <= n_blocks <= saf.MAX_BLOCKS:
        return "n_blocks=%d" % n_blocks
    if not 0 < rows < (1 << 30):
        return "rows=%d" % rows
    for name, r in (
        ("forward_recat", a2ai.forward_recat_tensor),
        ("backward_recat", a2ai.backward_recat_tensor),
    ):
        if (
            r.dtype not in (torch.int32, torch.int64)
            or r.numel() != n_blocks
            or not r.is_contiguous()
            or r.device != emb.device
        ):
            return name
    if (
        tok_lengths.dtype not in (torch.int32, torch.int64)
        or not tok_lengths.is_contiguous()
        or tok_lengths.device != emb.device
        or tok_lengths.numel() % n_blocks
        or tok_lengths.numel() >= (1 << 31)
    ):
        return "lengths"
    return None


def _fused_failed(where: str, e: BaseException) -> None:
    """A fused send kernel raised: rowperm path for this call and the rest of the run."""
    _state["fused_disabled"] = True
    if where not in _state["fail_logged"]:
        _state["fail_logged"].add(where)
        logger.warning(
            "DLRMV4 seq a2a fused send: %s raised %s: %s; fused send path DISABLED "
            "for the rest of the run (rowperm path)",
            where,
            type(e).__name__,
            e,
        )


def _fused_plan(
    a2ai: Any,
    emb: torch.Tensor,
    tok_lengths: torch.Tensor,
    n_blocks: int,
    world_size: int,
    rows: int,
    D: int,
) -> Tuple[bool, bool]:
    """(run K1 in this forward, run K2 in its backward)."""
    fwd, bwd = fused_config()
    if not (fwd or bwd) or _state["fused_disabled"]:
        return False, False
    why = _fused_reason(a2ai, emb, tok_lengths, n_blocks, world_size, rows, D)
    if why is not None:
        if why not in _state["fused_skip_logged"]:
            _state["fused_skip_logged"].add(why)
            logger.info(
                "DLRMV4 seq a2a fused send: not eligible (%s); rowperm path", why
            )
        return False, False
    return fwd, bwd


class _BwdPlan:
    """What the patched backward needs from its forward (``ctx._seq_rowperm``)."""

    __slots__ = ("pblk", "blk", "recat", "bwd_recat", "fused", "rows")

    def __init__(self, pblk, blk, recat, bwd_recat, fused, rows) -> None:
        self.pblk = pblk  # [n_blocks, 1] int64 (rowperm forward) or None
        self.blk = blk  # [n_blocks] int64; None when no fused kernel needs it
        self.recat = recat
        self.bwd_recat = bwd_recat
        self.fused = fused
        self.rows = rows

    def permuted_blk(self) -> torch.Tensor:
        if self.pblk is None:  # fused forward: blk[recat], same values as row_index's
            self.pblk = torch.index_select(self.blk.view(-1, 1), 0, self.recat)
        return self.pblk


# --------------------------------------------------------------------------
# Patched autograd Function methods
# --------------------------------------------------------------------------
def _eligible(a2ai: Any, emb: torch.Tensor) -> bool:
    if _state["disabled"] or not enabled():
        return False
    if a2ai.variable_batch_size or a2ai.forward_recat_tensor is None:
        return False
    if a2ai.backward_recat_tensor is None:
        return False
    if a2ai.lengths_after_sparse_data_all2all.shape[0] == 0:
        return False
    D = a2ai.embedding_dim
    if D <= 0 or emb.numel() % D != 0 or not emb.is_contiguous():
        return False
    return _codecs_ok(a2ai.codecs)


def _forward(ctx, pg, myreq, a2ai, sharded_input_embeddings):  # noqa: C901
    if not _eligible(a2ai, sharded_input_embeddings):
        ctx._seq_rowperm = None
        return _orig["forward"](ctx, pg, myreq, a2ai, sharded_input_embeddings)

    import torch.distributed as dist
    from torch.autograd.profiler import record_function

    world_size = dist.get_world_size(pg)
    my_rank = dist.get_rank(pg)
    D = a2ai.embedding_dim
    tok_lengths = a2ai.lengths_after_sparse_data_all2all
    local_T = tok_lengths.shape[0]
    n_blocks = local_T * world_size
    rows = sharded_input_embeddings.numel() // D
    fused_fwd, fused_bwd = _fused_plan(
        a2ai, sharded_input_embeddings, tok_lengths, n_blocks, world_size, rows, D
    )
    # The fused forward writes the permuted D-scaled lengths itself; the
    # unpermuted ones are then only needed by VERIFY (computed there).
    lengths_d = None if fused_fwd else tok_lengths * D
    input_splits = [i * D for i in a2ai.output_splits]
    output_splits = [i * D for i in a2ai.input_splits]

    if _state["check_left"] is None:
        _state["check_left"] = _env_int(_CHECK_ENV, 1)
    if _state["check_left"] > 0:
        _state["check_left"] -= 1
        tot = int(block_rows(tok_lengths, n_blocks).sum().item())
        if tot != rows:
            _state["disabled"] = True
            logger.warning(
                "DLRMV4 seq a2a rowperm: sum(lengths)=%d != rows=%d; "
                "stock permute for the rest of the run",
                tot,
                rows,
            )
            ctx._seq_rowperm = None
            return _orig["forward"](ctx, pg, myreq, a2ai, sharded_input_embeddings)

    a2ai.input_splits = input_splits
    a2ai.output_splits = output_splits

    codecs = a2ai.codecs
    if codecs is not None:
        qcomm_ctx = codecs.forward.create_context()
        encode = lambda t: codecs.forward.encode(t, qcomm_ctx)  # noqa: E731
    else:
        qcomm_ctx = None
        encode = None

    if not _state["announced"]:
        _state["announced"] = True
        logger.info(
            "DLRMV4 armed OK: seq a2a rowperm: row-index permute in wire dtype "
            "(rows=%d, D=%d, blocks=%d, world=%d, codec fwd=%s bwd=%s)",
            rows,
            D,
            n_blocks,
            world_size,
            getattr(getattr(codecs, "forward", None), "_comm_precision", "none"),
            getattr(getattr(codecs, "backward", None), "_comm_precision", "none"),
        )

    forward_recat_tensor = a2ai.forward_recat_tensor
    pblk: Optional[torch.Tensor] = None
    with record_function("## All2All_Seq_fwd_permute ##"):
        done = False
        if fused_fwd:
            # Prologue: the one aten::sum (block row counts); K1 does the rest.
            blk = block_rows(tok_lengths, n_blocks).view(-1)
            try:
                permuted_lengths_d, wire = _fused_module().send_fwd(
                    sharded_input_embeddings,
                    blk,
                    forward_recat_tensor,
                    tok_lengths,
                    n_blocks,
                    D,
                )
                done = True
            except Exception as e:  # noqa: BLE001 (Triton compile/launch failure)
                _fused_failed("send fwd (K1)", e)
                fused_fwd = fused_bwd = False
                lengths_d = tok_lengths * D
        if not done:
            permuted_lengths_d, wire, pblk, blk = _fwd_rowperm_impl(
                forward_recat_tensor,
                tok_lengths,
                lengths_d,
                sharded_input_embeddings,
                n_blocks,
                D,
                encode,
            )
    # Keep blk only where a fused kernel can read it: K2 (fused_bwd), or
    # permuted_blk() after a fused forward (pblk None). With the fused path off
    # the rowperm backward reads only pblk, so blk is not held from forward to
    # backward (the stock rowperm frees it in the forward).
    plan: Optional[_BwdPlan] = _BwdPlan(
        pblk,
        blk if (fused_fwd or fused_bwd) else None,
        forward_recat_tensor,
        a2ai.backward_recat_tensor,
        fused_bwd,
        rows,
    )

    if (fused_fwd or fused_bwd) and not _state["fused_announced"]:
        _state["fused_announced"] = True
        logger.info(
            "%s [fwd=%s bwd=%s rows=%d D=%d blocks=%d world=%d]",
            _ARMED_FUSED_MSG,
            "on" if fused_fwd else "off",
            "on" if fused_bwd else "off",
            rows,
            D,
            n_blocks,
            world_size,
        )

    if _state["verify_fwd_left"] is None:
        _state["verify_fwd_left"] = _verify_count()
    if _state["verify_fwd_left"] > 0:
        _state["verify_fwd_left"] -= 1
        if lengths_d is None:
            lengths_d = tok_lengths * D
        ref_pl, ref_wire = stock_fwd(
            forward_recat_tensor, lengths_d, sharded_input_embeddings, n_blocks, encode
        )
        ok = bits_equal(ref_pl, permuted_lengths_d) and bits_equal(ref_wire, wire)
        tag = "fused" if fused_fwd else "rowperm"
        logger.info("DLRMV4 seq a2a %s verify fwd: bitwise=%s", tag, ok)
        if not ok:
            if fused_fwd:
                _state["fused_disabled"] = True  # rowperm path from the next call
            else:
                _state["disabled"] = True
            permuted_lengths_d, wire = ref_pl, ref_wire
            plan = None
            logger.warning("DLRMV4 seq a2a %s: fwd MISMATCH; using stock", tag)

    if codecs is not None:
        output_splits = [
            codecs.forward.calc_quantized_size(x, qcomm_ctx) for x in output_splits
        ]
        input_splits = [
            codecs.forward.calc_quantized_size(x, qcomm_ctx) for x in input_splits
        ]

    sharded_output_embeddings = torch.empty(
        sum(output_splits), dtype=wire.dtype, device=wire.device
    )
    with record_function("## All2All_Seq_fwd ##"):
        req = dist.all_to_all_single(
            output=sharded_output_embeddings,
            input=wire,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=pg,
            async_op=True,
        )
    from torchrec.distributed import comm_ops

    a2ai.permuted_lengths_after_sparse_data_all2all = permuted_lengths_d
    myreq.req = req
    myreq.tensor = sharded_output_embeddings
    myreq.a2ai = a2ai
    myreq.wait_function = comm_ops.All2All_Seq_Req_Wait
    ctx.myreq = myreq
    myreq.qcomm_ctx = qcomm_ctx
    ctx.pg = pg
    ctx.my_rank = my_rank
    ctx.world_size = world_size
    ctx._seq_rowperm = plan  # None => stock backward
    return myreq.dummy_tensor


def _backward(ctx, *unused):  # noqa: C901
    plan = getattr(ctx, "_seq_rowperm", None)
    if plan is None:
        return _orig["backward"](ctx, *unused)

    import torch.distributed as dist
    from torch.autograd.profiler import record_function
    from torchrec.distributed import comm_ops

    myreq = ctx.myreq
    a2ai = myreq.a2ai
    D = a2ai.embedding_dim
    assert myreq.req is not None
    myreq.req.wait()
    wire = myreq.tensor
    codecs = a2ai.codecs
    qcomm_ctx = myreq.qcomm_ctx
    decode = (
        (lambda t: codecs.backward.decode(t, qcomm_ctx)) if codecs is not None else None
    )
    myreq.req = None
    myreq.tensor = None
    myreq.dummy_tensor = None

    bwd_recat = a2ai.backward_recat_tensor
    grad_div = bool(comm_ops.GRADIENT_DIVISION)
    world_size = dist.get_world_size(ctx.pg) if grad_div else None
    # K2 needs the forward's tables and an fp16 wire of the forward's size; it
    # applies GRADIENT_DIVISION itself (world_size is a power of two: checked
    # in the forward).
    use_fused = (
        plan.fused
        and not _state["fused_disabled"]
        and bwd_recat is plan.bwd_recat
        and wire.dtype == torch.float16
        and wire.is_contiguous()
        and wire.numel() == plan.rows * D
    )
    with record_function("## All2All_Seq_bwd_permute ##"):
        g: Optional[torch.Tensor] = None
        if use_fused:
            try:
                g = _fused_module().send_bwd(
                    wire, plan.blk, plan.recat, bwd_recat, D, world_size
                )
            except Exception as e:  # noqa: BLE001 (Triton compile/launch failure)
                _fused_failed("send bwd (K2)", e)
                use_fused = False
            if g is not None and not _state["send_bwd_announced"]:
                _state["send_bwd_announced"] = True
                logger.info("%s [rows=%d D=%d]", _ARMED_SEND_BWD_MSG, plan.rows, D)
        if g is None:
            g = bwd_rowperm(bwd_recat, plan.permuted_blk(), wire, D, decode)
    div_done = use_fused  # K2 scaled (or GRADIENT_DIVISION is off)

    if _state["verify_bwd_left"] is None:
        _state["verify_bwd_left"] = _verify_count()
    if _state["verify_bwd_left"] > 0:
        _state["verify_bwd_left"] -= 1
        ref = stock_bwd(
            bwd_recat,
            a2ai.permuted_lengths_after_sparse_data_all2all,
            wire,
            decode,
        )
        if div_done and grad_div:
            ref.div_(world_size)
        ok = bits_equal(ref, g)
        tag = "fused" if use_fused else "rowperm"
        logger.info("DLRMV4 seq a2a %s verify bwd: bitwise=%s", tag, ok)
        if not ok:
            if use_fused:
                _state["fused_disabled"] = True
            else:
                _state["disabled"] = True
            g = ref
            logger.warning("DLRMV4 seq a2a %s: bwd MISMATCH; using stock", tag)

    if grad_div and not div_done:
        g.div_(world_size)
    return (None, None, None, g.view(-1, D))


# --------------------------------------------------------------------------
# Fused receive path: patched All2All_Seq_Req_Wait (K3' forward, K4 backward)
# --------------------------------------------------------------------------
def recv_config() -> bool:
    """Fused receive switch (master and ..._FUSED_RECV), read once from the env."""
    cfg = _state["recv_cfg"]
    if cfg is None:
        cfg = _env_flag(_FUSED_ENV, False) and _env_flag(_FUSED_RECV_ENV, True)
        _state["recv_cfg"] = cfg
    return cfg


def _recv_verify_count() -> int:
    if os.environ.get(_FUSED_VERIFY_ENV, "").strip():
        return _env_int(_FUSED_VERIFY_ENV, 0)
    return _env_int(_VERIFY_ENV, 0)


def _ubg_module() -> Any:
    from generative_recommenders.dlrm_v4 import unbucketize_gather

    return unbucketize_gather


class _RecvHandoff:
    """The RW unbucketize permutation, parked on the a2a Request for the wait."""

    __slots__ = ("perm", "D", "consumed")

    def __init__(self, perm: torch.Tensor, D: int) -> None:
        self.perm = perm
        self.D = D
        self.consumed = False


class _RecvPlan:
    """What the patched wait backward needs from its forward (``ctx._seqa2a_recv``).

    mode: None (no permutation), "permute_rows" (``_unbucketize`` would use
    ``_PermuteRows``) or "index_select" (it would use ``torch.index_select``).
    """

    __slots__ = ("mode", "perm", "fused", "rows", "D")

    def __init__(self, mode, perm, fused, rows, D) -> None:
        self.mode = mode
        self.perm = perm
        self.fused = fused
        self.rows = rows
        self.D = D


def recv_handoff(tensor_awaitable: Any, perm: torch.Tensor, embedding_dim: int) -> Optional[_RecvHandoff]:
    """Park ``perm`` on the a2a Request for the patched wait (review item R3).

    Called by ``unbucketize_gather``'s ``SequenceEmbeddingsAwaitable.__init__``
    patch. Returns the hand-off (the caller then appends only a consumed-check
    callback), or None: the caller keeps today's ``_unbucketize`` callback. Only
    a ``comm_ops.Request`` whose ``wait_function`` is ``All2All_Seq_Req_Wait``
    with THIS module's patch installed qualifies (``alltoall_sequence`` can
    return a ``NoWait``, whose result would never pass through the wait).
    """
    if not _recv_installed or not recv_config() or _state["recv_disabled"]:
        return None
    try:
        from torchrec.distributed import comm_ops
    except Exception:  # noqa: BLE001
        return None
    wait_cls = comm_ops.All2All_Seq_Req_Wait
    why: Optional[str] = None
    if not isinstance(tensor_awaitable, comm_ops.Request):
        why = "awaitable %s" % type(tensor_awaitable).__name__
    elif getattr(tensor_awaitable, "wait_function", None) is not wait_cls:
        why = "wait function %s" % getattr(
            getattr(tensor_awaitable, "wait_function", None), "__name__", "?"
        )
    elif wait_cls.forward is not _wait_forward or wait_cls.backward is not _wait_backward:
        why = "All2All_Seq_Req_Wait re-patched"
    elif getattr(tensor_awaitable, "_seqa2a_recv", None) is not None:
        why = "hand-off already present"
    elif getattr(tensor_awaitable, "req", None) is None:
        why = "request already waited"
    elif (
        getattr(getattr(tensor_awaitable, "a2ai", None), "embedding_dim", None)
        != embedding_dim
    ):
        why = "embedding_dim"
    else:
        # Divert the permutation only where K3'/K4 can run at all: exact fp16
        # elementwise codecs (static, cached per codecs object). Any other
        # codec keeps _unbucketize's own callback; that also keeps the wait's
        # row count (wire numel / D) away from row-quantized wires.
        wire = getattr(tensor_awaitable, "tensor", None)
        if not isinstance(wire, torch.Tensor):
            why = "no wire"
        else:
            why = _static_reason(
                "recv", tensor_awaitable.a2ai.codecs, embedding_dim, 0, wire.device
            )
            if why is not None:
                why = "fused recv cannot run: " + why
    if why is not None:
        if why not in _state["handoff_refused_logged"]:
            _state["handoff_refused_logged"].add(why)
            logger.info(
                "DLRMV4 seq a2a fused recv: unbucketize hand-off refused (%s); "
                "stock unbucketize callback for such awaitables",
                why,
            )
        return None
    h = _RecvHandoff(perm, embedding_dim)
    tensor_awaitable._seqa2a_recv = h
    return h


def _fn_recorded(ctx: Any) -> bool:
    """Inside a custom Function forward: will autograd record this call?

    True iff grad mode was on at ``apply`` time and an input requires grad, i.e.
    iff the output will require grad, which is exactly ``_unbucketize``'s
    ``torch.is_grad_enabled() and x.requires_grad`` for the wait output ``x``.
    The node's next edges are collected before ``forward`` runs only when the
    call is recorded (``ctx.needs_input_grad`` ignores grad mode). Checked at
    install time by ``_recorded_probe_ok``.
    """
    return len(ctx.next_functions) > 0


def _recorded_probe_ok() -> bool:
    """Self-test of ``_fn_recorded`` on this torch build (CPU, microseconds)."""
    seen = []

    class _Probe(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):  # type: ignore[override]
            seen.append(_fn_recorded(ctx))
            return x.clone()

        @staticmethod
        def backward(ctx, g):  # type: ignore[override]
            return g

    try:
        x = torch.zeros(1, requires_grad=True)
        with torch.enable_grad():
            a = _Probe.apply(x)
            c = _Probe.apply(torch.zeros(1))
        with torch.no_grad():
            b = _Probe.apply(x)
        return (
            seen == [True, False, False]
            and a.requires_grad
            and not c.requires_grad
            and not b.requires_grad
        )
    except Exception:  # noqa: BLE001
        return False


def _ubg_decision(perm: torch.Tensor, rows: int, D: int, recorded: bool) -> str:
    """``unbucketize_gather._unbucketize``'s choice, same checks in the same order.

    The bijectivity check consumes the shared first-calls budget exactly when
    ``_unbucketize`` would, and a failure disables that module as it would.
    """
    ubg = _ubg_module()
    if (
        ubg._state["disabled"]
        or perm.dtype != torch.int64
        or not recorded
        or not ubg._bijective(perm, rows)
    ):
        return "index_select"
    ubg._announce(rows, D)
    return "permute_rows"


def _recv_reason(a2ai: Any, wire: Any, D: int, perm: Optional[torch.Tensor]) -> Optional[str]:
    """None if K3' (and later K4) may run for this wait, else why not."""
    if not recv_config():
        return "off"
    if _state["recv_disabled"]:
        return "disabled"
    if not isinstance(wire, torch.Tensor):
        return "no wire"
    if a2ai.codecs is None:
        return "no codec"
    why = _static_reason("recv", a2ai.codecs, D, 0, wire.device)
    if why is not None:
        return why
    # Reasons are logged once each: keep per-step sizes out of the strings.
    if wire.dtype != torch.float16:
        return "wire dtype %s" % wire.dtype
    if not wire.is_contiguous() or wire.numel() % D:
        return "wire layout"
    if perm is not None and (
        perm.dim() != 1
        or perm.numel() * D != wire.numel()
        or not perm.is_contiguous()
        or perm.device != wire.device
    ):
        return "perm layout"
    return None


def _recv_skip(why: str) -> None:
    if why in ("off", "disabled"):
        return
    if why not in _state["recv_skip_logged"]:
        _state["recv_skip_logged"].add(why)
        logger.info(
            "DLRMV4 seq a2a fused recv: not eligible (%s); stock decode/unbucketize", why
        )


def _recv_failed(where: str, e: BaseException) -> None:
    _state["recv_disabled"] = True
    if where not in _state["fail_logged"]:
        _state["fail_logged"].add(where)
        logger.warning(
            "DLRMV4 seq a2a fused recv: %s raised %s: %s; fused recv path DISABLED for "
            "the rest of the run (stock decode/unbucketize)",
            where,
            type(e).__name__,
            e,
        )


def _recv_mismatch(where: str) -> None:
    _state["recv_disabled"] = True
    logger.warning(
        "DLRMV4 seq a2a fused recv: verify %s MISMATCH; using the stock result, fused "
        "recv path DISABLED for the rest of the run",
        where,
    )


def _wait_forward(ctx, pg, myreq, *dummy_tensor):  # noqa: C901
    """Patched ``All2All_Seq_Req_Wait.forward``."""
    h = getattr(myreq, "_seqa2a_recv", None)
    if h is not None:
        myreq._seqa2a_recv = None
    ctx._seqa2a_recv = None
    a2ai = myreq.a2ai
    D = a2ai.embedding_dim
    wire = myreq.tensor
    perm = None if h is None else h.perm
    mode: Optional[str] = None
    rows = -1
    if perm is not None:
        rows = wire.numel() // h.D if isinstance(wire, torch.Tensor) and h.D > 0 else -1
        mode = _ubg_decision(perm, rows, h.D, _fn_recorded(ctx))
    fused = False
    if perm is None or mode == "permute_rows":
        why = _recv_reason(a2ai, wire, D, perm)
        if why is None:
            fused = True
        else:
            _recv_skip(why)

    if not fused:
        # Stock wait + decode (the original method), then, if a permutation was
        # handed over, exactly what _unbucketize would have run on the result.
        out = _orig["wait_forward"](ctx, pg, myreq, *dummy_tensor)
        if perm is not None:
            out = torch.index_select(out.view(-1, h.D), 0, perm)
            ctx._seqa2a_recv = _RecvPlan(mode, perm, False, rows, h.D)
            h.consumed = True
        return out

    from torch.autograd.profiler import record_function

    # Stock wait bookkeeping (All2All_Seq_Req_Wait.forward), then K3'.
    ctx.a2ai = a2ai
    assert myreq.req is not None
    myreq.req.wait()
    myreq.req = None
    myreq.tensor = None
    ctx.pg = pg
    ctx.myreq = myreq
    codecs = a2ai.codecs
    out = None
    try:
        with record_function("## All2All_Seq_recv_fwd_fused ##"):
            out = _fused_module().recv_fwd(wire, perm, D)
    except Exception as e:  # noqa: BLE001 (Triton compile/launch failure)
        _recv_failed("recv fwd (K3')", e)
    if out is None:
        fused = False
    else:
        if _state["recv_verify_fwd_left"] is None:
            _state["recv_verify_fwd_left"] = _recv_verify_count()
        if _state["recv_verify_fwd_left"] > 0:
            _state["recv_verify_fwd_left"] -= 1
            ref = codecs.forward.decode(wire, myreq.qcomm_ctx).view(-1, D)
            if perm is not None:
                ref = torch.index_select(ref, 0, perm)
            ok = bits_equal(ref, out)
            logger.info("DLRMV4 seq a2a fused recv verify fwd: bitwise=%s", ok)
            if not ok:
                _recv_mismatch("fwd")
                out, fused = ref, False
    if out is None:  # K3' raised: the stock math on the same received wire
        out = codecs.forward.decode(wire, myreq.qcomm_ctx).view(-1, D)
        if perm is not None:
            out = torch.index_select(out, 0, perm)
    if fused and not _state["recv_announced"]:
        _state["recv_announced"] = True
        logger.info(
            "%s [perm=%s rows=%d D=%d]",
            _ARMED_RECV_MSG,
            "on" if perm is not None else "off",
            out.size(0),
            D,
        )
    ctx._seqa2a_recv = _RecvPlan(mode, perm, fused, out.size(0), D)
    if h is not None:
        h.consumed = True
    return out


def _unpermute_stock(plan: _RecvPlan, grad: torch.Tensor) -> torch.Tensor:
    """The stock gradient of the permutation step (what autograd ran before)."""
    if plan.mode == "permute_rows":
        ubg = _ubg_module()
        return ubg._BACKWARDS[ubg._mode()](grad, plan.perm)
    if plan.mode == "index_select":
        # IndexSelectBackward0: zeros(self.sizes()).index_add_(0, index, grad).
        return torch.zeros(
            (plan.rows, plan.D), dtype=grad.dtype, device=grad.device
        ).index_add_(0, plan.perm, grad)
    return grad


def _wait_backward(ctx, sharded_grad_output):  # noqa: C901
    """Patched ``All2All_Seq_Req_Wait.backward``."""
    plan = getattr(ctx, "_seqa2a_recv", None)
    if plan is None:
        return _orig["wait_backward"](ctx, sharded_grad_output)
    grad = sharded_grad_output
    if not plan.fused:
        return _orig["wait_backward"](ctx, _unpermute_stock(plan, grad))
    if (
        _state["recv_disabled"]
        or grad.dtype != torch.float32
        or not grad.is_contiguous()
        or tuple(grad.shape) != (plan.rows, plan.D)
    ):
        if not _state["recv_disabled"]:
            _recv_skip(
                "grad dtype %s" % grad.dtype
                if grad.dtype != torch.float32
                else "grad layout (contiguous=%s)" % grad.is_contiguous()
            )
        return _orig["wait_backward"](ctx, _unpermute_stock(plan, grad))

    import torch.distributed as dist
    from torch.autograd.profiler import record_function

    myreq = ctx.myreq
    a2ai = ctx.a2ai
    pg = ctx.pg
    codecs = a2ai.codecs
    qcomm_ctx = codecs.backward.create_context()
    wire = None
    try:
        with record_function("## All2All_Seq_recv_bwd_fused ##"):
            wire = _fused_module().recv_bwd(grad, plan.perm, plan.D)
    except Exception as e:  # noqa: BLE001 (Triton compile/launch failure)
        _recv_failed("recv bwd (K4)", e)
        return _orig["wait_backward"](ctx, _unpermute_stock(plan, grad))
    if not _state["recv_bwd_announced"]:
        _state["recv_bwd_announced"] = True
        logger.info(
            "%s [perm=%s rows=%d D=%d]",
            _ARMED_RECV_BWD_MSG,
            "on" if plan.perm is not None else "off",
            plan.rows,
            plan.D,
        )
    if _state["recv_verify_bwd_left"] is None:
        _state["recv_verify_bwd_left"] = _recv_verify_count()
    if _state["recv_verify_bwd_left"] > 0:
        _state["recv_verify_bwd_left"] -= 1
        ref = codecs.backward.encode(_unpermute_stock(plan, grad), qcomm_ctx)
        ok = bits_equal(ref, wire)
        logger.info("DLRMV4 seq a2a fused recv verify bwd: bitwise=%s", ok)
        if not ok:
            _recv_mismatch("bwd")
            wire = ref

    # The rest is the stock All2All_Seq_Req_Wait.backward after its encode.
    input_splits = a2ai.output_splits
    output_splits = a2ai.input_splits
    output_splits = [codecs.backward.calc_quantized_size(x, qcomm_ctx) for x in output_splits]
    input_splits = [codecs.backward.calc_quantized_size(x, qcomm_ctx) for x in input_splits]
    sharded_grad_input = torch.empty(
        sum(output_splits), device=wire.device, dtype=wire.dtype
    )
    with record_function("## All2All_Seq_bwd ##"):
        req = dist.all_to_all_single(
            output=sharded_grad_input,
            input=wire.view(-1),
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=pg,
            async_op=True,
        )
    myreq.req = req
    myreq.tensor = sharded_grad_input
    myreq.qcomm_ctx = qcomm_ctx
    return (None, None, myreq.dummy_tensor)


def _install_recv() -> bool:
    """Patch All2All_Seq_Req_Wait.forward/backward (idempotent; originals kept)."""
    global _recv_installed
    if _recv_installed:
        return True
    if not _recorded_probe_ok():
        logger.warning(
            "DLRMV4 seq a2a fused recv: this torch build fails the autograd-recording "
            "probe; fused recv path off (stock decode/unbucketize)"
        )
        return False
    try:
        from torchrec.distributed import comm_ops
    except Exception:  # noqa: BLE001
        return False
    try:
        ubg = _ubg_module()
    except Exception:  # noqa: BLE001
        ubg = None
    cls = comm_ops.All2All_Seq_Req_Wait
    _orig["wait_forward"] = cls.forward
    _orig["wait_backward"] = cls.backward
    cls.forward = staticmethod(_wait_forward)
    cls.backward = staticmethod(_wait_backward)
    _recv_installed = True
    if ubg is not None:  # the permutation hand-off (unbucketize_gather.install())
        ubg._RECV_HOOK = recv_handoff
    logger.info(
        "DLRMV4 seq a2a fused recv requested; All2All_Seq_Req_Wait patched (arms at "
        "the first eligible wait)"
    )
    return True


def _uninstall_recv_for_tests() -> None:
    """Restore the stock All2All_Seq_Req_Wait methods (unit tests only)."""
    global _recv_installed
    if not _recv_installed:
        return
    from torchrec.distributed import comm_ops

    cls = comm_ops.All2All_Seq_Req_Wait
    cls.forward = staticmethod(_orig.pop("wait_forward"))
    cls.backward = staticmethod(_orig.pop("wait_backward"))
    _recv_installed = False
    _ubg_module()._RECV_HOOK = None


def _uninstall_for_tests() -> None:
    """Restore the stock All2All_Seq_Req (and _Wait) methods (unit tests only)."""
    global _installed
    _uninstall_recv_for_tests()
    if not _installed:
        return
    from torchrec.distributed import comm_ops

    cls = comm_ops.All2All_Seq_Req
    cls.forward = staticmethod(_orig.pop("forward"))
    cls.backward = staticmethod(_orig.pop("backward"))
    _installed = False


def install() -> None:
    """Patch All2All_Seq_Req.forward/backward (idempotent; originals kept)."""
    global _installed
    if _installed:
        return
    if not enabled():
        if _env_flag(_FUSED_ENV, False):
            logger.warning(
                "DLRMV4 seq a2a fused send: %s=1 needs %s=1; fused send path off",
                _FUSED_ENV,
                _ENV,
            )
        return
    try:
        from torchrec.distributed import comm_ops
    except Exception:  # noqa: BLE001
        return
    cls = comm_ops.All2All_Seq_Req
    _orig["forward"] = cls.forward
    _orig["backward"] = cls.backward
    cls.forward = staticmethod(_forward)
    cls.backward = staticmethod(_backward)
    _installed = True
    logger.info("DLRMV4 seq a2a rowperm: All2All_Seq_Req patched")
    fwd, bwd = fused_config()
    if fwd or bwd:
        logger.info(
            "DLRMV4 seq a2a fused send requested (fwd=%s bwd=%s); arms at the first "
            "eligible call",
            "on" if fwd else "off",
            "on" if bwd else "off",
        )
    # Receive part: the All2All_Seq_Req_Wait patch exists only with the master
    # switch (and ..._FUSED_RECV) on, so the default path is untouched.
    if recv_config():
        _install_recv()


def _reset_for_tests() -> None:
    """Forget the read-once switches, counters and fallbacks (unit tests only)."""
    _state.update(
        disabled=False,
        announced=False,
        check_left=None,
        verify_fwd_left=None,
        verify_bwd_left=None,
        fused_cfg=None,
        fused_disabled=False,
        fused_announced=False,
        fused_skip_logged=set(),
        fail_logged=set(),
        static_cache={},
        recv_cfg=None,
        recv_disabled=False,
        recv_announced=False,
        recv_skip_logged=set(),
        recv_verify_fwd_left=None,
        recv_verify_bwd_left=None,
        send_bwd_announced=False,
        recv_bwd_announced=False,
        handoff_refused_logged=set(),
    )
