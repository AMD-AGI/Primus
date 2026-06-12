"""TE override that routes eligible FMHA-fwd calls to the hand-tuned
gfx950 hd64 BF16 [SWA-]causal kernel from https://github.com/mawad-amd/fwd-attn-asm.

Wraps `transformer_engine.pytorch.cpp_extensions.fused_attn.fused_attn_fwd`.
On first eligible call we delegate to CK once to capture the aux_ctx_tensors
template TE's autograd later passes to fused_attn_bwd, then reuse that
template (with our bit-identical LSE swapped in) on every subsequent hit.

Activated by `aiter_hd64_asm_override.pth` and gated by env vars:
    MLPERF_ENABLE_FWD_ATTN_ASM   "1" to enable. Default off.
    FMHA_HD64_ASM_CO             path to hand-tuned .co.
    FMHA_HD64_ASM_LOG            "1" to log every dispatch decision.
"""
from __future__ import annotations

import ctypes
import logging
import math
import os
import struct
import sys
from typing import Tuple

logger = logging.getLogger("fwd_attn_asm_override")
if os.environ.get("FMHA_HD64_ASM_LOG", "0") == "1":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)


_ENABLED = os.environ.get("MLPERF_ENABLE_FWD_ATTN_ASM", "0") == "1"
_CO_PATH = os.environ.get(
    "FMHA_HD64_ASM_CO",
    "/workspace/deps/fwd-attn-asm/build/fwd_d64_opt128.co",
)
_KERNEL_NAME = b"fmha_fwd_d64_bf16_causal"

# Tile shape baked into the kernel (BlockFmhaPipelineQRKSVSAsync<128,64,...>).
# S must be a multiple of BLOCK_M.
_BLOCK_M = 128
_LDS_BYTES = 13056
_BLOCK_THREADS = 256

_HIP_LIB = None
_CO_DATA: bytes | None = None
# Per-device caches: HIP modules are bound to a device's primary context, so
# each rank loads its own. Loading once globally would bind the kernel to
# whatever device was current at import time and any other rank would fail
# its launches with hipErrorInvalidDeviceFunction (rc=101).
_KFUNC_BY_DEV: dict = {}
_KMODULE_BY_DEV: dict = {}

# Cached CK aux_ctx_tensors templates, keyed by (qkv_layout, mask_type, dropout).
_AUX_CTX_TEMPLATES: dict = {}

# ENABLE_CG=1: use pre-allocated round-robin buffer pools (CUDA-graph-safe).
# ENABLE_CG=0 (default): dynamically allocate on every call (eager-safe).
_ENABLE_CG: bool = os.environ.get("ENABLE_CG", "0") == "1"

# Round-robin buffer pools for CUDA-graph-safe reuse (active when ENABLE_CG=1).
#
# GPT-OSS 20B has ~40 transformer layers.  All attention fwd calls happen
# before any attention bwd call, so at any instant up to N (num_layers)
# buffers are simultaneously "live" (forward written, backward not yet read).
# A single static buffer per (device, shape, dtype) would be overwritten by
# each successive layer, corrupting earlier layers' outputs before their
# backward pass reads them.
#
# Solution: a pool of _POOL_SIZE buffers per (device, shape, dtype), accessed
# with a monotonically increasing counter.  Each call within a step gets a
# distinct slot.  The counter only advances in eager/capture Python execution;
# CUDA graph replay re-uses the pointers baked at capture time automatically.
#
# Safety invariant: _POOL_SIZE >= num_attention_layers_per_forward_pass.
# GPT-OSS 20B has 24 attention layers; 32 gives a one-third safety margin.
_POOL_SIZE: int = int(os.environ.get("FMHA_HD64_ASM_POOL_SIZE", "32"))

# Pool state: (dev_id, shape, dtype) -> {"bufs": [Tensor, ...], "next": int}
_STATIC_O: dict = {}
_STATIC_LSE: dict = {}


def _ensure_hip_lib():
    """Load libamdhip64 and bind argtypes once. Returns the CDLL handle."""
    global _HIP_LIB
    if _HIP_LIB is not None:
        return _HIP_LIB
    lib = ctypes.CDLL("libamdhip64.so")
    lib.hipModuleLoadData.restype = ctypes.c_int
    lib.hipModuleLoadData.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.hipModuleGetFunction.restype = ctypes.c_int
    lib.hipModuleGetFunction.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p,
    ]
    lib.hipModuleLaunchKernel.restype = ctypes.c_int
    lib.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    _HIP_LIB = lib
    return _HIP_LIB


def _try_load_kernel() -> bool:
    """Cheap install-time gate: verify the .co exists and libamdhip64 is
    loadable. Actual hipModuleLoadData is deferred to first launch so the
    module loads on the right device (see `_KFUNC_BY_DEV`)."""
    global _CO_DATA
    try:
        _ensure_hip_lib()
        if _CO_DATA is None:
            with open(_CO_PATH, "rb") as f:
                _CO_DATA = f.read()
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("could not stage hand-tuned hd64 kernel: %r", e)
        return False


def _get_kfunc_for_device(device) -> ctypes.c_void_p:
    """Return the hipModule kernel function bound to `device`'s primary
    context, loading on demand and caching by device id."""
    import torch
    dev_id = device.index if hasattr(device, "index") and device.index is not None \
        else torch.cuda.current_device()
    cached = _KFUNC_BY_DEV.get(dev_id)
    if cached is not None:
        return cached
    hip = _ensure_hip_lib()
    if _CO_DATA is None:
        with open(_CO_PATH, "rb") as f:
            globals()["_CO_DATA"] = f.read()
    with torch.cuda.device(dev_id):
        module = ctypes.c_void_p()
        rc = hip.hipModuleLoadData(
            ctypes.byref(module), ctypes.create_string_buffer(_CO_DATA)
        )
        if rc != 0:
            raise RuntimeError(
                f"hipModuleLoadData failed for {_CO_PATH} on device {dev_id} "
                f"(rc={rc})"
            )
        func = ctypes.c_void_p()
        rc = hip.hipModuleGetFunction(ctypes.byref(func), module, _KERNEL_NAME)
        if rc != 0:
            raise RuntimeError(
                f"hipModuleGetFunction failed on device {dev_id} (rc={rc})"
            )
    _KMODULE_BY_DEV[dev_id] = module
    _KFUNC_BY_DEV[dev_id] = func
    logger.info(
        "loaded hand-tuned hd64 kernel from %s on device %d", _CO_PATH, dev_id
    )
    return func


def _is_gfx950(device) -> bool:
    import torch
    try:
        return torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx950")
    except Exception:
        return False


def _to_bshd(q, k, v, qkv_layout: str):
    """Return (q, k, v, was_sbhd). Underlying memory is unchanged; we just
    flag the layout so `_launch` can pull strides from the right axes."""
    if "bshd" in qkv_layout:
        return q, k, v, False
    if "sbhd" in qkv_layout:
        return q, k, v, True
    raise ValueError(f"unsupported qkv_layout for hand-tuned hd64 kernel: {qkv_layout}")


def _ck_window_args(window_size, attn_mask_type: str) -> Tuple[int, int]:
    """Translate aiter/TE's window_size convention to the kargs convention.

    aiter:   causal + (-1,-1) -> dense causal;  causal + (W, 0) -> SWA(W).
    kargs:   window_left=-1, window_right=0 -> dense causal;
             window_left=W,  window_right=0 -> SWA(W).
    """
    if window_size is None:
        wl, wr = -1, -1
    else:
        wl, wr = int(window_size[0]), int(window_size[1])
    if "causal" in (attn_mask_type or ""):
        wr = 0
    return wl, wr


def _eligible(*, q, k, v, attn_mask_type, window_size, qkv_layout, dropout,
              attn_bias, softmax_offset, s_quantizer, o_quantizer,
              fp8) -> bool:
    """Strict allow-list. Anything outside the bench validation envelope bails."""
    import torch

    if not _ENABLED:
        return False
    if fp8:
        return False
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return False
    if any(x is not None for x in (s_quantizer, o_quantizer)):
        return False
    if attn_bias is not None or softmax_offset is not None:
        return False

    # Only single-tensor bshd / sbhd. Packed (bs3hd) and varlen (thd_thd_thd)
    # are not supported by the launcher below.
    if qkv_layout not in ("bshd_bshd_bshd", "sbhd_sbhd_sbhd"):
        return False

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False
    if q.shape[-1] != 64 or v.shape[-1] != 64:
        return False

    if "causal" not in (attn_mask_type or ""):
        return False

    if dropout != 0.0:
        return False

    if not _is_gfx950(q.device):
        return False

    # Resolve B, S, HQ, HKV based on layout.
    if qkv_layout.startswith("bshd"):
        B, S, HQ, _ = q.shape
        HKV = k.shape[2]
    else:  # sbhd_sbhd_sbhd
        S, B, HQ, _ = q.shape
        HKV = k.shape[2]

    if HQ % HKV != 0:
        return False
    if S % _BLOCK_M != 0:
        return False

    return True


def _get_static_buf(cache, shape, dtype, device):
    """Return a buffer for the kernel output / LSE.

    When ENABLE_CG=1: round-robin pool (CUDA-graph-safe).
      Each (device_id, shape, dtype) key owns a pool of _POOL_SIZE tensors.
      A per-pool counter selects the slot: slot = counter % _POOL_SIZE.  The
      counter advances only during eager/capture Python execution, never during
      CUDA graph replay (replay reuses the pointers baked at capture time).
      This guarantees that concurrent "live" layers each hold a distinct buffer.

    When ENABLE_CG=0 (default): dynamic allocation on every call (eager-safe).
      Equivalent to the original torch.empty() path — no persistent state.
    """
    import torch
    if not _ENABLE_CG:
        return torch.empty(shape, dtype=dtype, device=device)

    dev_id = device.index if hasattr(device, "index") and device.index is not None \
        else torch.cuda.current_device()
    key = (dev_id, shape, dtype)
    entry = cache.get(key)
    if entry is None:
        entry = {"bufs": [], "next": 0}
        cache[key] = entry

    pool = entry["bufs"]
    idx = entry["next"] % _POOL_SIZE
    entry["next"] += 1

    if idx >= len(pool):
        buf = torch.empty(shape, dtype=dtype, device=device)
        pool.append(buf)
        if dev_id == 0:
            print(f"[fwd-attn-asm] allocated static buf #{len(pool)}/{_POOL_SIZE}: "
                  f"shape={shape} dtype={dtype} ptr=0x{buf.data_ptr():x} "
                  f"device={dev_id}", flush=True)

    return pool[idx]


def _launch(q, k, v, qkv_layout: str, *, attn_scale: float,
            attn_mask_type: str, window_size, lse_out):
    """Launch the hand-tuned kernel. Writes output to a static `o` buffer
    and softmax_lse to `lse_out`. Returns `o`.

    CUDA-graph safe: uses a persistent output buffer and launches on the
    current CUDA stream so the kernel is recorded into any active graph.
    """
    import torch

    if qkv_layout.startswith("bshd"):
        B, S, HQ, D = q.shape
        HKV = k.shape[2]
        s_q, h_q, b_q = q.stride(1), q.stride(2), q.stride(0)
        s_k, h_k, b_k = k.stride(1), k.stride(2), k.stride(0)
        s_v, h_v, b_v = v.stride(1), v.stride(2), v.stride(0)
    else:  # sbhd_sbhd_sbhd
        S, B, HQ, D = q.shape
        HKV = k.shape[2]
        s_q, h_q, b_q = q.stride(0), q.stride(2), q.stride(1)
        s_k, h_k, b_k = k.stride(0), k.stride(2), k.stride(1)
        s_v, h_v, b_v = v.stride(0), v.stride(2), v.stride(1)

    o = _get_static_buf(_STATIC_O, q.shape, q.dtype, q.device)
    if qkv_layout.startswith("bshd"):
        s_o, h_o, b_o = o.stride(1), o.stride(2), o.stride(0)
    else:
        s_o, h_o, b_o = o.stride(0), o.stride(2), o.stride(1)

    scale_s = (1.0 / math.sqrt(D)) * math.log2(math.e)

    wl, wr = _ck_window_args(window_size, attn_mask_type)

    kargs = struct.pack(
        "<QQQQQ iiii iif iiii iiii ii iiii xxxx Q ii iiii QQ",
        q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), 0,
        S, S, D, D,
        HQ, HQ // HKV, scale_s,
        s_q, s_k, s_v, s_o,
        h_q, h_k, h_v, h_o,
        HQ, 0,
        wl, wr, 0, 2,            # mask_type=2 == CK::MaskType::Causal
        lse_out.data_ptr(),
        lse_out.stride(1), lse_out.stride(0),
        b_q, b_k, b_v, b_o,
        0, 0,
    )
    karg_buf = (ctypes.c_ubyte * len(kargs))(*kargs)
    karg_size = ctypes.c_size_t(len(kargs))
    extra = (ctypes.c_void_p * 5)(
        ctypes.c_void_p(0x01),
        ctypes.cast(karg_buf, ctypes.c_void_p),
        ctypes.c_void_p(0x02),
        ctypes.cast(ctypes.pointer(karg_size), ctypes.c_void_p),
        ctypes.c_void_p(0x03),
    )
    kfunc = _get_kfunc_for_device(q.device)
    stream_ptr = torch.cuda.current_stream(q.device).cuda_stream
    if not hasattr(_launch, "_logged_stream"):
        _launch._logged_stream = True
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        enable_cg = os.environ.get("ENABLE_CG", "0") == "1"
        if rank == 0 and enable_cg:
            o_pool_sizes = {k: len(v["bufs"]) for k, v in _STATIC_O.items()}
            lse_pool_sizes = {k: len(v["bufs"]) for k, v in _STATIC_LSE.items()}
            print(f"[fwd-attn-asm] CUDA-graph-safe launch: "
                  f"stream=0x{stream_ptr:x} (current_stream), "
                  f"pool_size={_POOL_SIZE}, "
                  f"o_pools={o_pool_sizes}, "
                  f"lse_pools={lse_pool_sizes}", flush=True)
    rc = _HIP_LIB.hipModuleLaunchKernel(
        kfunc,
        HQ, S // _BLOCK_M, B,
        _BLOCK_THREADS, 1, 1,
        _LDS_BYTES,
        ctypes.c_void_p(stream_ptr),
        ctypes.c_void_p(0),
        ctypes.cast(extra, ctypes.c_void_p),
    )
    if rc != 0:
        raise RuntimeError(
            f"hipModuleLaunchKernel returned rc={rc} on device {q.device}"
        )
    return o


def _install_fused_attn_override():
    """Wrap `fused_attn_fwd` so eligible calls hit the hand-tuned kernel.

    Returns True on success. If TE is not importable yet, returns False; the
    deferred installer (see bottom of file) will retry when TE loads.
    """
    try:
        from transformer_engine.pytorch.cpp_extensions import fused_attn as _fa_mod
    except Exception as e:  # noqa: BLE001
        logger.info("TE not importable yet, deferring install (%r)", e)
        return False

    if getattr(_fa_mod.fused_attn_fwd, "_fwd_attn_asm_patched", False):
        return True
    if not _try_load_kernel():
        return False

    import torch

    _orig_fwd = _fa_mod.fused_attn_fwd

    def _capture_aux_template(orig_kwargs):
        """Run CK once with the same shape to learn aux_ctx_tensors layout."""
        out_, aux_ = _orig_fwd(**orig_kwargs)
        return out_, list(aux_)

    def patched_fused_attn_fwd(
        is_training,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        k,
        v,
        fake_dtype,
        fused_attention_backend,
        attn_bias=None,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        page_table_k=None,
        page_table_v=None,
        s_quantizer=None,
        o_quantizer=None,
        attn_scale=None,
        dropout=0.0,
        fast_zero_fill=True,
        qkv_layout="sbh3d",
        attn_bias_type="no_bias",
        attn_mask_type="padding",
        softmax_type="vanilla",
        window_size=(-1, -1),
        rng_gen=None,
        softmax_offset=None,
        return_max_logit=False,
        cuda_graph=False,
    ):
        all_kwargs = dict(
            is_training=is_training,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            q=q, k=k, v=v,
            fake_dtype=fake_dtype,
            fused_attention_backend=fused_attention_backend,
            attn_bias=attn_bias,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            page_table_k=page_table_k,
            page_table_v=page_table_v,
            s_quantizer=s_quantizer,
            o_quantizer=o_quantizer,
            attn_scale=attn_scale,
            dropout=dropout,
            fast_zero_fill=fast_zero_fill,
            qkv_layout=qkv_layout,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            softmax_type=softmax_type,
            window_size=window_size,
            rng_gen=rng_gen,
            softmax_offset=softmax_offset,
            return_max_logit=return_max_logit,
            cuda_graph=cuda_graph,
        )
        # Anything outside the bench validation envelope falls through to CK.
        if not _eligible(
            q=q, k=k, v=v,
            attn_mask_type=attn_mask_type,
            window_size=window_size,
            qkv_layout=qkv_layout,
            dropout=dropout,
            attn_bias=attn_bias,
            softmax_offset=softmax_offset,
            s_quantizer=s_quantizer,
            o_quantizer=o_quantizer,
            fp8=False,
        ):
            return _orig_fwd(**all_kwargs)
        if return_max_logit or page_table_k is not None:
            return _orig_fwd(**all_kwargs)

        if qkv_layout.startswith("bshd"):
            B, S, HQ, _ = q.shape
        else:
            S, B, HQ, _ = q.shape

        # CK ships softmax_lse as (B, HQ, S, 1). Use a static buffer so
        # the address is stable across CUDA graph replays.
        lse = _get_static_buf(
            _STATIC_LSE, (B, HQ, S, 1), torch.float32, q.device,
        )
        try:
            out = _launch(
                q, k, v, qkv_layout,
                attn_scale=attn_scale or 0.0,
                attn_mask_type=attn_mask_type,
                window_size=window_size,
                lse_out=lse,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("hand-tuned launch failed (%r); falling back to CK", e)
            return _orig_fwd(**all_kwargs)

        if not is_training:
            return out, []

        # TE's autograd passes aux_ctx_tensors straight to fused_attn_bwd, so
        # its layout has to match CK's. Capture it once per
        # (qkv_layout, mask, dropout) by delegating to CK on the first call.
        cache_key = (qkv_layout, attn_mask_type, dropout)
        template = _AUX_CTX_TEMPLATES.get(cache_key)
        if template is None:
            ck_out, ck_aux = _capture_aux_template(all_kwargs)
            template = ck_aux
            _AUX_CTX_TEMPLATES[cache_key] = template
            lse0 = template[0] if template else None
            if (lse0 is None
                    or getattr(lse0, "ndim", -1) != lse.ndim
                    or lse0.dtype != lse.dtype):
                logger.warning(
                    "CK aux_ctx[0] shape=%s dtype=%s structurally mismatches our "
                    "lse shape=%s dtype=%s; falling back to CK for this call.",
                    getattr(lse0, "shape", None),
                    getattr(lse0, "dtype", None),
                    lse.shape, lse.dtype,
                )
                _AUX_CTX_TEMPLATES.pop(cache_key, None)
                return ck_out, ck_aux

        # Slot 0 is softmax_lse; remaining slots (rng_state, etc.) are unused
        # at dropout=0 and can be reused from the template.
        aux = [lse] + list(template[1:])

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "fwd-attn-asm dispatched: shape=%s qkv_layout=%s mask=%s window=%s",
                tuple(q.shape), qkv_layout, attn_mask_type, window_size,
            )
        return out, aux

    patched_fused_attn_fwd._fwd_attn_asm_patched = True
    _fa_mod.fused_attn_fwd = patched_fused_attn_fwd

    # Also re-export under the name pulled by `backends.py`.
    try:
        from transformer_engine.pytorch.attention.dot_product_attention import backends as _b
        _b.fused_attn_fwd = patched_fused_attn_fwd
    except Exception:
        pass

    logger.info("fused_attn_fwd patched (D=64 BF16 [SWA-]causal -> hand-tuned hd64 kernel)")
    return True


class _DeferredInstaller:
    """Meta-path finder that triggers install once TE's fused_attn module loads."""

    _TARGETS = {
        "transformer_engine.pytorch.cpp_extensions.fused_attn",
        "transformer_engine.pytorch.cpp_extensions",
    }

    def find_spec(self, fullname, path, target=None):
        # find_spec runs for every import; swallow exceptions so a bug here
        # cannot break the host import system.
        try:
            if fullname not in self._TARGETS:
                return None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                try:
                    spec = finder.find_spec(fullname, path, target)
                except (AttributeError, ImportError):
                    spec = None
                if spec is None:
                    continue
                orig_loader = spec.loader

                class _WrappedLoader:
                    def create_module(self, spec):
                        if hasattr(orig_loader, "create_module"):
                            return orig_loader.create_module(spec)
                        return None

                    def exec_module(self, module):
                        orig_loader.exec_module(module)
                        if fullname == "transformer_engine.pytorch.cpp_extensions.fused_attn":
                            try:
                                _install_fused_attn_override()
                            except Exception as e:  # noqa: BLE001
                                logger.warning("deferred install failed: %r", e)

                spec.loader = _WrappedLoader()
                return spec
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("fwd-attn-asm find_spec error for %s: %r", fullname, e)
            return None


def _register_deferred_install():
    if any(isinstance(f, _DeferredInstaller) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _DeferredInstaller())
    logger.info("deferred installer registered; will patch on TE.fused_attn load")


# Imported at Python startup via the .pth shim — long before TE / torch / HIP
# have done first-touch init. Don't import TE here; defer patching to when
# TE.cpp_extensions.fused_attn actually loads. Wrapped in try/except so a
# buggy installer can't crash the host process — worst case the fast path
# stays off and training runs through CK.
if _ENABLED:
    try:
        _register_deferred_install()
    except Exception as e:  # noqa: BLE001
        logger.warning("fwd-attn-asm deferred install failed at startup: %r", e)
