"""Route eligible TransformerEngine FMHA forward calls to the pinned gfx950
HD64 BF16 assembly kernel.

The module is imported at Python startup through ``aiter_hd64_asm_override.pth``
and remains inactive unless ``MLPERF_ENABLE_FWD_ATTN_ASM=1``.
"""

from __future__ import annotations

import ctypes
import functools
import inspect
import logging
import math
import os
import struct
import sys
from typing import Tuple

logger = logging.getLogger("fwd_attn_asm_override")
# Per-dispatch logging is opt-in only: it fires once per attention call, so an
# inherited INFO root level (verbose runs) must not switch it on.
_DISPATCH_LOG = os.environ.get("FMHA_HD64_ASM_LOG", "0") == "1"
if _DISPATCH_LOG:
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)


_ENABLED = os.environ.get("MLPERF_ENABLE_FWD_ATTN_ASM", "0") == "1"
_AITER_ROPE_ENABLED = os.environ.get("NVTE_USE_AITER_ROPE", "0") == "1"
_DEFAULT_CO_PATH = os.path.join(
    os.path.dirname(__file__),
    "aiter_hd64_asm_fwd_d64_opt128.co",
)
_CO_PATH = os.environ.get("FMHA_HD64_ASM_CO", _DEFAULT_CO_PATH)
_KERNEL_NAME = b"fmha_fwd_d64_bf16_causal"

# Tile shape baked into the kernel (BlockFmhaPipelineQRKSVSAsync<128,64,...>).
_BLOCK_M = 128
_LDS_BYTES = 13056
_BLOCK_THREADS = 256

_HIP_LIB = None
_CO_DATA: bytes | None = None
# HIP modules are bound to a device's primary context, so each rank/device
# needs its own handle.
_KFUNC_BY_DEV: dict = {}
_KMODULE_BY_DEV: dict = {}
_DISPATCH_COUNT = 0

# CK aux tensors are needed by TE's backward wrapper. Capture a compatible
# template on the first eligible call and then replace only its LSE tensor.
_AUX_CTX_TEMPLATES: dict = {}


def get_dispatch_count() -> int:
    """Return successful hand-tuned kernel launches in this process."""
    return _DISPATCH_COUNT


def _ensure_hip_lib():
    global _HIP_LIB
    if _HIP_LIB is not None:
        return _HIP_LIB
    # ROCm Python wheels ship runtime and development copies of libamdhip64.
    # PyTorch is linked against the versioned runtime SONAME; opening the
    # unversioned development symlink can create a second HIP runtime with a
    # separate module/context registry, causing hipModuleGetFunction to return
    # hipErrorNotFound even though the code object contains the symbol.
    lib = ctypes.CDLL("libamdhip64.so.7")
    lib.hipModuleLoadData.restype = ctypes.c_int
    lib.hipModuleLoadData.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.hipModuleGetFunction.restype = ctypes.c_int
    lib.hipModuleGetFunction.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.hipModuleLaunchKernel.restype = ctypes.c_int
    lib.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _HIP_LIB = lib
    return _HIP_LIB


def _try_load_kernel() -> bool:
    """Stage the code object; bind it to a HIP context on first launch."""
    global _CO_DATA
    try:
        _ensure_hip_lib()
        if _CO_DATA is None:
            with open(_CO_PATH, "rb") as file:
                _CO_DATA = file.read()
        return True
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "could not stage hand-tuned hd64 kernel path=%s exists=%s: %r",
            _CO_PATH,
            os.path.exists(_CO_PATH),
            error,
        )
        return False


def _get_kfunc_for_device(device) -> ctypes.c_void_p:
    import torch

    dev_id = (
        device.index if hasattr(device, "index") and device.index is not None else torch.cuda.current_device()
    )
    cached = _KFUNC_BY_DEV.get(dev_id)
    if cached is not None:
        return cached

    hip = _ensure_hip_lib()
    if _CO_DATA is None:
        with open(_CO_PATH, "rb") as file:
            globals()["_CO_DATA"] = file.read()

    with torch.cuda.device(dev_id):
        module = ctypes.c_void_p()
        rc = hip.hipModuleLoadData(
            ctypes.byref(module),
            ctypes.create_string_buffer(_CO_DATA),
        )
        if rc != 0:
            raise RuntimeError(f"hipModuleLoadData failed for {_CO_PATH} on device {dev_id} " f"(rc={rc})")
        func = ctypes.c_void_p()
        rc = hip.hipModuleGetFunction(
            ctypes.byref(func),
            module,
            _KERNEL_NAME,
        )
        if rc != 0:
            raise RuntimeError(f"hipModuleGetFunction failed on device {dev_id} (rc={rc})")

    _KMODULE_BY_DEV[dev_id] = module
    _KFUNC_BY_DEV[dev_id] = func
    logger.info(
        "loaded hand-tuned hd64 kernel from %s on device %d",
        _CO_PATH,
        dev_id,
    )
    return func


def _is_gfx950(device) -> bool:
    import torch

    try:
        return torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx950")
    except Exception:
        return False


def _ck_window_args(window_size, attn_mask_type: str) -> Tuple[int, int]:
    if window_size is None:
        window_left, window_right = -1, -1
    else:
        window_left, window_right = int(window_size[0]), int(window_size[1])
    if "causal" in (attn_mask_type or ""):
        window_right = 0
    return window_left, window_right


def _eligible(
    *,
    max_seqlen_q,
    max_seqlen_kv,
    q,
    k,
    v,
    attn_scale,
    attn_bias_type,
    attn_mask_type,
    softmax_type,
    window_size,
    bottom_right_diagonal,
    qkv_layout,
    dropout,
    attn_bias,
    softmax_offset,
    s_quantizer,
    o_quantizer,
    fp8,
) -> bool:
    import torch

    if not _ENABLED or fp8:
        return False
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return False
    # TE 2.15 passes quantizer objects through its BF16 fused-attention API even
    # when fp8=False. They are unused by this BF16 output path and must not make
    # an otherwise eligible call fall back to CK.
    if attn_bias is not None or softmax_offset is not None:
        return False
    if attn_bias_type != "no_bias" or softmax_type != "vanilla":
        return False
    if bottom_right_diagonal not in (None, False):
        return False
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

    if qkv_layout.startswith("bshd"):
        batch, sequence_length, query_heads, _ = q.shape
        key_value_sequence_length = k.shape[1]
        key_value_heads = k.shape[2]
    else:
        sequence_length, batch, query_heads, _ = q.shape
        key_value_sequence_length = k.shape[0]
        key_value_heads = k.shape[2]

    if sequence_length != key_value_sequence_length:
        return False
    if max_seqlen_q != sequence_length or max_seqlen_kv != sequence_length:
        return False
    if q.shape[:2] != k.shape[:2] or k.shape[:2] != v.shape[:2]:
        return False
    if query_heads % key_value_heads != 0:
        return False
    if sequence_length % _BLOCK_M != 0:
        return False
    # TE uses 0.0 as a sentinel for the default 1/sqrt(head_dim) scale.
    if attn_scale not in (None, 0.0) and not math.isclose(
        float(attn_scale),
        1.0 / math.sqrt(q.shape[-1]),
        rel_tol=1e-6,
        abs_tol=0.0,
    ):
        return False
    if batch <= 0:
        return False
    return True


def _launch(
    q,
    k,
    v,
    qkv_layout: str,
    *,
    attn_scale: float,
    attn_mask_type: str,
    window_size,
    lse_out,
):
    import torch

    if qkv_layout.startswith("bshd"):
        batch, sequence_length, query_heads, head_dim = q.shape
        key_value_heads = k.shape[2]
        stride_q_s, stride_q_h, stride_q_b = (
            q.stride(1),
            q.stride(2),
            q.stride(0),
        )
        stride_k_s, stride_k_h, stride_k_b = (
            k.stride(1),
            k.stride(2),
            k.stride(0),
        )
        stride_v_s, stride_v_h, stride_v_b = (
            v.stride(1),
            v.stride(2),
            v.stride(0),
        )
    else:
        sequence_length, batch, query_heads, head_dim = q.shape
        key_value_heads = k.shape[2]
        stride_q_s, stride_q_h, stride_q_b = (
            q.stride(0),
            q.stride(2),
            q.stride(1),
        )
        stride_k_s, stride_k_h, stride_k_b = (
            k.stride(0),
            k.stride(2),
            k.stride(1),
        )
        stride_v_s, stride_v_h, stride_v_b = (
            v.stride(0),
            v.stride(2),
            v.stride(1),
        )

    output = torch.empty_like(q)
    if qkv_layout.startswith("bshd"):
        stride_o_s, stride_o_h, stride_o_b = (
            output.stride(1),
            output.stride(2),
            output.stride(0),
        )
    else:
        stride_o_s, stride_o_h, stride_o_b = (
            output.stride(0),
            output.stride(2),
            output.stride(1),
        )

    # The assembly kernel expects the CK log2(e)-scaled convention.
    del attn_scale
    scale_s = (1.0 / math.sqrt(head_dim)) * math.log2(math.e)
    window_left, window_right = _ck_window_args(
        window_size,
        attn_mask_type,
    )

    kargs = struct.pack(
        "<QQQQQ iiii iif iiii iiii ii iiii xxxx Q ii iiii QQ",
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        output.data_ptr(),
        0,
        sequence_length,
        sequence_length,
        head_dim,
        head_dim,
        query_heads,
        query_heads // key_value_heads,
        scale_s,
        stride_q_s,
        stride_k_s,
        stride_v_s,
        stride_o_s,
        stride_q_h,
        stride_k_h,
        stride_v_h,
        stride_o_h,
        query_heads,
        0,
        window_left,
        window_right,
        0,
        2,
        lse_out.data_ptr(),
        lse_out.stride(1),
        lse_out.stride(0),
        stride_q_b,
        stride_k_b,
        stride_v_b,
        stride_o_b,
        0,
        0,
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
    kernel = _get_kfunc_for_device(q.device)
    rc = _HIP_LIB.hipModuleLaunchKernel(
        kernel,
        query_heads,
        sequence_length // _BLOCK_M,
        batch,
        _BLOCK_THREADS,
        1,
        1,
        _LDS_BYTES,
        ctypes.c_void_p(0),
        ctypes.c_void_p(0),
        ctypes.cast(extra, ctypes.c_void_p),
    )
    if rc != 0:
        raise RuntimeError(f"hipModuleLaunchKernel returned rc={rc} on device {q.device}")
    return output


def _install_fused_attn_override():
    try:
        from transformer_engine.pytorch.cpp_extensions import (
            fused_attn as fused_attn_module,
        )
    except Exception as error:  # noqa: BLE001
        logger.info("TE not importable yet, deferring install (%r)", error)
        return False

    if getattr(
        fused_attn_module.fused_attn_fwd,
        "_fwd_attn_asm_patched",
        False,
    ):
        return True
    if not _try_load_kernel():
        return False

    import torch

    original_fused_attn_fwd = fused_attn_module.fused_attn_fwd
    original_signature = inspect.signature(original_fused_attn_fwd)

    @functools.wraps(original_fused_attn_fwd)
    def patched_fused_attn_fwd(*args, **kwargs):
        global _DISPATCH_COUNT

        bound = original_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        parameters = bound.arguments

        is_training = parameters["is_training"]
        max_seqlen_q = parameters["max_seqlen_q"]
        max_seqlen_kv = parameters["max_seqlen_kv"]
        q = parameters["q"]
        k = parameters["k"]
        v = parameters["v"]
        attn_bias = parameters.get("attn_bias")
        page_table_k = parameters.get("page_table_k")
        s_quantizer = parameters.get("s_quantizer")
        o_quantizer = parameters.get("o_quantizer")
        attn_scale = parameters.get("attn_scale")
        dropout = parameters.get("dropout", 0.0)
        qkv_layout = parameters.get("qkv_layout", "sbh3d")
        attn_bias_type = parameters.get("attn_bias_type", "no_bias")
        attn_mask_type = parameters.get("attn_mask_type", "padding")
        softmax_type = parameters.get("softmax_type", "vanilla")
        window_size = parameters.get("window_size", (-1, -1))
        bottom_right_diagonal = parameters.get("bottom_right_diagonal")
        softmax_offset = parameters.get("softmax_offset")
        return_max_logit = parameters.get("return_max_logit", False)

        # TE 2.12 uses False as the "no softmax offset" sentinel, while its
        # C++ binding and TE 2.15 expect None.
        if softmax_offset is False:
            softmax_offset = None
            parameters["softmax_offset"] = None

        def fallback():
            return original_fused_attn_fwd(*bound.args, **bound.kwargs)

        # Newer TE revisions added independent output and FP8 scale-inverse
        # formats. The hand-tuned BF16 kernel writes the same layout as Q and
        # does not produce FP8 scale inverses, so preserve TE for other formats.
        expected_o_format = "bshd" if qkv_layout.startswith("bshd") else "sbhd"
        if parameters.get("o_format", expected_o_format) != expected_o_format:
            return fallback()
        if parameters.get("qkv_scale_inv_format") is not None:
            return fallback()

        if not _eligible(
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            q=q,
            k=k,
            v=v,
            attn_scale=attn_scale,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            softmax_type=softmax_type,
            window_size=window_size,
            bottom_right_diagonal=bottom_right_diagonal,
            qkv_layout=qkv_layout,
            dropout=dropout,
            attn_bias=attn_bias,
            softmax_offset=softmax_offset,
            s_quantizer=s_quantizer,
            o_quantizer=o_quantizer,
            fp8=False,
        ):
            return fallback()
        if return_max_logit or page_table_k is not None:
            return fallback()

        if qkv_layout.startswith("bshd"):
            batch, sequence_length, query_heads, _ = q.shape
        else:
            sequence_length, batch, query_heads, _ = q.shape

        lse = torch.empty(
            batch,
            query_heads,
            sequence_length,
            1,
            dtype=torch.float32,
            device=q.device,
        )
        try:
            output = _launch(
                q,
                k,
                v,
                qkv_layout,
                attn_scale=attn_scale or 0.0,
                attn_mask_type=attn_mask_type,
                window_size=window_size,
                lse_out=lse,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "hand-tuned launch failed (%r); falling back to CK",
                error,
            )
            return fallback()

        _DISPATCH_COUNT += 1
        if _DISPATCH_LOG:
            logger.info(
                "fwd-attn-asm dispatched: shape=%s layout=%s mask=%s window=%s",
                tuple(q.shape),
                qkv_layout,
                attn_mask_type,
                window_size,
            )
        if not is_training:
            return output, []

        cache_key = (qkv_layout, attn_mask_type, dropout)
        template = _AUX_CTX_TEMPLATES.get(cache_key)
        if template is None:
            ck_output = None
            ck_aux = None
            try:
                ck_output, ck_aux = fallback()
                template = list(ck_aux)
            except RuntimeError as error:
                if "fused attn configs not supported" not in str(error):
                    raise
                # TE 2.15 CK rejects the exact S=8192 gfx950 configuration,
                # although its backward contract only needs softmaxStats and
                # an RNG state. Dropout is zero for GPT-OSS, so a zero state is
                # a valid placeholder and is not consumed by the kernel.
                template = [
                    lse,
                    torch.zeros(2, dtype=torch.int64, device=q.device),
                ]
                logger.info("CK aux template unavailable; synthesized LSE/RNG aux " "for dropout=0")
            _AUX_CTX_TEMPLATES[cache_key] = template
            template_lse = template[0] if template else None
            if (
                template_lse is None
                or getattr(template_lse, "ndim", -1) != lse.ndim
                or template_lse.dtype != lse.dtype
            ):
                logger.warning(
                    "CK aux LSE shape=%s dtype=%s mismatches ASM LSE "
                    "shape=%s dtype=%s; falling back to CK",
                    getattr(template_lse, "shape", None),
                    getattr(template_lse, "dtype", None),
                    lse.shape,
                    lse.dtype,
                )
                _AUX_CTX_TEMPLATES.pop(cache_key, None)
                return ck_output, ck_aux

        aux = [lse] + list(template[1:])
        return output, aux

    patched_fused_attn_fwd._fwd_attn_asm_patched = True
    fused_attn_module.fused_attn_fwd = patched_fused_attn_fwd

    original_fused_attn_bwd = fused_attn_module.fused_attn_bwd

    def diagnostic_fused_attn_bwd(*args, **kwargs):
        if _DISPATCH_LOG:
            logger.info(
                "fused-attn-bwd config: q_shape=%s q_stride=%s k_stride=%s "
                "v_stride=%s out_stride=%s dout_stride=%s",
                getattr(args[4], "shape", None) if len(args) > 4 else None,
                args[4].stride() if len(args) > 4 else None,
                args[5].stride() if len(args) > 5 else None,
                args[6].stride() if len(args) > 6 else None,
                args[7].stride() if len(args) > 7 else None,
                args[8].stride() if len(args) > 8 else None,
            )
        try:
            return original_fused_attn_bwd(*args, **kwargs)
        except RuntimeError:
            aux = args[11] if len(args) > 11 else kwargs.get("aux_ctx_tensors")
            logger.error(
                "fused-attn-bwd rejected config: max_q=%s max_kv=%s "
                "q=%s backend=%s aux=%s layout=%s mask=%s window=%s "
                "bottom_right=%s deterministic=%s",
                args[0] if len(args) > 0 else None,
                args[1] if len(args) > 1 else None,
                getattr(args[4], "shape", None) if len(args) > 4 else None,
                args[12] if len(args) > 12 else None,
                [(tuple(t.shape), str(t.dtype)) for t in (aux or [])],
                args[21] if len(args) > 21 else None,
                args[23] if len(args) > 23 else None,
                args[25] if len(args) > 25 else None,
                args[26] if len(args) > 26 else None,
                args[27] if len(args) > 27 else None,
            )
            raise

    diagnostic_fused_attn_bwd._fwd_attn_asm_bwd_diagnostic = True
    fused_attn_module.fused_attn_bwd = diagnostic_fused_attn_bwd

    try:
        from transformer_engine.pytorch.attention.dot_product_attention import backends

        backends.fused_attn_fwd = patched_fused_attn_fwd
        backends.fused_attn_bwd = diagnostic_fused_attn_bwd
    except Exception:
        pass

    logger.info("fused_attn_fwd patched " "(D=64 BF16 [SWA-]causal -> hand-tuned hd64 kernel)")
    return True


def _install_aiter_rope_override(rope_module):
    """Restore the AITER RoPE route removed by newer ROCm TE revisions."""
    if not _AITER_ROPE_ENABLED:
        return False

    fused_rope = rope_module.FusedRoPEFunc
    if getattr(fused_rope, "_mlperf_aiter_rope_patched", False):
        return True
    # Older TE revisions already provide the same dispatch natively.
    if hasattr(fused_rope, "_can_use_aiter"):
        return True

    from aiter.ops.rope import rope_bwd as aiter_rope_bwd
    from aiter.ops.rope import rope_fwd as aiter_rope_fwd

    original_forward = fused_rope.forward
    original_backward = fused_rope.backward

    def aiter_aware_forward(
        ctx,
        tensor,
        freqs,
        start_positions=None,
        tensor_format="sbhd",
        interleaved=False,
        cu_seqlens=None,
        cp_size=1,
        cp_rank=0,
    ):
        use_aiter = (
            tensor_format == "sbhd"
            and not interleaved
            and cu_seqlens is None
            and cp_size == 1
            and start_positions is None
        )
        ctx._mlperf_use_aiter_rope = use_aiter
        if not use_aiter:
            return original_forward(
                ctx,
                tensor,
                freqs,
                start_positions,
                tensor_format,
                interleaved,
                cu_seqlens,
                cp_size,
                cp_rank,
            )

        if freqs.dtype != torch.float32:
            freqs = freqs.float()
        output = aiter_rope_fwd(tensor, freqs, 0, False, False)
        ctx.save_for_backward(freqs, cu_seqlens, start_positions)
        return output

    def aiter_aware_backward(ctx, grad_output):
        if not getattr(ctx, "_mlperf_use_aiter_rope", False):
            return original_backward(ctx, grad_output)
        freqs, _, _ = ctx.saved_tensors
        grad_input = aiter_rope_bwd(grad_output, freqs, 0, False, False)
        return grad_input, None, None, None, None, None, None, None, None

    import torch

    fused_rope.forward = staticmethod(aiter_aware_forward)
    fused_rope.backward = staticmethod(aiter_aware_backward)
    fused_rope._mlperf_aiter_rope_patched = True
    logger.info("restored AITER fused RoPE dispatch for this TE revision")
    return True


class _DeferredInstaller:
    _TARGETS = {
        "transformer_engine.pytorch.attention.rope",
        "transformer_engine.pytorch.cpp_extensions.fused_attn",
        "transformer_engine.pytorch.cpp_extensions",
    }

    def find_spec(self, fullname, path, target=None):
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
                original_loader = spec.loader

                class _WrappedLoader:
                    def create_module(self, module_spec):
                        if hasattr(original_loader, "create_module"):
                            return original_loader.create_module(module_spec)
                        return None

                    def exec_module(self, module):
                        original_loader.exec_module(module)
                        if fullname == "transformer_engine.pytorch.cpp_extensions.fused_attn":
                            try:
                                _install_fused_attn_override()
                            except Exception as error:  # noqa: BLE001
                                logger.warning(
                                    "deferred install failed: %r",
                                    error,
                                )
                        elif fullname == "transformer_engine.pytorch.attention.rope":
                            try:
                                _install_aiter_rope_override(module)
                            except Exception as error:  # noqa: BLE001
                                logger.warning(
                                    "deferred AITER RoPE install failed: %r",
                                    error,
                                )

                spec.loader = _WrappedLoader()
                return spec
            return None
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "fwd-attn-asm find_spec error for %s: %r",
                fullname,
                error,
            )
            return None


def _register_deferred_install():
    if any(isinstance(finder, _DeferredInstaller) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _DeferredInstaller())
    logger.info("deferred installer registered; will patch on TE load")


if _ENABLED or _AITER_ROPE_ENABLED:
    try:
        _register_deferred_install()
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "fwd-attn-asm deferred install failed at startup: %r",
            error,
        )
