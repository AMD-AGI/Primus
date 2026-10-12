"""FP8 forward + dgrad for LoRA-frozen base weights after healing.

Healing installs the FP8 stash pre_quantize_mxfp4 built at load: one E4M3 direct cast per
weight, scale 1, row-wise only. Under the Turbo FP8 autocast the Turbo linears hand
``gemm_fp8`` a weight pair from ``_maybe_create_quantized_weight_buffers``; for these
weights the hook in llama2_custom.py returns ``frozen_fp8_weight`` of that stash, and
``install()`` routes the GEMM here:

  forward   x  -> per-tensor E4M3 quantize -> FlyDSL tensorwise GEMM, NT
  backward  dy -> per-tensor E5M2 quantize -> FlyDSL tensorwise GEMM, NN (the stash as-is,
            no transpose)

E4M3 forward and E5M2 backward is TE's HYBRID format, which the healing DelayedScaling
recipe uses. Scales differ: Turbo's quantize takes the current tensor's amax where TE's
DelayedScaling takes the amax history (length 4, most_recent). No wgrad: LoRA freezes the
base weights. Measured at 9.34 ms per layer against TE's 9.94 ms, same forward and dgrad
SNR.
"""

import os

import torch

_STATE = {"installed": False, "frozen_calls": 0, "stock_calls": 0}


def _log(msg: str) -> None:
    if os.environ.get("TURBO_DEBUG", "0") in ("1", "true", "True"):
        try:
            with open("/results/turbo_decide.log", "a") as fh:
                fh.write(msg + "\n")
        except Exception:
            pass


def frozen_scale_inv(w):
    """The per-tensor scale_inv of a tensor ``frozen_fp8_weight`` returned, else None."""
    return getattr(w, "_frozen_fp8_scale_inv", None) if isinstance(w, torch.Tensor) else None


def frozen_fp8_weight(weight):
    """The [N, K] float8_e4m3fn data of a frozen TE Float8Tensor (per-tensor E4M3), else None.

    A plain tensor tagged with its scale_inv, because the Turbo linears store it in
    ``quantized_weight_buffer``, a registered buffer that only takes tensors.
    """
    cached = getattr(weight, "_turbo_fp8_frozen", None)
    if cached is not None:
        return cached
    if getattr(weight, "requires_grad", True):
        return None
    data = getattr(weight, "_data", None)
    scale_inv = getattr(weight, "_scale_inv", None)
    if data is None or scale_inv is None or data.ndim != 2 or scale_inv.numel() != 1:
        return None
    if data.dtype == torch.uint8:
        data = data.view(torch.float8_e4m3fn)
    if data.dtype != torch.float8_e4m3fn or not data.is_cuda:
        return None
    w = data.view(torch.float8_e4m3fn)
    w._frozen_fp8_scale_inv = scale_inv.reshape(1).float()
    try:
        weight._turbo_fp8_frozen = w
    except Exception:
        pass
    return w


class _FrozenWeightFP8Gemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, w_s, out_dtype):
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.core.low_precision import ScalingGranularity
        from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl
        from primus_turbo.pytorch.ops.quantization import quantize_fp8

        tw = ScalingGranularity.TENSORWISE
        xq, xs = quantize_fp8(x, torch.float8_e4m3fn, tw)
        out = gemm_fp8_impl(xq, xs, False, w, w_s, True, out_dtype, False, tw.value, BackendType.FLYDSL.value)
        ctx.save_for_backward(w, w_s)
        ctx.out_dtype = out_dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.core.low_precision import ScalingGranularity
        from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl
        from primus_turbo.pytorch.ops.quantization import quantize_fp8

        w, w_s = ctx.saved_tensors
        tw = ScalingGranularity.TENSORWISE
        gq, gs = quantize_fp8(grad_out.contiguous(), torch.float8_e5m2, tw)
        grad_x = gemm_fp8_impl(gq, gs, False, w, w_s, False, ctx.out_dtype, False, tw.value, BackendType.FLYDSL.value)
        return grad_x, None, None, None


def frozen_fp8_linear(x, w, out_dtype=torch.bfloat16):
    return _FrozenWeightFP8Gemm.apply(x.contiguous(), w, frozen_scale_inv(w), out_dtype)


def warmup(shapes, m: int) -> None:
    """Compile the FlyDSL FP8 GEMMs (NT forward, NN dgrad) for each (N, K) at M rows.

    They are first used at healing, and compiling them there cost 335 s inside the timed
    region of a 20-step smoke; the compile cache lives in the container and is lost with it.
    """
    for n, k in sorted(set(shapes)):
        w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02).to(torch.float8_e4m3fn)
        w._frozen_fp8_scale_inv = torch.ones(1, device="cuda", dtype=torch.float32)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        frozen_fp8_linear(x, w).backward(torch.ones(m, n, device="cuda", dtype=torch.bfloat16))
        del w, x
    torch.cuda.synchronize()
    _log(f"[FrozenFP8] warmed up FlyDSL FP8 GEMMs for M={m}, (N, K)={sorted(set(shapes))}")


def install() -> bool:
    """Route ``gemm_fp8`` calls whose weight came from ``frozen_fp8_weight`` through FlyDSL."""
    if _STATE["installed"]:
        return True
    import primus_turbo.pytorch.ops as turbo_ops
    from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensorPair

    stock = turbo_ops.gemm_fp8

    @torch._dynamo.disable
    def gemm_fp8(a, b, trans_a=False, trans_b=False, out_dtype=None, config=None, fuse_bgrad_accum_pattern=None):
        w = b.data if isinstance(b, QuantizedTensorPair) else b
        frozen = frozen_scale_inv(w) is not None
        if (
            frozen
            and not trans_a
            and trans_b
            and fuse_bgrad_accum_pattern is None
            and isinstance(a, torch.Tensor)
            and a.ndim == 2
            and a.dtype in (torch.bfloat16, torch.float16)
        ):
            _STATE["frozen_calls"] += 1
            if _STATE["frozen_calls"] == 1:
                _log(f"[FrozenFP8] frozen path: x={tuple(a.shape)} w={tuple(w.shape)}")
            return frozen_fp8_linear(a, w, out_dtype or a.dtype)
        if frozen:
            raise RuntimeError(
                f"Frozen FP8 weight reached gemm_fp8 outside the frozen path: trans_a={trans_a} "
                f"trans_b={trans_b} fuse={fuse_bgrad_accum_pattern} a={getattr(a, 'dtype', type(a))}"
            )
        _STATE["stock_calls"] += 1
        if _STATE["stock_calls"] == 1:
            _log(f"[FrozenFP8] stock gemm_fp8 for b={type(b).__name__}")
        return stock(a, b, trans_a, trans_b, out_dtype, config, fuse_bgrad_accum_pattern)

    gemm_fp8.__wrapped__ = stock
    turbo_ops.gemm_fp8 = gemm_fp8
    _STATE["installed"] = True
    return True
