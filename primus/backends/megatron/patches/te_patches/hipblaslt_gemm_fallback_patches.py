###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""TE hipBLASLt GEMM failure fallback.

PROBLEM
-------
On MI355X (gfx950) the ``mamba_370M`` pretrain dies in the first backward
pass with::

    File ".../transformer_engine/pytorch/module/layernorm_linear.py", in backward
      wgrad, grad_bias_ = wgrad_gemm(ln_out_total, grad_output)
    RuntimeError: .../common/gemm/rocm_gemm.hip:1934 in function
      hipblaslt_gemm: HIPBLASLT Error: 6

``6`` is ``HIPBLAS_STATUS_INTERNAL_ERROR``, raised by TE's own status check
around ``hipblasLtMatmul``. What is known about it:

* The failing call is the Mamba ``in_proj`` **wgrad** GEMM, layout ``NT``,
  ``m=1024 (hidden) n=4384 (fused in_proj width) k=65536 (tokens)``.
* The ``fprop`` GEMM of the very same layer succeeds; wgrad is the only one
  whose ``K`` is the token count (``micro_batch_size * seq_length``) rather
  than ``hidden_size``.
* ``hipblaslt-bench`` runs the exact problem TE dumps -- same shapes, layout
  and solution index -- and validates, so the Tensile kernel is fine.
* A standalone ``transformer_engine.pytorch.LayerNormLinear`` (no Megatron,
  no Primus) fails the same way for *every* output width at 65536 tokens,
  which rules out the gfx950 fused-width "dead zone" that
  ``kimi_delta_attention.py`` pads around.

So the defect is in TE's ROCm GEMM call path for large-``K`` wgrad, not in
the problem geometry and not in Primus. TE picks the first heuristic result
without re-checking its workspace requirement (``bestAlgo = firstAlgo`` when
tuning is disabled, which is how Primus runs it), and large-``K`` wgrad is
exactly where hipBLASLt wants split-K solutions that need more workspace
than the 64 MiB TE allocates for gfx950.

SOLUTION
--------
Wrap ``transformer_engine.pytorch.cpp_extensions.general_gemm`` so a
hipBLASLt status failure is recovered instead of killing the run. On the
first failure the wrapper escalates:

1. **Grow TE's hipBLASLt workspace once** and retry the same call. If the
   failure is workspace starvation this keeps the GEMM on hipBLASLt at full
   speed, and the log line tells us the workspace hypothesis was right.
2. **Recompute the GEMM with torch** (``torch.matmul`` plus the bias
   epilogue) when the retry still fails. The failing shape is remembered so
   subsequent calls go straight to torch rather than paying for an
   exception per layer per step -- TE caches its chosen algorithm, so a
   shape that failed once fails every time.

Fallback (2) only accepts calls it can reproduce exactly: plain 2D
bf16/fp16/fp32 operands, no quantization, no gelu epilogue, no
Userbuffers/comm-overlap, ``alpha=1``, ``beta in (0, 1)``. Anything else
re-raises the original TE error, so no FP8/MXFP4 or TP-overlap GEMM is ever
silently rerouted.

NUMERICS
--------
Fallback (2) is the same math: ``torch.matmul`` accumulates bf16 inputs in
fp32 exactly like hipBLASLt. The one difference is that a fused
``gradient_accumulation_fusion`` wgrad (fp32 ``main_grad`` output from bf16
operands) rounds the product to bf16 before accumulating, where TE writes
the fp32 accumulator straight into ``main_grad``. That configuration is
flagged in the log when it happens.

CONFIGURATION
-------------
Enabled by default on ROCm; it cannot change a run that never raises a
hipBLASLt error, because the fast path is the unmodified TE call.

    1. ``te_hipblaslt_gemm_fallback: false`` in the EXP YAML ``overrides:``
       block disables it (get the raw TE crash back).
    2. ``PRIMUS_TE_HIPBLASLT_GEMM_FALLBACK=0`` env var, same effect, for
       ad-hoc runs.
    3. ``PRIMUS_TE_HIPBLASLT_WORKSPACE_MIB=<N>`` sets the retry workspace
       (default 128 MiB, up from TE's 64 MiB on gfx950). Note that grouped
       GEMM allocates one workspace of this size per cuBLAS stream, so
       raising it far above the default costs memory on MoE runs. ``0``
       skips step (1) and goes straight to the torch fallback.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch

from primus.core.patches import PatchContext, get_param, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH_ID = "megatron.te.hipblaslt_gemm_fallback"
_WRAPPED_FLAG = "_primus_hipblaslt_gemm_fallback"

_DEFAULT_RETRY_WORKSPACE_MIB = 128

# TE raises through NVTE_CHECK_HIPBLASLT, which stringifies the hipblasStatus_t
# as "HIPBLASLT Error: <n>" and prefixes the rocm_gemm source location.
_HIPBLASLT_ERROR_MARKERS = ("HIPBLASLT Error", "hipblaslt_gemm")

_SUPPORTED_FALLBACK_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# Shapes whose TE GEMM is known to fail; routed straight to the torch fallback.
_broken_gemms: Dict[Tuple, bool] = {}
_workspace_grown = False
_blas_preference_settable = True


def _is_hipblaslt_error(exc: BaseException) -> bool:
    """Whether an exception is TE's hipBLASLt status check firing."""
    message = str(exc)
    return any(marker in message for marker in _HIPBLASLT_ERROR_MARKERS)


def _retry_workspace_bytes() -> int:
    """Workspace size for the step-1 retry, in bytes (0 disables the retry)."""
    raw = os.environ.get("PRIMUS_TE_HIPBLASLT_WORKSPACE_MIB")
    if raw is None:
        return _DEFAULT_RETRY_WORKSPACE_MIB * 1024 * 1024
    try:
        return max(int(raw), 0) * 1024 * 1024
    except (TypeError, ValueError):
        warning_rank_0(
            f"[Patch:{_PATCH_ID}] invalid PRIMUS_TE_HIPBLASLT_WORKSPACE_MIB={raw!r}, "
            f"using {_DEFAULT_RETRY_WORKSPACE_MIB} MiB"
        )
        return _DEFAULT_RETRY_WORKSPACE_MIB * 1024 * 1024


def _grow_te_workspace() -> bool:
    """Enlarge TE's hipBLASLt workspace for every subsequent GEMM.

    TE sizes the workspace from ``get_cublas_workspace_size_bytes()`` and
    caches the allocation with ``functools.lru_cache``, so raising the size
    means overriding that function and dropping the cached buffer. Returns
    False when the workspace was already grown, when the retry is disabled,
    or when this TE revision does not expose the helpers.
    """
    global _workspace_grown

    if _workspace_grown:
        return False

    target = _retry_workspace_bytes()
    if target == 0:
        return False

    try:
        from transformer_engine.pytorch.cpp_extensions import gemm as te_gemm
    except ImportError:
        return False

    size_fn = getattr(te_gemm, "get_cublas_workspace_size_bytes", None)
    workspace_fn = getattr(te_gemm, "get_cublas_workspace", None)
    if size_fn is None or workspace_fn is None or not hasattr(workspace_fn, "cache_clear"):
        return False

    current = size_fn()
    if current is None or current >= target:
        return False

    te_gemm.get_cublas_workspace_size_bytes = lambda: target
    workspace_fn.cache_clear()
    _workspace_grown = True
    log_rank_0(
        f"[Patch:{_PATCH_ID}] grew TE hipBLASLt workspace "
        f"{current // (1024 * 1024)} MiB -> {target // (1024 * 1024)} MiB and retrying the GEMM"
    )
    return True


def _make_binder(original):
    """Build a resolver from a ``general_gemm`` call to a name -> value mapping.

    ``general_gemm`` is called with keywords everywhere in TE, but Primus
    ships its own variant with a positional ``workspace``, so bind against
    the real signature instead of reading ``kwargs``. The signature is
    inspected once here because binding happens on the recovery path, which
    runs per GEMM once a shape is known to be broken.
    """
    signature = inspect.signature(original)
    operand_names = list(signature.parameters)[:2]

    def bind(A, B, args, kwargs) -> Optional[Dict[str, Any]]:
        try:
            bound = signature.bind(A, B, *args, **kwargs)
        except (TypeError, ValueError):
            return None
        bound.apply_defaults()
        params = dict(bound.arguments)
        params["A"] = params.pop(operand_names[0])
        params["B"] = params.pop(operand_names[1])
        return params

    return bind


def _is_plain_operand(tensor: Any) -> bool:
    """Whether a GEMM operand is a dense tensor the torch fallback can use."""
    return (
        type(tensor) in (torch.Tensor, torch.nn.Parameter)
        and tensor.dim() == 2
        and tensor.dtype in _SUPPORTED_FALLBACK_DTYPES
    )


def _fallback_unsupported_reason(params: Dict[str, Any]) -> Optional[str]:
    """Return why ``params`` cannot be recomputed with torch, or None."""
    layout = params.get("layout", "TN")
    if layout not in ("TN", "NN", "NT"):
        return f"layout={layout}"
    if not _is_plain_operand(params.get("A")) or not _is_plain_operand(params.get("B")):
        return "non-dense or non-2D operands (quantized GEMM)"
    if params.get("quantization_params") is not None:
        return "quantization_params set"
    if params.get("gelu") or params.get("gelu_in") is not None:
        return "gelu epilogue"
    if params.get("ub") is not None or params.get("ub_type") is not None:
        return "Userbuffers comm overlap"
    if params.get("extra_output") is not None or params.get("bulk_overlap"):
        return "comm-overlap extra output"
    if params.get("alpha", 1.0) not in (1.0, None):
        return f"alpha={params.get('alpha')}"
    if params.get("beta", None) not in (0.0, 1.0, None):
        return f"beta={params.get('beta')}"
    out_dtype = params.get("out_dtype")
    if out_dtype is not None and out_dtype not in _SUPPORTED_FALLBACK_DTYPES:
        return f"out_dtype={out_dtype}"
    bias = params.get("bias")
    if bias is not None:
        # grad=True asks the epilogue for dbias (wgrad, layout NT); grad=False
        # adds bias to the product (fprop, layout TN). Other combinations do
        # not appear in TE and are not worth guessing at.
        if params.get("grad") and layout != "NT":
            return f"dbias epilogue with layout={layout}"
        if not params.get("grad") and layout != "TN":
            return f"bias epilogue with layout={layout}"
    return None


@contextlib.contextmanager
def _prefer_non_hipblaslt_blas():
    """Route the fallback's matmul away from the backend that just failed.

    torch can dispatch GEMMs to hipBLASLt too, so ask for the hipBLAS /
    rocBLAS path while recomputing. Does nothing on builds where the
    preference cannot be set.
    """
    global _blas_preference_settable

    previous = None
    if _blas_preference_settable:
        try:
            previous = torch.backends.cuda.preferred_blas_library()
            torch.backends.cuda.preferred_blas_library("cublas")
        except Exception:
            _blas_preference_settable = False
            previous = None
    try:
        yield
    finally:
        if previous is not None:
            try:
                torch.backends.cuda.preferred_blas_library(previous)
            except Exception:
                pass


def _torch_matmul(A: torch.Tensor, B: torch.Tensor, layout: str) -> torch.Tensor:
    """Row-major equivalent of TE's column-major ``D = op(A) op(B)``.

    TE hands hipBLASLt column-major views of row-major torch tensors, so the
    row-major result is ``op(B) op(A)`` with the transposes swapped: ``TN``
    is fprop (``inp @ weight.T``), ``NN`` dgrad (``dgrad_out @ weight``) and
    ``NT`` wgrad (``dy.T @ x``).
    """
    if layout == "TN":
        return B @ A.transpose(0, 1)
    if layout == "NN":
        return B @ A
    return B.transpose(0, 1) @ A


def _torch_general_gemm(params: Dict[str, Any]) -> Tuple[Any, Any, None, None]:
    """Recompute a ``general_gemm`` call with torch ops.

    Mirrors ``general_gemm``'s return contract:
    ``(out, bias_grad, gelu_input, extra_output)``.
    """
    A, B = params["A"], params["B"]
    layout = params.get("layout", "TN")
    bias = params.get("bias")
    grad = bool(params.get("grad"))
    out = params.get("out")
    accumulate = bool(params.get("accumulate"))
    out_dtype = params.get("out_dtype")

    bias_grad = None
    with _prefer_non_hipblaslt_blas():
        result = _torch_matmul(A, B, layout)
        if bias is not None:
            if grad:
                # dbias sums grad_output over the token (K) dimension.
                bias_grad = B.sum(dim=0).to(bias.dtype)
            else:
                result = result + bias.to(result.dtype)

    if out is None:
        if out_dtype is not None and out_dtype != result.dtype:
            result = result.to(out_dtype)
        return result, bias_grad, None, None

    if accumulate:
        out.add_(result)
    else:
        out.copy_(result)
    return out, bias_grad, None, None


def _gemm_key(A, B, kwargs) -> Optional[Tuple]:
    """Identity of a GEMM problem, for remembering which ones TE cannot run.

    Built from the raw call rather than the bound arguments so that looking a
    problem up stays cheap enough for the per-GEMM path. A call site that
    passes ``layout``/``grad`` positionally simply misses the lookup and
    pays for TE's exception again, which is correct if slower.
    """
    shape_a = getattr(A, "shape", None)
    shape_b = getattr(B, "shape", None)
    if shape_a is None or shape_b is None:
        return None
    return (
        tuple(shape_a),
        tuple(shape_b),
        getattr(A, "dtype", None),
        getattr(B, "dtype", None),
        kwargs.get("layout", "TN"),
        bool(kwargs.get("grad", False)),
    )


def _describe(params: Dict[str, Any]) -> str:
    """Human-readable GEMM problem description for log lines."""
    A, B = params["A"], params["B"]
    return (
        f"layout={params.get('layout', 'TN')} "
        f"A={getattr(A, 'shape', None)} B={getattr(B, 'shape', None)} "
        f"dtype={getattr(A, 'dtype', None)} grad={bool(params.get('grad'))}"
    )


def _make_fallback_general_gemm(original):
    """Wrap ``general_gemm`` with the hipBLASLt failure recovery path."""

    bind = _make_binder(original)

    def general_gemm(A, B, *args, **kwargs):
        if _broken_gemms and _broken_gemms.get(_gemm_key(A, B, kwargs)):
            # The key only covers shapes and layout, so re-check the rest of
            # the call before rerouting: the same shape may show up with an
            # epilogue or comm overlap the fallback must not touch.
            params = bind(A, B, args, kwargs)
            if params is not None and _fallback_unsupported_reason(params) is None:
                return _torch_general_gemm(params)

        try:
            return original(A, B, *args, **kwargs)
        except RuntimeError as exc:
            if not _is_hipblaslt_error(exc):
                raise

            if _grow_te_workspace():
                try:
                    return original(A, B, *args, **kwargs)
                except RuntimeError as retry_exc:
                    if not _is_hipblaslt_error(retry_exc):
                        raise

            params = bind(A, B, args, kwargs)
            if params is None:
                raise

            reason = _fallback_unsupported_reason(params)
            if reason is not None:
                warning_rank_0(
                    f"[Patch:{_PATCH_ID}] hipBLASLt GEMM failed and cannot be recomputed "
                    f"with torch ({reason}); {_describe(params)}"
                )
                raise

            key = _gemm_key(A, B, kwargs)
            if key is not None:
                _broken_gemms[key] = True
            warning_rank_0(
                f"[Patch:{_PATCH_ID}] hipBLASLt GEMM failed ({exc}); "
                f"recomputing with torch and routing this shape to torch for the rest of "
                f"the run: {_describe(params)}"
            )
            out = params.get("out")
            if out is not None and out.dtype != params["A"].dtype:
                warning_rank_0(
                    f"[Patch:{_PATCH_ID}] this GEMM accumulates {params['A'].dtype} operands "
                    f"into a {out.dtype} output (fused wgrad accumulation); the fallback "
                    "rounds the product before accumulating"
                )
            return _torch_general_gemm(params)

    setattr(general_gemm, _WRAPPED_FLAG, True)
    setattr(general_gemm, "__wrapped__", original)
    general_gemm.__name__ = getattr(original, "__name__", "general_gemm")
    general_gemm.__doc__ = getattr(original, "__doc__", None)
    return general_gemm


def _fallback_enabled(ctx: PatchContext) -> bool:
    """YAML ``te_hipblaslt_gemm_fallback`` > env var > enabled on ROCm."""
    if getattr(torch.version, "hip", None) is None:
        return False

    yaml_value = get_param(ctx, "te_hipblaslt_gemm_fallback", None)
    if yaml_value is not None:
        return bool(yaml_value)

    env_value = os.environ.get("PRIMUS_TE_HIPBLASLT_GEMM_FALLBACK")
    if env_value is not None:
        return env_value.strip().lower() not in ("0", "false", "off", "no")

    return True


@register_patch(
    _PATCH_ID,
    backend="megatron",
    phase="before_train",
    description=(
        "Recover from TE hipBLASLt GEMM status failures (HIPBLASLT Error: 6 in the "
        "large-K wgrad path on gfx950) by retrying with a larger workspace and then "
        "recomputing the GEMM with torch"
    ),
    condition=_fallback_enabled,
    tags=["rocm", "transformer_engine"],
)
def patch_te_hipblaslt_gemm_fallback(ctx: PatchContext):
    """Rebind every ``general_gemm`` reference to the fallback wrapper.

    TE modules do ``from ..cpp_extensions import general_gemm`` at import
    time, so patching the definition alone would leave the already-imported
    call sites (``linear``, ``layernorm_linear``, ``layernorm_mlp``, ...) on
    the original. Rebind by identity everywhere instead, including the
    defining module so later imports also pick up the wrapper.
    """
    del ctx

    try:
        from transformer_engine.pytorch.cpp_extensions import gemm as te_gemm
    except ImportError as exc:
        log_rank_0(f"[Patch:{_PATCH_ID}] TE cpp_extensions unavailable ({exc}); patch skipped")
        return

    original = getattr(te_gemm, "general_gemm", None)
    if original is None:
        log_rank_0(f"[Patch:{_PATCH_ID}] TE has no general_gemm; patch skipped")
        return
    if getattr(original, _WRAPPED_FLAG, False):
        return

    wrapper = _make_fallback_general_gemm(original)
    te_gemm.general_gemm = wrapper
    rebound = [te_gemm.__name__]

    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith(("transformer_engine", "megatron", "primus")):
            continue
        try:
            bound_gemm = getattr(module, "general_gemm", None)
        except Exception:
            # Lazy-import shims can raise from __getattr__; they cannot be
            # holding a reference to the original function anyway.
            continue
        if bound_gemm is original:
            setattr(module, "general_gemm", wrapper)
            rebound.append(module_name)

    log_rank_0(
        f"[Patch:{_PATCH_ID}] wrapped general_gemm in {len(rebound)} module(s): "
        f"{', '.join(sorted(rebound))}"
    )
