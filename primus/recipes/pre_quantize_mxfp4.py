###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
PRE_QUANTIZED_MODEL=True parity with NeMo MLPerf 6.0 MI355X FP4.

Recipe-local pre-quantization wiring. The three byte-equivalent NeMo
reference functions (``_extract_module``, ``_get_quantized_params_cpu``,
``_pre_quantize_model``) are ports of:

  ``mlperf-training-6-0/llama2_sft/nemo/src/callbacks/custom_callbacks.py``
  lines 57-62 (``_extract_module``), 122-149 (``_pre_quantize_model``),
  151-212 (``_get_quantized_params_cpu``).

Adaptations vs NeMo (NOT behavioural changes):

* NeMo reads ``self.first_last_layers_bf16``,
  ``self.num_layers_at_start_in_bf16``, ``self.num_layers_at_end_in_bf16``,
  ``self.store_quantized_params_on_gpu``, ``self.fp8_quantizer``,
  ``self.mxfp4_quantizer`` from a ``CustomCallback`` instance bound to
  the Lightning trainer. Primus has no such instance, so these are read
  from environment variables with NeMo's *default* values
  (``False`` / ``0`` / ``False``) and the FP8/MXFP4 quantizers are
  module-locals built inside ``_pre_quantize_model``.
* NeMo passes ``trainer.model`` (single Lightning module, already
  unwrapped from MegatronParallel by PL). Primus passes a
  ``List[ModelChunk]`` from megatron-bridge; ``_extract_module`` strips
  the list wrapper before stripping ``.module`` chains, so the input
  semantically matches NeMo's.

Integration: ``install_pre_quantize_wrap(orig_train)`` returns a drop-in
wrapper around ``megatron.bridge.training.train.train`` that, on the
first call, stashes FP8 weights on CPU, swaps live weights to MXFP4, and
bridges the stash to ``primus.recipes.mxfp4_healing._ORDERED_FP8_STASH``
before delegating to ``orig_train``. When ``PRE_QUANTIZED_MODEL`` is not
enabled the wrap is a no-op (it just returns ``orig_train`` unchanged).

Historical note: this code used to live under
``primus.backends.megatron.patches.te_patches.pre_quantize_mxfp4_patches``
as a ``@register_patch(phase="before_train")`` module that fired during
``MegatronBridgeBaseTrainer.__init__``. Because the patch ran *before*
the recipe was imported, a separate wrap-aware setter mechanism
(``_primus_set_orig_train``) was needed to inject the recipe's own
``megatron_bridge_train_override`` into the already-installed wrapper.
By moving the install into the recipe itself we can wrap the override
directly at recipe import time, deleting that plumbing entirely.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, List

from primus.modules.module_utils import log_rank_0

_LOG = logging.getLogger(__name__)

_TAG = "[pre_quantize_mxfp4]"

# Layered FP8 CPU stash, populated by ``_pre_quantize_model``.
# Format mirrors NeMo's ``self.fp8_cpu_params``:
#   List[List[Float8Tensor]]
# indexed by ``decoder.layers`` index, then by intra-layer iteration
# order over ``isinstance(m, (te.Linear, te.LayerNormLinear))`` modules.
# Same ``Float8Tensor`` objects are also referenced from
# ``primus.recipes.mxfp4_healing._ORDERED_FP8_STASH`` (flat
# ``(module, fp8_tensor)`` view) for restore-time bookkeeping.
_FP8_CPU_PARAMS: List[List[Any]] = []


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def is_pre_quantized_enabled() -> bool:
    """Single switch for the pre-quantize path (``PRE_QUANTIZED_MODEL``)."""
    return _truthy_env("PRE_QUANTIZED_MODEL", default=False)


# ---------------------------------------------------------------------------
# NeMo MLPerf 6.0 MI355X FP4 byte-equivalent pre-quantization functions.
#
# Source of truth: the actual NeMo container code, NOT the older host-tree
# mirror. Inside the running ``nemo_mxfp4_lora`` Docker image:
#   /workspace/code/src/callbacks/custom_callbacks.py
#   - _extract_module                : lines 57-62
#   - _pre_quantize_model            : lines 122-148
#   - _get_quantized_params_cpu      : lines 150-213
# The host-tree file under ``mlperf-training-6-0/`` is 25 lines shorter
# (660 vs 685) and out of date in three places: the MXFP4Quantizer kwargs,
# the ``_columnwise_data`` assertion, and the CPU-pin/GPU-restore
# columnwise-None guard. We mirror the *container* code.
# ---------------------------------------------------------------------------


def _extract_module(model):
    """Unwrap the model from MegatronParallel / DDP wrappers to get the GPT module.

    NeMo (``custom_callbacks.py:57-62``)::

        m = model
        while hasattr(m, "module"):
            m = m.module
        return m

    Adaptation: Primus passes ``model`` as ``List[ModelChunk]`` from
    megatron-bridge (one chunk per pipeline stage). Pick chunk 0 then
    follow ``.module`` exactly as NeMo does on the unwrapped trainer
    model.
    """
    m = model
    if isinstance(m, (list, tuple)):
        m = m[0]
    while hasattr(m, "module"):
        m = m.module
    return m


def _pre_quantize_model(model):
    """Pre-quantize model: store FP8 clones on CPU for healing, replace weights with MXFP4.

    Byte-equivalent to NeMo
    ``custom_callbacks.CustomCallback._pre_quantize_model``
    (``mlperf-training-6-0/llama2_sft/nemo/src/callbacks/custom_callbacks.py:122-149``).
    """
    import torch
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer

    global _FP8_CPU_PARAMS

    device = next(_extract_module(model).parameters()).device

    # Clone current FP8 weights to CPU for healing.
    # Must happen BEFORE MXFP4 replacement since we reference the live FP8 params.
    #
    # ``columnwise=<ENABLE_TRANSPOSE_CACHE>`` mirrors NeMo's
    # ``custom_callbacks.py::_pre_quantize_model`` (line 205). When the env
    # is set, ``Float8Quantizer`` allocates a columnwise (transpose) buffer
    # alongside the rowwise data so the post-healing FP8 GEMMs can reuse a
    # precomputed transpose. When it is off, no transpose is kept -- and
    # ``_restore_fp8_weights_to_gpu`` / ``_pre_ttt_fp8_warmup`` additionally
    # set ``keep_fp8_weight_transpose_cache=False`` on each TE Linear so TE
    # recomputes the transpose on demand instead of dereferencing a cache
    # entry that was never populated.
    #
    # Default ``False`` matches NeMo's ``_env_enabled`` helper (returns False
    # for missing / empty env). The Primus MLPerf-parity shell
    # (``setup_llama2_70b_lora_fp4_training.sh``) exports
    # ``ENABLE_TRANSPOSE_CACHE=1`` so the reference run gets the fast path.
    _enable_transpose_cache = _truthy_env("ENABLE_TRANSPOSE_CACHE", default=False)
    fp8_quantizer = Float8Quantizer(
        scale=torch.ones(1, dtype=torch.float32, device=device),
        amax=torch.zeros(1, dtype=torch.float32, device=device),
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=_enable_transpose_cache,
    )
    _FP8_CPU_PARAMS = _get_quantized_params_cpu(
        model, fp8_quantizer, "FP8_DS"
    )

    _use_hadamard = os.environ.get("NVTE_MXFP4_USE_HADAMARD", "0") == "1"

    mxfp4_quantizer = MXFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_gemm_swizzled_scales=True,
        shuffle_rowwise_data=True,
        shuffle_columnwise_data=True,
        use_hadamard=_use_hadamard,
    )
    _get_quantized_params_cpu(
        model, mxfp4_quantizer, "MXFP4", replace=True
    )

    torch.cuda.empty_cache()


def _get_quantized_params_cpu(model, quantizer, qtype: str, replace: bool = False) -> List:
    """Byte-equivalent to NeMo
    ``custom_callbacks.CustomCallback._get_quantized_params_cpu``
    (``mlperf-training-6-0/llama2_sft/nemo/src/callbacks/custom_callbacks.py:151-212``).

    Adaptation:

    * NeMo reads ``self.first_last_layers_bf16`` / ``self.num_layers_at_start_in_bf16``
      / ``self.num_layers_at_end_in_bf16`` / ``self.store_quantized_params_on_gpu``
      from the OmegaConf ``cfg.model`` (env-defaulted to ``False`` /
      ``0`` / ``0`` / ``False`` in NeMo's MI355X FP4 config). Primus
      reads the same env vars (``FIRST_LAST_LAYERS_BF16``,
      ``NUM_LAYERS_AT_START_IN_BF16``, ``NUM_LAYERS_AT_END_IN_BF16``,
      ``STORE_GPU``) with NeMo's same defaults. NeMo's MI355X reference
      run does not set any of them so the effective behaviour is
      identical.
    """
    import torch
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
    from transformer_engine.pytorch.quantized_tensor import QuantizedTensor

    first_last_layers_bf16 = _truthy_env("FIRST_LAST_LAYERS_BF16", default=False)
    num_layers_at_start_in_bf16 = int(os.getenv("NUM_LAYERS_AT_START_IN_BF16", "0") or "0")
    num_layers_at_end_in_bf16 = int(os.getenv("NUM_LAYERS_AT_END_IN_BF16", "0") or "0")
    store_quantized_params_on_gpu = _truthy_env("STORE_GPU", default=False)

    extracted_module = _extract_module(model)
    layers = extracted_module.decoder.layers
    layer_count = len(layers)

    quantized_params = []
    for layer_idx, layer in enumerate(layers):
        if first_last_layers_bf16:
            if layer_idx < num_layers_at_start_in_bf16 or layer_idx >= layer_count - num_layers_at_end_in_bf16:
                quantized_params.append([])
                continue

        quantized_layer_params = []
        for name, module in layer.named_modules():
            if not isinstance(module, (te.Linear, te.LayerNormLinear)):
                continue
            if not hasattr(module, 'weight'):
                continue
            param = module.weight
            with torch.no_grad():
                if qtype == "MXFP4":
                    if isinstance(param, QuantizedTensor):
                        bf16_param = param.dequantize().detach()
                    else:
                        bf16_param = param.detach().to(torch.bfloat16)
                    bf16_param = bf16_param.contiguous()
                    qparam = quantizer(bf16_param)
                    del bf16_param
                    assert qparam._rowwise_data is not None, "No rowwise data."
                elif qtype == "FP8_DS":
                    fp8_data = param.data.to(torch.float8_e4m3fn).view(torch.uint8)
                    scale_inv = torch.ones(1, dtype=torch.float32, device=param.device)
                    qparam = Float8Tensor(
                        shape=param.shape,
                        dtype=torch.bfloat16,
                        data=fp8_data,
                        fp8_scale_inv=scale_inv,
                        fp8_dtype=tex.DType.kFloat8E4M3,
                        quantizer=quantizer,
                    )
                    assert qparam._data is not None, "No data."
                else:
                    raise ValueError(f"Unsupported quantization type: {qtype}")

                if replace:
                    qparam.requires_grad = False
                    module._parameters['weight'] = qparam
                else:
                    qparam = qparam.clone()
                    if not store_quantized_params_on_gpu:
                        if qtype == "MXFP4":
                            qparam._rowwise_data = qparam._rowwise_data.cpu().pin_memory()
                            if qparam._columnwise_data is not None:
                                qparam._columnwise_data = qparam._columnwise_data.cpu().pin_memory()
                        elif qtype == "FP8_DS":
                            qparam._data = qparam._data.cpu().pin_memory()
                            if qparam._transpose is not None:
                                qparam._transpose = qparam._transpose.cpu().pin_memory()
                    quantized_layer_params.append(qparam)
        quantized_params.append(quantized_layer_params)

    return quantized_params


# ---------------------------------------------------------------------------
# Memory diagnostics (Primus-only; not part of NeMo's pre-quantize logic).
# ---------------------------------------------------------------------------


def _log_gpu_mem(tag: str) -> None:
    """Print rank-0 GPU mem checkpoint (allocated / reserved / max-allocated, GiB).

    Gated by ``MXFP4_HEALING_DEBUG`` (default ``False``) so these diagnostic
    lines are opt-in and don't clutter normal runs.
    """
    if not _truthy_env("MXFP4_HEALING_DEBUG", default=False):
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        dev = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev) / (1024**3)
        reserved = torch.cuda.memory_reserved(dev) / (1024**3)
        max_alloc = torch.cuda.max_memory_allocated(dev) / (1024**3)
        max_reserved = torch.cuda.max_memory_reserved(dev) / (1024**3)
        log_rank_0(
            f"{_TAG}[mem] {tag:<48s} "
            f"allocated={alloc:7.2f} GiB | reserved={reserved:7.2f} GiB | "
            f"max_alloc={max_alloc:7.2f} GiB | max_reserved={max_reserved:7.2f} GiB"
        )
    except Exception as exc:  # noqa: BLE001
        log_rank_0(
            f"{_TAG}[mem] {tag}: failed to read "
            f"({type(exc).__name__}: {exc})"
        )


def _empty_cache_and_collect(tag: str) -> None:
    """Run gc.collect() + torch.cuda.empty_cache() to release any cached blocks
    holding onto the freed BF16 storage. Prints a mem checkpoint after."""
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        _log_gpu_mem(tag)
    except Exception as exc:  # noqa: BLE001
        log_rank_0(
            f"{_TAG}[mem] {tag}: empty_cache failed "
            f"({type(exc).__name__}: {exc})"
        )


# ---------------------------------------------------------------------------
# train() wrapper -- invokes _pre_quantize_model once before the first iter.
# ---------------------------------------------------------------------------


def _make_pre_quantizing_train(orig_train: Callable) -> Callable:
    """Return a thin wrapper around megatron-bridge's ``train`` that runs
    pre-quantization once on the first call, then delegates to ``orig_train``.

    ``orig_train`` is closed over directly; since this wrapper is installed
    from the recipe (after the recipe-level ``megatron_bridge_train_override``
    is defined), there is no need for the setter/getter swap dance the
    old ``@register_patch`` version had.
    """

    if getattr(orig_train, "_primus_pre_quantize_wrapped", False):
        return orig_train  # idempotent

    state = {"done": False}

    def _wrapped_train(forward_step_func, model, *args, **kwargs):
        if not state["done"]:
            try:
                try:
                    from primus.recipes.mxfp4_healing import log_healing_env_banner_once

                    log_healing_env_banner_once()
                except Exception:  # noqa: BLE001
                    pass
                log_rank_0(
                    f"{_TAG}   PRE_QUANTIZED_MODEL=True -> running "
                    "NeMo-equivalent _pre_quantize_model "
                    "(FP8 stash on CPU + MXFP4 swap)."
                )

                # Memory diagnostic: confirms whether the BF16 storage is freed
                # after the in-place MXFP4 swap. If `allocated` does NOT drop by
                # roughly half between "before swap" and "after swap (post empty_cache)",
                # something outside this loop (DDP grad bucket, distopt mapping, TE
                # FP8 weight cache, etc.) is still holding the old BF16 storage.
                _log_gpu_mem("before _pre_quantize_model")

                # NeMo MLPerf 6.0 MI355X FP4 reference (custom_callbacks.py:122-149).
                # Byte-equivalent: clones FP8 (E4M3) weights to CPU for healing,
                # then replaces every TE Linear/LayerNormLinear weight with an
                # MXFP4Tensor (rowwise+columnwise data populated). HEALING_ITER
                # gating is enforced by _ORDERED_FP8_STASH consumers; here we
                # *always* stash so PRE_QUANTIZED_MODEL=True and HEALING_ITER>0
                # have identical behaviour to NeMo (which always stashes when
                # cfg.model.pre_quantized_model is True).
                _pre_quantize_model(model)

                _log_gpu_mem("after _pre_quantize_model (BEFORE empty_cache)")
                _empty_cache_and_collect("after _pre_quantize_model (AFTER empty_cache)")

                # Bridge: populate primus.recipes.mxfp4_healing._ORDERED_FP8_STASH
                # with flat (module, fp8_cpu_tensor) pairs that share storage
                # with _FP8_CPU_PARAMS, so existing healing-side code
                # (restore-to-GPU, refcount audit, FP8 warmup) keeps working.
                try:
                    from primus.recipes.mxfp4_healing import (
                        _set_ordered_fp8_stash_from_layered,
                    )

                    _set_ordered_fp8_stash_from_layered(model, _FP8_CPU_PARAMS)
                except Exception as bridge_err:  # noqa: BLE001
                    log_rank_0(
                        f"{_TAG}   FP8 stash bridge to mxfp4_healing failed "
                        f"({type(bridge_err).__name__}: {bridge_err}); "
                        f"healing/restore will not work."
                    )

                # NeMo MLPerf MI355X parity: optional FP8 warmup (off by default).
                # Honors MXFP4_HEALING_FP8_WARMUP=1 to enable.
                try:
                    from primus.recipes.mxfp4_healing import (
                        healing_iter as _healing_iter,
                        run_fp8_warmup_for_kernel_jit,
                    )

                    if _healing_iter() > 0:
                        _optimizer = args[0] if len(args) > 0 else kwargs.get("optimizer")
                        _state = args[4] if len(args) > 4 else kwargs.get("state")
                        _model_cfg = None
                        try:
                            chunks = list(model) if isinstance(model, (list, tuple)) else [model]
                            _model_cfg = getattr(chunks[0], "config", None)
                        except Exception:  # noqa: BLE001
                            pass
                        if _model_cfg is None and _state is not None:
                            _model_cfg = getattr(getattr(_state, "cfg", None), "model", None)

                        if _model_cfg is None:
                            log_rank_0(
                                f"{_TAG}   FP8 warmup skipped: could not resolve "
                                "model_config from train() args (no model.config "
                                "and no state.cfg.model)."
                            )
                        else:
                            run_fp8_warmup_for_kernel_jit(
                                model=model,
                                model_config=_model_cfg,
                                optimizer=_optimizer,
                                forward_step_func=forward_step_func,
                                state=_state,
                            )
                except Exception as warm_err:  # noqa: BLE001
                    log_rank_0(
                        f"{_TAG}   FP8 warmup raised "
                        f"({type(warm_err).__name__}: {warm_err}); "
                        f"continuing to iter 1 without warmup."
                    )
            finally:
                # Even on failure, mark as done so we don't re-attempt every
                # call in case of caller retry. The exception will propagate.
                state["done"] = True
        return orig_train(forward_step_func, model, *args, **kwargs)

    _wrapped_train._primus_pre_quantize_wrapped = True  # type: ignore[attr-defined]
    return _wrapped_train


def install_pre_quantize_wrap(orig_train: Callable) -> Callable:
    """Recipe-facing entry point.

    If ``PRE_QUANTIZED_MODEL`` is enabled, returns a pre-quantizing
    wrapper around ``orig_train``; otherwise returns ``orig_train``
    unchanged. Idempotent: re-wrapping an already-wrapped callable is
    a no-op.
    """
    if not is_pre_quantized_enabled():
        return orig_train
    return _make_pre_quantizing_train(orig_train)
