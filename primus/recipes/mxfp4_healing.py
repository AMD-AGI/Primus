###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""NeMo-parity MXFP4 -> FP8 healing for Primus / Megatron-Bridge.

Reference: ``mlperf_code_llama2_70b_0430/src/callbacks/custom_callbacks.py``
``CustomCallback`` (``_healing_setup``, ``_set_quantized_params_cpu``,
``_reset_full_iteration_cuda_graphs``, ``on_train_batch_end`` healing
branch). This module mirrors that logic byte-for-byte; only the trigger
surface differs:

* NeMo's pre-quantize fires from ``on_train_start``; healing fires from
  ``on_train_batch_end`` when ``trainer.global_step + 1 == healing_iter``.
* Primus's pre-quantize fires from a megatron-bridge ``train()`` wrapper
  (``primus.recipes.pre_quantize_mxfp4``); healing fires when the recipe
  calls ``apply_healing_after_step(model, model_config, train_state.step)``
  immediately after ``train_state.step += 1`` -- the same arithmetic
  identity as NeMo.

All NeMo-side state lives on ``CustomCallback`` (``self.fp8_cpu_params``,
``self.healing_lambda``); Primus has no callback instance, so we use module
state. Configuration that NeMo reads from OmegaConf ``cfg.model.*`` is
read here from environment variables with NeMo's same defaults
(``HEALING_ITER`` / ``HEALING_PRECISION`` / ``ENABLE_TRANSPOSE_CACHE`` /
``RESET_CG_AFTER_HEALING`` / ``FIRST_LAST_LAYERS_BF16`` /
``NUM_LAYERS_AT_{START,END}_IN_BF16`` / ``STORE_GPU``).
"""

from __future__ import annotations

import os
from typing import Any, List

import torch

from primus.modules.module_utils import log_rank_0


# ---------------------------------------------------------------------------
# Module state.
# ---------------------------------------------------------------------------

# Layered FP8 (E4M3) CPU stash, populated from outside via
# ``set_fp8_cpu_params`` (the bridge from ``pre_quantize_mxfp4._FP8_CPU_PARAMS``)
# in the same shape as NeMo's ``CustomCallback.fp8_cpu_params``:
# ``List[List[Float8Tensor]]`` indexed first by ``decoder.layers`` index and
# then by intra-layer iteration over TE Linear / LayerNormLinear modules.
_FP8_CPU_PARAMS: List[List[Any]] = []
_HEALING_APPLIED: bool = False


# ---------------------------------------------------------------------------
# Env helpers (NeMo ``custom_callbacks._env_enabled`` parity).
# ---------------------------------------------------------------------------


def _env_enabled(name: str) -> bool:
    """NeMo ``custom_callbacks.py:37-43``: missing/unrecognized = False."""
    val = os.getenv(name, "").strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    return False


def healing_iter() -> int:
    """``cfg.model.healing_iter`` analog (0 = healing disabled)."""
    raw = os.getenv("HEALING_ITER", "0")
    try:
        return max(0, int(raw.strip()))
    except ValueError:
        return 0


def healing_precision() -> str:
    """``cfg.model.healing_precision`` analog: ``FP8_DS`` (default) or ``MXFP8``."""
    return os.getenv("HEALING_PRECISION", "FP8_DS").strip().upper()


# ---------------------------------------------------------------------------
# Model traversal helpers.
# ---------------------------------------------------------------------------


def _extract_module(model):
    """NeMo ``custom_callbacks.py:92-97``. Adapted: Primus passes
    ``List[ModelChunk]``; pick chunk 0 then strip ``.module``.
    """
    m = model
    if isinstance(m, (list, tuple)):
        m = m[0]
    while hasattr(m, "module"):
        m = m.module
    return m


def _all_modules(model):
    chunks = list(model) if isinstance(model, (list, tuple)) else [model]
    for ch in chunks:
        yield from ch.modules()


def _te_linear_types():
    import transformer_engine.pytorch as te

    types = (te.Linear, te.LayerNormLinear)
    if hasattr(te, "LayerNormMLP"):
        types = types + (te.LayerNormMLP,)
    return types


# ---------------------------------------------------------------------------
# NeMo ``_healing_setup`` / ``healing_lambda``.
# ---------------------------------------------------------------------------


def _build_healing_lambda(model_config: Any):
    """NeMo ``CustomCallback._healing_setup`` (``custom_callbacks.py:176-187``).

    Reads ``fp8_amax_history_len`` / ``fp8_amax_compute_algo`` /
    ``fp8_reduce_amax`` / ``fp8_dot_product_attention`` from the live
    ``TransformerConfig`` (NeMo reads them from OmegaConf ``cfg.model``;
    Megatron-Bridge stores the same fields directly on ``model_config``).
    NeMo defaults: ``amax_history_len=4``, ``amax_compute_algo="most_recent"``,
    ``reduce_amax=False``, ``fp8_dpa=False``.
    """
    import transformer_engine.common.recipe as _recipe

    prec = healing_precision()
    if prec == "FP8_DS":
        amax_len = int(getattr(model_config, "fp8_amax_history_len", 4) or 4)
        amax_algo = str(getattr(model_config, "fp8_amax_compute_algo", "most_recent") or "most_recent")
        reduce_amax = bool(getattr(model_config, "fp8_reduce_amax", False))
        fp8_dpa = bool(getattr(model_config, "fp8_dot_product_attention", False))

        log_rank_0(
            "[mxfp4_healing] Healing recipe = DelayedScaling("
            f"amax_history_len={amax_len}, "
            f"amax_compute_algo={amax_algo!r}, "
            f"reduce_amax={reduce_amax}, "
            f"fp8_dpa={fp8_dpa})"
        )

        def _delayed(_config: Any):
            return _recipe.DelayedScaling(
                amax_history_len=amax_len,
                amax_compute_algo=amax_algo,
                reduce_amax=reduce_amax,
                fp8_dpa=fp8_dpa,
            )

        return _delayed

    if prec == "MXFP8":
        log_rank_0("[mxfp4_healing] Healing recipe = MXFP8BlockScaling() (TE defaults)")

        def _mxfp8(_config: Any):
            return _recipe.MXFP8BlockScaling()

        return _mxfp8

    raise ValueError(f"Unsupported HEALING_PRECISION={prec!r} (expected FP8_DS or MXFP8)")


# ---------------------------------------------------------------------------
# NeMo ``_set_quantized_params_cpu`` (custom_callbacks.py:293-355).
# ---------------------------------------------------------------------------


def _set_quantized_params_cpu(model: Any, cpu_params: List[List[Any]], qtype: str) -> int:
    """Restore CPU-stashed quantized weights to GPU and reinstall on TE modules.

    Byte-equivalent to NeMo ``CustomCallback._set_quantized_params_cpu``
    (``custom_callbacks.py:293-355``). Iterates ``decoder.layers`` in the same
    order as ``_get_quantized_params_cpu`` (which built ``cpu_params``), so the
    positional index ``cpu_params[layer_idx][param_idx]`` (consumed via
    ``.pop(0)``) lines up with the ``param_idx``-th TE Linear /
    LayerNormLinear / LayerNormMLP module encountered inside
    ``layers[layer_idx].named_modules()``.

    NeMo's two-pass design (set ``module._parameters['weight']=None`` for all
    target modules first, then assign the restored weight) is preserved:
    decoupling the unlink from the install lets the caching allocator release
    the old MXFP4 weight storage (~75 GiB) before the FP8 GPU side is wired
    in (~64 GiB), preventing both sets resident at peak.
    """
    extracted = _extract_module(model)
    layers = extracted.decoder.layers
    layer_count = len(layers)
    dev = torch.cuda.current_device()

    first_last = _env_enabled("FIRST_LAST_LAYERS_BF16")
    n_start = int(os.getenv("NUM_LAYERS_AT_START_IN_BF16", "0") or "0")
    n_end = int(os.getenv("NUM_LAYERS_AT_END_IN_BF16", "0") or "0")
    store_gpu = _env_enabled("STORE_GPU")
    enable_tc = _env_enabled("ENABLE_TRANSPOSE_CACHE")
    use_fuser = bool(getattr(extracted.config, "use_transformer_engine_op_fuser", False))

    te_types = _te_linear_types()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    xfer_stream = torch.cuda.Stream()
    n_restored = 0

    for layer_idx, layer in enumerate(layers):
        if first_last:
            if layer_idx < n_start or layer_idx >= layer_count - n_end:
                continue

        target_modules = []
        for _name, module in layer.named_modules():
            if not isinstance(module, te_types):
                continue
            if not hasattr(module, "weight"):
                continue
            target_modules.append(module)

        with torch.no_grad():
            for module in target_modules:
                module._parameters["weight"] = None

        for module in target_modules:
            with torch.no_grad():
                weight = cpu_params[layer_idx].pop(0)

                if not store_gpu:
                    with torch.cuda.stream(xfer_stream):
                        if qtype == "FP8_DS":
                            weight._data = weight._data.to(dev, non_blocking=True)
                            if weight._transpose is not None:
                                if enable_tc:
                                    weight._transpose = weight._transpose.to(dev, non_blocking=True)
                                else:
                                    weight._transpose = None
                        elif qtype == "MXFP4":
                            weight._rowwise_data = weight._rowwise_data.to(dev, non_blocking=True)
                            if weight._columnwise_data is not None:
                                weight._columnwise_data = weight._columnwise_data.to(dev, non_blocking=True)
                        else:
                            raise ValueError(f"Unsupported quantization type: {qtype}")

                weight.requires_grad = False
                module._parameters["weight"] = weight
                n_restored += 1

        if use_fuser:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "_make_fused_impl"):
                mlp._fused_impl = (mlp._make_fused_impl(),)
            sa = getattr(layer, "self_attention", None)
            if sa is not None:
                for attr in ("linear_proj", "linear_qkv"):
                    mod = getattr(sa, attr, None)
                    if mod is not None and hasattr(mod, "_make_fused_branches"):
                        mod._fused_branches = mod._make_fused_branches()

    xfer_stream.synchronize()
    return n_restored


# ---------------------------------------------------------------------------
# NeMo ``custom_llama.reset_fp8_state`` (custom_llama.py:77-92).
# ---------------------------------------------------------------------------


def _reset_fp8_state(model: Any) -> None:
    """Force TE to re-initialize ``fp8_meta`` under the new recipe on next forward.

    Without this, modules keep MXFP4 ``RecipeState`` while the recipe and
    weights changed under them; the next GEMM dereferences scale/amax
    pointers from the wrong RecipeState type. Mirrors NeMo's
    ``custom_llama.reset_fp8_state``: MX recipes (MXFP4/MXFP8) use
    ``RecipeState`` without ``.scale``, and TE's ``reset_fp8_meta_tensors``
    assumes DelayedScaling-style state and raises against MX, so we detect
    the MX case by inspecting ``fp8_meta`` and skip the reset for those
    modules.
    """

    def reset_fp8(m: Any) -> None:
        if not hasattr(m, "fp8_initialized"):
            return
        m.fp8_initialized = False
        # MX recipes (MXFP4/MXFP8) use RecipeState without `.scale`; TE's
        # reset_fp8_meta_tensors assumes DelayedScaling-style state and raises.
        fp8_meta = getattr(m, "fp8_meta", None) or {}
        for key in ("scaling_fwd", "scaling_bwd"):
            state = fp8_meta.get(key)
            if state is not None and not hasattr(state, "scale"):
                return
        m.reset_fp8_meta_tensors()

    for m in _all_modules(model):
        reset_fp8(m)


# ---------------------------------------------------------------------------
# NeMo ``_reset_full_iteration_cuda_graphs`` (custom_callbacks.py:148-174).
# ---------------------------------------------------------------------------


def _reset_full_iteration_cuda_graphs() -> None:
    """Drop Megatron full-iteration CUDA graph captures before the recipe swap.

    After MXFP4 -> FP8 weight swap and recipe change, replaying old graphs
    is invalid. ``RESET_CG_AFTER_HEALING`` (NeMo: ``cfg.model.reset_cg_after_healing``)
    additionally resets per-stage warmup step counters so
    ``FullCudaGraphWrapper`` re-runs ``cuda_graph_warmup_steps`` before
    re-capture.
    """
    try:
        from megatron.core.full_cuda_graph import FullCudaGraphWrapper
    except ImportError:
        return

    FullCudaGraphWrapper.cuda_graph["training"] = None
    FullCudaGraphWrapper.cuda_graph["validation"] = None
    FullCudaGraphWrapper.result["training"] = None
    FullCudaGraphWrapper.result["validation"] = None

    if _env_enabled("RESET_CG_AFTER_HEALING"):
        FullCudaGraphWrapper.curr_iteration["training"] = 0
        FullCudaGraphWrapper.curr_iteration["validation"] = 0


# ---------------------------------------------------------------------------
# Public stash bridge.
# ---------------------------------------------------------------------------


def set_fp8_cpu_params(layered: List[List[Any]]) -> None:
    """Hand off NeMo-style ``List[List[Float8Tensor]]`` from pre-quantize.

    Mirrors the assignment ``self.fp8_cpu_params = self._get_quantized_params_cpu(...)``
    in NeMo ``CustomCallback._pre_quantize_model``. Storing a reference (not
    a copy) means ``_set_quantized_params_cpu``'s ``.pop(0)`` mutates the same
    list as the caller, identical to NeMo.
    """
    global _FP8_CPU_PARAMS, _HEALING_APPLIED
    _FP8_CPU_PARAMS = layered
    _HEALING_APPLIED = False


# ---------------------------------------------------------------------------
# NeMo ``on_train_batch_end`` healing branch (custom_callbacks.py:620-677).
# ---------------------------------------------------------------------------


def apply_healing_after_step(model: Any, model_config: Any, train_state_step: int) -> None:
    """Run NeMo's ``on_train_batch_end`` healing branch.

    Trigger condition mirrors NeMo:
      ``if trainer.global_step + 1 == self.cfg.model.healing_iter:``
    Primus's ``train_state.step`` is incremented before this call (in the
    recipe's ``megatron_bridge_train_override``), so it carries the same
    value as NeMo's ``trainer.global_step``.

    Steps below are 1:1 with the NeMo branch.
    """
    global _HEALING_APPLIED

    hi = healing_iter()
    if hi <= 0 or _HEALING_APPLIED:
        return
    if train_state_step + 1 != hi:
        return

    if not _FP8_CPU_PARAMS:
        log_rank_0(
            "[mxfp4_healing] HEALING_ITER is set but the FP8 CPU stash is empty; "
            "either PRE_QUANTIZED_MODEL=True is missing or pre-quantize did not run. "
            "Skipping healing."
        )
        _HEALING_APPLIED = True
        return

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # 1. NeMo: ``self._reset_full_iteration_cuda_graphs(trainer)``.
    _reset_full_iteration_cuda_graphs()

    # 2. NeMo: ``self._set_quantized_params_cpu(trainer.model, self.fp8_cpu_params, self.healing_precision)``.
    n_restored = _set_quantized_params_cpu(model, _FP8_CPU_PARAMS, healing_precision())
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # 3. NeMo: patch ``megatron.core.fp4_utils`` to the healing recipe and clear ``_mxfp4_phase``.
    import megatron.core.fp4_utils as fp4u

    fp4u.get_fp4_recipe = _build_healing_lambda(model_config)
    fp4u._mxfp4_phase = False

    # 4. NeMo: when ``ENABLE_TRANSPOSE_CACHE`` is off, tell every TE Linear to stop
    # keeping a transpose cache (custom_callbacks.py:637-642).
    if not _env_enabled("ENABLE_TRANSPOSE_CACHE"):
        te_types = _te_linear_types()
        for m in _all_modules(model):
            if isinstance(m, te_types):
                m.keep_fp8_weight_transpose_cache = False

    # 5. NeMo: drop activation recompute on the live TransformerConfig
    # (custom_callbacks.py:644-648).
    extracted = _extract_module(model)
    cfg = extracted.config
    if cfg.recompute_granularity is not None:
        cfg.recompute_granularity = None
        cfg.recompute_num_layers = None
        cfg.recompute_method = None

    # 6. NeMo: clear MXFP4 caches on every module (custom_callbacks.py:650-656).
    for m in _all_modules(model):
        if hasattr(m, "_mxfp4_weight_cache"):
            del m._mxfp4_weight_cache
        if hasattr(m, "_mxfp4_persist_columnwise"):
            del m._mxfp4_persist_columnwise

    # 6b. Primus delta vs NeMo: clear ``save_original_input`` on every TE module.
    #
    # Megatron-LM shipped in Primus's tree (third_party/Megatron-Bridge/3rdparty/
    # Megatron-LM) sets ``module.save_original_input = True`` on every TE
    # ``linear_proj`` (and some ``mlp.linear_fc1``) when ``config.fp4`` is
    # active -- it's an MX-only optimization that lets TE re-quantize at
    # backward time from the saved BF16 input (legal under MXFP4BlockScaling
    # because the per-block scale is reproducible from the BF16 input).
    # ``DelayedScaling`` derives its FP8 scale from amax history that is
    # updated each step, so the saved BF16 input cannot reproduce the
    # forward FP8 input. TE guards this combination with a hard error::
    #
    #   RuntimeError: DelayedScaling recipe is not supported with save_original_input
    #
    # ...which fires on the very next forward after we patch
    # ``get_fp4_recipe -> FP8_DS``. NeMo MLPerf doesn't hit this because
    # their (older) Megatron-LM lacks the FP4 branch that sets the flag in
    # the first place. We reset it defensively to mirror NeMo's behavior.
    n_save_orig_cleared = 0
    for m in _all_modules(model):
        if getattr(m, "save_original_input", False):
            try:
                m.save_original_input = False
                n_save_orig_cleared += 1
            except Exception:  # noqa: BLE001
                pass

    # 7. NeMo: ``reset_fp8_state(trainer.lightning_module)`` (custom_callbacks.py:668).
    _reset_fp8_state(model)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    _HEALING_APPLIED = True

    log_rank_0(
        f"[mxfp4_healing] healing applied at train_state.step={train_state_step} "
        f"(step+1==HEALING_ITER={hi}); recipe={healing_precision()}; "
        f"restored {n_restored} FP8 weights; "
        f"cleared save_original_input on {n_save_orig_cleared} TE modules."
    )


# ---------------------------------------------------------------------------
# Status helpers + test hygiene.
# ---------------------------------------------------------------------------


def is_fp8_healing_phase() -> bool:
    """True after ``apply_healing_after_step`` has fired (TE weights are FP8)."""
    return _HEALING_APPLIED


def reset_healing_state() -> None:
    """Test / multi-run hygiene: forget the stash and the healed flag."""
    global _FP8_CPU_PARAMS, _HEALING_APPLIED
    _FP8_CPU_PARAMS = []
    _HEALING_APPLIED = False


# ---------------------------------------------------------------------------
# Backwards-compatible shims for existing callers.
#
# These keep ``primus/recipes/pre_quantize_mxfp4.py`` and
# ``primus/recipes/llama2_custom.py`` importable without edits. The new
# code is ``set_fp8_cpu_params`` (above); the legacy names below just
# delegate / no-op.
# ---------------------------------------------------------------------------


def _set_ordered_fp8_stash_from_layered(model: Any, fp8_cpu_params: List[List[Any]]) -> None:
    """Legacy alias for ``set_fp8_cpu_params``. ``model`` is unused; we keep
    the layered stash exactly as NeMo does.
    """
    del model  # unused; layered stash is enough (NeMo parity)
    set_fp8_cpu_params(fp8_cpu_params)


def log_healing_env_banner_once() -> None:
    """No-op (NeMo doesn't have this; logging removed from main flow)."""
    return


def log_training_step_phase(train_state_step: int) -> None:
    """No-op (NeMo doesn't have this; logging removed from main flow)."""
    del train_state_step
    return


def run_fp8_warmup_for_kernel_jit(*_args: Any, **_kwargs: Any) -> None:
    """No-op. NeMo's ``_pre_ttt_fp8_warmup`` call site is commented out
    (custom_callbacks.py:509-510), so the warmup is dead in NeMo too.
    """
    return
