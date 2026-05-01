###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# MLPerf 6.0 NeMo MI355X FP4 "healing" parity for Primus (Megatron-Bridge).
#
# Reference: mlperf-training-6-0/llama2_sft/nemo/src/callbacks/custom_callbacks.py
#   CustomCallback._pre_quantize_model, _healing_setup, on_train_batch_end
#   (global_step + 1 == healing_iter).
#
# Behaviour (when HEALING_ITER > 0 and PRE_QUANTIZED_MODEL is active):
#   1) Before MXFP4 pre-quantize: stash FP8 (E4M3) clones of each TE Linear /
#      LayerNormLinear weight on CPU (pin_memory), in deterministic iteration order.
#   2) After training step (S) with S + 1 == HEALING_ITER: all-reduce barrier,
#      load FP8 weights back to GPU, monkey-patch megatron.core.fp4_utils.get_fp4_recipe
#      to FP8 DelayedScaling or MXFP8BlockScaling, set _mxfp4_phase = False, clear
#      MXFP4 caches, and drop activation recompute on the live TransformerConfig (NeMo
#      disables recompute for the FP8 phase to free VRAM).
###############################################################################

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Tuple

import torch

# (module, fp8_tensor_on_cpu) in the same order as pre-quantize walks the model.
_ORDERED_FP8_STASH: List[Tuple[Any, Any]] = []
_HEALING_APPLIED: bool = False
_ENV_BANNER_SHOWN: bool = False
_HEALING_OFF_NOTICE_SHOWN: bool = False


def _is_rank0() -> bool:
    try:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0
    except Exception:  # noqa: BLE001
        return True


def _emit_rank0(msg: str) -> None:
    """Primus logger + raw stdout so lines appear next to Megatron-Bridge train_utils INFO."""
    if not _is_rank0():
        return
    try:
        from primus.modules.module_utils import log_rank_0

        log_rank_0(msg)
    except Exception:  # noqa: BLE001
        pass
    print(msg, flush=True)


def log_healing_env_banner_once() -> None:
    """Always print once per process (rank 0): how healing is configured (stdout + Primus log)."""
    global _ENV_BANNER_SHOWN
    if _ENV_BANNER_SHOWN or not _is_rank0():
        return
    _ENV_BANNER_SHOWN = True
    hi = os.getenv("HEALING_ITER", "0")
    pq = os.getenv("PRE_QUANTIZED_MODEL", "")
    plog = os.getenv("MXFP4_HEALING_PHASE_LOG", "1")
    msg = (
        f"[mxfp4_healing] config | PRE_QUANTIZED_MODEL={pq!r} HEALING_ITER={hi} "
        f"HEALING_PRECISION={healing_precision()} MXFP4_HEALING_PHASE_LOG={plog!r} | "
        f"Per-step MXFP4/FP8 lines: only when HEALING_ITER>0 and MXFP4_HEALING_PHASE_LOG is true. "
        f"Healing fires when train_state.step+1==HEALING_ITER (NeMo parity)."
    )
    _emit_rank0(msg)


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def healing_iter() -> int:
    """NeMo ``healing_iter`` (0 = disabled)."""
    raw = os.getenv("HEALING_ITER", "0")
    try:
        v = int(raw.strip())
    except ValueError:
        return 0
    return max(0, v)


def healing_precision() -> str:
    """NeMo ``healing_precision``: FP8_DS (default) or MXFP8."""
    return os.getenv("HEALING_PRECISION", "FP8_DS").strip().upper()


def is_fp8_healing_phase() -> bool:
    """True after healing has run (TE weights restored to FP8; ``_mxfp4_phase`` cleared)."""
    return _HEALING_APPLIED


def phase_label() -> str:
    """Short label for stdout: MXFP4 until healing, then FP8."""
    return "FP8" if _HEALING_APPLIED else "MXFP4"


def phase_log_every_step_enabled() -> bool:
    """Per-step phase lines when ``HEALING_ITER > 0``; disable with ``MXFP4_HEALING_PHASE_LOG=0``."""
    if healing_iter() <= 0:
        return False
    return _truthy_env("MXFP4_HEALING_PHASE_LOG", default=False)


def _enable_transpose_cache() -> bool:
    """NeMo ``_env_enabled("ENABLE_TRANSPOSE_CACHE")`` parity.

    Single source of truth for the ENABLE_TRANSPOSE_CACHE gate used in three
    places: (1) the FP8 CPU stash ``Float8Quantizer`` creation in
    ``primus.recipes.pre_quantize_mxfp4``, (2) the FP8 restore in
    ``_restore_fp8_weights_to_gpu`` (healing), and (3) the FP8 warmup swap in
    ``_pre_ttt_fp8_warmup``. Default ``False`` matches NeMo's ``_env_enabled``
    (missing env == disabled); the Primus MI355X reference shell exports
    ``ENABLE_TRANSPOSE_CACHE=1`` so the reference run gets the fast path.
    """
    return _truthy_env("ENABLE_TRANSPOSE_CACHE", default=False)


def _disable_te_linear_transpose_cache(model: Any) -> int:
    """Walk ``model.modules()`` and set ``keep_fp8_weight_transpose_cache=False``
    on every TE ``Linear`` / ``LayerNormLinear`` / ``LayerNormMLP`` (NeMo parity,
    ``custom_callbacks.py`` lines 419-422 in ``_pre_ttt_fp8_warmup`` and lines
    637-642 in ``on_train_batch_end``). Returns the count of modules modified.

    Called unconditionally from the caller when ``ENABLE_TRANSPOSE_CACHE`` is
    off; this function is the inner walk so both call sites stay small.
    """
    try:
        import transformer_engine.pytorch as te
    except Exception:  # noqa: BLE001
        return 0
    te_types: Tuple[type, ...] = (te.Linear, te.LayerNormLinear)
    lnmlp = getattr(te, "LayerNormMLP", None)
    if lnmlp is not None:
        te_types = te_types + (lnmlp,)

    n = 0
    for m in _all_modules_depth_first(model):
        if isinstance(m, te_types):
            try:
                m.keep_fp8_weight_transpose_cache = False
                n += 1
            except Exception:  # noqa: BLE001
                pass
    return n


def log_training_step_phase(train_state_step: int) -> None:
    """Rank-0 line showing MXFP4 vs FP8 so healing is visible across the boundary."""
    global _HEALING_OFF_NOTICE_SHOWN

    if healing_iter() <= 0:
        if _HEALING_OFF_NOTICE_SHOWN or train_state_step != 1:
            return
        _HEALING_OFF_NOTICE_SHOWN = True
        if _truthy_env("MXFP4_HEALING_DEBUG", default=False):
            _emit_rank0(
                "[mxfp4_healing] HEALING_ITER=0 (or unset): mid-run FP8 healing is OFF — "
                "weights stay MXFP4 for the full run. Export HEALING_ITER=340 (and PRE_QUANTIZED_MODEL=True) "
                "for MLPerf MI355X-style MXFP4→FP8 switch; set MXFP4_HEALING_DEBUG=0 to hide this line."
            )
        return

    if not phase_log_every_step_enabled():
        return

    hi = healing_iter()
    lbl = phase_label()
    # After increment, ``train_state_step`` is the completed iteration index (NeMo-style).
    if lbl == "MXFP4":
        line = (
            f"[mxfp4_healing] train_state.step={train_state_step} | active=MXFP4 "
            f"(pre-healing; healing when step+1=={hi} -> FP8)"
        )
    else:
        line = (
            f"[mxfp4_healing] train_state.step={train_state_step} | active=FP8 "
            f"(post-healing; TE recipe={healing_precision()})"
        )
    _emit_rank0(line)


def reset_healing_state() -> None:
    """Test / multi-run hygiene (stash + healed flag only; banners stay once per process)."""
    global _ORDERED_FP8_STASH, _HEALING_APPLIED
    _ORDERED_FP8_STASH = []
    _HEALING_APPLIED = False


def _set_ordered_fp8_stash_from_layered(model: Any, fp8_cpu_params: List[List[Any]]) -> None:
    """Bridge from NeMo-style ``List[List[Float8Tensor]]`` FP8 stash to the
    flat ``(module, fp8_cpu_tensor)`` view consumed by the rest of this
    module (restore-to-GPU, refcount audit, FP8 warmup).

    Walks ``decoder.layers`` in the *exact* same order as
    ``primus.recipes.pre_quantize_mxfp4._get_quantized_params_cpu`` --
    which is itself byte-equivalent to NeMo
    ``custom_callbacks.CustomCallback._get_quantized_params_cpu``
    (``mlperf-training-6-0/llama2_sft/nemo/src/callbacks/custom_callbacks.py:151-212``)
    -- so positional indices line up: ``fp8_cpu_params[layer_idx][param_idx]``
    is the FP8 clone that was made from the ``param_idx``-th
    ``te.Linear``/``te.LayerNormLinear`` encountered inside
    ``layers[layer_idx].named_modules()``.

    The resulting ``_ORDERED_FP8_STASH`` shares the same ``Float8Tensor``
    objects as ``fp8_cpu_params``; we just expose them with their owning
    ``module`` for module-keyed restore-time lookups.

    Honors ``HEALING_ITER>0`` -- builds an empty stash when healing is
    disabled so downstream code degrades gracefully.
    """
    global _ORDERED_FP8_STASH

    # Mirror what the old ``stash_fp8_ordered_before_mxfp4`` did at the top:
    # reset BOTH the stash and the ``_HEALING_APPLIED`` flag so multi-run /
    # test scenarios start clean. (Banners stay once per process by design.)
    reset_healing_state()

    if healing_iter() <= 0:
        return

    try:
        import transformer_engine.pytorch as te
    except Exception as err:  # noqa: BLE001
        _emit_rank0(
            f"[mxfp4_healing] cannot bridge layered FP8 stash to module-keyed view "
            f"(transformer_engine.pytorch import failed: {type(err).__name__}: {err})."
        )
        return

    first_last_layers_bf16 = _truthy_env("FIRST_LAST_LAYERS_BF16", default=False)
    num_layers_at_start_in_bf16 = int(os.getenv("NUM_LAYERS_AT_START_IN_BF16", "0") or "0")
    num_layers_at_end_in_bf16 = int(os.getenv("NUM_LAYERS_AT_END_IN_BF16", "0") or "0")

    inner = model
    if isinstance(inner, (list, tuple)):
        inner = inner[0]
    seen = set()
    while hasattr(inner, "module") and id(inner) not in seen:
        seen.add(id(inner))
        inner = inner.module

    decoder = getattr(inner, "decoder", None)
    layers = getattr(decoder, "layers", None) if decoder is not None else None
    if not layers:
        _emit_rank0(
            "[mxfp4_healing] cannot bridge layered FP8 stash: no decoder.layers on extracted module."
        )
        return

    layer_count = len(layers)
    n_pairs = 0
    for layer_idx, layer in enumerate(layers):
        if first_last_layers_bf16:
            if layer_idx < num_layers_at_start_in_bf16 or layer_idx >= layer_count - num_layers_at_end_in_bf16:
                continue
        if layer_idx >= len(fp8_cpu_params):
            continue
        layer_params = fp8_cpu_params[layer_idx]
        if not layer_params:
            continue
        param_idx = 0
        for _name, module in layer.named_modules():
            if not isinstance(module, (te.Linear, te.LayerNormLinear)):
                continue
            if not hasattr(module, 'weight'):
                continue
            if param_idx >= len(layer_params):
                break
            _ORDERED_FP8_STASH.append((module, layer_params[param_idx]))
            param_idx += 1
            n_pairs += 1

    _emit_rank0(
        f"[mxfp4_healing] Bridged {n_pairs} FP8 (E4M3) CPU clones into module-keyed "
        f"_ORDERED_FP8_STASH (List[List]→flat) for HEALING_ITER={healing_iter()}."
    )


def _build_healing_lambda(model_config: Any) -> Callable[[Any], Any]:
    prec = healing_precision()
    if prec == "FP8_DS":
        import transformer_engine.common.recipe

        amax_len = int(getattr(model_config, "fp8_amax_history_len", 4) or 4)
        amax_algo = str(getattr(model_config, "fp8_amax_compute_algo", "most_recent") or "most_recent")
        reduce_amax = bool(getattr(model_config, "fp8_reduce_amax", _truthy_env("FP8_REDUCE_AMAX", False)))
        fp8_dpa = bool(getattr(model_config, "fp8_dot_product_attention", False))

        def _lambda(_config: Any) -> Any:
            return transformer_engine.common.recipe.DelayedScaling(
                amax_history_len=amax_len,
                amax_compute_algo=amax_algo,
                reduce_amax=reduce_amax,
                fp8_dpa=fp8_dpa,
            )

        return _lambda
    if prec == "MXFP8":
        import transformer_engine.common.recipe

        def _lambda_mx(_config: Any) -> Any:
            return transformer_engine.common.recipe.MXFP8BlockScaling()

        return _lambda_mx
    raise ValueError(f"Unsupported HEALING_PRECISION={prec!r} (expected FP8_DS or MXFP8)")


def _classify_referrer(ref: Any, target_id: int) -> str:
    """Best-effort one-line description of a ``gc.get_referrers`` entry.

    The goal is to surface *which* container is keeping the old MXFP4 weight
    storage alive across the FP8 swap (e.g. distributed-optimizer
    param→main_param dict, DDP grad bucket list, TE workspace cache,
    LoRA adapter ``base_linear`` attribute, autograd saved-tensor closure).
    We deliberately stay defensive — every introspection branch is
    wrapped in ``try`` so a malformed referrer never crashes the audit.
    """
    import inspect

    t = type(ref).__name__
    try:
        if isinstance(ref, dict):
            keys_with_val = []
            for k, v in ref.items():
                if id(v) == target_id:
                    try:
                        keys_with_val.append(repr(k))
                    except Exception:  # noqa: BLE001
                        keys_with_val.append(f"<unrepr:{type(k).__name__}>")
                if len(keys_with_val) >= 3:
                    break
            sample_keys = list(ref.keys())[:3]
            sample_keys_repr = []
            for k in sample_keys:
                try:
                    sample_keys_repr.append(repr(k))
                except Exception:  # noqa: BLE001
                    sample_keys_repr.append(f"<unrepr:{type(k).__name__}>")
            return (
                f"dict(len={len(ref)}, "
                f"holding_keys={keys_with_val}, "
                f"sample_keys={sample_keys_repr})"
            )
        if isinstance(ref, (list, tuple)):
            indices = [i for i, v in enumerate(ref[:128]) if id(v) == target_id]
            return f"{t}(len={len(ref)}, holding_indices={indices[:5]})"
        if isinstance(ref, set):
            return f"set(len={len(ref)})"
        if inspect.isframe(ref):
            try:
                code = ref.f_code
                holding_names = [
                    name for name, val in ref.f_locals.items() if id(val) == target_id
                ]
                return (
                    f"frame({code.co_filename}:{ref.f_lineno} in {code.co_name}, "
                    f"holding_locals={holding_names})"
                )
            except Exception:  # noqa: BLE001
                return "frame(<introspect failed>)"
        if t == "cell":
            try:
                contents = ref.cell_contents
                return f"cell(contents_type={type(contents).__name__})"
            except Exception:  # noqa: BLE001
                return "cell(<empty>)"
        if hasattr(ref, "__class__"):
            mod = type(ref).__module__
            qual = type(ref).__qualname__
            holding_attrs: list = []
            try:
                for attr in dir(ref):
                    if attr.startswith("__"):
                        continue
                    try:
                        val = getattr(ref, attr, None)
                    except Exception:  # noqa: BLE001
                        continue
                    if id(val) == target_id:
                        holding_attrs.append(attr)
                    if len(holding_attrs) >= 3:
                        break
            except Exception:  # noqa: BLE001
                pass
            return f"{mod}.{qual}(holding_attrs={holding_attrs})"
    except Exception as exc:  # noqa: BLE001
        return f"{t}(<classify failed: {type(exc).__name__}: {exc}>)"
    return t


def _audit_mxfp4_weight_referrers(
    model: Any,
    tag: str,
    sample_count: int = 1,
) -> None:
    """Dump ``gc.get_referrers`` + refcount for a few MXFP4 weights about to
    be swapped out by ``_restore_fp8_weights_to_gpu``.

    Gated by ``MXFP4_HEALING_REFCOUNT_AUDIT`` (default off) so it stays
    out of perf runs. When you see refcount > the expected baseline
    (``module._parameters['weight']`` + the local audit ref + the
    ``gc.get_referrers`` argument list = 3), the printed referrers tell
    you exactly which container is preventing the old MXFP4 storage from
    being released. Likely culprits on this stack:

    * Distributed‑optimizer param→main_param dict (Megatron‑Bridge
      ``DistributedOptimizer`` keeps a per‑param mapping that is keyed
      on the parameter tensor id captured at optimizer build time).
    * DDP grad bucket / param sync buckets if
      ``overlap_grad_reduce=True`` /
      ``overlap_param_gather=True`` / ``keep_fp8_transpose_cache=True``.
    * TE module workspace caches beyond ``_mxfp4_weight_cache``
      (``_fp8_weight_workspace``, version‑dependent).
    * LoRA adapter wrappers that captured ``base_linear.weight`` by
      attribute at install time.
    * Autograd saved‑tensor closures (only if ``save_original_input``
      was active during the last MXFP4 step — Primus already clears
      that flag, but worth confirming via this audit).
    """
    if not _truthy_env("MXFP4_HEALING_REFCOUNT_AUDIT", default=False):
        return
    if not _is_rank0():
        return
    if not _ORDERED_FP8_STASH:
        return

    import gc
    import sys

    _emit_rank0(f"[mxfp4_healing][refaudit] === {tag} ===")
    samples = _ORDERED_FP8_STASH[: max(1, int(sample_count))]
    for idx, (module, _fp8_cpu) in enumerate(samples):
        try:
            old_weight = module._parameters.get("weight", None)
        except Exception as exc:  # noqa: BLE001
            _emit_rank0(
                f"[mxfp4_healing][refaudit] sample[{idx}]: "
                f"could not read module._parameters['weight'] "
                f"({type(exc).__name__}: {exc})"
            )
            continue
        if old_weight is None:
            _emit_rank0(
                f"[mxfp4_healing][refaudit] sample[{idx}] "
                f"({type(module).__name__}): module.weight is None"
            )
            continue

        target_id = id(old_weight)
        rc = sys.getrefcount(old_weight)
        try:
            referrers = gc.get_referrers(old_weight)
        except Exception as exc:  # noqa: BLE001
            _emit_rank0(
                f"[mxfp4_healing][refaudit] sample[{idx}]: gc.get_referrers "
                f"failed ({type(exc).__name__}: {exc})"
            )
            continue

        try:
            shape = tuple(old_weight.shape)
            dtype = str(old_weight.dtype)
            wcls = type(old_weight).__name__
        except Exception:  # noqa: BLE001
            shape, dtype, wcls = ("?",), "?", type(old_weight).__name__

        _emit_rank0(
            f"[mxfp4_healing][refaudit] sample[{idx}] module={type(module).__name__} "
            f"weight={wcls} shape={shape} dtype={dtype} "
            f"sys.getrefcount={rc} (baseline ~3 = _parameters dict + "
            f"this local + gc.get_referrers arg)  "
            f"len(referrers)={len(referrers)}"
        )
        for r_idx, ref in enumerate(referrers[:12]):
            if ref is referrers or ref is samples or ref is _ORDERED_FP8_STASH:
                continue
            try:
                desc = _classify_referrer(ref, target_id)
            except Exception as exc:  # noqa: BLE001
                desc = f"<classify exception: {type(exc).__name__}: {exc}>"
            _emit_rank0(
                f"[mxfp4_healing][refaudit]   referrer[{r_idx}] {desc}"
            )
        if len(referrers) > 12:
            _emit_rank0(
                f"[mxfp4_healing][refaudit]   ... ({len(referrers) - 12} more referrers truncated)"
            )

        del referrers
        del old_weight


def _restore_fp8_weights_to_gpu(model: Any) -> int:
    if not _ORDERED_FP8_STASH:
        raise RuntimeError("[mxfp4_healing] Healing requested but FP8 stash is empty (PRE_QUANTIZED + stash failed?)")

    debug = _truthy_env("MXFP4_HEALING_DEBUG", default=False)
    # NeMo parity: ``ENABLE_TRANSPOSE_CACHE=0`` means the pre-quantize path
    # didn't allocate a columnwise (transpose) buffer on the stashed FP8
    # tensor, so there is nothing to hydrate. We still null the attr
    # defensively in case it's lingering from a previous run, then tell TE
    # to stop keeping a weight transpose cache for the rest of training.
    enable_tc = _enable_transpose_cache()
    dev = torch.cuda.current_device()
    xfer = torch.cuda.Stream()
    n_restored = 0
    total_bytes = 0
    if debug:
        _emit_rank0(
            f"[mxfp4_healing][debug] Restoring {len(_ORDERED_FP8_STASH)} FP8 weights from CPU -> "
            f"GPU (device={dev}) on dedicated CUDA stream "
            f"(ENABLE_TRANSPOSE_CACHE={int(enable_tc)})..."
        )
    for module, weight in _ORDERED_FP8_STASH:
        with torch.no_grad(), torch.cuda.stream(xfer):
            if debug:
                try:
                    total_bytes += int(weight._data.numel()) * int(weight._data.element_size())
                except Exception:  # noqa: BLE001
                    pass
            weight._data = weight._data.to(dev, non_blocking=True)
            if enable_tc:
                if getattr(weight, "_transpose", None) is not None:
                    weight._transpose = weight._transpose.to(dev, non_blocking=True)
            else:
                # NeMo ``_set_quantized_params_cpu`` lines 333-336: force
                # _transpose=None when ENABLE_TRANSPOSE_CACHE is off so TE
                # doesn't try to consume a stale / partially-hydrated cache.
                weight._transpose = None
            si = getattr(weight, "_scale_inv", None)
            if si is not None:
                weight._scale_inv = si.to(dev, non_blocking=True)
            weight.requires_grad = False
            module._parameters["weight"] = weight
            n_restored += 1
    xfer.synchronize()

    if not enable_tc:
        n_flagged = _disable_te_linear_transpose_cache(model)
        if debug:
            _emit_rank0(
                f"[mxfp4_healing][debug] ENABLE_TRANSPOSE_CACHE=0: set "
                f"keep_fp8_weight_transpose_cache=False on {n_flagged} TE modules."
            )

    if debug:
        _emit_rank0(
            f"[mxfp4_healing][debug] Restored {n_restored} FP8 weight tensors "
            f"(~{total_bytes / (1024 * 1024):.1f} MiB) to GPU."
        )
    return n_restored


def _maybe_rebuild_te_op_fuser_layers(model: Any) -> None:
    """NeMo rebuilds fused branches when ``use_transformer_engine_op_fuser`` (decoder.layers)."""
    use_fuser = False
    chunks = list(model) if isinstance(model, (list, tuple)) else [model]
    for ch in chunks:
        inner = ch
        seen = set()
        while hasattr(inner, "module") and id(inner) not in seen:
            seen.add(id(inner))
            inner = inner.module
        cfg = getattr(inner, "config", None)
        if cfg is not None and bool(getattr(cfg, "use_transformer_engine_op_fuser", False)):
            use_fuser = True
            break
    if not use_fuser:
        return

    for ch in chunks:
        inner = ch
        seen = set()
        while hasattr(inner, "module") and id(inner) not in seen:
            seen.add(id(inner))
            inner = inner.module
        dec = getattr(inner, "decoder", None)
        layers = getattr(dec, "layers", None) if dec is not None else None
        if not layers:
            continue
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "_make_fused_impl"):
                try:
                    mlp._fused_impl = (mlp._make_fused_impl(),)
                except Exception:  # noqa: BLE001
                    pass
            for attr in ("linear_proj", "linear_qkv"):
                mod = getattr(layer.self_attention, attr, None) if hasattr(layer, "self_attention") else None
                if mod is not None and hasattr(mod, "_make_fused_branches"):
                    try:
                        mod._fused_branches = mod._make_fused_branches()
                    except Exception:  # noqa: BLE001
                        pass


def _log_gpu_mem(tag: str) -> None:
    """Log allocated / reserved / max_alloc / max_reserved (rank-0 only).

    Same format as ``pre_quantize_mxfp4._log_gpu_mem`` so the two
    sequences compose into a single timeline you can diff. Tagged
    ``[mxfp4_healing][mem]`` so it's grep-distinct from the pre-quantize
    ``[pre_quantize_mxfp4][mem]`` lines.

    Gated by ``MXFP4_HEALING_DEBUG`` (default ``False``). Silent no-op unless
    debug is explicitly enabled.
    """
    if not _truthy_env("MXFP4_HEALING_DEBUG", default=False):
        return
    try:
        if not torch.cuda.is_available():
            return
        gib = 1024**3
        allocated = torch.cuda.memory_allocated() / gib
        reserved = torch.cuda.memory_reserved() / gib
        max_alloc = torch.cuda.max_memory_allocated() / gib
        max_reserved = torch.cuda.max_memory_reserved() / gib
        _emit_rank0(
            f"[mxfp4_healing][mem] {tag:<55s}"
            f" allocated={allocated:6.2f} GiB | reserved={reserved:6.2f} GiB"
            f" | max_alloc={max_alloc:6.2f} GiB | max_reserved={max_reserved:6.2f} GiB"
        )
    except Exception as exc:  # noqa: BLE001
        _emit_rank0(
            f"[mxfp4_healing][mem] {tag}: gpu mem read failed "
            f"({type(exc).__name__}: {exc})"
        )


def _force_release_unreferenced_storage(tag: str) -> None:
    """``gc.collect() + torch.cuda.empty_cache()`` and emit a mem checkpoint.

    Required at the MXFP4 -> FP8 healing transition: the old MXFP4
    ``module._parameters["weight"]`` tensors are unlinked when we install
    the restored FP8 weights, but PyTorch's caching allocator does not
    return that storage to the OS until something forces a flush. Without
    this, the FP8 restore (~64 GiB) stacks on top of the not-yet-freed
    MXFP4 weights (~75 GiB) and OOMs the next forward step. NeMo MLPerf
    happens to avoid the OOM because (i) their MXFP4 phase runs ~340
    steps before healing so the allocator pool has stabilized, and (ii)
    their ``_setup_partial_columnwise_cache`` intentionally pre-grows the
    pool to ~peak post-healing size during MXFP4, so the dereferenced
    MXFP4 slots get reused for FP8 + bigger activations. We don't have
    that pre-growth and heal at step 40, so we have to flush explicitly.
    """
    try:
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        _log_gpu_mem(tag)
    except Exception as exc:  # noqa: BLE001
        _emit_rank0(
            f"[mxfp4_healing][mem] {tag}: empty_cache failed "
            f"({type(exc).__name__}: {exc})"
        )


def _zero_grad_all(optimizer: Any, model: Any) -> None:
    """Best-effort: zero gradients on optimizer + every model parameter.

    Used after the FP8 warmup step to release gradient storage before
    restoring MXFP4 weights. Mirrors NeMo
    ``optimizer.zero_grad(set_to_none=True)``.

    .. WARNING::
        Megatron's distributed optimizer holds ``param.main_grad`` as a
        *persistent* view into a contiguous bucket allocated once at
        optimizer init. Setting ``param.main_grad = None`` breaks that view
        and the next forward crashes with ``'NoneType' object has no
        attribute 'dtype'`` when the grad-accumulation kernel reads
        ``param.main_grad.dtype``. So we never touch ``main_grad`` here —
        we only clear ``param.grad`` (the autograd accumulator, safe to
        None) and let the optimizer's own ``zero_grad`` decide how to handle
        ``main_grad`` (Megatron's distopt zeros the buffer in place rather
        than rebinding).
    """
    try:
        if optimizer is not None and hasattr(optimizer, "zero_grad"):
            try:
                optimizer.zero_grad(set_to_none=True)
            except TypeError:
                optimizer.zero_grad()
    except Exception:  # noqa: BLE001
        pass
    chunks = list(model) if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        try:
            for p in chunk.parameters():
                if getattr(p, "grad", None) is not None:
                    p.grad = None
        except Exception:  # noqa: BLE001
            pass


def _reset_te_fp8_global_recompute_state() -> None:
    """Wipe TE class-level FP8 recompute bookkeeping.

    The FP8 activation-recomputation path keeps two pieces of class-level
    state that must be paired across (first forward → recompute forward):

      * ``FP8GlobalStateManager.fp8_tensors_recompute_buffer`` — a list of
        deques. ``copy_forward_fp8_meta_tensors_for_recompute`` appends an
        entry on first forward, ``get_old_fp8_meta_tensors_for_recompute``
        pops it on recompute. Any unpopped entry is stale.
      * ``activation_recompute_forward._is_first_fp8_module`` — appended
        on first forward, popped on recompute.

    These are populated when the recipe is delayed-scaling. If a forward
    pass crashes mid-flight (as our FP8 warmup did), the entries are
    appended but the matching pop never runs. The next *real* training
    step then mixes stale and fresh entries, and the fp8_meta buffer
    position keys saved on per-module dicts point into now-shifted buffer
    indices. The result at recompute time is a ``KeyError:
    'global_fp8_buffer_pos_fwd_recompute'`` (when the key was never set
    for that module under the new recipe) or silently-wrong amax/scale
    snapshots (when the key points to a stale entry).

    Best-effort: walks every TE namespace it can find and resets these
    structures. Any failure on a given attribute is swallowed — losing
    one bookkeeping reset is far better than crashing the whole healing
    transition.
    """
    try:
        from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor  # noqa: F401
    except Exception:  # noqa: BLE001
        return  # TE not available; nothing to reset

    n_reset = 0
    try:
        from transformer_engine.pytorch.module.base import FP8GlobalStateManager

        if hasattr(FP8GlobalStateManager, "fp8_tensors_recompute_buffer"):
            FP8GlobalStateManager.fp8_tensors_recompute_buffer = []
            n_reset += 1
    except Exception:  # noqa: BLE001
        pass

    try:
        from transformer_engine.pytorch.distributed import (
            activation_recompute_forward,
        )

        if hasattr(activation_recompute_forward, "_is_first_fp8_module"):
            activation_recompute_forward._is_first_fp8_module = []
            n_reset += 1
    except Exception:  # noqa: BLE001
        pass

    if n_reset > 0 and _truthy_env("MXFP4_HEALING_DEBUG", default=False):
        _emit_rank0(
            f"[mxfp4_healing] Wiped {n_reset} TE class-level FP8 recompute bookkeeping "
            f"slots (FP8GlobalStateManager.fp8_tensors_recompute_buffer / "
            f"activation_recompute_forward._is_first_fp8_module)."
        )


def _reset_te_fp8_module_state(model: Any) -> None:
    """Reset per-module TE FP8 state so the next forward re-initializes cleanly.

    Mirrors NeMo's ``custom_llama.reset_fp8_state``
    (``mlperf-training-6-0/llama2_sft/nemo/src/custom_llama.py:77-83``):

    .. code-block:: python

        def reset_fp8(m):
            if hasattr(m, "fp8_initialized"):
                m.fp8_initialized = False
                m.reset_fp8_meta_tensors()

    Plus, we also drop the ``"global_fp8_buffer_pos_fwd_recompute"`` key
    from each module's ``fp8_meta`` (so the next ``copy_forward_fp8_meta_
    tensors_for_recompute`` allocates a fresh buffer position rather than
    appending to a stale one) and zero-out amax/scale buffers (defense in
    depth in case a downstream consumer expects clean state).
    """
    chunks = list(model) if isinstance(model, (list, tuple)) else [model]
    n_reinit = 0
    n_keys_dropped = 0
    n_amax_zeroed = 0
    for chunk in chunks:
        try:
            modules_iter = chunk.modules()
        except Exception:  # noqa: BLE001
            continue
        for m in modules_iter:
            # NeMo-equivalent reset
            try:
                if hasattr(m, "fp8_initialized"):
                    m.fp8_initialized = False
                    n_reinit += 1
                if hasattr(m, "reset_fp8_meta_tensors"):
                    m.reset_fp8_meta_tensors()
            except Exception:  # noqa: BLE001
                pass
            fp8_meta = getattr(m, "fp8_meta", None)
            if not isinstance(fp8_meta, dict):
                continue
            for key in (
                "global_fp8_buffer_pos_fwd_recompute",
                "updated_amax_history_fwd",
                "updated_scale_fwd",
            ):
                if key in fp8_meta:
                    try:
                        del fp8_meta[key]
                        n_keys_dropped += 1
                    except Exception:  # noqa: BLE001
                        pass
            for key, val in list(fp8_meta.items()):
                lkey = str(key).lower()
                if not any(tok in lkey for tok in ("amax", "history", "scale_inv", "scale")):
                    continue
                try:
                    if hasattr(val, "zero_"):
                        with torch.no_grad():
                            val.zero_()
                            n_amax_zeroed += 1
                    elif isinstance(val, dict):
                        for v2 in val.values():
                            if hasattr(v2, "zero_"):
                                with torch.no_grad():
                                    v2.zero_()
                                    n_amax_zeroed += 1
                except Exception:  # noqa: BLE001
                    continue
    if _truthy_env("MXFP4_HEALING_DEBUG", default=False):
        _emit_rank0(
            f"[mxfp4_healing] TE module reset: fp8_initialized=False on {n_reinit} modules, "
            f"dropped {n_keys_dropped} stale recompute/amax keys, zeroed {n_amax_zeroed} buffers."
        )


_TE_CHECKPOINT_FP8_PATCHED: bool = False
_ORIGINAL_TE_CHECKPOINT: Any = None


def _install_te_checkpoint_outer_fp8_autocast(model_config: Any) -> None:
    """Wrap ``transformer_block.te_checkpoint`` with an outer fp8_autocast.

    Background -- the bug this fixes
    -------------------------------
    Megatron's ``transformer_block.TransformerBlock.forward`` makes the
    ``outer_quantization_context`` choice based on ``config.fp8`` /
    ``config.fp4`` *flags*, not on the live recipe object. With
    ``config.fp4=True`` (which we never flip during healing -- only the
    recipe-builder is monkey-patched), the FP4 branch is taken::

        elif self.config.fp4:
            use_outer_quantization_context = False
            use_inner_quantization_context = True
            outer_quantization_context = nullcontext()

    So the entire block forward runs with ``outer = nullcontext`` and the
    fp8/fp4 autocast is established *per-layer*, *inside* each layer's
    ``custom_forward``. That ``custom_forward`` is what gets passed into
    ``te_checkpoint(custom_forward, ...)`` for the checkpointed layers.

    ``te_checkpoint`` (TE's ``transformer_engine.pytorch.distributed.checkpoint``)
    in turn does, in its ``_CheckpointFunction.forward``::

        forward_ctx, recompute_ctx = context_fn()  # default = noop -> nullcontext, nullcontext
        ...
        with torch.no_grad(), forward_ctx:
            with activation_recompute_forward(activation_recompute=True,
                                              recompute_phase=False):
                outputs = run_function(*args, **kwargs)

    Crucially, ``activation_recompute_forward.__enter__`` evaluates::

        _FP8_ACTIVATION_RECOMPUTE_ENABLED = (
            self.activation_recompute and FP8GlobalStateManager.is_fp8_enabled()
        )

    *at entry*. With ``forward_ctx = nullcontext`` and no surrounding
    fp8_autocast, ``is_fp8_enabled() == False`` at that moment, so
    ``_FP8_ACTIVATION_RECOMPUTE_ENABLED`` ends up ``False`` for the entire
    first-forward pass of the checkpointed layers. The per-layer
    ``inner_quantization_context = get_fp4_context(...)`` autocast is
    entered *inside* ``run_function``, but by then it's too late to flip
    the flag.

    Now look at what TE's ``Module.prepare_forward`` does with that flag
    (``transformer_engine/pytorch/module/base.py``)::

        # First forward pass.
        ...
        delayed_scaling_recipe = self.fp8 and self.fp8_meta["recipe"].delayed()
        if delayed_scaling_recipe:
            ...
            if self.training and is_fp8_activation_recompute_enabled():
                FP8GlobalStateManager.copy_forward_fp8_meta_tensors_for_recompute(
                    self.fp8_meta
                )

    With ``is_fp8_activation_recompute_enabled() == False``, the
    per-module ``copy_forward_fp8_meta_tensors_for_recompute`` is *never*
    called, so ``self.fp8_meta["global_fp8_buffer_pos_fwd_recompute"]``
    is never set. During the FP4 phase this is harmless: the recipe is
    ``NVFP4BlockScaling``, ``recipe.delayed() == False``, both
    ``copy_forward_*`` and ``get_old_*`` are short-circuited as no-ops.

    After healing, the recipe becomes ``DelayedScaling`` (via our
    monkey-patched ``get_fp4_recipe``). On the recompute pass during
    backward, ``in_fp8_activation_recompute_phase()`` *is* True (TE's
    ``CheckpointFunction.backward`` enters ``activation_recompute_forward(
    activation_recompute=True, recompute_phase=True)``), and
    ``prepare_forward`` takes the second-forward branch::

        if self.fp8 and in_fp8_activation_recompute_phase():
            delayed_scaling_recipe = self.fp8_meta["recipe"].delayed()
            FP8GlobalStateManager.get_old_fp8_meta_tensors_for_recompute(
                self.fp8_meta
            )

    With ``DelayedScaling``, ``get_old_fp8_meta_tensors_for_recompute``
    is no longer a no-op -- it tries to ``fp8_meta[
    "global_fp8_buffer_pos_fwd_recompute"]``. Key missing →
    ``KeyError: 'global_fp8_buffer_pos_fwd_recompute'`` at step
    ``HEALING_ITER``. Exactly what the user is hitting.

    NeMo MLPerf doesn't hit this because their fork of Megatron-LM
    pre-dates the FP4 branch in ``TransformerBlock.forward``; they
    only ever take the ``config.fp8`` path, where
    ``outer_quantization_context = get_fp8_context(...)`` *is* an
    fp8_autocast and ``activation_recompute_forward.__enter__`` sees
    ``is_fp8_enabled() == True``. ``copy_forward_*`` runs, the buffer
    position key gets populated, and recompute works as designed.

    The fix
    -------
    We monkey-patch the module-level binding
    ``megatron.core.transformer.transformer_block.te_checkpoint`` with a
    wrapper that *injects* a ``context_fn`` returning our own
    ``fp8_autocast(enabled=True, recipe=DelayedScaling, fp8_group=...)``
    for both the forward and recompute phases. This puts an outer FP8
    autocast around both phases of every checkpoint block, so:

      * On first forward, ``activation_recompute_forward.__enter__`` runs
        *inside* fp8_autocast → ``_FP8_ACTIVATION_RECOMPUTE_ENABLED ==
        True`` → ``copy_forward_*`` runs → buffer position key is set.
      * ``ctx.fp8 = is_fp8_enabled()`` (captured at line 387 of
        ``distributed.py``) is now ``True`` because we're still inside
        the outer autocast.
      * ``ctx.fp8_recipe = DelayedScaling`` is captured.
      * On recompute, ``ctx.recompute_ctx`` is also our fp8_autocast,
        which is entered before ``activation_recompute_forward(True,
        True)`` runs, so the inner ``autocast(enabled=ctx.fp8,
        recipe=ctx.fp8_recipe)`` finds matching state and the
        ``get_old_*`` lookup hits a populated key.

    Per-layer ``inner_quantization_context = get_fp4_context(...)`` (also
    DelayedScaling post-healing) is now *nested* inside our outer
    autocast. TE autocast handles nesting via stack save/restore in
    ``FP8GlobalStateManager.{get,set}_autocast_state``, so this is safe.

    We also store the original ``te_checkpoint`` and assert idempotency
    so repeated healing transitions (shouldn't happen in production but
    can during tests) don't double-wrap.
    """
    global _TE_CHECKPOINT_FP8_PATCHED, _ORIGINAL_TE_CHECKPOINT  # noqa: PLW0603

    if _TE_CHECKPOINT_FP8_PATCHED:
        if _truthy_env("MXFP4_HEALING_DEBUG", default=False):
            _emit_rank0(
                "[mxfp4_healing] te_checkpoint outer-fp8_autocast wrapper already installed; "
                "skipping re-install."
            )
        return

    try:
        from megatron.core.transformer import transformer_block as _tb
    except Exception as e:  # noqa: BLE001
        _emit_rank0(
            f"[mxfp4_healing] WARNING: cannot import megatron.core.transformer.transformer_block "
            f"to install te_checkpoint outer-fp8_autocast wrapper: {e!r}. "
            "Activation recompute will likely KeyError on the first FP8 backward."
        )
        return

    original = getattr(_tb, "te_checkpoint", None)
    if original is None or not callable(original):
        _emit_rank0(
            "[mxfp4_healing] WARNING: transformer_block.te_checkpoint is missing or not callable; "
            "outer-fp8_autocast wrapper not installed. "
            "Activation recompute will likely KeyError on the first FP8 backward."
        )
        return

    try:
        import transformer_engine.pytorch as te_pytorch
    except Exception as e:  # noqa: BLE001
        _emit_rank0(
            f"[mxfp4_healing] WARNING: cannot import transformer_engine.pytorch to install "
            f"te_checkpoint outer-fp8_autocast wrapper: {e!r}. "
            "Activation recompute will likely KeyError on the first FP8 backward."
        )
        return

    recipe_factory = _build_healing_lambda(model_config)

    def _resolve_amax_group() -> Any:
        try:
            from megatron.core import parallel_state

            if not parallel_state.model_parallel_is_initialized():
                return None
            tp_only = bool(getattr(model_config, "tp_only_amax_red", False))
            return parallel_state.get_amax_reduction_group(
                with_context_parallel=True, tp_only_amax_red=tp_only
            )
        except Exception:  # noqa: BLE001
            return None

    def _build_outer_fp8_context() -> Any:
        recipe = recipe_factory(model_config)
        group = _resolve_amax_group()
        return te_pytorch.fp8_autocast(
            enabled=True, fp8_recipe=recipe, fp8_group=group
        )

    def _outer_fp8_context_fn() -> Tuple[Any, Any]:
        # Returned to TE's ``_CheckpointFunction.forward``: the first ctx is
        # entered around the first forward, the second around the recompute
        # forward in backward. Each must be a *fresh* autocast instance because
        # contextmanager generators are single-use.
        return _build_outer_fp8_context(), _build_outer_fp8_context()

    def _wrapped_te_checkpoint(*args: Any, **kwargs: Any) -> Any:
        # Only inject our context_fn when we're actually past healing. Pre-healing
        # (FP4 phase) calls must keep their existing behavior because the per-layer
        # NVFP4 autocast is incompatible with an outer DelayedScaling wrap.
        if _HEALING_APPLIED and "context_fn" not in kwargs:
            kwargs["context_fn"] = _outer_fp8_context_fn
        return original(*args, **kwargs)

    _ORIGINAL_TE_CHECKPOINT = original
    _tb.te_checkpoint = _wrapped_te_checkpoint  # type: ignore[assignment]
    _TE_CHECKPOINT_FP8_PATCHED = True

    _emit_rank0(
        "[mxfp4_healing] Installed te_checkpoint outer-fp8_autocast wrapper "
        "(recipe=DelayedScaling). This puts an FP8 autocast around both forward and "
        "recompute phases of every checkpointed layer block, so TE's "
        "activation_recompute_forward sees is_fp8_enabled()=True at entry and "
        "copy_forward_fp8_meta_tensors_for_recompute / get_old_* pair up correctly. "
        "Required to avoid KeyError 'global_fp8_buffer_pos_fwd_recompute' on the "
        "first FP8 backward when config.fp4 is True (Megatron uses outer=nullcontext "
        "in that branch)."
    )


def _reset_te_fp8_state(model: Any) -> None:
    """Best-effort: zero TE FP8 metadata buffers (amax history, scale, scale_inv).

    NeMo MLPerf calls a ``reset_fp8_state`` helper after the warmup step so the
    real training run starts with an unpolluted amax history. We approximate
    by walking ``model.modules()`` and zero-ing every tensor in each TE
    module's ``fp8_meta`` whose key contains ``"amax"``, ``"history"``,
    ``"scale"``, or ``"scale_inv"``. Any failure on a given buffer is
    silently skipped — a slightly stale amax bucket is far better than
    losing the whole warmup.
    """
    chunks = list(model) if isinstance(model, (list, tuple)) else [model]
    n_reset = 0
    for chunk in chunks:
        try:
            modules_iter = chunk.modules()
        except Exception:  # noqa: BLE001
            continue
        for m in modules_iter:
            fp8_meta = getattr(m, "fp8_meta", None)
            if not isinstance(fp8_meta, dict):
                continue
            for key, val in list(fp8_meta.items()):
                lkey = str(key).lower()
                if not any(tok in lkey for tok in ("amax", "history", "scale_inv", "scale")):
                    continue
                try:
                    if hasattr(val, "zero_"):
                        with torch.no_grad():
                            val.zero_()
                            n_reset += 1
                    elif isinstance(val, dict):
                        for k2, v2 in val.items():
                            if hasattr(v2, "zero_"):
                                with torch.no_grad():
                                    v2.zero_()
                                    n_reset += 1
                except Exception:  # noqa: BLE001
                    continue
    if n_reset > 0 and _truthy_env("MXFP4_HEALING_DEBUG", default=False):
        _emit_rank0(
            f"[mxfp4_healing] FP8 warmup: reset {n_reset} TE FP8 metadata buffers."
        )


def _build_nemo_synthetic_data_iterator(
    seq_length: int,
    micro_batch_size: int,
) -> Any:
    """Build a synthetic data iterator that matches NeMo MLPerf byte-for-byte.

    Reproduces ``CustomLlamaModel.get_synthetic_input`` from
    ``mlperf-training-6-0/llama2_sft/nemo/src/custom_llama.py:101-132``
    verbatim for the per-sample tensors, then wraps it in
    ``itertools.repeat`` so the Megatron-Bridge
    ``get_batch_from_iterator`` loop can pull the same synthetic microbatch
    on every ``next()`` call.

    Per-sample tensor schema (identical to NeMo):

      * ``tokens``        : shape ``[seq_length]``,        dtype int64,
        value ``3545`` everywhere except ``tokens[-1] = 2``.
      * ``labels``        : shape ``[seq_length]``,        dtype int64,
        value ``3545`` everywhere (NeMo's ``text[1:]``) with
        ``labels[-1] = 2``.
      * ``loss_mask``     : shape ``[seq_length]``,        dtype int64,
        all ones except ``loss_mask[-1] = 0``.
      * ``position_ids``  : shape ``[seq_length]``,        dtype int64,
        ``[0, 1, ..., seq_length-1]`` with ``position_ids[-1] = 0``.
      * ``attention_mask``: shape ``[1, seq_length, seq_length]``, dtype
        bool, all ones.

    Two unavoidable deltas vs NeMo (different host-framework contracts —
    see docstring of ``run_fp8_warmup_for_kernel_jit`` for the rationale):

      1. NeMo's helper collates ``get_num_microbatches()`` samples into
         one big ``[gbs, seq_len]`` batch because PyTorch-Lightning's
         ``training_step`` splits a global batch into microbatches
         internally. Megatron-Bridge's ``get_batch_from_iterator`` is
         instead called once *per microbatch* and wants each ``next()``
         to yield a ``[mbs, seq_len]`` dict, so we collate
         ``micro_batch_size`` samples here.
      2. NeMo wraps the batch as ``(batch, 0, 0)`` (Lightning's
         ``(batch, batch_idx, dataloader_idx)`` contract).
         ``get_batch_from_iterator`` does ``next(iterator)`` then
         ``for key, val in batch.items()`` — a tuple has no ``.items()``
         and would crash, so we yield just the dict.
    """
    import itertools

    from torch.utils.data import default_collate

    text = torch.ones(seq_length + 1, dtype=torch.int64) * 3545

    tokens = text[:-1].contiguous()
    tokens[-1] = 2

    labels = text[1:].contiguous()
    labels[-1] = 2

    attention_mask_shape = [1, seq_length, seq_length]
    attention_mask = torch.ones(attention_mask_shape, dtype=torch.bool)

    loss_mask = torch.ones(seq_length, dtype=torch.int64)
    loss_mask[-1] = 0

    position_ids = torch.tensor(
        [i for i in range(seq_length)], dtype=torch.int64
    )
    position_ids[-1] = 0

    single_data = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }

    batch = default_collate([single_data] * micro_batch_size)
    return itertools.repeat(batch)


def run_fp8_warmup_for_kernel_jit(
    model: Any,
    model_config: Any,
    optimizer: Any,
    forward_step_func: Any,
    state: Any = None,
) -> None:
    """NeMo MLPerf-equivalent of ``CustomCallback._pre_ttt_fp8_warmup``.

    Called once at startup, AFTER ``stash_fp8_ordered_before_mxfp4`` and the
    MXFP4 pre-quantize swap, BEFORE the real training loop's first iteration.
    Runs exactly one synthetic forward+backward in FP8 mode so that:

      1. **FP8 GEMM kernels (Triton, hipBLASLt) get JIT-compiled and cached.**
         Without this, the first forward at ``HEALING_ITER`` stalls ~5-10 s
         while every FP8 kernel shape gets compiled on demand. NeMo measures
         this saving as ~10 s on step 310 (their healing iter is 340).

      2. **The PyTorch caching allocator sees the FP8 + larger-activation
         memory layouts at least once**, tuning split-fraction / segment-shape
         heuristics for FP8-shaped allocations. (The final ``empty_cache()``
         returns the pool to the OS, so this is more of a tuning hint than a
         persistent reservation.)

    Steps (mirrors NeMo ``custom_callbacks.py:260-344`` +
    ``custom_llama.py:101-132`` for the synthetic input):

      a. Build a synthetic data iterator via
         ``_build_nemo_synthetic_data_iterator`` whose per-sample tensors are
         **byte-identical** to NeMo's ``CustomLlamaModel.get_synthetic_input``
         (token id 3545 except last=2, position_ids 0..seq_len-1 with last=0,
         all-ones bool attention_mask, etc.). Synthetic data — does NOT consume
         a real microbatch from the training iterator.
      b. Move FP8 weights from CPU stash to GPU as **fresh** ``Float8Tensor``
         objects (the CPU stash is left untouched so the real healing event
         can still use it). Save the existing MXFP4 weights to a list.
      c. Install the GPU FP8 weights into ``module._parameters['weight']``,
         displacing the MXFP4 weights (which remain alive via the saved
         list -> both weight sets live on GPU at once, ~142 GiB).
      d. Patch ``megatron.core.fp4_utils.{_mxfp4_phase,get_fp4_recipe}`` to
         FP8 (``DelayedScaling`` or ``MXFP8BlockScaling``).
      e. Force aggressive activation recompute (default
         ``MXFP4_HEALING_FP8_WARMUP_RECOMPUTE_N=72`` -> ~72 of 80 layers
         recomputed, only 8 layers' activations saved) so the synthetic
         step actually fits with both weight sets resident.
      f. Run **one** microbatch through ``get_forward_backward_func``
         (``forward_only=False`` so backward + grad allocation happens).
         Wrapped in try/except: any failure is logged and the warmup is
         aborted (real training is unaffected).
      g. Zero gradients on the optimizer and all parameters.
      h. Restore MXFP4 weights into ``module._parameters['weight']``,
         dropping references to the GPU FP8 tensors (they get freed).
      i. Restore the saved fp4_utils recipe + recompute config.
      j. Best-effort reset of TE FP8 metadata buffers (amax history, scales).
      k. ``cuda.synchronize()`` + ``cuda.empty_cache()``.

    Honors env vars:

      * ``MXFP4_HEALING_FP8_WARMUP=0`` -> skip the warmup entirely.
      * ``MXFP4_HEALING_FP8_WARMUP_RECOMPUTE_N=<int>`` -> override the
        ``recompute_num_layers`` used for the synthetic step (default 72).
      * ``MXFP4_HEALING_FP8_WARMUP_SEQ_LEN=<int>`` -> override the synthetic
        sequence length (default: NeMo's hardcoded 8192, falling back to
        ``model_config.seq_length`` if the model wants a different seq_len
        to avoid shape mismatch).
    """
    if healing_iter() <= 0:
        return
    if not _ORDERED_FP8_STASH:
        return
    # Default OFF: the synthetic warmup step has been observed to crash
    # consistently (most recently with
    # ``AssertionError: Expected _transpose to be None or an empty tensor when
    # transpose cache is disabled.`` from TE -- our ``no_fp8_weight_transpose_cache``
    # config conflicts with the warmup's _transpose carrying-over from CPU
    # stash). Even when it crashes mid-forward, it leaves stale entries in
    # TE's class-level FP8 recompute bookkeeping, which then poisons the
    # real step 40 forward with
    # ``KeyError: 'global_fp8_buffer_pos_fwd_recompute'``. Until the warmup
    # step is fixed to *succeed* under the pre-quantize transpose-cache
    # constraint, opting in is unsafe. Set MXFP4_HEALING_FP8_WARMUP=1 to
    # force-enable.
    if not _truthy_env("MXFP4_HEALING_FP8_WARMUP", default=False):
        if _truthy_env("MXFP4_HEALING_DEBUG", default=False):
            _emit_rank0(
                "[mxfp4_healing] FP8 warmup is OFF by default (set MXFP4_HEALING_FP8_WARMUP=1 "
                "to opt in -- but the synthetic step currently crashes against "
                "no_fp8_weight_transpose_cache=true and pollutes TE recompute state)."
            )
        return

    try:
        import time

        import megatron.core.fp4_utils as fp4u
        import transformer_engine_torch as tex
        from megatron.core.pipeline_parallel import get_forward_backward_func
        from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
    except Exception as imp_err:  # noqa: BLE001
        _emit_rank0(
            f"[mxfp4_healing] FP8 warmup: imports failed "
            f"({type(imp_err).__name__}: {imp_err}); skipping warmup."
        )
        return

    debug = _truthy_env("MXFP4_HEALING_DEBUG", default=False)
    aggressive_n = int(os.getenv("MXFP4_HEALING_FP8_WARMUP_RECOMPUTE_N", "80"))

    _emit_rank0("[mxfp4_healing] " + "=" * 70)
    _emit_rank0(
        "[mxfp4_healing] >>> FP8 WARMUP <<< pre-compiling FP8 kernels "
        "(NeMo _pre_ttt_fp8_warmup parity) — one synthetic FP8 step before iter 1."
    )
    _emit_rank0("[mxfp4_healing] " + "=" * 70)

    _log_gpu_mem("FP8 warmup: entry (MXFP4 weights live)")

    t0 = time.perf_counter()
    dev = torch.cuda.current_device()
    xfer_stream = torch.cuda.Stream()

    saved_weights: List[Tuple[Any, Any]] = []
    gpu_fp8_holders: List[Any] = []  # keep refs alive across the step
    n_swapped = 0
    # NeMo ``_pre_ttt_fp8_warmup`` lines 393-396 + 419-422: when
    # ENABLE_TRANSPOSE_CACHE is off, skip hydrating the columnwise buffer to
    # GPU during the per-module swap and tell TE to not keep a transpose
    # cache for the synthetic FP8 forward/backward.
    enable_tc = _enable_transpose_cache()
    try:
        for module, cpu_fp8 in _ORDERED_FP8_STASH:
            with torch.no_grad(), torch.cuda.stream(xfer_stream):
                gpu_data = cpu_fp8._data.to(dev, non_blocking=True)
                if enable_tc:
                    gpu_tp = (
                        cpu_fp8._transpose.to(dev, non_blocking=True)
                        if getattr(cpu_fp8, "_transpose", None) is not None
                        else None
                    )
                else:
                    gpu_tp = None
                gpu_si = (
                    cpu_fp8._scale_inv.to(dev, non_blocking=True)
                    if getattr(cpu_fp8, "_scale_inv", None) is not None
                    else None
                )
                gpu_fp8 = Float8Tensor(
                    shape=cpu_fp8.shape,
                    dtype=cpu_fp8.dtype,
                    data=gpu_data,
                    fp8_scale_inv=gpu_si,
                    fp8_dtype=getattr(cpu_fp8, "_fp8_dtype", tex.DType.kFloat8E4M3),
                    data_transpose=gpu_tp,
                    quantizer=getattr(cpu_fp8, "_quantizer", None),
                )
                gpu_fp8.requires_grad = False
            saved_weights.append((module, module._parameters["weight"]))
            module._parameters["weight"] = gpu_fp8
            gpu_fp8_holders.append(gpu_fp8)
            n_swapped += 1
        xfer_stream.synchronize()
    except Exception as exc:  # noqa: BLE001
        _emit_rank0(
            f"[mxfp4_healing] FP8 warmup: weight swap failed "
            f"({type(exc).__name__}: {exc}); restoring MXFP4 and aborting warmup."
        )
        for module, mxfp4_weight in saved_weights:
            module._parameters["weight"] = mxfp4_weight
        return

    if debug:
        _emit_rank0(
            f"[mxfp4_healing] FP8 warmup: swapped {n_swapped} module weights "
            f"MXFP4 -> FP8 (both sets live on GPU)."
        )
    _log_gpu_mem("FP8 warmup: after MXFP4+FP8 weight install (BOTH sets live)")

    # Patch fp4_utils to FP8 mode for the warmup step.
    saved_phase = fp4u._mxfp4_phase  # type: ignore[attr-defined]
    saved_recipe = fp4u.get_fp4_recipe  # type: ignore[attr-defined]
    healing_lambda = _build_healing_lambda(model_config)
    fp4u._mxfp4_phase = False  # type: ignore[attr-defined]
    fp4u.get_fp4_recipe = healing_lambda  # type: ignore[method-assign]

    if not enable_tc:
        n_flagged = _disable_te_linear_transpose_cache(model)
        if debug:
            _emit_rank0(
                f"[mxfp4_healing] FP8 warmup: ENABLE_TRANSPOSE_CACHE=0: set "
                f"keep_fp8_weight_transpose_cache=False on {n_flagged} TE modules "
                f"(NeMo _pre_ttt_fp8_warmup parity)."
            )

    # Force aggressive activation recompute. With both MXFP4+FP8 weights
    # resident (~142 GiB), the synthetic step needs minimal activation
    # memory to fit. NeMo uses recompute_num_layers=72.
    saved_recompute_g = getattr(model_config, "recompute_granularity", None)
    saved_recompute_n = getattr(model_config, "recompute_num_layers", None)
    saved_recompute_m = getattr(model_config, "recompute_method", None)
    try:
        model_config.recompute_granularity = "full"
        model_config.recompute_method = "block"
        model_config.recompute_num_layers = aggressive_n
    except Exception:  # noqa: BLE001
        pass

    # NeMo MLPerf hard-codes seq_length=8192 in
    # ``CustomLlamaModel.get_synthetic_input``; do the same to match exactly.
    # Allow override only as an escape hatch for non-MLPerf seq lengths
    # (MXFP4_HEALING_FP8_WARMUP_SEQ_LEN). Falling back to model_config.seq_length
    # is the last resort.
    env_seq = os.getenv("MXFP4_HEALING_FP8_WARMUP_SEQ_LEN")
    if env_seq is not None:
        seq_length = int(env_seq)
    else:
        seq_length = 8192
        cfg_seq = getattr(model_config, "seq_length", None)
        if cfg_seq is not None and int(cfg_seq) != seq_length:
            _emit_rank0(
                f"[mxfp4_healing] FP8 warmup: model_config.seq_length="
                f"{int(cfg_seq)} differs from NeMo synthetic seq_length=8192; "
                f"using model_config value to avoid shape mismatch. "
                f"Override with MXFP4_HEALING_FP8_WARMUP_SEQ_LEN if you really "
                f"want NeMo's exact 8192."
            )
            seq_length = int(cfg_seq)
    micro_batch_size = int(getattr(model_config, "micro_batch_size", 1) or 1)

    synthetic_iter = _build_nemo_synthetic_data_iterator(
        seq_length=seq_length, micro_batch_size=micro_batch_size
    )

    if debug:
        _emit_rank0(
            f"[mxfp4_healing] FP8 warmup: running 1 microbatch of NeMo "
            f"synthetic data (token_id=3545, seq_len={seq_length}, "
            f"mbs={micro_batch_size}, recompute_num_layers={aggressive_n}) "
            f"via get_forward_backward_func()..."
        )

    # Megatron-Bridge's gpt_step.forward_step has signature
    # ``(state, data_iterator, model)``. ``train()`` would normally wrap it
    # via ``prepare_forward_step_func`` to inject the GlobalState, but our
    # pre-quantize wrapper runs *before* train() reaches that wrapping
    # point, so ``forward_step_func`` here is still the raw 3-arg version.
    # We have to do the wrapping ourselves; otherwise the warmup step
    # crashes with "TypeError: forward_step() missing 1 required
    # positional argument: 'model'" because get_forward_backward_func calls
    # the function with just (data_iterator, model).
    wrapped_forward_step_func = forward_step_func
    if state is not None:
        try:
            from megatron.bridge.training.utils.train_utils import (
                prepare_forward_step_func,
            )

            wrapped_forward_step_func = prepare_forward_step_func(
                forward_step_func, state
            )
        except Exception as exc:  # noqa: BLE001
            _emit_rank0(
                f"[mxfp4_healing] FP8 warmup: prepare_forward_step_func "
                f"unavailable ({type(exc).__name__}: {exc}); calling raw "
                f"forward_step_func directly (will likely fail signature check)."
            )
    else:
        _emit_rank0(
            "[mxfp4_healing] FP8 warmup: state was not passed; cannot inject "
            "GlobalState into forward_step_func. The synthetic step will "
            "likely fail with a signature error (best-effort)."
        )

    step_ok = False
    t_step_start = time.perf_counter()
    try:
        fwd_bwd_func = get_forward_backward_func()
        fwd_bwd_func(
            forward_step_func=wrapped_forward_step_func,
            data_iterator=synthetic_iter,
            model=model,
            num_microbatches=1,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
        )
        step_ok = True
    except Exception as exc:  # noqa: BLE001
        _emit_rank0(
            f"[mxfp4_healing] FP8 warmup: synthetic step failed "
            f"(best-effort, real training unaffected): "
            f"{type(exc).__name__}: {exc}"
        )
    t_step_end = time.perf_counter()

    if step_ok and debug:
        _emit_rank0(
            f"[mxfp4_healing] FP8 warmup: synthetic step completed in "
            f"{t_step_end - t_step_start:.2f}s."
        )
    _log_gpu_mem("FP8 warmup: after synthetic FP8 step (peak memory)")

    # Drop gradient storage produced by the warmup step. Only call the
    # optimizer's zero_grad when the step actually ran -- if the step failed
    # we never produced any grads, and calling optimizer.zero_grad on a
    # not-yet-stepped distributed optimizer can prematurely materialize its
    # bucket bookkeeping.
    if step_ok:
        _zero_grad_all(optimizer, model)

    # Restore MXFP4 weights; drop references to GPU FP8 tensors so they free.
    for module, mxfp4_weight in saved_weights:
        module._parameters["weight"] = mxfp4_weight
    gpu_fp8_holders.clear()

    # Restore fp4_utils recipe + recompute config.
    fp4u._mxfp4_phase = saved_phase  # type: ignore[attr-defined]
    fp4u.get_fp4_recipe = saved_recipe  # type: ignore[method-assign]
    try:
        model_config.recompute_granularity = saved_recompute_g
        model_config.recompute_num_layers = saved_recompute_n
        model_config.recompute_method = saved_recompute_m
    except Exception:  # noqa: BLE001
        pass

    # Reset TE FP8 amax/scale buffers so real training starts clean.
    _reset_te_fp8_state(model)

    # Final flush: this is what NeMo does (custom_callbacks.py:343-344).
    # Combined with a synchronize() to make sure all the warmup-step kernels
    # have finished before we tear down their tensors.
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    t_total = time.perf_counter() - t0
    _log_gpu_mem("FP8 warmup: after restore + empty_cache (back in MXFP4)")
    _emit_rank0(
        f"[mxfp4_healing] >>> FP8 WARMUP COMPLETE <<< total {t_total:.2f}s "
        f"(swap {n_swapped} weights -> {'OK' if step_ok else 'STEP FAILED'} -> restore MXFP4). "
        f"Model now back in MXFP4 phase; iter 1 begins next."
    )
    _emit_rank0("[mxfp4_healing] " + "=" * 70)


def apply_healing_after_step(
    model: Any,
    model_config: Any,
    train_state_step: int,
) -> None:
    """NeMo ``on_train_batch_end`` when ``global_step + 1 == healing_iter``.

    ``train_state_step`` must be the Megatron ``train_state.step`` value **after**
    the training step increment (same convention as NeMo's ``trainer.global_step``).
    """
    global _HEALING_APPLIED  # noqa: PLW0603
    import time

    hi = healing_iter()
    debug = _truthy_env("MXFP4_HEALING_DEBUG", default=False)

    if hi <= 0 or _HEALING_APPLIED:
        return
    if train_state_step + 1 != hi:
        return

    if not _ORDERED_FP8_STASH:
        _emit_rank0(
            "[mxfp4_healing] HEALING_ITER is set but the FP8 CPU stash is empty. "
            "Enable PRE_QUANTIZED_MODEL=True so pre-quantize runs the MXFP4 path with "
            "FP8 stashing, or set HEALING_ITER=0. Skipping healing."
        )
        _HEALING_APPLIED = True
        return

    import torch.distributed as dist

    t0 = time.perf_counter()
    if dist.is_initialized():
        if debug:
            _emit_rank0(
                f"[mxfp4_healing][debug] step={train_state_step}: entering all-reduce "
                f"barrier before healing (world_size={dist.get_world_size()})."
            )
        dist.barrier()

    _emit_rank0(
        "[mxfp4_healing] " + "=" * 70 + "\n"
        f"[mxfp4_healing] >>> HEALING TRIGGERED <<< train_state.step={train_state_step} "
        f"(step+1==HEALING_ITER={hi}): MXFP4 phase END — restoring FP8 weights, "
        f"fp4_utils recipe -> {healing_precision()}, clearing MXFP4 caches. "
        f"Subsequent steps: active=FP8.\n"
        "[mxfp4_healing] " + "=" * 70
    )

    _log_gpu_mem("entry (MXFP4 phase end, before FP8 restore)")

    # Referrer audit (off by default; gated by MXFP4_HEALING_REFCOUNT_AUDIT).
    # Use this when investigating whether a non-_parameters container is
    # holding the MXFP4 storage alive across the per-module swap (which is
    # the real cause of memory pressure at healing, NOT healing step number).
    # Expected baseline refcount with NO extra holders is 3:
    #   - module._parameters['weight']
    #   - the local 'old_weight' var inside _audit_mxfp4_weight_referrers
    #   - the gc.get_referrers call's own argument
    # Anything above 3 means something else (distopt main_param map, DDP
    # bucket, TE workspace, LoRA wrapper attribute, ...) is keeping the
    # MXFP4 storage live. The classified referrer dump tells you which.
    _audit_mxfp4_weight_referrers(
        model,
        tag="BEFORE FP8 install (MXFP4 weights still on module._parameters)",
        sample_count=int(os.getenv("MXFP4_HEALING_REFCOUNT_SAMPLES", "2")),
    )

    t_restore_start = time.perf_counter()
    n_restored = _restore_fp8_weights_to_gpu(model)
    t_restore_end = time.perf_counter()
    _log_gpu_mem("after FP8 restore (MXFP4 still in caching alloc)")

    # Same probe AFTER the swap. Now ``module._parameters['weight']`` is the
    # FP8 tensor; if anything still references the OLD MXFP4 weight we
    # captured above, this audit will surface it. (We re-resolve the
    # current weight, so this dump is on the FP8 tensor — useful as a
    # sanity check that no stale MXFP4 ref leaked into FP8's referrers
    # via shared workspace caches, op-fuser branches, etc.)
    _audit_mxfp4_weight_referrers(
        model,
        tag="AFTER FP8 install (module._parameters['weight'] is now FP8)",
        sample_count=int(os.getenv("MXFP4_HEALING_REFCOUNT_SAMPLES", "2")),
    )

    import megatron.core.fp4_utils as fp4u

    healing_lambda = _build_healing_lambda(model_config)
    fp4u.get_fp4_recipe = healing_lambda  # type: ignore[method-assign]
    fp4u._mxfp4_phase = False  # type: ignore[attr-defined]
    if debug:
        _emit_rank0(
            f"[mxfp4_healing][debug] Patched megatron.core.fp4_utils.get_fp4_recipe -> "
            f"{healing_precision()} and set fp4_utils._mxfp4_phase=False."
        )

    # Activation recompute policy at the healing transition.
    #
    # MXFP4 phase keeps activation recompute (full / block / 8) -- it's cheap
    # in MXFP4 because weight memory is ~3.5x smaller than FP8 and the
    # transformer-block-level recompute saves the bulk of activations.
    # FP8 phase (post-healing) drops activation recompute entirely. Two
    # reasons:
    #
    #   1. NeMo MLPerf parity. NeMo's CustomCallback._healing_setup
    #      explicitly clears recompute_granularity / method / num_layers when
    #      switching to FP8 because the recompute path doesn't pair cleanly
    #      with TE's FP8 kernels (CUDA-graph capture, transpose cache reuse,
    #      and the FP8 amax/scale recompute bookkeeping all interact in
    #      subtle ways).
    #   2. Throughput. FP8 forward kernels are ~2x faster than recomputed
    #      backward forward; keeping recompute on during FP8 throws away
    #      most of the FP8 perf win.
    #
    # Memory cost on this GPU (Llama-2 70B, MBS=1, seq=4096):
    #
    #   * MXFP4 phase peak (with recompute=full/block/8): ~65 GiB weights +
    #     ~137 GiB activations = ~202 GiB max_alloc.
    #   * FP8 phase, recompute=OFF: ~65 GiB weights + ~208 GiB activations =
    #     ~273 GiB. Within the ~277 GiB usable budget (after NCCL overhead),
    #     but very tight. If you OOM here, opt back into preserve-recompute
    #     with ``MXFP4_HEALING_KEEP_RECOMPUTE=1`` (gets you back to ~202 GiB
    #     peak at the cost of ~50% FP8 throughput) or reduce
    #     ``recompute_num_layers`` in the YAML so MXFP4 phase peak is lower
    #     (which raises the floor that FP8 phase has to fit under).
    if _truthy_env("MXFP4_HEALING_KEEP_RECOMPUTE", default=False):
        rg = getattr(model_config, "recompute_granularity", None)
        rn = getattr(model_config, "recompute_num_layers", None)
        rm = getattr(model_config, "recompute_method", None)
        _emit_rank0(
            f"[mxfp4_healing] MXFP4_HEALING_KEEP_RECOMPUTE=1 set: keeping activation "
            f"recompute as-configured for post-healing FP8 phase: granularity={rg!r}, "
            f"method={rm!r}, num_layers={rn!r}. FP8 throughput will be lower than "
            f"recompute=OFF, but memory peak is the same as MXFP4 phase."
        )
    else:
        rg_old = getattr(model_config, "recompute_granularity", None)
        rn_old = getattr(model_config, "recompute_num_layers", None)
        rm_old = getattr(model_config, "recompute_method", None)
        if rg_old is not None or rn_old is not None or rm_old is not None:
            model_config.recompute_granularity = None
            model_config.recompute_num_layers = None
            model_config.recompute_method = None
            _emit_rank0(
                f"[mxfp4_healing] Disabled activation recompute for post-healing FP8 "
                f"phase (was: granularity={rg_old!r}, method={rm_old!r}, "
                f"num_layers={rn_old!r} -> all None). MXFP4 phase keeps recompute, "
                f"FP8 phase runs full activations for max throughput. Set "
                f"MXFP4_HEALING_KEEP_RECOMPUTE=1 if this OOMs on your GPU."
            )
        else:
            _emit_rank0(
                "[mxfp4_healing] Recompute already disabled in model_config; "
                "post-healing FP8 phase will continue without recompute."
            )

    n_caches_cleared = 0
    for m in _all_modules_depth_first(model):
        if hasattr(m, "_mxfp4_weight_cache"):
            try:
                del m._mxfp4_weight_cache
                n_caches_cleared += 1
            except Exception:  # noqa: BLE001
                pass
        if hasattr(m, "_mxfp4_persist_columnwise"):
            try:
                del m._mxfp4_persist_columnwise
                n_caches_cleared += 1
            except Exception:  # noqa: BLE001
                pass
    if debug:
        _emit_rank0(
            f"[mxfp4_healing][debug] Cleared {n_caches_cleared} MXFP4 cache attributes "
            f"across model modules."
        )

    # Symmetric undo of Megatron-LM's ``set_save_original_input(...)``. During
    # the MXFP4 phase, ``Megatron-LM/megatron/core/transformer/attention.py``
    # (and ``transformer_layer.py`` under CUDA-graph + pre_mlp_layernorm
    # recompute) sets ``module.save_original_input = True`` on every TE
    # ``linear_proj`` (and possibly ``mlp.linear_fc1``) when ``config.fp4`` is
    # active. This optimization is only legal under MX-style block-scaled
    # recipes -- the per-block scale is reproducible from the BF16 input, so
    # TE can re-quantize at backward time. ``DelayedScaling`` derives its FP8
    # scale from amax history that is updated each step, so the saved BF16
    # input cannot reproduce the forward FP8 input. TE guards against this
    # combination with a hard error::
    #
    #   RuntimeError: DelayedScaling recipe is not supported with
    #   save_original_input
    #
    # ...which fires on the very next forward after we patch
    # ``get_fp4_recipe -> FP8_DS``. NeMo MLPerf doesn't hit this because
    # their (older) Megatron-LM lacks the FP4 branch that sets the flag in
    # the first place. We mirror that behaviour by resetting the flag here.
    # Memory cost: ~1 byte/elem of ``linear_proj`` activations (the FP8 input
    # is now saved alongside the BF16 input from fused core_attn). For
    # Llama-2 70B, MBS=1, seq=4096, hidden=8192 that is ~33 MiB/layer x 80
    # layers ~= 2.6 GiB total, which is small next to the multi-tens-of-GiB
    # delta from disabling activation recompute above.
    n_save_orig_input_cleared = 0
    for m in _all_modules_depth_first(model):
        if getattr(m, "save_original_input", False):
            try:
                m.save_original_input = False
                n_save_orig_input_cleared += 1
            except Exception:  # noqa: BLE001
                pass
    _emit_rank0(
        f"[mxfp4_healing] Reset save_original_input=False on "
        f"{n_save_orig_input_cleared} TE modules (DelayedScaling parity)."
    )

    _maybe_rebuild_te_op_fuser_layers(model)

    _log_gpu_mem("after cache + save_original_input cleanup (BEFORE flush)")

    # Unconditional flush after healing. The MXFP4 weights that were just
    # unlinked from ``module._parameters['weight']`` (and any TE workspace
    # caches that were attached to them) are unreferenced now, but PyTorch's
    # caching allocator does not return that storage to the device pool
    # until a flush happens. Without this, the next forward (now FP8 +
    # full activations because we just disabled recompute) stacks on top
    # of ~75 GiB of stale MXFP4 storage and OOMs at ~272 GiB allocated
    # (observed in healing_debug.log @ step 40). NeMo MLPerf doesn't need
    # this because they (a) heal at step 340 with a steady-state allocator
    # pool, and (b) intentionally pre-grow the pool with
    # ``_setup_partial_columnwise_cache`` so the dereferenced MXFP4 slots
    # are re-used by FP8 + activations. We have neither, so we flush.
    _force_release_unreferenced_storage("after empty_cache (post-healing steady state)")

    # Reset TE FP8 state: both class-level recompute bookkeeping and per-module
    # fp8_meta / fp8_initialized flags. This is required because:
    #   1. The optional FP8 warmup may have crashed mid-forward, leaving
    #      orphan entries in ``FP8GlobalStateManager.fp8_tensors_recompute_buffer``
    #      and ``activation_recompute_forward._is_first_fp8_module``. Without
    #      this reset, step 40's first FP8 forward mixes warmup-leftover entries
    #      with fresh ones and step 40 backward crashes with
    #      ``KeyError: 'global_fp8_buffer_pos_fwd_recompute'``.
    #   2. Even without warmup, MXFP4 phase steps populated ``fp8_initialized``
    #      against the MXFP4 recipe. We just switched to DelayedScaling -- the
    #      recipe-change branch in TE's ``init_fp8_metadata`` only updates
    #      ``meta["recipe"]`` and clears ``_fp8_workspaces``; it leaves any
    #      pre-existing buffer-position keys in fp8_meta untouched. Forcing
    #      ``fp8_initialized=False`` and dropping stale recompute keys gives
    #      step 40 a clean slate, mirroring NeMo's
    #      ``custom_llama.reset_fp8_state``.
    _reset_te_fp8_global_recompute_state()
    _reset_te_fp8_module_state(model)

    # Install the te_checkpoint outer-fp8_autocast wrapper. This is the
    # actual fix for ``KeyError: 'global_fp8_buffer_pos_fwd_recompute'`` on
    # the first FP8 backward post-healing. See the docstring of
    # ``_install_te_checkpoint_outer_fp8_autocast`` for the full root-cause
    # analysis. Short version: with ``config.fp4=True`` Megatron's
    # ``TransformerBlock.forward`` uses ``outer = nullcontext`` and only
    # establishes the fp8 autocast *per-layer* inside ``custom_forward``,
    # which means TE's ``activation_recompute_forward.__enter__`` evaluates
    # ``is_fp8_enabled() == False`` and silently disables recompute
    # tracking. ``copy_forward_fp8_meta_tensors_for_recompute`` is then
    # never called, so the buffer-position key never gets set, and the
    # paired ``get_old_*`` lookup KeyError's during recompute backward.
    # Wrapping ``te_checkpoint`` with an outer ``fp8_autocast(DelayedScaling)``
    # makes ``is_fp8_enabled()`` True at the right moment so the pairing works.
    if _truthy_env("MXFP4_HEALING_TE_CHECKPOINT_FP8_WRAP", default=True):
        _install_te_checkpoint_outer_fp8_autocast(model_config)
    elif _truthy_env("MXFP4_HEALING_DEBUG", default=False):
        _emit_rank0(
            "[mxfp4_healing] te_checkpoint outer-fp8_autocast wrapper DISABLED "
            "(MXFP4_HEALING_TE_CHECKPOINT_FP8_WRAP=0). Expect "
            "KeyError 'global_fp8_buffer_pos_fwd_recompute' on first FP8 backward "
            "if activation recompute is enabled."
        )

    if _truthy_env("RESET_CG_AFTER_HEALING", default=False):
        # Already flushed above. Keep this knob for ad-hoc post-flush logging
        # / extra empty_cache pass without changing the default behaviour.
        torch.cuda.empty_cache()
        _emit_rank0("[mxfp4_healing] RESET_CG_AFTER_HEALING=1: extra empty_cache() pass.")

    _HEALING_APPLIED = True
    torch.cuda.synchronize()
    t_total = time.perf_counter() - t0

    _emit_rank0(
        "[mxfp4_healing] " + "=" * 70 + "\n"
        f"[mxfp4_healing] >>> HEALING COMPLETED <<< train_state.step={train_state_step} | "
        f"restored {n_restored} FP8 weights in {t_restore_end - t_restore_start:.2f}s | "
        f"total healing time {t_total:.2f}s | active phase now: FP8 ({healing_precision()})\n"
        "[mxfp4_healing] " + "=" * 70
    )


def _all_modules_depth_first(model: Any):
    chunks = list(model) if isinstance(model, (list, tuple)) else [model]
    for ch in chunks:
        yield from ch.modules()
