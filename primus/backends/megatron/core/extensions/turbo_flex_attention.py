###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Route Megatron core attention through the Primus-Turbo ``flex_attention`` compat layer.

Enabled by ``use_turbo_flex_attention: true`` in the ``primus_turbo`` module config
(default ``false``). It only takes effect on top of ``use_turbo_attention: true`` --
it swaps the callable that :class:`PrimusTurboAttention` invokes, so the surrounding
Megatron plumbing (layout permutes, sink parameters, offload, spec selection) is
untouched.

Why route through the compat layer at all
-----------------------------------------
``flash_attn_func`` takes a fixed set of knobs: ``causal``, ``window_size``,
``alibi_slopes``, ``bias``, ``dropout_p``, ``sink``. The compat layer takes a
``block_mask`` / ``score_mod`` -- torch-FlexAttention's *programmable* interface --
classifies it, and dispatches the recognised patterns onto exactly those same Turbo
kernels. So with the default causal / no-mask configuration this switch is
**numerically identical** to the direct call (same kernel, same arguments); what it
buys is the ability to express masks and score modifications that have no
``flash_attn_func`` argument, without leaving Turbo's fast path.

``turbo_flex_attention_mask_mod`` / ``turbo_flex_attention_score_mod`` (both ``null``
by default) accept a ``"package.module:attribute"`` path to a user callable, which is
how a model config reaches that programmability.

Deliberately strict
-------------------
Every unsupported combination raises with an explicit message instead of quietly
falling back to the direct call. A silent fallback would make the switch look like it
worked while the run behaved exactly as before -- the failure mode that is hardest to
notice in a training log.
"""

import importlib
from typing import Any, Callable, Dict, Optional, Tuple

import torch

# The compat layer ships with Primus-Turbo. Import defensively so that a container
# with an older Turbo build still imports this module (and fails with a clear message
# only if the switch is actually turned on).
_FLEX_IMPORT_ERROR: Optional[BaseException] = None
try:
    from primus_turbo.pytorch.ops.attention import create_block_mask as turbo_create_block_mask
    from primus_turbo.pytorch.ops.attention import flex_attention as turbo_flex_attention
except (ImportError, ModuleNotFoundError, AttributeError) as exc:  # pragma: no cover
    turbo_create_block_mask = None
    turbo_flex_attention = None
    _FLEX_IMPORT_ERROR = exc

# ``flex_attention_bshd`` is the layout-native entry (bshd in, bshd out). Megatron has
# already permuted q/k/v to bshd by the time we are called, so this entry avoids the
# 4 transpose+contiguous copies per forward (and their backward mirrors) that the
# torch-layout ``flex_attention`` entry needs. Older Turbo builds do not have it; we
# transparently fall back to the bhsd entry there (correct, just with the copies).
try:
    from primus_turbo.pytorch.ops.attention import (
        flex_attention_bshd as turbo_flex_attention_bshd,
    )
except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover
    turbo_flex_attention_bshd = None


# =============================================================================
# mask_mod builders + BlockMask cache
# =============================================================================


def _causal_mask_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _make_window_causal_mask_mod(window: int) -> Callable:
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= window)

    return mask_mod


# BlockMask construction walks the whole mask once, and the compat layer memoises its
# classification by *object identity* -- so building one mask per (pattern, shape) and
# reusing it across layers and steps is what keeps the per-call cost at zero. Keyed by
# device string rather than the device object so replicas share entries.
_BLOCK_MASK_CACHE: Dict[Tuple, Any] = {}


def _get_block_mask(
    *,
    mask_mod: Callable,
    cache_key: Tuple,
    q_len: int,
    kv_len: int,
    device: torch.device,
):
    key = (cache_key, q_len, kv_len, str(device))
    cached = _BLOCK_MASK_CACHE.get(key)
    if cached is None:
        cached = turbo_create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )
        _BLOCK_MASK_CACHE[key] = cached
    return cached


def clear_turbo_flex_block_mask_cache() -> None:
    """Drop every cached ``BlockMask`` (tests / benchmarks wanting a cold measurement)."""
    _BLOCK_MASK_CACHE.clear()


# =============================================================================
# "package.module:attribute" resolution for the optional user hooks
# =============================================================================


def resolve_dotted_callable(path: str, *, what: str) -> Callable:
    """Import ``"package.module:attribute"`` and return the callable it names."""
    if not isinstance(path, str) or ":" not in path:
        raise ValueError(
            f"Primus-Turbo flex attention: {what} must be a 'package.module:attribute' string, "
            f"got {path!r}."
        )
    module_name, _, attr = path.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Primus-Turbo flex attention: could not import module '{module_name}' for {what}."
        ) from exc
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(
            f"Primus-Turbo flex attention: module '{module_name}' has no attribute '{attr}' "
            f"for {what}."
        ) from exc
    if not callable(obj):
        raise TypeError(f"Primus-Turbo flex attention: {what} '{path}' is not callable.")
    return obj


# =============================================================================
# The drop-in replacement for flash_attn_func
# =============================================================================


class TurboFlexAttention:
    """Callable with ``flash_attn_func``'s signature that dispatches via the compat layer.

    :class:`PrimusTurboAttention` keeps a bound ``self.attn`` and calls it with bshd
    ``[B, S, H, D]`` tensors plus keyword arguments. Presenting the same interface here
    means the switch is a one-line substitution in ``__init__`` and ``forward`` needs no
    changes at all.
    """

    def __init__(
        self,
        *,
        mask_mod: Optional[Callable] = None,
        score_mod: Optional[Callable] = None,
        mask_mod_key: str = "user",
    ):
        self.mask_mod = mask_mod
        self.score_mod = score_mod
        self.mask_mod_key = mask_mod_key
        self._entry = turbo_flex_attention_bshd or turbo_flex_attention
        self._entry_is_bshd = turbo_flex_attention_bshd is not None

    # -- mask selection -----------------------------------------------------
    def _block_mask_for(self, *, causal: bool, window: int, q_len: int, kv_len: int, device):
        """Pick the BlockMask for this call, or ``None`` for unmasked (full) attention.

        Precedence: an explicit user ``mask_mod`` wins; otherwise the Megatron mask type
        (``causal`` / ``no_mask``) and ``window_size`` decide.
        """
        if self.mask_mod is not None:
            return _get_block_mask(
                mask_mod=self.mask_mod,
                cache_key=("user", self.mask_mod_key),
                q_len=q_len,
                kv_len=kv_len,
                device=device,
            )
        if causal and window > 0:
            return _get_block_mask(
                mask_mod=_make_window_causal_mask_mod(window),
                cache_key=("window_causal", window),
                q_len=q_len,
                kv_len=kv_len,
                device=device,
            )
        if causal:
            return _get_block_mask(
                mask_mod=_causal_mask_mod,
                cache_key=("causal",),
                q_len=q_len,
                kv_len=kv_len,
                device=device,
            )
        if window > 0:
            raise NotImplementedError(
                "Primus-Turbo flex attention: a sliding window without a causal mask is not a "
                "pattern the compat layer can express; set attn_mask_type=causal or disable the "
                "window."
            )
        return None  # no_mask -> full attention

    # -- the call itself ----------------------------------------------------
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        return_lse: bool = False,
        return_attn_probs: bool = False,
        sink: Optional[torch.Tensor] = None,
        **extra: Any,
    ):
        if query.dim() != 4:
            raise NotImplementedError(
                "Primus-Turbo flex attention: only the bshd [B,S,H,D] path is wired up; got "
                f"{query.dim()}D input (THD packed sequences reach the compat layer through "
                "flex_attention_varlen, which is not routed from Megatron yet). Set "
                "use_turbo_flex_attention=false for packed-sequence runs."
            )
        if return_attn_probs:
            raise NotImplementedError(
                "Primus-Turbo flex attention: return_attn_probs is not supported (the compat "
                "layer never materialises the attention probabilities)."
            )
        if deterministic:
            raise NotImplementedError(
                "Primus-Turbo flex attention: deterministic mode is not supported (the compat "
                "layer always dispatches deterministic=False). Set deterministic_mode=false or "
                "use_turbo_flex_attention=false."
            )
        if extra:
            # ulysses_group / ring_group arrive here when context parallelism is on.
            raise NotImplementedError(
                "Primus-Turbo flex attention: unsupported backend arguments "
                f"{sorted(extra)} (context parallelism has no compat-layer path yet)."
            )

        left_window = int(window_size[0]) if window_size is not None else -1
        block_mask = self._block_mask_for(
            causal=causal,
            window=max(left_window, 0),
            q_len=query.shape[1],
            kv_len=key.shape[1],
            device=query.device,
        )

        kwargs = dict(
            score_mod=self.score_mod,
            block_mask=block_mask,
            scale=softmax_scale,
            enable_gqa=query.shape[2] != key.shape[2],
            return_lse=return_lse,
            alibi_slopes=alibi_slopes,
            dropout_p=dropout_p,
            sink=sink,
            bias=bias,
        )
        if self._entry_is_bshd:
            return self._entry(query, key, value, **kwargs)
        # Older Turbo build: the bhsd entry needs the layout round-trip.
        out = self._entry(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), **kwargs
        )
        if return_lse:
            out, lse = out
            return out.transpose(1, 2), lse
        return out.transpose(1, 2)


# =============================================================================
# Construction + config validation
# =============================================================================


def build_turbo_flex_attention(*, args, config) -> TurboFlexAttention:
    """Validate the configuration and build the flex-routed attention callable.

    Raises (rather than degrading to the direct ``flash_attn_func`` call) whenever the
    requested combination has no compat-layer path, so an unsupported run fails at
    model-build time instead of silently training on the old code path.
    """
    if turbo_flex_attention is None:
        raise RuntimeError(
            "use_turbo_flex_attention=true, but this Primus-Turbo build does not provide "
            "primus_turbo.pytorch.ops.attention.flex_attention "
            f"({type(_FLEX_IMPORT_ERROR).__name__}: {_FLEX_IMPORT_ERROR}). Upgrade Primus-Turbo "
            "or set use_turbo_flex_attention=false."
        )
    if getattr(config, "context_parallel_size", 1) > 1:
        raise NotImplementedError(
            "use_turbo_flex_attention=true is not supported with context parallelism "
            f"(context_parallel_size={config.context_parallel_size}); the compat layer has no "
            "ulysses/ring path yet. Set use_turbo_flex_attention=false for CP runs."
        )
    if getattr(args, "enable_turbo_attention_float8", False):
        raise NotImplementedError(
            "use_turbo_flex_attention=true cannot be combined with "
            "enable_turbo_attention_float8=true; the compat layer supports fp16/bf16 only."
        )

    mask_mod_path = getattr(args, "turbo_flex_attention_mask_mod", None)
    score_mod_path = getattr(args, "turbo_flex_attention_score_mod", None)
    mask_mod = (
        resolve_dotted_callable(mask_mod_path, what="turbo_flex_attention_mask_mod")
        if mask_mod_path
        else None
    )
    score_mod = (
        resolve_dotted_callable(score_mod_path, what="turbo_flex_attention_score_mod")
        if score_mod_path
        else None
    )
    return TurboFlexAttention(
        mask_mod=mask_mod,
        score_mod=score_mod,
        mask_mod_key=str(mask_mod_path),
    )
