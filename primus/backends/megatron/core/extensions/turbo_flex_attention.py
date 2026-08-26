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
import warnings
from collections import OrderedDict
from typing import Any, Callable, Optional, Tuple

import torch

# The compat layer ships with Primus-Turbo. Import defensively so that a container
# with an older Turbo build still imports this module (and fails with a clear message
# only if the switch is actually turned on).
_FLEX_IMPORT_ERROR: Optional[BaseException] = None
try:
    from primus_turbo.pytorch.ops.attention import (
        create_block_mask as turbo_create_block_mask,
    )
    from primus_turbo.pytorch.ops.attention import (
        flex_attention as turbo_flex_attention,
    )
except (ImportError, ModuleNotFoundError, AttributeError) as exc:  # pragma: no cover
    turbo_create_block_mask = None
    turbo_flex_attention = None
    _FLEX_IMPORT_ERROR = exc

# ``flex_attention_bshd`` is the layout-native entry (bshd in, bshd out). Megatron has
# already permuted q/k/v to bshd by the time we are called, so this entry needs no
# transposes at all. Older Turbo builds do not have it; we transparently fall back to
# the torch-layout ``flex_attention`` entry there, handing it a ``transpose(1, 2)``
# view. What that fallback costs depends on the Turbo build:
#   * Turbo with the bhsd passthrough: nothing. The view is non-contiguous, so the
#     compat layer restores the caller's own buffer and forwards it unchanged.
#   * Older Turbo: 3 input + 1 output copies per forward, plus the backward mirrors.
# Either way the result is correct, so no version probing is needed here.
try:
    from primus_turbo.pytorch.ops.attention import (
        flex_attention_bshd as turbo_flex_attention_bshd,
    )
except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover
    turbo_flex_attention_bshd = None

# ``flex_attention_varlen`` is the packed-sequence (THD) entry: q/k/v are
# ``[total_tokens, H, D]`` with the document boundaries carried out-of-band in
# ``cu_seqlens``. This is what makes ``qkv_format="thd"`` reachable -- the dense
# entries cannot express per-document masking, and the compat layer's probe-based
# classifier is not asked to: the boundaries are given, not inferred.
try:
    from primus_turbo.pytorch.ops.attention import (
        flex_attention_varlen as turbo_flex_attention_varlen,
    )
except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover
    turbo_flex_attention_varlen = None


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
_BLOCK_MASK_CACHE: "OrderedDict[Tuple, Any]" = OrderedDict()

# Pretraining sees one or two shapes for the whole run, so the cache holds a couple of
# entries forever. Variable-length workloads (SFT with per-batch bucketing, evaluation
# sweeps) hand us a new ``(q_len, kv_len)`` almost every step, and an unbounded dict
# would then grow one BlockMask per distinct length for the lifetime of the process.
# Bounded + LRU: the hot shapes stay resident, the long tail is evicted.
_BLOCK_MASK_CACHE_MAX = 64
_EVICTION_WARNED = False


def _get_block_mask(
    *,
    mask_mod: Callable,
    cache_key: Tuple,
    q_len: int,
    kv_len: int,
    device: torch.device,
):
    global _EVICTION_WARNED

    key = (cache_key, q_len, kv_len, str(device))
    cached = _BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        _BLOCK_MASK_CACHE.move_to_end(key)
        return cached

    cached = turbo_create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
    )
    _BLOCK_MASK_CACHE[key] = cached
    if len(_BLOCK_MASK_CACHE) > _BLOCK_MASK_CACHE_MAX:
        _BLOCK_MASK_CACHE.popitem(last=False)
        if not _EVICTION_WARNED:
            _EVICTION_WARNED = True
            warnings.warn(
                "Primus-Turbo flex attention: the BlockMask cache passed "
                f"{_BLOCK_MASK_CACHE_MAX} distinct (pattern, q_len, kv_len, device) keys and "
                "started evicting. Every miss rebuilds a BlockMask and re-runs the compat "
                "layer's pattern classification, which is orders of magnitude more expensive "
                "than the attention call itself. Bucket sequence lengths, or raise "
                "turbo_flex_attention._BLOCK_MASK_CACHE_MAX if the working set is genuinely "
                "this wide.",
                RuntimeWarning,
                stacklevel=2,
            )
    return cached


def clear_turbo_flex_block_mask_cache() -> None:
    """Drop every cached ``BlockMask`` (tests / benchmarks wanting a cold measurement)."""
    global _EVICTION_WARNED

    _BLOCK_MASK_CACHE.clear()
    _EVICTION_WARNED = False


# =============================================================================
# Shared build-time rejections
# =============================================================================


def reject_reset_attention_mask(args, *, is_flex: bool, where: str) -> None:
    """Reject ``reset_attention_mask`` on any path that cannot honour it.

    ``reset_attention_mask`` asks Megatron for per-document causal masking *inside a
    dense sample*: the document boundaries are baked into the ``attention_mask`` tensor
    as cross-document zeros. No Primus-Turbo attention entry point takes a mask tensor
    -- they take a ``causal`` flag -- so the boundaries have nowhere to go and would be
    dropped, letting tokens attend across documents with nothing raised anywhere.

    The default mask is *not* the problem. ``create_attention_mask_in_dataloader``
    defaults to true, so an ordinary causal run also hands the module a tensor; that one
    is exactly ``torch.tril(...)``, which the ``causal`` flag already expresses, so
    dropping it changes nothing. Only ``reset_attention_mask`` adds information the flag
    cannot carry (see ``megatron/core/datasets/gpt_dataset.py``,
    ``_get_ltor_masks_and_position_ids``).

    ``qkv_format="thd"`` is a different, supported mechanism: there the boundaries
    arrive out-of-band as ``cu_seqlens`` in ``packed_seq_params`` and are forwarded
    explicitly to ``flex_attention_varlen``.

    Args:
        args: the Megatron global args namespace.
        is_flex: whether the caller routes through the flex compat layer, which raises
            for itself in :func:`build_turbo_flex_attention` with a message naming the
            THD alternative. Passing True here makes this a no-op so that the more
            specific message wins.
        where: class name to name in the error message.
    """
    if not getattr(args, "reset_attention_mask", False) or is_flex:
        return
    raise NotImplementedError(
        f"{where} does not support reset_attention_mask=true. The per-document "
        "boundaries are carried in the attention_mask tensor, which flash_attn_func has "
        "no argument for, so they would be silently dropped and tokens would attend "
        "across documents. Use packed sequences (qkv_format='thd'), or set "
        "reset_attention_mask=false."
    )


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
            f"Primus-Turbo flex attention: module '{module_name}' has no attribute '{attr}' " f"for {what}."
        ) from exc
    if not callable(obj):
        raise TypeError(f"Primus-Turbo flex attention: {what} '{path}' is not callable.")
    return obj


# =============================================================================
# The drop-in replacement for flash_attn_func
# =============================================================================


def _coerce_sink(sink: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Present Megatron's sink parameter in the dtype the compat layer requires.

    ``PrimusTurboAttention`` allocates ``self.sinks`` as a **bfloat16** Parameter (it
    mirrors gpt-oss, and ``flash_attn_func`` takes it as-is), while the compat layer
    validates ``sink.dtype == torch.float32`` and rejects anything else. Handing the
    Parameter straight through therefore turns on ``use_sink_attention`` +
    ``use_turbo_flex_attention`` into a hard ValueError on the first forward.

    ``.float()`` is a differentiable cast: autograd casts the fp32 grad back to bf16 on
    the way out, so the Parameter still trains. Doing it here rather than loosening the
    compat layer keeps the dtype contract explicit and confines the Megatron-specific
    convention to the Megatron-side adapter.
    """
    if sink is None or sink.dtype == torch.float32:
        return sink
    return sink.float()


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

        An explicit user ``mask_mod`` replaces the built-in choice; combining it with a
        window is rejected rather than silently resolved. Otherwise the Megatron mask
        type (``causal`` / ``no_mask``) and ``window_size`` decide.
        """
        if self.mask_mod is not None:
            if window > 0:
                raise NotImplementedError(
                    "Primus-Turbo flex attention: turbo_flex_attention_mask_mod and a sliding "
                    f"window (window={window}) were both requested. A user mask_mod replaces "
                    "the window rather than composing with it, so honouring either one would "
                    "silently drop the other. Fold the window into the mask_mod itself."
                )
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

    # -- packed sequences (THD) ---------------------------------------------
    def _call_varlen(
        self,
        query,
        key,
        value,
        *,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        return_lse,
        sink,
    ):
        """Dispatch packed THD input through ``flex_attention_varlen``.

        The dense entries take a ``BlockMask`` that the compat layer has to *classify*
        by probing ``mask_mod`` on a grid. Packing is different in kind: Megatron
        already knows exactly where the document boundaries are and hands them over as
        ``cu_seqlens``, so there is nothing to infer. That is why this path bypasses
        the BlockMask cache entirely -- no probing, no classification, no per-shape
        cache entry, which also means the variable-length workloads that would thrash
        that cache do not touch it at all.

        ``causal=True`` here means *document-internal* causal (block-diagonal plus
        within-segment causal), which is exactly Megatron's intent for a packed batch
        and the thing the dense path could not express.
        """
        if turbo_flex_attention_varlen is None:
            raise RuntimeError(
                "Primus-Turbo flex attention: packed (THD) sequences need "
                "primus_turbo.pytorch.ops.attention.flex_attention_varlen, which this "
                "Primus-Turbo build does not provide. Upgrade Primus-Turbo or set "
                "use_turbo_flex_attention=false for packed-sequence runs."
            )
        if query.dim() != 3:
            raise NotImplementedError(
                "Primus-Turbo flex attention: cu_seqlens was supplied, so the packed (THD) "
                f"path was selected, but query is {query.dim()}D; THD expects "
                "[total_tokens, H, D]."
            )
        if cu_seqlens_q is None:
            raise NotImplementedError(
                "Primus-Turbo flex attention: packed (THD) input without cu_seqlens_q. The "
                "document boundaries are not recoverable from the tensor shape, and guessing "
                "them would let tokens attend across documents."
            )
        if self.mask_mod is not None:
            # A user mask_mod is a *dense* [q_idx, kv_idx] predicate; composing it with
            # packing would need the product mask, which the varlen backend cannot take.
            raise NotImplementedError(
                "Primus-Turbo flex attention: turbo_flex_attention_mask_mod cannot be combined "
                "with packed (THD) sequences; the varlen backend takes cu_seqlens, not a "
                "BlockMask. Use dense batches, or fold the pattern into the packing itself."
            )
        if self.score_mod is not None:
            # flex_attention_varlen takes explicit alibi_slopes/softcap rather than probing
            # a score_mod, so silently dropping the callable is the one thing we must not do.
            raise NotImplementedError(
                "Primus-Turbo flex attention: turbo_flex_attention_score_mod is not supported on "
                "the packed (THD) path; flex_attention_varlen takes explicit alibi_slopes "
                "instead of probing a score_mod."
            )
        if bias is not None:
            raise NotImplementedError(
                "Primus-Turbo flex attention: an attention bias is not supported on the packed "
                "(THD) path (flex_attention_varlen has no bias parameter)."
            )
        if cu_seqlens_kv is None:
            cu_seqlens_kv = cu_seqlens_q
        if max_seqlen_q is None or max_seqlen_kv is None:
            # Derivable from the prefix sums; one small D2H copy, and only on the first
            # call of a shape in practice because Megatron caches PackedSeqParams.
            diffs_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
            diffs_kv = (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).max().item()
            max_seqlen_q = int(max_seqlen_q or diffs_q)
            max_seqlen_kv = int(max_seqlen_kv or diffs_kv)

        return turbo_flex_attention_varlen(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_kv,
            int(max_seqlen_q),
            int(max_seqlen_kv),
            causal=causal,
            window_size=window_size if window_size is not None else (-1, -1),
            scale=softmax_scale,
            alibi_slopes=alibi_slopes,
            dropout_p=dropout_p,
            sink=_coerce_sink(sink),
            return_lse=return_lse,
        )

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
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        **extra: Any,
    ):
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

        if query.dim() == 3 or cu_seqlens_q is not None:
            return self._call_varlen(
                query,
                key,
                value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                bias=bias,
                alibi_slopes=alibi_slopes,
                return_lse=return_lse,
                sink=sink,
            )
        if query.dim() != 4:
            raise NotImplementedError(
                "Primus-Turbo flex attention: expected bshd [B,S,H,D] or packed THD "
                f"[total_tokens,H,D] input, got {query.dim()}D."
            )

        left_window = int(window_size[0]) if window_size is not None else -1
        right_window = int(window_size[1]) if window_size is not None else -1
        if right_window > 0:
            # The dense path expresses a window as a mask_mod, and every mask_mod the
            # compat layer can build is causal (q_idx >= kv_idx). A positive right bound
            # means attending forward, which none of them can represent -- and honouring
            # the left bound while quietly discarding the right one is exactly the silent
            # wrongness this layer exists to prevent. (The packed path forwards
            # window_size to the varlen backend verbatim, so it is not restricted here.)
            raise NotImplementedError(
                "Primus-Turbo flex attention: only causal windows are supported on the dense "
                "path, so window_size[1] must be 0 or -1; got "
                f"window_size={tuple(window_size)}."
            )
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
            sink=_coerce_sink(sink),
            bias=bias,
        )
        if self._entry_is_bshd:
            return self._entry(query, key, value, **kwargs)
        # Older Turbo build: go through the torch-layout entry (see the import block
        # above for what the round-trip costs on each Turbo version).
        out = self._entry(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), **kwargs)
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
    if getattr(args, "deterministic_mode", False):
        # PrimusTurboAttention.forward passes deterministic=self.deterministic_mode on
        # every call, so this would otherwise surface as a NotImplementedError from the
        # first forward -- after model init, data loading and (on a large job) minutes of
        # startup. Reject it here instead, next to the other build-time rejections.
        raise NotImplementedError(
            "use_turbo_flex_attention=true is not supported with deterministic_mode=true "
            "(the compat layer always dispatches deterministic=False). Note that the aiter "
            "backward accumulates dQ with fp32 atomics regardless, so bit-reproducible "
            "attention gradients are not available on this backend either way. Set "
            "deterministic_mode=false or use_turbo_flex_attention=false."
        )
    if getattr(args, "reset_attention_mask", False):
        # reset_attention_mask asks Megatron for per-document causal masking inside a
        # *dense* sample: the boundaries are baked into an attention_mask tensor that the
        # Turbo call signature has no argument for, so they would be dropped and tokens
        # would attend across documents with nothing raised anywhere.
        #
        # Note this is a different mechanism from qkv_format="thd", which IS supported:
        # there Megatron hands over cu_seqlens explicitly and the adapter forwards them
        # to flex_attention_varlen. Packing via THD is the supported way to get
        # per-document masking on this path.
        raise NotImplementedError(
            "use_turbo_flex_attention=true is not supported with reset_attention_mask=true. "
            "The dense flex dispatch receives only the causal flag, so the per-document "
            "boundaries encoded in the attention mask would be dropped and tokens would "
            "attend across documents. Use packed sequences (qkv_format='thd', which routes "
            "through flex_attention_varlen and IS supported) instead, or set "
            "reset_attention_mask=false / use_turbo_flex_attention=false."
        )

    mask_mod_path = getattr(args, "turbo_flex_attention_mask_mod", None)
    if mask_mod_path and getattr(args, "sink_sliding_window", 0) > 0:
        # Both describe the mask, and a user mask_mod replaces the window instead of
        # composing with it (see _block_mask_for). That conflict is reachable from a
        # plain config, so catch it here rather than at the first forward.
        raise NotImplementedError(
            f"turbo_flex_attention_mask_mod={mask_mod_path!r} and sink_sliding_window="
            f"{args.sink_sliding_window} both describe the attention mask, and the mask_mod "
            "replaces the window rather than composing with it. Fold the window into the "
            "mask_mod, or set sink_sliding_window=0."
        )
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
