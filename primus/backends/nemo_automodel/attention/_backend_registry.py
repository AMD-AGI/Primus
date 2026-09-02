###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared machinery for rebinding diffusers' attention backends.

THE SEAM:
  diffusers routes every model's attention through
  ``_AttentionBackendRegistry``: the config's ``model.attention_backend`` name
  selects a registered backend function, and ``dispatch_attention_fn`` looks it
  up in ``_backends[name]`` at call time. Rebinding the backend *function* lets a
  different kernel run while the backend NAME, its ``_supported_arg_names`` and
  its constraint list stay untouched -- so the config still selects the same
  backend by name and only the numerics change. No diffusers or AutoModel fork.

  The registered ``_supported_arg_names`` are deliberately left alone: dispatch
  filters kwargs down to the ORIGINAL supported set before calling, and the
  wrapper built here accepts a superset of both backend signatures, so it can be
  driven unchanged. The qkv/device/shape constraints still hold.

WHY THIS IS SHARED:
  Every override needs the same registry walk, the same superset signature, the
  same fallback conditions and the same idempotence marker; only the kernel
  differs. Holding one copy means an override is a kernel plus a name, and means
  a fix to the fallback logic reaches every override at once.

MUTUAL EXCLUSION:
  Overrides all claim the same registry entries, so only one can be active. A
  single shared marker records which override owns an entry, and a second,
  different override is refused rather than layered on top. That refusal matters:
  wrapping a wrapper would run one kernel inside the other's fallback path and
  change numerics with nothing in the logs to say so.

TIMING:
  Install before the transformer's first forward -- in practice before the recipe
  build and ``set_attention_backend``. The swap is a module-global dict entry
  resolved at forward, so being in place before the first forward is sufficient.

Kept free of torch and diffusers at import time so patch conditions can be
evaluated, and these tests can run, without them.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Records which override owns a rebound registry entry. Shared across overrides
# on purpose: checking only for one's OWN marker is how an override ends up
# wrapping another override's wrapper.
MARKER = "_primus_attn_override"

# The backends the diffusion recipes select. FLASH accepts ``window_size``; AITER
# does not, which matters only for the fallback call into the original function.
TARGET_BACKENDS: Tuple[Tuple[str, bool], ...] = (("FLASH", True), ("AITER", False))

_warned: Set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg, *args)


def unsupported_reason(
    attn_mask,
    dropout_p: float,
    window_size,
    return_lse: bool,
    parallel_config,
) -> Optional[str]:
    """Why the plain override path cannot serve this call, or None if it can.

    Correctness first: the overrides serve the plain, no-mask, dropout-free,
    non-context-parallel, single-output call, which is what the diffusion forwards
    use. Anything else falls back to the original kernel, so enabling an override
    never changes results for a call it does not handle.

    Context parallelism is in this list because none of the override kernels have
    a CP-aware forward yet, so a run raised to ``cp_size > 1`` for memory reasons
    quietly gets the original kernel instead.
    """
    if parallel_config is not None:
        return "context parallelism is active"
    if return_lse:
        return "return_lse was requested"
    if attn_mask is not None:
        return "an additive attn_mask was supplied"
    if dropout_p:
        return f"dropout_p={dropout_p} is nonzero"
    if window_size not in ((-1, -1), None):
        return f"a sliding window_size={window_size} was supplied"
    return None


def make_override(
    orig_fn: Callable,
    backend_value: str,
    supports_window: bool,
    *,
    kernel: Callable,
    override_name: str,
    log_prefix: str,
) -> Callable:
    """Wrap ``orig_fn`` so the plain path runs ``kernel`` and the rest falls back.

    Args:
        kernel: ``(q, k, v, softmax_scale=, causal=) -> out``, in the (B, S, H, D)
            layout diffusers hands over.
        supports_window: whether ``orig_fn`` accepts ``window_size``. Passing it
            to a backend that does not take it is a TypeError on the fallback
            path, which would only show up once a fallback actually triggers.
    """

    def _fallback(
        query, key, value, attn_mask, dropout_p, is_causal, scale, window_size, return_lse, parallel_config
    ):
        kwargs = dict(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            return_lse=return_lse,
            _parallel_config=parallel_config,
        )
        if supports_window:
            kwargs["window_size"] = window_size
        return orig_fn(**kwargs)

    # The signature is a superset of both the FLASH and AITER backend signatures,
    # so dispatch_attention_fn can drive it unchanged.
    def override(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale=None,
        window_size=(-1, -1),
        return_lse: bool = False,
        _parallel_config=None,
    ):
        reason = unsupported_reason(attn_mask, dropout_p, window_size, return_lse, _parallel_config)
        if reason is not None:
            _warn_once(
                f"{override_name}:{backend_value}:{reason}",
                "%s falling back to the original kernel on backend '%s' because %s.",
                log_prefix,
                backend_value,
                reason,
            )
            return _fallback(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                window_size,
                return_lse,
                _parallel_config,
            )
        return kernel(query, key, value, softmax_scale=scale, causal=is_causal)

    setattr(override, MARKER, override_name)
    override._primus_orig_fn = orig_fn
    return override


def install_override(
    *,
    kernel: Callable,
    override_name: str,
    log_prefix: str,
    description: str,
    probe: Callable[[], None],
) -> bool:
    """Rebind the target backends to ``kernel``. Returns whether anything took.

    Args:
        probe: called first and allowed to raise, so a missing kernel library
            fails the run clearly instead of silently leaving the original
            backend in place -- which would look like the override worked.
    """
    probe()

    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _AttentionBackendRegistry,
    )

    reg = _AttentionBackendRegistry
    installed: List[str] = []
    conflicts: List[Tuple[str, str]] = []

    for name_str, supports_window in TARGET_BACKENDS:
        name = getattr(AttentionBackendName, name_str, None)
        if name is None:
            continue
        orig = reg._backends.get(name)
        if orig is None:
            # Not registered in this diffusers build.
            continue
        owner = getattr(orig, MARKER, None)
        if owner == override_name:
            installed.append(name.value)  # already ours; installing twice is fine
            continue
        if owner is not None:
            conflicts.append((name.value, owner))
            continue
        reg._backends[name] = make_override(
            orig,
            name.value,
            supports_window,
            kernel=kernel,
            override_name=override_name,
            log_prefix=log_prefix,
        )
        installed.append(name.value)

    for backend_value, owner in conflicts:
        logger.warning(
            "%s NOT installed on backend '%s': the %s override already owns it. "
            "Only one attention override can be active; unset one of the two env gates.",
            log_prefix,
            backend_value,
            owner,
        )

    if not installed:
        if not conflicts:
            logger.warning(
                "%s requested but neither FLASH nor AITER is registered in diffusers; "
                "the override is NOT active.",
                log_prefix,
            )
        return False

    logger.info(
        "%s installed %s for backends: %s (the configured attention_backend name is unchanged)",
        log_prefix,
        description,
        ", ".join(installed),
    )
    return True


def uninstall_override(override_name: str) -> int:
    """Restore the original backend functions owned by ``override_name``.

    Exists so tests can rebind and unwind without leaking a patched registry into
    whatever runs next.
    """
    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _AttentionBackendRegistry,
    )

    reg = _AttentionBackendRegistry
    restored = 0
    for name_str, _ in TARGET_BACKENDS:
        name = getattr(AttentionBackendName, name_str, None)
        if name is None:
            continue
        current = reg._backends.get(name)
        if current is not None and getattr(current, MARKER, None) == override_name:
            reg._backends[name] = current._primus_orig_fn
            restored += 1
    return restored
