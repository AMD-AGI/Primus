###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``situ`` activation kernels — single entry point for every backend.

``eager``
    :func:`...situ_activation.situ_pre_mul_fused`, the numerical reference.
``flydsl``
    One fused FlyDSL kernel per direction (:mod:`._flydsl_gfx1250`), gfx1250 only,
    loaded lazily. On the routed experts the router probability is folded into the
    same kernel (``SituActivation.forward_with_probs``).

Selected by the ``situ_backend`` field of the Kimi K3 config, read when the activation
module is built; the layout mirrors :mod:`...attn_res_kernels`.
"""

SITU_BACKENDS = ("eager", "flydsl")

#: The architectures the FlyDSL kernels build for (they assume 32-lane waves).
FLYDSL_SITU_ARCHS = ("gfx1250",)


def _require_flydsl_arch() -> None:
    """Raise :class:`ImportError` unless a gfx1250 device is visible (as ``attn_res_kernels`` does)."""
    import torch

    if not torch.cuda.is_available():
        raise ImportError("no ROCm/CUDA device is visible")
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    if not arch.startswith(FLYDSL_SITU_ARCHS):
        raise ImportError(f"device arch is {arch!r}, not {' or '.join(FLYDSL_SITU_ARCHS)}")


def resolve_situ_backend(name: str):
    """The fused-input entry ``fn(x, *, beta, linear_beta)`` for ``name``.

    Raises:
        ValueError: on an unknown name.
        ImportError: when ``flydsl`` or a gfx1250 device is unavailable, naming the fallback.
    """
    if name == "eager":
        from primus.backends.megatron.core.transformer.kimi_k3.situ_activation import (
            situ_pre_mul_fused,
        )

        return situ_pre_mul_fused
    if name == "flydsl":
        try:
            import flydsl.compiler  # noqa: F401  (the kernels import it only when they first build)

            _require_flydsl_arch()
            from primus.backends.megatron.core.transformer.kimi_k3.situ_kernels._flydsl_gfx1250 import (
                flydsl_situ_pre_mul_fused,
            )
        except ImportError as exc:
            raise ImportError(
                "situ_backend = 'flydsl' requires the fused FlyDSL situ kernels (the `flydsl` pip "
                f"package on gfx1250), which are unavailable: {exc}. Select situ_backend: eager."
            ) from exc
        return flydsl_situ_pre_mul_fused
    raise ValueError(f"Unknown situ backend {name!r}; expected one of {list(SITU_BACKENDS)}.")


__all__ = ["SITU_BACKENDS", "FLYDSL_SITU_ARCHS", "resolve_situ_backend"]
