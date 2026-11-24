###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText config schema patch: the config keys Primus adds to MaxText.

MaxText's ``pyconfig`` rejects keys that are not fields of ``types.MaxTextConfig``,
so every Primus-only MaxText config key is added here. From ROCm/maxtext release/v26.10:

- ``expert_balance`` (38d321cff)
- ``jax_distributed_heartbeat_timeout_seconds`` (6d3eab29e)
- ``use_turbo_grouped_gemm`` / ``turbo_grouped_gemm_backend`` / ``turbo_grouped_gemm_quant``
  plus their validation (a1efcbc98 .. 2fb48f50e)

The fields live on a ``MaxTextConfig`` subclass that becomes the default
``config_class`` of ``pyconfig``. Fields that the installed MaxText already
defines are left alone, so the patch is a no-op on a MaxText that carries them.
"""

from typing import Literal

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH_ID = "maxtext.config.primus_fields"
_PREFIX = f"[Patch:{_PATCH_ID}]"

_TURBO_BACKEND_MODULES = {
    "ck": "primus_turbo.jax.lax.grouped_gemm",
    "hipblaslt": "primus_turbo.jax.lax.grouped_gemm_hipblaslt",
    "flydsl": "primus_turbo.jax.lax.grouped_gemm_flydsl",
}


def _new_fields(Field):
    return {
        "expert_balance": (bool, Field(False, description="Whether to use expert balancing.")),
        "use_turbo_grouped_gemm": (
            bool,
            Field(
                False,
                description="Use Primus-Turbo grouped GEMM for MoE sparse matmul. "
                "Requires sparse_matmul=True and megablox=False. "
                "Requires the primus_turbo package to be installed.",
            ),
        ),
        "turbo_grouped_gemm_quant": (
            Literal["bf16", "fp8"],
            Field(
                "bf16",
                description="Quantise the Primus-Turbo grouped GEMM operands to per-tensor float8 (e4m3): "
                "fp8 forward, dA and dW. Backends flydsl and hipblaslt on gfx950.",
            ),
        ),
        "turbo_grouped_gemm_backend": (
            Literal["ck", "hipblaslt", "flydsl"],
            Field(
                "ck",
                description="Primus-Turbo grouped GEMM backend: 'ck' (CK grouped GEMM kernels) or 'hipblaslt' "
                "(one hipBLASLt grouped GEMM launch for all experts, group sizes stay on the device; needs a "
                "primus_turbo build that provides primus_turbo.jax.lax.grouped_gemm_hipblaslt).",
            ),
        ),
        "jax_distributed_heartbeat_timeout_seconds": (
            int,
            Field(
                100,
                description="How long before a missing heartbeat marks a task as dead. "
                "Increase for slow NFS checkpoint restores.",
            ),
        ),
    }


def _validate_turbo_grouped_gemm(config, DType):
    if not config.use_turbo_grouped_gemm:
        return config
    if config.quantization:
        raise ValueError("use_turbo_grouped_gemm is not compatible with quantization.")
    if not config.sparse_matmul:
        raise ValueError("use_turbo_grouped_gemm requires sparse_matmul=True.")
    if config.megablox:
        raise ValueError("use_turbo_grouped_gemm requires megablox=False.")
    if config.use_tokamax_gmm:
        raise ValueError("use_turbo_grouped_gemm requires use_tokamax_gmm=False.")
    if config.dtype not in (DType.BFLOAT16, DType.FLOAT16):
        raise ValueError(f"use_turbo_grouped_gemm requires dtype bfloat16 or float16, got {config.dtype}.")
    try:
        __import__(_TURBO_BACKEND_MODULES[config.turbo_grouped_gemm_backend])
    except ImportError as e:
        raise ValueError(
            f"use_turbo_grouped_gemm with turbo_grouped_gemm_backend={config.turbo_grouped_gemm_backend} "
            "requires a primus_turbo package that provides it."
        ) from e
    return config


def _rebind_default_config_class(pyconfig, old_cls, new_cls) -> None:
    for name in ("initialize", "initialize_pydantic", "_initialize_pydantic", "_prepare_for_pydantic"):
        fn = getattr(pyconfig, name, None)
        if fn is None or not fn.__defaults__:
            continue
        fn.__defaults__ = tuple(new_cls if d is old_cls else d for d in fn.__defaults__)


@register_patch(
    patch_id=_PATCH_ID,
    backend="maxtext",
    phase="setup",
    description="Add the Primus MaxText config keys (expert_balance, heartbeat timeout, turbo grouped GEMM) to MaxTextConfig",
    condition=lambda ctx: True,
    priority=10,
)
def patch_maxtext_config_fields(ctx: PatchContext) -> None:
    del ctx
    try:
        from maxtext.configs import pyconfig, types
    except ImportError as e:
        warning_rank_0(f"{_PREFIX} maxtext.configs not importable (legacy MaxText layout?); skipped: {e}")
        return

    from pydantic import create_model
    from pydantic.fields import Field
    from pydantic.functional_validators import model_validator

    base_cls = types.MaxTextConfig
    if getattr(base_cls, "__primus_fields__", False):
        return
    missing = {k: v for k, v in _new_fields(Field).items() if k not in base_cls.model_fields}
    if not missing:
        log_rank_0(f"{_PREFIX} MaxTextConfig already defines every Primus key; not patched.")
        return

    validators = {}
    if "use_turbo_grouped_gemm" in missing:
        validators["_primus_validate_turbo_grouped_gemm"] = model_validator(mode="after")(
            lambda self: _validate_turbo_grouped_gemm(self, types.DType)
        )

    new_cls = create_model(
        "MaxTextConfig",
        __base__=base_cls,
        __module__=base_cls.__module__,
        __validators__=validators,
        **missing,
    )
    new_cls.__qualname__ = base_cls.__qualname__
    new_cls.__primus_fields__ = True
    _rebind_default_config_class(pyconfig, base_cls, new_cls)
    types.MaxTextConfig = new_cls
    log_rank_0(f"{_PREFIX} Added MaxTextConfig fields: {sorted(missing)}")
