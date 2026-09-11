###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Stack-aware preflight for the SpecForge ROCm overlay.

The Qwen3.5 DFlash recipe on MI355X is not a generic ``specforge train``. Capture
and serving run a hybrid Mamba target under AITER; SGLang's Mamba radix-cache
path asserts CUDA at init. A CUDA torch wheel in this image is a silent
clobber. These checks are GPU-free (imports and env only).

Called from the pretrain hook (fail before launch) and from the trainer
(fail before exec). The hook also emits the AITER / radix-cache env defaults
so capture and serve inherit them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

from primus.backends.specforge.argument_builder import (
    flatten_overrides,
    resolve_specforge_root,
    specforge_mode,
)

ROCM_STACK_ENV_DEFAULTS = (
    ("SGLANG_USE_AITER", "1"),
    ("SGLANG_USE_AITER_UNIFIED_ATTN", "1"),
    ("AITER_FLYDSL_FORCE", "1"),
    ("SGLANG_DISABLE_RADIX_CACHE", "1"),
)

FALSEY = {"0", "false", "no", "off"}
TRUTHY = {"1", "true", "yes", "on"}

DEFAULT_SGLANG_PIN_PREFIX = "0.5.14"


def _env_flag(env: Mapping[str, str], name: str) -> Optional[bool]:
    raw = env.get(name)
    if raw is None or raw == "":
        return None
    lowered = raw.strip().lower()
    if lowered in FALSEY:
        return False
    if lowered in TRUTHY:
        return True
    return None


def torch_kind(torch_mod: Any = None) -> str:
    """Return ``hip``, ``cuda``, ``unknown``, or ``missing``."""

    if torch_mod is None:
        try:
            import torch as torch_mod
        except Exception:
            return "missing"
    version = str(getattr(torch_mod, "__version__", "")).lower()
    hip = getattr(getattr(torch_mod, "version", None), "hip", None)
    if hip:
        return "hip"
    if "+cu" in version:
        return "cuda"
    if "git" in version or "+rocm" in version:
        return "hip"
    return "unknown"


def enforce_rocm_stack(env: Mapping[str, str], kind: Optional[str] = None) -> bool:
    """Whether AITER / radix / pin checks apply.

    HIP torch always enforces. Otherwise HIP_VISIBLE_DEVICES or an explicit
    ``PRIMUS_SPECFORGE_ENFORCE_ROCM=1`` opts in (the overlay smoke sets the
    former). A laptop CUDA torch without those stays a no-op so unit tests pass.
    """

    if _env_flag(env, "PRIMUS_SPECFORGE_ENFORCE_ROCM") is False:
        return False
    if _env_flag(env, "PRIMUS_SPECFORGE_ENFORCE_ROCM") is True:
        return True
    if kind is None:
        kind = torch_kind()
    if kind == "hip":
        return True
    return bool(env.get("HIP_VISIBLE_DEVICES"))


def apply_rocm_stack_env(env: Optional[MutableMapping[str, str]] = None) -> list[tuple[str, str]]:
    """Fill AITER / radix-cache defaults when unset. Does not override ``0``."""

    environ = os.environ if env is None else env
    applied: list[tuple[str, str]] = []
    for key, value in ROCM_STACK_ENV_DEFAULTS:
        if key in environ and str(environ[key]).strip() != "":
            continue
        environ[key] = value
        applied.append((key, value))
    return applied


def _hidden_states_path(params: Any, env: Mapping[str, str]) -> Optional[str]:
    overrides = flatten_overrides(getattr(params, "specforge_overrides", None))
    hidden = overrides.get("data.hidden_states_path") or overrides.get("hidden_states_path")
    return hidden or env.get("HIDDEN_STATES_PATH") or None


def collect_issues(params: Any, env: Optional[Mapping[str, str]] = None) -> list[str]:
    """Return human-readable problems. Empty means the stack looks runnable."""

    environ = os.environ if env is None else env
    issues: list[str] = []
    mode = specforge_mode(params)
    overrides = flatten_overrides(getattr(params, "specforge_overrides", None))
    capture = flatten_overrides(getattr(params, "specforge_capture", None))

    specforge_config = getattr(params, "specforge_config", None) or environ.get("SPECFORGE_CONFIG")
    if specforge_config:
        config_path = Path(str(specforge_config))
        if not config_path.is_file():
            issues.append(f"specforge_config is not a file: {config_path}")
    elif mode != "capture":
        issues.append("specforge_config is required for train")

    root = resolve_specforge_root(params, env=environ)
    if root is None:
        issues.append(
            "SpecForge root was not resolved; set SPECFORGE_ROOT or specforge_root "
            "(relative draft configs will not load; see job 89021)"
        )
    elif not (root / "configs").is_dir():
        issues.append(f"SpecForge root is missing configs/: {root}")

    if mode == "capture":
        data_path = capture.get("data_path") or environ.get("CAPTURE_DATA_PATH")
        if not data_path:
            issues.append("capture data_path is required (ShareGPT-style JSONL)")
        elif not Path(str(data_path)).is_file():
            issues.append(f"capture data_path is not a file: {data_path}")
        output_path = capture.get("output_path")
        if not output_path:
            issues.append("capture output_path is required")
        draft = capture.get("draft_model_config")
        if draft and root is not None:
            draft_path = Path(str(draft))
            if not draft_path.is_absolute():
                draft_path = root / draft_path
            if not draft_path.is_file():
                issues.append(f"draft model config is not a file: {draft_path}")
        if (
            capture.get("sglang_disable_radix_cache") in FALSEY
            or capture.get("sglang_disable_radix_cache") == "false"
        ):
            issues.append(
                "capture sglang_disable_radix_cache is false; Mamba + AITER on ROCm "
                "must disable the radix cache"
            )
    else:
        hidden = _hidden_states_path(params, environ)
        if hidden:
            path = Path(str(hidden))
            if not path.is_dir():
                issues.append(f"Hidden-states path is not a directory: {path}")
            elif not any(path.iterdir()):
                issues.append(f"Hidden-states path is empty: {path}")

    liger = overrides.get("model.use_liger_kernel")
    if liger in TRUTHY or liger == "true":
        try:
            import liger_kernel  # noqa: F401
        except Exception:
            issues.append(
                "model.use_liger_kernel is enabled but liger_kernel is not installed; "
                "the ROCm overlay does not ship Liger (set model.use_liger_kernel=false)"
            )

    radix_override = overrides.get("model.sglang_disable_radix_cache")
    kind = torch_kind()
    if enforce_rocm_stack(environ, kind=kind):
        if kind == "cuda":
            issues.append("Detected a CUDA torch wheel; the ROCm overlay stack has been clobbered")
        if kind == "unknown":
            issues.append("Unexpected torch build; expected HIP metadata or the SGLang ROCm git build")
        if _env_flag(environ, "SGLANG_USE_AITER") is False:
            issues.append("SGLANG_USE_AITER=0; Qwen3.5 Mamba capture/serve on this overlay needs AITER")
        if _env_flag(environ, "SGLANG_DISABLE_RADIX_CACHE") is False:
            issues.append("SGLANG_DISABLE_RADIX_CACHE=0; SGLang Mamba radix-cache asserts CUDA on ROCm")
        if radix_override in FALSEY or radix_override == "false":
            issues.append(
                "model.sglang_disable_radix_cache=false; Mamba + AITER on ROCm must disable "
                "the radix cache"
            )
        if kind == "hip":
            try:
                from sglang.srt.server_args import ServerArgs

                if not hasattr(ServerArgs, "enable_spec_capture"):
                    issues.append(
                        "sglang ServerArgs is missing enable_spec_capture; this is not the "
                        "SpecForge-patched overlay"
                    )
            except Exception as exc:
                issues.append(f"sglang is not importable on the ROCm overlay: {exc}")
            pin = environ.get("PRIMUS_SPECFORGE_PIN_SGLANG", DEFAULT_SGLANG_PIN_PREFIX)
            if pin:
                try:
                    import sglang

                    version = str(getattr(sglang, "__version__", ""))
                    if version and not version.startswith(pin):
                        issues.append(
                            f"sglang {version} does not match pin prefix {pin} "
                            "(set PRIMUS_SPECFORGE_PIN_SGLANG to override)"
                        )
                except Exception:
                    pass
            try:
                import aiter  # noqa: F401
            except ModuleNotFoundError:
                issues.append("aiter is not installed; AITER is required on this ROCm SpecForge stack")
            except Exception:
                # The wheel is present; JIT/GPU init can still fail in CPU-only pytest.
                pass

    return issues


def raise_if_issues(params: Any, env: Optional[Mapping[str, str]] = None) -> None:
    issues = collect_issues(params, env=env)
    if not issues:
        return
    bullets = "\n".join(f"  - {item}" for item in issues)
    raise RuntimeError(f"[Primus:specforge] stack preflight failed:\n{bullets}")
