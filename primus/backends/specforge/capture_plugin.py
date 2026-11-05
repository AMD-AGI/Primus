# Copyright (c) 2026, Advanced Micro Devices, Inc.
# See LICENSE for license information.

"""GPU-free checks and process environment for the SpecForge capture plugin."""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from typing import Mapping

PLUGIN_NAME = "specforge_spec_capture"
CAPTURE_BACKEND_KEY = "deployment.disaggregated.server_capture"


def capture_server_env(env: Mapping[str, str]) -> dict[str, str]:
    """Enable capture only in the server, preserving any plugin allowlist."""
    result = dict(env)
    result["SPECFORGE_SPEC_CAPTURE"] = "1"
    if result.get("SGLANG_PLUGINS"):
        plugins = [name.strip() for name in result["SGLANG_PLUGINS"].split(",") if name.strip()]
        if PLUGIN_NAME not in plugins:
            plugins.append(PLUGIN_NAME)
        result["SGLANG_PLUGINS"] = ",".join(plugins)
    return result


def validate_capture_backend(value: str | None) -> None:
    """Auto is compatible: Primus explicitly selects the plugin at launch."""
    if value not in (None, "", "auto", "plugin"):
        raise ValueError(
            f"[Primus:specforge] {CAPTURE_BACKEND_KEY}={value} is incompatible; "
            "Primus online capture requires the SGLang plugin (use plugin or auto)"
        )


def capture_plugin_issues() -> list[str]:
    """Probe extension APIs without loading a model or registering hooks."""
    issues: list[str] = []
    try:
        plugins = [ep for ep in entry_points(group="sglang.srt.plugins") if ep.name == PLUGIN_NAME]
        if not plugins:
            issues.append(
                "SpecForge capture plugin entry point is missing; install "
                "plugins/sglang-spec-capture with --no-deps in the SGLang environment"
            )
        elif not callable(plugins[0].load()):
            issues.append("SpecForge capture plugin entry point is not callable")
    except Exception as exc:
        issues.append(f"SpecForge capture plugin cannot be imported: {exc}")

    requirements = (
        ("sglang.srt.model_executor.forward_observer", None),
        ("sglang.srt.managers.deferred_output", "ResponseCompletion"),
        ("sglang.srt.plugins.hook_registry", "HookRegistry"),
        ("sglang.srt.plugins.hook_registry", "HookType"),
    )
    for module, attribute in requirements:
        try:
            loaded = importlib.import_module(module)
            if attribute and not hasattr(loaded, attribute):
                raise AttributeError(f"missing {attribute}")
        except Exception as exc:
            issues.append(
                f"SGLang capture plugin requires {module}{':' + attribute if attribute else ''}: {exc}"
            )

    try:
        server_args = importlib.import_module("sglang.srt.server_args").ServerArgs
        for name in (
            "speculative_algorithm",
            "enable_return_aux_hidden_states",
            "return_aux_hidden_state_layer_ids",
            "return_hidden_states_mode",
            "chunked_prefill_size",
        ):
            if not hasattr(server_args, name) and name not in getattr(
                server_args, "__dataclass_fields__", {}
            ):
                issues.append(f"SGLang capture plugin requires ServerArgs.{name}; use a plugin-capable image")
    except Exception as exc:
        issues.append(f"SGLang ServerArgs cannot be imported: {exc}")
    return issues
