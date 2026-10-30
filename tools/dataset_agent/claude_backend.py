# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Claude Agent SDK session runner.

The phase machine calls ``run`` and ``resume``. This module turns an
``AgentRunSpec`` into ``ClaudeAgentOptions`` and streams ``query()``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tools.dataset_agent.paths import bash_deny_reason, edit_deny_reason
from tools.dataset_agent.spec import AgentRunResult, AgentRunSpec

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5"
_PROGRESS_CHARS = 160


class ClaudeBackendError(RuntimeError):
    """The Claude SDK or CLI cannot run this session."""


def _bundled_claude() -> str:
    """Binary shipped inside the claude-agent-sdk wheel. It is not put on PATH."""
    try:
        import claude_agent_sdk
    except ImportError:
        return ""
    bundled = Path(claude_agent_sdk.__file__).resolve().parent / "_bundled" / "claude"
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return str(bundled)
    return ""


def resolve_claude_cli(explicit: str = "") -> str:
    if explicit.strip():
        return explicit.strip()
    found = shutil.which("claude")
    if found:
        return found
    bundled = _bundled_claude()
    if bundled:
        return bundled
    for candidate in (
        Path.home() / ".local/bin/claude",
        Path.home() / ".npm-global/bin/claude",
        Path("/usr/local/bin/claude"),
        Path("/usr/bin/claude"),
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return "claude"


def prepare_environment() -> None:
    """Drop a duplicated ``/v1`` tail. The Claude CLI appends ``/v1/messages`` itself."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        os.environ.setdefault("IS_SANDBOX", "1")
    configured = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    if not configured:
        return
    trimmed = configured.rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    if trimmed != configured:
        print(f"ANTHROPIC_BASE_URL {configured} -> {trimmed} (the CLI appends /v1/messages)")
        os.environ["ANTHROPIC_BASE_URL"] = trimmed


def gateway_auth_failure(status: int, body: str) -> str:
    """Explain a gateway rejection. Empty means the CLI should try the call itself."""
    if status == 401 and "subscription key" in body.lower():
        return (
            "The AMD gateway returned HTTP 401 because the APIM subscription key is missing or invalid. "
            "Export ANTHROPIC_CUSTOM_HEADERS=$'Ocp-Apim-Subscription-Key: <key>' along with ANTHROPIC_BASE_URL. "
            "ANTHROPIC_API_KEY is not that header. The Claude CLI retries this 401 for several minutes, "
            "which is the 90 second probe timeout."
        )
    if status in {401, 403}:
        return f"Gateway returned HTTP {status}: {body[:300]}"
    return ""


def _custom_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    raw = os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "")
    for line in raw.splitlines():
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        name = name.strip()
        if name:
            headers[name] = value.strip()
    return headers


def check_gateway(model: str) -> None:
    """One direct POST so a gateway 401 fails before the CLI's retry loop."""
    base = os.environ.get("ANTHROPIC_BASE_URL", "").strip().rstrip("/")
    if not base:
        return
    for suffix in ("/v1/messages", "/v1"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    url = base + "/v1/messages"
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or ""
    headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    if key:
        headers["x-api-key"] = key
        headers["authorization"] = f"Bearer {key}"
    headers.update(_custom_headers())
    body = json.dumps(
        {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "OK"}]}
    ).encode()
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    print(f"Checking gateway {url}")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            response.read(200)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        reason = gateway_auth_failure(exc.code, detail)
        if reason:
            raise ClaudeBackendError(reason) from exc
    except ClaudeBackendError:
        raise
    except Exception as exc:  # noqa: BLE001 — the probe should say the gateway was unreachable
        raise ClaudeBackendError(f"Could not reach {url}: {exc}") from exc


def _load_sdk():
    try:
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query
    except ImportError as exc:
        raise ClaudeBackendError(
            "claude-agent-sdk is not installed. Install it with: pip install claude-agent-sdk"
        ) from exc
    return query, ClaudeAgentOptions, HookMatcher


def _deny(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


class ClaudeBackend:
    def __init__(self, *, repo: Path, model: str = "", executable: str = "") -> None:
        prepare_environment()
        self.repo = repo
        self.model = model.strip() or os.environ.get("DATASET_AGENT_MODEL", DEFAULT_CLAUDE_MODEL)
        self.executable = resolve_claude_cli(executable or os.environ.get("DATASET_AGENT_CLI", ""))
        self._query, self._options_type, self._hook_matcher = _load_sdk()

    def probe(self) -> None:
        """One tool-free request so a bad key or model fails before the campaign."""
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            raise ClaudeBackendError(
                "Set ANTHROPIC_API_KEY in the environment. The SDK does not load .env files."
            )
        check_gateway(self.model)
        command = [
            self.executable,
            "--print",
            "Reply with exactly OK. Do not inspect files or run tools.",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--permission-mode",
            "dontAsk",
            "--tools",
            "",
            "--max-turns",
            "1",
            "--no-session-persistence",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=90,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ClaudeBackendError(f"Claude probe failed for {self.model!r}: {exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise ClaudeBackendError(self._probe_failure(detail))
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ClaudeBackendError(f"Claude probe returned invalid JSON for {self.model!r}") from exc
        if str(payload.get("result") or "").strip() != "OK":
            raise ClaudeBackendError(self._probe_failure(json.dumps(payload)))

    def _probe_failure(self, detail: str) -> str:
        """Turn the CLI's JSON error into a short instruction."""
        first_party = "first_party" in detail and "Invalid API key" in detail
        base = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
        if first_party and not base:
            return (
                "Claude probe failed: the key was sent to Anthropic's own API and rejected with HTTP 401. "
                "This shell has no ANTHROPIC_BASE_URL, so the AMD token never reached the AMD gateway. "
                "In this same shell, before the dataset command, export "
                "ANTHROPIC_BASE_URL=https://llm-api.amd.com/anthropic, "
                "ANTHROPIC_API_KEY, and "
                "ANTHROPIC_CUSTOM_HEADERS=$'Ocp-Apim-Subscription-Key: <amd-token>'. "
                "Do not paste those values into the chat."
            )
        if first_party:
            return (
                f"Claude probe failed: HTTP 401 Invalid API key from {base}. "
                "ANTHROPIC_API_KEY is not accepted by that gateway."
            )
        return f"Claude probe failed for {self.model!r}: {detail[-1200:]}"

    def _hooks(self, spec: AgentRunSpec) -> dict[str, list[Any]]:
        repo = self.repo
        policy = spec.policy

        async def on_edit(input_data: dict, tool_use_id: str | None, context: Any) -> dict:
            del tool_use_id, context
            if input_data.get("tool_name") not in {"Write", "Edit", "NotebookEdit"}:
                return {}
            tool_input = input_data.get("tool_input") or {}
            raw = tool_input.get("file_path") or tool_input.get("path") or ""
            reason = edit_deny_reason(repo, policy, str(raw))
            return _deny(reason) if reason else {}

        async def on_bash(input_data: dict, tool_use_id: str | None, context: Any) -> dict:
            del tool_use_id, context
            if input_data.get("tool_name") != "Bash":
                return {}
            tool_input = input_data.get("tool_input") or {}
            command = str(tool_input.get("command") or "")
            reason = bash_deny_reason(repo, policy, command)
            return _deny(reason) if reason else {}

        matcher = self._hook_matcher
        return {
            "PreToolUse": [
                matcher(matcher="Write|Edit|NotebookEdit", hooks=[on_edit]),
                matcher(matcher="Bash", hooks=[on_bash]),
            ]
        }

    def _options(self, spec: AgentRunSpec, *, resume: str = "") -> Any:
        options: dict[str, Any] = {
            "model": spec.model or self.model,
            "cwd": spec.cwd,
            "system_prompt": spec.system_prompt,
            "cli_path": self.executable,
            "allowed_tools": list(spec.allowed_tools),
            "permission_mode": spec.permission_mode,
            "max_turns": spec.max_turns,
            "hooks": self._hooks(spec),
        }
        if resume:
            options["resume"] = resume
        extra = os.environ.get("DATASET_AGENT_FALLBACK_MODEL", "").strip()
        if extra and extra != options["model"]:
            options["fallback_model"] = extra
        return self._options_type(**options)

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        return await self._query_once(spec, spec.user_prompt)

    async def resume(self, spec: AgentRunSpec, session_id: str, feedback: str) -> AgentRunResult:
        if not session_id.strip():
            raise ClaudeBackendError("resume requires a session id")
        result = await self._query_once(spec, feedback, resume=session_id.strip())
        if not result.session_id:
            result.session_id = session_id.strip()
        return result

    async def _query_once(self, spec: AgentRunSpec, prompt: str, *, resume: str = "") -> AgentRunResult:
        options = self._options(spec, resume=resume)
        text_parts: list[str] = []
        tool_calls: list[tuple[str, dict[str, Any]]] = []
        subtype = ""
        num_turns: int | None = None
        session_id = ""
        stream_error: Exception | None = None
        print(f"\n[{spec.name}] session start model={spec.model or self.model} resume={bool(resume)}")
        try:
            async for message in self._query(prompt=prompt, options=options):
                candidate = getattr(message, "session_id", "") or ""
                if isinstance(candidate, str) and candidate:
                    session_id = candidate
                if hasattr(message, "total_cost_usd"):
                    subtype = getattr(message, "subtype", "") or ""
                    turns = getattr(message, "num_turns", None)
                    if isinstance(turns, int):
                        num_turns = turns
                    print(f"[{spec.name}] end subtype={subtype or '?'}")
                content = getattr(message, "content", None) or ()
                for block in content:
                    if hasattr(block, "text"):
                        text = " ".join(str(block.text).split())
                        if text:
                            print(f"[{spec.name}] say: {text[:_PROGRESS_CHARS]}")
                            text_parts.append(block.text)
                    elif block.__class__.__name__ == "ToolUseBlock":
                        name = getattr(block, "name", "?")
                        raw_input = getattr(block, "input", {}) or {}
                        tool_calls.append((name, raw_input))
                        detail = ""
                        if isinstance(raw_input, dict):
                            for key in ("file_path", "path", "pattern", "command"):
                                value = raw_input.get(key)
                                if isinstance(value, str) and value.strip():
                                    detail = value.strip()[:_PROGRESS_CHARS]
                                    break
                        print(f"[{spec.name}] tool: {name} {detail}".rstrip())
        except Exception as exc:  # noqa: BLE001 — keep the session id when the stream dies late
            if not session_id:
                raise
            stream_error = exc
            print(f"[{spec.name}] sdk-error: {exc}")

        if stream_error is not None:
            end_reason = "sdk_error"
            text_parts.append(f"\n[session ended with SDK error: {stream_error}]")
        elif subtype and subtype != "success":
            end_reason = f"sdk_{subtype}"
        else:
            end_reason = "agent_stopped"
        return AgentRunResult(
            text="\n".join(text_parts).strip(),
            session_id=session_id,
            subtype=subtype,
            end_reason=end_reason,
            num_turns=num_turns,
            tool_calls=tool_calls,
            stderr_tail=str(stream_error)[:2000] if stream_error else "",
        )
