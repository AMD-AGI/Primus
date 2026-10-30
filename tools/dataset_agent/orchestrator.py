# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Phase machine for the Primus dataset agent.

Python decides when a phase is done. Each phase is one Claude Agent SDK session.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from tools.dataset_agent.gate import run_gate
from tools.dataset_agent.hub import fetch_dataset, format_hub_facts
from tools.dataset_agent.integration import is_infra_error, use_writable_hf_home
from tools.dataset_agent.locate import layout_for, repo_root
from tools.dataset_agent.prompts import (
    document_prompts,
    integrate_prompts,
    plan_prompts,
    test_prompts,
)
from tools.dataset_agent.spec import AgentRunResult, AgentRunSpec, PathPolicy, Task

_READ_TOOLS = ["Read", "Grep", "Glob"]
_EDIT_TOOLS = _READ_TOOLS + ["Edit", "Write", "Bash"]


def _stage(value: str) -> str:
    text = value.strip().lower().replace("-", "").replace("_", "")
    if text in {"pretrain", "pre"}:
        return "pretrain"
    if text in {"posttrain", "post", "sft"}:
        return "posttrain"
    raise argparse.ArgumentTypeError("stage must be pretrain or posttrain")


def _parse_args(argv: list[str] | None) -> Task:
    parser = argparse.ArgumentParser(
        description="Plug a Hugging Face dataset into a Primus backend and model.",
    )
    parser.add_argument("--backend", required=True, help="Primus backend, for example diffusion")
    parser.add_argument("--model", required=True, help="Model preset name, for example flux.1_schnell_t2i")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset id or published dataset name")
    parser.add_argument(
        "--stage",
        required=True,
        type=_stage,
        help="pretrain or posttrain. post-train and sft are accepted.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--llm-model", default="", help="Claude model id. Defaults to DATASET_AGENT_MODEL.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve files and the Hub record, then stop.")
    parser.add_argument("--no-branch", action="store_true", help="Stay on the current git branch.")
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    return Task(
        backend=args.backend.strip(),
        model=args.model.strip(),
        dataset=args.dataset.strip(),
        stage=args.stage,
        split=args.split.strip() or "train",
        samples=args.samples,
        max_retries=args.max_retries,
        llm_model=args.llm_model.strip(),
        dry_run=args.dry_run,
        no_branch=args.no_branch,
    )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


_TOOL_PREFIXES = (
    "tools/dataset_agent/",
    "tests/unit_tests/tools/dataset_agent/",
)


def _campaign_file(relative: str) -> bool:
    """Files that belong in the dataset patch, not the agent tool itself."""
    if relative == "tests/unit_tests/tools/test_dataset_agent.py":
        return False
    if relative.startswith(_TOOL_PREFIXES):
        return False
    if relative.startswith("primus/") or relative.startswith("docs/"):
        return True
    return "/dataset_agent/" in relative and relative.endswith((".yaml", ".yml"))


def ensure_branch(repo: Path, branch: str) -> str:
    """Create or check out ``branch``. Returns a note for the operator."""
    if _git(repo, "rev-parse", "--is-inside-work-tree").returncode != 0:
        return "Not a git checkout. Staying in the current directory."
    unrelated = [
        path
        for path in sorted(_porcelain(repo))
        if not _campaign_file(path)
        and not path.startswith(_TOOL_PREFIXES)
        and path != "tests/unit_tests/tools/test_dataset_agent.py"
    ]
    exists = _git(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}").returncode == 0
    command = ("checkout", branch) if exists else ("checkout", "-b", branch)
    switched = _git(repo, *command)
    if switched.returncode != 0:
        detail = (switched.stderr or switched.stdout).strip()
        return f"Could not switch to {branch}: {detail}"
    note = f"On branch {branch}."
    if unrelated:
        note += " Unrelated worktree files stay out of the dataset patch: " + ", ".join(unrelated)
    return note


def _porcelain(repo: Path) -> set[str]:
    if _git(repo, "rev-parse", "--is-inside-work-tree").returncode != 0:
        return set()
    listed = _git(repo, "status", "--porcelain", "-uall")
    paths: set[str] = set()
    for line in listed.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.add(path.strip())
    return paths


def _restore_paths(repo: Path, paths: set[str]) -> None:
    for relative in sorted(paths):
        target = repo / relative
        tracked = _git(repo, "ls-files", "--error-unmatch", "--", relative)
        if tracked.returncode == 0:
            _git(repo, "checkout", "--", relative)
        elif target.exists():
            if target.is_file():
                target.unlink()
            elif target.is_dir():
                # Only remove directories this phase created and that are now empty of tracked files.
                subprocess.run(["rm", "-rf", str(target)], check=False)


def _policy_delta(repo: Path, before: set[str], policy: PathPolicy) -> list[str]:
    from tools.dataset_agent.paths import write_allowed

    extra = _porcelain(repo) - before
    denied = [path for path in sorted(extra) if not write_allowed(policy, path)]
    return denied


def _writable_run_dir(preferred: Path) -> Path:
    """Use ``output/dataset_agent`` when the user can write there, otherwise the cache."""
    fallback = Path.home() / ".cache" / "primus-dataset-agent" / preferred.name
    parent = preferred.parent
    if parent.exists() and not os.access(parent, os.W_OK):
        print(f"Logs: {fallback}")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return preferred
    except OSError:
        print(f"Logs: {fallback}")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _knowledge(repo: Path) -> str:
    path = repo / "tools" / "dataset_agent" / "knowledge" / "backends.md"
    return path.read_text(encoding="utf-8")


def _spec(
    name: str, system: str, user: str, layout, task: Task, policy: PathPolicy, tools: list[str], turns: int
) -> AgentRunSpec:
    return AgentRunSpec(
        name=name,
        system_prompt=system,
        user_prompt=user,
        cwd=str(layout.repo),
        model=task.llm_model,
        max_turns=turns,
        allowed_tools=tools,
        policy=policy,
    )


async def _phase(
    backend, spec: AgentRunSpec, run_dir: Path, *, resume: str = "", feedback: str = ""
) -> AgentRunResult:
    result = await backend.resume(spec, resume, feedback) if resume else await backend.run(spec)
    _write_text(run_dir / f"{spec.name}.txt", result.text)
    return result


def _print_dry_run(task: Task, layout, hub_text: str, run_dir: Path) -> None:
    preset = layout.model_preset or "(not found)"
    template = layout.template_example or "(not found)"
    print("Dataset agent dry run")
    print(f"  backend:  {task.backend}")
    print(f"  model:    {task.model}")
    print(f"  dataset:  {task.dataset}")
    print(f"  stage:    {task.stage}")
    print(f"  preset:   {preset}")
    print(f"  template: {template}")
    print(f"  example:  {layout.example_yaml}")
    print(f"  test:     {layout.test_file}")
    print(f"  branch:   {layout.branch}")
    print(f"  log:      {run_dir}")
    print("Hub lookup:")
    print(hub_text)


async def campaign(task: Task) -> int:
    note = use_writable_hf_home()
    if note:
        print(note)
    repo = repo_root()
    layout = layout_for(repo, task.backend, task.model, task.dataset, task.stage)
    facts = fetch_dataset(task.dataset)
    hub_text = format_hub_facts(facts)
    run_dir = _writable_run_dir(layout.run_dir)
    _write_text(run_dir / "hub.json", json.dumps(facts, indent=2))
    if task.dry_run:
        _print_dry_run(task, layout, hub_text, run_dir)
        return 2 if facts.get("error") else 0

    from tools.dataset_agent.claude_backend import ClaudeBackend, ClaudeBackendError

    try:
        backend = ClaudeBackend(repo=repo, model=task.llm_model)
        backend.probe()
    except ClaudeBackendError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not task.no_branch:
        print(ensure_branch(repo, layout.branch))
    layout.example_yaml.parent.mkdir(parents=True, exist_ok=True)
    layout.test_file.parent.mkdir(parents=True, exist_ok=True)

    knowledge = _knowledge(repo)
    plan_system, plan_user = plan_prompts(task, layout, hub_text, knowledge)
    plan_policy = PathPolicy()
    plan = await _phase(
        backend,
        _spec("plan", plan_system, plan_user, layout, task, plan_policy, _READ_TOOLS, 12),
        run_dir,
    )

    integrate_system, integrate_user = integrate_prompts(task, layout, hub_text, knowledge, plan.text)
    integrate_policy = PathPolicy(write_roots=("primus", "examples"), allow_bash=True)
    integrate_spec = _spec(
        "integrate", integrate_system, integrate_user, layout, task, integrate_policy, _EDIT_TOOLS, 30
    )
    before = _porcelain(repo)
    integrated = await _phase(backend, integrate_spec, run_dir)
    _revert_if_needed(repo, before, integrate_policy, "integrate")

    test_system, test_user = test_prompts(task, layout, hub_text, knowledge)
    test_root = "tests/unit_tests/tools/dataset_agent"
    test_policy = PathPolicy(write_roots=(test_root,), allow_bash=True)
    test_spec = _spec("test", test_system, test_user, layout, task, test_policy, _EDIT_TOOLS, 20)
    before = _porcelain(repo)
    tested = await _phase(backend, test_spec, run_dir)
    _revert_if_needed(repo, before, test_policy, "test")

    ok, log = run_gate(
        repo,
        layout.test_file,
        layout.example_yaml,
        backend=task.backend,
        dataset=task.dataset,
        samples=task.samples,
        model_preset=layout.model_preset,
        split=task.split,
        stage=task.stage,
    )
    _write_text(run_dir / "gate.txt", log)
    attempt = 0
    while not ok and attempt < task.max_retries:
        fix_test = _gate_needs_test_fix(log, layout.test_file)
        if not fix_test and is_infra_error(log):
            print(
                "\n[gate] stopping: the failure is a host or network problem, not something an edit "
                f"to Primus can fix:\n{log}"
            )
            break
        attempt += 1
        if fix_test:
            if not tested.session_id:
                break
            print(f"\n[gate] test file rejected, resuming tester ({attempt}/{task.max_retries})")
            feedback = (
                "The Python gate rejected the pytest. Fix that file only. "
                "Keep the real Hugging Face load. Do not mock the loader. "
                "Do not skip. Do not import torch or primus.backends.megatron.sft.preprocessing. "
                "Stream the rows and call tools.dataset_agent.integration.check_megatron_rows.\n\n"
                f"{log}"
            )
            before = _porcelain(repo)
            tested = await _phase(backend, test_spec, run_dir, resume=tested.session_id, feedback=feedback)
            _revert_if_needed(repo, before, test_policy, "test")
        else:
            if not integrated.session_id:
                break
            print(f"\n[gate] failed, resuming integrator ({attempt}/{task.max_retries})")
            feedback = (
                "The Python gate failed. Fix the Primus loader or the example YAML. "
                "Do not edit the pytest.\n\n"
                f"{log}"
            )
            before = _porcelain(repo)
            integrated = await _phase(
                backend,
                integrate_spec,
                run_dir,
                resume=integrated.session_id,
                feedback=feedback,
            )
            _revert_if_needed(repo, before, integrate_policy, "integrate")
        previous_log = log
        ok, log = run_gate(
            repo,
            layout.test_file,
            layout.example_yaml,
            backend=task.backend,
            dataset=task.dataset,
            samples=task.samples,
            model_preset=layout.model_preset,
            split=task.split,
            stage=task.stage,
        )
        _write_text(run_dir / "gate.txt", log)
        if not ok and log == previous_log:
            print("\n[gate] stopping: the resume did not change the gate failure.")
            break

    if not ok:
        _write_results(repo, layout, run_dir, task, hub_text, passed=False, gate_log=log)
        print("Gate failed. Diff left in the worktree.", file=sys.stderr)
        print(log, file=sys.stderr)
        return 1

    doc_system, doc_user = document_prompts(task, layout, hub_text, knowledge)
    example_rel = layout.example_yaml.relative_to(repo).as_posix()
    doc_policy = PathPolicy(write_roots=("docs",), write_files=(example_rel,), allow_bash=False)
    before = _porcelain(repo)
    await _phase(
        backend,
        _spec(
            "document",
            doc_system,
            doc_user,
            layout,
            task,
            doc_policy,
            ["Read", "Grep", "Glob", "Edit", "Write"],
            15,
        ),
        run_dir,
    )
    _revert_if_needed(repo, before, doc_policy, "document")

    ok, log = run_gate(
        repo,
        layout.test_file,
        layout.example_yaml,
        backend=task.backend,
        dataset=task.dataset,
        samples=task.samples,
        model_preset=layout.model_preset,
        split=task.split,
        stage=task.stage,
    )
    _write_text(run_dir / "gate-final.txt", log)
    _write_results(repo, layout, run_dir, task, hub_text, passed=ok, gate_log=log)
    if not ok:
        print("Documentation edit broke the gate.", file=sys.stderr)
        print(log, file=sys.stderr)
        return 1
    print(f"\nDone. Example config: {layout.example_yaml}")
    print(f"Run log: {run_dir}")
    return 0


def file_patch(repo: Path, relative: str) -> str:
    """A git patch for one tracked change or one untracked file."""
    target = repo / relative
    if not target.is_file():
        return ""
    tracked = _git(repo, "ls-files", "--error-unmatch", "--", relative)
    if tracked.returncode == 0:
        diff = _git(repo, "diff", "--binary", "HEAD", "--", relative)
        text = diff.stdout
    else:
        diff = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--", "/dev/null", relative],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        text = diff.stdout
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def build_patch(repo: Path, example_relative: str) -> str:
    """Primus and docs edits, plus the example YAML the operator should run."""
    paths = [path for path in sorted(_porcelain(repo)) if _campaign_file(path)]
    if example_relative not in paths and (repo / example_relative).is_file():
        paths.append(example_relative)
    chunks = [file_patch(repo, path) for path in paths]
    return "".join(chunk for chunk in chunks if chunk.strip())


def render_report(
    *,
    backend: str,
    model: str,
    dataset: str,
    stage: str = "",
    branch: str,
    example_yaml: str,
    test_file: str,
    patch_path: str,
    hub_text: str,
    plan_text: str,
    passed: bool,
    gate_log: str,
    changed_files: list[str],
) -> str:
    """One page the operator can read after the campaign."""
    status = "passed" if passed else "failed"
    files = "\n".join(f"- `{path}`" for path in changed_files) or "- (no Primus or docs edits)"
    plan = plan_text.strip() or "(planner did not return a summary)"
    if len(plan) > 4000:
        plan = plan[:4000] + "\n…"
    gate = gate_log.strip() or "(no pytest output)"
    if len(gate) > 4000:
        gate = gate[-4000:]
    return f"""# Dataset agent report

## Result
The host gate **{status}**. It checked the example YAML, streamed real rows through the backend field parser, and ran pytest. No GPU and no container.

Use this config to try the dataset:

`{example_yaml}`

## Request
- Backend: `{backend}`
- Model: `{model}`
- Dataset: `{dataset}`
- Stage: `{stage or "(not recorded)"}`
- Branch: `{branch}`

## What changed
{files}

The patch with the Primus edits and that YAML is `{patch_path}`.

Apply it from the Primus repo with:

```bash
git apply --check {patch_path} && git apply {patch_path}
```

## Test
Pytest file: `{test_file}`

```bash
CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= python3 -m pytest {test_file} -q --tb=short
```

```
{gate}
```

## Hub lookup
```
{hub_text.strip()}
```

## Plan
{plan}
"""


def write_campaign_patch(repo: Path, layout, run_dir: Path) -> Path | None:
    """Write the patch next to the example YAML and in the run log."""
    example_relative = layout.example_yaml.relative_to(repo).as_posix()
    patch = build_patch(repo, example_relative)
    if not patch.strip():
        print("No Primus or example YAML changes to write into a patch.")
        return None
    destination = layout.example_yaml.with_suffix(".patch")
    _write_text(destination, patch)
    _write_text(run_dir / destination.name, patch)
    return destination


def _changed_files(repo: Path, example_relative: str) -> list[str]:
    paths = [path for path in sorted(_porcelain(repo)) if _campaign_file(path)]
    if example_relative not in paths and (repo / example_relative).is_file():
        paths.append(example_relative)
    return paths


def _write_results(
    repo: Path,
    layout,
    run_dir: Path,
    task: Task,
    hub_text: str,
    *,
    passed: bool,
    gate_log: str,
) -> tuple[Path | None, Path]:
    """Write the patch and the operator report. Returns their paths."""
    patch = write_campaign_patch(repo, layout, run_dir)
    example_relative = layout.example_yaml.relative_to(repo).as_posix()
    plan_path = run_dir / "plan.txt"
    plan_text = plan_path.read_text(encoding="utf-8") if plan_path.is_file() else ""
    report_text = render_report(
        backend=task.backend,
        model=task.model,
        dataset=task.dataset,
        stage=task.stage,
        branch=layout.branch,
        example_yaml=str(layout.example_yaml),
        test_file=str(layout.test_file),
        patch_path=str(patch) if patch is not None else "(no patch)",
        hub_text=hub_text,
        plan_text=plan_text,
        passed=passed,
        gate_log=gate_log,
        changed_files=_changed_files(repo, example_relative),
    )
    report = layout.example_yaml.with_suffix(".report.md")
    _write_text(report, report_text)
    _write_text(run_dir / report.name, report_text)
    if patch is not None:
        print(f"Patch: {patch}")
    print(f"Report: {report}")
    return patch, report


def _gate_needs_test_fix(log: str, test_file: Path) -> bool:
    """The tester owns a missing or rejected pytest. The integrator owns loader and YAML failures."""
    if not test_file.is_file():
        return True
    return log.startswith("test file") or log.startswith("pytest file is missing")


def _revert_if_needed(repo: Path, before: set[str], policy: PathPolicy, phase: str) -> None:
    denied = _policy_delta(repo, before, policy)
    if not denied:
        return
    print(f"[{phase}] reverting writes outside the phase policy: {', '.join(denied)}")
    _restore_paths(repo, set(denied))


def main(argv: list[str] | None = None) -> int:
    task = _parse_args(argv)
    try:
        return asyncio.run(campaign(task))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
