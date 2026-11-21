###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Publish a brought-up op: its own branch, commit and pull request.

Registration leaves the op model in the working tree.  With ``--op-pr`` the
projection agent also turns each registered (or acknowledged) op into a branch
off the current HEAD -- built in a temporary git worktree, so the checkout is
not touched -- with one commit holding the module and its regression test (or
the ``acknowledged.yaml`` entries), and, depending on the mode:

* ``commit`` -- leave the branch local;
* ``push``   -- push it to ``--op-pr-remote``;
* ``pr``     -- push it and open a draft pull request against ``--op-pr-base``
  (the remote's default branch), with ``gh`` when installed, else the GitHub
  REST API with ``$GITHUB_TOKEN`` / ``$GH_TOKEN``; with neither, the compare
  URL to open it from is printed.

The pull request body is the bring-up report: the gap, the validation checks
and cases, and the references the module cites.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from .bringup import BringupResult, _op_models_dir, acknowledge_keys

MODES = ("off", "commit", "push", "pr")


@dataclass
class PublishConfig:
    mode: str = "off"
    remote: str = "origin"
    base: Optional[str] = None
    repo: Optional[str] = None  # owner/name the pull request goes to; default: the remote's
    branch_prefix: str = "projection-agent"
    signoff: bool = False
    draft: bool = True
    author: Optional[str] = None  # "Name <email>"; default: git's configured identity


@dataclass
class PublishResult:
    op_id: str
    branch: Optional[str] = None
    commit: Optional[str] = None
    pushed: bool = False
    pr_url: Optional[str] = None
    compare_url: Optional[str] = None
    message: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _git(root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        cmd = next((a for a in args if not a.startswith("-") and "=" not in a), "")
        err = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
        raise RuntimeError(f"git {cmd}: {err[-1] if err else proc.returncode}")
    return proc.stdout.strip()


def _identity(root: Path, author: Optional[str]) -> list:
    """``git -c`` options for *author*, or ``[]`` when git already has an identity."""
    if author:
        m = re.fullmatch(r"\s*(.+?)\s*<([^<>]+)>\s*", author)
        if not m:
            raise ValueError(f"--op-pr-author must look like 'Name <email>', got {author!r}")
        return ["-c", f"user.name={m.group(1)}", "-c", f"user.email={m.group(2)}"]
    if subprocess.run(["git", "-C", str(root), "var", "GIT_COMMITTER_IDENT"], capture_output=True).returncode:
        raise RuntimeError("git has no user identity here; pass --op-pr-author 'Name <email>'")
    return []


def github_repo(url: str) -> Optional[str]:
    """``owner/name`` of a GitHub remote URL (https or ssh), else None."""
    m = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", url or "")
    return f"{m.group(1)}/{m.group(2)}" if m else None


def _default_base(root: Path, remote: str) -> str:
    ref = _git(root, "symbolic-ref", "--quiet", f"refs/remotes/{remote}/HEAD", check=False)
    return ref.rsplit("/", 1)[-1] if ref else "main"


def _module_meta(path: Path) -> Tuple[str, Tuple[str, ...]]:
    """First docstring paragraph and the ``references`` of a generated op module."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return "", ()
    doc = (ast.get_docstring(tree) or "").split("\n\n")[0].strip()
    refs: Tuple[str, ...] = ()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "references" for t in node.targets):
            try:
                refs = tuple(str(r) for r in ast.literal_eval(node.value))
            except ValueError:
                pass
    return doc, refs


def pr_title(res: BringupResult) -> str:
    if res.status == "acknowledged":
        return f"projection: acknowledge `{res.op_id}` as negligible ({', '.join(res.acknowledged)})"
    return f"projection: op model `{res.op_name}` for unmodeled op `{res.op_id}`"


def pr_body(res: BringupResult, *, gap: Optional[dict], workload: Path, module: Optional[Path]) -> str:
    gap = gap or {}
    lines = [
        f"The projection agent found an op the projection does not model (`{res.op_id}`) "
        f"in `{workload.name}` and brought it up.",
        "",
        "## Gap",
        "",
        f"- slot: `{gap.get('slot', '?')}`",
        f"- reason: {gap.get('reason', '?')}",
        f"- keys: {', '.join(f'`{k}`' for k in (gap.get('keys') or {})) or '?'}",
        "",
    ]
    if res.status == "acknowledged":
        lines += ["## Acknowledged as negligible", ""]
        lines += [f"- `{k}`: {v}" for k, v in res.acknowledged.items()]
    else:
        doc, refs = _module_meta(module) if module else ("", ())
        lines += [f"## Op model `{res.op_name}`", ""]
        if doc:
            lines += [doc, ""]
        if refs:
            lines += ["References:", ""] + [f"- {r}" for r in refs] + [""]
        val = res.validation or {}
        checks = val.get("checks", [])
        if checks:
            passed = sum(c.get("status") == "pass" for c in checks)
            lines += [f"## Validation (attempt {res.attempts}): {passed}/{len(checks)} checks pass", ""]
            lines += [
                f"- {c['status'].upper()} {c['name']}" + (f": {c['detail']}" if c.get("detail") else "")
                for c in checks
            ]
            lines.append("")
        cases = val.get("cases", [])
        if cases:
            lines += [
                "| case | batch | seq | fwd ms | bwd ms | roofline fwd ms |",
                "|---|---|---|---|---|---|",
            ]
            lines += [
                f"| {c['label']} | {c['batch_size']} | {c['seq_len']} | {c['fwd_ms']:.4f} | {c['bwd_ms']:.4f} "
                f"| {c['roofline_fwd_ms']:.4f} |"
                for c in cases
            ]
            lines.append("")
    lines += ["", "Opened by the Primus projection agent (`--op-pr`). Review the equations before merging."]
    return "\n".join(lines) + "\n"


def _open_pr(repo: str, head: str, base: str, title: str, body: str, draft: bool, cwd: Path) -> str:
    if shutil.which("gh"):
        cmd = ["gh", "pr", "create", "--repo", repo, "--base", base, "--head", head, "--title", title]
        cmd += ["--body", body] + (["--draft"] if draft else [])
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"gh pr create: {proc.stderr.strip()}")
        return proc.stdout.strip().splitlines()[-1]
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not token:
        raise LookupError("no gh and no GITHUB_TOKEN / GH_TOKEN")
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/pulls",
        data=json.dumps({"title": title, "body": body, "head": head, "base": base, "draft": draft}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["html_url"]


def publish_op(
    res: BringupResult,
    cfg: PublishConfig,
    *,
    primus_root: Path,
    workload: Path,
    gap: Optional[dict] = None,
    log: Callable[[str], None] = print,
) -> PublishResult:
    """Commit one brought-up op on its own branch, then push / open a PR as *cfg* says."""
    out = PublishResult(res.op_id)
    if cfg.mode == "off" or res.status not in ("registered", "acknowledged"):
        out.message = "nothing to publish"
        return out
    try:
        root = Path(_git(primus_root, "rev-parse", "--show-toplevel"))
        ident = _identity(root, cfg.author)
    except Exception as exc:
        out.message = f"publish skipped: {exc}"
        log(f"[op-pr] {res.op_id}: {out.message}")
        return out
    module = Path(res.module_path) if res.module_path else None
    files = [p for p in (module, Path(res.test_path) if res.test_path else None) if p is not None]
    if res.status == "registered" and not all(p.is_file() and root in p.resolve().parents for p in files):
        out.message = "registered files are not inside the repository (run with registration on)"
        return out

    out.branch = (
        f"{cfg.branch_prefix}/{re.sub(r'[^A-Za-z0-9_.-]+', '-', res.op_id)}-{time.strftime('%Y%m%d%H%M%S')}"
    )
    tmp = Path(tempfile.mkdtemp(prefix="op-pr-"))
    try:
        _git(root, "worktree", "add", "-q", "-b", out.branch, str(tmp), "HEAD")
        if res.status == "registered":
            rel = [str(p.resolve().relative_to(root)) for p in files]
            for r in rel:
                (tmp / r).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / r, tmp / r)
            init = Path(rel[-1]).parent / "__init__.py"
            if (root / init).is_file() and not (tmp / init).exists():
                shutil.copy2(root / init, tmp / init)
                rel.append(str(init))
        else:
            ack = acknowledge_keys(tmp, res.acknowledged)
            rel = [str(ack.relative_to(tmp))]
        _git(tmp, "add", "--", *rel)
        title = pr_title(res)
        body = pr_body(res, gap=gap, workload=workload, module=module)
        _git(tmp, *ident, "commit", "-q", *(["-s"] if cfg.signoff else []), "-m", title, "-m", body)
        out.commit = _git(tmp, "rev-parse", "HEAD")
        log(f"[op-pr] {res.op_id}: committed {out.commit[:9]} on {out.branch}")
        if cfg.mode == "commit":
            out.message = "committed on a local branch"
            return out

        _git(root, "push", "-q", "-u", cfg.remote, out.branch)
        out.pushed = True
        push_repo = github_repo(_git(root, "remote", "get-url", cfg.remote))
        repo = cfg.repo or push_repo
        base = cfg.base or _default_base(root, cfg.remote)
        ahead = _git(root, "rev-list", "--count", f"{cfg.remote}/{base}..{out.branch}", check=False)
        if ahead.isdigit() and int(ahead) > 1:
            log(
                f"[op-pr] {res.op_id}: the branch carries {int(ahead) - 1} commit(s) not on {base} besides the op"
            )
        log(f"[op-pr] {res.op_id}: pushed {out.branch} to {cfg.remote}")
        if cfg.mode == "push" or not repo:
            out.message = "pushed" + ("" if repo else " (remote is not on GitHub: no pull request)")
            return out

        owner = (push_repo or repo).split("/")[0]
        head = out.branch if push_repo == repo else f"{owner}:{out.branch}"
        out.compare_url = f"https://github.com/{repo}/compare/{base}...{head}?expand=1"
        try:
            out.pr_url = _open_pr(repo, head, base, title, body, cfg.draft, root)
            out.message = "pull request opened"
            log(f"[op-pr] {res.op_id}: {out.pr_url}")
        except Exception as exc:
            out.message = f"pushed; open the pull request at {out.compare_url} ({exc})"
            log(f"[op-pr] {res.op_id}: {out.message}")
        return out
    except Exception as exc:
        out.message = f"publish failed: {exc}"
        log(f"[op-pr] {res.op_id}: {out.message}")
        return out
    finally:
        _git(root, "worktree", "remove", "--force", str(tmp), check=False)
        shutil.rmtree(tmp, ignore_errors=True)
        if out.branch and out.commit is None:
            _git(root, "branch", "-q", "-D", out.branch, check=False)
            out.branch = None
