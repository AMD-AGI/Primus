###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Projection agent ``--op-pr``: a brought-up op becomes its own branch, commit
and pull request, against a throwaway repository and a local bare remote."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path

import pytest

from primus.agents.projection_agent import cli as pa_cli
from primus.agents.projection_agent import publish as pub
from primus.agents.projection_agent.bringup import BringupResult

_GEN = Path("primus/core/projection/op_models/generated")
_TESTS = Path("tests/unit_tests/core/projection/op_models")

_MODULE = '''"""Foo op: a made-up op for the publish test.

More detail.
"""

class FooOp:
    name = "foo_op"
    references = ("Foo et al., 2026", "https://example.com/foo")
'''


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for k in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{k}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{k}_EMAIL", "test@example.com")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    root = tmp_path / "repo"
    (root / _GEN).mkdir(parents=True)
    (root / _GEN / "__init__.py").write_text("")
    (root / _GEN / "acknowledged.yaml").write_text("old_key: >-\n  negligible\n")
    _git(tmp_path, "init", "-q", "-b", "main", str(root))
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "init")
    _git(root, "remote", "add", "origin", str(remote))
    _git(root, "push", "-q", "-u", "origin", "main")
    _git(root, "remote", "set-head", "origin", "main")
    return root, remote


def _registered(root: Path) -> BringupResult:
    mod, test = root / _GEN / "foo_op.py", root / _TESTS / "test_generated_foo_op.py"
    test.parent.mkdir(parents=True, exist_ok=True)
    mod.write_text(_MODULE)
    test.write_text("def test_foo():\n    pass\n")
    return BringupResult(
        "foo",
        "registered",
        op_name="foo_op",
        module_path=str(mod),
        test_path=str(test),
        attempts=2,
        validation={
            "ok": True,
            "checks": [{"name": "roofline", "status": "pass", "detail": ""}],
            "cases": [
                {
                    "label": "workload",
                    "batch_size": 1,
                    "seq_len": 4096,
                    "fwd_ms": 1.0,
                    "bwd_ms": 2.0,
                    "roofline_fwd_ms": 0.5,
                }
            ],
        },
    )


_GAP = {"op_id": "foo", "slot": "attention_extra", "reason": "foo_type set", "keys": {"foo_type": "x"}}


def _publish(root, res, mode, **kw):
    return pub.publish_op(
        res,
        pub.PublishConfig(mode=mode, **kw),
        primus_root=root,
        workload=Path("w.yaml"),
        gap=_GAP,
        log=lambda s: None,
    )


def test_commit_mode_branches_off_head_without_touching_the_checkout(repo):
    root, _ = repo
    res = _registered(root)
    status = _git(root, "status", "--porcelain")
    out = _publish(root, res, "commit")
    assert out.commit and not out.pushed, out.message
    files = _git(root, "show", "--name-only", "--format=", out.branch).split()
    assert sorted(files) == sorted([str(_GEN / "foo_op.py"), str(_TESTS / "test_generated_foo_op.py")])
    msg = _git(root, "log", "-1", "--format=%B", out.branch)
    assert msg.startswith("projection: op model `foo_op` for unmodeled op `foo`")
    assert "Foo et al., 2026" in msg and "| workload | 1 | 4096 |" in msg and "`foo_type`" in msg
    assert _git(root, "status", "--porcelain") == status
    assert _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    assert len(_git(root, "worktree", "list").splitlines()) == 1


def test_push_mode_pushes_the_branch(repo):
    root, remote = repo
    out = _publish(root, _registered(root), "push", signoff=True)
    assert out.pushed and out.pr_url is None
    assert _git(remote, "rev-parse", out.branch).strip() == out.commit
    assert "Signed-off-by: Test <test@example.com>" in _git(root, "log", "-1", "--format=%B", out.branch)
    assert "not on GitHub" in out.message


def test_acknowledged_op_commits_only_the_new_entries(repo):
    root, _ = repo
    res = BringupResult("bar", "acknowledged", acknowledged={"bar_key": "a tiny elementwise op"})
    out = _publish(root, res, "commit")
    assert _git(root, "show", "--name-only", "--format=", out.branch).split() == [
        str(_GEN / "acknowledged.yaml")
    ]
    ack = _git(root, "show", f"{out.branch}:{_GEN / 'acknowledged.yaml'}")
    assert "old_key" in ack and "bar_key: >-" in ack
    assert "bar_key" not in (root / _GEN / "acknowledged.yaml").read_text()


@pytest.fixture
def github_repo(repo):
    root, remote = repo
    _git(root, "remote", "set-url", "origin", "https://github.com/AMD-AGI/Primus.git")
    _git(root, "remote", "set-url", "--push", "origin", str(remote))
    return root


def test_pr_mode_without_gh_or_token_prints_the_compare_url(github_repo, monkeypatch):
    monkeypatch.setattr(pub.shutil, "which", lambda _: None)
    out = _publish(github_repo, _registered(github_repo), "pr")
    assert out.pushed and out.pr_url is None
    assert out.compare_url == f"https://github.com/AMD-AGI/Primus/compare/main...{out.branch}?expand=1"
    assert out.compare_url in out.message


def test_pr_mode_opens_a_draft_pull_request_through_the_rest_api(github_repo, monkeypatch):
    sent = {}

    def urlopen(req, timeout):
        sent.update(url=req.full_url, auth=req.headers["Authorization"], body=json.loads(req.data))
        return io.BytesIO(json.dumps({"html_url": "https://github.com/AMD-AGI/Primus/pull/1"}).encode())

    monkeypatch.setattr(pub.shutil, "which", lambda _: None)
    monkeypatch.setenv("GITHUB_TOKEN", "t0k")
    monkeypatch.setattr(pub.urllib.request, "urlopen", urlopen)
    out = _publish(github_repo, _registered(github_repo), "pr", repo="AMD-AGI/Primus", base="dev")
    assert out.pr_url == "https://github.com/AMD-AGI/Primus/pull/1"
    assert sent["url"] == "https://api.github.com/repos/AMD-AGI/Primus/pulls"
    assert sent["auth"] == "Bearer t0k"
    body = sent["body"]
    assert body["head"] == out.branch and body["base"] == "dev" and body["draft"] is True
    assert "Validation (attempt 2): 1/1 checks pass" in body["body"]


def test_pr_to_upstream_from_a_fork_names_the_fork_owner(github_repo, monkeypatch):
    sent = {}
    monkeypatch.setattr(pub.shutil, "which", lambda _: None)
    monkeypatch.setenv("GH_TOKEN", "t")
    monkeypatch.setattr(
        pub.urllib.request,
        "urlopen",
        lambda req, timeout: sent.update(json.loads(req.data)) or io.BytesIO(b'{"html_url": "u"}'),
    )
    _git(github_repo, "remote", "set-url", "origin", "git@github.com:araina/Primus.git")
    out = _publish(github_repo, _registered(github_repo), "pr", repo="AMD-AGI/Primus")
    assert sent["head"] == f"araina:{out.branch}"


def test_nothing_is_published_for_failed_bringups_or_mode_off(repo):
    root, _ = repo
    assert _publish(root, BringupResult("x", "failed"), "pr").branch is None
    assert _publish(root, _registered(root), "off").branch is None
    assert "projection-agent/" not in _git(root, "branch", "--list")


@pytest.fixture
def no_identity(repo, tmp_path, monkeypatch):
    for k in ("AUTHOR", "COMMITTER"):
        monkeypatch.delenv(f"GIT_{k}_NAME")
        monkeypatch.delenv(f"GIT_{k}_EMAIL")
    monkeypatch.delenv("EMAIL", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.useConfigOnly")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")
    return repo[0]


def test_without_a_git_identity_nothing_is_branched(no_identity):
    out = _publish(no_identity, _registered(no_identity), "commit")
    assert out.branch is None and "--op-pr-author" in out.message
    assert "projection-agent/" not in _git(no_identity, "branch", "--list")


def test_op_pr_author_commits_without_touching_git_config(no_identity):
    out = _publish(no_identity, _registered(no_identity), "commit", author="Op Bot <op-bot@example.com>")
    assert (
        _git(no_identity, "log", "-1", "--format=%an <%ae>", out.branch).strip()
        == "Op Bot <op-bot@example.com>"
    )
    assert "user." not in _git(no_identity, "config", "--local", "--list")


def test_op_pr_flags_build_the_publish_config():
    p = argparse.ArgumentParser()
    pa_cli.add_op_bringup_args(p)
    assert pa_cli.publish_from_args(p.parse_args([])) is None
    cfg = pa_cli.publish_from_args(p.parse_args(["--op-pr", "pr", "--op-pr-base", "dev", "--op-pr-ready"]))
    assert (cfg.mode, cfg.base, cfg.draft, cfg.remote) == ("pr", "dev", False, "origin")
