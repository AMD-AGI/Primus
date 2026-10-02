###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the prepare hook's AutoModel pin check.

Base images ship their own ``nemo_automodel``. The hook must replace any copy
that is not the commit ``primus/_thirdparty.lock`` pins, because the Primus
patches look up AutoModel APIs that older copies lack and then stand aside.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from runner.helpers.hooks.train.pretrain.nemo_automodel import prepare

REPO_ROOT = Path(__file__).resolve().parents[4]
PIN = "bd1ca5a07fbff806063bca8ce3a9071753ad2036"
OTHER = "44f2acde0000000000000000000000000000beef"

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


_REAL_STALE_INSTALL_REASON = prepare.stale_install_reason


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("AUTOMODEL_REINSTALL", "AUTOMODEL_PATH", "BACKEND_PATH", "PRIMUS_SKIP_PIP"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    # The keep decision should not depend on what the test environment has installed.
    monkeypatch.setattr(prepare, "stale_install_reason", lambda commit: None)


def write_lock(path, entries):
    path.write_text(json.dumps({"third_party": entries}), encoding="utf-8")
    return path


def test_pin_is_read_from_the_lock_entry_for_the_submodule(tmp_path):
    lock = write_lock(
        tmp_path / "lock",
        [
            {"name": "torchtitan", "path": "third_party/torchtitan", "commit": OTHER},
            {"name": "Automodel", "path": "third_party/Automodel", "commit": PIN},
        ],
    )
    assert prepare.pinned_automodel_commit(lock) == PIN


@pytest.mark.parametrize("content", ["not json", json.dumps({"third_party": []}), json.dumps({})])
def test_no_pin_without_a_readable_entry(tmp_path, content):
    lock = tmp_path / "lock"
    lock.write_text(content, encoding="utf-8")
    assert prepare.pinned_automodel_commit(lock) is None
    assert prepare.pinned_automodel_commit(tmp_path / "missing") is None


@requires_git
def test_shipped_lock_agrees_with_the_submodule_pointer():
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    # safe.directory, as the hook's own git calls do: run as root over a user-owned
    # checkout, git refuses the repository and the test would skip where it matters.
    out = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(REPO_ROOT), "ls-tree", "HEAD", "third_party/Automodel"],
        capture_output=True,
        text=True,
        check=True,
    )
    gitlink = out.stdout.split()[2]
    assert prepare.pinned_automodel_commit(REPO_ROOT / "primus" / "_thirdparty.lock") == gitlink


@pytest.mark.parametrize(
    "found, reinstall, pinned, keep",
    [
        (None, "", PIN, False),
        (("/image/site-packages", PIN), "", PIN, True),
        (("/image/site-packages", PIN[:12]), "", PIN, True),
        (("/image/site-packages", OTHER), "", PIN, False),
        (("/image/site-packages", None), "", PIN, False),
        (("/image/site-packages", OTHER), "0", PIN, True),
        (("/image/site-packages", None), "", None, True),
    ],
    ids=["absent", "pinned", "pinned-abbrev", "other-commit", "unrecorded", "opt-out", "no-lock-entry"],
)
def test_which_importable_copies_are_kept(tmp_path, capsys, found, reinstall, pinned, keep):
    if found is not None:
        found = (Path(found[0]), found[1])
    assert prepare.keep_importable_automodel(found, tmp_path / "checkout", pinned, reinstall) is keep
    if found is not None and found[1] == OTHER and pinned:
        assert PIN[:12] in capsys.readouterr().err


def test_the_checkout_itself_is_kept_even_off_the_pin(tmp_path, capsys):
    checkout = tmp_path / "Automodel"
    checkout.mkdir()
    assert prepare.keep_importable_automodel((checkout, OTHER), checkout, PIN, "")
    err = capsys.readouterr().err
    assert f"Primus pins {PIN[:12]}" in err
    assert "[WARN]" in err and "git submodule update --init third_party/Automodel" in err


def test_a_stale_install_of_the_checkout_is_reinstalled(tmp_path, monkeypatch, capsys):
    """The submodule moved under an editable install: same path, old metadata and deps."""
    checkout = tmp_path / "Automodel"
    checkout.mkdir()
    monkeypatch.setattr(
        prepare, "stale_install_reason", lambda commit: "its installed metadata is from c852b16ff972"
    )
    assert prepare.keep_importable_automodel((checkout, PIN), checkout, PIN, "") is False
    assert "reinstalling" in capsys.readouterr().err


def test_a_stale_pinned_copy_is_reinstalled(monkeypatch, capsys):
    """Another backend's hook moved a dependency after the pin was installed."""
    monkeypatch.setattr(
        prepare, "stale_install_reason", lambda commit: "unsatisfied requirements: transformers"
    )
    assert prepare.keep_importable_automodel((Path("/image"), PIN), Path("/checkout"), PIN, "") is False


def test_opting_out_keeps_a_stale_copy_but_says_so(monkeypatch, capsys):
    monkeypatch.setattr(
        prepare, "stale_install_reason", lambda commit: "unsatisfied requirements: transformers"
    )
    assert prepare.keep_importable_automodel((Path("/image"), OTHER), Path("/checkout"), PIN, "0") is True
    assert "[WARN]" in capsys.readouterr().err


class TestStaleInstallReason:
    @pytest.fixture(autouse=True)
    def real(self, monkeypatch):
        monkeypatch.setattr(prepare, "stale_install_reason", _REAL_STALE_INSTALL_REASON)

    def test_importable_but_not_installed(self, monkeypatch):
        monkeypatch.setattr(prepare, "_unsatisfied_requirements", lambda: None)
        assert "not installed" in prepare.stale_install_reason(PIN)

    def test_metadata_from_another_commit(self, monkeypatch):
        """Its requirements can all be met, which is why checking them alone is not enough."""
        monkeypatch.setattr(prepare, "_unsatisfied_requirements", lambda: [])
        monkeypatch.setattr(prepare, "_installed_dist_commit", lambda: "c852b16ff")
        assert "c852b16ff" in prepare.stale_install_reason(PIN)

    def test_unmet_requirements(self, monkeypatch):
        monkeypatch.setattr(
            prepare, "_unsatisfied_requirements", lambda: ["transformers==5.15.1 (installed 5.10.1)"]
        )
        monkeypatch.setattr(prepare, "_installed_dist_commit", lambda: PIN[:9])
        assert "transformers==5.15.1" in prepare.stale_install_reason(PIN)

    def test_a_matching_install(self, monkeypatch):
        monkeypatch.setattr(prepare, "_unsatisfied_requirements", lambda: [])
        monkeypatch.setattr(prepare, "_installed_dist_commit", lambda: PIN[:9])
        assert prepare.stale_install_reason(PIN) is None


@pytest.mark.parametrize(
    "version, expected",
    [("0.7.0+bd1ca5a07", "bd1ca5a07"), ("0.7.0+gbd1ca5a07.d20261001", "bd1ca5a07"), ("0.7.0", None)],
)
def test_installed_dist_commit_reads_the_local_version(monkeypatch, version, expected):
    import importlib.metadata as md

    monkeypatch.setattr(md, "version", lambda name: version)
    monkeypatch.setattr(prepare, "_direct_url_commit", lambda: None)
    assert prepare._installed_dist_commit() == expected


def test_requirements_on_the_held_rocm_packages_are_not_counted(monkeypatch):
    """The install holds those at the image's versions, so reinstalling cannot satisfy them."""
    import importlib.metadata as md

    monkeypatch.setattr(md, "requires", lambda name: ["torch>=99", "flash_attn>=99", "transformers==5.15.1"])
    monkeypatch.setattr(md, "version", lambda name: "1.0")
    assert prepare._unsatisfied_requirements() == ["transformers==5.15.1 (installed 1.0)"]


def test_skip_pip_keeps_the_importable_copy(monkeypatch, tmp_path, recorder, capsys):
    monkeypatch.setenv("PRIMUS_SKIP_PIP", "1")
    monkeypatch.setattr(prepare, "importable_automodel", lambda: (Path("/image/site-packages"), OTHER))
    prepare.ensure_automodel_installed(None, tmp_path)
    assert recorder == []
    assert "kept, as PRIMUS_SKIP_PIP=1" in capsys.readouterr().err


def test_skip_pip_with_nothing_importable_stops(monkeypatch, tmp_path, recorder):
    monkeypatch.setenv("PRIMUS_SKIP_PIP", "1")
    monkeypatch.setattr(prepare, "importable_automodel", lambda: None)
    with pytest.raises(SystemExit):
        prepare.ensure_automodel_installed(None, tmp_path)
    assert recorder == []


@pytest.fixture
def recorder(monkeypatch, tmp_path):
    installs = []
    monkeypatch.setattr(prepare, "install_automodel_editable", installs.append)
    monkeypatch.setattr(prepare, "pinned_automodel_commit", lambda: PIN)
    monkeypatch.setattr(
        prepare, "resolve_backend_path", lambda cli, root: tmp_path / "third_party" / "Automodel"
    )
    return installs


def test_image_copy_off_the_pin_is_reinstalled(monkeypatch, tmp_path, recorder):
    monkeypatch.setattr(prepare, "importable_automodel", lambda: (Path("/image/site-packages"), None))
    prepare.ensure_automodel_installed(None, tmp_path)
    assert recorder == [tmp_path / "third_party" / "Automodel"]


def test_pinned_copy_is_not_reinstalled(monkeypatch, tmp_path, recorder):
    monkeypatch.setattr(prepare, "importable_automodel", lambda: (Path("/elsewhere"), PIN))
    prepare.ensure_automodel_installed(None, tmp_path)
    assert recorder == []


@pytest.mark.parametrize("env", [("AUTOMODEL_REINSTALL", "1"), ("AUTOMODEL_PATH", "/explicit")])
def test_forced_or_explicit_installs_skip_the_check(monkeypatch, tmp_path, recorder, env):
    monkeypatch.setenv(*env)
    monkeypatch.setattr(prepare, "importable_automodel", lambda: pytest.fail("must not inspect the copy"))
    prepare.ensure_automodel_installed(None, tmp_path)
    assert len(recorder) == 1


def test_version_changes_name_every_moved_package(capsys):
    before = {"transformers": "5.12.0", "torch": "2.9.0", "nemo-automodel": "0.6.0"}
    after = {"transformers": "5.15.1", "torch": "2.9.0", "nemo-automodel": "0.7.0", "diffusers": "0.39.0"}
    prepare.log_version_changes(before, after)
    err = capsys.readouterr().err
    assert "changed 3 package version(s)" in err
    assert "transformers 5.12.0 -> 5.15.1" in err
    assert "diffusers (new) -> 0.39.0" in err
    assert "torch" not in err

    prepare.log_version_changes(after, after)
    assert "changed no package versions" in capsys.readouterr().err


def fake_git(monkeypatch, probe_stderr, probe_rc=128, add_rc=0):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("env")))
        if "rev-parse" in cmd:
            return subprocess.CompletedProcess(cmd, probe_rc, "", probe_stderr)
        return subprocess.CompletedProcess(cmd, add_rc, "", "" if add_rc == 0 else "read-only")

    monkeypatch.setattr(prepare.subprocess, "run", run)
    return calls


def test_checkout_refused_for_ownership_is_marked_safe(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    calls = fake_git(monkeypatch, "fatal: detected dubious ownership in repository at '/x'")
    prepare.ensure_git_trusts_checkout(tmp_path)
    assert calls[1][0] == ["git", "config", "--global", "--add", "safe.directory", str(tmp_path)]
    assert all(not k.startswith("GIT_") for _, env in calls for k in env)
    assert "added it to safe.directory" in capsys.readouterr().err


@pytest.mark.parametrize("stderr, rc", [("", 0), ("fatal: not a git repository", 128)])
def test_git_config_is_left_alone_otherwise(monkeypatch, tmp_path, stderr, rc):
    calls = fake_git(monkeypatch, stderr, probe_rc=rc)
    prepare.ensure_git_trusts_checkout(tmp_path)
    assert len(calls) == 1


def test_failure_to_mark_safe_says_what_to_run(monkeypatch, tmp_path, capsys):
    fake_git(monkeypatch, "fatal: detected dubious ownership", add_rc=1)
    prepare.ensure_git_trusts_checkout(tmp_path)
    assert f"git config --global --add safe.directory {tmp_path}" in capsys.readouterr().err


@requires_git
def test_git_head_only_for_the_checkout_root(tmp_path):
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert prepare._git_head(repo) == head
    assert prepare._git_head(repo / "sub") is None
    assert prepare._git_head(tmp_path) is None


def test_importable_copy_is_located_without_importing_it(monkeypatch, tmp_path):
    package = tmp_path / "src" / "nemo_automodel"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('imported')\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "nemo_automodel", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    monkeypatch.setattr(prepare, "_direct_url_commit", lambda: None)

    root, commit = prepare.importable_automodel()
    assert root == (tmp_path / "src").resolve()
    assert commit is None
