###############################################################################
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Backend prepare entry for the NeMo AutoModel framework.

Mirrors the torchtitan hook next door: it resolves the backend checkout
(the ``third_party/Automodel`` submodule by default) and installs it editable so
the ``nemo_automodel`` package is importable for the training phase. Two
AutoModel-specific differences from the torchtitan flow:

  1. **ROCm-safe install.** AutoModel's dependency graph would otherwise let pip
     pull CUDA torch wheels on top of the image's ROCm build. We pin the native
     ROCm packages (torch/torchvision/torchaudio/triton/aiter/flash-attn) to the
     versions already installed and pass them as a pip *constraint*, so the
     editable install never replaces the ROCm stack. The ``diffusion`` extras
     are installed for the Wan diffusion recipe.
  2. **Pinned, not merely present.** Base images may ship no ``nemo_automodel`` or
     an older one, and the Primus patches need the commit ``primus/_thirdparty.lock``
     pins. An importable copy is kept only when it is that commit (or is the
     checkout itself) and its installed metadata and dependencies match that code;
     any other copy is replaced from the checkout, and every package version the
     install moves (transformers, typically) is logged. ``AUTOMODEL_REINSTALL=1``
     always reinstalls; ``AUTOMODEL_REINSTALL=0`` keeps whatever copy is importable
     and warns if it looks stale; ``PRIMUS_SKIP_PIP=1`` installs nothing.

Unlike torchtitan there is no tokenizer asset to pre-download: the Wan diffusion
recipe pulls model weights from HuggingFace / a local ``/models`` mount at
runtime. The heavy lifting (config -> AutoModel ConfigNode -> TrainDiffusionRecipe)
is done later by ``NemoAutomodelPretrainTrainer`` inside the Primus core runtime.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from primus.core.launcher.parser import PrimusParser
from runner.helpers.hooks.train.pretrain.utils import (
    get_env_case_insensitive,
    log_error_and_exit,
    log_info,
    log_warning,
    write_patch_args,
)

# Native ROCm packages to pin so the editable install never swaps them for CUDA
# wheels.
_ROCM_PINS = [
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "pytorch-triton-rocm",
    "amd-aiter",
    "flash-attn",
    "flash_attn",
]

# diffusers gates its ``aiter`` attention backend behind a minimum amd-aiter
# version. Some ROCm base images ship a functional but dev-versioned aiter (e.g.
# ``0.1.1.dev*``) that fails that guard even though the kernel works (its
# ``flash_attn_func`` supports ``return_lse``). The shipped Wan preset uses
# ``flash``: the name ``aiter`` is not in the pinned diffusers' backend list and
# raises before this version check runs. The shim still covers a config that
# asks for ``aiter`` on a diffusers that has the name.
_REQUIRED_AITER_VERSION = "0.1.5"

_AUTOMODEL_SUBMODULE = "third_party/Automodel"
_DEFAULT_EXTRAS = "diffusion,diffusion-media"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Primus NeMo AutoModel backend")
    parser.add_argument("--primus_path", type=str, required=True, help="Root path to the Primus project")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--patch_args",
        type=str,
        default="/tmp/primus_patch_args.txt",
        help="Path to write additional args (used during training phase)",
    )
    parser.add_argument(
        "--backend_path",
        type=str,
        default=None,
        help="Optional AutoModel checkout path; overrides AUTOMODEL_PATH/BACKEND_PATH and the default submodule.",
    )
    args, _ = parser.parse_known_args()
    return args


def resolve_backend_path(cli_path, primus_path: Path) -> Path:
    """CLI --backend_path > AUTOMODEL_PATH/BACKEND_PATH env > third_party/Automodel."""
    if cli_path:
        path = Path(cli_path).resolve()
        log_info(f"Using AutoModel path from CLI: {path}")
        return path
    env_value = get_env_case_insensitive("AUTOMODEL_PATH") or get_env_case_insensitive("BACKEND_PATH")
    if env_value:
        path = Path(env_value).resolve()
        log_info(f"Using AutoModel path from env: {path}")
        return path
    path = (primus_path / "third_party" / "Automodel").resolve()
    log_info(f"Using default AutoModel submodule path: {path}")
    return path


def write_rocm_constraints() -> str:
    """Pin currently-installed ROCm packages so pip treats them as satisfied."""
    import importlib.metadata as md

    lines = []
    for name in _ROCM_PINS:
        try:
            lines.append(f"{name}=={md.version(name)}")
        except md.PackageNotFoundError:
            continue
    fd, path = tempfile.mkstemp(prefix="primus_rocm_constraints.", suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    log_info(f"ROCm constraints ({len(lines)} pins) -> {path}")
    return path


def install_automodel_editable(automodel_path: Path):
    """ROCm-safe editable install of AutoModel with the diffusion extras."""
    if not (automodel_path / "pyproject.toml").exists() and not (automodel_path / "setup.py").exists():
        log_error_and_exit(
            f"AutoModel checkout not found at {automodel_path} (no pyproject.toml/setup.py).\n"
            "Initialize the submodule first:\n"
            "    git submodule update --init third_party/Automodel\n"
            "or run `primus-cli deps sync`, or pass --backend_path / set AUTOMODEL_PATH.\n"
            "AUTOMODEL_REINSTALL=0 keeps an already-importable nemo_automodel instead, pinned or not."
        )

    ensure_git_trusts_checkout(automodel_path)
    extras = os.environ.get("AUTOMODEL_EXTRAS", _DEFAULT_EXTRAS)
    spec = f"{automodel_path}[{extras}]" if extras else str(automodel_path)
    constraints = write_rocm_constraints()
    before = installed_versions()

    env = os.environ.copy()
    env["PIP_CONSTRAINT"] = constraints
    log_info(f"Installing AutoModel (editable, ROCm-pinned): pip install --no-build-isolation -e {spec!r}")
    ret = None
    try:
        ret = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", spec, "-q"],
            env=env,
            check=False,
        )
    except OSError as e:
        log_error_and_exit(f"Failed to invoke pip for AutoModel editable install: {e}")
    finally:
        try:
            os.unlink(constraints)
        except OSError as e:
            # Best-effort cleanup of the temp constraints file; a failure here is
            # non-fatal and must not mask the install result.
            log_info(f"Non-fatal: could not remove temp ROCm constraints file {constraints!r}: {e}")
    if ret is None or ret.returncode != 0:
        rc = ret.returncode if ret is not None else "n/a"
        log_error_and_exit(
            f"AutoModel editable install failed (exit {rc}). To use a nemo_automodel that is "
            "already importable instead (offline nodes, prepared images), set AUTOMODEL_REINSTALL=0 "
            "or PRIMUS_SKIP_PIP=1."
        )
    log_info("AutoModel editable install complete.")
    log_version_changes(before, installed_versions())


def _git_env() -> dict:
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def ensure_git_trusts_checkout(path: Path):
    """Add ``path`` to git's global ``safe.directory`` when git refuses it for ownership.

    A container running as root over a host-owned bind mount trips that check, and the
    install's setuptools-scm file finder then fails. setuptools-scm strips ``GIT_*``
    variables from the git it runs, so only the global config can reach it.
    """
    try:
        probe = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            env=_git_env(),
            timeout=30,
            check=False,
        )
        if probe.returncode == 0 or "dubious ownership" not in probe.stderr:
            return
        added = subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", str(path)],
            capture_output=True,
            text=True,
            env=_git_env(),
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return
    if added.returncode == 0:
        log_info(
            f"git refused {path} as owned by another user; added it to safe.directory in the "
            "global git config, which the AutoModel install needs to read the checkout."
        )
    else:
        log_warning(
            f"git refuses {path} as owned by another user, and adding it to safe.directory failed "
            f"({added.stderr.strip()}). Run: git config --global --add safe.directory {path}"
        )


def installed_versions() -> dict:
    """Installed distribution versions, keyed by normalised project name."""
    import importlib
    import importlib.metadata as md

    importlib.invalidate_caches()
    versions = {}
    for dist in md.distributions():
        name = dist.metadata["Name"] if dist.metadata else None
        if name:
            versions[name.lower().replace("_", "-")] = dist.version
    return versions


def log_version_changes(before: dict, after: dict):
    """Say which packages the install added, upgraded or downgraded."""
    changes = [
        f"{name} {before.get(name, '(new)')} -> {after[name]}"
        for name in sorted(after)
        if before.get(name) != after[name]
    ]
    if not changes:
        log_info("The AutoModel install changed no package versions.")
        return
    log_info(f"The AutoModel install changed {len(changes)} package version(s):")
    for change in changes:
        log_info(f"    {change}")


def pinned_automodel_commit(lock_path=None):
    """The AutoModel commit ``primus/_thirdparty.lock`` pins, or None without a lock entry."""
    if lock_path is None:
        import primus

        lock_path = Path(primus.__file__).resolve().parent / "_thirdparty.lock"
    try:
        entries = json.loads(Path(lock_path).read_text(encoding="utf-8"))["third_party"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    for entry in entries:
        if entry.get("path") == _AUTOMODEL_SUBMODULE:
            return entry.get("commit")
    return None


def _git_head(path: Path):
    """HEAD of the git checkout rooted exactly at ``path``, else None."""
    try:
        out = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(path), "rev-parse", "--show-toplevel", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = out.stdout.splitlines()
    if out.returncode != 0 or len(lines) != 2:
        return None
    toplevel, head = lines
    return head if Path(toplevel).resolve() == path.resolve() else None


def _direct_url_commit():
    """The commit pip recorded for a VCS install of nemo_automodel (PEP 610), else None."""
    import importlib.metadata as md

    try:
        raw = md.distribution("nemo_automodel").read_text("direct_url.json")
        return json.loads(raw or "{}").get("vcs_info", {}).get("commit_id")
    except (md.PackageNotFoundError, ValueError, AttributeError):
        return None


def importable_automodel():
    """``(source root, commit)`` of the nemo_automodel Python would import, or None.

    The commit is None when nothing records it -- a wheel baked into an image, say.
    Found without importing, so a mismatched copy never runs its import side effects.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec("nemo_automodel")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    root = Path(list(spec.submodule_search_locations)[0]).resolve().parent
    return root, _git_head(root) or _direct_url_commit()


def _same_commit(a, b) -> bool:
    if not a or not b:
        return False
    n = min(len(a), len(b))
    return n >= 7 and a[:n] == b[:n]


def _installed_dist_commit():
    """The commit the installed nemo_automodel metadata was built from, else None.

    setuptools-scm puts it in the local version (``0.7.0+bd1ca5a07``); a VCS install
    records it in ``direct_url.json`` instead.
    """
    import importlib.metadata as md

    try:
        local = md.version("nemo_automodel").partition("+")[2]
    except md.PackageNotFoundError:
        return None
    candidate = local.lstrip("g").split(".")[0]
    if len(candidate) >= 7 and all(c in "0123456789abcdef" for c in candidate):
        return candidate
    return _direct_url_commit()


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def _unsatisfied_requirements():
    """Declared requirements of the installed nemo_automodel that are not met, or None if not installed.

    The ROCm packages are left out: the install holds them at the image's
    versions on purpose, so a mismatch there is not something reinstalling fixes,
    and counting it would reinstall on every launch.
    """
    import importlib.metadata as md

    from packaging.requirements import Requirement

    try:
        declared = md.requires("nemo_automodel") or []
    except md.PackageNotFoundError:
        return None
    extras = [
        "",
        *filter(None, (e.strip() for e in os.environ.get("AUTOMODEL_EXTRAS", _DEFAULT_EXTRAS).split(","))),
    ]
    held = {_canonical(p) for p in _ROCM_PINS}
    unmet = []
    for line in declared:
        req = Requirement(line)
        if _canonical(req.name) in held:
            continue
        if req.marker is not None and not any(req.marker.evaluate({"extra": extra}) for extra in extras):
            continue
        try:
            version = md.version(req.name)
        except md.PackageNotFoundError:
            unmet.append(f"{req} (missing)")
            continue
        if req.specifier and not req.specifier.contains(version, prereleases=True):
            unmet.append(f"{req} (installed {version})")
    return unmet


def stale_install_reason(commit):
    """Why the installed nemo_automodel does not match the code at ``commit``, else None."""
    unmet = _unsatisfied_requirements()
    if unmet is None:
        return "nemo_automodel is importable but not installed, so none of its dependencies were"
    built_from = _installed_dist_commit()
    if commit and built_from and not _same_commit(built_from, commit):
        return f"its installed metadata is from {built_from[:12]}, not {commit[:12]}"
    if unmet:
        return "unsatisfied requirements: " + ", ".join(unmet)
    return None


def maybe_shim_aiter_version():
    """Make diffusers' ``aiter`` backend selectable on dev-versioned aiter images.

    Rewrites only the ``Version:`` field in the installed ``amd-aiter`` dist
    metadata so it satisfies diffusers' minimum-version guard. This touches
    neither aiter, diffusers, nor AutoModel source; it reverts on image rebuild
    and is idempotent (skipped when the installed version already satisfies the
    guard, when aiter is absent, or when its metadata is read-only). Disable with
    ``AITER_VERSION_SHIM=0``.

    """
    if os.environ.get("AITER_VERSION_SHIM", "1") != "1":
        log_info("AITER version shim disabled (AITER_VERSION_SHIM=0).")
        return

    import importlib.metadata as md
    import re

    try:
        dist = md.distribution("amd-aiter")
        cur = dist.version
    except md.PackageNotFoundError:
        log_info("amd-aiter not installed; AITER version shim skipped.")
        return

    try:
        from packaging.version import Version

        if Version(cur) >= Version(_REQUIRED_AITER_VERSION):
            log_info(f"AITER shim: amd-aiter {cur} already satisfies >={_REQUIRED_AITER_VERSION}; no change.")
            return
    except Exception:
        pass  # unparseable version -> attempt the bump below

    # Locate the metadata file via the public API (RECORD-relative paths resolved
    # through Distribution.locate_file). Fall back to the .dist-info dir only if
    # the distribution ships no RECORD (dist.files is None).
    meta_path = None
    try:
        for f in dist.files or []:
            if f.name in ("METADATA", "PKG-INFO"):
                located = dist.locate_file(f)
                if os.path.exists(located):
                    meta_path = str(located)
                    break
    except Exception:
        meta_path = None
    if not meta_path:
        try:
            base = dist._path  # PathDistribution .dist-info dir (fallback only)
            for cand in ("METADATA", "PKG-INFO"):
                p = os.path.join(str(base), cand)
                if os.path.exists(p):
                    meta_path = p
                    break
        except Exception:
            meta_path = None

    if not meta_path or not os.access(meta_path, os.W_OK):
        log_info(
            f"AITER shim: amd-aiter metadata not writable ({meta_path}); leaving {cur} as-is. "
            "If the diffusers aiter guard blocks it, set model.attention_backend: flash."
        )
        return

    with open(meta_path, encoding="utf-8") as f:
        txt = f.read()
    new, n = re.subn(r"(?im)^Version: .*$", f"Version: {_REQUIRED_AITER_VERSION}", txt, count=1)
    if n == 0 or new == txt:
        log_info(
            f"AITER shim: no Version field found in amd-aiter metadata ({meta_path}); leaving {cur} as-is. "
            "If the diffusers aiter guard blocks it, set model.attention_backend: flash."
        )
        return
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(new)
    log_info(
        f"AITER shim: bumped amd-aiter metadata Version {cur} -> {_REQUIRED_AITER_VERSION} "
        "to satisfy the diffusers aiter guard."
    )


def keep_importable_automodel(
    found, automodel_path: Path, pinned, reinstall: str, kept_by: str = "AUTOMODEL_REINSTALL=0"
) -> bool:
    """Whether the importable nemo_automodel can stand in for installing ``automodel_path``."""
    if found is None:
        log_info("nemo_automodel not importable; installing from the AutoModel checkout.")
        return False
    root, commit = found
    where = f"{root} at {commit[:12] if commit else 'an unrecorded commit'}"
    pin_note = f"; Primus pins {pinned[:12]}" if pinned and not _same_commit(commit, pinned) else ""
    if reinstall == "0":
        log_info(f"nemo_automodel importable from {where}; kept, as {kept_by}{pin_note}.")
        reason = stale_install_reason(commit)
        if reason:
            log_warning(f"the kept nemo_automodel may not run: {reason}.")
        return True
    if root == automodel_path.resolve():
        log_info(f"nemo_automodel importable from the AutoModel checkout itself ({where}){pin_note}.")
        if pin_note:
            log_warning(
                f"the AutoModel checkout is not the commit Primus pins ({pinned[:12]}). Run "
                "`git submodule update --init third_party/Automodel`; off the pin, the Primus "
                "repairs that cannot find the AutoModel APIs they patch stand aside."
            )
        reason = stale_install_reason(commit)
        if reason:
            log_warning(f"the checkout's install is stale ({reason}); reinstalling from {automodel_path}.")
            return False
        return True
    if pinned is None:
        log_info(
            f"nemo_automodel importable from {where}, and _thirdparty.lock pins no AutoModel commit; "
            "kept. Set AUTOMODEL_REINSTALL=1 to install from the checkout."
        )
        return True
    if _same_commit(commit, pinned):
        reason = stale_install_reason(commit)
        if reason:
            log_warning(f"nemo_automodel at {where} is the pinned commit, but {reason}; reinstalling.")
            return False
        log_info(f"nemo_automodel importable from {where}, the pinned commit; skipping install.")
        return True
    log_info(
        f"nemo_automodel importable from {where}, but Primus pins {pinned[:12]}; "
        f"reinstalling from {automodel_path}. Set AUTOMODEL_REINSTALL=0 to keep the importable copy."
    )
    return False


def ensure_automodel_installed(cli_path, primus_path: Path):
    """Install AutoModel from the checkout unless the importable copy is the pinned commit.

    ``AUTOMODEL_REINSTALL=1`` always installs, ``AUTOMODEL_REINSTALL=0`` keeps any
    importable copy. An explicit --backend_path / AUTOMODEL_PATH / BACKEND_PATH is
    always installed, so it beats a nemo_automodel the base image happens to ship.
    """
    reinstall = os.environ.get("AUTOMODEL_REINSTALL", "").strip()
    explicit_source = (
        bool(cli_path)
        or bool(get_env_case_insensitive("AUTOMODEL_PATH"))
        or bool(get_env_case_insensitive("BACKEND_PATH"))
    )
    automodel_path = resolve_backend_path(cli_path, primus_path)
    if os.environ.get("PRIMUS_SKIP_PIP", "").strip() == "1":
        found = importable_automodel()
        if found is None:
            log_error_and_exit(
                "PRIMUS_SKIP_PIP=1, but nemo_automodel is not importable. Unset it to let the hook "
                f"install AutoModel from {automodel_path}."
            )
        keep_importable_automodel(
            found, automodel_path, pinned_automodel_commit(), "0", kept_by="PRIMUS_SKIP_PIP=1"
        )
        return
    if reinstall != "1" and not explicit_source:
        if keep_importable_automodel(
            importable_automodel(), automodel_path, pinned_automodel_commit(), reinstall
        ):
            return
    install_automodel_editable(automodel_path)


def main():
    args = parse_args()

    primus_path = Path(args.primus_path).resolve()
    exp_path = Path(args.config).resolve()
    patch_args_file = Path(args.patch_args).resolve()

    log_info(f"PRIMUS_PATH: {primus_path}")
    log_info(f"DATA_PATH: {Path(args.data_path).resolve()}")
    log_info(f"EXP: {exp_path}")
    if args.backend_path:
        log_info(f"BACKEND_PATH (--backend_path): {args.backend_path}")
    log_info(f"PATCH-ARGS: {patch_args_file}")

    if not exp_path.is_file():
        log_error_and_exit(f"EXP file not found: {exp_path}")

    # 1) Make the nemo_automodel package importable (submodule -> editable install).
    ensure_automodel_installed(args.backend_path, primus_path)

    # 1b) Make diffusers' aiter attention backend selectable on images that ship a
    #     dev-versioned amd-aiter. The shipped preset uses flash; see the note above.
    maybe_shim_aiter_version()

    # 2) Validate the experiment parses and routes to the nemo_automodel backend.
    primus_cfg = PrimusParser().parse(args)
    try:
        pre_trainer_cfg = primus_cfg.get_module_config("pre_trainer")
    except Exception:
        log_error_and_exit("Missing required module config: pre_trainer")

    framework = getattr(pre_trainer_cfg, "framework", None)
    if framework != "nemo_automodel":
        log_error_and_exit(
            f"pre_trainer.framework must be 'nemo_automodel' (got {framework!r}). "
            "Check the experiment config."
        )

    if not getattr(pre_trainer_cfg, "model", None):
        log_error_and_exit("Missing required field: pre_trainer.model (model preset)")

    log_info(
        f"NeMo AutoModel backend ready (framework={framework}, model={pre_trainer_cfg.model}). "
        "Weights resolve via HF cache / /models at runtime."
    )

    # 3) Keep multi-GPU stdout to rank 0 only, consistent with other backends.
    write_patch_args(patch_args_file, "torchrun_args", {"local-ranks-filter": "0"})


if __name__ == "__main__":
    log_info("========== Prepare NeMo AutoModel backend ==========")
    main()
