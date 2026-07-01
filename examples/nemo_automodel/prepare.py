###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Backend prepare entry for the NeMo AutoModel framework.

Mirrors ``examples/torchtitan/prepare.py``: it resolves the backend checkout
(the ``third_party/Automodel`` submodule by default) and installs it editable so
the ``nemo_automodel`` package is importable for the training phase. Two
AutoModel-specific differences from the torchtitan flow:

  1. **ROCm-safe install.** AutoModel's dependency graph would otherwise let pip
     pull CUDA torch wheels on top of the image's ROCm build. We pin the native
     ROCm packages (torch/torchvision/torchaudio/triton/aiter/flash-attn) to the
     versions already installed and pass them as a pip *constraint*, so the
     editable install never replaces the ROCm stack. The ``diffusion`` extras
     are installed for the Wan diffusion recipe.
  2. **Idempotent / skip-if-present.** If ``nemo_automodel`` is already importable
     (e.g. a base image that ships it pre-installed) we skip the install unless
     ``AUTOMODEL_REINSTALL=1`` is set.

Unlike torchtitan there is no tokenizer asset to pre-download: the Wan diffusion
recipe pulls model weights from HuggingFace / a local ``/models`` mount at
runtime. The heavy lifting (config -> AutoModel ConfigNode -> TrainDiffusionRecipe)
is done later by ``NemoAutomodelPretrainTrainer`` inside the Primus core runtime.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from examples.scripts.utils import (
    get_env_case_insensitive,
    log_error_and_exit,
    log_info,
    write_patch_args,
)
from primus.core.launcher.parser import PrimusParser

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
# ``flash_attn_func`` supports ``return_lse``). The configs default to
# ``model.attention_backend: aiter``, so without a fix diffusers would refuse it.
_REQUIRED_AITER_VERSION = "0.1.5"


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
    return parser.parse_args()


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
            "or run `primus-cli deps sync`, or pass --backend_path / set AUTOMODEL_PATH."
        )

    extras = os.environ.get("AUTOMODEL_EXTRAS", "diffusion,diffusion-media")
    spec = f"{automodel_path}[{extras}]" if extras else str(automodel_path)
    constraints = write_rocm_constraints()

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
        except OSError:
            pass
    if ret is None or ret.returncode != 0:
        rc = ret.returncode if ret is not None else "n/a"
        log_error_and_exit(f"AutoModel editable install failed (exit {rc}).")
    log_info("AutoModel editable install complete.")


def maybe_shim_aiter_version():
    """Make diffusers' ``aiter`` backend selectable on dev-versioned aiter images.

    Rewrites only the ``Version:`` field in the installed ``amd-aiter`` dist
    metadata so it satisfies diffusers' minimum-version guard. This touches
    neither aiter, diffusers, nor AutoModel source; it reverts on image rebuild
    and is idempotent (skipped when the installed version already satisfies the
    guard, when aiter is absent, or when its metadata is read-only). Disable with
    ``AITER_VERSION_SHIM=0``.

    NOTE (single source of truth): the same metadata-rewrite is performed by the
    NeMo-AutoModel dev-env ``entrypoint.sh`` (tiger-training-internal #208). The
    two must stay in sync -- keep ``_REQUIRED_AITER_VERSION`` and the rewrite
    logic identical, or fold both onto a shared helper if the dev-env starts
    importing from this repo. Kept duplicated for now because the dev-env image
    does not depend on Primus at container-build time.
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


def ensure_automodel_installed(cli_path, primus_path: Path):
    """Install AutoModel from the submodule unless it is already importable."""
    reinstall = os.environ.get("AUTOMODEL_REINSTALL", "0") == "1"
    # Only take the skip-if-present fast path when the user did not explicitly
    # point us at a checkout (CLI --backend_path / AUTOMODEL_PATH / BACKEND_PATH).
    # If they did, honor it and (re)install so an explicit override beats a
    # nemo_automodel that the base image happens to ship (matches --backend_path help).
    explicit_source = (
        bool(cli_path)
        or bool(get_env_case_insensitive("AUTOMODEL_PATH"))
        or bool(get_env_case_insensitive("BACKEND_PATH"))
    )
    if not reinstall and not explicit_source:
        try:
            import nemo_automodel  # noqa: F401

            log_info(
                f"nemo_automodel already importable ({getattr(nemo_automodel, '__file__', '?')}); "
                "skipping install. Set AUTOMODEL_REINSTALL=1 to force a reinstall from third_party."
            )
            return
        except Exception:
            log_info("nemo_automodel not importable; installing from the AutoModel checkout.")

    automodel_path = resolve_backend_path(cli_path, primus_path)
    install_automodel_editable(automodel_path)


def main():
    args = parse_args()

    primus_path = Path(args.primus_path).resolve()
    exp_path = Path(args.config).resolve()
    patch_args_file = Path(args.patch_args).resolve()

    log_info(f"PRIMUS_PATH: {primus_path}")
    log_info(f"DATA_PATH: {Path(args.data_path).resolve()}")
    log_info(f"EXP: {exp_path}")
    log_info(f"BACKEND_PATH: {args.backend_path}")
    log_info(f"PATCH-ARGS: {patch_args_file}")

    if not exp_path.is_file():
        log_error_and_exit(f"EXP file not found: {exp_path}")

    # 1) Make the nemo_automodel package importable (submodule -> editable install).
    ensure_automodel_installed(args.backend_path, primus_path)

    # 1b) Make diffusers' aiter attention backend selectable on images that ship a
    #     dev-versioned amd-aiter (configs default to attention_backend: aiter).
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
