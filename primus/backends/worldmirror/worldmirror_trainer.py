###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run HunyuanWorld-Mirror stage 1 / stage 2 inside the Primus trainer lifecycle.

Primus owns the launch, the config, and the attention-backend switch. World
Mirror keeps its Lightning + Hydra loop, DDP, and bf16 recipe. The only
training optimization applied here is aiter flash attention on the geometry
transformer. Gaussian rasterization stays in gsplat and stays float32.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

from primus.core.trainer.base_trainer import BaseTrainer
from primus.core.utils.module_utils import log_rank_0
from primus.core.utils.yaml_utils import nested_namespace_to_dict

_PRIMUS_ROOT = Path(__file__).resolve().parents[3]


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    return nested_namespace_to_dict(value)


_SUBMODULE_NAME = "HunyuanWorld-Mirror-rocm"


def _has_launch(path: Path) -> bool:
    return (path / "training" / "launch.py").is_file()


def resolve_worldmirror_repo(configured: str | None, primus_root: Path = _PRIMUS_ROOT) -> Path:
    """Find the World Mirror submodule checkout that contains training/launch.py."""
    explicit = (configured or "").strip() or os.environ.get("WORLD_MIRROR_PATH", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not _has_launch(path):
            raise FileNotFoundError(
                "HunyuanWorld-Mirror checkout is missing training/launch.py: " f"{path}"
            )
        return path.resolve()

    third_party_root = os.environ.get("PRIMUS_THIRDPARTY_DIR", "").strip()
    candidates = [primus_root / "third_party" / _SUBMODULE_NAME]
    if third_party_root:
        candidates.append(Path(third_party_root).expanduser() / _SUBMODULE_NAME)
    else:
        candidates.append(Path.home() / ".cache" / "Primus" / "third_party" / _SUBMODULE_NAME)

    for candidate in candidates:
        if _has_launch(candidate):
            return candidate.resolve()
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find the HunyuanWorld-Mirror submodule. Initialize it with "
        "`git submodule update --init third_party/HunyuanWorld-Mirror-rocm`. "
        f"Looked in: {tried}."
    )


def require_stage2_checkpoint(hydra_config: str, pretrained: str | None) -> None:
    """Stage 2 must load weights only. Resuming the stage-1 optimizer fails."""
    if "stage2" in hydra_config and not str(pretrained or "").strip():
        raise ValueError(
            "Stage 2 needs the stage-1 checkpoint in worldmirror.pretrained "
            "(or PRETRAINED_CKPT). Load weights only. Do not resume the stage-1 "
            "optimizer with ckpt_path."
        )


def build_hydra_overrides(cfg: Mapping[str, Any], env: Mapping[str, str] | None = None) -> list[str]:
    """Translate the Primus worldmirror block into Hydra overrides."""
    env = os.environ if env is None else env
    pretrained = str(cfg.get("pretrained") or "").strip()
    require_stage2_checkpoint(str(cfg["hydra_config"]), pretrained)

    overrides = [
        f"train={cfg['hydra_config']}",
        f"trainer.max_steps={int(cfg['max_steps'])}",
        f"data.max_images_per_gpu={int(cfg['max_images_per_gpu'])}",
        f"paths.hypersim_dir={cfg['hypersim_dir']}",
        f"paths.root_dir={cfg['output_dir']}",
    ]
    if pretrained:
        overrides.append(f"wrapper.pretrained={pretrained}")
    log_every = str(cfg.get("log_every_n_steps") or env.get("LOG_EVERY_N_STEPS") or "").strip()
    if log_every:
        overrides.append(f"+trainer.log_every_n_steps={int(log_every)}")
    # torchrun already created the processes. Tell Lightning how many, so it
    # joins that group instead of launching another one.
    if "LOCAL_RANK" in env and env.get("LOCAL_WORLD_SIZE"):
        overrides.append(f"trainer.devices={env['LOCAL_WORLD_SIZE']}")
        overrides.append(f"trainer.num_nodes={env.get('NNODES', '1')}")
    return overrides


def primus_hydra_overrides(overrides: list[str]) -> list[str]:
    """Select Primus Hydra configs without writing them into the submodule."""
    search_path = Path(__file__).resolve().parent / "hydra_configs"
    return [f"hydra.searchpath=[file://{search_path}]", *overrides]


def launch_worldmirror(repo: Path, overrides: list[str]) -> None:
    """Compose the World Mirror Hydra config and run its Lightning trainer."""
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from primus.backends.worldmirror.runtime import (
        disable_msc_hydra_search_path,
        install_runtime_fixes,
    )

    install_runtime_fixes()
    disable_msc_hydra_search_path()

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    config_dir = str(repo / "training" / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="train.yaml", overrides=primus_hydra_overrides(overrides))

    from training.launch import train as worldmirror_train

    worldmirror_train(cfg)


class WorldMirrorTrainer(BaseTrainer):
    """Primus lifecycle wrapper around World Mirror stage 1 and stage 2."""

    def __init__(self, backend_args: Any, *args, **kwargs):
        super().__init__(backend_args=backend_args, *args, **kwargs)
        self._cfg: dict | None = None
        self._repo: Path | None = None
        self._overrides: list[str] | None = None

    def setup(self):
        cfg = _as_dict(self.backend_args.worldmirror)
        backend = str(cfg.get("attention_backend") or "aiter").strip().lower()
        os.environ["WORLD_MIRROR_ATTN_BACKEND"] = backend
        log_rank_0(f"[Primus:WorldMirror] attention_backend={backend}")
        self._cfg = cfg

    def init(self):
        assert self._cfg is not None
        self._repo = resolve_worldmirror_repo(self._cfg.get("repo_path"))
        self._overrides = build_hydra_overrides(self._cfg)
        log_rank_0(f"[Primus:WorldMirror] repo={self._repo}")
        log_rank_0("[Primus:WorldMirror] hydra overrides: " + " ".join(self._overrides))

    def train(self):
        if self._repo is None or self._overrides is None:
            raise RuntimeError("WorldMirrorTrainer.init() must run before train().")
        launch_worldmirror(self._repo, self._overrides)
        log_rank_0("[Primus:WorldMirror] training finished.")
