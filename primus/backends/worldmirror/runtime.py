###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ROCm training fixes that stay in Primus.

These wrap World Mirror objects after import. They do not write into the
third_party checkout. gsplat is compiled from a temporary copy, with the
ROCm source fixes and the missing GLM headers applied only in that copy.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _f32(value):
    import torch

    if torch.is_tensor(value) and value.is_floating_point():
        return value.float()
    return value


def install_raster_float32() -> None:
    """gsplat kernels accept float32. bf16 validation cameras are cast here."""
    import src.models.models.rasterization as rasterization

    if getattr(rasterization.Rasterizer.rasterize_splats, "_primus_wrapped", False):
        return
    original = rasterization.Rasterizer.rasterize_splats

    def rasterize_splats(
        self,
        means,
        quats,
        scales,
        opacities,
        colors,
        camtoworlds,
        Ks,
        width,
        height,
        **kwargs,
    ):
        return original(
            self,
            _f32(means),
            _f32(quats),
            _f32(scales),
            _f32(opacities),
            _f32(colors),
            _f32(camtoworlds),
            _f32(Ks),
            width,
            height,
            **kwargs,
        )

    rasterize_splats._primus_wrapped = True
    rasterization.Rasterizer.rasterize_splats = rasterize_splats


def install_render_depth_mask() -> None:
    """Depth confidence is a float. Compare it with > 0 before combining masks."""
    import torch
    import training.losses.render as render

    if getattr(render.RenderDepthLoss.compute_loss, "_primus_wrapped", False):
        return

    def compute_loss(self, preds, context_view_nums, dataset_name=None):
        depth = render.check_and_fix_inf_nan(preds["rendered_depths"], "render_depths")
        if self.depth_head_sup:
            gt_depth = preds["depth"].detach()
        else:
            gt_depth = preds["gt_depths"].unsqueeze(-1)
        if self.depth_conf_mask:
            valid_mask = preds["depth_conf"].detach()
        else:
            valid_mask = preds["valid_masks"].detach()

        gt_depth = render.check_and_fix_inf_nan(gt_depth, "render_gt_depth")
        valid_mask = valid_mask.clone()
        mask = torch.full_like(valid_mask, fill_value=True, dtype=bool)
        if self.mask_mode == "context_mask + target_mask":
            valid_mask = valid_mask
        elif self.mask_mode == "context_mask":
            mask[:, :context_view_nums, :, :] = valid_mask[:, :context_view_nums, :, :]
            valid_mask = mask
        elif self.mask_mode == "target_mask":
            mask[:, context_view_nums:, :, :] = valid_mask[:, context_view_nums:, :, :]
            valid_mask = mask
        else:
            valid_mask = mask
        if depth.shape[1] == 1 or valid_mask.sum() < 100:
            dummy_loss = (0.0 * depth).mean()
            loss_dict = {
                "loss_conf_renderdepth": dummy_loss,
                "loss_reg_renderdepth": dummy_loss,
                "loss_grad_renderdepth": dummy_loss,
            }
        else:
            gt_depth, valid_mask = gt_depth[:, : depth.shape[1]], valid_mask[:, : depth.shape[1]]
            weight = torch.ones(gt_depth.shape[0], 1, 1, 1, device=gt_depth.device)
            for i, name in enumerate(dataset_name):
                if name in self.norender_datatsets:
                    weight[i] *= 0.0
            dataset_mask = weight > 0
            valid_mask = (valid_mask > 0) & dataset_mask.expand_as(valid_mask)
            loss_conf, loss_grad, loss_reg = render.regression_loss(
                depth,
                gt_depth,
                valid_mask,
                conf=None,
                gradient_loss_fn=self.gradient_loss_fn,
                gamma=self.gamma,
                alpha=self.alpha,
                valid_range=self.valid_range,
            )
            loss_dict = {
                "loss_conf_renderdepth": loss_conf,
                "loss_reg_renderdepth": loss_reg,
                "loss_grad_renderdepth": loss_grad,
            }
        if "only" in self.gradient_loss_fn:
            loss_value = loss_dict["loss_grad_renderdepth"]
        else:
            loss_value = (
                loss_dict["loss_conf_renderdepth"]
                + loss_dict["loss_reg_renderdepth"]
                + loss_dict["loss_grad_renderdepth"]
            )
        return loss_value, loss_dict

    compute_loss._primus_wrapped = True
    render.RenderDepthLoss.compute_loss = compute_loss


def _drop_failed_gsplat_import() -> None:
    for name in list(sys.modules):
        if name == "gsplat" or name.startswith("gsplat."):
            sys.modules.pop(name, None)


def gsplat_imports() -> bool:
    try:
        import gsplat  # noqa: F401
    except ModuleNotFoundError:
        _drop_failed_gsplat_import()
        return False
    return True


_ROCM_PATCH = Path(__file__).with_name("gsplat_rocm.patch")
_GLM_URL = "https://github.com/g-truc/glm.git"
_GLM_COMMIT = "6f14f4792a0cde5d0cf2c910506724d61cb95834"


def _apply_rocm_patch(staged: Path) -> None:
    """Apply the ROCm gsplat fixes to the temporary build tree."""
    setup = staged / "setup.py"
    if not setup.is_file() or not (staged / "gsplat" / "cuda" / "include" / "Utils.cuh").is_file():
        return
    if "gsplat_glm" in setup.read_text(encoding="utf-8"):
        return
    subprocess.check_call(
        ["patch", "-p1", "--forward", "--batch", "-i", str(_ROCM_PATCH)],
        cwd=staged,
    )


def _ensure_glm(staged: Path) -> None:
    """Fetch GLM into the build tree. The submodule does not vendor it."""
    glm_root = staged / "gsplat" / "cuda" / "csrc" / "third_party" / "glm"
    if (glm_root / "glm" / "glm.hpp").is_file():
        return
    if not (staged / "gsplat" / "cuda" / "csrc").is_dir():
        return
    if glm_root.exists():
        shutil.rmtree(glm_root)
    glm_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--filter=blob:none", "--no-checkout", _GLM_URL, str(glm_root)])
    subprocess.check_call(["git", "fetch", "--depth", "1", "origin", _GLM_COMMIT], cwd=glm_root)
    subprocess.check_call(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=glm_root)


def _stage_gsplat_source(source: Path) -> Path:
    """Copy gsplat out of the git checkout before building it.

    Setuptools lists package files with git. The container user does not own
    the bind-mounted checkout, so that listing fails and the editable install
    stops. A copy under the temp directory is not inside the repository, so
    the build uses the filesystem instead of git. The submodule is not changed.
    """
    destination = Path(tempfile.mkdtemp(prefix="gsplat-build-"))

    def _ignore(directory: str, names: list[str]) -> set[str]:
        skipped = {".git", "build", "hip", "__pycache__"}
        return {name for name in names if name in skipped or name.endswith(".egg-info")}

    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=_ignore)
    return destination


def _git_config_for_install(env: dict) -> None:
    """Point this install at a throwaway git config that trusts every directory."""
    config = tempfile.NamedTemporaryFile("w", prefix="gsplat-gitconfig-", delete=False)
    config.write("[safe]\n\tdirectory = *\n")
    config.close()
    env["GIT_CONFIG_GLOBAL"] = config.name
    env["GIT_CONFIG_SYSTEM"] = os.devnull


def install_gsplat(repo: Path) -> None:
    """Build gsplat from the submodule when it is not already installed.

    The World Mirror install scripts call this once, before torchrun. The
    submodule checkout is not modified.
    """
    if gsplat_imports():
        return

    source = repo / "submodules" / "gsplat"
    if not (source / "setup.py").is_file():
        raise FileNotFoundError(f"gsplat source is missing setup.py: {source}")

    from primus.core.utils.module_utils import log_rank_0

    staged = _stage_gsplat_source(source)
    if (staged / "gsplat" / "cuda" / "csrc").is_dir():
        log_rank_0("[Primus:WorldMirror] applying ROCm gsplat fixes and fetching GLM")
    _apply_rocm_patch(staged)
    _ensure_glm(staged)
    env = os.environ.copy()
    env.setdefault("MAX_JOBS", "16")
    _git_config_for_install(env)
    log_rank_0(f"[Primus:WorldMirror] building gsplat from {staged}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", str(staged)],
        cwd=staged,
        env=env,
    )
    importlib.invalidate_caches()
    _drop_failed_gsplat_import()
    if not gsplat_imports():
        raise RuntimeError(f"gsplat install finished, but import still fails: {source}")


def disable_msc_hydra_search_path() -> None:
    """Keep the Multi-Storage Client plugin out of World Mirror's Hydra search path.

    The container ships multistorageclient, whose Hydra plugin adds ``msc://``
    to every search path. With no MSC profile configured, Hydra logs a full
    traceback while composing the config. World Mirror configs are local files.
    """
    from hydra.core.plugins import Plugins
    from hydra.plugins.search_path_plugin import SearchPathPlugin

    plugins = Plugins.instance()
    if getattr(plugins.discover, "_primus_wrapped", False):
        return
    original = plugins.discover

    def discover(plugin_type=None):
        found = original(plugin_type)
        if plugin_type is SearchPathPlugin:
            found = [
                plugin
                for plugin in found
                if "multistorageclient" not in plugin.__module__ and "msc" not in plugin.__module__.lower()
            ]
        return found

    discover._primus_wrapped = True
    plugins.discover = discover


def install_sampler_aspect_ratio() -> None:
    """Sample one aspect ratio as a scalar.

    ``Generator.uniform(..., size=1)`` returns a length-1 array. ``float()``
    of that array fails on this NumPy, so the training loader never yields a
    batch. Omitting ``size`` returns one scalar.
    """
    import training.data.sampler.dynamic_sampler as dynamic_sampler

    method = dynamic_sampler.DynamicBatchSampler._sample_view_idxs_and_ar_and_tp
    if getattr(method, "_primus_wrapped", False):
        return

    def _sample_view_idxs_and_ar_and_tp(self):
        _view_idxs = int(self.rng.choice(self.possible_nums, p=self.normalized_weights))
        _source_view_idxs = self._sample_source_view_idxs(_view_idxs)
        if self.aspect_ratio_range is not None:
            _aspect_ratio = float(self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]))
        else:
            _aspect_ratio = 1.0
        min_pixels = self.num_pixels_range[0]
        max_pixels = self.num_pixels_range[1]
        _target_pixels = int(self.rng.integers(min_pixels, max_pixels + 1))
        return _view_idxs, _source_view_idxs, _aspect_ratio, _target_pixels

    _sample_view_idxs_and_ar_and_tp._primus_wrapped = True
    dynamic_sampler.DynamicBatchSampler._sample_view_idxs_and_ar_and_tp = _sample_view_idxs_and_ar_and_tp


def install_runtime_fixes() -> None:
    from primus.backends.worldmirror.attention import install_aiter_attention

    install_aiter_attention()
    install_raster_float32()
    install_render_depth_mask()
    install_sampler_aspect_ratio()
