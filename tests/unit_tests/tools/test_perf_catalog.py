###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Guards for the tools/perf release performance suite.

The catalog is a hand-curated list of `examples/` paths, so its one real
failure mode is drift: a config gets renamed or moved and the suite quietly
points at nothing. Without this test that surfaces as a wasted benchmark run;
with it, it fails the PR that renamed the config.

Pure CPU, no GPU and no yq binary required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG = REPO_ROOT / "tools" / "perf" / "configs.yaml"


def _catalog() -> dict:
    with CATALOG.open() as fh:
        return yaml.safe_load(fh) or {}


def _entries() -> list[tuple[str, str, str]]:
    """Flatten the catalog into (gpu, backend, path) triples."""
    out = []
    for gpu, backends in _catalog().items():
        for backend, configs in (backends or {}).items():
            for path in configs or []:
                out.append((gpu, backend, path))
    return out


def test_catalog_exists_and_parses():
    assert CATALOG.is_file(), f"missing catalog: {CATALOG}"
    assert _catalog(), "catalog is empty"


def test_catalog_is_gpu_then_backend_then_list():
    for gpu, backends in _catalog().items():
        assert isinstance(backends, dict), f"{gpu}: expected a mapping of backend -> list"
        for backend, configs in backends.items():
            # A null value means every entry under that backend is commented
            # out, which is allowed; a non-list of anything else is not.
            if configs is None:
                continue
            assert isinstance(configs, list), f"{gpu}.{backend}: expected a list of paths"
            for path in configs:
                assert isinstance(path, str), f"{gpu}.{backend}: expected path strings, got {path!r}"


def test_catalog_is_not_empty_after_flattening():
    assert _entries(), "no enabled configs in the catalog; is everything commented out?"


@pytest.mark.parametrize("gpu,backend,rel_path", _entries())
def test_catalog_path_exists(gpu: str, backend: str, rel_path: str):
    assert (REPO_ROOT / rel_path).is_file(), f"{gpu}.{backend}: config not found: {rel_path}"


@pytest.mark.parametrize("gpu,backend,rel_path", _entries())
def test_catalog_config_declares_a_train_module(gpu: str, backend: str, rel_path: str):
    """Every entry must be something the runner can actually launch.

    The runner picks its CLI verb from `modules.pre_trainer` (`train
    pretrain`) or `modules.post_trainer` (`train posttrain`). A config with
    neither -- a hardware descriptor, an include fragment -- would be
    selected and then fail at launch.
    """
    path = REPO_ROOT / rel_path
    if not path.is_file():
        pytest.skip("covered by test_catalog_path_exists")

    with path.open() as fh:
        config = yaml.safe_load(fh) or {}

    modules = config.get("modules") or {}
    assert (
        "pre_trainer" in modules or "post_trainer" in modules
    ), f"{rel_path}: declares neither modules.pre_trainer nor modules.post_trainer"


def test_catalog_has_no_duplicate_paths():
    """Duplicates would produce two runs writing the same log filename.

    The log name is <framework>-<config>-<hash>-MBS..-GBS..-rep<n>_<stamp>,
    so a config listed twice collides with itself and the second run silently
    overwrites the first.
    """
    seen: dict[str, str] = {}
    for gpu, backend, path in _entries():
        key = f"{gpu}/{path}"
        assert key not in seen, f"duplicate entry {path} under {gpu} ({seen[key]} and {backend})"
        seen[key] = backend


def test_catalog_backend_matches_config_framework():
    """The backend key must agree with the framework the config declares.

    Filing a maxtext config under `megatron:` would make BACKEND=megatron
    select it and then run it with the wrong step-override flag.
    """
    for gpu, backend, rel_path in _entries():
        path = REPO_ROOT / rel_path
        if not path.is_file():
            continue
        with path.open() as fh:
            config = yaml.safe_load(fh) or {}
        modules = config.get("modules") or {}
        module = modules.get("pre_trainer") or modules.get("post_trainer") or {}
        framework = module.get("framework")
        if framework is None:
            continue
        assert framework == backend, f"{rel_path}: listed under '{backend}' but framework is '{framework}'"
