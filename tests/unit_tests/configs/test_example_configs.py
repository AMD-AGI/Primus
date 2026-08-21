###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Schema smoke test for every shipped experiment recipe under `examples/`.

The trainer E2E suites only launch one model per architecture, so most of the
~380 example recipes are never loaded by any other test -- a rename in
`primus/configs/models/`, a broken `extends:` chain or a `${VAR}` without a
default would ship unnoticed. This test loads all of them through the real
config stack (env interpolation -> extends merge -> module/model preset merge ->
experiment overrides) and asserts the resulting experiment is well formed.

It is CPU-only and needs no GPU: `load_primus_config` resolves YAML and presets
without importing megatron / torchtitan / jax (backends are only imported later,
by `PrimusRuntime._initialize_adapter`).
"""

from pathlib import Path

import pytest

from primus.core.config.primus_config import load_primus_config
from primus.core.utils import file_utils

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples"

# Backend dir -> the `framework` every experiment under it must declare.
BACKENDS = ("megatron", "torchtitan", "maxtext", "megatron_bridge")

MODULE_NAMES = {"pre_trainer", "post_trainer"}

# Files shipped next to the recipes that are not runnable experiments, so
# resolving them is not meaningful:
#   - lfm2...te-precision: a Transformer Engine precision-matcher fragment with
#     no work_group / modules of its own.
#   - native_hf_to_megatron_sft.template: a copy-and-fill template whose `model`
#     preset and paths are <...> placeholders by design, so it only resolves
#     once a user substitutes them.
NOT_AN_EXPERIMENT = {
    "examples/megatron/configs/MI355X/lfm2_8B_A1B-FP8-te-precision.yaml",
    "examples/megatron/configs/MI355X/native_hf_to_megatron_sft.template.yaml",
}


def _discover():
    found = []
    for backend in BACKENDS:
        for path in sorted((EXAMPLES / backend / "configs").rglob("*.yaml")):
            rel = path.relative_to(ROOT).as_posix()
            if rel not in NOT_AN_EXPERIMENT:
                found.append(rel)
    return found


EXAMPLE_CONFIGS = _discover()


@pytest.fixture(scope="module", autouse=True)
def _no_workspace_side_effects():
    """Keep loading side-effect free.

    `PrimusConfig.__init__` mkdir's `<workspace>/<group>/<user>/<exp_name>`, and
    most recipes hardcode `workspace: ./output`, so a bare load would litter the
    repo (and fail outright when `output/` is not writable by the test user).
    Only directory creation is stubbed; the config resolution under test is
    untouched.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(file_utils, "create_path_if_not_exists", lambda *args, **kwargs: None)
        yield


def test_every_backend_contributes_configs():
    """Guard the glob itself: a typo'd path would otherwise make the suite pass
    with zero recipes checked."""
    per_backend = {b: sum(1 for c in EXAMPLE_CONFIGS if c.startswith(f"examples/{b}/")) for b in BACKENDS}
    assert all(per_backend.values()), f"no example configs discovered for some backend: {per_backend}"


@pytest.mark.parametrize("rel_path", EXAMPLE_CONFIGS, ids=EXAMPLE_CONFIGS)
def test_example_config_loads(rel_path):
    cfg = load_primus_config(ROOT / rel_path, None)

    modules = getattr(cfg, "modules", [])
    assert modules, f"{rel_path}: no trainer module resolved"

    backend = rel_path.split("/")[1]
    for module in modules:
        assert module.name in MODULE_NAMES, f"{rel_path}: unexpected module '{module.name}'"
        # A recipe filed under examples/<backend>/ that declares another
        # framework would be launched by the wrong adapter.
        assert module.framework == backend, (
            f"{rel_path}: declares framework '{module.framework}' " f"but lives under examples/{backend}/"
        )
        assert hasattr(module, "params"), f"{rel_path}: module '{module.name}' resolved no params"
