###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Wrapper patches must forward arguments newer MaxText adds (no JAX import; run on the torch CI image)."""

import sys
import types

import pytest

from primus.backends.maxtext.patches import distributed_patches, train_loop_patches

_TRAIN_SOURCE = """
def training_loop_iteration(jax_device_state, python_vars, immutable_data, elastic_retry_cache=None):
  jax_device_state["state"] += 1
  return {"retry_cache": elastic_retry_cache}


def train_loop(config, recorder, state=None, elastic_retry_cache=None):
  jax_device_state = {"state": 0}
  metrics = training_loop_iteration(jax_device_state, {}, {}, elastic_retry_cache=elastic_retry_cache)
  return metrics, jax_device_state
"""

_MAX_UTILS_SOURCE = """
import jax


def maybe_initialize_jax_distributed_system(raw_keys, coordinator=None):
  return jax.distributed.initialize(coordinator_address=coordinator)


def print_system_information():
  pass
"""


@pytest.fixture
def fake_maxtext(tmp_path, monkeypatch):
    files = {
        "maxtext/__init__.py": "",
        "maxtext/trainers/__init__.py": "",
        "maxtext/trainers/pre_train/__init__.py": "",
        "maxtext/trainers/pre_train/train.py": _TRAIN_SOURCE,
        "maxtext/utils/__init__.py": "",
        "maxtext/utils/max_utils.py": _MAX_UTILS_SOURCE,
    }
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    calls = {"blocked": [], "initialize": []}
    jax = types.ModuleType("jax")
    jax.block_until_ready = calls["blocked"].append
    jax.distributed = types.SimpleNamespace(initialize=lambda **kw: calls["initialize"].append(kw) or kw)

    for name in [m for m in sys.modules if m == "maxtext" or m.startswith("maxtext.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.syspath_prepend(str(tmp_path))
    for module in (train_loop_patches, distributed_patches):
        monkeypatch.setattr(module, "log_rank_0", lambda *a, **k: None)
        monkeypatch.setattr(module, "warning_rank_0", lambda *a, **k: None)
    yield calls
    for name in [m for m in sys.modules if m == "maxtext" or m.startswith("maxtext.")]:
        sys.modules.pop(name)


def test_sync_each_step_forwards_elastic_retry_cache(fake_maxtext):
    from maxtext.trainers.pre_train import train

    train_loop_patches.patch_sync_each_step(None)
    metrics, jax_device_state = train.train_loop(None, None, elastic_retry_cache="cache")

    assert metrics == {"retry_cache": "cache"}
    assert fake_maxtext["blocked"] == [1] and jax_device_state["state"] == 1


def test_heartbeat_timeout_forwards_extra_args(fake_maxtext):
    from maxtext.utils import max_utils

    distributed_patches.patch_heartbeat_timeout(None)
    max_utils.maybe_initialize_jax_distributed_system(
        {"jax_distributed_heartbeat_timeout_seconds": 7}, coordinator="host:1234"
    )

    assert fake_maxtext["initialize"] == [
        {"coordinator_address": "host:1234", "heartbeat_timeout_seconds": 7}
    ]
