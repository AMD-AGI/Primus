###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the opt-in first-iteration log reset (training_log wrapper)."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from primus.backends.megatron.patches.training_log import (
    first_iteration_reset_patches as rp,
)
from tests.utils import require_megatron

REPO = Path(__file__).resolve().parents[5]


def _calls():
    calls = []

    def training_log(
        loss_dict, total_loss_dict, learning_rate, iteration, loss_scale, is_first_iteration=False
    ):
        calls.append((iteration, is_first_iteration))
        return "flag"

    return calls, training_log


@pytest.mark.parametrize(
    "iteration,first,log_interval,expected",
    [
        (1, True, 1, False),
        (10, True, 5, False),
        (11, True, 5, True),
        (2, False, 1, False),
        (3, True, 0, True),
    ],
)
@pytest.mark.parametrize("by_keyword", [True, False])
def test_first_iteration_on_a_boundary_is_logged_as_any_boundary(
    iteration, first, log_interval, expected, by_keyword
):
    calls, original = _calls()
    wrapped = rp.wrap(original, lambda: log_interval)
    if by_keyword:
        result = wrapped("ld", "tld", 0.1, iteration=iteration, loss_scale=1.0, is_first_iteration=first)
    else:
        result = wrapped("ld", "tld", 0.1, iteration, 1.0, first)
    assert result == "flag"
    assert calls == [(iteration, expected)]


def test_not_first_iteration_never_reads_log_interval():
    calls, original = _calls()
    rp.wrap(original, lambda: 1 // 0)("ld", "tld", 0.1, 4, 1.0)
    assert calls == [(4, False)]


def test_wrap_checks_the_signature_at_install():
    with pytest.raises(RuntimeError, match="is_first_iteration"):
        rp.wrap(lambda loss_dict, iteration: None)


def _ctx(value):
    from primus.core.patches import PatchContext

    params = SimpleNamespace(reset_log_on_first_iteration=value)
    return PatchContext(
        backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=params)}
    )


def test_condition_reads_the_flag_and_never_raises():
    assert rp.reset_log_on_first_iteration_enabled(_ctx(True))
    assert not rp.reset_log_on_first_iteration_enabled(_ctx(False))
    with pytest.raises(ValueError):
        rp.reset_log_on_first_iteration_enabled(_ctx("maybe"))
    assert rp.reset_log_on_first_iteration_condition(_ctx("maybe")) is True
    assert rp.reset_log_on_first_iteration_condition(_ctx(False)) is False


def test_a_bad_value_stops_the_run_with_a_clear_error():
    from primus.core.patches import run_patches

    params = SimpleNamespace(reset_log_on_first_iteration="ture")
    extra = {"module_config": SimpleNamespace(params=params)}
    with pytest.raises(SystemExit, match="reset_log_on_first_iteration"):
        run_patches(backend="megatron", phase="before_train", enabled_ids=[rp._KEY], extra=extra)


def test_install_wraps_megatron_training_log_once(monkeypatch):
    require_megatron()
    import megatron.training.training as megatron_training

    calls, original = _calls()
    monkeypatch.setattr(megatron_training, "training_log", original)
    monkeypatch.setattr(megatron_training, "_primus_applied_patch_keys", set(), raising=False)
    monkeypatch.setattr(rp, "_megatron_log_interval", lambda: 2)
    assert rp.install() is True
    wrapped = megatron_training.training_log
    assert wrapped is not original and wrapped.__wrapped__ is original
    assert rp.install() is False and megatron_training.training_log is wrapped
    wrapped("ld", "tld", 0.1, 4, 1.0, is_first_iteration=True)
    assert calls == [(4, False)]


def test_runs_before_the_mlperf_training_log_wrapper():
    import primus.backends.megatron.patches.mlperf_logging_patches  # noqa: F401
    from primus.core.patches import PatchRegistry

    assert (
        PatchRegistry.get(rp._KEY).priority < PatchRegistry.get("megatron.training.mlperf_logging").priority
    )


def test_megatron_training_log_has_what_the_wrapper_needs():
    """Upstream training_log takes ``iteration`` and ``is_first_iteration`` (checked by
    wrap) and resets only when ``is_first_iteration`` is False."""
    require_megatron()
    import megatron.training.training as megatron_training

    rp.wrap(megatron_training.training_log)
    assert "should_reset = not is_first_iteration" in inspect.getsource(megatron_training.training_log)


def test_module_default_is_off():
    import yaml

    cfg = yaml.safe_load((REPO / "primus/configs/modules/megatron/primus_megatron_module.yaml").read_text())
    assert cfg["reset_log_on_first_iteration"] is False
