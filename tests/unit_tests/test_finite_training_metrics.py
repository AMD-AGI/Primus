###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Detection of non-finite per-step metrics in trainer logs.

A diverged run still exits 0 and still prints the completion marker, so this
check is what keeps such a run from passing as green. It fails open by design:
a log it cannot parse reports zero checked steps rather than erroring, which
makes a silent parsing regression the failure mode worth testing for.
"""

import pytest

from tests.utils import assert_finite_training_metrics

TORCHTITAN = "step: {step}  loss: {loss}  grad_norm: {grad}  memory: 30.5GiB(38.10%)"
MEGATRON = (
    " iteration {step}/  3 | consumed samples: 24 | elapsed time per iteration (ms): 1234.5 |"
    " lm loss: {loss} | grad norm: {grad} | number of skipped iterations: 0 |"
)


@pytest.mark.parametrize("fmt", [TORCHTITAN, MEGATRON], ids=["torchtitan", "megatron"])
class TestFiniteTrainingMetrics:
    def test_clean_run_is_parsed(self, fmt):
        checked = assert_finite_training_metrics("t", fmt.format(step=1, loss="1.17E+01", grad="5.885"))
        assert checked == 1, "log format no longer recognized; divergence would go undetected"

    @pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
    def test_non_finite_loss_fails(self, fmt, bad):
        with pytest.raises(AssertionError, match="non-finite"):
            assert_finite_training_metrics("t", fmt.format(step=2, loss=bad, grad="5.885"))

    def test_non_finite_grad_norm_fails(self, fmt):
        with pytest.raises(AssertionError, match="non-finite"):
            assert_finite_training_metrics("t", fmt.format(step=3, loss="1.17E+01", grad="nan"))

    def test_only_the_diverged_step_is_reported(self, fmt):
        log = "\n".join(
            (
                fmt.format(step=1, loss="1.17E+01", grad="5.885"),
                fmt.format(step=2, loss="nan", grad="5.885"),
            )
        )
        with pytest.raises(AssertionError, match=r"step 2: loss=nan"):
            assert_finite_training_metrics("t", log)


def test_ansi_colored_log_is_still_parsed():
    colored = "\x1b[32m" + TORCHTITAN.format(step=1, loss="nan", grad="5.885") + "\x1b[0m"
    with pytest.raises(AssertionError, match="non-finite"):
        assert_finite_training_metrics("t", colored)


def test_unrecognized_format_is_not_a_failure():
    assert assert_finite_training_metrics("t", "some launcher output\nwithout metrics\n") == 0
