###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the explicit validation timestep source.

Before this knob existed, a validation batch with no ``timestep`` field silently
fell back to injecting ``arange(B) % 8``. That is indistinguishable from a
dataset that legitimately has no per-sample timestep, so an ingest that dropped
the column produced a plausible-looking val_loss computed against the wrong
image-to-timestep pairing. These tests pin the two sources apart.
"""

import logging

import pytest
import torch

from primus.backends.megatron.training.diffusion.forward_step import (
    DATASET_TIMESTEPS,
    EQUIDISTANT_TIMESTEPS,
    NUM_VALIDATION_TIMESTEPS,
    resolve_validation_timesteps,
)


def _resolve(batch, source, batch_size, compute_dtype=torch.float32):
    return resolve_validation_timesteps(
        batch,
        source,
        batch_size=batch_size,
        device=torch.device("cpu"),
        compute_dtype=compute_dtype,
    )


class TestDatasetSource:
    def test_uses_the_dataset_values_verbatim(self):
        """The dataset's own pairing must survive, not be re-derived by position."""
        # Deliberately not arange % 8: an injection regression would show up here.
        timestep = torch.tensor([3, 7, 3, 0, 5, 5])
        batch = {"timestep": timestep}

        sigmas = _resolve(batch, DATASET_TIMESTEPS, batch_size=6)

        assert torch.equal(batch["timestep"], timestep)
        torch.testing.assert_close(sigmas, timestep.float() / NUM_VALIDATION_TIMESTEPS, check_dtype=False)

    def test_sigma_is_the_index_over_eight(self):
        batch = {"timestep": torch.arange(NUM_VALIDATION_TIMESTEPS)}

        sigmas = _resolve(batch, DATASET_TIMESTEPS, batch_size=NUM_VALIDATION_TIMESTEPS)

        expected = torch.tensor([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
        torch.testing.assert_close(sigmas, expected, check_dtype=False)

    def test_missing_field_is_an_error_naming_the_cause(self):
        """The whole point of the knob: fail loudly instead of injecting."""
        with pytest.raises(ValueError, match="carries no 'timestep' field"):
            _resolve({}, DATASET_TIMESTEPS, batch_size=64)

    def test_missing_field_error_suggests_the_remedy(self):
        with pytest.raises(ValueError) as excinfo:
            _resolve({}, DATASET_TIMESTEPS, batch_size=64)

        message = str(excinfo.value)
        assert "Re-ingest" in message
        assert "equidistant" in message

    def test_a_batch_spanning_only_two_timesteps_is_accepted(self):
        """The published val set is ordered so a contiguous batch has two sigmas.

        An assertion that every batch covers all eight would reject the real
        data, so confirm no such check crept in.
        """
        batch = {"timestep": torch.tensor([0, 4] * 32)}

        sigmas = _resolve(batch, DATASET_TIMESTEPS, batch_size=64)

        assert set(sigmas.tolist()) == {0.0, 0.5}


class TestEquidistantSource:
    def test_injects_index_modulo_eight(self):
        batch = {}

        _resolve(batch, EQUIDISTANT_TIMESTEPS, batch_size=16)

        assert torch.equal(batch["timestep"], torch.arange(16) % NUM_VALIDATION_TIMESTEPS)

    def test_does_not_override_a_present_dataset_timestep(self):
        """Equidistant is a fallback, not an override."""
        timestep = torch.tensor([6, 1, 6, 1])
        batch = {"timestep": timestep}

        _resolve(batch, EQUIDISTANT_TIMESTEPS, batch_size=4)

        assert torch.equal(batch["timestep"], timestep)

    def test_full_width_batch_covers_every_timestep(self):
        batch = {}

        _resolve(batch, EQUIDISTANT_TIMESTEPS, batch_size=64)

        assert set(batch["timestep"].tolist()) == set(range(NUM_VALIDATION_TIMESTEPS))

    def test_narrow_batch_warns_that_it_misses_timesteps(self, caplog, monkeypatch):
        """micro_batch_size 2 evaluates only t=0 and t=1/8. Say so."""
        monkeypatch.setattr(
            "primus.backends.megatron.training.diffusion.forward_step._warned_uncovered_equidistant",
            False,
        )
        batch = {}

        with caplog.at_level(logging.WARNING):
            _resolve(batch, EQUIDISTANT_TIMESTEPS, batch_size=2)

        assert "not a multiple of 8" in caplog.text
        assert set(batch["timestep"].tolist()) == {0, 1}

    def test_multiple_of_eight_does_not_warn(self, caplog, monkeypatch):
        monkeypatch.setattr(
            "primus.backends.megatron.training.diffusion.forward_step._warned_uncovered_equidistant",
            False,
        )

        with caplog.at_level(logging.WARNING):
            _resolve({}, EQUIDISTANT_TIMESTEPS, batch_size=40)

        assert caplog.text == ""


class TestSourceValidation:
    @pytest.mark.parametrize("source", ["", "Dataset", "arange", None, 8])
    def test_unknown_source_is_rejected(self, source):
        with pytest.raises(ValueError, match="eval_timestep_source must be one of"):
            _resolve({"timestep": torch.zeros(4, dtype=torch.long)}, source, batch_size=4)

    def test_rejection_happens_even_when_the_batch_would_have_worked(self):
        """A typo must not be masked by the batch happening to carry timesteps."""
        batch = {"timestep": torch.arange(8)}

        with pytest.raises(ValueError, match="eval_timestep_source"):
            _resolve(batch, "datset", batch_size=8)

        assert "timesteps" not in batch
