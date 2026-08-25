###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
End-to-end parity for Ideogram-4 multi-sample packing: K samples in one row must produce
the same per-sample prediction as K rows of one sample.

WHY THIS IS THE TEST THAT MATTERS:
  Packing has one catastrophic failure mode and it is silent. If two samples sharing a row can
  see each other, every prediction is conditioned on an unrelated caption and an unrelated
  image -- and the loss still descends, because the model simply learns a worse function.
  Nothing raises, no shape is wrong, no NaN appears. The run looks fine and the model is ruined.

  So the assertion here is exact agreement against an independent computation: the SAME samples
  pushed through the SAME weights with ``pack_size=1``, which is the layout that was already in
  production. Anything the folding gets wrong -- an index tensor off by the left-pad offset, a
  timestep broadcast to the wrong span, a segment id shared between neighbours, an image block
  gathered from the wrong offset -- breaks that agreement.

  :class:`TestLeakageDetection` then verifies the test can actually detect the thing it is
  guarding against, by deliberately merging two samples' segments and checking that parity
  BREAKS. Without that, a parity test that accidentally compared two identical unpacked runs
  would pass forever while proving nothing.

WHY A REFERENCE MODEL AND NOT THE REAL TRANSFORMER:
  Packed/unpacked agreement is a property of the ADAPTER's fold and unfold, not of any
  particular attention kernel. What the model has to supply is the contract the real
  ``Ideogram4Transformer2DModel`` documents -- mask attention by ``(seg_i == seg_j)``, accept
  ``timestep`` as ``(B,)`` or ``(B,S)``, treat positions and indicators per token, and reduce
  over nothing but the feature axis. :class:`_ReferenceIdeogram` implements exactly that in a
  few dozen lines, which also means these tests need no GPU, no aiter, and no particular
  diffusers version. Every term in it is there to make the output SENSITIVE to a specific piece
  of the layout, so that getting that piece wrong fails a test instead of passing quietly.
"""

import pytest

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    pytest.skip("Ideogram-4 packing parity tests require torch", allow_module_level=True)

pytest.importorskip(
    "nemo_automodel",
    reason="the Ideogram-4 adapter is built against nemo_automodel's ModelAdapter base",
)

from nemo_automodel.components.flow_matching.adapters.base import (  # noqa: E402
    FlowMatchingContext,
)

from primus.backends.nemo_automodel.models.ideogram4.adapter import (  # noqa: E402
    get_ideogram4_adapter_class,
)
from primus.backends.nemo_automodel.models.ideogram4.packing import (  # noqa: E402
    SLACK_SEGMENT_ID,
)

CHANNELS = 8
FEATURE_DIM = 12
GRID_H, GRID_W = 2, 3


class _ReferenceIdeogram(nn.Module):
    """Minimal stand-in implementing the Ideogram-4 forward contract.

    Every term is chosen to make the output depend on one part of the layout, so that a folding
    bug cannot hide:

      * ``txt_in`` on ``encoder_hidden_states`` -- catches text features scattered to the wrong
        slot, which is the bug the left-pad offset invites.
      * ``ind_emb`` on ``indicator`` -- catches a token labelled text when it is image.
      * a bounded function of ``position_ids`` -- catches positions that do not restart per
        sample. Bounded because the image positions carry a +65536 offset, and a linear term
        would let it swamp everything else.
      * the ``timestep`` modulation, broadcast from ``(B,1)`` or applied per token from
        ``(B,S)`` exactly as the real model does -- catches a per-token timestep that covers
        the wrong span.
      * attention masked by ``(seg_i == seg_j)`` -- the isolation being tested. It mixes tokens,
        so if the mask lets two samples meet, the output changes.

    It reduces over nothing but the feature axis, which is what makes per-sample outputs
    independent of how rows are composed.
    """

    def __init__(self, hidden: int = 16, heads: int = 2, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.heads = heads
        self.img_in = nn.Linear(CHANNELS, hidden)
        self.txt_in = nn.Linear(FEATURE_DIM, hidden)
        self.ind_emb = nn.Embedding(8, hidden)
        self.t_proj = nn.Linear(1, hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, CHANNELS)
        # Deterministic weights so two runs of the same test compare like with like.
        for param in self.parameters():
            with torch.no_grad():
                param.copy_(torch.randn(param.shape, generator=generator) * 0.2)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        position_ids,
        segment_ids,
        indicator,
        return_dict=False,
    ):
        x = self.img_in(hidden_states) + self.txt_in(encoder_hidden_states)
        x = x + self.ind_emb(indicator.clamp(min=0))
        x = x + torch.sin(position_ids.float() * 0.01).sum(-1, keepdim=True)

        # Mirrors the real model: unsqueeze only when timestep is per-sample, so a (B,) input
        # broadcasts over the sequence and a (B,S) input applies per token.
        time = timestep if timestep.dim() > 1 else timestep.unsqueeze(-1)
        x = x * (1.0 + self.t_proj(time.unsqueeze(-1)))

        rows, seq_len, hidden = x.shape
        mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)
        query, key, value = self.qkv(x).chunk(3, dim=-1)

        def heads(z):
            return z.view(rows, seq_len, self.heads, hidden // self.heads).transpose(1, 2)

        attended = F.scaled_dot_product_attention(heads(query), heads(key), heads(value), attn_mask=mask)
        x = x + self.mix(attended.transpose(1, 2).reshape(rows, seq_len, hidden))
        return (self.head(x),)


def _samples(num_samples, text_capacity, lengths, seed=0):
    """Left-padded features + latents, matching the dataloader's output contract."""
    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn(num_samples, CHANNELS, GRID_H, GRID_W, generator=generator)
    features = torch.zeros(num_samples, text_capacity, FEATURE_DIM)
    for i, length in enumerate(lengths):
        # Real tokens occupy the LAST `length` slots -- the left-padding the adapter's index
        # tensors have to account for.
        features[i, text_capacity - length :] = torch.randn(length, FEATURE_DIM, generator=generator)
    sigma = torch.rand(num_samples, generator=generator) * 0.8 + 0.1
    return latents, features, sigma


def _run(adapter, model, latents, features, lengths, sigma, *, pack_size, text_budget, cfg=0.0, seed=0):
    """One adapter round trip. Returns ``(prediction [N,C,H,W], row count)``."""
    batch = {
        "image_latents": latents,
        "llm_features": features,
        "text_lengths": torch.tensor(lengths, dtype=torch.long),
        "data_type": "image",
        "pack_size": pack_size,
        "text_budget": text_budget,
    }
    context = FlowMatchingContext(
        noisy_latents=latents,
        latents=latents,
        timesteps=1.0 - sigma,
        sigma=sigma,
        task_type="t2i",
        data_type="image",
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch=batch,
        cfg_dropout_prob=cfg,
    )
    # The CFG dropout mask is drawn with the global RNG; seeding makes the packed and unpacked
    # runs draw the same one, which is what lets them be compared at all.
    torch.manual_seed(seed)
    inputs = adapter.prepare_inputs(context)
    rows = inputs["hidden_states"].shape[0]
    return adapter.forward(model, inputs), rows


@pytest.fixture(scope="module")
def adapter():
    return get_ideogram4_adapter_class()(in_channels=CHANNELS)


@pytest.fixture(scope="module")
def model():
    return _ReferenceIdeogram().eval()


# (label, lengths, pack_size, text_capacity, packed_budget)
PARITY_CASES = [
    ("k2_ragged", [3, 7, 2, 6], 2, 8, 12),
    ("k2_equal", [5, 5, 5, 5], 2, 8, 12),
    ("k2_extremes", [1, 8, 8, 1], 2, 8, 12),
    ("k4_ragged", [3, 7, 2, 6, 1, 8, 4, 5], 4, 8, 22),
    ("k4_single_row", [2, 3, 4, 1], 4, 8, 12),
    ("k8_wide", [1, 2, 3, 4, 5, 6, 7, 8], 8, 8, 40),
]
PARITY_IDS = [c[0] for c in PARITY_CASES]
PARITY_ARGS = [c[1:] for c in PARITY_CASES]


class TestPackedMatchesUnpacked:
    """The load-bearing assertion: folding rows must not change any sample's prediction."""

    @pytest.mark.parametrize("args", PARITY_ARGS, ids=PARITY_IDS)
    def test_predictions_agree(self, adapter, model, args):
        lengths, pack_size, text_capacity, packed_budget = args
        latents, features, sigma = _samples(len(lengths), text_capacity, lengths)

        packed, packed_rows = _run(
            adapter,
            model,
            latents,
            features,
            lengths,
            sigma,
            pack_size=pack_size,
            text_budget=packed_budget,
        )
        unpacked, unpacked_rows = _run(
            adapter,
            model,
            latents,
            features,
            lengths,
            sigma,
            pack_size=1,
            text_budget=text_capacity + 1,
        )

        # Without this the test could be comparing two unpacked runs and proving nothing.
        assert packed_rows == len(lengths) // pack_size, "the packed run did not actually pack"
        assert unpacked_rows == len(lengths)
        assert packed.shape == unpacked.shape == (len(lengths), CHANNELS, GRID_H, GRID_W), (
            "the adapter must hand back one prediction per SAMPLE regardless of pack_size, or "
            "the pipeline's target and sigma stop lining up"
        )
        torch.testing.assert_close(
            packed,
            unpacked,
            rtol=1e-5,
            atol=1e-5,
            msg=lambda m: (
                f"packed and unpacked predictions diverge for lengths={lengths}, K={pack_size}.\n{m}\n"
                "Something in the fold is wrong: an index tensor off by the left-pad offset, a "
                "timestep covering the wrong span, or samples in a row able to see each other."
            ),
        )

    @pytest.mark.parametrize("args", PARITY_ARGS, ids=PARITY_IDS)
    def test_parity_holds_under_cfg_dropout(self, adapter, model, args):
        """CFG dropout must be drawn per SAMPLE; a per-ROW draw would correlate neighbours."""
        lengths, pack_size, text_capacity, packed_budget = args
        latents, features, sigma = _samples(len(lengths), text_capacity, lengths, seed=3)

        packed, _ = _run(
            adapter,
            model,
            latents,
            features,
            lengths,
            sigma,
            pack_size=pack_size,
            text_budget=packed_budget,
            cfg=0.5,
            seed=11,
        )
        unpacked, _ = _run(
            adapter,
            model,
            latents,
            features,
            lengths,
            sigma,
            pack_size=1,
            text_budget=text_capacity + 1,
            cfg=0.5,
            seed=11,
        )
        torch.testing.assert_close(
            packed,
            unpacked,
            rtol=1e-5,
            atol=1e-5,
            msg=lambda m: (
                f"{m}\nWith the RNG seeded identically, both runs must draw the SAME per-sample "
                "dropout mask. A divergence here means the mask is shaped per row, so one draw "
                "decides several samples' conditioning at once."
            ),
        )

    def test_each_sample_keeps_its_own_timestep(self, adapter, model):
        """A packed row holds samples at different flow-matching times.

        If the per-token timestep were built per row, every sample in a row would train at one
        sigma. Parity against the unpacked run catches that -- but only if the sigmas actually
        differ, which is what this pins down explicitly.
        """
        lengths, pack_size, capacity, budget = [3, 6, 2, 7], 2, 8, 12
        latents, features, _ = _samples(4, capacity, lengths, seed=7)
        spread = torch.tensor([0.05, 0.95, 0.5, 0.15])

        packed, _ = _run(
            adapter,
            model,
            latents,
            features,
            lengths,
            spread,
            pack_size=pack_size,
            text_budget=budget,
        )
        unpacked, _ = _run(
            adapter,
            model,
            latents,
            features,
            lengths,
            spread,
            pack_size=1,
            text_budget=capacity + 1,
        )
        torch.testing.assert_close(packed, unpacked, rtol=1e-5, atol=1e-5)

        # And the sigmas really are far enough apart that sharing one would show.
        uniform, _ = _run(
            adapter,
            model,
            latents,
            features,
            lengths,
            torch.full((4,), 0.5),
            pack_size=pack_size,
            text_budget=budget,
        )
        assert not torch.allclose(packed, uniform, rtol=1e-3, atol=1e-3), (
            "the reference model is insensitive to the timestep, so the parity check above "
            "would not have detected a wrongly broadcast one"
        )

    @pytest.mark.parametrize("pack_size", [2, 4])
    def test_row_count_and_width_do_not_depend_on_the_captions(self, adapter, model, pack_size):
        """torch.compile keys on shapes: two batches of the same size must produce the same ones."""
        capacity, budget = 8, 12 if pack_size == 2 else 22
        shapes = set()
        # These bypass the sampler, so the row sums have to fit the budget by construction: at
        # K=2 consecutive pairs, at K=4 consecutive quads, both under budget - 1.
        for lengths in ([1, 8, 3, 6, 5, 4, 2, 7], [8, 1, 4, 2, 1, 8, 2, 4], [3] * 8):
            latents, features, sigma = _samples(8, capacity, lengths)
            batch = {
                "image_latents": latents,
                "llm_features": features,
                "text_lengths": torch.tensor(lengths),
                "data_type": "image",
                "pack_size": pack_size,
                "text_budget": budget,
            }
            context = FlowMatchingContext(
                noisy_latents=latents,
                latents=latents,
                timesteps=1 - sigma,
                sigma=sigma,
                task_type="t2i",
                data_type="image",
                device=torch.device("cpu"),
                dtype=torch.float32,
                batch=batch,
            )
            inputs = adapter.prepare_inputs(context)
            shapes.add(
                (
                    tuple(inputs["hidden_states"].shape),
                    tuple(inputs["encoder_hidden_states"].shape),
                    tuple(inputs["segment_ids"].shape),
                    tuple(inputs["timestep"].shape),
                )
            )
        assert len(shapes) == 1, f"an input shape varied with the captions: {shapes}"


class TestLeakageDetection:
    """Proof that the parity test above can actually see cross-sample attention."""

    def test_merging_two_segments_breaks_parity(self, adapter, model):
        """Mutation test. Merge the samples in a row into ONE segment and parity must fail.

        This is the whole reason to trust :class:`TestPackedMatchesUnpacked`. If merging the
        segments still agreed with the unpacked run, the parity assertion would be blind to the
        exact corruption packing risks, and it would keep passing while the feature was broken.
        """
        lengths, pack_size, capacity, budget = [3, 6, 2, 7], 2, 8, 12
        latents, features, sigma = _samples(4, capacity, lengths, seed=5)

        unpacked, _ = _run(
            adapter,
            model,
            latents,
            features,
            lengths,
            sigma,
            pack_size=1,
            text_budget=capacity + 1,
        )

        batch = {
            "image_latents": latents,
            "llm_features": features,
            "text_lengths": torch.tensor(lengths),
            "data_type": "image",
            "pack_size": pack_size,
            "text_budget": budget,
        }
        context = FlowMatchingContext(
            noisy_latents=latents,
            latents=latents,
            timesteps=1 - sigma,
            sigma=sigma,
            task_type="t2i",
            data_type="image",
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch=batch,
        )
        inputs = adapter.prepare_inputs(context)

        # Sanity: unmutated, this is the parity case.
        clean = adapter.forward(model, dict(inputs))
        torch.testing.assert_close(clean, unpacked, rtol=1e-5, atol=1e-5)

        # Now collapse every sample in a row onto one segment id, leaving slack alone. That is
        # exactly what a mask bug or a dense-attention fallback would do.
        merged = inputs["segment_ids"].clone()
        merged[merged != SLACK_SEGMENT_ID] = 1
        inputs["segment_ids"] = merged
        leaked = adapter.forward(model, inputs)

        assert not torch.allclose(leaked, unpacked, rtol=1e-4, atol=1e-4), (
            "merging the samples in a row into one segment did NOT change the prediction, so "
            "the parity test cannot detect cross-sample attention -- the reference model is not "
            "mixing tokens, or the segment ids are not reaching its mask."
        )

    def test_segment_ids_isolate_every_sample(self, adapter):
        """Stated directly on the tensor the model masks with, independent of any model."""
        lengths, pack_size, capacity, budget = [3, 6, 2, 7], 2, 8, 12
        latents, features, sigma = _samples(4, capacity, lengths)
        batch = {
            "image_latents": latents,
            "llm_features": features,
            "text_lengths": torch.tensor(lengths),
            "data_type": "image",
            "pack_size": pack_size,
            "text_budget": budget,
        }
        context = FlowMatchingContext(
            noisy_latents=latents,
            latents=latents,
            timesteps=1 - sigma,
            sigma=sigma,
            task_type="t2i",
            data_type="image",
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch=batch,
        )
        inputs = adapter.prepare_inputs(context)
        segment_ids = inputs["segment_ids"]

        for row in range(segment_ids.shape[0]):
            ids = segment_ids[row]
            present = [int(v) for v in ids.unique()]
            assert len(present) == pack_size + 1, (
                f"row {row} has segments {present}; expected {pack_size} samples plus slack. "
                "Two samples sharing an id can attend to each other."
            )
            # Each sample's tokens must be one contiguous run, or the row layout has drifted
            # from what cu_seqlens describes.
            for value in present:
                positions = (ids == value).nonzero().flatten()
                assert torch.equal(
                    positions, torch.arange(int(positions[0]), int(positions[-1]) + 1)
                ), f"segment {value} in row {row} is not contiguous"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
