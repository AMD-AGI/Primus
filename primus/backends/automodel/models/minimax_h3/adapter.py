###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax-H3 flow-matching adapter, registered without forking AutoModel.

WHY A WRAPPER AND NOT AN EDIT:
  AutoModel picks a flow-matching adapter by name from a closed dictionary in
  ``flow_matching.pipeline.create_adapter``, and MiniMax-H3 is not in it. ``install``
  wraps that function at runtime so ``flow_matching.adapter_type: minimax_h3``
  resolves here, the way the Ideogram-4 adapter does. The wrapper only adds a
  route, so every other adapter type behaves as before. It patches two namespaces,
  because the recipe binds ``create_adapter`` by value when it is imported.

WHAT THE ADAPTER DOES:
  MiniMax-H3 jointly denoises video and audio in one packed sequence. AutoModel's
  pipeline noises the video latents and compares the prediction with ``noise - x0``;
  this adapter additionally noises the audio latents, packs both streams and the
  text into the layout the transformer reads, and reports the audio flow-matching
  loss through ``auxiliary_losses``, as the LTX-2 adapter does. Both streams train
  from a single forward.

CONVENTIONS, EACH OF WHICH FAILS SILENTLY WHEN WRONG:
  - The transformer predicts the velocity toward the DATA, ``x0 - eps``; AutoModel's
    target is ``eps - x0``. Both predictions are NEGATED, so the sign is stated
    here rather than left to the reader.
  - H3's timestep is ``t = 1 - sigma`` in [0, 1], where t = 1 means clean. AutoModel's
    ``timesteps`` are ``sigma * 1000``, the opposite direction on another scale, so
    the adapter builds ``t`` from ``sigma`` and ignores ``timesteps``.
  - Text rows carry the video timestep (they never reach an output head, and their
    modulation is selected by modality), not t = 1.
  - Micro-batch 1: the structural tensors and the timestep table are shared by the
    batch (see ``layout.py``).

AUDIO NOISE LEVEL:
  MiniMax-H3 samples two schedules at inference, both stepping the same base grid:
  shift 12 for the video, shift 3 for the audio. ``audio_sigma: derived`` keeps that
  pairing in training, by undoing the video shift and applying the audio one.
  ``same`` uses the video sigma for both, as LTX-2 does. MiniMax has not published
  its training schedule, so this is a proxy choice; it does not change the cost of
  a step.

Batch keys consumed:
  video_latents    ``[1, 24, F, H, W]``  consumed by AutoModel's pipeline
  audio_latents    ``[1, 2, 32, L]``     stereo channel, latent channels, length
  text_embeddings  ``[1, T, 5120]``      Qwen3-VL-32B hidden states
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from primus.backends.automodel.models.minimax_h3 import layout as layout_lib

logger = logging.getLogger(__name__)

ADAPTER_TYPE = "minimax_h3"
_AUDIO_SIGMA_MODES = ("derived", "same")


def audio_sigma_from_video(sigma_video: torch.Tensor, video_shift: float, audio_shift: float) -> torch.Tensor:
    """The audio noise level paired with a video noise level at inference.

    Both modalities step one base grid ``u``, each through its own exponential shift
    ``sigma = s * u / (1 + (s - 1) * u)``. Undo the video shift to recover ``u``, then
    apply the audio shift. The map fixes 0 and 1, and is the identity when the two
    shifts are equal.
    """
    u = sigma_video / (video_shift - sigma_video * (video_shift - 1.0))
    return audio_shift * u / (1.0 + (audio_shift - 1.0) * u)


def _base_adapter_cls():
    from nemo_automodel.components.flow_matching.adapters.base import ModelAdapter

    return ModelAdapter


def _build_adapter_class():
    """Build the adapter class on the first call.

    A factory rather than a module-level class so that importing this module never
    requires AutoModel, which keeps it import-safe for tests and for the patch
    machinery, which loads patch modules before anything has decided what a run needs.
    """
    ModelAdapter = _base_adapter_cls()

    class MiniMaxH3Adapter(ModelAdapter):
        """Flow-matching adapter for MiniMax-H3 (text-to-audio-video)."""

        def __init__(
            self,
            audio_loss_weight: float = 1.0,
            video_flow_shift: float = 12.0,
            audio_flow_shift: float = 3.0,
            audio_sigma: str = "derived",
        ):
            if audio_sigma not in _AUDIO_SIGMA_MODES:
                raise ValueError(f"audio_sigma must be one of {_AUDIO_SIGMA_MODES}, got {audio_sigma!r}")
            if video_flow_shift <= 0 or audio_flow_shift <= 0:
                raise ValueError("flow shifts must be positive")
            self.audio_loss_weight = audio_loss_weight
            self.video_flow_shift = video_flow_shift
            self.audio_flow_shift = audio_flow_shift
            self.audio_sigma = audio_sigma

        def _audio_sigma(self, sigma_video: torch.Tensor) -> torch.Tensor:
            if self.audio_sigma == "same":
                return sigma_video
            return audio_sigma_from_video(sigma_video, self.video_flow_shift, self.audio_flow_shift)

        def prepare_inputs(self, context) -> Dict[str, Any]:
            """Noise the audio, pack all three streams, and build the transformer's arguments.

            Returns the transformer's keyword arguments plus private ``_`` keys that
            ``forward`` and ``auxiliary_losses`` read and the model never sees.
            """
            batch = context.batch
            for key in ("audio_latents", "text_embeddings"):
                if key not in batch:
                    raise KeyError(
                        f"MiniMax-H3 batch is missing {key!r}; the synthetic dataloader or the "
                        "preprocessor must provide audio_latents and text_embeddings"
                    )
            c = layout_lib.constants()
            device, dtype = context.device, context.dtype

            noisy_video = context.noisy_latents
            if noisy_video.ndim != 5 or noisy_video.shape[0] != 1:
                raise ValueError(
                    "MiniMax-H3 trains at micro-batch 1: the batch shares one packed layout and one "
                    f"timestep table, so each item needs its own forward. Got latents of shape "
                    f"{tuple(noisy_video.shape)}; set step_scheduler.local_batch_size: 1."
                )
            _, channels, frames, height, width = noisy_video.shape

            audio_x0 = batch["audio_latents"].to(device=device, dtype=torch.float32, non_blocking=True)
            stereo, audio_channels, audio_length = audio_x0.shape[1:]
            if stereo != c.audio_channels or audio_channels != c.audio_latent_channels:
                raise ValueError(
                    f"audio_latents must be [1, {c.audio_channels}, {c.audio_latent_channels}, L], "
                    f"got {tuple(audio_x0.shape)}"
                )
            text = batch["text_embeddings"].to(device=device, dtype=dtype, non_blocking=True)

            sigma_video = context.sigma.to(device=device, dtype=torch.float32).reshape(1)
            sigma_audio = self._audio_sigma(sigma_video)
            audio_noise = torch.randn_like(audio_x0)
            s = sigma_audio.view(1, 1, 1, 1)
            noisy_audio = (1.0 - s) * audio_x0 + s * audio_noise

            layout = layout_lib.get_layout(text.shape[1], frames, height, width, audio_length, device)

            # t = 1 - sigma; table row 0 is the video and text rows, row 1 the audio rows.
            timestep = torch.stack([1.0 - sigma_video[0], 1.0 - sigma_audio[0]])

            inputs: Dict[str, Any] = {
                "hidden_states": layout_lib.patchify_video(noisy_video, c.patch_size).to(dtype),
                "audio_hidden_states": layout_lib.pack_audio(noisy_audio).to(dtype),
                "encoder_hidden_states": text,
                "timestep": timestep,
                **layout.model_kwargs(),
                # Private: not passed to the model.
                "_video_shape": (channels, frames, height, width),
                "_audio_shape": (stereo, audio_length),
                # eps - x0, AutoModel's convention, so it is compared with the negated output.
                "_audio_target": audio_noise - audio_x0,
            }
            return inputs

        def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
            """Run the transformer and return the video prediction as ``eps - x0``.

            Mutates ``inputs``: the audio prediction is stashed under ``_audio_pred``
            for ``auxiliary_losses``.
            """
            c = layout_lib.constants()
            kwargs = {k: v for k, v in inputs.items() if not k.startswith("_")}
            output = model(**kwargs, return_dict=False)
            if isinstance(output, tuple):
                video_rows, audio_rows = output[0], output[1]
            else:
                video_rows, audio_rows = output.sample, output.audio_sample

            stereo, audio_length = inputs["_audio_shape"]
            # The transformer predicts x0 - eps; AutoModel's target is eps - x0.
            inputs["_audio_pred"] = -layout_lib.unpack_audio(audio_rows, stereo, audio_length)
            channels, frames, height, width = inputs["_video_shape"]
            return -layout_lib.unpatchify_video(video_rows, c.patch_size, channels, frames, height, width)

        def auxiliary_losses(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor] | None:
            audio_loss = nn.functional.mse_loss(inputs["_audio_pred"].float(), inputs["_audio_target"])
            return {"audio_loss": self.audio_loss_weight * audio_loss}

    return MiniMaxH3Adapter


# Cached so the class identity is stable across calls, which matters for isinstance
# checks and for anything that keys off the type.
_ADAPTER_CLS = None


def get_adapter_class():
    global _ADAPTER_CLS
    if _ADAPTER_CLS is None:
        _ADAPTER_CLS = _build_adapter_class()
    return _ADAPTER_CLS


def install() -> bool:
    """Make ``adapter_type: minimax_h3`` resolve to this adapter.

    Additive and idempotent: it only adds a route. The recipe's binding is
    re-checked on every call, including after the pipeline is already patched.
    """
    import nemo_automodel.components.flow_matching.pipeline as pipeline

    current = pipeline.create_adapter
    if getattr(current, "_minimax_h3_patched", False):
        create_adapter = current
    else:
        original = current

        def create_adapter(adapter_type: str, **kwargs):
            if adapter_type == ADAPTER_TYPE:
                return get_adapter_class()(**kwargs)
            return original(adapter_type, **kwargs)

        create_adapter._minimax_h3_patched = True
        create_adapter._minimax_h3_original = original
        pipeline.create_adapter = create_adapter
        logger.info("[PrimusMiniMaxH3] registered the '%s' flow-matching adapter.", ADAPTER_TYPE)

    _patch_recipe_binding(create_adapter)
    return True


def _patch_recipe_binding(create_adapter) -> None:
    """Point the recipe's own ``create_adapter`` name at the patched function.

    A recipe imported after the pipeline was patched already holds the patched
    function; one imported before holds the original and is rebound here. Anything
    else bound there is someone else's wrapper, which this does not replace.
    """
    try:
        import nemo_automodel.recipes.diffusion.train as recipe
    except ImportError as exc:
        logger.warning(
            "[PrimusMiniMaxH3] could not import the diffusion recipe (%s), so its create_adapter "
            "was not checked; 'adapter_type: %s' may be reported as unknown.",
            exc,
            ADAPTER_TYPE,
        )
        return

    bound = getattr(recipe, "create_adapter", None)
    if bound is create_adapter or getattr(bound, "_minimax_h3_patched", False):
        return
    if bound is create_adapter._minimax_h3_original:
        recipe.create_adapter = create_adapter
        return
    logger.warning(
        "[PrimusMiniMaxH3] the diffusion recipe's create_adapter is %r, neither AutoModel's nor "
        "the MiniMax-H3 route, so it was left alone and 'adapter_type: %s' will only resolve if "
        "that function forwards to the pipeline's.",
        bound,
        ADAPTER_TYPE,
    )
