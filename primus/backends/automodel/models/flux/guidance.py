###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Make the FLUX.1 flow-matching adapter honour ``use_guidance_embeds``.

WHY:
  AutoModel's ``FluxAdapter`` accepts ``use_guidance_embeds`` and stores it, but
  ``prepare_inputs`` builds a guidance tensor unconditionally and ``forward`` always
  passes it to the transformer. FLUX.1-schnell is guidance-distilled: its transformer
  is built with ``guidance_embeds=False`` and has no guidance embedder, so step 0
  dies inside diffusers with ``CombinedTimestepTextProjEmbeddings.forward() takes 3
  positional arguments but 4 were given``. ``use_guidance_embeds: false`` is accepted
  by the config and then ignored. (The FLUX.2 adapter already honours it.)

WHAT (NO diffusers / AutoModel fork):
  Wraps ``FluxAdapter.prepare_inputs`` so the guidance entry is ``None`` when
  ``use_guidance_embeds`` is false; diffusers then takes its no-guidance branch.
  Wraps ``FluxAdapter.forward`` to check the setting against the transformer's own
  ``config.guidance_embeds``, and on a mismatch in either direction raises a
  ``ValueError`` that names the config key, instead of the ``TypeError`` from deep
  inside diffusers.

  Unconditional, because with ``use_guidance_embeds: true`` -- the default, and what
  FLUX.1-dev needs -- the adapter behaves exactly as before.
"""
from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[PrimusFluxGuidance]"
_PATCHED_ATTR = "_primus_guidance_patched"
_CONFIG_KEY = "flow_matching.adapter_kwargs.use_guidance_embeds"


def _transformer_has_guidance_embedder(model):
    """``config.guidance_embeds`` of the transformer, or None if it cannot be read.

    FSDP2 shards in place and ``torch.compile`` forwards attribute access to the
    original module, so the config is reachable on the wrapped model as well.
    """
    config = getattr(model, "config", None)
    value = getattr(config, "guidance_embeds", None)
    return None if value is None else bool(value)


def install() -> bool:
    """Patch ``FluxAdapter`` to honour ``use_guidance_embeds``.

    Idempotent: returns True if it is already patched. Returns False, and changes
    nothing, when the adapter is not importable.
    """
    try:
        from nemo_automodel.components.flow_matching.adapters.flux import FluxAdapter
    except ImportError as exc:
        logger.warning("%s FluxAdapter is not importable (%s); not installed", _LOG_PREFIX, exc)
        return False

    if getattr(FluxAdapter, _PATCHED_ATTR, False):
        return True

    original_prepare_inputs = FluxAdapter.prepare_inputs
    original_forward = FluxAdapter.forward

    @functools.wraps(original_prepare_inputs)
    def prepare_inputs(self, context):
        inputs = original_prepare_inputs(self, context)
        if not self.use_guidance_embeds:
            inputs["guidance"] = None
        return inputs

    @functools.wraps(original_forward)
    def forward(self, model, inputs):
        has_embedder = _transformer_has_guidance_embedder(model)
        wants_guidance = bool(self.use_guidance_embeds)
        if has_embedder is not None and has_embedder != wants_guidance:
            raise ValueError(
                f"{_LOG_PREFIX} the transformer has guidance_embeds={has_embedder} but "
                f"{_CONFIG_KEY}={wants_guidance}. Set {_CONFIG_KEY}: "
                f"{'true' if has_embedder else 'false'} for this model "
                f"(false for FLUX.1-schnell, true for FLUX.1-dev)."
            )
        if not getattr(self, "_primus_guidance_logged", False):
            self._primus_guidance_logged = True
            logger.info(
                "%s use_guidance_embeds=%s: %s",
                _LOG_PREFIX,
                wants_guidance,
                "passing a guidance tensor" if wants_guidance else "no guidance tensor is passed",
            )
        return original_forward(self, model, inputs)

    FluxAdapter.prepare_inputs = prepare_inputs
    FluxAdapter.forward = forward
    setattr(FluxAdapter, _PATCHED_ATTR, True)
    logger.info("%s FluxAdapter now honours use_guidance_embeds", _LOG_PREFIX)
    return True
