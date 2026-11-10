###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A transformer-only diffusers pipeline for MiniMax-H3 training.

WHY THIS EXISTS:
  AutoModel's fine-tune path, which is the only path that applies LoRA, loads the
  model with ``DiffusionPipeline.from_pretrained`` on a directory with a
  ``model_index.json``. Diffusers integrates MiniMax-H3 as modular blocks only, with
  no ``DiffusionPipeline`` class, so that call cannot load an H3 directory. The
  random-initialization path (``from_config``) can build the transformer, but has no
  LoRA.

  This class closes the gap with the smallest possible pipeline: it holds the
  transformer and nothing else. Training needs no VAE, no text encoder and no
  scheduler, because the data is pre-encoded and AutoModel's flow-matching pipeline
  owns the noise. With it registered, the same ``model_index.json`` + ``transformer/``
  directory loads for full fine-tuning and for LoRA, through AutoModel's own code
  (mesh, dtype, LoRA injection, sharding), and loads the released weights the same way
  if they are ever linked in as ``transformer/``.

HOW IT IS FOUND:
  ``DiffusionPipeline.from_pretrained`` resolves the class named in
  ``model_index.json`` with ``getattr(diffusers, name)``. ``install`` sets that
  attribute, so no diffusers source is edited and nothing is registered for a run
  that does not train MiniMax-H3.

Only the diffusers-importing part is behind a factory, so importing this module
costs nothing for a run of another model.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

PIPELINE_CLASS_NAME = "MiniMaxH3TrainingPipeline"
TRANSFORMER_CLASS_NAME = "MiniMaxH3Transformer3DModel"

_PIPELINE_CLS = None


def get_pipeline_class():
    """The training pipeline class, built on first use."""
    global _PIPELINE_CLS
    if _PIPELINE_CLS is None:
        from diffusers import DiffusionPipeline

        class MiniMaxH3TrainingPipeline(DiffusionPipeline):
            """The MiniMax-H3 transformer, wrapped as a diffusers pipeline so AutoModel can load it."""

            def __init__(self, transformer):
                super().__init__()
                self.register_modules(transformer=transformer)

        MiniMaxH3TrainingPipeline.__name__ = PIPELINE_CLASS_NAME
        MiniMaxH3TrainingPipeline.__qualname__ = PIPELINE_CLASS_NAME
        _PIPELINE_CLS = MiniMaxH3TrainingPipeline
    return _PIPELINE_CLS


def model_index(diffusers_version: str) -> dict:
    """The ``model_index.json`` that makes a directory load as this pipeline."""
    return {
        "_class_name": PIPELINE_CLASS_NAME,
        "_diffusers_version": diffusers_version,
        "transformer": ["diffusers", TRANSFORMER_CLASS_NAME],
    }


def install() -> bool:
    """Make ``model_index.json`` files naming ``MiniMaxH3TrainingPipeline`` loadable.

    Idempotent. Refuses to replace an attribute that is not this class, which would
    mean a diffusers that grew a pipeline of that name, and refuses a diffusers that has
    no MiniMax-H3, naming its version.
    """
    import diffusers

    # Before anything else: a diffusers without H3 would otherwise fail much later, in the
    # recipe's model build, with an error that says nothing about the version.
    if getattr(diffusers, TRANSFORMER_CLASS_NAME, None) is None:
        raise RuntimeError(
            f"this diffusers ({diffusers.__version__}) has no {TRANSFORMER_CLASS_NAME}; MiniMax-H3 needs "
            "diffusers 0.40 or later (the Primus AutoModel prepare hook caps it below 0.41)."
        )

    cls = get_pipeline_class()
    existing = getattr(diffusers, PIPELINE_CLASS_NAME, None)
    if existing is cls:
        return True
    if existing is not None:
        raise RuntimeError(
            f"diffusers already defines {PIPELINE_CLASS_NAME} ({existing!r}); the Primus training "
            "pipeline would shadow it. Rename PIPELINE_CLASS_NAME."
        )
    setattr(diffusers, PIPELINE_CLASS_NAME, cls)
    logger.info(
        "[PrimusMiniMaxH3] registered diffusers.%s for transformer-only training.", PIPELINE_CLASS_NAME
    )
    return True
