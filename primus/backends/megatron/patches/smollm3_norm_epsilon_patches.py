###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""SmolLM3: the two RMSNorm epsilon keys must agree.

Training reads ``layernorm_epsilon`` (TransformerConfig's field, the dest of --norm-epsilon); the
native converters and the conversion/parity tooling read ``norm_epsilon``. The SmolLM3 preset sets
both to 1e-6. If a run overrides only one of them (experiment YAML or CLI), training and
conversion would use different epsilons without a sign; this stops the run (SystemExit) with a
clear error before the trainer is built.
"""

from primus.core.patches import PatchContext, get_param, register_patch
from primus.core.utils.module_utils import log_rank_0

KEY = "megatron.smollm3.norm_epsilon_match"


def is_smollm3(ctx) -> bool:
    """A SmolLM3 run: the preset is ``smollm3*.yaml``, or the tokenizer is a SmolLM3 one (a copied
    or renamed preset that extends ``smollm3_3B_Base.yaml`` keeps its ``tokenizer_model``)."""
    model_name = str(getattr(ctx, "model_name", None) or "")
    if model_name.rsplit("/", 1)[-1].lower().startswith("smollm3"):
        return True
    tokenizer = str(get_param(ctx, "tokenizer_model", "") or "")
    return "smollm3" in tokenizer.lower()


def check_epsilons(layernorm_epsilon, norm_epsilon):
    """Raise ValueError when both are set and differ."""
    if layernorm_epsilon is None or norm_epsilon is None:
        return
    if float(layernorm_epsilon) != float(norm_epsilon):
        raise ValueError(
            f"SmolLM3: layernorm_epsilon={layernorm_epsilon} (used by training) and "
            f"norm_epsilon={norm_epsilon} (used by the checkpoint converter) differ; set both "
            "to the checkpoint's rms_norm_eps"
        )


@register_patch(
    KEY,
    backend="megatron",
    phase="build_args",
    description="SmolLM3: layernorm_epsilon and norm_epsilon must agree",
    condition=is_smollm3,
)
def patch_norm_epsilon_match(ctx: PatchContext):
    try:
        check_epsilons(get_param(ctx, "layernorm_epsilon"), get_param(ctx, "norm_epsilon"))
    except ValueError as exc:
        # The patch runner logs and continues on an Exception; a mismatch must stop the run.
        raise SystemExit(f"[{KEY}] {exc}") from exc
    log_rank_0(f"[Patch:{KEY}] layernorm_epsilon and norm_epsilon agree")
