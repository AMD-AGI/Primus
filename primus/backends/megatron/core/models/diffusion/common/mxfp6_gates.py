###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Runtime registry for the MXFP6 fusion gates.

Every gate here used to be an environment variable read at module import. That
worked while they were experiments but is wrong for a shipped recipe, for two
reasons that both bit this campaign:

* An arm that forgets one still runs, still converges and is simply several
  percent slow, because Inductor compiles the unfused graph instead of calling
  the fused kernel. A set of arms built without five of these read as a 25 ms
  *hardware* regression and was chased through clocks, thermals, GPU tenants and
  the compiler cache before a kernel census located it in the compiled layer.
* Read at import, a gate cannot be described by the config file that is supposed
  to define the run, so the YAML and the measured throughput can diverge
  silently.

So the values live here, are populated once from ``BaseDiffusionConfig`` in its
``__post_init__`` (before any model code runs), and are read through
``gates()`` at the point of use rather than captured at import.

There is deliberately no environment fallback. A gate that can be set from two
places is a gate that will be set from the wrong one.
"""

from dataclasses import dataclass, fields
from typing import Optional

# Accepted values for the tri-state gates. "auto" engages the fusion when the
# module shape allows it and falls back silently otherwise; "on" turns a
# fallback into an error, which is what a submission run wants, so that a
# configuration the fusion cannot reproduce exactly cannot quietly become a
# slower unfused run with the same logged throughput claim.
TRISTATE = ("auto", "on", "off")


@dataclass
class Mxfp6Gates:
    """The MXFP6 fusion gates for one run.

    Defaults are all-off / "auto", matching the pre-migration environment
    defaults, so a config that sets none of these behaves exactly as an
    unset environment did.
    """

    # --- packer / GEMM fusions -------------------------------------------
    # MLP folds its bias-add + GELU into the MXFP6 packer.
    fused_mlp: str = "auto"
    # Feed the attention V operand to the packer strided, skipping a repack.
    strided_v: bool = False
    # Run the two MLP linears as one grouped A6W6 GEMM.
    grouped_mlp: bool = False
    # Project Q, K and V in a single joint GEMM.
    joint_qkv: bool = False
    # Route small reduction gradients (biases, QK-norm weights) into main_grad
    # instead of letting AccumulateGrad materialise them.
    fused_small_grads: bool = False
    # Call the grouped MLP's aiter GEMM through a functional custom op. aiter registers it
    # with mutates_args="unknown", so functionalization clones every live operand -- the
    # saved weight column pair and its scales in the backward, four in-place identity
    # copies per joint block.
    grouped_gemm_functional: bool = False
    # Run a Flux single block's MLP and attention out-projection as one autograd Function
    # (MXFP6MLPProjFunction) whose backward packs their shared output gradient once. Saves
    # a [M, 3072] dual pack and a store of gate*dy per single block.
    shared_grad_pack: bool = False
    # With shared_grad_pack: take fc2's bias gradient from that pack's column sums instead
    # of a separate reduction of gate*dy. Not bit-identical (summation order).
    shared_grad_pack_bias: bool = False
    # Inductor post-grad pass: ln_mod_bwd(a + b) -> one kernel that sums a and b while
    # loading. Removes the single block's separate gradient-accumulation add.
    ln_bwd_fused_sum: bool = False
    # Joint block: both attention out-projections as one Function whose backward writes
    # the two dgrads into halves of one dO, removing autograd's slice-scatter reassembly.
    joint_proj: bool = False

    # --- norm / RoPE fusions ---------------------------------------------
    # Fuse QK-norm and RoPE into one kernel.
    fused_qk_rope: bool = False
    # Fuse the whole QKV norm+RoPE prologue. Tri-state.
    fused_qkv: str = "off"
    # Single-pass backward for the LN-modulate fusion.
    fused_ln_mod_bwd: bool = False
    # Collapse the norm/RoPE autotune space to the measured best compromise
    # instead of retuning per shape.
    norm_rope_pin: bool = False
    # Pre-fusion RoPE slicing order. Kept because it changes numerics, so an
    # A/B against older results needs it; not something a recipe should set.
    rope_slice_legacy: bool = False

    # --- A6W4 ------------------------------------------------------------
    # Pack the wgrad column operand as MXFP4 too, narrowing a second operand on
    # top of A6W4's weight. Worth about as much again as everything A6W4
    # delivers, but the weight-gradient cosine against fp32 lands at 0.9928
    # where A6W4's forward already sits at 0.99293 -- so it needs its own
    # convergence gate and is opt-in. Requires mxfp6_weight_format='mxfp4';
    # without A6W4 there is no narrowed weight to narrow a second operand
    # against, and the combination is rejected in BaseDiffusionConfig.
    wgrad_a6w4: bool = False

    def validate(self) -> None:
        """Reject nonsense values at config time rather than at first use."""
        for name in ("fused_mlp", "fused_qkv"):
            value = getattr(self, name)
            if value not in TRISTATE:
                raise ValueError(f"mxfp6_{name} must be one of {list(TRISTATE)}, got {value!r}.")


_GATES = Mxfp6Gates()


def gates() -> Mxfp6Gates:
    """The active gate set.

    Returns the all-default set if :func:`configure` was never called, which is
    what non-diffusion callers and unit tests importing these modules see.
    """
    return _GATES


def configure(config) -> Mxfp6Gates:
    """Populate the registry from a diffusion config.

    Reads ``mxfp6_<gate>`` off ``config`` for each field, leaving any the config
    does not define at its default. Mutates the module-level registry in place
    so that modules which captured a reference to it still observe the update.
    """
    global _GATES
    resolved = Mxfp6Gates()
    defaults = Mxfp6Gates()
    for f in fields(Mxfp6Gates):
        key = f"mxfp6_{f.name}"
        if hasattr(config, key):
            setattr(resolved, f.name, getattr(config, key))
    resolved.validate()
    _GATES = resolved

    # Log the RESOLVED gates, not the requested ones. The two can differ: the
    # trainer copies a fixed list of fields onto the model config, so a gate the
    # YAML sets and the argument dump reports as True can still be dropped before
    # it reaches here -- which costs ~6% of step time and changes nothing visible
    # otherwise. This line is the one place the truth is recorded, so read it
    # rather than the argument dump when a run comes back mysteriously slow.
    active = {f.name: getattr(resolved, f.name) for f in fields(Mxfp6Gates)}
    changed = {k: v for k, v in active.items() if v != getattr(defaults, k)}
    try:
        from primus.core.utils.module_utils import log_rank_0

        log_rank_0(f"[mxfp6-gates] active: {changed or 'none (all default)'}")
    except ImportError:
        pass
    return _GATES


def reset(gate_set: Optional[Mxfp6Gates] = None) -> None:
    """Restore defaults, or install an explicit set. For tests."""
    global _GATES
    _GATES = gate_set if gate_set is not None else Mxfp6Gates()
