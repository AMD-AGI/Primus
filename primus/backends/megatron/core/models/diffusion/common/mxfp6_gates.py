###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Runtime registry for the MXFP6 fusion gates.

Every gate here used to be an environment variable read at module import. That
worked while they were experiments but is wrong for a shipped recipe, for two
reasons:

* A run that forgets one still runs, still converges and is simply several
  percent slow, because Inductor compiles the unfused graph instead of calling
  the fused kernel. That is easy to misread as a hardware regression rather than
  a missing gate.
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
FP4_SCALE_ROUNDING = ("rceil", "m0", "m1", "m2")
FP4_HADAMARD = ("h32", "h16", "none")
GEMM_LAYOUTS = ("blob", "tilescale")


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
    # Single block, with shared_grad_pack: the gated residual's multiply moves inside the
    # shared Function, whose backward packs dy with Turbo's GateMul prologue, so gate * dy
    # is never materialised. Bit-identical.
    gate_mul_pack: bool = False
    # With gate_mul_pack: fc2's bias gradient from the GateMul pack's column sums instead of a
    # separate reduction of gate * dy. Not bit-identical (summation order, bf16 product).
    gate_mul_pack_bias: bool = False
    # LN-modulate backward split over 16 sequence slices instead of 8 (twice the programs).
    # dx and dshift bit-identical; dscale's fp32 partial sums regroup.
    ln_bwd_ns16: bool = False
    # AdaLN modulation linears write their weight gradient straight into main_grad
    # (the GEMM's output is main_grad) instead of AccumulateGrad + the DDP hook's copy.
    adaln_wgrad_main_grad: bool = False

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
    # convergence validation and is opt-in. Requires mxfp6_weight_format='mxfp4';
    # without A6W4 there is no narrowed weight to narrow a second operand
    # against, and the combination is rejected in BaseDiffusionConfig.
    wgrad_a6w4: bool = False

    # --- A4W4 backward -----------------------------------
    # Run the dgrad / wgrad GEMMs as A4W4 on AITER's f4gemm kernels, both operands MXFP4
    # (Hadamard-rotated like MXFP6), packed in f4gemm's layouts by Primus-Turbo's
    # quantize_mx_* ops. The forward stays MXFP6. dgrad and wgrad are separate gates.
    # Changes numerics; exclusive with A6W4.
    bwd_fp4_dgrad: bool = False
    bwd_fp4_wgrad: bool = False
    # Stochastic rounding for the A4W4 gradient operand: each FP4 code of a
    # gradient pack rounds up or down with probability proportional to distance, so the
    # gradient GEMMs are unbiased. Activations and weights stay round-to-nearest.
    bwd_fp4_sr: bool = False
    # The grouped MLP under the A4W4 backward: format-aware pair packers, grouped
    # A6W6 forward, per-stream A4W4 dgrads. (there is no grouped A4W4 kernel).
    # Bit-identical to the ungrouped A4W4 path. Off: without it the grouped MLP falls back to two MLPs while a bwd_fp4 gate is on.
    bwd_fp4_grouped_mlp: bool = False
    # Operand layout of every MX GEMM, and with it the kernels (all AITER asm):
    #   "blob":      AITER's tile blobs -- A6W6 on the tuned table, A4W4 on f4gemm.
    #   "tilescale": the tilescale layout (aiter.ops.tilescale) -- forward A6W6, the MXFP4 forward and the A4W4
    #                backward on aiter's tilescale kernels where one exists for the shape; other A6W6 shapes stay on
    #                the blob kernels, other MXFP4-forward shapes on the A4W4 tile-blob kernels. The A4W4 backward
    #                packs store the scales per consuming GEMM. Needs both bwd_fp4 gates if either is on.
    gemm_layout: str = "blob"
    # MXFP4 forward for the single blocks' linear2 (attention out-projection + MLP fc2): their
    # activations and weights pack as
    # plain FP4 both ways (fmt 8) and the forward GEMMs run FlyDSL A4W4. Changes forward numerics.
    # Needs both bwd_fp4 gates and gemm_layout "tilescale" (whose backward columns are the layouts its forward
    # packs write).
    fwd_fp4_single_linear2: bool = False
    # MXFP4 forward for the single blocks' MLP fc1 too: same packs / backends / requirements as
    # fwd_fp4_single_linear2. Changes forward numerics.
    fwd_fp4_single_fc1: bool = False
    # MXFP4 forward for the joint blocks' stream MLPs, fc1 and fc2; same requirements.
    fwd_fp4_joint_mlp: bool = False
    # Forward GEMMs as A6W4 on the tilescale layout (aiter `gemm_a6w4_tilescale`): MXFP6 activations times MXFP4
    # weights (H32, RCEIL, round to nearest; K128-blocked codes), wherever aiter has the kernel for the
    # (M, N, K, bias). Columns (the backward's operands) are unchanged. GEMMs on the MXFP4 forward (fwd_fp4_*) keep
    # it. Changes forward numerics. Needs gemm_layout "tilescale", and neither wgrad_a6w4 nor the MXFP4 weight format.
    fwd_a6w4: bool = False
    # Gather the MXFP6 linear weights as their tilescale packs instead of bf16 (distributed optimizer): each rank
    # packs the rows it owns after the optimizer step and the packs are all-gathered; the forward / dgrad read them
    # (the per-step weight packs leave the forward). Lays the DDP buckets out so every shard boundary inside a weight
    # is on a 32-row edge. Needs gemm_layout "tilescale". Bit-identical forward; the column (dgrad) copy's SR seed
    # becomes per (step, weight).
    packed_param_gather: bool = False
    # Selective A4W4: the first / last N transformer blocks keep the A6W6
    # backward while the bwd_fp4 gates are on. Forward hooks switch the three bwd_fp4 gates off
    # around those blocks' forwards; every MXFP6 Function captures the flags at forward time
    # (ctx.b4), so its backward follows the forward's choice. per_block compile only.
    bwd_fp4_a6w6_first: int = 0
    bwd_fp4_a6w6_last: int = 0

    # --- MXFP4 quantization options (the FP4 packs of the A4W4 GEMMs; MXFP6 packs are unaffected) ---
    # E8M0 scale rule per operand class: "rceil" = ceil_pow2(amax / 6), never saturates (the default);
    # "m0" / "m1" / "m2" = Turbo's scale_rounding_mode 0 / 1 / 2, whose scale steps up only at amax mantissa
    # >= 1.75 / 1.5 / 1.8125, so a group's largest values may saturate to 6 in exchange for a finer grid.
    # _grad: gradients. _actw_hp: the FP4 backward copies of activations / weights of layers whose forward is
    # MXFP6. _actw_fp4fwd: activations / weights of the MXFP4-forward layers (fwd_fp4_*), both directions.
    fp4_scale_rounding_grad: str = "rceil"
    fp4_scale_rounding_actw_hp: str = "rceil"
    fp4_scale_rounding_actw_fp4fwd: str = "rceil"
    # Hadamard of the FP4 operands per GEMM, along its contraction axis: "h32" (the default, as MXFP6), "h16"
    # (two 16-point transforms per 32-group) or "none". Per GEMM, not per operand: both operands of a GEMM must
    # be rotated alike, so each setting drives both. _fwd covers only the MXFP4-forward layers.
    fp4_hadamard_fwd: str = "h32"
    fp4_hadamard_dgrad: str = "h32"
    fp4_hadamard_wgrad: str = "h32"
    # 2-D 32x32 block scaling of the weights' FP4 copies: one E8M0 scale per tile, shared by the row and column
    # directions, so one FP4 weight serves forward and dgrad. Needs fp4_hadamard_dgrad "none" (and _fwd "none"
    # with an MXFP4-forward layer): a rotation along one axis would break the shared grid.
    fp4_weight_2d: bool = False
    # Stochastic rounding of the activations' and weights' MXFP4 backward copies (the wgrad / dgrad B operands), on
    # top of bwd_fp4_sr's gradient operand: with both operands of a backward GEMM rounded stochastically and
    # independently the product is unbiased; with one round-to-nearest it carries that operand's rounding bias. The
    # forward rows of MXFP4-forward layers stay round-to-nearest. Needs gemm_layout "tilescale".
    fp4_sr_actw: bool = False
    # The joint blocks' image-stream fc1 forward as a bf16 GEMM (its most sensitive GEMM); its backward stays as the
    # gates above set it, on FP4 / MXFP6 column packs of the bf16 operands. Disables the grouped joint MLP (the two
    # streams' fc1 would differ in precision).
    fwd_bf16_joint_img_fc1: bool = False

    def validate(self) -> None:
        """Reject nonsense values at config time rather than at first use."""
        for name in ("fused_mlp", "fused_qkv"):
            value = getattr(self, name)
            if value not in TRISTATE:
                raise ValueError(f"mxfp6_{name} must be one of {list(TRISTATE)}, got {value!r}.")
        for name in ("grad", "actw_hp", "actw_fp4fwd"):
            value = getattr(self, f"fp4_scale_rounding_{name}")
            if value not in FP4_SCALE_ROUNDING:
                raise ValueError(
                    f"mxfp6_fp4_scale_rounding_{name} must be one of {list(FP4_SCALE_ROUNDING)}, got {value!r}."
                )
        for name in ("fwd", "dgrad", "wgrad"):
            value = getattr(self, f"fp4_hadamard_{name}")
            if value not in FP4_HADAMARD:
                raise ValueError(
                    f"mxfp6_fp4_hadamard_{name} must be one of {list(FP4_HADAMARD)}, got {value!r}."
                )
        if self.gemm_layout not in GEMM_LAYOUTS:
            raise ValueError(f"mxfp6_gemm_layout must be one of {list(GEMM_LAYOUTS)}, got {self.gemm_layout!r}.")
        if self.fp4_sr_actw and self.gemm_layout != "tilescale":
            raise ValueError("mxfp6_fp4_sr_actw needs mxfp6_gemm_layout 'tilescale'.")
        if self.fp4_weight_2d:
            fwd_fp4 = self.fwd_fp4_single_linear2 or self.fwd_fp4_single_fc1 or self.fwd_fp4_joint_mlp
            if self.fp4_hadamard_dgrad != "none" or (fwd_fp4 and self.fp4_hadamard_fwd != "none"):
                raise ValueError(
                    "mxfp6_fp4_weight_2d shares one scale grid between a weight's two directions; it needs "
                    "mxfp6_fp4_hadamard_dgrad 'none' (and mxfp6_fp4_hadamard_fwd 'none' with an MXFP4-forward layer)."
                )

    def fp4_options_set(self) -> bool:
        """Whether any MXFP4 quantization option differs from the default."""
        d = Mxfp6Gates()
        return any(
            getattr(self, f.name) != getattr(d, f.name) for f in fields(self) if f.name.startswith("fp4_")
        )


_GATES = Mxfp6Gates()


def gates() -> Mxfp6Gates:
    """The active gate set.

    Returns the all-default set if :func:`configure` was never called, which is
    what non-diffusion callers and unit tests importing these modules see.
    """
    return _GATES


# Retired config keys -> what replaces them. Setting one is an error, not a silent no-op.
RETIRED_KEYS = {
    "mxfp6_bwd_fp4_backend": "mxfp6_gemm_layout ('aiter' -> 'blob'; 'aiter_fly' / 'flydsl_packed' -> 'tilescale'; "
    "'flydsl' is gone)",
    "mxfp6_fwd_a6w6_fly": "mxfp6_gemm_layout: tilescale",
    "mxfp6_a6w6_backend": "nothing (the GEMMs run on AITER's kernels; the FlyDSL backend is gone)",
    "mxfp6_fwd_a6w4_ts": "mxfp6_fwd_a6w4",
}


def check_retired(source) -> None:
    """Raise if ``source`` (a config or the parsed arguments) sets a retired key."""
    hit = [k for k in RETIRED_KEYS if getattr(source, k, None) is not None]
    if hit:
        raise ValueError(
            "retired MXFP6 config keys: "
            + "; ".join(f"{k} -> use {RETIRED_KEYS[k]}" for k in hit)
        )


def _pin_aiter_backend() -> None:
    """Keep Primus-Turbo's A6W6 GEMMs on AITER whatever its environment says (the FlyDSL backend is not used)."""
    try:
        from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import set_a6w6_backend
    except ImportError:
        return
    set_a6w6_backend("aiter")


def configure(config) -> Mxfp6Gates:
    """Populate the registry from a diffusion config.

    Reads ``mxfp6_<gate>`` off ``config`` for each field, leaving any the config
    does not define at its default. Mutates the module-level registry in place
    so that modules which captured a reference to it still observe the update.
    """
    global _GATES
    check_retired(config)
    resolved = Mxfp6Gates()
    defaults = Mxfp6Gates()
    for f in fields(Mxfp6Gates):
        key = f"mxfp6_{f.name}"
        if hasattr(config, key):
            setattr(resolved, f.name, getattr(config, key))
    resolved.validate()
    _GATES = resolved
    _pin_aiter_backend()

    # Log the RESOLVED gates, not the requested ones. The two can differ: the
    # trainer copies a fixed list of fields onto the model config, so a gate the
    # YAML sets and the argument dump reports as True can still be dropped before
    # it reaches here -- which costs step time and changes nothing visible
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
    _pin_aiter_backend()


_A6W6_BWD_SAVED = []


def _a6w6_bwd_pre(module, args, kwargs=None):
    g = gates()
    _A6W6_BWD_SAVED.append((g.bwd_fp4_dgrad, g.bwd_fp4_wgrad, g.bwd_fp4_sr))
    g.bwd_fp4_dgrad = g.bwd_fp4_wgrad = g.bwd_fp4_sr = False


def _a6w6_bwd_post(module, args, output):
    g = gates()
    g.bwd_fp4_dgrad, g.bwd_fp4_wgrad, g.bwd_fp4_sr = _A6W6_BWD_SAVED.pop()


def install_a6w6_backward_blocks(layers, compile_strategy) -> list:
    """Hook the first ``bwd_fp4_a6w6_first`` and last ``bwd_fp4_a6w6_last`` of ``layers`` so
    they run the A6W6 backward under the A4W4 gates. Returns the hooked indices.

    The hooks sit in ``Module.__call__``, outside a per-block compiled ``forward``; Dynamo
    guards on the gate values, so the hooked blocks compile their own variant. Under a
    strategy that compiles across blocks the hooks would be traced, so that is rejected.
    """
    g = gates()
    first, last = g.bwd_fp4_a6w6_first, g.bwd_fp4_a6w6_last
    if not (first or last) or not (g.bwd_fp4_dgrad or g.bwd_fp4_wgrad):
        return []
    if compile_strategy not in (None, "per_block"):
        raise ValueError(
            f"mxfp6_bwd_fp4_a6w6_first/last need torch_compile strategy per_block, got {compile_strategy!r}"
        )
    n = len(layers)
    idx = [i for i in range(n) if i < first or i >= n - last]
    for i in idx:
        layers[i].register_forward_pre_hook(_a6w6_bwd_pre)
        layers[i].register_forward_hook(_a6w6_bwd_post)
    return idx
