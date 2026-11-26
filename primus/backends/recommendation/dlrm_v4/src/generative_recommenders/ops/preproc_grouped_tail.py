# pyre-strict
"""E-M1: ContextualPreprocessor's final LayerNorm(512)s as ONE grouped launch.

Gate: DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL=1, read once when
modules/preprocessors.py is imported. Default ON through the gin binding
apply_env_bootstrap.DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL = True (yambda_5b.gin
item (30)). Absent or "0" in code runs the shipped path, unchanged; opt out per
run with DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL=0.

WHAT CHANGES. With F1+F2 on (fuse_linear_norm=1, fuse_ln_residual=1) the
preprocessor computes

    O = mlp_x(a_x, residual=mlp_a(a_a, residual=mlp_c(seq_emb)))

Each MLP is FusedLinearNormMLP: (Linear, SwishLayerNorm), (Linear, LayerNorm),
and each pair is one _LinearNormFunction. In each MLP's last pair, addmm is
followed by one _weighted_layer_norm_fwd launch. Content has no residual;
additional and action add F2's round-before-add residual. The content output
and the running sum after the additional MLP are each [N, 512] bf16. Both are
written to HBM only for the next launch to read them back. Here the three last
pairs make the same three addmm calls, then ONE
triton_preproc_tail.grouped_ln_residual_fwd launch. That launch reads h_c,
h_a and h_x once and writes O once, so LN-tail traffic drops from 8 KB to
4 KB per row. The first pair of every MLP (Linear -> SwishLayerNorm) is
unchanged: it is the same _LinearNormFunction call that
FusedLinearNormMLP.forward makes.

BITWISE CONTRACT (proven end to end by probes/iter42/tail/t2_e2e.py; the
output and every gradient match with torch.equal):
  * forward, per MLP k: the addmm is _LinearNormFunction.forward's, with the
    same autocast casts, the same contiguity and therefore the same TunableOp
    key (BLAS pin preproc_l2_fwd). The grouped kernel reproduces the pinned
    standalone LayerNorm launches bit for bit (probes/iter42/tail/
    t1_bitwise.py). That proof covers bf16 h with fp32 norm parameters, D=512
    and the autotune pins as shipped (DLRM_HSTU_FP8_AUTOTUNE_PINS=1, gin
    default). Without the pins the shipped launch autotunes its own config, so
    the two paths can differ by a reduction order (ulp-level in rstd/out).
  * backward, per MLP k, in reverse forward order (action, additional,
    content): _LinearNormFunction.backward's body with dy = dO. Each shipped
    pair receives exactly that, because F2's residual gradient is the incoming
    gradient itself. The tensors saved per k are the shipped pair's eight:
    (x_c, lin_w_c, lin_b_c, h, norm_w, norm_b, mean, rstd).
  * E-M2 (sub-switch DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL_BWD; OFF in code,
    on through yambda_5b.gin item (45), "0" = off): the
    three LayerNorm dx-sum backwards that body starts with all read the same
    dy, so they run as ONE launch, ops/triton/triton_preproc_tail_bwd.py, that
    reads each dy row block once (9 -> 7 KB/row). It is bit-identical to the
    three v2 calls: the same grid and strided row walk, per-problem
    arithmetic, num_warps 1, and finalize. Each problem's LayerNorm bias sum is
    still computed separately (no computation is shared between problems).
    The dgrad/wgrad matmuls then run per k as before. Out of place (without
    _BWD_INPLACE) the three dh_k are fresh [N, 512] tensors, live together
    until each k's matmuls free its own (the per-pair loop holds one at a time).
    Opt-in ..._BWD_INPLACE=1 writes dh_k over h_k, which is saved only for
    this backward, so no dh is allocated at all; it donates only when
    autograd will not reuse the node (no retain_graph, no create_graph).
    WHY SHIP11 LEFT IT OFF (Arbor actckpt iter 45; this Triton grouped
    kernel, see "E-M2 on PR #7" below for item (45)). Not a measured step
    regression: at 1n the step got shorter with it (iter-44 bench, E-M1+E-M2:
    GPU span -2.0% on every step, against -0.8% for E-M1 alone). The reason
    is the per-rank kernel evidence. In the step, the DDP all-reduce of the
    41 MB gradient bucket (ncclDevKernel_Generic_1) is launched just before
    this backward and spins during the grouped launch on every rank except
    the last to arrive. There the grouped kernel runs ~1.8x slower: in-model
    the ratio to an N proxy rises from 0.60 to 1.10. The three v2 launches
    are ~0.94 either way.
    On the last rank, which sets the step time, it saves ~1.8 ms; on the
    other seven it costs time they would otherwise spend waiting. So the
    kernel is not consistently faster per rank, the bench's kernel-evidence
    leg rejects it, and its effect where such a collective is on the
    critical path (multi-node) is unmeasured.
    Reproduced off-model at N=2.333M (probes/iter45/verify/t7_contention.py,
    8 ranks, the all-reduce issued from a side stream): 3.2 -> 5.6 ms, while
    the v2 launches go 4.19 -> 4.39 ms.
    Every alternative fails:
      * NUM_STAGES=1 (249 VGPR, occupancy 2) is not bitwise and is just as slow
        under contention;
      * finer contiguous partitions (up to 8192 programs) and s_setprio are
        just as slow;
      * waves_per_eu=2 spills to scratch (23 ms even free).
    Opt in to keep the critical-rank gain.
  * E-M2 on PR #7 (sub-switch DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL_BWD_FLYDSL,
    default ON, consulted only while E-M2 is on; "0" = E-M2 exactly as
    above). The Triton grouped launch is bitwise to the Triton v2 calls,
    not to PR #7's FlyDSL ln_bwd that the per-pair calls run when gin item
    (29) is on. PR #7 owns those calls: when its dispatch would take all
    three (triton_layer_norm.flydsl_ln_bwd_decline_reason, the dispatch's
    own checks, no side effects), E-M2 runs ONE FlyDSL grouped launch
    instead, ops/flydsl/ln_bwd_grouped.py (Arbor P7 M1b, main
    ln_bwd_flydsl_grp3_d512_dxsum_x16_c3 + one grouped finalize): dy is
    loaded once per row (9 -> 7 KB/row) and every output is bit-identical to
    the three PR #7 v2.1 calls (same grid, XCD block order and cache policy,
    per-problem arithmetic; each problem keeps its own dW/dB/dx-sum
    accumulators, so no computation is shared). Same in-place rule
    (_BWD_INPLACE). When PR #7 owns the calls but the grouped kernel cannot
    take them (two problems, a grouped-kernel error), the per-pair loop runs,
    so the result stays PR #7's; when PR #7 owns none of them, the Triton
    grouped launch is considered, under the guard below. One "armed OK:
    [preproc-grouped-tail-bwd-flydsl]" line per process (every rank), or one
    DECLINED warning naming the fallback.
  * Triton grouped launch: bitwise guard (Arbor actckpt iter 51). That kernel
    is bit-identical to item (14)'s v2 launch ONLY, so it runs only when every
    per-pair call it replaces would run v2
    (triton_layer_norm.wln_bwd_dxsum_path, the dispatch's own branch
    helpers, no side effects). Where PR #7 owns a call (also with the
    _BWD_FLYDSL sub-switch "0"), item (28)(b) DLRM_HSTU_FP8_WLN_DXSUM_V3
    would run it, or v1 would (WLN_BWD_V2=0), the per-pair loop runs instead
    and the result stays the per-pair dispatch's (one "[preproc-grouped-tail-
    bwd] grouped LayerNorm backward DECLINED (...)" warning). Under the gin
    defaults (items (28)(b) and (29) on) the Triton grouped launch therefore
    never runs; it runs with DLRM_HSTU_FP8_WLN_BWD_FLYDSL=0 and
    DLRM_HSTU_FP8_WLN_DXSUM_V3=0.

E-M3a (ops/preproc_fp8_l2.py; gate DLRM_HSTU_FP8_PREPROC_L2, default OFF, a
PRECISION CHANGE): with the gate "1" the content and additional MLPs' first pair
emits e4m3 a1 (P2) and their last Linear runs FP8 (_scaled_mm forward/dgrad,
hipBLASLt NN wgrad) inside _GroupedLinearLNTailFP8Function, with e5m2 dh2 from
the LayerNorm(512) dx-sum backward. That backward is ONE stacked launch for
all three MLPs when PR #7 owns the three per-pair calls
(ops/flydsl/ln_bwd_grouped_fp8.py: E-M2's FlyDSL grouped kernel with e5m2 dx
for content/additional and bf16 dx for action, the action outputs bitwise
the per-pair call's), else P3 (an e5m2 emitter) per FP8 MLP and the per-pair
dispatch for the action MLP. The action MLP's forward and every warmup call
run the E-M1 forward above statement for statement. E-M2's grouped backward
(item (45); Triton or FlyDSL) is not used under E-M3a (a WARNING says so). With
the gate unset GroupedTail only asks preproc_fp8_l2.enabled() (a cached bool)
once per forward and runs no E-M3a code (bitwise == the tree without E-M3a);
the module itself is imported at the top of this file.
With the action prefill (DLRM_HSTU_FP8_ACTION_PREFILL, modules/
action_prefill.py) the action MLP arrives through head_prefilled: its plan
slot (preproc_fp8_l2.prefilled_slot) is BF16 and carries the PrefilledLinear,
so the FP8 node keeps ONE per-entry non-tensor argument (eps, plans, *flat)
and x_k stays at ctx.needs_input_grad[2 + 5k].

WHEN IT DECLINES. The shipped path runs and one WARNING is logged on a
non-ROCm (non-HIP) build, or when F1 or
F2 is off, there is no additional MLP, the output dim is not 512, autocast
(cuda, bf16) is off, an MLP is not a FusedLinearNormMLP whose _can_fuse(x)
holds, or a last norm is not an fp32 LayerNorm(512) with the content MLP's eps.
If an input that is only available later (additional, action) fails a check,
the queued last pairs run the shipped way, in shipped order. The rest of the
chain then goes through the caller's shipped _mlp_add, so a late decline is
still exactly the shipped computation. Under fx tracing or torch.compile it
declines silently.

MEMORY. The forward never materialises the two intermediate [N, 512] bf16
sums. The backward holds more: the three h_k (and the y_k saved with them)
are released only when the grouped backward finishes, not pair by pair, and
the three first-pair input gradients are live together. Measured for the
preprocessor alone at N=2.333M (t2_bench.py, E-M1 per-pair backward): forward
peak -2.18 GiB, fwd+bwd peak +2.23 GiB (one [N, 512] bf16). E-M2 (item (45)),
out of place, adds one more [N, 512] bf16 at that point
(+4.45 GiB against the shipped chain in total, measured with the same bench,
Arbor actckpt iter 44 t2_e2e); with the opt-in donation it needs no dh at all
and the preprocessor's fwd+bwd peak returns to the shipped value (12.3298 vs
12.3293 GiB at N=1.5M, iter 44 review). That backward peak falls at the end of
the step's backward. In a full 1n step
(Arbor iter-43 smoke, 276 steps, DLRM_HSTU_FP8_ALLOC_REPORT=1, E-M1 only)
max_memory_allocated was unchanged (213.81 GiB both ways); the one allocator
growth event per run that the shipped path also has came earlier and larger
(+1.06 vs +0.29 GiB reserved).
Under no_grad (eval) the shipped chain frees each pair's h/y right after its
own LayerNorm, while the grouped path holds the three [N, 256] first-pair
outputs and the three h_k at once: about +2 KB/row (~4.4 GiB at N=2.33M),
transient, in a forward that saves no activations.
"""

import ast
import logging
import os
from typing import Callable, List, Optional, Tuple

import torch
from generative_recommenders.ops import preproc_fp8_l2
from generative_recommenders.ops.fused_linear_norm import (
    _LinearNormFunction,
    FusedLinearNormMLP,
)
from generative_recommenders.ops.layer_norm import LayerNorm, SwishLayerNorm
from generative_recommenders.ops.triton import preproc_fp8_snr
from generative_recommenders.ops.triton import triton_ec_dgrad_segreduce as _xs
from generative_recommenders.ops.triton.triton_layer_norm import (
    compute_BLOCK_D,
    flydsl_ln_bwd_decline_reason,
    triton_weighted_layer_norm_bwd_dxsum,
    wln_bwd_dxsum_path,
)
from generative_recommenders.ops.triton.triton_preproc_tail import (
    grouped_ln_residual_fwd,
)
from generative_recommenders.ops.triton.triton_preproc_tail_bwd import (
    grouped_ln_bwd_dxsum,
    grouped_ln_bwd_dxsum_reason,
)
from torch.fx._symbolic_trace import is_fx_tracing

logger: logging.Logger = logging.getLogger(__name__)

_IX_ATTR: str = "_dlrmv4_ec_indexed_l1"  # ops/ec_indexed_l1.IX_ATTR (literal)

ENV_VAR: str = "DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL"
# E-M2 sub-switches, read once per process (forwarded by launch_slurm.sh's
# DLRM_HSTU_FP8_ prefix). _BWD is consulted only when the grouped tail itself
# is on, and _BWD_INPLACE only when _BWD is "1" (set alone it is inert, and a
# one-time WARNING says so).
#  * _BWD (OFF in code, on through yambda_5b.gin item (45), "0" = off; see
#    "WHY SHIP11 LEFT IT OFF" above): the
#    three last-pair LayerNorm(512) dx-sum backwards run as ONE grouped launch
#    that reads dy once (ops/triton/triton_preproc_tail_bwd.py, bitwise to
#    three v2 calls). Off = E-M1's per-pair loop.
#  * _BWD_INPLACE (default OFF, opt in with "1"): that launch writes dh_k
#    over h_k (h_k is saved only for this backward), so the three dh_k cost no
#    memory. Even when opted in it donates only when autograd will not run
#    this node's backward again (no retain_graph, no create_graph). Off: three
#    fresh [N, 512] dh_k, so the in-backward peak is one [N, 512] bf16 above
#    the per-pair loop's (three dh + one first-pair dx against one dh + three
#    dx), at the end of the step's backward.
#  * _BWD_FLYDSL (default ON, "0" = off; consulted only when _BWD is "1"):
#    when PR #7's FlyDSL LayerNorm backward (DLRM_HSTU_FP8_WLN_BWD_FLYDSL)
#    would take all three calls, the grouped launch is the FlyDSL grouped
#    kernel ops/flydsl/ln_bwd_grouped.py (bitwise to the three PR #7 calls)
#    instead of the Triton one (bitwise to Triton v2 only). See the module
#    docstring, "E-M2 on PR #7". The Triton one runs only where the per-pair
#    calls would run v2 (module docstring, "bitwise guard"), so "0" never
#    changes numerics: with PR #7 on it leaves the per-pair loop.
BWD_ENV_VAR: str = "DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL_BWD"
BWD_INPLACE_ENV_VAR: str = "DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL_BWD_INPLACE"
BWD_FLYDSL_ENV_VAR: str = "DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL_BWD_FLYDSL"
BWD_ARMED_PREFIX: str = "armed OK: [preproc-grouped-tail-bwd]"
# "armed OK: [preproc-grouped-tail-bwd-flydsl] kernel=<main kernel> inputs=3
# N=<rows> remap=<R> cache=<load>/<store> inplace=<yes|no> rank=<r>", once
# per process on every rank, the first time the FlyDSL grouped kernel runs.
BWD_FLYDSL_ARMED_PREFIX: str = "armed OK: [preproc-grouped-tail-bwd-flydsl]"
# Grep-able arming literal, emitted once per process (every rank) by the code
# that does the work: "armed OK: [preproc-grouped-tail] inputs=<2|3> N=<rows>
# ln512_pins=<BLOCK_N>x<num_warps>/<..> bitwise=<yes|no..> rank=<r>".
ARMED_PREFIX: str = "armed OK: [preproc-grouped-tail]"
# The only width at which the grouped kernel is proven bitwise (t1_bitwise.py).
TAIL_DIM: int = 512
# The two shipped standalone LayerNorm(512) forward autotune keys the grouped
# kernel replaces (ops/triton/autotune_pin_table.py): content (no residual) and
# the F2 residual chain. The bitwise contract holds when both run num_warps=1,
# which the pin table seeds (BLOCK_N 8 / 4, num_warps 1). With the pins off the
# shipped launches autotune their own config, and the two paths can then
# differ by a reduction order (ulp-level in rstd/out).
_LN512_FWD_KEYS: Tuple[str, str] = (
    "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', "
    "'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32')",
    "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', "
    "'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', "
    "'torch.bfloat16')",
)

_ARMED_LOGGED: bool = False
_DECLINE_WARNED: bool = False
_BWD_ARMED_LOGGED: bool = False
_BWD_DECLINE_WARNED: bool = False
_BWD_ENABLED: Optional[bool] = None
_BWD_INPLACE_ENABLED: Optional[bool] = None
_BWD_FLYDSL_ENABLED: Optional[bool] = None
_BWD_FLYDSL_ARMED_LOGGED: bool = False
_BWD_FLYDSL_DECLINE_WARNED: bool = False
# First host-side error of the FlyDSL grouped kernel (import, compile, launch):
# the grouped FlyDSL path stays off for the process (PR #7's per-call path is
# not touched by it).
_BWD_FLYDSL_BROKEN: Optional[str] = None
_FP8_EM2_WARNED: bool = False

# (mlp, x, residual) -> mlp(x) + residual, i.e. ContextualPreprocessor._mlp_add.
MlpAdd = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def warn_decline_once(reason: str) -> None:
    """One WARNING per process (every rank): the gate is on but the shipped
    path ran."""
    global _DECLINE_WARNED
    if _DECLINE_WARNED:
        return
    _DECLINE_WARNED = True
    logger.warning(
        "[preproc-grouped-tail] %s=1 but the grouped tail DECLINED (%s); "
        "ContextualPreprocessor ran the shipped per-MLP LayerNorm tails rank=%d",
        ENV_VAR,
        reason,
        _rank(),
    )


def ln512_pin_state() -> Tuple[str, bool]:
    """(state, bitwise) of the shipped LN512 fwd autotune entries.

    state is "<BLOCK_N>x<num_warps>" per key, e.g. "8x1/4x1" with the pins on;
    "none" for a key not in the Autotuner cache (pins off and the shipped
    launch never ran); "unknown(...)" if the cache cannot be read. bitwise is
    True only when both keys hold a config the grouped kernel was proven
    against: num_warps=1 and BLOCK_N in {2, 4, 8} (iter-42 t1_bitwise; BLOCK_N
    1 and 16 differ for inputs over 2 GB).
    """
    try:
        from generative_recommenders.ops.triton import triton_layer_norm as tln

        at = tln._weighted_layer_norm_fwd
        # unwrap @triton.heuristics or other decorators over the Autotuner
        for _ in range(4):
            if hasattr(at, "cache") and hasattr(at, "configs"):
                break
            at = at.fn
        cache = at.cache
        states = []
        bitwise = True
        for key_str in _LN512_FWD_KEYS:
            cfg = cache.get(ast.literal_eval(key_str))
            if cfg is None:
                states.append("none")
                bitwise = False
                continue
            block_n = cfg.kwargs.get("BLOCK_N")
            states.append(f"{block_n}x{cfg.num_warps}")
            bitwise = bitwise and cfg.num_warps == 1 and block_n in (2, 4, 8)
        return "/".join(states), bitwise
    except Exception as e:  # pragma: no cover - environment specific
        return f"unknown({type(e).__name__})", False


def _report_armed_once(num_in: int, rows: int) -> None:
    global _ARMED_LOGGED
    if _ARMED_LOGGED:
        return
    _ARMED_LOGGED = True
    # Once per process on every rank, so a multi-node log proves the arm on
    # each rank.
    pins, bitwise = ln512_pin_state()
    logger.info(
        "%s inputs=%d N=%d ln512_pins=%s bitwise=%s rank=%d",
        ARMED_PREFIX,
        num_in,
        rows,
        pins,
        "yes" if bitwise else "no (ulp-level vs autotuned shipped)",
        _rank(),
    )


def _grouped_bwd_enabled() -> bool:
    global _BWD_ENABLED
    if _BWD_ENABLED is None:
        _BWD_ENABLED = os.environ.get(BWD_ENV_VAR, "0") == "1"
        if not _BWD_ENABLED:
            for name in (BWD_INPLACE_ENV_VAR, BWD_FLYDSL_ENV_VAR):
                if os.environ.get(name, "0") == "1":
                    logger.warning(
                        "[preproc-grouped-tail-bwd] %s=1 is INERT: the grouped "
                        "LayerNorm backward is off (opt in with %s=1) rank=%d",
                        name,
                        BWD_ENV_VAR,
                        _rank(),
                    )
    return _BWD_ENABLED


def _grouped_bwd_flydsl_enabled() -> bool:
    """E-M2's FlyDSL sub-switch (default ON, "0" = off), read once."""
    global _BWD_FLYDSL_ENABLED
    if _BWD_FLYDSL_ENABLED is None:
        _BWD_FLYDSL_ENABLED = os.environ.get(BWD_FLYDSL_ENV_VAR, "1") != "0"
    return _BWD_FLYDSL_ENABLED


def flydsl_grouped_bwd_wanted() -> bool:
    """True when this process may run the FlyDSL grouped LayerNorm backward:
    E-M1 (DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL=1), E-M2 (..._BWD=1), its FlyDSL
    sub-switch and PR #7 (DLRM_HSTU_FP8_WLN_BWD_FLYDSL) all on. For the
    startup prebuild (dlrm_v4/train/blas_warmup.py)."""
    return (
        os.environ.get(ENV_VAR, "0") == "1"
        and _grouped_bwd_enabled()
        and _grouped_bwd_flydsl_enabled()
        and os.environ.get("DLRM_HSTU_FP8_WLN_BWD_FLYDSL", "0") != "0"
    )


def _disable_bwd_flydsl(where: str, error: BaseException) -> None:
    """Latch the FlyDSL grouped path off for the process (first error wins),
    with one loud WARNING; PR #7's per-call path is unaffected."""
    global _BWD_FLYDSL_BROKEN
    if _BWD_FLYDSL_BROKEN is not None:
        return
    lines = str(error).strip().splitlines()
    _BWD_FLYDSL_BROKEN = (
        f"{where}: {type(error).__name__}: {lines[0][:200] if lines else ''}"
    )
    logger.warning(
        "[preproc-grouped-tail-bwd-flydsl] DISABLED (%s); the per-pair PR #7 "
        "LayerNorm backward runs instead rank=%d",
        _BWD_FLYDSL_BROKEN,
        _rank(),
        exc_info=error,
    )


def prebuild_flydsl_grouped(device: torch.device) -> str:
    """Startup prebuild of the FlyDSL grouped kernels (main + finalize) on
    synthetic zero/one tensors; an error latches only this path off."""
    try:
        from generative_recommenders.ops.flydsl import ln_bwd_grouped

        ln_bwd_grouped.prebuild_grouped(device, M=1024)
        return "grp3=1"
    except Exception as error:
        _disable_bwd_flydsl("startup prebuild", error)
        return "grp3=error"


def _grouped_bwd_inplace_ok() -> bool:
    """Donate h_k's storage to dh_k. Opt-in (default off): only with the
    switch set to "1", outside a create_graph backward, and when the running
    graph task does not keep the graph (retain_graph=False), so this node's
    backward and its saved h_k are never used again. AOTAutograd asks the
    same keep_graph question before it frees saved tensors early."""
    global _BWD_INPLACE_ENABLED
    if _BWD_INPLACE_ENABLED is None:
        _BWD_INPLACE_ENABLED = os.environ.get(BWD_INPLACE_ENV_VAR, "0") == "1"
    if not _BWD_INPLACE_ENABLED or torch.is_grad_enabled():
        return False
    try:
        return not torch._C._autograd._get_current_graph_task_keep_graph()
    except Exception:  # pragma: no cover - API absent: never donate
        return False


def _report_bwd_once(num_in: int, rows: int, inplace: bool) -> None:
    global _BWD_ARMED_LOGGED
    if _BWD_ARMED_LOGGED:
        return
    _BWD_ARMED_LOGGED = True
    logger.info(
        "%s inputs=%d N=%d inplace=%s rank=%d",
        BWD_ARMED_PREFIX,
        num_in,
        rows,
        "yes" if inplace else "no",
        _rank(),
    )


def _warn_bwd_decline_once(reason: str) -> None:
    global _BWD_DECLINE_WARNED
    if _BWD_DECLINE_WARNED:
        return
    _BWD_DECLINE_WARNED = True
    logger.warning(
        "[preproc-grouped-tail-bwd] grouped LayerNorm backward DECLINED (%s); "
        "the per-pair LayerNorm backward ran rank=%d",
        reason,
        _rank(),
    )


def _report_bwd_flydsl_once(num_in: int, rows: int, inplace: bool) -> None:
    global _BWD_FLYDSL_ARMED_LOGGED
    if _BWD_FLYDSL_ARMED_LOGGED:
        return
    _BWD_FLYDSL_ARMED_LOGGED = True
    from generative_recommenders.ops.flydsl import ln_bwd_grouped

    remap, c_ld, c_st = ln_bwd_grouped.grouped_kernel_options()
    logger.info(
        "%s kernel=%s inputs=%d N=%d remap=%d cache=%d/%d inplace=%s rank=%d",
        BWD_FLYDSL_ARMED_PREFIX,
        ln_bwd_grouped.grouped_kernel_name(),
        num_in,
        rows,
        remap,
        c_ld,
        c_st,
        "yes" if inplace else "no",
        _rank(),
    )


def _warn_bwd_flydsl_decline_once(reason: str, fallback: str) -> None:
    global _BWD_FLYDSL_DECLINE_WARNED
    if _BWD_FLYDSL_DECLINE_WARNED:
        return
    _BWD_FLYDSL_DECLINE_WARNED = True
    logger.warning(
        "[preproc-grouped-tail-bwd-flydsl] %s is on but the FlyDSL grouped "
        "LayerNorm backward DECLINED (%s); %s ran rank=%d",
        BWD_FLYDSL_ENV_VAR,
        reason,
        fallback,
        _rank(),
    )


def _flydsl_grouped_reason(
    dy: torch.Tensor,
    hs: List[torch.Tensor],
    norm_ws: List[torch.Tensor],
    norm_bs: List[torch.Tensor],
    means: List[torch.Tensor],
    rstds: List[torch.Tensor],
) -> Tuple[bool, Optional[str]]:
    """(PR #7 owns some of these calls, why the FlyDSL grouped kernel does
    not take them or None). PR #7 owns call k when its dispatch would run it
    on FlyDSL (same checks, no side effects); the grouped kernel takes the
    calls only when PR #7 owns all of them."""
    whys = [
        flydsl_ln_bwd_decline_reason(
            hs[k], dy, norm_ws[k], norm_bs[k], means[k], rstds[k], sum_dx=True
        )
        for k in range(len(hs))
    ]
    owned = [w is None for w in whys]
    if not any(owned):
        return False, f"PR #7 does not take these calls: {whys[0]}"
    if not all(owned):
        return True, "PR #7 takes only problems " + ",".join(
            str(k) for k in range(len(hs)) if owned[k]
        )
    if _BWD_FLYDSL_BROKEN is not None:
        return True, f"latched off: {_BWD_FLYDSL_BROKEN}"
    try:
        from generative_recommenders.ops.flydsl import ln_bwd_grouped

        return True, ln_bwd_grouped.grouped_reason(dy, hs, norm_ws, means, rstds)
    except Exception as error:  # import error: latch, run the per-pair path
        _disable_bwd_flydsl("import", error)
        return True, f"latched off: {_BWD_FLYDSL_BROKEN}"


def _triton_grouped_guard_reason(
    dy: torch.Tensor,
    hs: List[torch.Tensor],
    norm_ws: List[torch.Tensor],
    norm_bs: List[torch.Tensor],
    means: List[torch.Tensor],
    rstds: List[torch.Tensor],
    block_ds: List[int],
) -> Optional[str]:
    """Bitwise guard of the Triton grouped launch (module docstring): None
    when every per-pair call it replaces would run item (14)'s v2 launch, the
    only path that kernel is bit-identical to; else why not (stable text,
    never N). Same checks as the per-pair dispatch, no side effects."""
    paths = [
        wln_bwd_dxsum_path(
            hs[k], dy, norm_ws[k], norm_bs[k], means[k], rstds[k], block_ds[k],
            is_swish=False,
        )
        for k in range(len(hs))
    ]
    if all(p == "v2" for p in paths):
        return None
    return (
        "the per-pair LayerNorm backward would run "
        + "/".join(sorted(set(paths)))
        + ", and the Triton grouped kernel is bitwise to item (14) v2 only"
    )


def _flydsl_grouped_bwd(
    dy: torch.Tensor,
    hs: List[torch.Tensor],
    norm_ws: List[torch.Tensor],
    norm_bs: List[torch.Tensor],
    means: List[torch.Tensor],
    rstds: List[torch.Tensor],
    inplace: bool,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """The FlyDSL grouped kernel's (dh_k, dnorm_w_k, dnorm_b_k, dh_colsum_k),
    or None after a host-side error that left every h_k intact (latched off;
    the caller runs the per-pair loop)."""
    state = {"main_launched": False}
    try:
        from generative_recommenders.ops.flydsl import ln_bwd_grouped

        return ln_bwd_grouped.ln_bwd_grouped3(
            dy,
            hs,
            norm_ws,
            norm_bs,
            means,
            rstds,
            out={"dx": list(hs)} if inplace else None,
            state=state,
        )
    except Exception as error:
        if inplace and state["main_launched"]:
            # dh_k already overwrote the saved h_k: no input left to rerun from
            raise RuntimeError(
                "[preproc-grouped-tail-bwd-flydsl] the grouped FlyDSL LayerNorm "
                "backward failed after its main kernel wrote dh over the saved "
                f"inputs ({BWD_INPLACE_ENV_VAR}=1); no fallback is possible"
            ) from error
        _disable_bwd_flydsl("dispatch", error)
        return None


def mlp_decline_reason(
    mlp: torch.nn.Module, x: torch.Tensor, eps: Optional[float]
) -> Optional[str]:
    """None iff `mlp(x)`'s LAST (Linear, LayerNorm) pair may join the group.

    `eps` is the group's eps (the content MLP's), or None for the first MLP.
    """
    if not isinstance(mlp, FusedLinearNormMLP):
        return f"{type(mlp).__name__} is not FusedLinearNormMLP"
    if isinstance(x, torch.fx.Proxy):
        return "fx proxy input"
    if not mlp._can_fuse(x):
        # Same conditions FusedLinearNormMLP.forward falls back on: F1 off for
        # this container, non-CUDA or non-2-D input, an unexpected child, or a
        # norm on a non-Triton ladder rung.
        return (
            f"FusedLinearNormMLP._can_fuse declined (enabled={mlp._f1_enabled} "
            f"cuda={x.is_cuda} dim={x.dim()})"
        )
    return _last_pair_reason(mlp, eps)


def _last_pair_reason(mlp: FusedLinearNormMLP, eps: Optional[float]) -> Optional[str]:
    """mlp_decline_reason's checks on the LAST (Linear, LayerNorm) pair only
    (shared with GroupedTail.head_prefilled, whose head already ran)."""
    lin, norm = mlp._pairs()[-1]
    if not isinstance(norm, LayerNorm):
        return f"last norm is {type(norm).__name__}, not LayerNorm"
    if (
        lin.out_features != TAIL_DIM
        or tuple(norm.weight.shape) != (TAIL_DIM,)
        or tuple(norm.bias.shape) != (TAIL_DIM,)
    ):
        return f"last LayerNorm is not {TAIL_DIM} wide"
    if norm.weight.dtype != torch.float32 or norm.bias.dtype != torch.float32:
        return (
            f"last LayerNorm params are {norm.weight.dtype}/{norm.bias.dtype}, "
            "not fp32"
        )
    if eps is not None and norm._eps != eps:
        return f"last LayerNorm eps {norm._eps} != content MLP's {eps}"
    return None


def _run_head(mlp: FusedLinearNormMLP, x: torch.Tensor) -> torch.Tensor:
    """Every pair of `mlp` but the last, exactly as FusedLinearNormMLP.forward
    runs them (residual only ever goes to the last pair)."""
    # Indexed-X (ops/ec_indexed_l1.py, default off): an EC placeholder is rebuilt into the exact member output
    # (identity backward to the placeholder) BEFORE E-X2b's claim below; every pair then runs unchanged.
    if isinstance(x, torch.Tensor) and getattr(x, _IX_ATTR, None) is not None:
        from generative_recommenders.ops import ec_indexed_l1 as _ixp

        x = _ixp.rebuild(x, "_run_head")
    if not mlp._f1_reported:
        # The shipped forward emits F1's "armed OK: fuse_linear_norm FUSED"
        # line from here; keep it, since every pair of this container is
        # still fused.
        mlp._report_once(True, x)
    for i, (lin, norm) in enumerate(mlp._pairs()[:-1]):
        # E-X2b: pair 0 may take x's EC dh port (None, None when off). A
        # _complete_shipped re-run finds the port already claimed and declines.
        rec, port = _xs.claim(x, lin.weight) if i == 0 else (None, None)
        x = _LinearNormFunction.apply(
            x,
            lin.weight,
            lin.bias,
            norm.weight,
            norm.bias,
            norm._eps,
            isinstance(norm, SwishLayerNorm),
            None,
            None,  # prefilled (65): never here
            rec,
            port,
        )
    return x


class _GroupedLinearLNTailFunction(torch.autograd.Function):
    """O = LN_{K-1}(x_{K-1} W^T + b) (+) ... (+) LN_0(x_0 W^T + b), K in {2, 3}.

    Forward args: eps, then `prefilled` (None, or per MLP k a
    modules/action_prefill.py PrefilledLinear or None: that k's casts + addmm
    issued earlier, DLRM_HSTU_FP8_ACTION_PREFILL), then per MLP k the five
    tensors (x_k, lin_w_k, lin_b_k, norm_w_k, norm_b_k), where x_k is the
    input of that MLP's LAST Linear (the SwishLayerNorm output). Because eps
    and prefilled come first, x_k's ctx.needs_input_grad slot is 2 + 5k.
    (+) is F2's
    round-before-add. Gradients come back in the dtypes of the tensors passed
    in, as _LinearNormFunction returns them.
    """

    @staticmethod
    # pyre-ignore[14]
    def forward(  # pyre-ignore[3]
        ctx,  # pyre-ignore[2]
        eps: float,
        # pyre-ignore[2]: None, or per MLP k a modules/action_prefill.py
        # PrefilledLinear (or None) holding that k's casts + addmm issued
        # earlier (GroupedTail.head_prefilled)
        prefilled,
        *flat: torch.Tensor,
    ) -> torch.Tensor:
        num_in = len(flat) // 5
        if num_in not in (2, 3) or len(flat) != 5 * num_in:
            raise ValueError(
                f"grouped LN tail: need 2 or 3 x 5 tensors, got {len(flat)}"
            )
        if prefilled is not None and len(prefilled) != num_in:
            raise ValueError(
                f"grouped LN tail: {len(prefilled)} prefilled entries for "
                f"{num_in} MLPs"
            )
        saved: List[torch.Tensor] = []
        hs: List[torch.Tensor] = []
        norm_ws: List[torch.Tensor] = []
        norm_bs: List[torch.Tensor] = []
        grad_dtypes: List[Tuple[torch.dtype, torch.dtype, torch.dtype]] = []
        block_ds: List[int] = []
        for k in range(num_in):
            x, lin_w, lin_b, norm_w, norm_b = flat[5 * k : 5 * k + 5]
            # == _LinearNormFunction.forward, statement for statement (see the
            # AUTOCAST note there): cast HERE so the matmul runs at the shipped
            # precision and backward reuses the same cast tensors.
            x = x.contiguous()
            # DLRM_HSTU_FP8_ACTION_PREFILL: this k's casts + addmm may already
            # have run (fused_linear_norm.linear_fwd, the same statements on the
            # same operands under the same autocast state) inside the forward
            # all-to-all window; reuse them. take() returns None on any
            # mismatch -> computed here as before.
            pf = None if prefilled is None else prefilled[k]
            got = None if pf is None else pf.take(x, lin_w, lin_b)
            if got is not None:
                x_c, lin_w_c, lin_b_c, h = got
            else:
                dev = x.device.type
                if torch.is_autocast_enabled(dev):
                    amp_dtype = torch.get_autocast_dtype(dev)
                    x_c = x.to(amp_dtype)
                    lin_w_c = lin_w.to(amp_dtype)
                    lin_b_c = lin_b.to(amp_dtype)
                else:
                    x_c, lin_w_c, lin_b_c = x, lin_w, lin_b
                h = torch.addmm(lin_b_c, x_c, lin_w_c.t())
            if k > 0 and (h.dtype != hs[0].dtype or h.shape != hs[0].shape):
                # The shipped chain raises here too (_LinearNormFunction's
                # residual dtype/shape check): it cannot type-promote.
                raise RuntimeError(
                    f"grouped LN tail: MLP {k} norm input {tuple(h.shape)}/"
                    f"{h.dtype} != MLP 0's {tuple(hs[0].shape)}/{hs[0].dtype}"
                )
            if preproc_fp8_snr.ENABLED:
                # Default-off FP8 numerics probe (Arbor E-M3); reads only.
                preproc_fp8_snr.on_forward(
                    ctx, k, preproc_fp8_snr.tail_site(k), x_c, lin_w_c, lin_b_c, h
                )
            hs.append(h)
            norm_ws.append(norm_w)
            norm_bs.append(norm_b)
            saved += [x_c, lin_w_c, lin_b_c, h, norm_w, norm_b]
            grad_dtypes.append((x.dtype, lin_w.dtype, lin_b.dtype))
            block_ds.append(compute_BLOCK_D(h))
        out, means, rstds = grouped_ln_residual_fwd(hs, norm_ws, norm_bs, eps)
        packed: List[torch.Tensor] = []
        for k in range(num_in):
            packed += saved[6 * k : 6 * k + 6] + [means[k], rstds[k]]
        ctx.save_for_backward(*packed)
        ctx.num_in = num_in
        ctx.grad_dtypes = grad_dtypes
        ctx.block_ds = block_ds
        ctx.eps = eps
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(  # pyre-ignore[3]
        ctx,  # pyre-ignore[2]
        dy: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        num_in: int = ctx.num_in
        saved = ctx.saved_tensors
        needs = ctx.needs_input_grad
        # Every shipped pair receives the output gradient itself: action's is
        # dO, and F2 hands it on unchanged as each residual's gradient (a
        # same-dtype .to() is the identity). Each pair then calls .contiguous()
        # on it, which yields the same values every time.
        dy = dy.contiguous()
        grads: List[Optional[torch.Tensor]] = [None] * (5 * num_in)

        def _match(
            g: Optional[torch.Tensor], dt: torch.dtype
        ) -> Optional[torch.Tensor]:
            return None if g is None else g.to(dt)

        # E-M2: the three last-pair LayerNorm dx-sum backwards as ONE launch
        # that reads dy once; bit-identical to the per-pair calls below.
        grouped: Optional[
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None
        if _grouped_bwd_enabled():
            hs = [saved[8 * k + 3] for k in range(num_in)]
            norm_ws = [saved[8 * k + 4] for k in range(num_in)]
            norm_bs = [saved[8 * k + 5] for k in range(num_in)]
            means = [saved[8 * k + 6] for k in range(num_in)]
            rstds = [saved[8 * k + 7] for k in range(num_in)]
            # E-M2 on PR #7 (module docstring): PR #7 owns these calls when its
            # dispatch would take them; then only its FlyDSL grouped kernel or
            # the per-pair loop (PR #7 per call) may run them.
            run_triton = True
            fly_decline: Optional[str] = None
            if _grouped_bwd_flydsl_enabled():
                pr7_owns, fly_why = _flydsl_grouped_reason(
                    dy, hs, norm_ws, norm_bs, means, rstds
                )
                if fly_why is None:
                    inplace = _grouped_bwd_inplace_ok()
                    grouped = _flydsl_grouped_bwd(
                        dy, hs, norm_ws, norm_bs, means, rstds, inplace
                    )
                    if grouped is not None:
                        _report_bwd_flydsl_once(num_in, dy.shape[0], inplace)
                    else:
                        fly_why = f"latched off: {_BWD_FLYDSL_BROKEN}"
                if grouped is not None:
                    run_triton = False
                else:
                    # declined: where PR #7 owns (some of) the calls, the
                    # per-pair loop keeps them on PR #7; otherwise E-M2's
                    # Triton grouped path is considered (bitwise guard)
                    run_triton = not pr7_owns
                    fly_decline = str(fly_why)
            if run_triton:
                # Bitwise guard (module docstring): only where every per-pair
                # call would run item (14) v2; else the per-pair loop runs.
                reason = _triton_grouped_guard_reason(
                    dy, hs, norm_ws, norm_bs, means, rstds, ctx.block_ds
                )
                if reason is None:
                    reason = grouped_ln_bwd_dxsum_reason(
                        dy, hs, norm_ws, means, rstds, ctx.block_ds
                    )
                if reason is None:
                    inplace = _grouped_bwd_inplace_ok()
                    grouped = grouped_ln_bwd_dxsum(
                        dy, hs, norm_ws, norm_bs, means, rstds, inplace=inplace
                    )
                    _report_bwd_once(num_in, dy.shape[0], inplace)
                else:
                    _warn_bwd_decline_once(reason)
            if fly_decline is not None:
                _warn_bwd_flydsl_decline_once(
                    fly_decline,
                    "E-M2's Triton grouped path"
                    if grouped is not None
                    else "the per-pair LayerNorm backward",
                )
            del hs, norm_ws, norm_bs, means, rstds

        # Reverse forward order (action, additional, content), the order in
        # which autograd reaches the shipped pairs.
        for k in reversed(range(num_in)):
            x, lin_w, lin_b, h, norm_w, norm_b, mean, rstd = saved[8 * k : 8 * k + 8]
            if grouped is not None:
                dh, dnorm_w, dnorm_b, dh_colsum = grouped[k]
                # Drop the list's reference so this k's dh (fresh when not
                # donated) is freed after its matmuls, as in the per-pair path.
                grouped[k] = None  # pyre-ignore[6]
            else:
                # == _LinearNormFunction.backward with is_swish=False.
                (
                    dh,
                    dnorm_w,
                    dnorm_b,
                    dh_colsum,
                ) = triton_weighted_layer_norm_bwd_dxsum(
                    dy=dy,
                    x=h,
                    weight=norm_w,
                    bias=norm_b,
                    mean=mean,
                    rstd=rstd,
                    eps=ctx.eps,
                    BLOCK_D=ctx.block_ds[k],
                    is_swish=False,
                )
            # needs_input_grad covers forward's (eps, prefilled, *flat): x_k is
            # at 2 + 5k (eps and the non-tensor `prefilled` come first).
            n_x, n_w, n_b = needs[2 + 5 * k], needs[3 + 5 * k], needs[4 + 5 * k]
            dx = dh @ lin_w if n_x else None
            dlin_w = dh.t() @ x if n_w else None
            dlin_b = dh_colsum if n_b else None
            if preproc_fp8_snr.ENABLED:
                preproc_fp8_snr.on_backward(ctx, k, dh, lin_w, x, dx, dlin_w)
            x_dt, w_dt, b_dt = ctx.grad_dtypes[k]
            grads[5 * k : 5 * k + 5] = [
                _match(dx, x_dt),
                _match(dlin_w, w_dt),
                _match(dlin_b, b_dt),
                _match(dnorm_w, norm_w.dtype),
                _match(dnorm_b, norm_b.dtype),
            ]
            # Drop this k's [N, 512] dh before the next k allocates its own:
            # otherwise two are live at once (+N x 1 KB at the backward peak).
            del dh, dx, dlin_w, dlin_b, dnorm_w, dnorm_b, dh_colsum
        # eps, prefilled (not tensors), then the five gradients per MLP.
        return (None, None, *grads)


class GroupedTail:
    """One ContextualPreprocessor forward's MLP chain, with the last pairs grouped.

    head(mlp, x) runs every pair of `mlp` except the last and queues
    (mlp, last-pair input). finish() runs all queued last pairs as ONE
    _GroupedLinearLNTailFunction and returns the chain's output. If a later
    input fails the checks, the queued last pairs run the shipped way (see
    _complete_shipped) and the rest of the chain goes through `mlp_add`.
    """

    def __init__(
        self, mlp_add: MlpAdd, eps: float, rows: int, device: torch.device
    ) -> None:
        self._mlp_add: MlpAdd = mlp_add
        self._eps: float = eps
        self._rows: int = rows
        self._device: torch.device = device
        self._mlps: List[FusedLinearNormMLP] = []
        self._ys: List[torch.Tensor] = []
        # Per queued MLP: a modules/action_prefill.py PrefilledLinear holding
        # its last pair's casts + addmm issued earlier (head_prefilled), or
        # None (head()). Consumed once, inside the node that runs that pair.
        self._pfs: List[Optional[object]] = []
        # Set once the chain has declined; from then on it is the shipped chain.
        self._acc: Optional[torch.Tensor] = None
        self._declined: bool = False
        # E-M3a (ops/preproc_fp8_l2.py, DLRM_HSTU_FP8_PREPROC_L2, default OFF):
        # one plan slot and the head input per queued MLP (a prefilled head's
        # input is None: its slot is BF16). Off: both stay empty and nothing
        # below changes.
        self._fp8: bool = preproc_fp8_l2.enabled()
        self._plans: List[preproc_fp8_l2.Slot] = []
        self._x0s: List[Optional[torch.Tensor]] = []
        if self._fp8 and _grouped_bwd_enabled():
            global _FP8_EM2_WARNED
            if not _FP8_EM2_WARNED:
                _FP8_EM2_WARNED = True
                logger.warning(
                    "[preproc-grouped-tail-bwd] %s=1 is INACTIVE under %s=1: the FP8 "
                    "tail runs its own LayerNorm(512) backward (stacked grouped "
                    "call when PR #7 owns all three, else per MLP) rank=%d",
                    BWD_ENV_VAR,
                    preproc_fp8_l2.ENV_VAR,
                    _rank(),
                )

    @property
    def declined(self) -> bool:
        """True once the chain fell back to the shipped per-MLP tails."""
        return self._declined

    def head(self, mlp: torch.nn.Module, x: torch.Tensor) -> None:
        if not self._declined:
            reason = mlp_decline_reason(mlp, x, self._eps)
            if reason is None and (
                x.shape[0] != self._rows or x.device != self._device
            ):
                reason = (
                    f"input {tuple(x.shape)} on {x.device} does not match "
                    f"{self._rows} rows on {self._device}"
                )
            if reason is None and len(self._mlps) >= 3:
                reason = "more than 3 MLPs"
            if reason is None:
                assert isinstance(mlp, FusedLinearNormMLP)
                self._mlps.append(mlp)
                if self._fp8:
                    y, slot = preproc_fp8_l2.run_head(
                        len(self._mlps) - 1, mlp, x, _run_head
                    )
                    self._plans.append(slot)
                    self._x0s.append(x)
                    self._ys.append(y)
                    self._pfs.append(None)
                    return
                self._ys.append(_run_head(mlp, x))
                self._pfs.append(None)
                return
            warn_decline_once(reason)
            self._declined = True
            self._acc = self._complete_shipped()
        acc = self._acc
        self._acc = mlp(x) if acc is None else self._mlp_add(mlp, x, acc)

    def prefilled_reason(
        self, mlp: torch.nn.Module, y: torch.Tensor, fused: bool
    ) -> Optional[str]:
        """None when head_prefilled(mlp, y, fused, ...) may run: the chain has
        already declined (the prefilled head then completes the shipped chain
        exactly as head() would), or this MLP passes head()'s checks with
        _can_fuse taken from `fused`. Otherwise the reason head() would decline
        the group AT THIS MLP (or that E-M3a plans this MLP FP8, see below).
        The caller must then run the original order
        (DLRM_HSTU_FP8_ACTION_PREFILL fallback): in that one case the shipped
        chain completes the queued pairs BEFORE this MLP's head, so a head
        created earlier (the prefill bridge) would change the backward order."""
        if self._declined:
            return None
        if not isinstance(mlp, FusedLinearNormMLP):
            return f"{type(mlp).__name__} is not FusedLinearNormMLP"
        if not fused:
            return "prefilled head did not take the fused path"
        reason = _last_pair_reason(mlp, self._eps)
        if reason is None and (y.shape[0] != self._rows or y.device != self._device):
            reason = (
                f"input {tuple(y.shape)} on {y.device} does not match "
                f"{self._rows} rows on {self._device}"
            )
        if reason is None and len(self._mlps) >= 3:
            reason = "more than 3 MLPs"
        if reason is None and self._fp8 and len(self._mlps) in preproc_fp8_l2.FP8_MLPS:
            # E-M3a plans MLP k = len(self._mlps) FP8 (or warms its a1/dh2
            # sites), but a prefilled head ran the shipped BF16 way and gets a
            # BF16 slot (prefilled_slot): prefill on would then differ from
            # off. Today FP8_MLPS = (0, 1) and the action MLP is k = 2.
            reason = "E-M3a plans this MLP FP8 (preproc_fp8_l2.FP8_MLPS)"
        return reason

    def head_prefilled(
        self,
        mlp: torch.nn.Module,
        y: torch.Tensor,
        fused: bool,
        prefilled: Optional[object],
    ) -> None:
        """head() for an MLP whose head already ran (DLRM_HSTU_FP8_ACTION_PREFILL,
        modules/action_prefill.py): `y` is the output of every pair but the
        last, computed by FusedLinearNormMLP.forward_head -- the same
        _LinearNormFunction calls _run_head makes -- and `fused` is its
        _can_fuse(x) answer. `prefilled` optionally holds the last pair's
        casts + addmm, also issued earlier; the node that runs the last pair
        reuses it (bitwise: same ops on the same operands). Call only when
        prefilled_reason(mlp, y, fused) is None."""
        if not self._declined:
            reason = self.prefilled_reason(mlp, y, fused)
            if reason is not None:
                raise RuntimeError(
                    "GroupedTail.head_prefilled: the group would decline here "
                    f"({reason}); the caller must run the original order"
                )
            assert isinstance(mlp, FusedLinearNormMLP)
            self._mlps.append(mlp)
            self._ys.append(y)
            self._pfs.append(prefilled)
            if self._fp8:
                # E-M3a: this head already ran the shipped way, so the entry is
                # BF16; its slot carries the PrefilledLinear into the FP8 node
                # (one per-entry non-tensor argument, see preproc_fp8_l2.Slot).
                self._plans.append(
                    preproc_fp8_l2.prefilled_slot(len(self._mlps) - 1, prefilled)
                )
                self._x0s.append(None)
            return
        acc = self._acc
        # The shipped _mlp_add (F2 on, which the group requires) cut after the
        # head: the last pair with residual = the running sum.
        if isinstance(mlp, FusedLinearNormMLP):
            self._acc = mlp.forward_tail(y, fused, residual=acc, prefilled=prefilled)
        else:  # pragma: no cover - the prefill only issues for nn.Sequential
            for m in list(mlp)[-2:]:
                y = m(y)
            self._acc = y if acc is None else acc + y

    def _complete_shipped(self) -> Optional[torch.Tensor]:
        """The queued last pairs, launched as FusedLinearNormMLP.forward does
        under _mlp_add with F2 on: content without a residual, then each next
        MLP's last pair with residual = the running sum."""
        acc: Optional[torch.Tensor] = None
        for i, (mlp, y, pf) in enumerate(zip(self._mlps, self._ys, self._pfs)):
            if self._fp8 and self._plans[i].mode == "fp8":
                # E-M3a: this head ran P2 and y is its stride-0 proxy. Re-run
                # the head the shipped way; the FP8 node's output is dropped,
                # so autograd never reaches it.
                self._plans[i].payload = None
                self._plans[i].x2lb = None  # X2L-B (item (93)): its handle, unused now
                x0 = self._x0s[i]
                assert x0 is not None
                y = _run_head(mlp, x0)
            lin, norm = mlp._pairs()[-1]
            acc = _LinearNormFunction.apply(
                y,
                lin.weight,
                lin.bias,
                norm.weight,
                norm.bias,
                norm._eps,
                False,
                acc,
                pf,
            )
        self._mlps, self._ys, self._pfs = [], [], []
        self._plans, self._x0s = [], []
        return acc

    def finish(self) -> torch.Tensor:
        if not self._declined and len(self._mlps) < 2:
            warn_decline_once(f"{len(self._mlps)} MLP(s) in the chain, need 2 or 3")
            self._declined = True
            self._acc = self._complete_shipped()
        if self._declined:
            acc = self._acc
            assert acc is not None
            return acc
        flat: List[torch.Tensor] = []
        for mlp, y in zip(self._mlps, self._ys):
            lin, norm = mlp._pairs()[-1]
            flat += [y, lin.weight, lin.bias, norm.weight, norm.bias]
        if self._fp8:
            # E-M3a armed: the tail with the FP8 last Linear for the MLPs whose
            # slot is "fp8"; every other MLP runs this tail's shipped body (a
            # prefilled one reuses its PrefilledLinear, carried by its slot).
            out = preproc_fp8_l2.grouped_tail_apply(self._eps, self._plans, flat)
        else:
            pfs = tuple(self._pfs) if any(pf is not None for pf in self._pfs) else None
            out = _GroupedLinearLNTailFunction.apply(self._eps, pfs, *flat)
        _report_armed_once(len(self._mlps), self._rows)
        self._mlps, self._ys, self._pfs = [], [], []
        self._plans, self._x0s = [], []
        return out


def begin_grouped_tail(
    mlp_add: MlpAdd,
    content_mlp: torch.nn.Module,
    content_x: torch.Tensor,
    fuse_linear_norm: bool,
    fuse_ln_residual: bool,
    has_additional: bool,
    output_dim: int,
) -> Optional[GroupedTail]:
    """A GroupedTail for this forward, or None (shipped path) if the
    preconditions known before the first MLP fail."""
    if is_fx_tracing() or isinstance(content_x, torch.fx.Proxy):
        return None  # symbolic trace: nothing executes, nothing to report
    if torch.compiler.is_compiling():
        return None
    reason: Optional[str] = None
    if not (fuse_linear_norm and fuse_ln_residual):
        reason = (
            f"fuse_linear_norm={fuse_linear_norm} fuse_ln_residual="
            f"{fuse_ln_residual}; both are required"
        )
    elif not has_additional:
        reason = "no additional-embedding MLP"
    elif output_dim != TAIL_DIM:
        reason = f"output dim {output_dim} != {TAIL_DIM}"
    elif not content_x.is_cuda:
        reason = f"input on {content_x.device}, not CUDA"
    elif torch.version.hip is None:
        # The kernel and its bitwise proof are gfx950 (ROCm) only; a CUDA host
        # reusing this gin keeps the shipped per-MLP tails.
        reason = "not a ROCm/HIP build (the grouped kernel is verified on gfx950 only)"
    elif not (
        torch.is_autocast_enabled(content_x.device.type)
        and torch.get_autocast_dtype(content_x.device.type) == torch.bfloat16
    ):
        # The kernel's bitwise proof is for bf16 norm inputs, which only the
        # autocast(bf16) path produces from fp32 Linear weights.
        reason = "not under torch.autocast(cuda, bfloat16)"
    else:
        reason = mlp_decline_reason(content_mlp, content_x, None)
    if reason is not None:
        warn_decline_once(reason)
        return None
    assert isinstance(content_mlp, FusedLinearNormMLP)
    eps = content_mlp._pairs()[-1][1]._eps
    return GroupedTail(mlp_add, eps, content_x.shape[0], content_x.device)
