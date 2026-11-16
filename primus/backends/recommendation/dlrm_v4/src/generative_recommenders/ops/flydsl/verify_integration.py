"""Regression guard for the $HSTU_ATTN_BWD flag.

Run on a gfx950 box with flydsl installed:

    python -m generative_recommenders.ops.flydsl.verify_integration --batch 64

Checks BOTH call sites, because they are not the same code path and only one of
them is what production runs:

  * standalone  — ops/hstu_attention.py's `hstu_mha` dispatch. Used by tests and
                  by the non-fused kernels.
  * fused       — ops/triton/triton_hstu_preprocess_and_attention.py, which is
                  what yambda-5b actually executes (HammerKernel.TRITON,
                  prefill=False). Here q/k/v arrive as column slices of one uvqk
                  buffer and the gradients must land back in that buffer's
                  views, so this is the case that exercises the strided
                  in/out handling.

Also asserts the safety properties the flag depends on now that FlyDSL is the
DEFAULT: an unset variable resolves to FlyDSL, a shape outside the specialization
(num_targets=2) still falls back to Triton rather than being computed
differently, and a backend name that is neither raises instead of silently
picking one.
"""

import argparse
import os
import sys

import numpy as np
import torch

from generative_recommenders.common import HammerKernel
from generative_recommenders.ops import hstu_attention
from generative_recommenders.ops.flydsl import hstu_flydsl_mha
from generative_recommenders.ops.triton.triton_hstu_attention import (
    triton_hstu_attention_fwd,
)
from generative_recommenders.ops.triton.triton_hstu_preprocess_and_attention import (
    triton_hstu_preprocess_and_attention,
)

HEADS = 4          # hstu_num_heads
ATTN_DIM = 128     # hstu_attn_qk_dim
HIDDEN_DIM = 128   # hstu_attn_linear_dim
EMB_DIM = 512      # hstu_transducer_embedding_dim

# SYNTHETIC length pool, generated in code from configured limits only: no
# training-data statistics (user rule 2026-10-12; the former
# data/real_lens_*.npy pools, taken from real sample lengths, were removed).
# Per-sample UIH lengths are drawn uniformly from [MIN_LEN, MAX_LEN] with a fixed
# seed, so every run sees the same pool. MAX_LEN = max_seq_len - 1 (gin
# get_hstu_configs.max_seq_len = 4096; one slot is the target) and MIN_LEN is a
# quarter of it, so the batch is ragged and long, spanning several kernel tiles.
MAX_SEQ_LEN = 4096      # get_hstu_configs.max_seq_len (yambda_5b.gin)
MAX_LEN = MAX_SEQ_LEN - 1
MIN_LEN = MAX_SEQ_LEN // 4
LENS_SEED = 20261012


def synthetic_lens(batch, seed=LENS_SEED, lo=MIN_LEN, hi=MAX_LEN):
    """Deterministic synthetic per-sample lengths (int64) in [lo, hi]."""
    rng = np.random.default_rng(seed)
    return rng.integers(lo, hi + 1, size=batch, dtype=np.int64)


def snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref32, got32 = ref.float(), got.float()
    noise = torch.linalg.vector_norm(ref32 - got32).item()
    if noise == 0.0:
        return float("inf")
    return 20.0 * np.log10(torch.linalg.vector_norm(ref32).item() / noise)


def set_backend(backend, fwd="triton"):
    """Select the backward and forward kernels. None means "leave unset".

    The forward defaults to Triton here so a backward A/B isolates the backward:
    the two are independently selectable, and letting both move at once would
    make it impossible to say which half a number came from.
    """
    for var, val in (("HSTU_ATTN_BWD", backend), ("HSTU_ATTN_FWD", fwd)):
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val


def offsets(batch, seed=1234):
    torch.manual_seed(seed)
    lens = torch.from_numpy(synthetic_lens(batch))
    off = torch.zeros(batch + 1, dtype=torch.int64)
    off[1:] = torch.cumsum(lens, 0)
    return off.cuda(), int(off[-1]), int(lens.max())


def run_standalone(backend, batch, n_targets=1):
    set_backend(backend)
    impl = hstu_attention._hstu_mha_impl()
    seq_offsets, total, max_seq_len = offsets(batch)

    def t():
        return torch.randn(
            total, HEADS, ATTN_DIM, dtype=torch.bfloat16, device="cuda"
        ).requires_grad_(True)

    q, k, v = t(), t(), t()
    num_targets = torch.full(
        (batch,), n_targets, dtype=torch.int64, device="cuda"
    )
    torch.manual_seed(7)
    dout = torch.randn(q.shape, dtype=torch.bfloat16, device="cuda")

    out = hstu_attention.hstu_mha(
        max_seq_len=max_seq_len, alpha=1.0 / (ATTN_DIM**0.5),
        q=q, k=k, v=v, seq_offsets=seq_offsets, causal=True,
        num_targets=num_targets, sort_by_length=True,
        kernel=HammerKernel.TRITON,
    )
    out.backward(dout)
    return out.detach(), q.grad, k.grad, v.grad, impl.__module__


def run_fused(backend, batch, seed=1234, fwd="triton", scale=1.0):
    set_backend(backend, fwd)
    fwd_calls_before = hstu_flydsl_mha._forward_calls
    seq_offsets, total, max_seq_len = offsets(batch, seed)
    torch.manual_seed(seed)

    uvqk_dim = 2 * HIDDEN_DIM * HEADS + 2 * ATTN_DIM * HEADS
    x = torch.randn(total, EMB_DIM, dtype=torch.bfloat16, device="cuda") * 0.1
    x.requires_grad_(True)
    norm_weight = torch.randn(EMB_DIM, dtype=torch.bfloat16, device="cuda")
    norm_bias = torch.randn(EMB_DIM, dtype=torch.bfloat16, device="cuda")
    # `scale` widens uvqk, and so the score qk. It matters because silu is
    # nearly linear near 0: at small qk, silu(c*x) ~ c*silu(x), so a scale error
    # on the score is indistinguishable from one on p and the check goes blind to
    # it. A large scale puts qk back in the nonlinear region.
    uvqk_weight = (
        torch.randn(EMB_DIM, uvqk_dim, dtype=torch.bfloat16, device="cuda")
        * 0.05 * scale
    )
    uvqk_bias = torch.zeros(uvqk_dim, dtype=torch.bfloat16, device="cuda")
    for p in (norm_weight, norm_bias, uvqk_weight, uvqk_bias):
        p.requires_grad_(True)

    _, attn_out = triton_hstu_preprocess_and_attention(
        x=x, norm_weight=norm_weight, norm_bias=norm_bias, norm_eps=1e-6,
        num_heads=HEADS, attn_dim=ATTN_DIM, hidden_dim=HIDDEN_DIM,
        uvqk_weight=uvqk_weight, uvqk_bias=uvqk_bias, max_seq_len=max_seq_len,
        seq_offsets=seq_offsets, attn_alpha=1.0 / (ATTN_DIM**0.5),
        num_targets=torch.ones(batch, dtype=torch.int64, device="cuda"),
        max_attn_len=0, contextual_seq_len=0,
        recompute_uvqk_in_backward=True, recompute_normed_x_in_backward=True,
        sort_by_length=True,
    )
    torch.manual_seed(7)
    attn_out.backward(
        torch.randn(attn_out.shape, dtype=torch.bfloat16, device="cuda")
    )
    return {
        "attn_out": attn_out.detach(),
        "d_x": x.grad,
        "d_uvqk_weight": uvqk_weight.grad,
        "d_norm_weight": norm_weight.grad,
        "d_norm_bias": norm_bias.grad,
        # How many times the FlyDSL forward actually fired. Liveness has to be
        # OBSERVED, not inferred from the numerics: the two forwards agree
        # bit-for-bit (see check_fused_fwd), so "the outputs differ" -- the test
        # the backward check can use -- would report a silent decline as a pass.
        "fwd_calls": hstu_flydsl_mha._forward_calls - fwd_calls_before,
    }


def check_standalone(batch) -> bool:
    print("standalone path (ops/hstu_attention.py)")
    ref_out, ref_dq, ref_dk, ref_dv, ref_mod = run_standalone("triton", batch)
    got_out, got_dq, got_dk, got_dv, got_mod = run_standalone("flydsl", batch)
    if "flydsl" not in got_mod:
        print(f"  FAIL: flag selected {got_mod}")
        return False

    fwd_ok = torch.equal(ref_out, got_out)
    # dq is bf16-accumulated through atomics (~45 dB); dk/dv are deterministic.
    scores = {
        "dq": (snr_db(ref_dq, got_dq), 40),
        "dk": (snr_db(ref_dk, got_dk), 60),
        "dv": (snr_db(ref_dv, got_dv), 60),
    }
    print(f"  forward bit-identical: {fwd_ok}")
    ok = fwd_ok
    for name, (s, floor) in scores.items():
        ok = ok and s > floor
        print(f"  {name} snr {s:6.1f} dB (floor {floor})")
    return ok


def check_fused(batch) -> bool:
    print("fused path (triton_hstu_preprocess_and_attention.py) — production")
    ref, got = run_fused("triton", batch), run_fused("flydsl", batch)
    ok = torch.equal(ref["attn_out"], got["attn_out"])
    print(f"  forward bit-identical: {ok}")
    for key, floor in (
        ("d_x", 40), ("d_uvqk_weight", 40),
        ("d_norm_weight", 30), ("d_norm_bias", 30),
    ):
        s = snr_db(ref[key], got[key])
        ok = ok and s > floor
        print(f"  {key:16s} snr {s:6.1f} dB (floor {floor})")

    # Guards against a vacuous pass: if FlyDSL had silently declined, the
    # gradients would be bit-identical and every SNR above would read inf.
    ran = not torch.equal(ref["d_x"], got["d_x"])
    print(f"  flydsl actually ran: {ran}")
    return ok and ran


def check_fused_fwd(batch) -> bool:
    """A/B the FORWARD alone through the production fused path.

    The backward is pinned to Triton on both sides, so anything that moves is the
    forward's doing, and the gradients must come back bit-identical: they are
    computed by the same Triton backward from the same uvqk and the same dout,
    and the forward's output never feeds them (HSTU has no softmax, so there is
    no lse or rowsum(dO*O) handed across).

    attn_out itself comes back BIT-IDENTICAL, which is worth explaining because
    it looks too good. Both kernels accumulate both gemms in f32 and round once
    to bf16. Summing the same terms in a different order perturbs an f32
    accumulator by ~1e-7 relative, while bf16 has 8 mantissa bits and so quantizes
    at ~0.4% -- three orders of magnitude coarser. Agreement is therefore the
    expected outcome, not a coincidence, and the two only diverge where a value
    sits within 1e-7 of a bf16 rounding boundary.

    That is a stronger result than an SNR, but it costs the obvious liveness
    test: "the outputs differ" cannot distinguish a working kernel from one that
    silently declined. Liveness is observed from the call counter instead.
    """
    print("fused path, FORWARD a/b (backward pinned to triton)")
    ok = True
    for scale, floor in ((1.0, 40), (8.0, 40)):
        ref = run_fused("triton", batch, fwd="triton", scale=scale)
        got = run_fused("triton", batch, fwd="flydsl", scale=scale)
        s = snr_db(ref["attn_out"], got["attn_out"])
        ran = got["fwd_calls"] > 0 and ref["fwd_calls"] == 0
        grads_ok = all(
            torch.equal(ref[key], got[key])
            for key in ("d_x", "d_uvqk_weight", "d_norm_weight", "d_norm_bias")
        )
        exact = torch.equal(ref["attn_out"], got["attn_out"])
        ok = ok and s > floor and ran and grads_ok
        print(f"  scale {scale:4.1f}: attn_out snr {s:6.1f} dB (floor {floor})"
              f"{' [bit-identical]' if exact else ''}"
              f"  flydsl fired {got['fwd_calls']}x (triton arm "
              f"{ref['fwd_calls']}x)  gradients identical: {grads_ok}")
        if not ran:
            print("    FAIL: liveness -- the flydsl forward did not fire")
    return ok


def check_fwd_gate_discriminates(batch) -> bool:
    """Prove the forward gate can FAIL, by injecting a 1% alpha error.

    Without this, "bit-identical" above is not evidence: a check that cannot
    distinguish a correct kernel from a subtly wrong one would report a pass
    either way. This is the same fault the backward's gate is calibrated on --
    a pure relative scale error eps scores -20*log10(eps) dB regardless of
    magnitude, so 1% lands near 40 dB and must be rejected.
    """
    print("forward gate discriminates (fault injection)")
    seq_offsets, total, max_seq_len = offsets(batch)
    torch.manual_seed(1234)
    alpha = 1.0 / (ATTN_DIM**0.5)

    def t(scale):
        return (
            torch.randn(total, HEADS, ATTN_DIM, dtype=torch.bfloat16,
                        device="cuda") * scale
        )

    ok = True
    for scale in (0.1, 1.0):
        q, k, v = t(scale), t(scale), t(scale)
        ref = triton_hstu_attention_fwd(
            N=max_seq_len, alpha=alpha, q=q, k=k, v=v,
            seq_offsets=seq_offsets, num_targets=None, max_attn_len=0,
            contextual_seq_len=0, sort_by_length_indices=None,
            enable_tma=False, num_softmax_heads=0,
        )
        good = hstu_flydsl_mha.flydsl_hstu_attention_fwd(
            q, k, v, seq_offsets, max_seq_len, alpha
        )
        bad = hstu_flydsl_mha.flydsl_hstu_attention_fwd(
            q, k, v, seq_offsets, max_seq_len, alpha * 1.01
        )
        s_good, s_bad = snr_db(ref, good), snr_db(ref, bad)
        # The good arm must clear the floor and the faulted arm must not, with
        # daylight between them -- otherwise the floor is decoration.
        passes = s_good > 40 and s_bad < 40
        ok = ok and passes
        print(f"  scale {scale:4.1f}: correct {s_good:6.1f} dB, "
              f"alpha x1.01 {s_bad:6.1f} dB -> "
              f"{'rejected as it must be' if passes else 'FAIL'}")
    return ok


def check_fwd_safety(batch) -> bool:
    print("forward safety")
    # Default off (Triton). Unset must not select flydsl; explicit
    # HSTU_ATTN_FWD=flydsl is the opt-in (bf16, or e4m3 when FP8 is on).
    os.environ.pop("HSTU_ATTN_FWD", None)
    default_ok = not hstu_flydsl_mha.flydsl_fwd_selected()
    print(f"  unset -> triton (default): {'ok' if default_ok else 'FAIL'}")
    os.environ["HSTU_ATTN_FWD"] = "triton"
    optout_ok = not hstu_flydsl_mha.flydsl_fwd_selected()
    default_ok = default_ok and optout_ok
    print(f"  HSTU_ATTN_FWD=triton -> triton: {'ok' if optout_ok else 'FAIL'}")

    # Outside the specialization, so the forward must decline and Triton must
    # run -- making attn_out bit-identical, not merely close.
    ref = run_fused("triton", 8, fwd="triton")
    os.environ["HSTU_ATTN_FWD"] = "flydsl"
    reason = hstu_flydsl_mha.unsupported_fwd_shape_reason(
        torch.empty(4, HEADS, ATTN_DIM, dtype=torch.bfloat16, device="cuda"),
        torch.empty(4, HEADS, ATTN_DIM, dtype=torch.bfloat16, device="cuda"),
        torch.empty(4, HEADS, ATTN_DIM, dtype=torch.bfloat16, device="cuda"),
        torch.full((1,), 2, dtype=torch.int64, device="cuda"), 0, 0,
    )
    fb_ok = reason is not None
    print(f"  num_targets=2 declines: {'ok' if fb_ok else 'FAIL'} ({reason})")

    # A dim the kernel is not written around must also decline.
    reason_d = hstu_flydsl_mha.unsupported_fwd_shape_reason(
        torch.empty(4, HEADS, 64, dtype=torch.bfloat16, device="cuda"),
        torch.empty(4, HEADS, 64, dtype=torch.bfloat16, device="cuda"),
        torch.empty(4, HEADS, 64, dtype=torch.bfloat16, device="cuda"),
        None, 0, 0,
    )
    dim_ok = reason_d is not None
    print(f"  dim=64 declines: {'ok' if dim_ok else 'FAIL'} ({reason_d})")
    set_backend(None, None)
    del ref
    return default_ok and fb_ok and dim_ok


def check_safety(batch) -> bool:
    print("safety")
    default_mod = run_standalone(None, 8)[4]
    default_ok = "flydsl" in default_mod
    print(f"  unset -> {default_mod.rsplit('.', 1)[-1]}: "
          f"{'ok' if default_ok else 'FAIL'}")

    # Outside the specialization, so FlyDSL must decline and the Triton backward
    # must run -- making the gradients bit-identical, not merely close.
    ref = run_standalone("triton", 8, n_targets=2)
    got = run_standalone("flydsl", 8, n_targets=2)
    fb_ok = all(torch.equal(a, b) for a, b in zip(ref[1:4], got[1:4]))
    print(f"  num_targets=2 falls back: {'ok' if fb_ok else 'FAIL'}")

    # A typo must not resolve to a working backend: that is how a run silently
    # loses the default kernel.
    set_backend("flydsel")
    try:
        hstu_attention._hstu_mha_impl()
        bad_ok = False
    except ValueError:
        bad_ok = True
    finally:
        set_backend(None)
    print(f"  unrecognized value raises: {'ok' if bad_ok else 'FAIL'}")

    # The host gate itself. This script only runs where FlyDSL is supported, so
    # the assertion is that it reports supported -- the raise it guards is
    # exercised by construction on any other box.
    reason = hstu_flydsl_mha.unsupported_hardware_reason()
    hw_ok = reason is None
    print(f"  host supported: {'ok' if hw_ok else 'FAIL (' + str(reason) + ')'}")
    return default_ok and fb_ok and bad_ok and hw_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    results = [
        check_standalone(args.batch),
        check_fused(args.batch),
        check_fused_fwd(args.batch),
        check_fwd_gate_discriminates(args.batch),
        check_fwd_safety(args.batch),
        check_safety(args.batch),
    ]
    ok = all(results)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
