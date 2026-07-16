# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Numerical equivalence tests: TE spec vs Primus-Turbo local spec attention.

Unlike the earlier version of this file, which compared in-file replicas of the
attention kernels, these tests build the *production* ``core_attention`` modules
returned by the two spec providers and feed them identical inputs:

  * TE spec    -> ``TEDotProductAttention``     (FusedAttention/CK, SBHD native)
  * local spec -> ``PrimusTurboLocalAttention`` (Primus-Turbo flash attention)

Both modules are constructed via ``build_module(...)`` with the exact keyword
arguments Megatron's ``transformer/attention.py`` uses when it wires up
``submodules.core_attention``, so a regression in how either spec selects or
configures its attention kernel is caught here.

The (formerly user-visible) contiguous-vs-non-contiguous split is now a
production-internal detail: ``PrimusTurboLocalAttention`` decides whether to
force a contiguous BSHD copy from the device capability (``force_contiguous_qkv``
on gfx942). The four tests below therefore exercise the two real production
paths in eager and compiled form rather than permutations of hand-rolled
replicas.

NOTE: this module is GPU-only (``skip_if_no_cuda()``) and additionally requires
the Megatron + Primus-Turbo stack that only imports inside the ROCm training
container, so it is validated on the GPU runner lane, not in CPU CI.
"""

import copy
import os

os.environ["NVTE_FUSED_ATTN"] = "1"
os.environ["NVTE_FUSED_ATTN_CK"] = "1"
os.environ["NVTE_CK_USES_FWD_V3"] = "1"
os.environ["NVTE_CK_USES_BWD_V3"] = "1"

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

import torch.nn as nn

# NOTE: the diffusion ``conftest`` installs Primus' aiter RTLD_DEEPBIND import
# hook (``install_aiter_deepbind_hook``) before this module loads, so Turbo's
# attention backward binds the pinned ``aiter::mha_bwd`` instead of the stale one
# vendored by transformer_engine below (ROCm/aiter#1332). Without it, attention
# backward crashes on gfx942/gfx950.
import transformer_engine.pytorch  # noqa: F401  (import side effects only)
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig

from primus.backends.megatron.core.extensions.primus_turbo_local_spec import (
    PrimusTurboLocalAttention,
)
from tests.utils import PrimusUT

_has_cuda = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _has_cuda, reason="CUDA required")

# head_dim=128 is required for FAv3 eligibility in AITER.
# seq=256 > 128 so the Python API fmha_v3_fwd path is taken.
_DIM = 512
_HEADS = 4
_HEAD_DIM = _DIM // _HEADS  # 128
_SEQ = 256
_BATCH = 2
_N_STEPS = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_snr(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.float(), y.float()
    signal_power = torch.norm(x).pow(2)
    noise_power = torch.norm(x - y).pow(2)
    return 10 * torch.log10(signal_power / (noise_power + 1e-12)).detach().item()


def _make_sbhd_inputs(n, seq, dim, batch, seed=42):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(seq, batch, dim, dtype=torch.bfloat16, generator=g).cuda() for _ in range(n)]


def _clone_state(model):
    return copy.deepcopy(model.state_dict())


def _run_training_loop(model, inputs, n_steps, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(inputs[step % len(inputs)])
        loss = out.sum()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def _print_comparison(label_a, losses_a, label_b, losses_b, milestones=None):
    if milestones is None:
        milestones = [0, 1, 5, 10, 20, 50, 99]
    milestones = [m for m in milestones if m < len(losses_a) and m < len(losses_b)]
    print(f"\n{'Step':>6} | {label_a:>20} | {label_b:>20} | {'Rel Diff':>10}")
    print("-" * 65)
    for m in milestones:
        a, b = losses_a[m], losses_b[m]
        rel = abs(a - b) / max(abs(a), 1e-12)
        print(f"{m:>6} | {a:>20.6f} | {b:>20.6f} | {rel:>10.6f}")


def _make_attention_config() -> TransformerConfig:
    """Minimal TransformerConfig matching the test attention dims.

    ``kv_channels`` is pinned to ``_HEAD_DIM`` and ``softmax_scale`` is left at
    its default (None) so both production attention modules fall back to the same
    ``1/sqrt(head_dim)`` scale, exactly as they do in a real Flux layer.
    """
    return TransformerConfig(
        num_layers=1,
        hidden_size=_DIM,
        num_attention_heads=_HEADS,
        kv_channels=_HEAD_DIM,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        bf16=True,
        params_dtype=torch.bfloat16,
    )


def _build_core_attention(core_attention_cls):
    """Build a production ``core_attention`` module the way Megatron's
    ``attention.py`` does (``build_module`` with the same kwargs).
    """
    config = _make_attention_config()
    return build_module(
        core_attention_cls,
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.no_mask,
        attention_type="self",
        softmax_scale=config.softmax_scale,
    )


# ---------------------------------------------------------------------------
# Wrapper that swaps only the production core_attention kernel
# ---------------------------------------------------------------------------


class _SpecAttentionModule(nn.Module):
    """norm -> qkv -> production core_attention -> proj.

    The norm/qkv/proj linears are identical across instances; the only thing
    that differs is ``core_attention_cls`` (the class a spec provider's
    ``core_attention()`` returns), so a numerical divergence is attributable to
    the attention kernel/wrapper alone.
    """

    def __init__(self, dim, num_heads, core_attention_cls):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.core = _build_core_attention(core_attention_cls)

    def forward(self, x):
        S, B, D = x.shape
        x = self.norm(x)
        qkv = self.qkv(x).view(S, B, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # Production core_attention forward: (q, k, v, attention_mask, attn_mask_type)
        # q/k/v are SBHD [S, B, H, head_dim]; output is merged [S, B, H*head_dim].
        out = self.core(q, k, v, None, AttnMaskType.no_mask)
        out = out.reshape(S, B, D)
        return self.proj(out)


def _te_module():
    return _SpecAttentionModule(_DIM, _HEADS, TEDotProductAttention)


def _local_module():
    return _SpecAttentionModule(_DIM, _HEADS, PrimusTurboLocalAttention)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@requires_cuda
class TestTEvsLocalSpecAttention(PrimusUT):
    """Numerical equivalence: TE spec vs Primus-Turbo local spec attention.

    Part A: Single-pass SNR (fast, precise kernel-level comparison).
    Part B: Eager training-loop convergence (detects accumulated drift).
    Part C: Compiled vs eager for the local spec.
    Part D: Cross-path end-to-end (TE eager vs local compiled).
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        # init_parallel_state sets up TP=1 parallel state, the TP RNG tracker,
        # and Megatron global args (enable_turbo_attention_float8=False), all of
        # which PrimusTurboLocalAttention.__init__ relies on.
        pass

    def setup_method(self, method):
        torch._dynamo.reset()

    # -----------------------------------------------------------------------
    # Part A: Single-forward-pass SNR
    # -----------------------------------------------------------------------

    def _run_single_pass(self, build_fn):
        """Run one forward + backward, return output, input grad, and state."""
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)

        model = build_fn().to(dtype=torch.bfloat16, device="cuda")

        x = torch.randn(_SEQ, _BATCH, _DIM, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        grad_out = torch.randn(_SEQ, _BATCH, _DIM, dtype=torch.bfloat16, device="cuda")

        out = model(x)
        out.backward(grad_out)
        torch.cuda.synchronize()
        return out.detach(), x.grad.detach(), model.state_dict()

    def _run_single_pass_with_state(self, build_fn, state_dict):
        """Run one forward + backward with pre-loaded norm/qkv/proj weights."""
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)

        model = build_fn().to(dtype=torch.bfloat16, device="cuda")
        own_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in own_keys}
        model.load_state_dict(filtered, strict=False)

        x = torch.randn(_SEQ, _BATCH, _DIM, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        grad_out = torch.randn(_SEQ, _BATCH, _DIM, dtype=torch.bfloat16, device="cuda")

        out = model(x)
        out.backward(grad_out)
        torch.cuda.synchronize()
        return out.detach(), x.grad.detach()

    def _compare_snr(self, label, out_a, grad_a, out_b, grad_b, threshold=40.0):
        out_snr = _compute_snr(out_a, out_b)
        grad_snr = _compute_snr(grad_a, grad_b)
        print(f"\n  [{label}] output SNR={out_snr:.2f} dB, grad SNR={grad_snr:.2f} dB")
        assert out_snr > threshold, f"[{label}] output SNR too low: {out_snr:.2f}"
        assert grad_snr > threshold, f"[{label}] grad SNR too low: {grad_snr:.2f}"

    def test_single_pass_te_vs_local(self):
        """TE-spec and local-spec attention produce near-identical single-pass
        output and input gradients given identical weights/inputs."""
        out_te, grad_te, state_te = self._run_single_pass(_te_module)
        out_local, grad_local = self._run_single_pass_with_state(_local_module, state_te)
        self._compare_snr("TE vs Local", out_te, grad_te, out_local, grad_local)

    # -----------------------------------------------------------------------
    # Part B: Eager training-loop convergence
    # -----------------------------------------------------------------------

    def test_eager_te_vs_local(self):
        """Eager: TE-spec and local-spec attention converge to the same loss."""
        torch._dynamo.reset()
        inputs = _make_sbhd_inputs(20, _SEQ, _DIM, _BATCH)

        model_te = _te_module().to(dtype=torch.bfloat16, device="cuda")
        init_state = _clone_state(model_te)
        losses_te = _run_training_loop(model_te, inputs, _N_STEPS)

        model_local = _local_module().to(dtype=torch.bfloat16, device="cuda")
        own_keys = set(model_local.state_dict().keys())
        filtered = {k: v for k, v in init_state.items() if k in own_keys}
        model_local.load_state_dict(filtered, strict=False)
        losses_local = _run_training_loop(model_local, inputs, _N_STEPS)

        _print_comparison("TE Eager", losses_te, "Local Eager", losses_local)

        final_rel = abs(losses_te[-1] - losses_local[-1]) / max(abs(losses_te[-1]), 1e-12)
        print(f"\n  Final loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.02, (
            f"TE vs Local eager: final loss diverged: "
            f"te={losses_te[-1]:.6f}, local={losses_local[-1]:.6f}, rel={final_rel:.6f}"
        )

    # -----------------------------------------------------------------------
    # Part C/D: Compiled local spec (in-process)
    #
    # An earlier revision ran the compiled training in a subprocess because
    # ``torch.compile`` + ``allow_in_graph(AiterFlashAttnFunc)`` could corrupt
    # AOT-autograd view-replay metadata when sharing a process with the rest of
    # the pytest/Megatron harness. That is avoided here by resetting Dynamo
    # (``torch._dynamo.reset()``) immediately before each compiled build, so the
    # local-attention graph is traced from a clean state regardless of what
    # earlier tests in the session compiled. The compiled run builds the
    # *production* ``PrimusTurboLocalAttention`` via the same ``_local_module``
    # helper as the eager paths, so the two paths differ only by compilation.
    # -----------------------------------------------------------------------

    @staticmethod
    def _run_compiled_local(n_steps: int = 100, init_state=None):
        """Compile + train the production local-spec attention in-process.

        Resets Dynamo and allows ``AiterFlashAttnFunc`` into the graph, then
        compiles a fresh ``_local_module`` and runs the training loop. When
        ``init_state`` is given the model is seeded with it so a comparison run
        shares identical norm/qkv/proj weights; otherwise the freshly
        initialized weights are captured. Returns ``(losses, init_state)``.
        """
        from primus_turbo.pytorch.ops.attention.flash_attn_interface import (
            AiterFlashAttnFunc,
        )

        torch._dynamo.reset()
        torch._dynamo.allow_in_graph(AiterFlashAttnFunc)

        model = _local_module().to(dtype=torch.bfloat16, device="cuda")
        if init_state is not None:
            own_keys = set(model.state_dict().keys())
            model.load_state_dict({k: v for k, v in init_state.items() if k in own_keys}, strict=False)
        captured_state = _clone_state(model)

        inputs = _make_sbhd_inputs(20, _SEQ, _DIM, _BATCH)
        compiled = torch.compile(model)
        losses = _run_training_loop(compiled, inputs, n_steps)
        return losses, captured_state

    def test_compiled_local_vs_eager(self):
        """Compiled local-spec attention converges like eager local-spec."""
        losses_compiled, init_state = self._run_compiled_local(_N_STEPS)

        torch._dynamo.reset()
        model_eager = _local_module().to(dtype=torch.bfloat16, device="cuda")
        own_keys = set(model_eager.state_dict().keys())
        model_eager.load_state_dict({k: v for k, v in init_state.items() if k in own_keys}, strict=False)
        inputs = _make_sbhd_inputs(20, _SEQ, _DIM, _BATCH)
        losses_eager = _run_training_loop(model_eager, inputs, _N_STEPS)

        _print_comparison("Local Eager", losses_eager, "Local Compiled", losses_compiled)

        final_rel = abs(losses_eager[-1] - losses_compiled[-1]) / max(abs(losses_eager[-1]), 1e-12)
        print(f"\n  Final loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.02, f"Diverged: {final_rel:.6f}"

    def test_te_eager_vs_local_compiled(self):
        """End-to-end cross-path: TE-spec eager and local-spec compiled, sharing
        initial weights + inputs, converge to the same loss."""
        losses_local, init_state = self._run_compiled_local(_N_STEPS)

        torch._dynamo.reset()
        model_te = _te_module().to(dtype=torch.bfloat16, device="cuda")
        own_keys = set(model_te.state_dict().keys())
        model_te.load_state_dict({k: v for k, v in init_state.items() if k in own_keys}, strict=False)
        inputs = _make_sbhd_inputs(20, _SEQ, _DIM, _BATCH)
        losses_te = _run_training_loop(model_te, inputs, _N_STEPS)

        _print_comparison("TE Eager", losses_te, "Local Compiled", losses_local)

        final_rel = abs(losses_te[-1] - losses_local[-1]) / max(abs(losses_te[-1]), 1e-12)
        print(f"\n  Final loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.02, f"Diverged: {final_rel:.6f}"
