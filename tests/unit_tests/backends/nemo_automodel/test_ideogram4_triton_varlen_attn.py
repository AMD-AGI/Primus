###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the head-dim-tuned Triton var-len attention path (``triton_varlen_attn``).

WHY THIS TEST EXISTS:
  The Triton path is 2.1x faster than CK-tile at hd=256 only because Primus overrides five
  of aiter's block-size fields. Every failure mode of that arrangement is SILENT:

    * a config with ``BLOCK_M2 < BLOCK_N1`` leaves the tail of Q at its zero init, so dq is
      wrong while dk/dv stay right and nothing raises (four sweep candidates hit this, dq
      rel-err 0.37-0.62). ``test_dq_coverage_*`` and the zero-row check in the parity test
      are the guards.
    * aiter's ``_get_config`` is ``lru_cache``d and returns the one dict it parsed, so
      overriding it in place would retune every other Triton MHA caller in the process.
      ``test_aiter_shipped_config_is_not_mutated`` pins the deep copy.
    * the override is only measured for (gfx950, hd=256). Any other shape must fall through
      to stock aiter rather than inherit blocks nobody swept -- ``test_untuned_shape_*``.
      And since stock aiter at hd=256 is 2x SLOWER than CK, falling through is not enough on
      its own now that ``triton`` is the preset's default: an unswept GPU has to end up on CK
      instead -- ``test_untuned_arch_degrades_to_ck``.

  The block sizes themselves are pinned by ``test_tuned_deltas_are_the_measured_values`` so
  an edit has to be deliberate; the numbers come from varlen_sweep/triton_hd256_sweep.json.

The config tests need only aiter importable (they monkeypatch the arch). The parity test
needs a GPU and compiles Triton kernels, so it is skipped without CUDA.
"""

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - CPU-less lint environments
    pytest.skip("Ideogram-4 Triton attention tests require torch", allow_module_level=True)

from primus.backends.nemo_automodel.models.ideogram4 import triton_varlen_attn as tva


def _has_aiter() -> bool:
    try:
        import aiter  # noqa: F401

        return True
    except Exception:
        return False


needs_aiter = pytest.mark.skipif(not _has_aiter(), reason="aiter is not importable")
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


@pytest.fixture
def gfx950(monkeypatch):
    """Pretend we are on the arch the sweep ran on, so the deltas apply off-GPU too."""
    monkeypatch.setattr(tva, "_arch", lambda: "gfx950")
    return "gfx950"


# --------------------------------------------------------------------------- #
# The tuned values                                                             #
# --------------------------------------------------------------------------- #
def test_tuned_deltas_are_the_measured_values():
    """Pin the swept block sizes; changing them must be a deliberate, reviewed edit."""
    assert tva._TUNED_DELTAS[("gfx950", 256)] == {
        "fwd": {"BLOCK_N": 16, "num_warps": 8, "num_stages": 2},
        "bwd": {"BLOCK_N1": 64, "num_stages": 2},
    }


# --------------------------------------------------------------------------- #
# dQ coverage: the silent-wrong-gradient guard                                 #
# --------------------------------------------------------------------------- #
def test_dq_coverage_rejects_undersized_block_m2():
    with pytest.raises(ValueError, match="silently wrong dq"):
        tva._check_dq_coverage({"BLOCK_M2": 32, "BLOCK_N1": 64})


@pytest.mark.parametrize("block_m2, block_n1", [(128, 64), (64, 64), (128, 128)])
def test_dq_coverage_accepts_covering_configs(block_m2, block_n1):
    tva._check_dq_coverage({"BLOCK_M2": block_m2, "BLOCK_N1": block_n1})


@needs_aiter
def test_env_override_that_breaks_coverage_is_rejected(gfx950, monkeypatch):
    """An operator re-sweeping via env knobs cannot reach a wrong-dq config unnoticed."""
    monkeypatch.setenv("PRIMUS_IDEOGRAM_ATTN_BWD_BLOCK_M2", "32")
    with pytest.raises(ValueError, match="silently wrong dq"):
        tva.bwd_config(256)


# --------------------------------------------------------------------------- #
# Config assembly: deltas over aiter's shipped base                            #
# --------------------------------------------------------------------------- #
@needs_aiter
def test_bwd_config_applies_only_the_deltas(gfx950):
    from aiter.ops.triton.attention.mha_onekernel_bwd import _get_config

    shipped = _get_config()["onekernel"]
    tuned = tva.bwd_config(256)["onekernel"]

    assert tuned["BLOCK_N1"] == 64
    assert tuned["num_stages"] == 2
    # Everything else is inherited, so an upstream retune of those fields reaches us.
    for field, value in shipped.items():
        if field not in ("BLOCK_N1", "num_stages"):
            assert tuned[field] == value, field


@needs_aiter
def test_fwd_config_applies_only_the_deltas(gfx950):
    from aiter.ops.triton._triton_kernels.attention.mha import _get_config

    shipped = _get_config(False, torch.bfloat16, has_pe=False)
    tuned = tva.fwd_config(256, torch.bfloat16)

    assert (tuned["BLOCK_N"], tuned["num_warps"], tuned["num_stages"]) == (16, 8, 2)
    for field, value in shipped.items():
        if field not in ("BLOCK_N", "num_warps", "num_stages"):
            assert tuned[field] == value, field


@needs_aiter
def test_aiter_shipped_config_is_not_mutated(gfx950):
    """Our override must not leak into aiter's cached dict and retune other callers."""
    from aiter.ops.triton.attention.mha_onekernel_bwd import _get_config

    before = dict(_get_config()["onekernel"])
    tva.bwd_config(256)
    tva.bwd_config(256)
    assert _get_config()["onekernel"] == before


@needs_aiter
@pytest.mark.parametrize("head_dim", [64, 128])
def test_untuned_head_dim_falls_through_to_aiter(gfx950, head_dim):
    from aiter.ops.triton.attention.mha_onekernel_bwd import _get_config

    assert tva.bwd_config(head_dim)["onekernel"] == _get_config()["onekernel"]


@needs_aiter
def test_untuned_arch_falls_through_to_aiter(monkeypatch):
    from aiter.ops.triton.attention.mha_onekernel_bwd import _get_config

    monkeypatch.setattr(tva, "_arch", lambda: "gfx942")
    assert tva.bwd_config(256)["onekernel"] == _get_config()["onekernel"]


@needs_aiter
def test_env_override_beats_the_table(gfx950, monkeypatch):
    monkeypatch.setenv("PRIMUS_IDEOGRAM_ATTN_BWD_STAGES", "3")
    monkeypatch.setenv("PRIMUS_IDEOGRAM_ATTN_FWD_BLOCK_N", "32")
    assert tva.bwd_config(256)["onekernel"]["num_stages"] == 3
    assert tva.fwd_config(256, torch.bfloat16)["BLOCK_N"] == 32


# --------------------------------------------------------------------------- #
# Selection: model.varlen_attn_impl reaches the hook, env wins, junk is loud    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def published(monkeypatch):
    """Publish/restore a params dict the way the trainer does before installing hooks."""
    from primus.backends.nemo_automodel import argument_builder

    def _publish(params):
        monkeypatch.setattr(argument_builder, "_PUBLISHED_PARAMS", params)

    return _publish


def test_impl_defaults_to_triton_when_nothing_is_published():
    from primus.backends.nemo_automodel.models.ideogram4.attention import (
        varlen_attn_impl,
    )

    assert varlen_attn_impl() == "triton"


@pytest.mark.parametrize("impl", ["triton", "ck"])
def test_impl_reads_the_model_preset(published, impl):
    from primus.backends.nemo_automodel.models.ideogram4.attention import (
        varlen_attn_impl,
    )

    published({"model": {"varlen_attn_impl": impl}})
    assert varlen_attn_impl() == impl


def test_env_overrides_the_model_preset(published, monkeypatch):
    from primus.backends.nemo_automodel.models.ideogram4.attention import (
        varlen_attn_impl,
    )

    published({"model": {"varlen_attn_impl": "ck"}})
    monkeypatch.setenv("PRIMUS_IDEOGRAM_VARLEN_ATTN_IMPL", "triton")
    assert varlen_attn_impl() == "triton"


def test_unknown_impl_is_rejected(published):
    from primus.backends.nemo_automodel.models.ideogram4.attention import (
        varlen_attn_impl,
    )

    published({"model": {"varlen_attn_impl": "cktile"}})
    with pytest.raises(ValueError, match="varlen_attn_impl"):
        varlen_attn_impl()


def test_shipped_preset_defaults_to_triton():
    """The preset ships Triton: -20% step time at 8 ranks, same memory, same loss."""
    from pathlib import Path

    import yaml

    import primus

    preset = Path(primus.__file__).parent / "configs/models/nemo_automodel/ideogram4.yaml"
    assert yaml.safe_load(preset.read_text())["model"]["varlen_attn_impl"] == "triton"


def test_tuned_shape_keeps_the_requested_triton(published, gfx950):
    from primus.backends.nemo_automodel.models.ideogram4.attention import (
        resolve_varlen_impl,
    )

    published({"model": {"varlen_attn_impl": "triton"}})
    assert resolve_varlen_impl() == "triton"


def test_untuned_arch_degrades_to_ck(published, monkeypatch):
    """Untuned Triton loses to CK, so an unswept GPU must not be left on it silently."""
    from primus.backends.nemo_automodel.models.ideogram4.attention import (
        resolve_varlen_impl,
    )

    monkeypatch.setattr(tva, "_arch", lambda: "gfx942")
    published({"model": {"varlen_attn_impl": "triton"}})
    assert resolve_varlen_impl() == "ck"


@pytest.mark.parametrize(
    "arch, head_dim, tuned",
    [("gfx950", 256, True), ("gfx950", 128, False), ("gfx942", 256, False)],
)
def test_is_tuned_reports_only_swept_pairs(monkeypatch, arch, head_dim, tuned):
    monkeypatch.setattr(tva, "_arch", lambda: arch)
    assert tva.is_tuned(head_dim) is tuned


# --------------------------------------------------------------------------- #
# Numerics: matches CK, and every query row gets a gradient                    #
# --------------------------------------------------------------------------- #
def _reference_varlen_attention(q, k, v, cu_seqlens):
    """Per-segment fp32 attention; the arbiter both kernels are checked against."""
    scale = q.shape[-1] ** -0.5
    bounds = cu_seqlens.tolist()
    chunks = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        qs, ks, vs = (t[start:end].transpose(0, 1).float() for t in (q, k, v))
        probs = torch.softmax(qs @ ks.transpose(-1, -2) * scale, dim=-1)
        chunks.append((probs @ vs).transpose(0, 1))
    return torch.cat(chunks, dim=0)


def _rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


@needs_gpu
@needs_aiter
def test_triton_matches_ck_and_covers_every_query_row():
    """Output and all three gradients agree with CK, and no dq row is left unwritten.

    An unwritten dq row is exactly zero, which a norm-based comparison can hide when it is a
    small fraction of the tensor -- so the coverage of dq is asserted directly.
    """
    from primus.backends.nemo_automodel.models.ideogram4.attention import (
        varlen_flash_attention,
    )

    torch.manual_seed(0)
    seglens = [128, 96, 288]  # ragged, and none a multiple of the largest block
    total, heads, head_dim = sum(seglens), 4, 256
    cu_seqlens = torch.tensor([0, *torch.tensor(seglens).cumsum(0).tolist()], dtype=torch.int32).cuda()

    base = [torch.randn(total, heads, head_dim, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    grad_out = torch.randn_like(base[0])

    results = {}
    for impl in ("ck", "triton"):
        q, k, v = (t.clone().detach().requires_grad_(True) for t in base)
        out = varlen_flash_attention(q, k, v, cu_seqlens.clone(), max(seglens), impl=impl)
        out.backward(grad_out)
        results[impl] = (out.detach(), q.grad, k.grad, v.grad)

    reference = _reference_varlen_attention(*base, cu_seqlens)
    for impl in ("ck", "triton"):
        assert _rel_err(results[impl][0], reference) < 2e-2, f"{impl} output disagrees with fp32"

    for name, ck_t, triton_t in zip(("out", "dq", "dk", "dv"), results["ck"], results["triton"]):
        assert _rel_err(triton_t, ck_t) < 2e-2, f"{name}: Triton disagrees with CK"

    dq = results["triton"][1]
    unwritten = (dq.abs().amax(dim=(1, 2)) == 0).sum().item()
    assert unwritten == 0, f"{unwritten} query rows have an all-zero dq (dq under-coverage)"


# --------------------------------------------------------------------------- #
# torch.compile: no graph break inside the block                               #
# --------------------------------------------------------------------------- #
@needs_gpu
@needs_aiter
def test_compiles_fullgraph_with_no_graph_breaks():
    """Ideogram-4 trains under per-layer ``torch.compile``, and a break inside the block
    splits the region FSDP2 registers its all-gather/reshard around -- a silent multi-rank
    desync. Dynamo cannot trace aiter's Triton launch path (``num_ctas``, ``triton._C``), so
    this passes only while the kernels stay wrapped as ``torch.library`` custom ops; as a
    plain ``autograd.Function`` the same block produced 57 breaks.

    The compiled unit has to extend PAST the attention call: with an autograd Function's
    output as the region's only output, inductor's partitioner fails on the CK arm too, so a
    bare-attention version of this test would fail for reasons unrelated to tracing.
    """
    from torch._dynamo.utils import counters

    from primus.backends.nemo_automodel.models.ideogram4.attention import (
        varlen_flash_attention,
    )

    torch.manual_seed(0)
    seglens = [64, 32, 96]
    total, heads, head_dim = sum(seglens), 2, 256
    cu = torch.tensor([0, *torch.tensor(seglens).cumsum(0).tolist()], dtype=torch.int32).cuda()

    class Block(torch.nn.Module):
        """Attention plus its output projection -- what per-layer compile wraps."""

        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(heads * head_dim, heads * head_dim, dtype=torch.bfloat16, device="cuda")

        def forward(self, q, k, v, cu_seqlens):
            out = varlen_flash_attention(q, k, v, cu_seqlens.clone(), max(seglens), impl="triton")
            return self.proj(out.flatten(1, 2))

    q, k, v = (
        torch.randn(total, heads, head_dim, device="cuda", dtype=torch.bfloat16).requires_grad_(True) for _ in range(3)
    )

    torch._dynamo.reset()
    counters.clear()
    out = torch.compile(Block(), fullgraph=True)(q, k, v, cu)
    out.backward(torch.randn_like(out))

    breaks = sum(counters["graph_break"].values())
    assert breaks == 0, f"{breaks} graph break(s): {sorted(counters['graph_break'])}"
    assert q.grad is not None and torch.isfinite(q.grad).all()
