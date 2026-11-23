###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Bit-fingerprint writer (``PRIMUS_DEBUG_CHECKSUM_FILE``): after the last
``train_step`` the patch writes one fingerprint per parameter, ``main_grad`` and optimizer state tensor.

Unit checks of the writer only (no training run): the fingerprint is deterministic and sensitive to a single bit
(including the sign of zero, for every element width); the file is written after the final step and not before,
covers parameters, main grads and optimizer state, is identical for identical state and differs after a one-bit
change of a parameter.
"""

import sys
import types

import pytest
import torch

from .helpers import requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"


def _patch_mod():
    from primus.backends.megatron.patches import overlap_patches

    return overlap_patches


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.int64, torch.uint8])
def test_fingerprint_is_deterministic_and_bit_sensitive(dtype):
    fp = _patch_mod()._fingerprint
    torch.manual_seed(0)
    t = (torch.randn(5000, device=DEV) * 4).to(dtype)
    assert fp(t) == fp(t.clone())
    u = t.clone()
    iu = u.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[u.element_size()])
    iu[1234] ^= 1  # flip the lowest bit of one element
    assert fp(u) != fp(t)
    if not torch.equal(t[10], t[11]):  # swapping two elements changes it (position weighted)
        s = t.clone()
        s[10], s[11] = t[11].clone(), t[10].clone()
        assert fp(s) != fp(t)


def test_fingerprint_distinguishes_signed_zero():
    fp = _patch_mod()._fingerprint
    a = torch.zeros(64, device=DEV, dtype=torch.bfloat16)
    b = a.clone()
    b[3] = -0.0
    assert fp(a) != fp(b)


def _fake_state():
    torch.manual_seed(0)
    lin = torch.nn.Linear(16, 8, bias=True, device=DEV, dtype=torch.float32)
    for p in lin.parameters():
        p.main_grad = torch.randn_like(p)
        p.grad = torch.randn_like(p)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    opt.step()
    return lin, types.SimpleNamespace(chained_optimizers=None, optimizer=opt)


def _install(monkeypatch, path):
    """Fresh fake train_step wrapped by the checksum patch (its step counter starts at zero)."""
    # Stand-ins for ``megatron.training`` / ``megatron.training.training`` (the writer only needs ``train_step``
    # and ``get_args``), so the unit test does not import the training stack.
    mtr = types.ModuleType("megatron.training")
    mtr.__path__ = []
    mt = types.ModuleType("megatron.training.training")
    mt.train_step = lambda *a, **k: "stepped"
    mtr.training = mt
    mtr.get_args = lambda: types.SimpleNamespace(train_iters=2)
    monkeypatch.setitem(sys.modules, "megatron.training", mtr)
    monkeypatch.setitem(sys.modules, "megatron.training.training", mt)
    # ``import megatron.training.training as mt`` resolves through the package attributes: if an earlier test
    # imported the real megatron.training, point the package attribute at the stand-in too.
    meg = sys.modules.get("megatron")
    if meg is not None:
        monkeypatch.setattr(meg, "training", mtr, raising=False)
    monkeypatch.setenv("PRIMUS_DEBUG_CHECKSUM_FILE", str(path))
    _patch_mod().patch_overlap_checksum(None)
    return mt


def test_checksum_file_written_after_last_step_only(monkeypatch, tmp_path):
    path = tmp_path / "ck.txt"
    mt = _install(monkeypatch, path)
    lin, opt = _fake_state()
    assert mt.train_step(None, None, [lin], opt) == "stepped"
    assert not path.exists(), "written before the last step"
    assert mt.train_step(None, None, [lin], opt) == "stepped"
    lines = path.read_text().strip().splitlines()
    assert sum(ln.startswith("param ") for ln in lines) == 2
    assert sum(ln.startswith("main_grad ") for ln in lines) == 2
    assert any(ln.startswith("opt0.") for ln in lines), "no optimizer state fingerprints"
    assert any(".exp_avg " in ln for ln in lines)


def test_checksum_file_is_reproducible_and_detects_one_bit(monkeypatch, tmp_path):
    lin, opt = _fake_state()

    def write(name):
        path = tmp_path / name
        mt = _install(monkeypatch, path)
        mt.train_step(None, None, [lin], opt)
        mt.train_step(None, None, [lin], opt)
        return path.read_text()

    first, again = write("a.txt"), write("b.txt")
    assert first == again
    lin.weight.data.view(torch.int32)[0, 0] ^= 1
    changed = write("c.txt")
    diff = [x for x, y in zip(first.splitlines(), changed.splitlines()) if x != y]
    assert diff and all(x.startswith(("param", "opt")) for x in diff)
