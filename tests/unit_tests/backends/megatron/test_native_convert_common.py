###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared native-converter helpers (native_convert_common): used by the Llama, Qwen3, DeepSeek and
OLMo-3 converters."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _unique(rows, hidden, base):
    """Every element distinct: element (r, c) is base + r * 1000 + c, so a reshape or transpose
    along either axis shows."""
    r = torch.arange(rows, dtype=torch.float64).unsqueeze(1)
    c = torch.arange(hidden, dtype=torch.float64).unsqueeze(0)
    return base + r * 1000 + c


@pytest.mark.parametrize("nh, ng, hdim, hidden", [(8, 2, 4, 16), (4, 4, 8, 32), (6, 1, 2, 12), (8, 2, 4, 24)])
def test_fuse_qkv_golden_row_order(nh, ng, hdim, hidden):
    """Golden layout, written row by row (no reshape in the oracle): per query group g, the rows of
    its nh / ng query heads, then key head g, then value head g; each row is copied unchanged."""
    from primus.backends.megatron.checkpoint.native_convert_common import fuse_qkv

    q = _unique(nh * hdim, hidden, 1e7)
    k = _unique(ng * hdim, hidden, 2e7)
    v = _unique(ng * hdim, hidden, 3e7)
    fused = fuse_qkv(q, k, v, heads=nh, groups=ng, head_dim=hdim)
    per = nh // ng
    golden = []
    for g in range(ng):
        golden += [q[r] for r in range(g * per * hdim, (g + 1) * per * hdim)]
        golden += [k[r] for r in range(g * hdim, (g + 1) * hdim)]
        golden += [v[r] for r in range(g * hdim, (g + 1) * hdim)]
    assert torch.equal(fused, torch.stack(golden))


def test_fuse_qkv_is_the_inkling_mapping_helper(monkeypatch):
    """One implementation (core/models/common/qkv_layout.py), used by the converters and by
    inkling_hf_mapping; it round-trips through inkling's split_qkv."""
    import importlib.util
    import sys

    from primus.backends.megatron.checkpoint import native_convert_common as common

    # loaded by file path: the inkling package __init__ imports megatron.training (needs a GPU here)
    path = Path(common.__file__).resolve().parents[1] / "core/models/inkling/inkling_hf_mapping.py"
    spec = importlib.util.spec_from_file_location("_inkling_hf_mapping_by_path", path)
    mapping = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mapping)  # its dataclasses look the module up
    spec.loader.exec_module(mapping)
    from primus.backends.megatron.core.models.common.qkv_layout import (
        fuse_qkv as layout_fuse_qkv,
    )

    assert mapping.fuse_qkv is common.fuse_qkv is layout_fuse_qkv  # one implementation
    q, k, v = _unique(12, 5, 0), _unique(6, 5, 1e6), _unique(6, 5, 2e6)
    fused = mapping.fuse_qkv(q, k, v, heads=4, groups=2, head_dim=3)
    q2, k2, v2 = mapping.split_qkv(fused, heads=4, groups=2, head_dim=3)  # inkling's inverse
    assert torch.equal(q2, q) and torch.equal(k2, k) and torch.equal(v2, v)


@pytest.mark.parametrize("nh, ng", [(6, 4), (4, 0), (0, 2), (4, -2), (8, 3)])
def test_fuse_qkv_refuses_head_counts(nh, ng):
    from primus.backends.megatron.checkpoint.native_convert_common import fuse_qkv

    q, k, v = _unique(8, 8, 0), _unique(4, 8, 0), _unique(4, 8, 0)
    with pytest.raises(ValueError, match="positive multiple"):
        fuse_qkv(q, k, v, heads=nh, groups=ng, head_dim=2)


def test_fuse_qkv_refuses_a_transposed_or_mis_sized_weight():
    """Non-square GQA weights (q [32, 24], k / v [8, 24]): a transposed or wrongly sized weight is
    refused with the expected shape, not reshaped into scrambled rows."""
    from primus.backends.megatron.checkpoint.native_convert_common import fuse_qkv

    nh, ng, hdim, hidden = 8, 2, 4, 24
    q, k, v = _unique(nh * hdim, hidden, 0), _unique(ng * hdim, hidden, 0), _unique(ng * hdim, hidden, 0)
    fuse_qkv(q, k, v, heads=nh, groups=ng, head_dim=hdim)
    for bad in ((q.t(), k, v), (q, k.t(), v), (q, k, v.t()), (q[:, :-1], k, v), (q, k[:-1], v)):
        with pytest.raises(ValueError, match="expected"):
            fuse_qkv(*bad, heads=nh, groups=ng, head_dim=hdim)
    with pytest.raises(TypeError):
        fuse_qkv(q, k, v, nh, ng, hdim)  # keyword-only


def _drive_shared_init(
    monkeypatch,
    legacy_error=None,
    legacy_package=False,
    initialized=True,
    on_init=None,
    master_addr="0",
    world_size=1,
    master_port=29555,
    env=None,
):
    """Run native_convert_common.init_megatron_single_process on stand-in Megatron modules; the
    import of megatron.legacy raises ``legacy_error`` (None: megatron.legacy is absent)."""
    import builtins
    import sys
    from types import ModuleType, SimpleNamespace

    from primus.backends.megatron.checkpoint import native_convert_common as common

    calls = []
    mpu = ModuleType("megatron.core.mpu")
    mpu.initialize_model_parallel = lambda **kw: calls.append(("mpu", kw))
    core = ModuleType("megatron.core")
    core.mpu = mpu
    gv = ModuleType("megatron.training.global_vars")
    margs = SimpleNamespace(
        padded_vocab_size=128,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
    gv.set_global_variables = lambda a, build_tokenizer=True: calls.append(("set", build_tokenizer))
    gv.get_args = lambda: margs
    for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setenv("MASTER_ADDR", master_addr)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(common, "neutralize_nvte_attention_env", lambda: None)
    if "megatron" not in sys.modules:
        monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    training = ModuleType("megatron.training")
    training.global_vars = gv
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu)
    monkeypatch.setitem(sys.modules, "megatron.training.global_vars", gv)
    if legacy_package:  # megatron.legacy exists, its fused_kernels submodule does not
        legacy = ModuleType("megatron.legacy")
        legacy.__path__ = []
        monkeypatch.setitem(sys.modules, "megatron.legacy", legacy)
        monkeypatch.setitem(sys.modules, "megatron.legacy.fused_kernels", None)
    else:
        monkeypatch.setitem(sys.modules, "megatron.legacy", None)  # import raises ModuleNotFoundError
    if legacy_error is not None:
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name.startswith("megatron.legacy"):
                raise legacy_error
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(common.torch.distributed, "is_initialized", lambda: initialized)
    monkeypatch.setattr(common.torch.distributed, "get_world_size", lambda *a, **k: world_size)
    if on_init is not None:
        monkeypatch.setattr(common.torch.distributed, "init_process_group", lambda **kw: on_init(**kw))
    logs = []
    out = common.init_megatron_single_process(margs, master_port, log=logs.append)
    return out, margs, calls, logs


def test_shared_init_without_megatron_legacy(monkeypatch):
    """native_convert_common.init_megatron_single_process runs on a Megatron-LM tree without
    megatron.legacy (core 0.19+): it skips the legacy fused-kernel load instead of failing."""
    out, margs, calls, logs = _drive_shared_init(monkeypatch)
    assert out is margs
    assert ("set", True) in calls and any(c[0] == "mpu" for c in calls)
    assert any("not present; no legacy fused kernels to load" in m for m in logs)


def test_shared_init_reraises_a_missing_dependency_inside_megatron_legacy(monkeypatch):
    """Only the absence of megatron.legacy itself is tolerated; a missing module inside it is raised."""
    err = ModuleNotFoundError("No module named 'some_dep'", name="some_dep")
    with pytest.raises(ModuleNotFoundError) as info:
        _drive_shared_init(monkeypatch, legacy_error=err)
    assert info.value.name == "some_dep"


def test_shared_init_with_megatron_legacy_but_no_fused_kernels(monkeypatch):
    """megatron.legacy present without fused_kernels: skipped too, and the log names the module."""
    out, margs, _calls, logs = _drive_shared_init(monkeypatch, legacy_package=True)
    assert out is margs
    assert any("megatron.legacy.fused_kernels not present" in m for m in logs)


def test_shared_init_private_world_on_localhost(monkeypatch):
    """Without a process group, the one-rank world's store is on 127.0.0.1 whatever MASTER_ADDR the
    launcher exported; a busy port falls back to a free one bound by the store."""
    import socket

    from primus.backends.megatron.checkpoint import native_convert_common as common

    made, inits = [], []

    class FakeStore:
        def __init__(self, host, port, **kw):
            made.append((host, port))
            if port == busy:
                raise RuntimeError(
                    "The server socket has failed to listen on any local network address. port: x, useIpv6: 0, code: -98, name: EADDRINUSE, message: address already in use"
                )
            if port == broken:
                raise RuntimeError("permission denied")
            self.port = port or 40001

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        busy = s.getsockname()[1]
    broken = busy + 1 if busy < 65535 else busy - 1
    monkeypatch.setattr(common.torch.distributed, "TCPStore", FakeStore)
    store = common._private_store(busy, log=lambda *a: None)
    assert made == [("127.0.0.1", busy), ("127.0.0.1", 0)] and store.port == 40001
    made.clear()
    assert common._private_store(None).port == 40001 and made == [("127.0.0.1", 0)]
    # any other bind error is raised, not masked by a retry on another port
    made.clear()
    with pytest.raises(RuntimeError, match="permission denied"):
        common._private_store(broken)
    assert made == [("127.0.0.1", broken)]

    # through init_megatron_single_process: no world yet -> the private store is used
    out, margs, calls, logs = _drive_shared_init(
        monkeypatch,
        initialized=False,
        on_init=lambda **kw: inits.append(kw),
        master_addr="node0.example",
    )
    import os

    assert os.environ["MASTER_ADDR"] == "127.0.0.1" and os.environ["MASTER_PORT"] == "29555"
    assert len(inits) == 1 and isinstance(inits[0]["store"], FakeStore)


def test_shared_init_keeps_an_existing_world_env(monkeypatch):
    """An existing one-rank process group is kept and its rendezvous env vars are not touched (also
    with master_port None); a multi-rank world is refused."""
    import os

    live = {"MASTER_PORT": "29500", "RANK": "3", "WORLD_SIZE": "1", "LOCAL_RANK": "3"}
    out, margs, calls, logs = _drive_shared_init(
        monkeypatch, master_addr="node0.example", master_port=None, env=live
    )
    assert out is margs
    assert os.environ["MASTER_ADDR"] == "node0.example"
    assert {k: os.environ[k] for k in live} == live
    with pytest.raises(RuntimeError, match="one-rank world"):
        _drive_shared_init(monkeypatch, world_size=8)


def test_private_store_real_busy_port_falls_back():
    """A real TCPStore on a port another socket holds: EADDRINUSE -> a free port bound by the store."""
    import socket

    from primus.backends.megatron.checkpoint import native_convert_common as common

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy = s.getsockname()[1]
        logs = []
        store = common._private_store(busy, log=logs.append)
        assert store.port not in (0, busy) and any("in use" in m for m in logs)
