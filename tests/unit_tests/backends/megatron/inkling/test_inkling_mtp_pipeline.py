###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling MTP under pipeline parallelism, on CPU without a process group:

* ``patch_inkling_mtp_embed_norm_grads``: its condition (Inkling + MTP + PP > 1), the one-time
  install over the three modules that hold ``finalize_model_grads``, and the wrapper order
  (the original first, then the embed_norm all-reduce);
* ``allreduce_mtp_embed_norm_grads``: a no-op without MTP or PP; otherwise the weight getter it
  hands Megatron's ``_allreduce_embedding_grad`` picks ``embed_norm.weight`` only on a chunk
  with ``pre_process`` or ``mtp_process``;
* ``data_loader_store._mtp_on_this_rank`` / ``get_batch_func``: an intermediate stage that runs
  MTP gets the batch, a stage without MTP keeps the ``(None,) * 5`` early return, and the cached
  TransformerConfig is rebuilt when ``args`` changes.

The distributed parts (the process groups, Megatron's all-reduce, the TP broadcast) are
replaced by recording stubs; the multi-rank behaviour is covered by the PP training gates."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "megatron.core.transformer.module", reason="megatron.core is not importable in this environment"
)

import importlib  # noqa: E402

import megatron.core.distributed as dist_module  # noqa: E402

# importlib, not ``import ...finalize_model_grads as m``: the package re-exports the function under
# the submodule's name, so the import statement would bind the function.
fmg_module = importlib.import_module("megatron.core.distributed.finalize_model_grads")

try:
    # megatron.training imports optional kernels (mamba_ssm) that, on some images, probe the GPU at
    # import time; such a host cannot import data_loader_store either.
    import megatron.training.training as training_module  # noqa: E402

    from primus.backends.megatron import data_loader_store  # noqa: E402
except RuntimeError as exc:  # pragma: no cover - environment
    pytest.skip(f"megatron.training is not importable here: {exc}", allow_module_level=True)

from primus.backends.megatron.core.models.inkling import inkling_model  # noqa: E402
from primus.backends.megatron.patches import inkling_mtp_patches  # noqa: E402
from primus.core.patches import PatchContext  # noqa: E402
from primus.core.patches.patch_registry import PatchRegistry  # noqa: E402


def _ctx(**params):
    base = dict(model_type="inkling", mtp_num_layers=2, pipeline_model_parallel_size=2)
    module_config = SimpleNamespace(params=SimpleNamespace(**{**base, **params}))
    return PatchContext(backend="megatron", phase="before_train", extra={"module_config": module_config})


@pytest.mark.parametrize(
    "params, applies",
    [
        ({}, True),
        ({"model_type": "gpt"}, False),
        ({"mtp_num_layers": None}, False),
        ({"mtp_num_layers": 0}, False),
        ({"pipeline_model_parallel_size": 1}, False),
    ],
)
def test_patch_condition(params, applies):
    patch = PatchRegistry.get(inkling_mtp_patches._PATCH_KEY)
    assert patch is not None and patch.phase == "before_train"
    assert bool(patch.condition(_ctx(**params))) is applies


def test_patch_installs_once_and_runs_original_first(monkeypatch):
    calls = []

    def original(model, *args, **kwargs):
        calls.append(("finalize", args, kwargs))
        return "out"

    for module in (training_module, dist_module, fmg_module):
        monkeypatch.setattr(module, "finalize_model_grads", original, raising=False)
    monkeypatch.setattr(
        inkling_model, "allreduce_mtp_embed_norm_grads", lambda model, config: calls.append(("embed_norm",))
    )
    monkeypatch.setattr("megatron.core.utils.get_model_config", lambda module: "cfg")

    inkling_mtp_patches.patch_inkling_mtp_embed_norm_grads(_ctx())
    wrapped = training_module.finalize_model_grads
    assert wrapped is not original and wrapped._inkling_mtp_wrapper
    for module in (dist_module, fmg_module):
        assert module.finalize_model_grads._inkling_mtp_wrapper
    # a second application (or a later patch run) does not wrap again
    inkling_mtp_patches.patch_inkling_mtp_embed_norm_grads(_ctx())
    assert training_module.finalize_model_grads is wrapped

    assert wrapped([object()], 7, num_tokens=3) == "out"
    assert calls == [("finalize", (7,), {"num_tokens": 3}), ("embed_norm",)]


def _chunk(pre_process, mtp_process, embed_norm=True):
    chunk = torch.nn.Module()
    chunk.pre_process, chunk.mtp_process = pre_process, mtp_process
    if embed_norm:
        chunk.embed_norm = torch.nn.LayerNorm(4)
    return chunk


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(mtp_num_layers=None, pipeline_model_parallel_size=2),
        SimpleNamespace(mtp_num_layers=2, pipeline_model_parallel_size=1),
    ],
)
def test_allreduce_is_noop_without_mtp_or_pp(monkeypatch, config):
    called = []
    monkeypatch.setattr(fmg_module, "_allreduce_embedding_grad", lambda *a, **k: called.append(a))
    inkling_model.allreduce_mtp_embed_norm_grads([_chunk(True, False)], config)
    assert called == []


def test_allreduce_weight_getter(monkeypatch):
    seen = {}

    def fake_allreduce(model, embd_group, pp_group, weight_getter, config=None):
        seen["model"], seen["config"] = model, config
        seen["getter"] = weight_getter

    from megatron.core import parallel_state

    monkeypatch.setattr(fmg_module, "_allreduce_embedding_grad", fake_allreduce)
    monkeypatch.setattr(parallel_state, "get_embedding_group", lambda check_initialized=False: "embd")
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_group", lambda: "pp")
    config = SimpleNamespace(mtp_num_layers=2, pipeline_model_parallel_size=2)
    model = [_chunk(True, False)]
    inkling_model.allreduce_mtp_embed_norm_grads(model, config)
    assert seen["model"] is model and seen["config"] is config
    getter = seen["getter"]
    first, mtp_stage, middle = _chunk(True, False), _chunk(False, True), _chunk(False, False)
    assert getter(first) is first.embed_norm.weight
    assert getter(mtp_stage) is mtp_stage.embed_norm.weight
    assert getter(middle) is None
    assert getter(_chunk(True, False, embed_norm=False)) is None


# ---------------- data_loader_store ----------------


@pytest.fixture
def loader_env(monkeypatch):
    """Stub Megatron's args / stage / TP-broadcast helpers; record what get_batch_func asks for."""
    env = SimpleNamespace(
        args=SimpleNamespace(mtp_num_layers=2, patch_zero_bubble=False, context_parallel_size=1),
        first_or_last=False,
        mtp_here=True,
        built=[],
        tp_calls=[],
    )
    monkeypatch.setattr(data_loader_store, "_mtp_config", None)
    monkeypatch.setattr(data_loader_store, "get_args", lambda: env.args)
    monkeypatch.setattr(data_loader_store, "is_first_or_last_pipeline_stage", lambda vp: env.first_or_last)

    def build(args):
        env.built.append(args)
        return SimpleNamespace(args=args)

    monkeypatch.setattr(data_loader_store, "core_transformer_config_from_args", build)
    monkeypatch.setattr(
        data_loader_store, "mtp_on_this_rank", lambda cfg, ignore_virtual, vp_stage: env.mtp_here
    )

    def tp_batch(it, mtp_on_this_rank=False):
        env.tp_calls.append(mtp_on_this_rank)
        return {k: torch.zeros(1, 4) for k in ("tokens", "labels", "loss_mask", "position_ids")} | {
            "attention_mask": None
        }

    monkeypatch.setattr(data_loader_store, "get_batch_on_this_tp_rank", tp_batch)
    monkeypatch.setattr(data_loader_store, "get_batch_on_this_cp_rank", lambda batch: batch)
    return env


def test_middle_stage_with_mtp_gets_the_batch(loader_env):
    out = data_loader_store.get_batch_func(iter(()), vp_stage=0)
    assert out[0] is not None and len(out) == 5
    assert loader_env.tp_calls == [True]


def test_middle_stage_without_mtp_returns_none(loader_env):
    loader_env.mtp_here = False
    assert data_loader_store.get_batch_func(iter(()), vp_stage=0) == (None,) * 5
    assert loader_env.tp_calls == []


def test_no_mtp_keeps_old_behaviour(loader_env):
    loader_env.args.mtp_num_layers = None
    assert data_loader_store._mtp_on_this_rank(None) is False
    assert loader_env.built == []  # no TransformerConfig is built without MTP
    loader_env.first_or_last = True
    data_loader_store.get_batch_func(iter(()), vp_stage=None)
    assert loader_env.tp_calls == [False]


def test_config_cache_rebuilt_when_args_change(loader_env):
    assert data_loader_store._mtp_on_this_rank(None)
    assert data_loader_store._mtp_on_this_rank(None)
    assert len(loader_env.built) == 1
    loader_env.args = SimpleNamespace(**vars(loader_env.args))
    assert data_loader_store._mtp_on_this_rank(None)
    assert loader_env.built == [loader_env.built[0], loader_env.args]
