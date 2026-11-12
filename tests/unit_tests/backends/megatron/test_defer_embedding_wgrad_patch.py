###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for draining deferred output-layer wgrads before the bucket's grad sync."""

from types import SimpleNamespace

import pytest

from primus.backends.megatron.patches import defer_embedding_wgrad_patches as patch_mod


class _Module:
    """Stand-in for the last-stage GPTModel that owns the deferred wgrad buffers."""

    def __init__(self, num_microbatches=2):
        self.output_layer = SimpleNamespace(weight=object(), tp_group="tp")
        self.share_embeddings_and_output_weights = True
        self.grad_output_buffer = [object()] * num_microbatches
        self.embedding_activation_buffer = [object()] * num_microbatches


@pytest.fixture(autouse=True)
def _clear_pending():
    patch_mod._pending.clear()
    yield
    patch_mod._pending.clear()


@pytest.fixture
def events():
    return []


@pytest.fixture
def drain(events):
    def drain(config, embedding_module, is_last_stage, tp_group):
        events.append(("drain", tp_group))
        embedding_module.grad_output_buffer.clear()
        embedding_module.embedding_activation_buffer.clear()

    return drain


def _register(module, config="cfg"):
    clear = patch_mod.wrap_clear_embedding_activation_buffer(lambda config, model, is_last_stage: module)
    clear(config, [None], True)


def _start_grad_sync(events, drain):
    return patch_mod.wrap_start_grad_sync(lambda self: events.append(("sync", self.name)), drain)


class TestStartGradSync:
    def test_drains_before_syncing_the_output_layer_bucket(self, events, drain):
        module = _Module()
        _register(module)
        bucket = SimpleNamespace(name="output", params={module.output_layer.weight})

        _start_grad_sync(events, drain)(bucket)

        assert events == [("drain", "tp"), ("sync", "output")]
        assert module.grad_output_buffer == []

    def test_other_buckets_are_not_drained(self, events, drain):
        module = _Module()
        _register(module)
        bucket = SimpleNamespace(name="other", params={object()})

        _start_grad_sync(events, drain)(bucket)

        assert events == [("sync", "other")]
        assert len(module.grad_output_buffer) == 2

    def test_drains_once_per_step(self, events, drain):
        module = _Module()
        _register(module)
        bucket = SimpleNamespace(name="output", params={module.output_layer.weight})
        start_grad_sync = _start_grad_sync(events, drain)

        start_grad_sync(bucket)
        start_grad_sync(bucket)

        assert [e[0] for e in events] == ["drain", "sync", "sync"]

    def test_non_last_stage_registers_nothing(self):
        clear = patch_mod.wrap_clear_embedding_activation_buffer(lambda config, model, is_last_stage: None)
        clear("cfg", [None], False)
        assert patch_mod._pending == []

    def test_untied_output_layer(self, events, drain):
        module = _Module()
        module.share_embeddings_and_output_weights = False
        weight = object()
        module.shared_embedding_or_output_weight = lambda: weight
        _register(module)

        _start_grad_sync(events, drain)(SimpleNamespace(name="output", params={weight}))

        assert events[0][0] == "drain"


class TestFinishEmbeddingWgradCompute:
    @pytest.fixture
    def warnings(self, monkeypatch):
        warnings = []
        monkeypatch.setattr(patch_mod, "warning_rank_0", warnings.append)
        return warnings

    def test_skips_after_early_drain(self, events, drain, warnings):
        finish = patch_mod.wrap_finish_embedding_wgrad_compute(drain)
        module = _Module()
        module.grad_output_buffer.clear()

        finish("cfg", module, True, "tp")

        assert events == []
        assert warnings == []

    def test_drains_and_warns_once_when_bypassed(self, events, drain, warnings):
        # A start_grad_sync override skipped the early drain.
        finish = patch_mod.wrap_finish_embedding_wgrad_compute(drain)

        finish("cfg", _Module(), True, "tp")
        finish("cfg", _Module(), True, "tp")

        assert events == [("drain", "tp"), ("drain", "tp")]
        assert len(warnings) == 1

    def test_passes_non_last_stage_through(self, warnings):
        calls = []
        finish = patch_mod.wrap_finish_embedding_wgrad_compute(lambda *a: calls.append(a))

        finish("cfg", None, False, "tp")

        assert calls == [("cfg", None, False, "tp")]
        assert warnings == []


class TestCondition:
    @pytest.mark.parametrize(
        "overrides, enabled",
        [
            ({}, True),
            ({"overlap_grad_reduce": False}, False),
            ({"defer_embedding_wgrad_compute": False}, False),
            ({"pipeline_model_parallel_size": 1}, False),
            ({"use_megatron_fsdp": True}, False),
            ({"use_torch_fsdp2": True}, False),
        ],
    )
    def test_condition(self, monkeypatch, overrides, enabled):
        args = SimpleNamespace(
            defer_embedding_wgrad_compute=True,
            overlap_grad_reduce=True,
            pipeline_model_parallel_size=2,
            use_megatron_fsdp=False,
            use_torch_fsdp2=False,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        monkeypatch.setattr(patch_mod, "get_args", lambda ctx: args)
        assert bool(patch_mod._is_enabled(None)) is enabled
