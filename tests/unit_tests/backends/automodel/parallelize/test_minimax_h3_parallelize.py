###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit and contract tests for the MiniMax-H3 sidecar.

THE CLAIM WORTH DEFENDING: under FSDP2, MiniMax-H3's keep-in-FP32 modules (the time
embedder, the four patch projections, the rotary buffer) compute in FP32, in units of
their own, while every other parameter computes in the policy dtype, FP32 master
weights included. Without the sidecar the whole root is cast to BF16 and nothing
fails, so the only way to see the difference is to look at the units.

The contract tests run AutoModel's real ``fully_shard_by_dtype`` and default
``ModelParallelizer`` on a tiny real H3 transformer, with only torch's ``fully_shard``
recorded, so they also check that diffusers' names match the parameters they mean.
"""

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")

from primus.backends.automodel.models.minimax_h3 import parallelize as h3  # noqa: E402
from tests.unit_tests.backends.automodel.parallelize._support import (  # noqa: E402
    registered_patch,
    requires_automodel,
)

FP32_MODULES = {"proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out"}
# ``rope`` holds no parameters, only the FP32 rotary buffer, so it is pinned and becomes a
# unit of its own with nothing to shard.
FP32_UNITS = FP32_MODULES | {"rope"}
TINY = dict(
    num_attention_heads=2,
    attention_head_dim=96,
    hidden_size=64,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=128,
    text_dim=48,
    freq_dim=32,
    time_embed_hidden_dim=64,
    time_embed_dim=32,
    rope_freq_dim=16,
)


def tiny_h3(dtype=torch.float32):
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )

    torch.manual_seed(0)
    return MiniMaxH3Transformer3DModel(**TINY).to(dtype)


def bf16_policy():
    from torch.distributed.fsdp import MixedPrecisionPolicy

    return MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)


class TestNames:
    def test_they_are_diffusers_declaration(self):
        assert set(h3.fp32_compute_names()) == FP32_MODULES | {"rope"}

    def test_they_match_exactly_the_modules_diffusers_keeps_in_fp32(self):
        """The names are matched as substrings of parameter names, so this pins what they reach:
        exactly the five top-level modules, and nothing inside a transformer block."""
        model = tiny_h3()
        names = h3.fp32_compute_names()
        hit = {n.split(".")[0] for n, _ in model.named_parameters() if any(t in n for t in names)}
        assert hit == FP32_MODULES
        for block in model.transformer_blocks:
            assert not [n for n, _ in block.named_parameters() if any(t in n for t in names)]


@requires_automodel
class TestInstall:
    @pytest.fixture
    def registry(self, monkeypatch):
        from nemo_automodel._diffusers import parallelization

        monkeypatch.setattr(parallelization, "_PARALLELIZERS", dict(parallelization._PARALLELIZERS))
        return parallelization._PARALLELIZERS

    def test_it_registers_on_top_of_the_default(self, registry):
        from nemo_automodel.components.distributed import ModelParallelizer

        assert "MiniMaxH3Transformer3DModel" not in registry
        assert h3.install() is True
        sidecar = registry["MiniMaxH3Transformer3DModel"]
        assert isinstance(sidecar, ModelParallelizer)
        assert type(sidecar).__name__ == "MiniMaxH3Parallelizer"

    def test_it_is_idempotent(self, registry):
        h3.install()
        first = registry["MiniMaxH3Transformer3DModel"]
        h3.install()
        assert registry["MiniMaxH3Transformer3DModel"] is first

    def test_an_upstream_sidecar_becomes_the_base(self, registry):
        class Upstream(__import__("nemo_automodel.components.distributed", fromlist=["x"]).ModelParallelizer):
            pass

        registry["MiniMaxH3Transformer3DModel"] = Upstream()
        h3.install()
        assert isinstance(registry["MiniMaxH3Transformer3DModel"], Upstream)


@requires_automodel
class TestUnits:
    """AutoModel's real fully_shard_by_dtype, with torch's fully_shard recorded."""

    @pytest.fixture
    def recorded(self, monkeypatch):
        from nemo_automodel.components.distributed import parallelizer_utils

        units = []

        def record(module, **kwargs):
            units.append((module, kwargs))

        monkeypatch.setattr(parallelizer_utils, "fully_shard", record)
        return units

    @pytest.fixture
    def sidecar(self):
        from nemo_automodel.components.distributed import ModelParallelizer

        return h3.build(ModelParallelizer)()

    @staticmethod
    def _names(model, module):
        return {n for n, m in model.named_modules() if m is module}

    def test_fp32_master_weights_get_fp32_units_for_the_pinned_modules_only(self, sidecar, recorded):
        model = tiny_h3(torch.float32)
        sidecar._fully_shard_module(model, mesh=None, mp_policy=bf16_policy(), reshard_after_forward=False)
        by_name = {next(iter(self._names(model, m))): kw for m, kw in recorded}
        fp32_units = {name for name, kw in by_name.items() if kw["mp_policy"].param_dtype == torch.float32}
        assert fp32_units == FP32_UNITS
        # The root, holding everything else (embedders, refiner, blocks, output norm), computes in BF16.
        assert by_name[""]["mp_policy"].param_dtype == torch.bfloat16

    def test_without_the_sidecar_everything_is_one_bf16_unit(self, recorded, monkeypatch):
        """What the sidecar changes: the default shards the root once, FP32 modules included."""
        from nemo_automodel.components.distributed import (
            ModelParallelizer,
            parallelizer,
        )

        monkeypatch.setattr(parallelizer, "fully_shard", lambda module, **kw: recorded.append((module, kw)))
        model = tiny_h3(torch.float32)
        ModelParallelizer()._fully_shard_module(model, mesh=None, mp_policy=bf16_policy())
        assert len(recorded) == 1 and recorded[0][1]["mp_policy"].param_dtype == torch.bfloat16

    @pytest.fixture
    def upstream(self, monkeypatch):
        """Upstream's own primitive, which the sidecar falls back to, recorded separately."""
        from nemo_automodel.components.distributed import parallelizer

        calls = []
        monkeypatch.setattr(
            parallelizer, "fully_shard", lambda module, **kw: calls.append((module, kw)) or module
        )
        return calls

    def test_a_block_is_still_one_unit(self, sidecar, recorded, upstream):
        """Nothing in a transformer block is pinned, so it goes to upstream's primitive unchanged."""
        model = tiny_h3(torch.float32)
        block = model.transformer_blocks[0]
        sidecar._fully_shard_module(block, mesh=None, mp_policy=bf16_policy(), reshard_after_forward=True)
        assert recorded == [] and len(upstream) == 1
        module, kwargs = upstream[0]
        assert module is block
        assert kwargs["mp_policy"].param_dtype == torch.bfloat16
        assert kwargs["reshard_after_forward"] is True

    def test_a_model_stored_all_in_bf16_is_left_to_upstream(self, sidecar, recorded, upstream):
        """AutoModel's LoRA path casts the whole model to BF16, keep-in-FP32 modules included.
        Pinning them then would make FSDP2 cast them up only inside their forward, while the model
        picks their input's dtype from the sharded BF16 weight: a mixed-dtype matmul (seen on
        8x MI355X). So nothing is pinned and the model is sharded as without the sidecar."""
        model = tiny_h3(torch.bfloat16)
        sidecar._fully_shard_module(model, mesh=None, mp_policy=None, reshard_after_forward=False)
        assert recorded == []
        assert len(upstream) == 1 and upstream[0][0] is model

    def test_only_names_stored_in_fp32_are_pinned(self):
        model = tiny_h3(torch.bfloat16)
        model.time_embedder.to(torch.float32)
        assert h3.pinned_in_fp32_storage(model, h3.fp32_compute_names()) == ("time_embedder",)

    def test_mixed_storage_keeps_the_fp32_modules_fp32(self, sidecar, recorded):
        """A model loaded the diffusers way: BF16, with the keep-in-FP32 modules still in FP32."""
        from diffusers.models.transformers.transformer_minimax_h3 import (
            MiniMaxH3Transformer3DModel,
        )

        torch.manual_seed(0)
        model = MiniMaxH3Transformer3DModel(**TINY)
        for name, param in model.named_parameters():
            if name.split(".")[0] not in FP32_MODULES:
                param.data = param.data.to(torch.bfloat16)
        sidecar._fully_shard_module(model, mesh=None, mp_policy=None, reshard_after_forward=False)
        for module, kwargs in recorded:
            if module is model:
                continue
            dtypes = {p.dtype for p in module.parameters(recurse=True)}
            # A unit is storage-uniform: one dtype, or none for the buffer-only rotary module.
            assert len(dtypes) <= 1, (self._names(model, module), dtypes)
        units = {next(iter(self._names(model, m))) for m, _ in recorded}
        assert FP32_MODULES <= units and "" in units

    def test_the_root_keeps_its_arguments(self, sidecar, recorded):
        model = tiny_h3(torch.float32)
        sidecar._fully_shard_module(
            model, mesh="the-mesh", mp_policy=bf16_policy(), reshard_after_forward=False
        )
        root = [kw for m, kw in recorded if m is model][0]
        assert root["mesh"] == "the-mesh" and root["reshard_after_forward"] is False

    def test_an_unknown_argument_falls_back_to_upstream_and_says_so(self, sidecar, caplog, monkeypatch):
        from nemo_automodel.components.distributed import parallelizer

        calls = []
        monkeypatch.setattr(parallelizer, "fully_shard", lambda module, **kw: calls.append(kw) or module)
        with caplog.at_level("WARNING"):
            sidecar._fully_shard_module(
                tiny_h3(), mesh=None, mp_policy=bf16_policy(), shard_placement_fn=None
            )
        assert calls and "shard_placement_fn" in calls[0]
        assert any("does not forward" in r.getMessage() for r in caplog.records)


@requires_automodel
class TestThroughTheDefaultParallelizer:
    """The default ModelParallelizer's whole FSDP2 path, with the sidecar and fully_shard recorded."""

    def test_blocks_and_fp32_modules_are_their_own_units(self, monkeypatch):
        from nemo_automodel.components.distributed import (
            ModelParallelizer,
            parallelizer,
            parallelizer_utils,
        )

        units = []
        record = lambda module, **kw: units.append((module, kw)) or module  # noqa: E731
        monkeypatch.setattr(parallelizer_utils, "fully_shard", record)
        monkeypatch.setattr(parallelizer, "fully_shard", record)

        class Mesh:
            mesh_dim_names = ("dp_shard_cp",)

            def __getitem__(self, name):
                return types.SimpleNamespace(size=lambda: 1)

        mesh = Mesh()
        monkeypatch.setattr(parallelizer, "get_fsdp_dp_mesh", lambda *a, **kw: mesh)
        # A torch-version shim upstream applies before sharding; it patches FSDP internals that
        # differ between torch releases and is beside the point of this test.
        monkeypatch.setattr(parallelizer, "_patch_fsdp_accumulated_grad_guard", lambda: None)

        model = tiny_h3(torch.float32)
        sidecar = h3.build(ModelParallelizer)()
        sidecar._apply(
            model, mesh, mp_policy=bf16_policy(), activation_checkpointing=False, enable_fsdp2_prefetch=False
        )
        name_of = {id(m): n for n, m in model.named_modules()}
        names = [name_of[id(u)] for u, _ in units]  # in the order they were sharded
        assert {f"transformer_blocks.{i}" for i in range(TINY["num_layers"])} <= set(names)
        assert FP32_UNITS <= set(names)
        assert "" in names
        # The FP32 units come before the root that encloses them, as FSDP2 requires.
        assert names.index("") > max(names.index(n) for n in FP32_UNITS)


class TestPatch:
    def test_it_is_registered_after_the_pipeline_and_adapter(self):
        patch = registered_patch("automodel.models.minimax_h3.parallelize")
        assert patch.priority > registered_patch("automodel.models.minimax_h3.adapter").priority

    def test_it_is_gated_like_the_others(self):
        from primus.core.patches import PatchContext

        patch = registered_patch("automodel.models.minimax_h3.parallelize")

        def ctx(adapter_type):
            params = types.SimpleNamespace(flow_matching=types.SimpleNamespace(adapter_type=adapter_type))
            return PatchContext(
                backend="automodel",
                phase="before_train",
                extra={"module_config": types.SimpleNamespace(params=params)},
            )

        assert patch.condition(ctx("minimax_h3")) is True
        assert patch.condition(ctx("wan")) is False
        assert patch.condition(None) is True


class TestReleaseCachedMemory:
    """The load's cached memory is released once, after sharding, and only on a GPU."""

    @pytest.fixture
    def fake_cuda(self, monkeypatch):
        state = {"reserved": 124 * 2**30, "calls": []}
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: state["reserved"])
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 15 * 2**30)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: state["calls"].append("synchronize"))

        def empty_cache():
            state["calls"].append("empty_cache")
            state["reserved"] = 16 * 2**30

        monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
        return state

    def test_it_empties_the_cache_and_reports_what_it_released(self, fake_cuda, caplog):
        with caplog.at_level("INFO"):
            assert h3.release_cached_memory() == pytest.approx(108.0)
        assert fake_cuda["calls"] == ["synchronize", "empty_cache"]
        assert "released 108.0 GiB" in caplog.text

    def test_without_an_initialized_gpu_it_does_nothing(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("emptied without a GPU"))
        assert h3.release_cached_memory() == 0.0

    def test_parallelize_releases_after_the_sidecar_chain_returns(self, monkeypatch):
        order = []

        class Base:
            def parallelize(self, model, mesh_context, /):
                order.append("sharded")
                return model

        monkeypatch.setattr(h3, "release_cached_memory", lambda: order.append("released") or 0.0)
        no_ac = types.SimpleNamespace(activation_checkpointing=False, strategy_config=types.SimpleNamespace())
        model = object()
        assert h3.build(Base)().parallelize(model, no_ac) is model
        assert order == ["sharded", "released"]

    def test_the_class_keeps_its_name(self):
        assert h3.build(type("Base", (), {})).__name__ == "MiniMaxH3Parallelizer"


class TestReleasingCachedMemory:
    """After sharding, the sidecar returns the load's cached memory to the device, once."""

    def test_it_runs_once_after_the_base_has_sharded(self, monkeypatch):
        order = []

        class Base:
            def parallelize(self, model, mesh_context, /):
                order.append("sharded")
                return model

        monkeypatch.setattr(h3, "release_cached_memory", lambda: order.append("released") or 0.0)
        context = types.SimpleNamespace(
            activation_checkpointing=False, strategy_config=types.SimpleNamespace(activation_checkpointing=False)
        )
        sidecar = h3.build(Base)()
        model = object()
        assert sidecar.parallelize(model, context) is model
        assert order == ["sharded", "released"]

    def test_it_keeps_the_sidecar_name(self):
        class Base:
            pass

        assert h3.build(Base).__name__ == "MiniMaxH3Parallelizer"

    def test_it_does_nothing_without_a_gpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert h3.release_cached_memory() == 0.0

    def test_it_empties_the_cache_and_reports_what_it_released(self, monkeypatch):
        reserved = iter([130 * 2**30, 16 * 2**30, 16 * 2**30])
        calls = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: next(reserved))
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 15 * 2**30)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: calls.append("sync"))
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
        assert h3.release_cached_memory() == pytest.approx(114.0)
        assert calls == ["sync", "empty"]
