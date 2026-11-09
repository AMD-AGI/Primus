###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for Megatron-FSDP on the diffusion path (``distributed/megatron_fsdp``,
``distributed/mfsdp_average_in_collective``).

WHY THIS TEST EXISTS:
  Every failure here trains. Route an fsdp section past the patch and the run is
  FSDP2 under a config that says Megatron-FSDP. Forget the optimizer and ZeRO-1
  is DDP with extra steps. Let split dtypes through and the forward runs in fp32
  at half the speed. Remove the wrong backward hook and every gradient is zero.
  None of those raises, so the tests pin each one:

    - key translation, the dtype guard and the refusals (CP, LoRA, CPU offload,
      ZeRO-1 on the DDP path, a missing megatron-fsdp, an optimizer it cannot
      shard);
    - that a patch that fails to install stops the run instead of being logged;
    - that each patched AutoModel seam falls through for anything that is not
      Megatron-FSDP;
    - the hook removal, against a stand-in that registers hooks the way the
      library does, including the root gradient-reduction hook that must stay;
    - the average_in_collective rewrite, against a stub of the library method
      and against the installed one's source.

  The contract tests run against the pinned AutoModel and megatron-fsdp when they
  are importable. No GPU and no process group needed.
"""

import functools
import importlib.util
import inspect
import logging
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from primus.backends.automodel.distributed import megatron_fsdp as mod  # noqa: E402
from primus.backends.automodel.distributed import (  # noqa: E402
    mfsdp_average_in_collective as avg,
)

LOGGER = "primus.backends.automodel.distributed.megatron_fsdp"

requires_automodel = pytest.mark.skipif(
    importlib.util.find_spec("nemo_automodel") is None, reason="needs the pinned AutoModel"
)


_REAL_FIND_SPEC = importlib.util.find_spec


def _find_spec_with_megatron_fsdp(present):
    def find_spec(name, *args):
        if name == "megatron_fsdp":
            return object() if present else None
        return _REAL_FIND_SPEC(name, *args)

    return find_spec


def _ctx(**params):
    from primus.core.patches import PatchContext

    namespace = SimpleNamespace(**params)
    return PatchContext(
        backend="automodel",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=namespace), "backend_args": namespace},
    )


_UPSTREAM = "nemo_automodel.components.distributed.megatron_fsdp"


@pytest.fixture
def upstream(monkeypatch):
    """AutoModel's megatron_fsdp module, as far as these patches use it, so the tests run without AutoModel."""
    module = ModuleType(_UPSTREAM)

    class MegatronFSDP:
        pass

    def fully_shard_optimizer(model, optimizer):
        module.sharded.append(optimizer)
        return optimizer

    module.HAS_MEGATRON_FSDP = True
    module.MegatronFSDP = MegatronFSDP
    module.fully_shard_optimizer = fully_shard_optimizer
    module.sharded = []
    monkeypatch.setitem(sys.modules, _UPSTREAM, module)
    return module


# --------------------------------------------------------------------------- #
# Patch registration and its refusals                                         #
# --------------------------------------------------------------------------- #
class TestRegistration:
    def _patch(self, patch_id):
        from primus.core.patches.patch_registry import PatchRegistry

        patch = PatchRegistry.get(patch_id)
        assert patch is not None
        return patch

    @pytest.mark.parametrize("backend", [None, "fsdp2", ""])
    def test_another_backend_is_left_alone(self, backend):
        fsdp = SimpleNamespace(backend=backend, average_in_collective=True)

        assert self._patch("automodel.distributed.megatron_fsdp").condition(_ctx(fsdp=fsdp)) is False
        assert (
            self._patch("automodel.distributed.mfsdp_average_in_collective").condition(_ctx(fsdp=fsdp))
            is False
        )

    @pytest.mark.parametrize("backend", ["megatron_fsdp", "Megatron-FSDP", "mfsdp"])
    def test_megatron_fsdp_is_routed(self, monkeypatch, backend):
        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_with_megatron_fsdp(True))

        ctx = _ctx(fsdp=SimpleNamespace(backend=backend))
        assert self._patch("automodel.distributed.megatron_fsdp").condition(ctx) is True

    def test_a_missing_megatron_fsdp_stops_the_run(self, monkeypatch):
        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_with_megatron_fsdp(False))

        with pytest.raises(ImportError, match="megatron-fsdp"):
            self._patch("automodel.distributed.megatron_fsdp").condition(
                _ctx(fsdp=SimpleNamespace(backend="megatron_fsdp"))
            )

    def test_ideogram4_zero1_on_top_is_refused(self, monkeypatch, set_option):
        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_with_megatron_fsdp(True))
        set_option("primus_ideogram4.zero1", True)

        with pytest.raises(ValueError, match="cannot be combined"):
            self._patch("automodel.distributed.megatron_fsdp").condition(
                _ctx(fsdp=SimpleNamespace(backend="megatron_fsdp"))
            )

    @pytest.mark.parametrize(
        "value,expected",
        [(True, True), ("true", True), ("off", True), (False, False), ("false", False), (None, False)],
    )
    def test_the_average_in_collective_fix_follows_the_setting(self, monkeypatch, value, expected):
        """Read as AutoModel reads it: "off" reaches megatron-fsdp as a non-empty string, which is on."""
        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_with_megatron_fsdp(True))
        fsdp = SimpleNamespace(backend="megatron_fsdp", average_in_collective=value)

        patch = self._patch("automodel.distributed.mfsdp_average_in_collective")
        assert patch.condition(_ctx(fsdp=fsdp)) is expected
        assert patch.priority < self._patch("automodel.distributed.megatron_fsdp").priority

    def test_the_setting_is_declared(self, set_option):
        from primus.backends.automodel import options

        set_option("primus_mfsdp.no_param_reshard", True)

        assert options.undeclared() == []


class TestApply:
    """The registered patch bodies, whose exceptions the patch runner only logs."""

    def _apply(self, patch_id, **params):
        from primus.core.patches.patch_registry import PatchRegistry

        PatchRegistry.get(patch_id).handler(_ctx(**params))

    def test_every_patch_is_installed(self, monkeypatch, upstream):
        installed = []
        for name in (
            "_install_mesh_context_patch",
            "_install_parallelize_patches",
            "_install_optimizer_patch",
        ):
            monkeypatch.setattr(mod, name, functools.partial(installed.append, name))

        self._apply("automodel.distributed.megatron_fsdp")

        assert installed == [
            "_install_mesh_context_patch",
            "_install_parallelize_patches",
            "_install_optimizer_patch",
        ]

    def test_a_broken_megatron_fsdp_stops_the_run(self, upstream):
        upstream.HAS_MEGATRON_FSDP = False

        with pytest.raises(ImportError, match="megatron-fsdp"):
            mod.install()
        with pytest.raises(SystemExit, match="megatron-fsdp"):
            self._apply("automodel.distributed.megatron_fsdp")

    @pytest.mark.parametrize(
        "patch_id,target",
        [
            ("automodel.distributed.megatron_fsdp", mod),
            ("automodel.distributed.mfsdp_average_in_collective", avg),
        ],
    )
    def test_a_failed_install_stops_the_run(self, monkeypatch, patch_id, target):
        def install():
            raise AttributeError("AutoModel moved a seam")

        monkeypatch.setattr(target, "install", install)

        with pytest.raises(SystemExit, match="AutoModel moved a seam"):
            self._apply(patch_id)

    @pytest.mark.parametrize(
        "checkpoint,warned",
        [(None, True), (True, True), ("off", True), (False, False), ("false", False)],
    )
    def test_the_checkpoint_warning_follows_automodel(self, monkeypatch, caplog, checkpoint, warned):
        """AutoModel saves for "off": a non-empty string."""
        monkeypatch.setattr(mod, "install", lambda: None)
        params = {} if checkpoint is None else {"checkpoint": SimpleNamespace(enabled=checkpoint)}

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            self._apply("automodel.distributed.megatron_fsdp", **params)

        assert ("checkpoint saving is untested" in caplog.text) is warned


# --------------------------------------------------------------------------- #
# Patch A: translating the fsdp section                                       #
# --------------------------------------------------------------------------- #
# The fsdp section of primus/configs/modules/automodel/flux_pre_trainer.yaml with
# the Megatron-FSDP overrides of the example experiment on top, as the launcher's
# deep merge leaves it.
MERGED_FLUX_SECTION = {
    "tp_size": 1,
    "cp_size": 1,
    "pp_size": 1,
    "dp_replicate_size": 1,
    "defer_fsdp_grad_sync": True,
    "reshard_after_forward": False,
    "enable_fsdp2_prefetch": True,
    "fsdp2_backward_prefetch_depth": 3,
    "fsdp2_forward_prefetch_depth": 2,
    "reduce_dtype": "bfloat16",
    "enable_compile": True,
    "activation_checkpointing": False,
    "backend": "megatron_fsdp",
    "zero_dp_strategy": 1,
    "overlap_grad_reduce": True,
    "overlap_param_gather": True,
    "average_in_collective": True,
}


@requires_automodel
class TestStrategySection:
    def test_the_merged_flux_section_translates_and_filters(self):
        section, enable_compile, dropped = mod.strategy_section(MERGED_FLUX_SECTION)

        assert section["strategy"] == "megatron_fsdp"
        assert section["zero_dp_strategy"] == 1 and section["average_in_collective"] is True
        assert section["grad_reduce_in_fp32"] is False, "reduce_dtype bfloat16 means a bf16 reduction"
        assert enable_compile is True and "enable_compile" not in section
        assert dropped == [
            "defer_fsdp_grad_sync",
            "enable_fsdp2_prefetch",
            "fsdp2_backward_prefetch_depth",
            "fsdp2_forward_prefetch_depth",
            "reshard_after_forward",
        ]

    def test_automodel_accepts_the_section(self):
        from nemo_automodel.components.distributed.config import MegatronFSDPConfig
        from nemo_automodel.recipes._dist_utils import parse_distributed_section

        section, _, _ = mod.strategy_section(MERGED_FLUX_SECTION)
        parsed = parse_distributed_section(section)

        assert isinstance(parsed["strategy_config"], MegatronFSDPConfig)
        assert parsed["strategy_config"].zero_dp_strategy == 1
        assert parsed["strategy_config"].average_in_collective is True
        assert parsed["activation_checkpointing"] is False

    @pytest.mark.parametrize(
        "reduce_dtype,expected",
        [("float32", True), ("fp32", True), ("torch.float32", True), (None, True), ("bfloat16", False)],
    )
    def test_the_reduce_dtype_sets_the_reduction(self, reduce_dtype, expected):
        section, _, _ = mod.strategy_section({"reduce_dtype": reduce_dtype})

        assert section["grad_reduce_in_fp32"] is expected

    def test_without_a_reduce_dtype_the_reduction_is_fp32_as_on_fsdp2(self):
        section, _, _ = mod.strategy_section({})

        assert section["grad_reduce_in_fp32"] is True

    def test_an_explicit_grad_reduce_setting_wins_over_reduce_dtype(self):
        section, _, _ = mod.strategy_section({"reduce_dtype": "float32", "grad_reduce_in_fp32": False})

        assert section["grad_reduce_in_fp32"] is False

    def test_context_parallelism_is_refused(self):
        with pytest.raises(ValueError, match="cp_size > 1 is not supported"):
            mod.strategy_section({"cp_size": 2})

    def test_cpu_offload_is_refused(self):
        with pytest.raises(ValueError, match="fsdp.cpu_offload is not supported"):
            mod.strategy_section({"cpu_offload": True})

        section, _, dropped = mod.strategy_section({"cpu_offload": False})
        assert "cpu_offload" not in section and "cpu_offload" not in dropped


class TestDtypes:
    def test_bf16_storage_and_compute_pass(self):
        mod.check_dtypes(torch.bfloat16, torch.bfloat16)

    def test_split_dtypes_are_refused(self):
        with pytest.raises(ValueError, match="split storage/compute dtypes"):
            mod.check_dtypes(torch.float32, torch.bfloat16)

    def test_fp32_is_refused(self):
        with pytest.raises(ValueError, match="would train in fp32"):
            mod.check_dtypes(torch.float32, torch.float32)

    def test_fp16_is_refused(self):
        with pytest.raises(ValueError, match="bfloat16 only"):
            mod.check_dtypes(torch.float16, torch.float16)

    def test_the_mesh_context_refuses_before_it_imports_automodel(self, monkeypatch):
        """Fails if the check is dropped or moved after the imports, also where AutoModel is absent."""
        for name in (
            "nemo_automodel.components.distributed",
            "nemo_automodel.components.distributed.config",
            "nemo_automodel.recipes._dist_utils",
        ):
            monkeypatch.setitem(sys.modules, name, None)

        with pytest.raises(ValueError, match="would train in fp32"):
            mod.build_mesh_context(
                fsdp_options={"backend": "megatron_fsdp"},
                world_size=8,
                dtype=torch.float32,
                compute_dtype=None,
            )


# --------------------------------------------------------------------------- #
# Patch A-C wiring, against stand-ins for the AutoModel modules               #
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_train_module(monkeypatch):
    """The real recipe module with its mesh builder swapped for a recorder, restored afterwards."""
    import nemo_automodel.recipes.diffusion.train as module

    calls = []

    def _build_diffusion_mesh_context(
        *, fsdp_cfg, ddp_cfg, world_size, dtype, compute_dtype=None, lora_enabled
    ):
        calls.append(fsdp_cfg)
        return "fsdp2 mesh context"

    monkeypatch.setattr(module, "_build_diffusion_mesh_context", _build_diffusion_mesh_context)
    return module, calls


class _Section(dict):
    """The recipe hands ConfigNode sections, which convert with to_dict()."""

    def to_dict(self):
        return dict(self)


def _call_mesh_builder(module, fsdp_cfg, **overrides):
    kwargs = dict(fsdp_cfg=fsdp_cfg, ddp_cfg=None, world_size=8, dtype=torch.bfloat16)
    kwargs.update(compute_dtype=torch.bfloat16, lora_enabled=False)
    kwargs.update(overrides)
    return module._build_diffusion_mesh_context(**kwargs)


@requires_automodel
class TestMeshContextPatch:
    def test_an_fsdp2_section_falls_through(self, fake_train_module):
        module, calls = fake_train_module
        mod._install_mesh_context_patch()

        assert _call_mesh_builder(module, _Section(tp_size=1)) == "fsdp2 mesh context"
        assert _call_mesh_builder(module, None) == "fsdp2 mesh context"
        assert len(calls) == 2

    def test_a_megatron_fsdp_section_builds_its_own_context(self, fake_train_module, monkeypatch):
        module, calls = fake_train_module
        built = []
        monkeypatch.setattr(mod, "build_mesh_context", lambda **kwargs: built.append(kwargs) or "mfsdp")
        mod._install_mesh_context_patch()

        assert _call_mesh_builder(module, _Section(backend="megatron_fsdp", zero_dp_strategy=1)) == "mfsdp"
        assert calls == []
        assert built[0]["fsdp_options"] == {"backend": "megatron_fsdp", "zero_dp_strategy": 1}

    def test_lora_and_ddp_are_refused(self, fake_train_module):
        module, _ = fake_train_module
        mod._install_mesh_context_patch()
        section = _Section(backend="megatron_fsdp")

        with pytest.raises(ValueError, match="LoRA"):
            _call_mesh_builder(module, section, lora_enabled=True)
        with pytest.raises(ValueError, match="both 'fsdp' and 'ddp'"):
            _call_mesh_builder(module, section, ddp_cfg={})

    def test_the_patch_is_installed_once(self, fake_train_module):
        module, _ = fake_train_module
        mod._install_mesh_context_patch()
        patched = module._build_diffusion_mesh_context

        mod._install_mesh_context_patch()

        assert module._build_diffusion_mesh_context is patched


@pytest.fixture
def fake_pipeline(monkeypatch):
    """The real pipeline module with both seams swapped for recorders, restored afterwards."""
    import nemo_automodel._diffusers.auto_diffusion_pipeline as module
    import nemo_automodel.components.distributed.parallelizer as parallelizer

    calls = {"parallelize": [], "compile": [], "per_layer": []}

    def parallelize_model(model, mesh_context):
        calls["parallelize"].append(model)
        return SimpleNamespace(module=model)

    def compile_parallelized_model(model, mesh_context):
        calls["compile"].append(model)

    monkeypatch.setattr(module, "parallelize_model", parallelize_model)
    monkeypatch.setattr(module, "compile_parallelized_model", compile_parallelized_model)
    monkeypatch.setattr(parallelizer, "_apply_per_layer_compile", lambda m: calls["per_layer"].append(m))
    monkeypatch.setitem(mod._STATE, "enable_compile", False)
    return module, calls


@requires_automodel
class TestParallelizePatches:
    def _contexts(self):
        from nemo_automodel.components.distributed.config import (
            FSDP2Config,
            MegatronFSDPConfig,
        )

        return SimpleNamespace(strategy_config=FSDP2Config()), SimpleNamespace(
            strategy_config=MegatronFSDPConfig()
        )

    def test_fsdp2_falls_through_to_upstream(self, fake_pipeline):
        module, calls = fake_pipeline
        mod._install_parallelize_patches()
        fsdp2, _ = self._contexts()

        module.compile_parallelized_model("model", fsdp2)

        assert calls["compile"] == ["model"] and calls["per_layer"] == []

    @pytest.mark.parametrize("enable_compile", [False, True])
    def test_megatron_fsdp_compiles_the_inner_module_per_layer(self, fake_pipeline, enable_compile):
        module, calls = fake_pipeline
        mod._install_parallelize_patches()
        mod._STATE["enable_compile"] = enable_compile
        _, mfsdp = self._contexts()
        wrapped = module.parallelize_model("inner", mfsdp)

        module.compile_parallelized_model(wrapped, mfsdp)

        assert calls["compile"] == [], "upstream compiles FSDP2 models only"
        assert calls["per_layer"] == (["inner"] if enable_compile else [])

    def test_the_wrapped_model_is_inspected_after_parallelize(self, fake_pipeline, monkeypatch):
        module, _ = fake_pipeline
        seen = []
        monkeypatch.setattr(mod, "_after_wrap", seen.append)
        mod._install_parallelize_patches()
        fsdp2, mfsdp = self._contexts()

        module.parallelize_model("a", fsdp2)
        wrapped = module.parallelize_model("b", mfsdp)

        assert seen == [wrapped]


# --------------------------------------------------------------------------- #
# Patch C: the optimizer                                                      #
# --------------------------------------------------------------------------- #
class TestOptimizerSharding:
    def test_every_optimizer_of_a_megatron_fsdp_model_is_sharded(self, upstream):
        optimizers = [SimpleNamespace(), SimpleNamespace()]

        assert mod.shard_optimizers(upstream.MegatronFSDP(), optimizers) == optimizers
        assert upstream.sharded == optimizers

    def test_any_other_model_passes_through(self, upstream):
        optimizers = [SimpleNamespace()]

        assert mod.shard_optimizers(torch.nn.Linear(2, 2), optimizers) is optimizers
        assert upstream.sharded == []

    def test_an_optimizer_is_sharded_once(self, upstream):
        """A build that calls a wrapped super().build() gets back optimizers that are sharded already."""
        model, optimizer = upstream.MegatronFSDP(), SimpleNamespace()

        mod.shard_optimizers(model, mod.shard_optimizers(model, [optimizer]))

        assert upstream.sharded == [optimizer]

    def test_an_optimizer_megatron_fsdp_cannot_shard_is_refused(self, upstream):
        with pytest.raises(ValueError, match="supports_megatron_fsdp_sharding = False"):
            mod.shard_optimizers(upstream.MegatronFSDP(), [SimpleNamespace()], allow=False)

        assert mod.shard_optimizers(torch.nn.Linear(2, 2), [], allow=False) == []


@pytest.fixture
def optimizer_config_restored():
    """The real OptimizerConfig, with the subclass hook the optimizer patch adds taken off afterwards."""
    from nemo_automodel.components.optim.optimizer import OptimizerConfig

    keys = ("__init_subclass__", mod._SUBCLASS_HOOK)
    saved = {key: vars(OptimizerConfig)[key] for key in keys if key in vars(OptimizerConfig)}
    yield OptimizerConfig
    for key in keys:
        if key in saved:
            setattr(OptimizerConfig, key, saved[key])
        elif key in vars(OptimizerConfig):
            delattr(OptimizerConfig, key)


@requires_automodel
@pytest.mark.usefixtures("optimizer_config_restored")
class TestOptimizerPatch:
    def test_every_build_in_the_optimizer_config_hierarchy_is_wrapped(self, monkeypatch):
        classes = mod._optimizer_config_classes()
        assert classes, "the hierarchy has at least the base config"
        for cls in classes:
            monkeypatch.setattr(cls, "build", vars(cls)["build"])

        mod._install_optimizer_patch()

        assert all(getattr(vars(cls)["build"], mod._PATCHED, False) for cls in classes)

    @pytest.mark.parametrize("supported", [None, True, False])
    def test_a_wrapped_build_shards_what_it_built(self, monkeypatch, supported):
        class Config:
            def build(self, model, device_mesh=None, is_peft=False):
                return ["opt"]

        if supported is not None:
            Config.supports_megatron_fsdp_sharding = supported
        monkeypatch.setattr(mod, "_optimizer_config_classes", lambda: [Config])
        monkeypatch.setattr(
            mod, "shard_optimizers", lambda model, optimizers, allow: [(model, o, allow) for o in optimizers]
        )
        mod._install_optimizer_patch()

        assert Config().build("model", device_mesh=None) == [("model", "opt", supported is not False)]

    def test_a_config_defined_after_the_install_is_wrapped(self, monkeypatch, optimizer_config_restored):
        monkeypatch.setattr(mod, "_optimizer_config_classes", lambda: [])
        monkeypatch.setattr(
            mod, "shard_optimizers", lambda model, optimizers, allow: [(model, o) for o in optimizers]
        )
        mod._install_optimizer_patch()

        class LaterConfig(optimizer_config_restored):
            def build(self, model, device_mesh=None, is_peft=False):
                return ["opt"]

        assert LaterConfig.build(SimpleNamespace(), "model") == [("model", "opt")]


@requires_automodel
def test_the_seams_this_patches_still_exist():
    """Fails when AutoModel moves a function the patches rebind."""
    import nemo_automodel._diffusers.auto_diffusion_pipeline as pipeline
    import nemo_automodel.recipes.diffusion.train as train_mod
    from nemo_automodel.components.distributed.megatron_fsdp import (
        fully_shard_optimizer,
    )
    from nemo_automodel.components.distributed.parallelizer import (
        _apply_per_layer_compile,
    )
    from nemo_automodel.components.optim.optimizer import MuonConfig, OptimizerConfig

    params = inspect.signature(train_mod._build_diffusion_mesh_context).parameters
    assert {"fsdp_cfg", "ddp_cfg", "world_size", "dtype", "compute_dtype", "lora_enabled"} <= set(params)
    assert list(inspect.signature(pipeline.parallelize_model).parameters)[:2] == ["model", "mesh_context"]
    assert list(inspect.signature(pipeline.compile_parallelized_model).parameters)[:2] == [
        "model",
        "mesh_context",
    ]
    assert list(inspect.signature(fully_shard_optimizer).parameters)[:2] == ["model", "optimizer"]
    assert callable(_apply_per_layer_compile)
    assert OptimizerConfig.supports_megatron_fsdp_sharding is True
    assert MuonConfig.supports_megatron_fsdp_sharding is False


# --------------------------------------------------------------------------- #
# Patch D: dropping the release and re-gather hooks                           #
# --------------------------------------------------------------------------- #
def _root_pre_backward(wrapped, *args):
    """The handler that queues the gradient-reduction callback. Must survive."""
    return args


def _unshard_params(*args):
    """The handler that all-gathers the parameters again. Must not."""
    return args


def create_custom_backward_hook(custom_backward_handler):
    """Mirrors the library factory: the free-variable name is what the matcher looks for."""

    def hook(module, args, output):
        return custom_backward_handler(output)

    return hook


class FakeUnit:
    """Provides a ``_post_forward``, the name of the library's release hook."""

    def _post_forward(self, module, args, output):
        return output


def _unrelated_hook(module, args, output):
    return output


ROOT_ENTRY = " _root_pre_backward"


class FakeMegatronFSDP(torch.nn.Module):
    """Root wrapper carrying both hook families on its units, registered and recorded the way the library does.

    The release handles are recorded under each unit's name, the re-gather ones
    under its class name, so the three units share one entry. Unit 0 registers
    its release hook ``with_kwargs`` and its re-gather hook ``always_call``, so
    the side tables those add are covered too.
    """

    def __init__(self, strategy="optim", n_units=3, root_handler=_root_pre_backward):
        super().__init__()
        self.data_parallel_sharding_strategy = strategy
        self.module = torch.nn.Linear(2, 2)
        self.units = torch.nn.ModuleList(torch.nn.Identity() for _ in range(n_units))
        if root_handler is not None:
            self.pre_backward = functools.partial(root_handler, self)

        self.forward_hooks = {}
        self.backward_pre_hooks = {}
        for i, unit in enumerate(self.units):
            self.forward_hooks[f"release module units.{i} parameters"] = unit.register_forward_hook(
                FakeUnit()._post_forward, with_kwargs=(i == 0)
            )
            regather = create_custom_backward_hook(_unshard_params)
            self.backward_pre_hooks[f"all-gather {unit._get_name()} parameters"] = unit.register_forward_hook(
                torch.compiler.disable(regather) if i % 2 else regather,
                always_call=(i == 0),
            )

        self.backward_pre_hooks[ROOT_ENTRY] = self.register_forward_hook(
            torch.compiler.disable(create_custom_backward_hook(_root_pre_backward))
        )
        self.register_forward_hook(_unrelated_hook)
        self.forward_hooks["keep me"] = object()


def unit_hook_count(wrapped):
    return [len(unit._forward_hooks) for unit in wrapped.units]


def root_handlers(wrapped):
    return [mod._wrapped_backward_handler(fn) for fn in wrapped._forward_hooks.values()]


@pytest.mark.parametrize("strategy", ["optim", "optim_grads"])
def test_both_hook_families_are_dropped_below_zero3(caplog, strategy):
    wrapped = FakeMegatronFSDP(strategy=strategy)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        mod.drop_param_reshard_hooks(wrapped)

    assert unit_hook_count(wrapped) == [0, 0, 0]
    assert wrapped.units[0]._forward_hooks_with_kwargs == {}
    assert wrapped.units[0]._forward_hooks_always_called == {}
    assert list(wrapped.forward_hooks) == ["keep me"]
    assert list(wrapped.backward_pre_hooks) == [ROOT_ENTRY], "the root entry stays, as its hook does"
    assert "dropped 3 post-forward parameter releases and 3 backward re-gather hooks" in caplog.text
    assert "parameters are gathered once per step" in caplog.text


def test_a_post_forward_hook_megatron_fsdp_did_not_record_stays():
    """Release hooks go by the handles Megatron-FSDP recorded, not by their name."""
    wrapped = FakeMegatronFSDP()
    foreign = FakeUnit()._post_forward
    wrapped.units[1].register_forward_hook(foreign)

    mod.drop_param_reshard_hooks(wrapped)

    assert list(wrapped.units[1]._forward_hooks.values()) == [foreign]


def test_the_gradient_reduction_hook_is_never_dropped():
    """It shares its factory, qualname and shape with the re-gather hook; without it every gradient is zero."""
    wrapped = FakeMegatronFSDP()

    mod.drop_param_reshard_hooks(wrapped)

    assert _root_pre_backward in root_handlers(wrapped)
    assert _unrelated_hook in wrapped._forward_hooks.values()
    assert len(wrapped._forward_hooks) == 2


def test_zero3_keeps_everything(caplog):
    """At optim_grads_params the parameters really are sharded, so the pair is required."""
    wrapped = FakeMegatronFSDP(strategy="optim_grads_params")
    forward_hooks, backward_pre_hooks = dict(wrapped.forward_hooks), dict(wrapped.backward_pre_hooks)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        mod.drop_param_reshard_hooks(wrapped)

    assert unit_hook_count(wrapped) == [2, 2, 2]
    assert wrapped.forward_hooks == forward_hooks and wrapped.backward_pre_hooks == backward_pre_hooks
    assert "ignored under zero_dp_strategy 3" in caplog.text


def test_an_unidentifiable_root_handler_keeps_the_re_gather(caplog):
    """Without ``pre_backward`` the two are indistinguishable, so neither may go."""
    wrapped = FakeMegatronFSDP(root_handler=None)
    backward_pre_hooks = dict(wrapped.backward_pre_hooks)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        mod.drop_param_reshard_hooks(wrapped)

    assert unit_hook_count(wrapped) == [1, 1, 1], "releases go, re-gathers stay"
    assert all(_unshard_params in root_handlers(unit) for unit in wrapped.units)
    assert wrapped.backward_pre_hooks == backward_pre_hooks
    assert "cannot identify the root pre-backward handler" in caplog.text
    assert "the backward re-gather stays" in caplog.text
    assert "gathered once per step" not in caplog.text


@pytest.mark.parametrize("enabled,expected", [(False, [2, 2, 2]), (True, [0, 0, 0])])
def test_after_wrap_honours_the_setting(monkeypatch, set_option, enabled, expected):
    monkeypatch.setattr(mod, "is_megatron_fsdp_model", lambda model: True)
    set_option("primus_mfsdp.no_param_reshard", enabled)
    wrapped = FakeMegatronFSDP()

    mod._after_wrap(wrapped)

    assert unit_hook_count(wrapped) == expected


def test_an_all_fp32_model_is_reported(monkeypatch, caplog):
    monkeypatch.setattr(mod, "is_megatron_fsdp_model", lambda model: True)

    with caplog.at_level(logging.ERROR, logger=LOGGER):
        mod._after_wrap(FakeMegatronFSDP())

    assert "this run computes in fp32" in caplog.text


# --------------------------------------------------------------------------- #
# average_in_collective without an expert-parallel group                      #
# --------------------------------------------------------------------------- #
# The scaling-factor block of megatron-fsdp 0.5.0's
# MegatronFSDP._init_fsdp_param_and_grad_buffer, at its upstream indentation.
STUB_SOURCE = textwrap.dedent(
    """
    from types import SimpleNamespace


    class MegatronFSDP:
        def __init__(self, dist_index, average_in_collective=True):
            self.dist_index = dist_index
            self.calculate_per_token_loss = False
            self.ddp_config = SimpleNamespace(average_in_collective=average_in_collective)

        def _init_fsdp_param_and_grad_buffer(self):
            if self.calculate_per_token_loss:
                gradient_scaling_factor = None
                expert_gradient_scaling_factor = None
            else:
                if self.ddp_config.average_in_collective:
                    gradient_scaling_factor = 1.0
                    expert_gradient_scaling_factor = (
                        self.dist_index.get_dp_group(is_expert_parallel=True).size()
                        / self.dist_index.get_dp_group().size()
                    )
                else:
                    data_parallel_world_size = self.dist_index.get_dp_group().size()
                    gradient_scaling_factor = 1.0 / data_parallel_world_size
                    expert_gradient_scaling_factor = 1.0 / data_parallel_world_size
            self.factors = (gradient_scaling_factor, expert_gradient_scaling_factor)
    """
)


class _Group:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class _DistIndex:
    def __init__(self, dp, expert_dp):
        self._dp = _Group(dp)
        self._expert_dp = None if expert_dp is None else _Group(expert_dp)

    def get_dp_group(self, is_expert_parallel=False):
        return self._expert_dp if is_expert_parallel else self._dp


def _load(tmp_path, monkeypatch, source, name):
    path = tmp_path / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stub_cls(tmp_path, monkeypatch):
    return _load(tmp_path, monkeypatch, STUB_SOURCE, "mfsdp_stub").MegatronFSDP


def _factors(cls, dp, expert_dp, average_in_collective=True):
    instance = cls(_DistIndex(dp, expert_dp), average_in_collective)
    instance._init_fsdp_param_and_grad_buffer()
    return instance.factors


def test_upstream_crashes_without_an_expert_group(stub_cls):
    with pytest.raises(AttributeError, match="size"):
        _factors(stub_cls, dp=8, expert_dp=None)


def test_patched_falls_back_to_the_dp_group(stub_cls):
    assert avg.patch_class(stub_cls) is True
    assert _factors(stub_cls, dp=8, expert_dp=None) == (1.0, 1.0)


@pytest.mark.parametrize("expert_dp,average_in_collective", [(2, True), (8, True), (2, False), (None, False)])
def test_patched_is_unchanged_where_upstream_works(stub_cls, expert_dp, average_in_collective):
    want = _factors(stub_cls, 8, expert_dp, average_in_collective)
    avg.patch_class(stub_cls)
    assert _factors(stub_cls, 8, expert_dp, average_in_collective) == want


def test_the_rewrite_is_idempotent(stub_cls):
    avg.patch_class(stub_cls)
    patched = stub_cls._init_fsdp_param_and_grad_buffer
    assert avg.patch_class(stub_cls) is True
    assert stub_cls._init_fsdp_param_and_grad_buffer is patched


def test_a_fixed_upstream_is_left_alone(tmp_path, monkeypatch):
    fixed_source = STUB_SOURCE.replace(
        "self.dist_index.get_dp_group(is_expert_parallel=True).size()",
        "self.dist_index.get_expert_dp_size()",
    )
    cls = _load(tmp_path, monkeypatch, fixed_source, "mfsdp_fixed_stub").MegatronFSDP
    original = cls._init_fsdp_param_and_grad_buffer

    assert avg.patch_class(cls) is False
    assert cls._init_fsdp_param_and_grad_buffer is original


def test_the_anchor_matches_the_installed_megatron_fsdp():
    """Fails when the installed megatron-fsdp drifts away from the rewrite's anchor."""
    module = pytest.importorskip("megatron_fsdp.megatron_fsdp")
    if getattr(module.MegatronFSDP, avg._PATCH_FLAG, False):
        pytest.skip("MegatronFSDP was already patched in this process")
    source = inspect.getsource(module.MegatronFSDP.__dict__[avg._METHOD])

    assert source.count(avg.ANCHOR) == 1


def test_the_hook_layout_matches_the_installed_megatron_fsdp():
    """Hook removal relies on these names and records; fails when megatron-fsdp changes them."""
    module = pytest.importorskip("megatron_fsdp.megatron_fsdp")
    source = inspect.getsource(module.MegatronFSDP)

    assert "def _post_forward(" in source
    assert "def create_custom_backward_hook(module, custom_backward_handler)" in source
    assert "self.pre_backward = partial(_root_pre_backward" in source
    assert 'self.forward_hooks[f"release module {name} parameters"]' in source
    assert "module.register_forward_hook(_post_forward" in source, "the release handles are what is removed"
    assert 'self.backward_pre_hooks[f"all-gather {module._get_name()} parameters"]' in source
