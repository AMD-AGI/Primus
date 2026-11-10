###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the MiniMax-H3 training pipeline and the proxy-checkpoint tool.

THE CLAIMS WORTH DEFENDING:

  1. THE DIRECTORY THE TOOL WRITES LOADS THROUGH DIFFUSERS' OWN LOADER. AutoModel's
     fine-tune path, which is the only one that applies LoRA, calls
     ``DiffusionPipeline.from_pretrained``. Diffusers has no pipeline class for H3, so
     without the registered training pipeline the call fails; with it, the same directory
     serves full fine-tuning and LoRA.

  2. THE FP32 MODULES SURVIVE A BF16 LOAD. The transformer keeps its time embedder and
     patch projections in FP32 whatever the load dtype, and the sharding code has to
     know which they are.

  3. NOTHING IS SHADOWED. Registering the class must not replace one diffusers grew.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")

from primus.backends.automodel.models.minimax_h3 import (  # noqa: E402
    pipeline as pipeline_mod,
)

TINY = {
    "num_attention_heads": 2,
    "attention_head_dim": 96,
    "hidden_size": 64,
    "ffn_dim": 128,
    "text_dim": 48,
    "freq_dim": 32,
    "time_embed_hidden_dim": 64,
    "time_embed_dim": 32,
    "rope_freq_dim": 16,
    "num_refiner_layers": 1,
}
FP32_MODULES = {"proj_in", "proj_out", "audio_proj_in", "audio_proj_out", "time_embedder"}

TOOL = Path(__file__).resolve().parents[4] / "tools" / "automodel" / "make_minimax_h3_proxy_checkpoint.py"


@pytest.fixture(autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def unregistered():
    """diffusers without the class registered, restored afterwards: the attribute is process-global."""
    saved = getattr(diffusers, pipeline_mod.PIPELINE_CLASS_NAME, None)
    if saved is not None:
        delattr(diffusers, pipeline_mod.PIPELINE_CLASS_NAME)
    yield
    if getattr(diffusers, pipeline_mod.PIPELINE_CLASS_NAME, None) is not None:
        delattr(diffusers, pipeline_mod.PIPELINE_CLASS_NAME)
    if saved is not None:
        setattr(diffusers, pipeline_mod.PIPELINE_CLASS_NAME, saved)


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("make_minimax_h3_proxy_checkpoint", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def proxy(tool, tmp_path):
    out = tmp_path / "proxy"
    rc = tool.main(
        ["--out", str(out), "--num-layers", "2", "--overrides", json.dumps(TINY), "--dtype", "float32"]
    )
    assert rc == 0
    return out


class TestRegistration:
    def test_it_is_not_there_until_installed(self, unregistered):
        assert not hasattr(diffusers, pipeline_mod.PIPELINE_CLASS_NAME)
        assert pipeline_mod.install() is True
        assert getattr(diffusers, pipeline_mod.PIPELINE_CLASS_NAME) is pipeline_mod.get_pipeline_class()

    def test_it_is_idempotent(self, unregistered):
        pipeline_mod.install()
        pipeline_mod.install()
        assert getattr(diffusers, pipeline_mod.PIPELINE_CLASS_NAME) is pipeline_mod.get_pipeline_class()

    def test_a_class_diffusers_defines_is_never_shadowed(self, unregistered):
        class Theirs:
            pass

        setattr(diffusers, pipeline_mod.PIPELINE_CLASS_NAME, Theirs)
        with pytest.raises(RuntimeError, match="would shadow"):
            pipeline_mod.install()
        assert getattr(diffusers, pipeline_mod.PIPELINE_CLASS_NAME) is Theirs

    def test_a_diffusers_without_h3_is_refused_naming_its_version(self, monkeypatch):
        """The failure it replaces comes much later, in the recipe's model build, and says nothing about the version."""
        old = types.ModuleType("diffusers")
        old.__version__ = "0.38.0"
        monkeypatch.setitem(sys.modules, "diffusers", old)
        with pytest.raises(RuntimeError, match=r"0\.38\.0.*0\.40"):
            pipeline_mod.install()

    def test_the_class_is_cached_and_named_as_the_index_says(self):
        cls = pipeline_mod.get_pipeline_class()
        assert pipeline_mod.get_pipeline_class() is cls
        assert cls.__name__ == pipeline_mod.PIPELINE_CLASS_NAME

    def test_the_model_index_names_the_class_and_the_transformer(self):
        index = pipeline_mod.model_index("0.40.0")
        assert index["_class_name"] == "MiniMaxH3TrainingPipeline"
        assert index["transformer"] == ["diffusers", "MiniMaxH3Transformer3DModel"]


class TestLoading:
    def test_the_directory_loads_through_diffusers(self, unregistered, proxy):
        from diffusers import DiffusionPipeline

        pipeline_mod.install()
        pipe = DiffusionPipeline.from_pretrained(str(proxy), torch_dtype=torch.float32)
        assert type(pipe).__name__ == "MiniMaxH3TrainingPipeline"
        assert type(pipe.transformer).__name__ == "MiniMaxH3Transformer3DModel"
        assert list(pipe.components) == ["transformer"]

    def test_it_does_not_load_without_the_registration(self, unregistered, proxy):
        """The failure the pipeline exists to prevent."""
        from diffusers import DiffusionPipeline

        with pytest.raises(AttributeError):
            DiffusionPipeline.from_pretrained(str(proxy), torch_dtype=torch.float32)

    def test_the_weights_are_the_seeds_and_survive_the_round_trip(self, unregistered, proxy):
        """The tool seeds torch before building, so the loaded weights equal a model built the same way."""
        from diffusers import DiffusionPipeline, MiniMaxH3Transformer3DModel

        pipeline_mod.install()
        torch.manual_seed(1234)
        expected = MiniMaxH3Transformer3DModel(num_layers=2, **TINY)
        pipe = DiffusionPipeline.from_pretrained(str(proxy), torch_dtype=torch.float32)
        loaded = dict(pipe.transformer.named_parameters())
        for name, param in expected.named_parameters():
            assert torch.equal(param, loaded[name]), name

    def test_a_bf16_load_keeps_exactly_the_fp32_modules(self, unregistered, proxy):
        """The sharding code has to put these in their own units."""
        from diffusers import DiffusionPipeline

        pipeline_mod.install()
        pipe = DiffusionPipeline.from_pretrained(str(proxy), torch_dtype=torch.bfloat16)
        fp32 = {n.split(".")[0] for n, p in pipe.transformer.named_parameters() if p.dtype == torch.float32}
        bf16 = {n.split(".")[0] for n, p in pipe.transformer.named_parameters() if p.dtype == torch.bfloat16}
        assert fp32 == FP32_MODULES
        assert not fp32 & bf16


class TestTool:
    def test_it_writes_what_the_loader_reads(self, proxy):
        assert (proxy / "model_index.json").is_file()
        assert (proxy / "transformer" / "config.json").is_file()
        assert (proxy / "transformer" / "diffusion_pytorch_model.safetensors").is_file()
        assert (
            json.loads((proxy / "model_index.json").read_text())["_class_name"] == "MiniMaxH3TrainingPipeline"
        )

    def test_it_records_how_the_proxy_was_made(self, proxy):
        info = json.loads((proxy / "PROXY.json").read_text())
        assert info["seed"] == 1234 and info["weights"] == "float32"
        assert info["num_parameters"] > 0
        assert "not MiniMax's weights" in info["what"]

    def test_the_same_seed_gives_the_same_weights(self, tool, tmp_path):
        outs = []
        for name in ("a", "b"):
            out = tmp_path / name
            assert tool.main(["--out", str(out), "--num-layers", "1", "--overrides", json.dumps(TINY)]) == 0
            outs.append((out / "transformer" / "diffusion_pytorch_model.safetensors").read_bytes())
        assert outs[0] == outs[1]

    def test_another_seed_gives_other_weights(self, tool, tmp_path):
        blobs = []
        for seed in (1, 2):
            out = tmp_path / str(seed)
            assert (
                tool.main(
                    [
                        "--out",
                        str(out),
                        "--num-layers",
                        "1",
                        "--overrides",
                        json.dumps(TINY),
                        "--seed",
                        str(seed),
                    ]
                )
                == 0
            )
            blobs.append((out / "transformer" / "diffusion_pytorch_model.safetensors").read_bytes())
        assert blobs[0] != blobs[1]

    def test_config_only_writes_no_weights(self, tool, tmp_path):
        out = tmp_path / "cfg"
        assert (
            tool.main(
                ["--out", str(out), "--config-only", "--num-layers", "2", "--overrides", json.dumps(TINY)]
            )
            == 0
        )
        assert (out / "transformer" / "config.json").is_file()
        assert not list((out / "transformer").glob("*.safetensors"))
        assert json.loads((out / "PROXY.json").read_text())["weights"] is None

    def test_the_default_config_is_the_released_models_size(self, tool, tmp_path, capsys):
        """33B parameters, checked on the meta device, so no storage is allocated."""
        out = tmp_path / "full"
        assert tool.main(["--out", str(out), "--config-only"]) == 0
        info = json.loads((out / "PROXY.json").read_text())
        assert 33.0e9 < info["num_parameters"] < 33.3e9
        assert json.loads((out / "transformer" / "config.json").read_text())["num_layers"] == 50

    def test_the_layer_count_is_applied(self, tool, tmp_path):
        out = tmp_path / "cfg"
        assert tool.main(["--out", str(out), "--config-only", "--num-layers", "4"]) == 0
        assert json.loads((out / "transformer" / "config.json").read_text())["num_layers"] == 4

    @pytest.mark.parametrize(
        "args",
        [["--overrides", "not json"], ["--overrides", "[1, 2]"], ["--num-layers", "0"]],
    )
    def test_bad_arguments_are_refused_with_a_message(self, tool, tmp_path, capsys, args):
        assert tool.main(["--out", str(tmp_path / "x"), "--config-only", *args]) == 2
        assert "make_minimax_h3_proxy_checkpoint" in capsys.readouterr().err

    def test_it_runs_from_a_checkout_without_primus_installed(self):
        """The tool puts the repo root on the path itself."""
        source = TOOL.read_text()
        assert "sys.path.insert(0, str(Path(__file__).resolve().parents[2]))" in source
