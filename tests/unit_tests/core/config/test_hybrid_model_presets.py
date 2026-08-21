import os
from pathlib import Path

import pytest
import yaml

from primus.core.config.preset_loader import PresetLoader
from primus.core.config.yaml_loader import parse_yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = REPO_ROOT / "primus" / "configs" / "models" / "megatron"
CONFIG_ROOT = REPO_ROOT / "examples" / "megatron" / "configs"

HYBRID_MODELS = [
    "zebra_mamba_1B_hybrid",
    "zebra_mamba_3B_hybrid",
    "zebra_mamba_8B_hybrid",
    "zebra_kda_1B_hybrid",
    "zebra_gdn_1B_hybrid",
    "zebra_mamba_300M_hybrid",
    "zebra_gdn_300M_hybrid",
]

PURE_MODELS = [
    "kda_1B_pure",
    "gdn_1B_pure",
    "kda_300M_pure",
    "gdn_300M_pure",
]

PRETRAIN_CONFIGS = [
    "MI300X/zebra_mamba_1B_hybrid-pretrain.yaml",
    "MI300X/zebra_kda_1B_hybrid-pretrain.yaml",
    "MI300X/zebra_gdn_1B_hybrid-pretrain.yaml",
    "MI300X/kda_1B_pure-pretrain.yaml",
    "MI300X/gdn_1B_pure-pretrain.yaml",
    "MI300X/kda_300M_pure-pretrain.yaml",
    "MI355X/zebra_mamba_1B_hybrid-pretrain.yaml",
    "MI355X/kda_1B_pure-pretrain.yaml",
]

DEPRECATED_SYMLINKS = [
    "examples/megatron/configs/MI300X/zebra_llama_1B-pretrain.yaml",
    "examples/megatron/configs/MI300X/zebra_llama_1B_kda_pure-pretrain.yaml",
    "examples/megatron/configs/MI300X/zebra_llama_1B_kda-pretrain.yaml",
    "primus/configs/models/megatron/zebra_llama_1B.yaml",
    "primus/configs/models/megatron/zebra_llama_1B_kda_pure.yaml",
]


@pytest.mark.parametrize("model_name", HYBRID_MODELS + PURE_MODELS)
def test_hybrid_and_pure_model_presets_load(model_name):
    cfg = PresetLoader.load(model_name, "megatron", config_type="models")
    assert cfg["model_type"] == "mamba"
    assert cfg["num_layers"] > 0
    assert cfg["hidden_size"] > 0


@pytest.mark.parametrize("model_name", HYBRID_MODELS)
def test_hybrid_models_use_mla_or_recurrent_stack(model_name):
    cfg = PresetLoader.load(model_name, "megatron", config_type="models")
    assert cfg.get("is_hybrid_model") is True
    if cfg.get("multi_latent_attention"):
        assert cfg.get("hybrid_attention_ratio", 0) > 0


@pytest.mark.parametrize("model_name", PURE_MODELS)
def test_pure_models_have_no_mla(model_name):
    cfg = PresetLoader.load(model_name, "megatron", config_type="models")
    assert cfg.get("hybrid_attention_ratio") == 0.0
    assert cfg.get("multi_latent_attention") is False


@pytest.mark.parametrize("rel_path", PRETRAIN_CONFIGS)
def test_pretrain_configs_reference_existing_models(rel_path):
    cfg_path = CONFIG_ROOT / rel_path
    assert cfg_path.exists(), f"missing pretrain config: {rel_path}"
    data = parse_yaml(str(cfg_path))
    model_yaml = data["modules"]["pre_trainer"]["model"]
    model_stem = model_yaml.removesuffix(".yaml")
    preset = PresetLoader.load(model_stem, "megatron", config_type="models")
    assert preset["model_type"] == "mamba"


@pytest.mark.parametrize("rel_path", DEPRECATED_SYMLINKS)
def test_deprecated_symlinks_resolve(rel_path):
    path = REPO_ROOT / rel_path
    assert path.exists(), f"missing deprecated symlink: {rel_path}"
    assert path.resolve().exists(), f"broken symlink: {rel_path} -> {os.readlink(path) if path.is_symlink() else path}"


def test_kda_1B_pure_inherits_mamba_base_defaults():
    cfg = PresetLoader.load("kda_1B_pure", "megatron", config_type="models")
    base = PresetLoader.load("mamba_base", "megatron", config_type="models")
    assert cfg["use_legacy_models"] == base["use_legacy_models"]
    assert cfg["attention_dropout"] == base["attention_dropout"]


def test_mi300x_kda_hybrid_batch_size():
    cfg_path = CONFIG_ROOT / "MI300X/zebra_kda_1B_hybrid-pretrain.yaml"
    overrides = parse_yaml(str(cfg_path))["modules"]["pre_trainer"]["overrides"]
    assert overrides["micro_batch_size"] == 8
    assert overrides["global_batch_size"] == 64
