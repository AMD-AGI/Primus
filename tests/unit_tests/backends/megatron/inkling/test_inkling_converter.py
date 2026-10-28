###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``tools/checkpoint_conversion/convert_inkling_hf_megatron.py`` end to end on a synthetic checkpoint."""

from __future__ import annotations

import json

import pytest
import torch

from primus.backends.megatron.core.models.inkling.inkling_builders import (
    build_inkling_model,
    inkling_config_from_model_yaml,
)
from tests.unit_tests.backends.megatron.inkling.inkling_test_utils import (
    load_converter,
    preset,
    random_release_state,
    release_text_config,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="TE modules need a GPU")

VOCAB = 640  # checkpoint rows; the test preset sets unpadded_vocab_size: 600, padded by 128 to 640


@pytest.fixture
def tiny_checkpoint(tmp_path, init_parallel_state):
    from safetensors.torch import save_file

    config = inkling_config_from_model_yaml(preset("inkling_small_tiny"), unpadded_vocab_size=600)
    release = random_release_state(config, vocab=VOCAB, dtype=torch.bfloat16)
    release["model.mtp.layers.0.embed_norm.weight"] = torch.ones(config.hidden_size, dtype=torch.bfloat16)
    release["model.visual.final_norm.weight"] = torch.ones(8, dtype=torch.bfloat16)

    names = sorted(release)
    shards = {"model-00001-of-00002.safetensors": names[::2], "model-00002-of-00002.safetensors": names[1::2]}
    for shard, keys in shards.items():
        save_file({k: release[k].contiguous() for k in keys}, str(tmp_path / shard))
    index = {"weight_map": {k: shard for shard, keys in shards.items() for k in keys}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "config.json").write_text(json.dumps({"text_config": release_text_config(config, VOCAB)}))

    yaml = tmp_path / "inkling_tiny_test.yaml"
    yaml.write_text(f"extends:\n  - {preset('inkling_small_tiny')}\nunpadded_vocab_size: 600\n")
    return tmp_path, yaml, config


def test_dry_run(tiny_checkpoint, capsys):
    hf_dir, yaml, _ = tiny_checkpoint
    assert load_converter().main(["--hf-dir", str(hf_dir), "--model-yaml", str(yaml), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "skipped 2 non-text tensors" in out and "dry run OK" in out


def test_convert_and_load(tiny_checkpoint):
    hf_dir, yaml, config = tiny_checkpoint
    out_dir = hf_dir / "megatron"
    converter = load_converter()
    assert converter.main(["--hf-dir", str(hf_dir), "--model-yaml", str(yaml), "--output", str(out_dir)]) == 0
    assert (out_dir / "latest_checkpointed_iteration.txt").read_text() == "release"

    ckpt = torch.load(out_dir / "release" / "mp_rank_00" / "model_optim_rng.pt", weights_only=False)
    model = build_inkling_model(config, vocab_size=VOCAB, max_sequence_length=4096)
    model.load_state_dict(ckpt["model"], strict=True)
    assert ckpt["model"]["decoder.layers.1.mlp.router.expert_bias"].dtype == torch.float32


@pytest.mark.parametrize(
    "flags, rows",
    [([], 640), (["--make-vocab-size-divisible-by", "8"], 600), (["--tensor-model-parallel-size", "2"], 768)],
    ids=["default", "trim", "pad"],
)
def test_vocab_follows_training_padding(tiny_checkpoint, flags, rows):
    hf_dir, yaml, _ = tiny_checkpoint
    out_dir = hf_dir / "megatron"
    argv = ["--hf-dir", str(hf_dir), "--model-yaml", str(yaml), "--output", str(out_dir), *flags]
    assert load_converter().main(argv) == 0
    ckpt = torch.load(out_dir / "release" / "mp_rank_00" / "model_optim_rng.pt", weights_only=False)
    for key in ("embedding.word_embeddings.weight", "output_layer.weight"):
        assert ckpt["model"][key].shape[0] == rows, key


def test_preset_mismatch_is_rejected(tiny_checkpoint):
    hf_dir, yaml, _ = tiny_checkpoint
    cfg = json.loads((hf_dir / "config.json").read_text())
    cfg["text_config"]["route_scale"] = 2.5
    (hf_dir / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="moe_router_topk_scaling_factor"):
        load_converter().main(["--hf-dir", str(hf_dir), "--model-yaml", str(yaml), "--dry-run"])
