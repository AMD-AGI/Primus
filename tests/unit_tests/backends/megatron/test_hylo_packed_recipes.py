###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from pathlib import Path

import pytest

from primus.core.config.primus_config import load_primus_config
from primus.core.utils import file_utils

CONFIGS = Path(__file__).resolve().parents[4] / "examples/megatron/configs/MI300X"

# Settings the packed recipes share with the SFT-path recipes: model, teacher, and optimizer.
SHARED = (
    "spec",
    "hybrid_override_pattern",
    "hybrid_attention_ratio",
    "hylo_distill_stage",
    "hylo_teacher_num_layers",
    "hylo_teacher_use_rope_scaling",
    "hylo_kl_weight",
    "hylo_ce_weight",
    "lr",
    "min_lr",
    "lr_decay_style",
    "seed",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
    "clip_grad",
    "micro_batch_size",
    "global_batch_size",
    "finetune",
    "no_load_optim",
    "no_load_rng",
)


@pytest.mark.parametrize("kind", ["m2", "mla"])
def test_packed_recipe_is_the_sft_recipe_on_the_reference_schedule(kind, monkeypatch):
    monkeypatch.setattr(file_utils, "create_path_if_not_exists", lambda *args, **kwargs: None)
    for name in (
        "HYLO_PACKED_TRAIN_DATA",
        "HYLO_ILD_TRAIN_ITERS",
        "HYLO_MICRO_BATCH_SIZE",
        "HYLO_GLOBAL_BATCH_SIZE",
        "HYLO_ILD_WARMUP_ITERS",
        "HYLO_ORDER_DATA_DIR",
        "HYLO_TEACHER_YARN_FACTOR",
    ):
        monkeypatch.delenv(name, raising=False)
    packed = load_primus_config(CONFIGS / f"hylo_{kind}_1B-enhanced-ild-packed.yaml", None).modules[0].params
    sft = load_primus_config(CONFIGS / f"hylo_{kind}_1B-enhanced-ild.yaml", None).modules[0].params

    for name in SHARED:
        assert getattr(packed, name) == getattr(sft, name), name
    if kind == "mla":
        assert (packed.hylo_mla_kv_heads, packed.rotary_scaling_factor) == (8, 32.0)
        assert packed.original_max_position_embeddings == sft.original_max_position_embeddings

    # The reference's schedule: 6934 steps of 96 packed 2048-token windows, no padding.
    assert (packed.seq_length, packed.train_iters) == (2048, 6934)
    # The Trainer's warmup_ratio 0.01, which rounds up: ceil(69.34) steps.
    assert (packed.lr_warmup_iters, packed.lr_warmup_fraction) == (70, None)
    assert sft.lr_warmup_fraction == 0.01
    assert packed.mock_data is False
    assert str(packed.train_data_path).endswith("data/hylo/packed/train")
    assert getattr(packed, "sft_dataset_name", None) is None
    assert (packed.tensor_model_parallel_size, packed.pipeline_model_parallel_size) == (1, 1)
