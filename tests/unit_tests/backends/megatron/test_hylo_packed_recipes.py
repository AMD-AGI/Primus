###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from pathlib import Path
from types import SimpleNamespace

import pytest

from primus.backends.megatron.training.hylo_distillation import _derive_warmup
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

# Packed recipe, SFT-path recipe, and the reference schedule: window length, steps, the Trainer's
# warmup steps ceil(0.01 * steps), and the packed data directory.
RECIPES = {
    "m2": ("hylo_m2_1B-enhanced-ild", 2048, 6934, 70, "data/hylo/packed/train"),
    "mla": ("hylo_mla_1B-enhanced-ild", 2048, 6934, 70, "data/hylo/packed/train"),
    "gdn": ("hylo_gdn_1B-enhanced-ild", 2048, 6934, 70, "data/hylo/packed/train"),
    "4MLA12GDN": ("hylo_4MLA12GDN_1B-8K-sft", 8192, 1000, 10, "data/hylo/packed-8k/train"),
}

ENV = (
    "HYLO_PACKED_TRAIN_DATA",
    "HYLO_ILD_TRAIN_ITERS",
    "HYLO_SFT_TRAIN_ITERS",
    "HYLO_MICRO_BATCH_SIZE",
    "HYLO_GLOBAL_BATCH_SIZE",
    "HYLO_ORDER_DATA_DIR",
    "HYLO_TEACHER_YARN_FACTOR",
)


@pytest.mark.parametrize("kind", sorted(RECIPES))
def test_packed_recipe_is_the_sft_recipe_on_the_reference_schedule(kind, monkeypatch):
    monkeypatch.setattr(file_utils, "create_path_if_not_exists", lambda *args, **kwargs: None)
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    sft_name, seq_length, steps, warmup, data = RECIPES[kind]
    packed_name = f"{sft_name}-packed" if kind == "4MLA12GDN" else f"hylo_{kind}_1B-enhanced-ild-packed"
    packed = load_primus_config(CONFIGS / f"{packed_name}.yaml", None).modules[0].params
    sft = load_primus_config(CONFIGS / f"{sft_name}.yaml", None).modules[0].params

    for name in SHARED:
        assert getattr(packed, name, None) == getattr(sft, name, None), name
    if kind == "mla":
        assert (packed.hylo_mla_kv_heads, packed.rotary_scaling_factor) == (8, 32.0)
        assert packed.original_max_position_embeddings == sft.original_max_position_embeddings

    assert (packed.seq_length, packed.train_iters) == (seq_length, steps)
    assert (packed.hylo_warmup_ratio, packed.lr_warmup_fraction) == (0.01, None)
    assert sft.lr_warmup_fraction == 0.01
    args = SimpleNamespace(**vars(packed))
    _derive_warmup(args)
    assert args.lr_warmup_iters == warmup
    assert packed.mock_data is False
    assert str(packed.train_data_path).endswith(data)
    assert getattr(packed, "sft_dataset_name", None) is None
    assert (packed.tensor_model_parallel_size, packed.pipeline_model_parallel_size) == (1, 1)


@pytest.mark.parametrize(
    "train_iters, decay_iters, expected",
    [(6934, None, 70), (1000, None, 10), (40, None, 1), (1, None, 0), (100, 50, 1), (6934, 60, 59)],
)
def test_warmup_ratio_rounds_up_below_the_decay_steps(train_iters, decay_iters, expected):
    args = SimpleNamespace(
        hylo_warmup_ratio=0.01, lr_warmup_fraction=None, train_iters=train_iters, lr_decay_iters=decay_iters
    )
    _derive_warmup(args)
    assert args.lr_warmup_iters == expected


def test_warmup_ratio_leaves_the_megatron_settings_alone_when_unset():
    args = SimpleNamespace(
        hylo_warmup_ratio=None, lr_warmup_iters=40, lr_warmup_fraction=None, train_iters=10
    )
    _derive_warmup(args)
    assert args.lr_warmup_iters == 40


@pytest.mark.parametrize(
    "overrides, match",
    [({"lr_warmup_fraction": 0.01}, "not both"), ({"train_iters": None}, "train_iters")],
)
def test_warmup_ratio_rejects_ambiguous_settings(overrides, match):
    args = SimpleNamespace(
        hylo_warmup_ratio=0.01, lr_warmup_fraction=None, train_iters=100, lr_decay_iters=None
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    with pytest.raises(ValueError, match=match):
        _derive_warmup(args)
