###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest

from primus.backends.diffusion.argument_builder import WanArgBuilder


def _minimal_params():
    return {
        "model": {
            "name": "wan",
            "config": {
                "model_type": "t2v",
            },
        },
    }


def test_rejects_legacy_public_dataset_override():
    params = _minimal_params()
    params["dataset"] = {"name": "wan"}

    builder = WanArgBuilder()
    builder.update(params)

    with pytest.raises(ValueError, match="no longer accepts public `dataset` overrides"):
        builder.finalize()


def test_rejects_legacy_public_trainer_override():
    params = _minimal_params()
    params["trainer"] = {"name": "fsdp2"}

    builder = WanArgBuilder()
    builder.update(params)

    with pytest.raises(ValueError, match="no longer accepts public `trainer` overrides"):
        builder.finalize()


def test_maps_primus_style_sections_to_wan_runtime_config():
    params = {
        **_minimal_params(),
        "stage": "posttrain",
        "primus": {"experiment": "wan-smoke"},
        "training": {
            "steps": 7,
            "local_batch_size": 2,
            "global_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "output_dir": "/tmp/wan-out",
            "save_steps": 3,
            "run_name": "wan-test",
            "num_train_epochs": 9,
            "dataloader_num_workers": 0,
            "resume_from_checkpoint": "/tmp/resume",
        },
        "data": {
            "dataset_path": "/data/meta.jsonl",
            "data_folder": "/data/videos",
            "frame_num": 17,
            "video_backend": "decord",
            "text_tokenizer": "/models/umt5",
            "height": 256,
            "width": 384,
        },
        "parallelism": {
            "sp_size": 4,
            "dp_replicate": 2,
        },
        "optimizer": {
            "lr": 2.0e-5,
            "weight_decay": 0.02,
            "adam_beta1": 0.8,
            "adam_beta2": 0.95,
            "adam_epsilon": 1.0e-7,
            "max_grad_norm": 0.5,
        },
        "runtime": {
            "attention_backend": "sdpa",
            "report_to": "none",
            "seed": 1234,
            "fsdp2_reshard_after_forward": False,
        },
        "scheduler": {
            "shift": 3.0,
            "sigma_min": 0.875,
            "sigma_max": 1.0,
            "extra_one_step": True,
        },
        "metrics": {
            "log_freq": 5,
            "enable_wandb": False,
        },
    }

    builder = WanArgBuilder()
    builder.update(params)
    result = builder.finalize()

    dataset_cfg = result.dataset["config"]
    processor_cfg = dataset_cfg["processor_config"]
    trainer_args = result.trainer["args"]

    assert result.stage == "posttrain"
    assert result.primus == {"experiment": "wan-smoke"}
    assert result.model == params["model"]

    assert dataset_cfg["dataset_path"] == "/data/meta.jsonl"
    assert dataset_cfg["data_folder"] == "/data/videos"
    assert dataset_cfg["frame_num"] == 17
    assert dataset_cfg["video_backend"] == "decord"
    assert processor_cfg["text_tokenizer"] == "/models/umt5"
    assert processor_cfg["extra_kwargs"]["size"] == {"height": 256, "width": 384}

    assert trainer_args["max_steps"] == 7
    assert trainer_args["per_device_train_batch_size"] == 2
    assert trainer_args["global_batch_size"] == 16
    assert trainer_args["gradient_accumulation_steps"] == 4
    assert trainer_args["output_dir"] == "/tmp/wan-out"
    assert trainer_args["save_steps"] == 3
    assert trainer_args["run_name"] == "wan-test"
    assert trainer_args["num_train_epochs"] == 9
    assert trainer_args["dataloader_num_workers"] == 0
    assert trainer_args["resume_from_checkpoint"] == "/tmp/resume"

    assert trainer_args["sp_size"] == 4
    assert trainer_args["dp_replicate"] == 2
    assert trainer_args["learning_rate"] == 2.0e-5
    assert trainer_args["weight_decay"] == 0.02
    assert trainer_args["adam_beta1"] == 0.8
    assert trainer_args["adam_beta2"] == 0.95
    assert trainer_args["adam_epsilon"] == 1.0e-7
    assert trainer_args["max_grad_norm"] == 0.5
    assert trainer_args["attention_backend"] == "sdpa"
    assert trainer_args["report_to"] == "none"
    assert trainer_args["seed"] == 1234
    assert trainer_args["fsdp2_reshard_after_forward"] is False
    assert trainer_args["logging_steps"] == 5
    assert trainer_args["flow_match_scheduler"]["shift"] == 3.0
    assert trainer_args["flow_match_scheduler"]["sigma_min"] == 0.875
    assert trainer_args["flow_match_scheduler"]["sigma_max"] == 1.0
    assert trainer_args["flow_match_scheduler"]["extra_one_step"] is True


def test_defaults_propagate_when_optional_sections_are_omitted():
    builder = WanArgBuilder()
    builder.update(_minimal_params())
    result = builder.finalize()

    dataset_cfg = result.dataset["config"]
    trainer_args = result.trainer["args"]

    assert result.stage == "pretrain"
    assert dataset_cfg["dataset_path"] == "/path/to/meta.jsonl"
    assert dataset_cfg["data_folder"] == "/path/to/videos"
    assert dataset_cfg["processor_config"]["text_tokenizer"] == "/path/to/umt5-xxl"
    assert result.trainer["name"] == "fsdp2"
    assert trainer_args["max_steps"] == 100
    assert trainer_args["attention_backend"] == "flash_attn_aiter"
    assert trainer_args["sp_size"] == 1
    assert trainer_args["dp_replicate"] == 1
    assert trainer_args["report_to"] == "none"


def test_metrics_enable_wandb_sets_report_to_when_runtime_omits_it():
    builder = WanArgBuilder()
    builder.update(
        {
            **_minimal_params(),
            "metrics": {
                "enable_wandb": True,
            },
        }
    )

    result = builder.finalize()

    assert result.trainer["args"]["report_to"] == "wandb"


def test_update_accepts_simplenamespace_input():
    builder = WanArgBuilder()
    builder.update(
        SimpleNamespace(
            model=SimpleNamespace(
                name="wan",
                config=SimpleNamespace(model_type="t2v"),
            ),
            data=SimpleNamespace(dataset_path="/data/meta.jsonl"),
        )
    )

    result = builder.finalize()

    assert result.model["name"] == "wan"
    assert result.model["config"]["model_type"] == "t2v"
    assert result.dataset["config"]["dataset_path"] == "/data/meta.jsonl"
