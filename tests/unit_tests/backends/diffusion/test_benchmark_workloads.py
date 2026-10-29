from __future__ import annotations

from pathlib import Path

import pytest

from primus.backends.diffusion.data.benchmark import AMDBProcessor
from primus.backends.diffusion.argument_builder import DiffusionArgBuilder
from primus.backends.diffusion.registry import DATASET_BUILDERS, MODEL_BUILDERS

BENCHMARK_NAMES = ("flux-dev",)


def _finalize(params: dict):
    builder = DiffusionArgBuilder()
    builder.update(params)
    return builder.finalize()


def test_benchmark_workloads_are_registered():
    for name in BENCHMARK_NAMES:
        assert name in MODEL_BUILDERS
    assert "amdb" in DATASET_BUILDERS


def test_benchmark_argument_builder_selects_flux_dev_defaults():
    args = _finalize(
        {
            "model": {"name": "flux-dev", "config": {"model_path": "black-forest-labs/FLUX.1-dev"}},
            "training": {"steps": 4, "local_batch_size": 2},
            "data": {"dataset_path": "some/images"},
        }
    )

    assert args.dataset["name"] == "flux"
    assert args.dataset["config"]["dataset_type"] == "raw"
    assert args.dataset["config"]["dataset_path"] == "some/images"
    assert args.trainer["args"]["max_steps"] == 4
    assert args.trainer["args"]["per_device_train_batch_size"] == 2
    assert args.trainer["args"]["learning_rate"] == 1.0e-5
    assert args.trainer["args"]["attention_backend"] == "flash_attn_aiter"
    assert args.trainer["args"]["compile_transformer_blocks"] is False
    assert args.trainer["args"]["fsdp2_wrap_target"] == "dit"
    assert args.trainer["args"]["fsdp_transformer_layer_cls_to_wrap"] == "DoubleStreamBlock,SingleStreamBlock"


def test_benchmark_argument_builder_rejects_sequence_parallelism():
    with pytest.raises(ValueError, match="sp_size"):
        _finalize(
            {
                "model": {"name": "flux-dev", "config": {}},
                "parallelism": {"sp_size": 2},
            }
        )


def test_amdb_processor_groups_raw_samples():
    prepared = AMDBProcessor({}).prepare_batch(
        [
            {"image": "a", "prompt": "cat"},
            {"image": "b", "prompt": "dog"},
        ]
    )
    assert prepared["images"] == ["a", "b"]
    assert prepared["prompts"] == ["cat", "dog"]


def test_benchmark_downloads_keep_datasets_separate(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIMUS_HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("DATA_PATH", raising=False)
    import primus.backends.diffusion.utils.download as download

    calls = {}

    def fake_snapshot(repo_id, *, revision=None):
        calls["repo"] = repo_id
        calls["revision"] = revision
        folder = download.dataset_folder(repo_id)
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder)

    monkeypatch.setattr(download, "_snapshot_dataset", fake_snapshot)
    path = download.materialize_dataset("flux-dev", "")
    home = tmp_path / "hf"
    assert Path(path) == home / "datasets" / "frank-chieng" / "chinese_architecture_siheyuan"
    assert calls == {
        "repo": "frank-chieng/chinese_architecture_siheyuan",
        "revision": None,
    }
    assert home / "hub" not in Path(path).parents

    weights = tmp_path / "flux1-dev.safetensors"
    weights.write_bytes(b"x")

    def fail_download(*_args, **_kwargs):
        raise AssertionError("checkpoint download should not run")

    monkeypatch.setattr(download, "_snapshot", fail_download)
    assert download.materialize_pretrained("flux-dev", str(weights)) == str(weights)

    data_root = tmp_path / "data"
    monkeypatch.delenv("PRIMUS_HF_HOME", raising=False)
    monkeypatch.setenv("DATA_PATH", str(data_root))
    monkeypatch.setenv("HF_HOME", str(data_root / "huggingface"))
    assert download.huggingface_home() == Path.home() / "cache" / "huggingface"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "custom"))
    assert download.huggingface_home() == tmp_path / "custom"


def test_flux_autoencoder_comes_from_the_downloaded_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIMUS_HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("DATA_PATH", raising=False)
    import primus.backends.diffusion.utils.download as download

    snapshot = tmp_path / "flux"
    snapshot.mkdir()
    dit = snapshot / "flux1-dev.safetensors"
    ae = snapshot / "ae.safetensors"
    dit.write_bytes(b"dit")
    ae.write_bytes(b"ae")
    monkeypatch.setattr(download, "prefetch_huggingface_repo", lambda _value: None)
    env = download.stage_benchmark_assets(
        "flux-dev",
        dataset_path=str(tmp_path / "cached.pt"),
        pretrained=str(dit),
        encoder={"t5_encoder": "google/t5-v1_1-xxl", "clip_encoder": "openai/clip-vit-large-patch14"},
    )
    assert env["VAE_CHECKPOINT"] == str(ae)
    assert "PRETRAINED_PATH" not in env


def test_siheyuan_train_split_includes_both_archives_and_log_pngs(tmp_path):
    import zipfile

    from primus.backends.diffusion.data.benchmark import (
        caption_from_image_filename,
        read_siheyuan_images,
        siheyuan_image_members,
    )

    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
    )
    first = tmp_path / "siheyuan.zip"
    second = tmp_path / "sdxl_lora_architecture_siheyuan_dataset.zip"
    with zipfile.ZipFile(first, "w") as zipped:
        zipped.writestr("notes.txt", "not an image")
        zipped.writestr("courtyard_gate.jpg", png)
    with zipfile.ZipFile(second, "w") as zipped:
        zipped.writestr("layout.npz", b"not an image")
        zipped.writestr("west_wing.jpg", png)
    log = tmp_path / "logs" / "20230816061827"
    log.mkdir(parents=True)
    (log / "sample.png").write_bytes(png)
    members = siheyuan_image_members(["notes.txt", "courtyard_gate.jpg", "dir/"])
    assert members == ["courtyard_gate.jpg"]
    rows = read_siheyuan_images(str(tmp_path))
    assert [name for name, _payload in rows] == ["west_wing.jpg", "courtyard_gate.jpg", "sample.png"]
    assert (
        caption_from_image_filename(type("Img", (), {"filename": "courtyard_gate.jpg"})()) == "courtyard gate"
    )


def test_benchmark_step_rates_match_amdiffusionbenchmark():
    from primus.backends.diffusion.trainers.base import benchmark_step_rates

    # AMDiffusionBenchmark: fps_gpu = batch / step_time_gpu,
    # tflops/s = total_flops * 1e-12 / step_time_gpu.
    fps_gpu, tflops_per_gpu = benchmark_step_rates(
        local_batch_size=10,
        step_time_gpu_s=2.5,
        total_flops=1.0e15,
    )
    assert fps_gpu == 4.0
    assert tflops_per_gpu == 400.0


def test_amdb_dataset_skips_decode_errors_only(tmp_path):
    from primus.backends.diffusion.data.benchmark import (
        AMDBDataset,
        _TemporaryVideoReader,
    )

    class Corrupt:
        def load(self):
            raise OSError("corrupt image")

    class Fine:
        def load(self):
            return None

    dataset = AMDBDataset(
        [{"image": Corrupt(), "prompt": "bad"}, {"image": Fine(), "prompt": "good"}],
        modality="image",
    )
    assert dataset[0]["prompt"] == "good"
    with pytest.raises(KeyError):
        AMDBDataset([{"prompt": "missing image"}], modality="image")[0]

    scratch = tmp_path / "amdb-temp-video.mp4"
    scratch.write_bytes(b"mp4")

    class DummyReader:
        def get_batch(self, indices):
            return indices

        def __len__(self):
            return 1

    reader = _TemporaryVideoReader(str(scratch), DummyReader())
    assert reader.get_batch([0]) == [0]
    assert not scratch.exists()


def test_clip_grad_norm_skips_the_collective_when_clipping_is_off():
    import torch

    from primus.backends.diffusion.trainers.base import BaseWanTrainer

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))

    model = _Tiny()
    model.weight.grad = torch.ones(2)
    trainer = BaseWanTrainer.__new__(BaseWanTrainer)
    trainer.max_grad_norm = 0.0
    trainer.model = model
    assert BaseWanTrainer._clip_grad_norm(trainer) == 0.0
    assert torch.equal(model.weight.grad, torch.ones(2))


def test_even_batches_pad_short_tail_like_accelerate():
    from primus.backends.diffusion.trainers.base import EvenSizeBatchSampler

    # 20 samples per rank and batch 16 is the Siheyuan 8-GPU case: a full
    # batch, then a tail of 4. Accelerate repeats the front of the loader
    # so the tail is also train_batch_size.
    batches = list(EvenSizeBatchSampler(range(20), 16))
    assert batches == [list(range(16)), list(range(16, 20)) + list(range(12))]
    assert all(len(batch) == 16 for batch in batches)
    assert len(EvenSizeBatchSampler(range(16), 16)) == 1
