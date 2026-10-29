from __future__ import annotations

from pathlib import Path

import pytest

from primus.backends.diffusion.argument_builder import DiffusionArgBuilder
from primus.backends.diffusion.data.benchmark import AMDBProcessor
from primus.backends.diffusion.registry import DATASET_BUILDERS, MODEL_BUILDERS, TRAINER_BUILDERS

BENCHMARK_NAMES = ("hunyuan-video",)


def _finalize(params: dict):
    builder = DiffusionArgBuilder()
    builder.update(params)
    return builder.finalize()


def test_benchmark_workloads_are_registered():
    for name in BENCHMARK_NAMES:
        assert name in MODEL_BUILDERS
    assert "amdb" in DATASET_BUILDERS
    assert "amdb_benchmark" in TRAINER_BUILDERS


def test_hunyuan_keeps_checkpointing_off_and_makes_the_vae_optional(tmp_path):
    import primus.backends.diffusion.utils.download as download

    args = _finalize(
        {
            "model": {"name": "hunyuan-video", "config": {}},
            "data": {"dataset_path": "x", "resolution": "960,544", "num_frames": 9},
        }
    )
    assert args.dataset["config"]["modality"] == "video"
    assert args.dataset["config"]["model_name"] == "hunyuan-video"
    assert args.model["config"]["resolution"] == "960,544"
    assert args.model["config"]["num_frames"] == 9
    assert args.trainer["args"]["gradient_checkpointing"] is False
    assert (
        args.trainer["args"]["fsdp_transformer_layer_cls_to_wrap"]
        == "MMDoubleStreamBlock,MMSingleStreamBlock"
    )

    root = tmp_path / "hunyuan"
    (root / "transformer").mkdir(parents=True)
    (root / "text_encoder").mkdir()
    (root / "tokenizer").mkdir()
    (root / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    env = download._companion_paths("hunyuan-video", str(root / "transformer"), {})
    assert env["LLM_ENCODER"] == str(root / "text_encoder")
    assert "VAE_CHECKPOINT" not in env
    (root / "vae").mkdir()
    with_vae = download._companion_paths("hunyuan-video", str(root / "transformer"), {})
    assert with_vae["VAE_CHECKPOINT"] == str(root / "vae")


def test_benchmark_argument_builder_rejects_sequence_parallelism():
    with pytest.raises(ValueError, match="sp_size"):
        _finalize(
            {
                "model": {"name": "hunyuan-video", "config": {}},
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
    path = download.materialize_dataset("hunyuan-video", "")
    home = tmp_path / "hf"
    assert Path(path) == home / "datasets" / "Wild-Heart" / "Disney-VideoGeneration-Dataset"
    assert calls == {
        "repo": "Wild-Heart/Disney-VideoGeneration-Dataset",
        "revision": "316b85f5d7c263260cc526a25eece96cc30e0c06",
    }
    assert home / "hub" not in Path(path).parents

    download.materialize_dataset("hunyuan-video", "", revision="abc123")
    assert calls["revision"] == "abc123"

    weights = tmp_path / "transformer"
    weights.write_bytes(b"x")

    def fail_download(*_args, **_kwargs):
        raise AssertionError("checkpoint download should not run")

    monkeypatch.setattr(download, "_snapshot", fail_download)
    assert download.materialize_pretrained("hunyuan-video", str(weights)) == str(weights)

    data_root = tmp_path / "data"
    monkeypatch.delenv("PRIMUS_HF_HOME", raising=False)
    monkeypatch.setenv("DATA_PATH", str(data_root))
    monkeypatch.setenv("HF_HOME", str(data_root / "huggingface"))
    assert download.huggingface_home() == Path.home() / "cache" / "huggingface"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "custom"))
    assert download.huggingface_home() == tmp_path / "custom"


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
    from primus.backends.diffusion.data.benchmark import AMDBDataset, _TemporaryVideoReader

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


@pytest.mark.parametrize("name", ["sdxl", "mochi-1", "wan2.1-i2v", "flux-dev"])
def test_unregistered_benchmark_names_fail_while_building_arguments(name):
    with pytest.raises(ValueError, match="Unsupported diffusion model"):
        _finalize({"model": {"name": name, "config": {}}})


def test_bright_frame_lists_are_normalized_once():
    import numpy as np
    import torch

    from primus.backends.diffusion.utils.pixels import videos_to_tensor

    frame = np.full((4, 4, 3), 200, dtype=np.uint8)
    clips = videos_to_tensor([[frame, frame]], num_frames=2, height=4, width=4)
    expected = 200 / 127.5 - 1.0
    assert torch.allclose(clips, torch.full_like(clips, expected), atol=1e-4)


def test_checkpoint_directory_does_not_overlay_mixed_formats(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file

    from primus.backends.diffusion.utils.checkpoint_io import load_state_dict

    folder = tmp_path / "ckpt"
    folder.mkdir()
    save_file({"weight": torch.ones(2)}, folder / "model.safetensors")
    torch.save({"weight": torch.zeros(2)}, folder / "model.bin")
    loaded = load_state_dict(str(folder))
    assert torch.equal(loaded["weight"], torch.ones(2))

    shards = tmp_path / "shards"
    shards.mkdir()
    save_file({"a": torch.ones(1)}, shards / "model-00001.safetensors")
    save_file({"b": torch.zeros(1)}, shards / "model-00002.safetensors")
    merged = load_state_dict(str(shards))
    assert set(merged) == {"a", "b"}

    overlap = tmp_path / "overlap"
    overlap.mkdir()
    save_file({"weight": torch.ones(1)}, overlap / "first.safetensors")
    save_file({"weight": torch.zeros(1)}, overlap / "second.safetensors")
    with pytest.raises(ValueError, match="repeats keys"):
        load_state_dict(str(overlap))

    mixed = tmp_path / "mixed"
    mixed.mkdir()
    torch.save({"weight": torch.ones(1)}, mixed / "a.bin")
    torch.save({"weight": torch.zeros(1)}, mixed / "b.pt")
    with pytest.raises(ValueError, match="Mixed checkpoint formats"):
        load_state_dict(str(mixed))

    single = tmp_path / "one.pt"
    torch.save({"weight": torch.ones(1)}, single)
    seen = {}
    real_load = torch.load

    def spy(path, *args, **kwargs):
        seen.update(kwargs)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    assert torch.equal(load_state_dict(str(single))["weight"], torch.ones(1))
    assert seen.get("weights_only") is True


def test_cached_latent_loader_keeps_even_batches():
    import torch

    from primus.backends.diffusion.trainers.benchmark import cached_sample_loader

    samples = [{"latents": torch.tensor([float(index)]), "prompt": str(index)} for index in range(7)]
    _sampler, loader = cached_sample_loader(
        samples,
        batch_size=3,
        shuffle=False,
        even_batches=True,
        drop_last=False,
    )
    prompts = [tuple(batch["prompts"]) for batch in loader]
    assert prompts == [("0", "1", "2"), ("3", "4", "5"), ("6", "0", "1")]
    assert all(batch["latents"].shape[0] == 3 for batch in loader)


def test_hunyuan_rope_tables_are_cached(monkeypatch):
    import torch

    from primus.backends.diffusion.models.hunyuan.train_pipeline import HunyuanTrainPipeline

    calls = {"count": 0}

    def fake(*_args, **_kwargs):
        calls["count"] += 1
        return torch.zeros(2, 4), torch.ones(2, 4)

    monkeypatch.setattr(
        "primus.backends.diffusion.models.hunyuan.train_pipeline.get_nd_rotary_pos_embed",
        fake,
    )
    pipeline = HunyuanTrainPipeline()
    dit = type("Dit", (), {"rope_dim_list": [2, 2]})()
    first_cos, first_sin = pipeline._rotary_embeddings(
        dit, (1, 2, 2), device=torch.device("cpu"), dtype=torch.float32
    )
    second_cos, second_sin = pipeline._rotary_embeddings(
        dit, (1, 2, 2), device=torch.device("cpu"), dtype=torch.float32
    )
    assert calls["count"] == 1
    assert first_cos is second_cos
    assert first_sin is second_sin
