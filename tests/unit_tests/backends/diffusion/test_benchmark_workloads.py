from __future__ import annotations

from pathlib import Path

import pytest

from primus.backends.diffusion.argument_builder import DiffusionArgBuilder
from primus.backends.diffusion.data.benchmark import AMDBProcessor
from primus.backends.diffusion.registry import DATASET_BUILDERS, MODEL_BUILDERS

BENCHMARK_NAMES = (
    "sdxl",
    "stable-diffusion-xl",
)


def _finalize(params: dict):
    builder = DiffusionArgBuilder()
    builder.update(params)
    return builder.finalize()


def test_benchmark_workloads_are_registered():
    for name in BENCHMARK_NAMES:
        assert name in MODEL_BUILDERS
    assert "amdb" in DATASET_BUILDERS


def test_sdxl_uses_the_amdb_dataset_and_unet_blocks():
    args = _finalize({"model": {"name": "sdxl", "config": {}}, "data": {"dataset_path": "x"}})
    assert args.dataset["name"] == "amdb"
    assert args.dataset["config"]["modality"] == "image"
    assert args.trainer["args"]["fsdp_transformer_layer_cls_to_wrap"] == "BasicTransformerBlock"
    assert args.trainer["args"]["learning_rate"] == 1.0e-4


def test_sdxl_companion_checkpoint_is_the_unet_file(tmp_path):
    import primus.backends.diffusion.utils.download as download

    weights = tmp_path / "sd_xl_base_1.0.safetensors"
    weights.write_bytes(b"sdxl")
    assert download._companion_paths("sdxl", str(weights), {})["VAE_CHECKPOINT"] == str(weights)


def test_benchmark_argument_builder_rejects_sequence_parallelism():
    with pytest.raises(ValueError, match="sp_size"):
        _finalize(
            {
                "model": {"name": "sdxl", "config": {}},
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
    path = download.materialize_dataset("sdxl", "")
    home = tmp_path / "hf"
    assert Path(path) == home / "datasets" / "frank-chieng" / "chinese_architecture_siheyuan"
    assert calls == {
        "repo": "frank-chieng/chinese_architecture_siheyuan",
        "revision": None,
    }
    assert home / "hub" not in Path(path).parents

    weights = tmp_path / "sd_xl_base_1.0.safetensors"
    weights.write_bytes(b"x")

    def fail_download(*_args, **_kwargs):
        raise AssertionError("checkpoint download should not run")

    monkeypatch.setattr(download, "_snapshot", fail_download)
    assert download.materialize_pretrained("sdxl", str(weights)) == str(weights)

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


def test_sdxl_benchmark_disables_grad_clipping_by_default():
    args = _finalize({"model": {"name": "sdxl", "config": {}}, "data": {"dataset_path": "x"}})
    assert args.trainer["args"]["max_grad_norm"] == 0.0


def test_unregistered_benchmark_presets_are_rejected():
    with pytest.raises(ValueError, match="Unsupported diffusion model"):
        _finalize({"model": {"name": "mochi-1", "config": {}}})


def test_benchmark_data_overrides_reach_the_dataset_and_model():
    args = _finalize(
        {
            "model": {"name": "stable-diffusion-xl", "config": {}},
            "data": {
                "modality": "image",
                "max_train_samples": 4,
                "revision": "main",
                "split": "validation",
                "resolution": 512,
                "num_frames": 1,
                "center_crop": True,
                "shift": 3.0,
            },
        }
    )
    cfg = args.dataset["config"]
    assert cfg["model_name"] == "sdxl"
    assert cfg["modality"] == "image"
    assert cfg["max_train_samples"] == 4
    assert cfg["revision"] == "main"
    assert cfg["split"] == "validation"
    assert args.model["config"]["resolution"] == 512
    assert args.model["config"]["num_frames"] == 1
    assert args.model["config"]["center_crop"] is True
    assert args.model["config"]["shift"] == 3.0


def test_dataloader_even_batches_follows_the_flag(tmp_path):
    import torch

    from primus.backends.diffusion.trainers.base import (
        BaseWanTrainer,
        EvenSizeBatchSampler,
    )

    class _Trainer(BaseWanTrainer):
        def _apply_parallelism(self):
            return None

        def _setup_mlperf(self):
            return None

    class _Rows:
        def __init__(self, count: int):
            self.count = count

        def __len__(self) -> int:
            return self.count

        def __getitem__(self, index: int) -> int:
            return index

        def get_collator(self):
            return lambda batch: batch

    def _make(flag: bool, *, mlperf: bool = False):
        rows = _Rows(8)
        args = {
            "dataloader_even_batches": flag,
            "mlperf_enable": mlperf,
            "per_device_train_batch_size": 4,
            "dataloader_num_workers": 0,
            "output_dir": str(tmp_path / f"loader-{flag}-{mlperf}"),
            "report_to": "none",
        }
        if mlperf:
            args["mlperf_train_samples"] = len(rows)
            args["mlperf_eval_total_samples"] = len(rows)
        return _Trainer(
            model=torch.nn.Linear(1, 1),
            args=args,
            train_dataset=rows,
            data_collator=lambda batch: batch,
            processing_class=None,
            rank=0,
            world_size=1,
            local_rank=0,
            eval_dataset=rows if mlperf else None,
        )

    even = _make(True)
    assert isinstance(even.dataloader.batch_sampler, EvenSizeBatchSampler)
    plain = _make(False)
    assert not isinstance(plain.dataloader.batch_sampler, EvenSizeBatchSampler)
    mlperf = _make(True, mlperf=True)
    assert not isinstance(mlperf.dataloader.batch_sampler, EvenSizeBatchSampler)


def test_placeholder_pretrained_downloads_only_the_unet_file(monkeypatch, tmp_path):
    import primus.backends.diffusion.utils.download as download

    monkeypatch.setenv("PRIMUS_HF_HOME", str(tmp_path / "hf"))
    calls = {}

    def fake_snapshot(repo_id, *, revision=None, allow_patterns=None):
        calls["repo"] = repo_id
        calls["allow_patterns"] = allow_patterns
        folder = tmp_path / "snap"
        folder.mkdir()
        (folder / "sd_xl_base_1.0.safetensors").write_bytes(b"sdxl")
        return str(folder)

    monkeypatch.setattr(download, "_snapshot", fake_snapshot)
    path = download.materialize_pretrained("sdxl", "/path/to/sd_xl_base_1.0.safetensors")
    assert path.endswith("sd_xl_base_1.0.safetensors")
    assert calls["repo"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert calls["allow_patterns"] == ["sd_xl_base_1.0.safetensors"]


def test_load_matching_rejects_an_unrelated_checkpoint(tmp_path):
    import torch

    from primus.backends.diffusion.utils.checkpoint_io import load_matching

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

    wrong = tmp_path / "wrong.pt"
    torch.save({"unrelated": torch.zeros(2)}, wrong)
    with pytest.raises(RuntimeError, match="partial initialization"):
        load_matching(_Tiny(), str(wrong))

    right = tmp_path / "right.pt"
    torch.save({"weight": torch.ones(2), "extra": torch.zeros(1)}, right)
    missing, unexpected = load_matching(_Tiny(), str(right))
    assert missing == 0
    assert unexpected == 1


def test_sdxl_timesteps_match_the_applied_noise_level():
    import torch

    from primus.backends.diffusion.utils.sampling import (
        amdb_flow_sigma_grid,
        sample_amdb_sdxl_sigmas,
    )

    grid = amdb_flow_sigma_grid(shift=3.0, num_train_timesteps=1000, zero_terminal=True)
    timesteps, sigma = sample_amdb_sdxl_sigmas(64, device=torch.device("cpu"), sigma_grid=grid)
    indices = 999 - timesteps
    assert torch.equal(sigma, grid[indices])
    assert int(timesteps[int(sigma.argmax())]) > int(timesteps[int(sigma.argmin())])


def test_sdxl_compute_loss_noises_with_the_timestep_schedule(monkeypatch):
    import torch
    from torch import nn

    from primus.backends.diffusion.models.sdxl import train_pipeline as pipeline_module
    from primus.backends.diffusion.models.sdxl.train_pipeline import SDXLTrainPipeline

    pipeline = SDXLTrainPipeline(resolution=64)
    fixed_timesteps = torch.tensor([999])

    def fake_sample(batch_size, *, device, **kwargs):
        del batch_size, kwargs
        return fixed_timesteps.to(device), torch.tensor([0.9], device=device)

    monkeypatch.setattr(pipeline_module, "sample_amdb_sdxl_sigmas", fake_sample)
    noise = torch.ones(1, 4, 2, 2)

    def _ones_like(latents):
        return noise.to(device=latents.device, dtype=latents.dtype)

    monkeypatch.setattr(torch, "randn_like", _ones_like)

    class _Dit(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))
            self.timesteps = None
            self.noisy = None

        def forward(self, noisy, timesteps, context=None, y=None):
            del context, y
            self.timesteps = timesteps.detach().clone()
            self.noisy = noisy.detach().clone()
            return torch.zeros_like(noisy)

    dit = _Dit()
    latents = torch.zeros(1, 4, 2, 2)
    loss = pipeline.compute_loss(
        dit=dit,
        autoencoder=None,
        text_encoder=None,
        batch={
            "latents": latents,
            "context": torch.zeros(1, 4, 8),
            "pooled": torch.zeros(1, 4),
        },
    )
    assert int(dit.timesteps[0]) == 999
    alpha = pipeline._alphas[999]
    expected = (alpha.sqrt() * latents + (1.0 - alpha).sqrt() * noise).to(latents.dtype)
    assert torch.allclose(dit.noisy, expected)
    assert loss["loss"].ndim == 0
    assert torch.isfinite(loss["loss"])

    calls = {"grid": 0}
    real_grid = pipeline_module.amdb_flow_sigma_grid

    def counting_grid(**kwargs):
        calls["grid"] += 1
        return real_grid(**kwargs)

    monkeypatch.setattr(pipeline_module, "amdb_flow_sigma_grid", counting_grid)
    pipeline.compute_loss(
        dit=dit,
        autoencoder=None,
        text_encoder=None,
        batch={
            "latents": latents,
            "context": torch.zeros(1, 4, 8),
            "pooled": torch.zeros(1, 4),
        },
    )
    assert calls["grid"] == 0


def test_accumulated_update_fps_counts_every_microbatch():
    from primus.backends.diffusion.trainers.base import benchmark_step_rates

    one_microbatch, _ = benchmark_step_rates(local_batch_size=4, step_time_gpu_s=2.0, total_flops=0.0)
    full_update, _ = benchmark_step_rates(local_batch_size=8, step_time_gpu_s=2.0, total_flops=0.0)
    assert one_microbatch == 2.0
    assert full_update == 4.0
