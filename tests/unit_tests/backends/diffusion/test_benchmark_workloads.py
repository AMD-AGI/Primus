from __future__ import annotations

from pathlib import Path

import pytest

from primus.backends.diffusion.argument_builder import DiffusionArgBuilder
from primus.backends.diffusion.data.benchmark import AMDBProcessor
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


def test_unbuilt_benchmark_models_fail_before_download():
    for name in ("sdxl", "stable-diffusion-xl", "mochi-1", "hunyuan-video", "wan2.1-i2v"):
        with pytest.raises(ValueError, match="no diffusion model builder"):
            _finalize({"model": {"name": name, "config": {}}, "data": {"dataset_path": "x"}})


def test_flux_dev_keeps_explicit_img_size_and_parses_resolution():
    kept = _finalize(
        {
            "model": {"name": "flux-dev", "config": {"resolution": 512}},
            "data": {"img_size": 256, "dataset_path": "images"},
        }
    )
    assert kept.dataset["name"] == "flux"
    assert kept.dataset["config"]["processor_config"]["img_size"] == 256

    square = _finalize(
        {
            "model": {"name": "flux-dev", "config": {}},
            "data": {"resolution": "512x512", "dataset_path": "images"},
        }
    )
    assert square.dataset["config"]["processor_config"]["img_size"] == 512

    decimal = _finalize(
        {
            "model": {"name": "flux-dev", "config": {"resolution": "768.0"}},
            "data": {"dataset_path": "images"},
        }
    )
    assert decimal.dataset["config"]["processor_config"]["img_size"] == 768

    with pytest.raises(ValueError, match="square"):
        _finalize(
            {
                "model": {"name": "flux-dev", "config": {}},
                "data": {"resolution": "1024,768"},
            }
        )


def test_flux_dev_precomputed_stays_on_the_flux_loader():
    args = _finalize(
        {
            "model": {"name": "flux-dev", "config": {}},
            "data": {"dataset_type": "precomputed", "dataset_path": "encoded"},
        }
    )
    assert args.dataset["name"] == "flux"
    assert args.dataset["config"]["dataset_type"] == "precomputed"


def test_sigma_grid_matches_the_shifted_flow_endpoints():
    import torch

    from primus.backends.diffusion.utils.sampling import (
        amdb_flow_sigma_grid,
        sample_shifted_logit_normal_sigmas,
    )

    grid = amdb_flow_sigma_grid()
    assert grid.shape == (1000,)
    assert grid.dtype == torch.float32
    torch.testing.assert_close(grid[0], torch.tensor(1.0))
    expected_last = 3.0 * 0.001 / (1.0 + 2.0 * 0.001)
    torch.testing.assert_close(grid[-1], torch.tensor(expected_last))

    drawn = sample_shifted_logit_normal_sigmas(4, device=torch.device("cpu"), sigma_grid=grid)
    assert drawn.shape == (4,)
    assert torch.all((drawn >= grid[-1]) & (drawn <= grid[0]))


def test_missing_flops_leave_tflops_unset():
    from primus.backends.diffusion.trainers.base import benchmark_step_rates

    fps_gpu, tflops_per_gpu = benchmark_step_rates(
        local_batch_size=8,
        step_time_gpu_s=2.0,
        total_flops=None,
    )
    assert fps_gpu == 4.0
    assert tflops_per_gpu is None


def test_benchmark_fsdp2_hooks_extend_the_parent_wrapper():
    import torch

    from primus.backends.diffusion.trainers.benchmark import AMDBBenchmarkTrainer
    from primus.backends.diffusion.trainers.fsdp2 import FSDP2Trainer

    assert AMDBBenchmarkTrainer._apply_fsdp2 is FSDP2Trainer._apply_fsdp2
    trainer = AMDBBenchmarkTrainer.__new__(AMDBBenchmarkTrainer)
    trainer.args = {"fsdp2_pure_low_precision": True}
    trainer.rank = 0
    assert trainer._fsdp2_reduce_dtype_name(torch.bfloat16) == "bf16"
    assert trainer._fsdp2_reduce_dtype_name(torch.float32) == "fp32"


def test_precomputed_cache_does_not_look_up_image():
    import torch

    from primus.backends.diffusion.trainers.benchmark import AMDBBenchmarkTrainer

    trainer = AMDBBenchmarkTrainer.__new__(AMDBBenchmarkTrainer)
    trainer.rank = 0
    trainer.sampler = type("Sampler", (), {"set_epoch": lambda self, epoch: None})()
    trainer.processing_class = None
    trainer.dataloader = [
        {
            "mean": torch.zeros(2, 1),
            "logvar": torch.zeros(2, 1),
            "t5_encodings": torch.zeros(2, 1),
            "clip_encodings": torch.zeros(2, 1),
        }
    ]
    trainer._prepare_host_batch = lambda batch: batch
    trainer._prepare_batch = lambda batch, processor: batch

    def encode(_batch):
        raise AssertionError("precomputed batches must not be re-encoded")

    trainer.model = type("Model", (), {"encode_batch": staticmethod(encode)})()
    trainer._cache_encoded_batches()
    assert trainer._encoded_cache is False


def test_cached_loader_pads_the_short_tail():
    import torch

    from primus.backends.diffusion.trainers.benchmark import AMDBBenchmarkTrainer

    trainer = AMDBBenchmarkTrainer.__new__(AMDBBenchmarkTrainer)
    trainer.rank = 0
    trainer.per_device_train_batch_size = 4
    trainer.args = {"shuffle": False, "dataloader_even_batches": True}
    trainer.sampler = type("Sampler", (), {"set_epoch": lambda self, epoch: None})()
    trainer.processing_class = None
    trainer.dataloader = [{"image": torch.zeros(5, 1), "prompts": ["p"] * 5}]
    trainer._prepare_host_batch = lambda batch: batch
    trainer._prepare_batch = lambda batch, processor: batch

    class Model:
        def encode_batch(self, batch):
            count = batch["image"].shape[0]
            return {
                "latents": torch.arange(count).float(),
                "t5_encodings": torch.zeros(count, 1),
                "clip_encodings": torch.zeros(count, 1),
            }

    trainer.model = Model()
    trainer._cache_encoded_batches()
    sizes = [batch["latents"].shape[0] for batch in trainer.dataloader]
    assert sizes == [4, 4]


def test_frame_list_videos_are_normalized_once():
    import torch

    from primus.backends.diffusion.utils.pixels import videos_to_tensor

    frame = torch.full((4, 4, 3), 200.0)
    clip = videos_to_tensor([[frame]], num_frames=1, height=4, width=4)
    expected = 200.0 / 127.5 - 1.0
    torch.testing.assert_close(clip[0, 0, 0, 0, 0], torch.tensor(expected))


def test_flux_prepare_hook_rewrites_paths_and_reports_auth_failures(monkeypatch, tmp_path, capsys):
    import primus.backends.diffusion.utils.download as download
    from runner.helpers.hooks.train.pretrain.diffusion import (
        prepare as diffusion_prepare,
    )

    dit = tmp_path / "flux1-dev.safetensors"
    data = tmp_path / "images"
    ae = tmp_path / "ae.safetensors"

    def fake_stage(model_name, **kwargs):
        assert model_name == "flux-dev"
        assert kwargs["encoder"] == {}
        return {
            "PRETRAINED_PATH": str(dit),
            "DATASET_PATH": str(data),
            "VAE_CHECKPOINT": str(ae),
        }

    monkeypatch.setattr(download, "stage_benchmark_assets", fake_stage)
    model_cfg: dict = {}
    dataset_cfg: dict = {"dataset_path": ""}
    env = diffusion_prepare._stage_benchmark_downloads("flux-dev", dataset_cfg, model_cfg, {})
    assert model_cfg["load_from_pretrained_path"] == str(dit)
    assert dataset_cfg["dataset_path"] == str(data)
    assert model_cfg["encoder"]["autoencoder"] == str(ae)
    assert env["VAE_CHECKPOINT"] == str(ae)

    def unauthorized(*_args, **_kwargs):
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(download, "stage_benchmark_assets", unauthorized)
    with pytest.raises(SystemExit):
        diffusion_prepare._stage_benchmark_downloads("flux-dev", {"dataset_path": ""}, {}, {})
    assert "hf_token" in capsys.readouterr().err

    def refuse_download(*_args, **_kwargs):
        raise AssertionError("unbuilt benchmark must not download assets")

    monkeypatch.setattr(download, "stage_benchmark_assets", refuse_download)
    with pytest.raises(SystemExit):
        diffusion_prepare._reject_unbuilt_benchmark("sdxl")
    assert "no diffusion model builder" in capsys.readouterr().err
    assert diffusion_prepare._BENCHMARK_DOWNLOADS == {"flux-dev"}
