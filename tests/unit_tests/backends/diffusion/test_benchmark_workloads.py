from __future__ import annotations

from pathlib import Path

import pytest

from primus.backends.diffusion.argument_builder import DiffusionArgBuilder
from primus.backends.diffusion.data.benchmark import AMDBProcessor
from primus.backends.diffusion.registry import DATASET_BUILDERS, MODEL_BUILDERS

BENCHMARK_NAMES = ("flux-dev", "wan2.1-i2v", "sdxl", "stable-diffusion-xl", "mochi-1", "hunyuan-video")


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
    for name in ("flux-dev", "wan2.1-i2v", "sdxl", "mochi-1", "hunyuan-video"):
        with pytest.raises(ValueError, match="sp_size"):
            _finalize(
                {
                    "model": {"name": name, "config": {}},
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

    path = download.materialize_dataset("sdxl", "")
    assert Path(path) == home / "datasets" / "frank-chieng" / "chinese_architecture_siheyuan"
    assert calls == {
        "repo": "frank-chieng/chinese_architecture_siheyuan",
        "revision": None,
    }

    path = download.materialize_dataset("mochi-1", "")
    assert Path(path) == home / "datasets" / "Wild-Heart" / "Disney-VideoGeneration-Dataset"
    assert calls == {
        "repo": "Wild-Heart/Disney-VideoGeneration-Dataset",
        "revision": "316b85f5d7c263260cc526a25eece96cc30e0c06",
    }
    download.materialize_dataset("hunyuan-video", "", revision="abc123")
    assert calls["revision"] == "abc123"

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
    monkeypatch.setattr(download, "prefetch_huggingface_repo", lambda _value, **_kwargs: None)
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


def test_registered_benchmarks_build_without_a_download():
    args = _finalize({"model": {"name": "hunyuan-video", "config": {}}, "data": {"dataset_path": "x"}})
    assert args.dataset["config"]["model_name"] == "hunyuan-video"


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
    parent = FSDP2Trainer.__new__(FSDP2Trainer)
    assert parent._fsdp2_reduce_dtype_name(torch.bfloat16) == "fp32"


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
    diffusion_prepare._reject_unbuilt_benchmark("hunyuan-video")
    assert diffusion_prepare._UNBUILT_BENCHMARKS == set()
    assert diffusion_prepare._BENCHMARK_DOWNLOADS == {
        "flux-dev",
        "wan2.1-i2v",
        "sdxl",
        "stable-diffusion-xl",
        "mochi-1",
        "hunyuan-video",
    }


def test_build_flux_dev_benchmark_loads_amdb_defaults(monkeypatch, tmp_path):
    from primus.backends.diffusion.models.registrations import flux as flux_reg

    checkpoint = tmp_path / "flux1-dev.safetensors"
    autoencoder = tmp_path / "ae.safetensors"
    checkpoint.write_bytes(b"dit")
    autoencoder.write_bytes(b"ae")
    seen = {}

    def fake_build(config):
        seen["config"] = config
        return "flux-model"

    monkeypatch.setattr(flux_reg, "build_flux_model", fake_build)
    built = flux_reg.build_flux_dev_benchmark({"load_from_pretrained_path": str(checkpoint), "config": {}})

    assert built == "flux-model"
    config = seen["config"]
    assert config["model_preset"] == "flux.1-dev"
    assert config["load_from_pretrained_path"] == str(checkpoint)
    assert config["config"]["timestep_sampling"] == "amdb_logit_normal"
    assert config["config"]["shift"] == 3.0
    assert config["config"]["strict_pretrained_load"] is True
    assert config["config"]["params"]["qk_norm_eps"] == 1e-6
    assert config["encoder"]["autoencoder"] == str(autoencoder)


def test_collate_encoded_stacks_latents_and_keeps_prompts():
    import torch

    from primus.backends.diffusion.trainers.benchmark import _collate_encoded

    batch = _collate_encoded(
        [
            {"latents": torch.zeros(1), "prompt": "cat"},
            {"latents": torch.ones(1), "prompt": "dog"},
        ]
    )
    assert batch["latents"].shape == (2, 1)
    assert batch["prompts"] == ["cat", "dog"]
    assert "prompt" not in batch


def test_benchmark_update_fps_counts_every_micro_batch(monkeypatch):
    import torch

    from primus.backends.diffusion.trainers.benchmark import AMDBBenchmarkTrainer

    class _Event:
        def __init__(self, enable_timing=True):
            del enable_timing

        def record(self):
            return None

        def elapsed_time(self, other):
            del other
            return 2000.0

    monkeypatch.setattr(torch.cuda, "Event", _Event, raising=False)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None, raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None, raising=False)

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    model = _Model()
    logged = {}

    def compute_loss(batch):
        return model.weight * batch["latents"].sum()

    trainer = AMDBBenchmarkTrainer.__new__(AMDBBenchmarkTrainer)
    trainer.rank = 0
    trainer.world_size = 1
    trainer.model = model
    trainer.args = {"cache_latents": False, "compile_transformer_blocks": True}
    trainer.mlperf_enabled = False
    trainer.optimizer = type(
        "Opt",
        (),
        {"zero_grad": lambda self, set_to_none=True: None, "step": lambda self: None},
    )()
    trainer.lr_scheduler = type("Sched", (), {"step": lambda self: None})()
    trainer.global_step = 0
    trainer.grad_accum_steps = 2
    trainer.per_device_train_batch_size = 4
    trainer.num_train_epochs = 1
    trainer.max_steps = 1
    trainer.total_steps = 1
    trainer.save_steps = 0
    trainer.max_grad_norm = 0.0
    trainer.sampler = type("Sampler", (), {"set_epoch": lambda self, epoch: None})()
    trainer.processing_class = None
    trainer.dataloader = [
        {"latents": torch.ones(4, 1)},
        {"latents": torch.ones(4, 1)},
    ]
    trainer.compute_loss = compute_loss
    trainer._log_step = lambda *args, **kwargs: logged.update(kwargs)

    trainer.train()

    # Two micro-batches of 4 over a 2s CUDA window. Counting only the configured
    # batch size would report 2.0.
    assert logged["fps_gpu"] == 4.0


def test_flux_prepare_hook_prints_benchmark_env(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    import primus.backends.diffusion.utils.download as download
    from runner.helpers.hooks.train.pretrain.diffusion import (
        prepare as diffusion_prepare,
    )

    dit = tmp_path / "flux1-dev.safetensors"
    data = tmp_path / "images"
    ae = tmp_path / "ae.safetensors"
    dit.write_bytes(b"dit")
    ae.write_bytes(b"ae")
    data.mkdir()

    def fake_stage(model_name, **kwargs):
        assert model_name == "flux-dev"
        return {
            "PRETRAINED_PATH": str(dit),
            "DATASET_PATH": str(data),
            "VAE_CHECKPOINT": str(ae),
        }

    module = SimpleNamespace(framework="diffusion", params={})
    backend_args = SimpleNamespace(
        stage="pretrain",
        model={"name": "flux-dev", "config": {}},
        dataset={
            "config": {
                "dataset_path": "",
                "dataset_type": "precomputed",
                "processor_config": {},
            }
        },
        trainer={"name": "amdb_benchmark"},
    )
    monkeypatch.setattr(download, "stage_benchmark_assets", fake_stage)
    monkeypatch.setattr(diffusion_prepare, "load_primus_config", lambda path: object())
    monkeypatch.setattr(diffusion_prepare, "get_module_config", lambda cfg, name: module)
    monkeypatch.setattr(
        diffusion_prepare,
        "DiffusionAdapter",
        lambda: SimpleNamespace(convert_config=lambda params: backend_args),
    )

    diffusion_prepare.validate_diffusion_config(tmp_path / "experiment.yaml", module_name="post_trainer")
    printed = capsys.readouterr().out
    assert f"env.PRETRAINED_PATH={dit}" in printed
    assert f"env.DATASET_PATH={data}" in printed
    assert f"env.VAE_CHECKPOINT={ae}" in printed
    assert "env.PREPARED=1" in printed


def test_wan_i2v_uses_dit_blocks_and_the_official_clip_files(monkeypatch, tmp_path):
    import primus.backends.diffusion.utils.download as download

    args = _finalize({"model": {"name": "wan2.1-i2v", "config": {}}, "data": {"dataset_path": "x"}})
    assert args.dataset["name"] == "amdb"
    assert args.dataset["config"]["modality"] == "video"
    assert args.trainer["args"]["fsdp_transformer_layer_cls_to_wrap"] == "DiTBlock"
    assert args.trainer["args"]["gradient_checkpointing"] is True

    snapshot = tmp_path / "wan"
    snapshot.mkdir()
    for name in (
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    ):
        (snapshot / name).write_bytes(b"w")
    monkeypatch.setattr(download, "_snapshot", lambda *_args, **_kwargs: str(snapshot))
    env = download._companion_paths("wan2.1-i2v", str(tmp_path / "transformer"), {})
    assert env["VAE_CHECKPOINT"] == str(snapshot / "Wan2.1_VAE.pth")
    assert env["TEXT_ENCODER"] == str(snapshot / "models_t5_umt5-xxl-enc-bf16.pth")
    assert env["CLIP_CHECKPOINT"] == str(snapshot / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")


def test_blank_dataset_path_selects_the_wan_default():
    args = _finalize({"model": {"name": "wan2.1-i2v", "config": {}}, "data": {}})
    assert args.dataset["config"]["model_name"] == "wan2.1-i2v"


def test_wan_diffusers_keys_map_onto_the_native_dit():
    import torch

    from primus.backends.diffusion.utils.checkpoint_convert import (
        wan_diffusers_to_native,
    )

    converted = wan_diffusers_to_native(
        {
            "condition_embedder.time_proj.weight": torch.zeros(1),
            "blocks.0.attn1.to_q.weight": torch.zeros(1),
            "blocks.0.attn2.add_k_proj.weight": torch.zeros(1),
            "blocks.0.attn2.norm_added_q.weight": torch.zeros(1),
            "blocks.0.ffn.net.2.weight": torch.zeros(1),
            "proj_out.weight": torch.zeros(1),
        }
    )
    assert set(converted) == {
        "time_projection.1.weight",
        "blocks.0.self_attn.q.weight",
        "blocks.0.cross_attn.k_img.weight",
        "blocks.0.ffn.2.weight",
        "head.head.weight",
    }


def test_single_file_diffusers_checkpoint_is_converted_and_checked(tmp_path):
    import torch

    from primus.backends.diffusion.models.registrations.wan import (
        _load_dit_weights_into_module,
    )

    # No parameter matches the converted Diffusers key, so a file checkpoint
    # must raise the same mismatch error as a directory checkpoint.
    dit = torch.nn.Linear(1, 1)
    path = tmp_path / "dit.pth"
    torch.save({"condition_embedder.time_proj.weight": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError, match="Wan Diffusers checkpoint mismatch"):
        _load_dit_weights_into_module(dit, str(path), accept_diffusers=True)


def test_tiny_i2v_forward_uses_the_clip_condition():
    import torch

    from primus.backends.diffusion.models.wan.wan_dit import WanModel

    model = WanModel(
        model_type="i2v",
        patch_size=(1, 2, 2),
        text_len=4,
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=16,
        text_dim=16,
        out_dim=4,
        num_heads=4,
        num_layers=1,
        qk_norm=False,
        cross_attn_norm=False,
    )
    model.eval()
    frames = model(
        [torch.randn(2, 1, 2, 2)],
        torch.zeros(1),
        [torch.randn(4, 16)],
        seq_len=1,
        y=[torch.randn(2, 1, 2, 2)],
        clip_fea=torch.randn(1, 2, 1280),
    )
    assert frames[0].shape == (4, 1, 2, 2)
    with pytest.raises(ValueError, match="clip_fea"):
        model([torch.randn(2, 1, 2, 2)], torch.zeros(1), [torch.randn(4, 16)], seq_len=1)


def test_denoiser_only_rejects_a_raw_video_batch():
    import torch

    from primus.backends.diffusion.models.wan.train_pipeline import (
        WanFlowMatchTrainPipeline,
    )

    pipeline = WanFlowMatchTrainPipeline()
    components = type("Encoders", (), {"vae": torch.nn.Identity(), "tokenizer": None})()
    with pytest.raises(ValueError, match="BENCHMARK_DENOISER_ONLY"):
        pipeline._maybe_prepare_raw_batch(components, {"videos": ["clip.mp4"], "prompts": ["a"]}, object())


def test_checkpoint_loads_disable_pickle(monkeypatch, tmp_path):
    import torch

    from primus.backends.diffusion.utils.checkpoint_io import load_state_dict

    path = tmp_path / "weights.pth"
    torch.save({"weight": torch.zeros(1)}, path)
    seen = {}
    real_load = torch.load

    def spy(file, *args, **kwargs):
        seen["weights_only"] = kwargs.get("weights_only")
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    assert "weight" in load_state_dict(str(path))
    assert seen["weights_only"] is True


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


def test_sdxl_benchmark_disables_grad_clipping_by_default():
    args = _finalize({"model": {"name": "sdxl", "config": {}}, "data": {"dataset_path": "x"}})
    assert args.trainer["args"]["max_grad_norm"] == 0.0


def test_unregistered_benchmark_presets_are_rejected():
    from primus.backends.diffusion.argument_builder import _UNBUILT_BENCHMARKS

    assert _UNBUILT_BENCHMARKS == set()
    _finalize({"model": {"name": "hunyuan-video", "config": {}}})


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


def test_cached_latents_replace_the_raw_loader(monkeypatch):
    import torch

    from primus.backends.diffusion.trainers.benchmark import AMDBBenchmarkTrainer

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None, raising=False)
    trainer = AMDBBenchmarkTrainer.__new__(AMDBBenchmarkTrainer)
    trainer.rank = 0
    trainer.per_device_train_batch_size = 4
    trainer.args = {"shuffle": False, "dataloader_drop_last": False}
    trainer.sampler = type("Sampler", (), {"set_epoch": lambda self, epoch: None})()
    trainer.processing_class = None
    trainer.dataloader = [{"images": torch.zeros(4, 3, 8, 8), "prompts": ["p"] * 4}]
    trainer._prepare_host_batch = lambda batch: batch
    trainer._prepare_batch = lambda batch, processor: batch

    class _Model:
        def encode_batch(self, batch):
            assert "images" in batch
            count = batch["images"].shape[0]
            return {
                "latents": torch.arange(count).float().view(count, 1),
                "context": torch.zeros(count, 1),
                "pooled": torch.zeros(count, 1),
            }

    trainer.model = _Model()
    trainer._cache_encoded_batches()
    batches = list(trainer.dataloader)
    assert trainer._encoded_cache is True
    assert [batch["latents"].shape[0] for batch in batches] == [4]
    assert "images" not in batches[0]
    assert batches[0]["prompts"] == ["p", "p", "p", "p"]


def test_mochi_uses_video_defaults_and_joint_blocks():
    args = _finalize({"model": {"name": "mochi-1", "config": {}}, "data": {"dataset_path": "x"}})
    assert args.dataset["name"] == "amdb"
    assert args.dataset["config"]["modality"] == "video"
    assert args.trainer["args"]["gradient_checkpointing"] is False
    assert args.trainer["args"]["fsdp_transformer_layer_cls_to_wrap"] == "AsymmetricJointBlock"


def test_mochi_companion_checkpoint_is_the_vae_encoder(tmp_path):
    import primus.backends.diffusion.utils.download as download

    root = tmp_path / "mochi"
    root.mkdir()
    (root / "dit.safetensors").write_bytes(b"dit")
    (root / "encoder.safetensors").write_bytes(b"vae")
    env = download._companion_paths("mochi-1", str(root / "dit.safetensors"), {})
    assert env["VAE_CHECKPOINT"] == str(root / "encoder.safetensors")


def test_mochi_downloads_only_required_model_and_t5_files(monkeypatch, tmp_path):
    import primus.backends.diffusion.utils.download as download

    calls = []

    def fake_snapshot(repo_id, *, revision=None, allow_patterns=None):
        calls.append((repo_id, allow_patterns))
        root = tmp_path / "mochi"
        root.mkdir(exist_ok=True)
        (root / "dit.safetensors").write_bytes(b"dit")
        return str(root)

    monkeypatch.setattr(download, "_snapshot", fake_snapshot)
    assert download.materialize_pretrained("mochi-1", "") == str(tmp_path / "mochi" / "dit.safetensors")
    download.prefetch_huggingface_repo(
        "google/t5-v1_1-xxl", allow_patterns=download._MOCHI_T5_DOWNLOAD_PATTERNS
    )
    assert calls == [
        ("genmo/mochi-1-preview", ["dit.safetensors", "encoder.safetensors"]),
        ("google/t5-v1_1-xxl", download._MOCHI_T5_DOWNLOAD_PATTERNS),
    ]


def test_hunyuan_denoiser_only_downloads_the_transformer(monkeypatch, tmp_path):
    import primus.backends.diffusion.utils.download as download

    calls = []
    data = tmp_path / "data"
    data.mkdir()

    def fake_snapshot(repo_id, *, revision=None, allow_patterns=None):
        del revision
        calls.append((repo_id, allow_patterns))
        root = tmp_path / "hunyuan"
        (root / "transformer").mkdir(parents=True, exist_ok=True)
        return str(root)

    monkeypatch.setenv("PRIMUS_HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("BENCHMARK_DENOISER_ONLY", "1")
    monkeypatch.setattr(download, "_snapshot", fake_snapshot)
    env = download.stage_benchmark_assets("hunyuan-video", dataset_path=str(data), pretrained="")
    assert calls == [("hunyuanvideo-community/HunyuanVideo", ["transformer/*"])]
    assert "LLM_ENCODER" not in env
    assert "VAE_CHECKPOINT" not in env


def test_hunyuan_full_run_still_fetches_encoder_and_vae(monkeypatch, tmp_path):
    import primus.backends.diffusion.utils.download as download

    calls = []
    data = tmp_path / "data"
    data.mkdir()

    def fake_snapshot(repo_id, *, revision=None, allow_patterns=None):
        del revision
        calls.append((repo_id, allow_patterns))
        root = tmp_path / "hunyuan"
        (root / "transformer").mkdir(parents=True, exist_ok=True)
        if allow_patterns and "text_encoder/*" in allow_patterns:
            (root / "text_encoder").mkdir(exist_ok=True)
            (root / "tokenizer").mkdir(exist_ok=True)
        if allow_patterns and "vae/*" in allow_patterns:
            (root / "vae").mkdir(exist_ok=True)
        return str(root)

    monkeypatch.setenv("PRIMUS_HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("BENCHMARK_DENOISER_ONLY", raising=False)
    monkeypatch.setattr(download, "_snapshot", fake_snapshot)
    env = download.stage_benchmark_assets("hunyuan-video", dataset_path=str(data), pretrained="")
    assert calls == [
        ("hunyuanvideo-community/HunyuanVideo", ["transformer/*"]),
        (
            "hunyuanvideo-community/HunyuanVideo",
            ["text_encoder/*", "tokenizer/*", "vae/*"],
        ),
    ]
    assert env["LLM_ENCODER"] == str(tmp_path / "hunyuan" / "text_encoder")
    assert env["VAE_CHECKPOINT"] == str(tmp_path / "hunyuan" / "vae")


def test_hunyuan_companions_already_on_disk_are_not_redownloaded(monkeypatch, tmp_path):
    import primus.backends.diffusion.utils.download as download

    root = tmp_path / "snap"
    (root / "transformer").mkdir(parents=True)
    (root / "text_encoder").mkdir()
    (root / "vae").mkdir()
    monkeypatch.setattr(
        download, "_snapshot", lambda *_args, **_kwargs: pytest.fail("companions are already local")
    )
    env = download._companion_paths("hunyuan-video", str(root / "transformer"), {})
    assert env["LLM_ENCODER"] == str(root / "text_encoder")
    assert env["VAE_CHECKPOINT"] == str(root / "vae")


def test_aiter_jit_dir_follows_package_writability(monkeypatch, tmp_path):
    import importlib.util
    import os
    import sys
    import types

    import primus.backends.diffusion.attention.aiter as aiter_backend

    home = tmp_path / "home"
    user_jit = home / ".aiter" / "jit"
    user_jit.mkdir(parents=True)
    package = tmp_path / "aiter_pkg"
    package_jit = package / "jit"
    package_jit.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_text("")
    spec = importlib.util.spec_from_file_location("aiter_pkg_probe", init)

    monkeypatch.setenv("HOME", str(home))
    # ``_import_aiter`` uses ``setdefault``, which monkeypatch does not track.
    # Register the variable for restore (whether or not it is exported), then
    # clear it so the first assert sees an unset variable.
    monkeypatch.setenv("AITER_JIT_DIR", "")
    os.environ.pop("AITER_JIT_DIR")

    def find_spec(name, *args, **kwargs):
        del args, kwargs
        return spec if name == "aiter" else None

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    fake = types.ModuleType("aiter")
    monkeypatch.setitem(sys.modules, "aiter", fake)

    # The torch unit-test container runs as root. os.access(W_OK) is true for
    # uid 0 even on a mode 0555 directory, so chmod cannot stand in for a
    # read-only aiter package there.
    state = {"writable": False}
    real_access = os.access

    def access(path, mode):
        if os.path.abspath(path) == str(package_jit) and mode == os.W_OK:
            return state["writable"]
        return real_access(path, mode)

    monkeypatch.setattr(os, "access", access)

    aiter_backend._import_aiter()
    assert os.environ["AITER_JIT_DIR"] == str(user_jit)

    monkeypatch.setenv("AITER_JIT_DIR", "/already/set")
    aiter_backend._import_aiter()
    assert os.environ["AITER_JIT_DIR"] == "/already/set"

    monkeypatch.delenv("AITER_JIT_DIR")
    state["writable"] = True
    aiter_backend._import_aiter()
    assert "AITER_JIT_DIR" not in os.environ


def test_synthetic_mochi_dataset_skips_asset_downloads(monkeypatch, tmp_path):
    import primus.backends.diffusion.utils.download as download

    monkeypatch.setenv("PRIMUS_HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr(
        download,
        "materialize_pretrained",
        lambda *_args, **_kwargs: pytest.fail("synthetic runs must not download checkpoints"),
    )
    assert (
        download.stage_benchmark_assets(
            "mochi-1",
            dataset_path="synthetic:mochi",
            pretrained="",
        ).get("PRETRAINED_PATH")
        is None
    )


def test_mochi_packed_attention_excludes_padding_tokens():
    import torch

    from primus.backends.diffusion.models.mochi.train_pipeline import MochiTrainPipeline

    class _Dit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.packed_indices = None

        def forward(self, latents, _sigma, **kwargs):
            self.packed_indices = kwargs["packed_indices"]
            return torch.zeros_like(latents) * self.weight

    dit = _Dit()
    pipeline = MochiTrainPipeline()
    pipeline.compute_loss(
        dit=dit,
        encoder=None,
        text_encoder=None,
        batch={
            "latents": torch.zeros(2, 12, 1, 2, 2),
            "text_hidden": torch.zeros(2, 3, 4),
            "text_mask": torch.tensor([[True, True, False], [True, False, False]]),
        },
    )
    assert torch.equal(dit.packed_indices["cu_seqlens_kv"], torch.tensor([0, 3, 5]))
    # torch.equal ignores dtype. AITER's varlen kernel rejects int64 boundaries.
    assert dit.packed_indices["cu_seqlens_kv"].dtype == torch.int32
    assert dit.packed_indices["max_seqlen_in_batch_kv"] == 3
    assert torch.equal(dit.packed_indices["valid_token_indices_kv"], torch.tensor([0, 1, 2, 4, 5]))


def _import_mochi_registration():
    import sys
    import types

    if "transformers" not in sys.modules:
        try:
            import importlib

            importlib.import_module("transformers")
        except ModuleNotFoundError:
            transformers = types.ModuleType("transformers")
            transformers.T5EncoderModel = object
            transformers.T5Tokenizer = object
            sys.modules["transformers"] = transformers
    from primus.backends.diffusion.models.registrations import mochi

    return mochi


def test_mochi_checkpoint_loading_rejects_incomplete_weights(monkeypatch):
    import torch

    mochi = _import_mochi_registration()

    monkeypatch.setattr(mochi, "load_matching", lambda *_args, **_kwargs: (1, 0))
    with pytest.raises(RuntimeError, match="missing=1, unexpected=0"):
        mochi._load_required_checkpoint(
            torch.nn.Identity(),
            "checkpoint.safetensors",
            prefixes=(),
            description="Mochi DiT",
        )


def test_relative_checkpoint_path_is_not_a_hub_repo():
    import primus.backends.diffusion.utils.download as download

    with pytest.raises(FileNotFoundError, match="does not exist"):
        download.materialize_pretrained("mochi-1", "ckpts/mochi/dit.safetensors")


def test_data_shift_override_reaches_the_mochi_grid(monkeypatch):
    import torch

    from primus.backends.diffusion.utils.sampling import amdb_flow_sigma_grid

    mochi = _import_mochi_registration()

    class _Tiny(torch.nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    args = _finalize({"model": {"name": "mochi-1", "config": {}}, "data": {"shift": 7}})
    assert args.model["config"]["shift"] == 7

    class _Built:
        def __init__(self, train_pipeline, **_kwargs):
            self.train_pipeline = train_pipeline

        def freeze_except(self):
            return None

    monkeypatch.setenv("BENCHMARK_DENOISER_ONLY", "1")
    monkeypatch.setattr(mochi, "AsymmDiTJoint", _Tiny)
    monkeypatch.setattr(mochi, "MochiForTraining", _Built)
    monkeypatch.setattr(mochi, "count_parameters", lambda _model: (0, 0))
    monkeypatch.setattr(mochi, "materialize_pretrained", lambda *_args, **_kwargs: "")
    pipeline = mochi.build_mochi_model(args.model["config"]).train_pipeline
    assert torch.allclose(pipeline.sigma_grid, amdb_flow_sigma_grid(shift=7.0))


def test_sdpa_fallback_rejects_packed_batches(monkeypatch):
    import torch

    import primus.backends.diffusion.attention as attention
    from primus.backends.diffusion.models.mochi.attn_imports import flash_varlen_attn

    monkeypatch.setattr(attention, "get_attention_backend", lambda: "sdpa")
    query = torch.randn(4, 1, 8)
    boundaries = torch.tensor([0, 2, 4], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="sample boundaries"):
        flash_varlen_attn(
            query,
            query,
            query,
            cu_seqlens_q=boundaries,
            cu_seqlens_k=boundaries,
            max_seqlen_q=2,
            max_seqlen_k=2,
        )


def test_prepare_host_batch_uses_the_processor_before_the_model():
    from primus.backends.diffusion.trainers.benchmark import AMDBBenchmarkTrainer

    calls = []
    trainer = AMDBBenchmarkTrainer.__new__(AMDBBenchmarkTrainer)
    trainer._encoded_cache = None
    trainer.processing_class = type(
        "Processor",
        (),
        {"prepare_host_batch": lambda self, batch: calls.append("processor") or {"ready": True}},
    )()
    trainer.model = type("Model", (), {"prepare_host_batch": lambda self, batch: calls.append("model")})()
    assert trainer._prepare_host_batch({"raw": 1}) == {"ready": True}
    assert calls == ["processor"]

    trainer.processing_class = object()
    trainer._prepare_host_batch({"latents": 1})
    assert calls == ["processor", "model"]


def test_even_batch_loader_is_full_size_and_mlperf_keeps_the_tail(tmp_path, monkeypatch):
    import torch
    from torch.utils.data import Dataset

    from primus.backends.diffusion.trainers.base import (
        BaseWanTrainer,
        ContiguousDistributedSampler,
    )

    class _Rows(Dataset):
        def __len__(self):
            return 20

        def __getitem__(self, index):
            return index

    class _Eval(_Rows):
        def get_collator(self):
            return lambda batch: batch

    class _Sampler:
        def __init__(self, dataset, num_replicas=1, rank=0, shuffle=True):
            del num_replicas, rank, shuffle
            self.indices = list(range(len(dataset)))

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    monkeypatch.setattr(torch.utils.data.distributed, "DistributedSampler", _Sampler)

    class _Trainer(BaseWanTrainer):
        def _apply_parallelism(self):
            return None

        def save_model(self):
            return None

        def _setup_mlperf(self):
            return None

    def build(mlperf: bool):
        return _Trainer(
            torch.nn.Linear(1, 1),
            {
                "output_dir": str(tmp_path / ("mlperf" if mlperf else "even")),
                "per_device_train_batch_size": 16,
                "dataloader_even_batches": True,
                "dataloader_num_workers": 0,
                "mlperf_enable": mlperf,
                "mlperf_train_samples": 20,
                "mlperf_eval_total_samples": 20,
                "report_to": "none",
                "gradient_checkpointing": False,
            },
            _Rows(),
            lambda batch: batch,
            object(),
            rank=0,
            world_size=1,
            local_rank=0,
            eval_dataset=_Eval() if mlperf else None,
        )

    even = build(False)
    assert all(len(batch) == 16 for batch in even.dataloader)
    mlperf = build(True)
    assert isinstance(mlperf.sampler, ContiguousDistributedSampler)
    assert [len(batch) for batch in mlperf.dataloader] == [16, 4]


def test_hunyuan_keeps_checkpointing_off_and_makes_the_vae_optional(tmp_path, monkeypatch):
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
    monkeypatch.setattr(
        download,
        "_snapshot",
        lambda *_args, **_kwargs: pytest.fail("a local Hunyuan snapshot must not be re-downloaded"),
    )
    env = download._companion_paths("hunyuan-video", str(root / "transformer"), {})
    assert env["LLM_ENCODER"] == str(root / "text_encoder")
    assert "VAE_CHECKPOINT" not in env
    (root / "vae").mkdir()
    with_vae = download._companion_paths("hunyuan-video", str(root / "transformer"), {})
    assert with_vae["VAE_CHECKPOINT"] == str(root / "vae")


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

    from primus.backends.diffusion.models.hunyuan.train_pipeline import (
        HunyuanTrainPipeline,
    )

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
