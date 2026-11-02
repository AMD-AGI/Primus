from __future__ import annotations

from pathlib import Path

import pytest

from primus.backends.diffusion.argument_builder import DiffusionArgBuilder
from primus.backends.diffusion.data.benchmark import AMDBProcessor
from primus.backends.diffusion.registry import DATASET_BUILDERS, MODEL_BUILDERS

BENCHMARK_NAMES = ("wan2.1-i2v",)


def _finalize(params: dict):
    builder = DiffusionArgBuilder()
    builder.update(params)
    return builder.finalize()


def test_benchmark_workloads_are_registered():
    for name in BENCHMARK_NAMES:
        assert name in MODEL_BUILDERS
    assert "amdb" in DATASET_BUILDERS


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


def test_benchmark_argument_builder_rejects_sequence_parallelism():
    with pytest.raises(ValueError, match="sp_size"):
        _finalize(
            {
                "model": {"name": "wan2.1-i2v", "config": {}},
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
    path = download.materialize_dataset("wan2.1-i2v", "")
    home = tmp_path / "hf"
    assert Path(path) == home / "datasets" / "Wild-Heart" / "Disney-VideoGeneration-Dataset"
    assert calls == {
        "repo": "Wild-Heart/Disney-VideoGeneration-Dataset",
        "revision": "refs/pr/4",
    }
    assert home / "hub" not in Path(path).parents

    weights = tmp_path / "transformer"
    weights.write_bytes(b"x")

    def fail_download(*_args, **_kwargs):
        raise AssertionError("checkpoint download should not run")

    monkeypatch.setattr(download, "_snapshot", fail_download)
    assert download.materialize_pretrained("wan2.1-i2v", str(weights)) == str(weights)

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


def test_unbuilt_benchmark_names_fail_before_download():
    for name in ("sdxl", "mochi-1", "hunyuan-video", "flux-dev"):
        with pytest.raises(ValueError, match="wan2.1-i2v"):
            _finalize({"model": {"name": name, "config": {}}})


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


def test_benchmark_fsdp2_hooks_extend_the_parent_wrapper():
    import torch

    from primus.backends.diffusion.trainers.benchmark import AMDBBenchmarkTrainer
    from primus.backends.diffusion.trainers.fsdp2 import FSDP2Trainer

    assert AMDBBenchmarkTrainer._apply_fsdp2 is FSDP2Trainer._apply_fsdp2
    trainer = AMDBBenchmarkTrainer.__new__(AMDBBenchmarkTrainer)
    trainer.args = {"fsdp2_pure_low_precision": True}
    trainer.rank = 0
    assert trainer._fsdp2_reduce_dtype_name(torch.bfloat16) == "bf16"
    parent = FSDP2Trainer.__new__(FSDP2Trainer)
    assert parent._fsdp2_reduce_dtype_name(torch.bfloat16) == "fp32"


def test_benchmark_update_fps_counts_every_micro_batch(monkeypatch):
    import torch

    from primus.backends.diffusion.trainers.benchmark import AMDBBenchmarkTrainer

    class _Event:
        def __init__(self, enable_timing=False):
            del enable_timing

        def record(self):
            return None

        def elapsed_time(self, _other):
            return 2000.0

    class _Loader:
        def __iter__(self):
            yield {"latents": torch.ones(4, 1)}
            yield {"latents": torch.ones(4, 1)}

        def __len__(self):
            return 2

    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    weight = torch.nn.Parameter(torch.ones(1))
    logged = {}
    trainer = AMDBBenchmarkTrainer.__new__(AMDBBenchmarkTrainer)
    trainer.rank = 0
    trainer.world_size = 1
    trainer.grad_accum_steps = 2
    trainer.per_device_train_batch_size = 4
    trainer.num_train_epochs = 1
    trainer.global_step = 0
    trainer.total_steps = 1
    trainer.max_steps = 1
    trainer.save_steps = 0
    trainer.mlperf_enabled = False
    trainer.use_wandb = False
    trainer.max_grad_norm = 0.0
    trainer.args = {"compile_transformer_blocks": True, "cache_latents": False}
    trainer._encoded_cache = True
    trainer.dataloader = _Loader()
    trainer.sampler = type("Sampler", (), {"set_epoch": lambda self, epoch: None})()
    trainer.model = torch.nn.Linear(1, 1)
    trainer.optimizer = torch.optim.SGD([weight], lr=0.1)
    trainer.lr_scheduler = type("Schedule", (), {"step": lambda self: None})()

    def compute_loss(batch):
        return (weight * batch["latents"].sum()).sum()

    trainer.compute_loss = compute_loss
    trainer._log_step = lambda *_args, **kwargs: logged.update(kwargs)
    trainer.train()
    assert logged["fps_gpu"] == 4.0
