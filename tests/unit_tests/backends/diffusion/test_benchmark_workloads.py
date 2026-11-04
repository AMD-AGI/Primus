from __future__ import annotations

from pathlib import Path

import pytest

from primus.backends.diffusion.argument_builder import DiffusionArgBuilder
from primus.backends.diffusion.data.benchmark import AMDBProcessor
from primus.backends.diffusion.registry import DATASET_BUILDERS, MODEL_BUILDERS

BENCHMARK_NAMES = ("mochi-1",)


def _finalize(params: dict):
    builder = DiffusionArgBuilder()
    builder.update(params)
    return builder.finalize()


def test_benchmark_workloads_are_registered():
    for name in BENCHMARK_NAMES:
        assert name in MODEL_BUILDERS
    assert "amdb" in DATASET_BUILDERS


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


def test_benchmark_argument_builder_rejects_sequence_parallelism():
    with pytest.raises(ValueError, match="sp_size"):
        _finalize(
            {
                "model": {"name": "mochi-1", "config": {}},
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

    def fake_snapshot(repo_id, *, revision=None, allow_patterns=None):
        calls["repo"] = repo_id
        calls["revision"] = revision
        folder = download.dataset_folder(repo_id)
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder)

    monkeypatch.setattr(download, "_snapshot_dataset", fake_snapshot)
    path = download.materialize_dataset("mochi-1", "")
    home = tmp_path / "hf"
    assert Path(path) == home / "datasets" / "Wild-Heart" / "Disney-VideoGeneration-Dataset"
    assert calls == {
        "repo": "Wild-Heart/Disney-VideoGeneration-Dataset",
        "revision": "refs/pr/4",
    }
    assert home / "hub" not in Path(path).parents

    weights = tmp_path / "dit.safetensors"
    weights.write_bytes(b"x")

    def fail_download(*_args, **_kwargs):
        raise AssertionError("checkpoint download should not run")

    monkeypatch.setattr(download, "_snapshot", fail_download)
    assert download.materialize_pretrained("mochi-1", str(weights)) == str(weights)

    data_root = tmp_path / "data"
    monkeypatch.delenv("PRIMUS_HF_HOME", raising=False)
    monkeypatch.setenv("DATA_PATH", str(data_root))
    monkeypatch.setenv("HF_HOME", str(data_root / "huggingface"))
    assert download.huggingface_home() == Path.home() / "cache" / "huggingface"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "custom"))
    assert download.huggingface_home() == tmp_path / "custom"


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
    for name in ("sdxl", "hunyuan-video", "wan2.1-i2v", "flux-dev"):
        with pytest.raises(ValueError, match="mochi-1"):
            _finalize({"model": {"name": name, "config": {}}})


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


def test_collate_encoded_stacks_latents_and_keeps_prompts():
    import torch

    from primus.backends.diffusion.trainers.benchmark import _collate_encoded

    batch = _collate_encoded(
        [
            {"latents": torch.ones(2), "prompt": "a"},
            {"latents": torch.ones(2) * 2, "prompt": "b"},
        ]
    )
    assert batch["prompts"] == ["a", "b"]
    assert torch.equal(batch["latents"], torch.tensor([[1.0, 1.0], [2.0, 2.0]]))


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
