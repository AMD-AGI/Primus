from pathlib import Path

from primus.backends.diffusion.trainers.base import ContiguousDistributedSampler
from primus.backends.diffusion.trainers.fsdp2 import FSDP2Trainer


def test_contiguous_sampler_resumes_at_offset():
    sampler = ContiguousDistributedSampler(list(range(16)), num_replicas=2, rank=1)
    sampler.set_offset(3)

    assert list(sampler) == [11, 12, 13, 14, 15]
    assert len(sampler) == 5


def test_latest_checkpoint_ignores_incomplete_save(tmp_path):
    trainer = FSDP2Trainer.__new__(FSDP2Trainer)
    trainer.output_dir = str(tmp_path)
    for step, complete in ((10, True), (20, True), (30, False)):
        path = tmp_path / f"checkpoint-{step}"
        path.mkdir()
        if complete:
            (path / ".complete").touch()

    assert trainer._latest_checkpoint() == str(tmp_path / "checkpoint-20")


def test_checkpoint_retention_keeps_latest_three(tmp_path):
    trainer = FSDP2Trainer.__new__(FSDP2Trainer)
    trainer.output_dir = str(tmp_path)
    trainer.rank = 0
    trainer.save_total_limit = 3
    for step in range(5):
        path = tmp_path / f"checkpoint-{step}"
        path.mkdir()
        (path / ".complete").touch()

    trainer._prune_checkpoints()

    assert sorted(path.name for path in Path(tmp_path).glob("checkpoint-*")) == [
        "checkpoint-2",
        "checkpoint-3",
        "checkpoint-4",
    ]
