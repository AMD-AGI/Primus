###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU-only tests for rank-local Energon dataloader checkpoint restore."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.data import energon_dataset_provider as provider_module
from primus.backends.megatron.data.dataloader import (
    DATALOADER_STATE_FORMAT_VERSION,
    DATALOADER_STATE_KEY,
    DATALOADER_STATE_PAYLOAD_KEY,
    MegatronDataloaderWrapper,
    restore_dataloader_state_from_checkpoint,
)
from primus.backends.megatron.data.energon_dataset_provider import (
    EnergonDatasetProvider,
)


class _StatefulLoader:
    def __init__(self):
        self.position = 0
        self.restored_states = []

    def __iter__(self):
        return iter([self.position])

    def restore_state_rank(self, state):
        self.restored_states.append(state)
        self.position = state["position"]


class _StatelessLoader:
    def __iter__(self):
        return iter([0])


class _SequentialStatefulLoader:
    def __init__(self):
        self.position = 0

    def __iter__(self):
        while True:
            sample_id = self.position
            self.position += 1
            yield sample_id

    def save_state_rank(self):
        return {"position": self.position}

    def restore_state_rank(self, state):
        self.position = state["position"]


def test_restore_dataloader_state_uses_rank_specific_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "train_dataloader_dprank003.pt"
    checkpoint_path.touch()
    calls = []

    def checkpoint_name(root, iteration, **kwargs):
        calls.append((root, iteration, kwargs))
        return str(checkpoint_path)

    loader = _StatefulLoader()
    wrapper = MegatronDataloaderWrapper(loader)

    restored_path = restore_dataloader_state_from_checkpoint(
        wrapper,
        str(tmp_path),
        iteration=17,
        data_parallel_rank=3,
        checkpoint_name_fn=checkpoint_name,
        load_fn=lambda path: {"dataloader_state_dict": {"position": 41}},
    )

    assert restored_path == checkpoint_path
    assert loader.restored_states == [{"position": 41}]
    assert next(wrapper) == 41
    assert calls == [
        (
            str(tmp_path),
            17,
            {
                "tensor_rank": 0,
                "pipeline_rank": 0,
                "basename": "train_dataloader_dprank003.pt",
            },
        )
    ]


def test_restore_dataloader_state_fails_when_rank_file_is_missing(tmp_path):
    missing_path = tmp_path / "missing.pt"
    wrapper = MegatronDataloaderWrapper(_StatefulLoader())

    with pytest.raises(FileNotFoundError, match="data_parallel_rank=2"):
        restore_dataloader_state_from_checkpoint(
            wrapper,
            str(tmp_path),
            iteration=8,
            data_parallel_rank=2,
            checkpoint_name_fn=lambda *args, **kwargs: str(missing_path),
        )


def test_restore_dataloader_state_fails_when_payload_key_is_missing(tmp_path):
    checkpoint_path = tmp_path / "state.pt"
    checkpoint_path.touch()
    wrapper = MegatronDataloaderWrapper(_StatefulLoader())

    with pytest.raises(KeyError, match="dataloader_state_dict"):
        restore_dataloader_state_from_checkpoint(
            wrapper,
            str(tmp_path),
            iteration=8,
            data_parallel_rank=0,
            checkpoint_name_fn=lambda *args, **kwargs: str(checkpoint_path),
            load_fn=lambda path: {"wrong_key": {}},
        )


def test_restore_dataloader_state_rejects_empty_state(tmp_path):
    checkpoint_path = tmp_path / "state.pt"
    checkpoint_path.touch()
    wrapper = MegatronDataloaderWrapper(_StatefulLoader())

    with pytest.raises(RuntimeError, match="empty state"):
        restore_dataloader_state_from_checkpoint(
            wrapper,
            str(tmp_path),
            iteration=8,
            data_parallel_rank=0,
            checkpoint_name_fn=lambda *args, **kwargs: str(checkpoint_path),
            load_fn=lambda path: {DATALOADER_STATE_KEY: None},
        )


def test_restore_dataloader_state_rejects_changed_data_parallel_topology():
    source = MegatronDataloaderWrapper(
        _SequentialStatefulLoader(),
        data_parallel_rank=3,
        data_parallel_world_size=8,
    )
    state = source.save_state()

    assert state == {
        "format_version": DATALOADER_STATE_FORMAT_VERSION,
        "data_parallel_rank": 3,
        "data_parallel_world_size": 8,
        DATALOADER_STATE_PAYLOAD_KEY: {"position": 0},
    }

    matching = MegatronDataloaderWrapper(
        _SequentialStatefulLoader(),
        data_parallel_rank=3,
        data_parallel_world_size=8,
    )
    assert matching.restore_state(state, strict=True)

    changed_world_size = MegatronDataloaderWrapper(
        _SequentialStatefulLoader(),
        data_parallel_rank=3,
        data_parallel_world_size=4,
    )
    with pytest.raises(RuntimeError, match="topology does not match"):
        changed_world_size.restore_state(state, strict=True)


def test_restore_dataloader_state_requires_restore_capability(tmp_path):
    checkpoint_path = tmp_path / "state.pt"
    checkpoint_path.touch()
    wrapper = MegatronDataloaderWrapper(_StatelessLoader())

    with pytest.raises(RuntimeError, match="does not support restore_state_rank"):
        restore_dataloader_state_from_checkpoint(
            wrapper,
            str(tmp_path),
            iteration=8,
            data_parallel_rank=0,
            checkpoint_name_fn=lambda *args, **kwargs: str(checkpoint_path),
            load_fn=lambda path: {"dataloader_state_dict": {}},
        )


def test_restore_dataloader_state_real_path_and_torch_round_trip(tmp_path, monkeypatch):
    from megatron.training import checkpointing

    monkeypatch.setattr(checkpointing.mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(checkpointing.mpu, "get_expert_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(checkpointing.mpu, "get_expert_model_parallel_rank", lambda: 0)

    uninterrupted_loader = _SequentialStatefulLoader()
    uninterrupted = MegatronDataloaderWrapper(uninterrupted_loader)
    assert [next(uninterrupted) for _ in range(5)] == list(range(5))
    state = uninterrupted.save_state()
    expected_tail = [next(uninterrupted) for _ in range(5)]

    checkpoint_path = Path(
        checkpointing.get_checkpoint_name(
            str(tmp_path),
            5,
            pipeline_parallel=False,
            tensor_rank=0,
            pipeline_rank=0,
            expert_parallel=False,
            expert_rank=0,
            basename="train_dataloader_dprank000.pt",
        )
    )
    checkpoint_path.parent.mkdir(parents=True)
    torch.save({DATALOADER_STATE_KEY: state}, checkpoint_path)

    resumed = MegatronDataloaderWrapper(_SequentialStatefulLoader())
    restored_path = restore_dataloader_state_from_checkpoint(
        resumed,
        str(tmp_path),
        iteration=5,
        data_parallel_rank=0,
    )

    assert restored_path == checkpoint_path
    assert [next(resumed) for _ in range(5)] == expected_tail == list(range(5, 10))


def test_provider_restores_before_first_resumed_batch(monkeypatch, tmp_path):
    provider = EnergonDatasetProvider(lambda: None)
    loader = MegatronDataloaderWrapper(_SequentialStatefulLoader())
    calls = []

    def restore(dataloader, checkpoint_root, iteration, data_parallel_rank):
        calls.append((checkpoint_root, iteration, data_parallel_rank))
        dataloader.restore_state({"position": 5}, strict=True)
        return tmp_path / "train_dataloader_dprank003.pt"

    monkeypatch.setattr(provider_module.parallel_state, "get_data_parallel_rank", lambda: 3)
    monkeypatch.setattr(provider_module, "restore_dataloader_state_from_checkpoint", restore)
    args = SimpleNamespace(
        load="/checkpoints",
        dataloader_save="/dataloader-state",
        iteration=5,
        finetune=False,
        require_dataloader_restore=True,
    )

    restored_path = provider._restore_train_dataloader_state(args, loader)

    assert restored_path == str(tmp_path / "train_dataloader_dprank003.pt")
    assert calls == [("/dataloader-state", 5, 3)]
    assert next(loader) == 5


def test_provider_skips_restore_when_exact_continuation_is_not_required(
    monkeypatch,
):
    provider = EnergonDatasetProvider(lambda: None)
    loader = MegatronDataloaderWrapper(_SequentialStatefulLoader())
    args = SimpleNamespace(
        load="/checkpoints",
        dataloader_save="/dataloader-state",
        iteration=5,
        finetune=False,
        require_dataloader_restore=False,
    )

    def unexpected_restore(*args, **kwargs):
        raise AssertionError("optional resume must not partially restore rank state")

    monkeypatch.setattr(
        provider_module,
        "restore_dataloader_state_from_checkpoint",
        unexpected_restore,
    )

    assert provider._restore_train_dataloader_state(args, loader) is None
    assert next(loader) == 0


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (
            SimpleNamespace(
                load=None,
                iteration=0,
                finetune=False,
                require_dataloader_restore=False,
            ),
            None,
        ),
        (
            SimpleNamespace(
                load="/checkpoints",
                iteration=5,
                finetune=True,
                require_dataloader_restore=False,
            ),
            None,
        ),
        (
            SimpleNamespace(
                load="/checkpoints",
                iteration=0,
                finetune=False,
                require_dataloader_restore=False,
            ),
            None,
        ),
        (
            SimpleNamespace(
                load=None,
                iteration=0,
                finetune=False,
                require_dataloader_restore=True,
            ),
            RuntimeError,
        ),
        (
            SimpleNamespace(
                load="/checkpoints",
                iteration=5,
                finetune=True,
                require_dataloader_restore=True,
            ),
            RuntimeError,
        ),
        (
            SimpleNamespace(
                load="/checkpoints",
                iteration=0,
                finetune=False,
                require_dataloader_restore=True,
            ),
            RuntimeError,
        ),
    ],
)
def test_provider_classifies_resume_state(args, error):
    provider = EnergonDatasetProvider(lambda: None)
    loader = MegatronDataloaderWrapper(_StatefulLoader())

    if error is None:
        assert provider._restore_train_dataloader_state(args, loader) is None
    else:
        with pytest.raises(error, match="successfully loaded"):
            provider._restore_train_dataloader_state(args, loader)


@pytest.mark.parametrize("required", [False, True])
def test_provider_handles_missing_dataloader_save(required):
    provider = EnergonDatasetProvider(lambda: None)
    loader = MegatronDataloaderWrapper(_StatefulLoader())
    args = SimpleNamespace(
        load="/checkpoints",
        iteration=5,
        finetune=False,
        require_dataloader_restore=required,
    )

    if required:
        with pytest.raises(RuntimeError, match="without dataloader_save"):
            provider._restore_train_dataloader_state(args, loader)
    else:
        assert provider._restore_train_dataloader_state(args, loader) is None


def test_provider_propagates_missing_rank_state(monkeypatch):
    provider = EnergonDatasetProvider(lambda: None)
    loader = MegatronDataloaderWrapper(_StatefulLoader())
    args = SimpleNamespace(
        load="/checkpoints",
        dataloader_save="/dataloader-state",
        iteration=5,
        finetune=False,
        require_dataloader_restore=True,
    )

    monkeypatch.setattr(provider_module.parallel_state, "get_data_parallel_rank", lambda: 7)

    def missing(*args, **kwargs):
        raise FileNotFoundError("train_dataloader_dprank007.pt")

    monkeypatch.setattr(provider_module, "restore_dataloader_state_from_checkpoint", missing)
    with pytest.raises(FileNotFoundError, match="dprank007"):
        provider._restore_train_dataloader_state(args, loader)


def test_provider_rejects_non_boolean_required_flag():
    provider = EnergonDatasetProvider(lambda: None)
    args = SimpleNamespace(require_dataloader_restore="true")

    with pytest.raises(TypeError, match="must be a boolean"):
        provider._restore_train_dataloader_state(args, MegatronDataloaderWrapper(_StatefulLoader()))


@pytest.mark.parametrize(
    ("iteration", "data_parallel_rank", "message"),
    [
        (-1, 0, "iteration must be non-negative"),
        (0, -1, "data_parallel_rank must be non-negative"),
    ],
)
def test_restore_dataloader_state_rejects_invalid_coordinates(
    tmp_path: Path, iteration: int, data_parallel_rank: int, message: str
):
    with pytest.raises(ValueError, match=message):
        restore_dataloader_state_from_checkpoint(
            MegatronDataloaderWrapper(_StatefulLoader()),
            str(tmp_path),
            iteration=iteration,
            data_parallel_rank=data_parallel_rank,
        )
