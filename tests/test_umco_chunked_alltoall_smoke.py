from __future__ import annotations

import torch

from primus.moe_umco.dispatcher import _all_to_all_single_chunked


def test_chunked_all_to_all_single_smoke(monkeypatch):
    class _FakeDist:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def get_world_size(group=None) -> int:
            _ = group
            return 2

        @staticmethod
        def all_to_all_single(
            output,
            input,
            output_split_sizes=None,
            input_split_sizes=None,
            group=None,
            async_op=False,
        ):
            _ = output_split_sizes, input_split_sizes, group
            output.copy_(input)
            if async_op:

                class _Work:
                    @staticmethod
                    def wait():
                        return None

                return _Work()
            return None

    monkeypatch.setattr(torch.distributed, "is_available", _FakeDist.is_available)
    monkeypatch.setattr(torch.distributed, "is_initialized", _FakeDist.is_initialized)
    monkeypatch.setattr(torch.distributed, "get_world_size", _FakeDist.get_world_size)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", _FakeDist.all_to_all_single)

    x = torch.arange(10, dtype=torch.float32).view(5, 2)
    out = _all_to_all_single_chunked(
        group=None,
        input_tensor=x,
        output_splits=[3, 2],
        input_splits=[3, 2],
        chunk_tokens=2,
        max_inflight=2,
        phase="dispatch",
        expert_compute_fn=None,
    )
    assert torch.equal(out, x)
