# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for FSDP2 transformer_impl wrapping logic.

Tests that PrimusTorchFullyShardedDataParallel correctly excludes/includes
ColumnParallelLinear based on transformer_impl setting, aligning with Megatron's pattern.
"""

from unittest.mock import patch

import pytest
import torch
from megatron.core import tensor_parallel
from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
from megatron.core.transformer.transformer_config import TransformerConfig

from primus.backends.megatron.core.distributed import (
    torch_fully_sharded_data_parallel as fsdp_mod,
)
from primus.backends.megatron.core.distributed.torch_fully_sharded_data_parallel import (
    PrimusTorchFullyShardedDataParallel,
)
from tests.utils import PrimusUT


class TestFSDP2TransformerImpl(PrimusUT):
    """Tests for FSDP2 wrapping logic with transformer_impl."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        """Initialize parallel state for FSDP2 tests."""

    def _build_and_collect_wrapped(self, transformer_impl):
        """Construct the FSDP2 wrapper with fully_shard mocked at the call site.

        Returns the list of module types that fully_shard was invoked on. The
        mock must patch the name bound INSIDE the production module (it does
        ``from torch.distributed.fsdp import fully_shard`` at import time), not
        ``torch.distributed.fsdp.fully_shard`` -- otherwise the real kernel runs
        and the wrapping decision is never exercised.
        """

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                from megatron.core.tensor_parallel.layers import ColumnParallelLinear

                inner = TransformerConfig(
                    hidden_size=128,
                    num_attention_heads=2,
                    num_layers=1,
                    transformer_impl=transformer_impl,
                )
                self.linear = ColumnParallelLinear(
                    inner.hidden_size,
                    inner.hidden_size,
                    config=inner,
                    init_method=lambda x: None,
                )

        model = MockModel().cuda()
        config = TransformerConfig(
            hidden_size=128,
            num_attention_heads=2,
            num_layers=1,
            transformer_impl=transformer_impl,
        )
        ddp_config = DistributedDataParallelConfig()

        wrapped_modules = []

        def mock_fully_shard(module, **kwargs):
            wrapped_modules.append(type(module))

        with patch.object(fsdp_mod, "fully_shard", side_effect=mock_fully_shard):
            PrimusTorchFullyShardedDataParallel(
                config=config,
                ddp_config=ddp_config,
                module=model,
            )
        return wrapped_modules

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_fsdp2_excludes_column_parallel_linear_when_local(self):
        """ColumnParallelLinear must be excluded when transformer_impl == 'local'."""
        if not fsdp_mod.HAVE_FSDP:
            pytest.skip("torch.distributed.fsdp (FSDP2) is unavailable")

        wrapped_modules = self._build_and_collect_wrapped("local")

        assert (
            tensor_parallel.ColumnParallelLinear not in wrapped_modules
        ), "ColumnParallelLinear should be excluded when transformer_impl == 'local'"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_fsdp2_includes_column_parallel_linear_when_not_local(self):
        """Positive control: ColumnParallelLinear IS wrapped for non-local impl."""
        if not fsdp_mod.HAVE_FSDP:
            pytest.skip("torch.distributed.fsdp (FSDP2) is unavailable")

        wrapped_modules = self._build_and_collect_wrapped("transformer_engine")

        assert (
            tensor_parallel.ColumnParallelLinear in wrapped_modules
        ), "ColumnParallelLinear should be wrapped when transformer_impl != 'local'"
