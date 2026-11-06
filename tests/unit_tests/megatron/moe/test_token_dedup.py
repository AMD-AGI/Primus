###############################################################################
# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the MoE token-dedup dispatcher port.

CPU tests cover the dedup plan helpers, the dedup autograd kernels (torch
fallback) and the build_args validation. The EP8 GPU test checks that the dedup
dispatcher (experts-side and combine-side weighting) matches the stock
all-to-all dispatcher bit-for-bit up to summation order.
"""

import functools
from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.core.transformer.moe.token_dedup import (
    expand_token_dedup_received,
    pack_token_dedup_payload,
    permute_token_dedup_expert_output,
    reduce_token_dedup_expert_output,
    restore_token_dedup_output,
)
from primus.backends.megatron.core.transformer.moe.token_dedup_dispatcher import (
    _compact_mask_indices,
    _token_destination_hit_map,
    dedup_weighted_combine_enabled,
)

# --------------------------------------------------------------------------- #
# Dedup plan helpers and kernels (CPU, torch fallback)
# --------------------------------------------------------------------------- #


def test_destination_hit_map_deduplicates_same_rank_experts():
    # Two experts per EP rank: {0, 1} -> rank 0 and {2, 3} -> rank 1.
    indices = torch.tensor([[0, 1], [0, 2], [3, 2], [1, 1]], dtype=torch.long)
    destinations, hits = _token_destination_hit_map(indices, num_local_experts=2, ep_size=2)
    assert torch.equal(destinations, torch.tensor([[0, 0], [0, 1], [1, 1], [0, 0]]))
    assert torch.equal(hits, torch.tensor([[True, True, False, True], [False, True, True, False]]))
    # Eight token/expert assignments become five token/destination wire rows.
    assert hits.sum().item() == 5


def test_destination_major_compaction_needs_no_sort():
    mask = torch.tensor([[True, False, True, False], [False, True, True, False]])
    assert torch.equal(_compact_mask_indices(mask, count=4), torch.tensor([0, 2, 5, 6]))


def test_compaction_handles_empty_mask():
    compact = _compact_mask_indices(torch.zeros((2, 0), dtype=torch.bool), count=0)
    assert compact.dtype == torch.long and compact.numel() == 0


def test_packed_metadata_uses_same_wire_rows_and_preserves_gradients():
    hidden = torch.arange(12, dtype=torch.float32).view(3, 4).requires_grad_()
    probs = torch.tensor([[0.25, 0.75], [0.4, 0.6], [0.125, 0.875]], requires_grad=True)
    wire_tokens = torch.tensor([0, 2, 0])
    wire_experts = torch.tensor([[1, -1], [0, 2], [-1, 3]])
    token_wire_rows = torch.tensor([[0, 2], [-1, -1], [1, -1]])
    slot_wire_rows = torch.tensor([[0, 2], [-1, -1], [1, 1]])

    packed = pack_token_dedup_payload(
        hidden, probs, wire_tokens, wire_experts, token_wire_rows, slot_wire_rows
    )
    assert packed.shape == (3, 8)
    assert torch.equal(packed[:, :4], hidden.detach().index_select(0, wire_tokens))
    assert torch.equal(packed[:, 4:6].long(), wire_experts)
    assert torch.equal(packed[:, 6:8], torch.tensor([[0.25, 0.0], [0.125, 0.875], [0.0, 0.75]]))

    packed.sum().backward()
    assert torch.equal(hidden.grad, torch.tensor([[2.0] * 4, [0.0] * 4, [1.0] * 4]))
    assert torch.equal(probs.grad, torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]))


def test_received_expansion_and_wire_permutation_preserve_gradients():
    received = torch.arange(12, dtype=torch.float32).view(3, 4).requires_grad_()
    wire_rows = torch.tensor([0, 2, 1, 0])
    wire_to_expert_rows = torch.tensor([[0, 3], [2, -1], [1, -1]])
    expanded = expand_token_dedup_received(received, wire_rows, wire_to_expert_rows)
    expanded.sum().backward()
    assert torch.equal(received.grad, torch.tensor([[2.0] * 4, [1.0] * 4, [1.0] * 4]))

    expert_output = torch.arange(16, dtype=torch.float32).view(4, 4).requires_grad_()
    wire_output = permute_token_dedup_expert_output(
        expert_output, torch.tensor([0, 3, 2, 1]), torch.tensor([0, 3, 2, 1])
    )
    (wire_output * torch.arange(1, 5, dtype=torch.float32).unsqueeze(1)).sum().backward()
    assert torch.equal(expert_output.grad[:, 0], torch.tensor([1.0, 4.0, 3.0, 2.0]))


def test_reduce_and_restore_sum_expert_rows():
    expert_output = torch.arange(8, dtype=torch.float32).view(4, 2).requires_grad_()
    wire_to_expert_rows = torch.tensor([[0, 1], [2, -1], [-1, 3]])
    expert_wire_rows = torch.tensor([0, 0, 1, 2])
    token_wire_rows = torch.tensor([[0, -1], [1, 2]])
    wire_tokens = torch.tensor([0, 1, 1])
    partials = reduce_token_dedup_expert_output(expert_output, expert_wire_rows, wire_to_expert_rows)
    output = restore_token_dedup_output(partials, token_wire_rows, wire_tokens)
    e = expert_output.detach()
    assert torch.equal(output, torch.stack((e[0] + e[1], e[2] + e[3])))
    grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    output.backward(grad)
    assert torch.equal(expert_output.grad, grad[[0, 0, 1, 1]])


def test_weighted_combine_condition():
    base = dict(moe_token_dedup_weighted_combine=True)
    assert dedup_weighted_combine_enabled(SimpleNamespace(**base))
    assert not dedup_weighted_combine_enabled(SimpleNamespace(moe_token_dedup_weighted_combine=False))
    assert not dedup_weighted_combine_enabled(SimpleNamespace(**base, moe_apply_probs_on_input=True))
    assert not dedup_weighted_combine_enabled(SimpleNamespace(**base, use_transformer_engine_op_fuser=True))
    # fused bias+act weights inside the kernel unless TE's activation is used
    assert not dedup_weighted_combine_enabled(SimpleNamespace(**base, bias_activation_fusion=True))
    assert dedup_weighted_combine_enabled(
        SimpleNamespace(**base, bias_activation_fusion=True, use_te_activation_func=True)
    )


# --------------------------------------------------------------------------- #
# Patch plumbing (CPU)
# --------------------------------------------------------------------------- #


def _fake_ctx(params: dict, backend: dict, phase="build_args"):
    from primus.core.patches import PatchContext

    return PatchContext(
        backend="megatron",
        phase=phase,
        extra={
            "module_config": SimpleNamespace(params=SimpleNamespace(**params)),
            "backend_args": SimpleNamespace(**backend),
        },
    )


_MEGATRON_DEFAULTS = dict(
    moe_token_dispatcher_type="alltoall",
    moe_expert_capacity_factor=None,
    moe_pad_expert_input_to_capacity=False,
    moe_router_padding_for_quantization=False,
    expert_tensor_parallel_size=1,
    cuda_graph_impl="none",
    num_experts=8,
    expert_model_parallel_size=4,
)


class TestBuildArgsPatch:
    @pytest.mark.parametrize(
        "params,backend_override",
        [
            (dict(moe_token_dedup=True), dict(moe_token_dispatcher_type="flex")),
            (dict(moe_token_dedup=True, use_turbo_deepep=True), {}),
            (dict(moe_token_dedup=True, use_turbo_mega_moe=True), {}),
            (dict(moe_token_dedup=True), dict(moe_expert_capacity_factor=1.0)),
            (dict(moe_token_dedup=True), dict(expert_tensor_parallel_size=2)),
            (dict(moe_token_dedup=True), dict(cuda_graph_impl="local")),
        ],
    )
    def test_rejects_unsupported_settings(self, params, backend_override):
        from primus.backends.megatron.patches.moe_patches.token_dedup_patches import patch_token_dedup_args

        ctx = _fake_ctx(params, {**_MEGATRON_DEFAULTS, **backend_override})
        with pytest.raises(ValueError):
            patch_token_dedup_args(ctx)

    def test_reads_dispatcher_type_from_backend_defaults(self):
        """Keys the YAML never set must come from the Megatron defaults, not fail closed."""
        from primus.backends.megatron.patches.moe_patches.token_dedup_patches import patch_token_dedup_args

        patch_token_dedup_args(_fake_ctx(dict(moe_token_dedup=True), _MEGATRON_DEFAULTS))


# --------------------------------------------------------------------------- #
# EP8 GPU parity tests
# --------------------------------------------------------------------------- #

if torch.cuda.is_available():
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.training.global_vars import set_args
    from megatron.training.initialize import _set_random_seed
    from torch.testing._internal.common_distributed import MultiProcessTestCase, skip_if_lt_x_gpu
    from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize

    from primus.backends.megatron.core.transformer.moe.token_dedup_dispatcher import (
        PrimusDedupAlltoAllTokenDispatcher,
    )

    def _build_config(ep_size, num_moe_experts, topk):
        return TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=ep_size,
            pipeline_model_parallel_size=1,
            num_moe_experts=num_moe_experts,
            moe_ffn_hidden_size=512,
            moe_router_topk=topk,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0.0,
            moe_token_dispatcher_type="alltoall",
            moe_grouped_gemm=False,
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            add_bias_linear=False,
            bias_activation_fusion=False,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )

    def _submodules(num_moe_experts):
        spec = get_gpt_layer_local_spec(num_experts=num_moe_experts, moe_grouped_gemm=False)
        mlp = spec.submodules.mlp
        return mlp.keywords["submodules"] if isinstance(mlp, functools.partial) else mlp.submodules

    def _run_layer(layer, x, grad):
        inp = x.clone().requires_grad_()
        out, _ = layer(inp)
        out.backward(grad)
        return out, inp.grad, layer.router.weight.grad.clone()

    @instantiate_parametrized_tests
    class TestTokenDedupDispatcher(MultiProcessTestCase):
        def setUp(self) -> None:
            super().setUp()
            self._spawn_processes()

        @property
        def world_size(self) -> int:
            return min(8, torch.cuda.device_count())

        @property
        def device(self) -> torch.device:
            return torch.device("cuda", self.rank)

        def _init(self):
            torch.cuda.set_device(self.device)
            store = dist.FileStore(self.file_name, self.world_size)
            dist.init_process_group(backend="nccl", world_size=self.world_size, rank=self.rank, store=store)
            set_args(SimpleNamespace(moe_router_force_load_balancing=False, enable_primus_turbo=False))
            import logging

            from primus.core.utils import logger as _primus_logger

            if _primus_logger._logger is None:
                _primus_logger._logger = logging.getLogger("token_dedup_test")
            parallel_state.destroy_model_parallel()
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=self.world_size,
            )

        def _finish(self):
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()

        def _new_layer(self, config, num_experts):
            _set_random_seed(seed_=123, data_parallel_random_init=False)
            layer = MoELayer(config, _submodules(num_experts)).cuda()
            layer.set_layer_number(0)
            return layer

        def _inputs(self, config):
            gen = torch.Generator(device="cuda").manual_seed(self.rank)
            x = torch.randn(64, 2, config.hidden_size, device="cuda", generator=gen)
            grad = torch.randn(x.shape, device="cuda", generator=gen)
            return x, grad

        @skip_if_lt_x_gpu(2)
        @parametrize("topk", [2, 6])
        @parametrize("weighted_combine", [False, True])
        def test_dedup_matches_alltoall_dispatcher(self, topk, weighted_combine):
            self._init()
            try:
                num_experts = 32
                config = _build_config(self.world_size, num_experts, topk)
                config.moe_token_dedup = True
                config.moe_token_dedup_weighted_combine = weighted_combine
                ref = self._new_layer(config, num_experts)
                dedup = self._new_layer(config, num_experts)
                dedup.load_state_dict(ref.state_dict())
                pg = ProcessGroupCollection.use_mpu_process_groups()
                dedup.token_dispatcher = PrimusDedupAlltoAllTokenDispatcher(
                    dedup.num_local_experts, dedup.local_expert_indices, config=config, pg_collection=pg
                )
                assert dedup.token_dispatcher.weighted_combine == weighted_combine
                if weighted_combine:
                    from primus.backends.megatron.patches.moe_patches.token_dedup_patches import (
                        _wrap_forward_probs_optional,
                    )

                    _wrap_forward_probs_optional(type(dedup.experts), "SequentialMLP")
                x, grad = self._inputs(config)
                ref_out = _run_layer(ref, x, grad)
                got_out = _run_layer(dedup, x, grad)
                for r, g in zip(ref_out, got_out):
                    torch.testing.assert_close(g, r, rtol=1e-4, atol=1e-4)
            finally:
                self._finish()
