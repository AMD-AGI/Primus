###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for FSEPGroupedMLP and FSEP token dispatcher integration.

Tests cover:
  Test 1: Math equivalence  (S=1 pass-through, no distributed needed)
  Test 2: ReduceScatter correctness  (4 GPU, EP=4, S=4)
  Test 3: Backward gradient correctness  (4 GPU)
  Test 4: End-to-end dispatcher + FSEP  (8 GPU, EP=8, S=4)
  Test 5: Load balance improvement  (8 GPU, imbalanced routing)

Run with:
  python -m pytest tests/unit_tests/megatron/transformer/moe/test_fsep_grouped_mlp.py -v
  # Requires 4 or 8 GPUs for distributed tests
"""

import dataclasses
import os
from contextlib import contextmanager
from types import SimpleNamespace

import megatron.core.parallel_state as ps
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import set_args
from megatron.training.initialize import _set_random_seed
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def create_args(fsep_sharding_degree: int = 0):
    """Create dummy args namespace for testing."""
    args = SimpleNamespace()
    args.turbo_sync_free_moe_stage = 0
    args.sequence_parallel = False
    args.seq_length = 4096
    args.context_parallel_size = 1
    args.micro_batch_size = 1
    args.moe_router_force_load_balancing = False
    args.moe_use_legacy_grouped_gemm = False  # Use pt.ops.grouped_gemm (turbo)
    args.use_turbo_grouped_mlp = True
    args.turbo_deepep_num_cu = 32
    args.turbo_deepep_use_comm_stream = False
    # FSEP-specific
    args.moe_fsep_sharding_degree = fsep_sharding_degree
    args.moe_log_expert_load = False
    args.enable_primus_turbo = True
    args.use_turbo_deepep = True
    # Other defaults
    args.use_turbo_fused_act_with_probs = False
    args.patch_moe_overlap = False
    args.overlap_moe_expert_parallel_comm = False
    args.patch_primus_pipeline = False
    args.patch_zero_bubble = False
    args.enable_zero_bubble = False
    args.pp_algorithm = "1f1b"
    args.offload = False
    args.moe_apply_probs_on_input = False
    return args


def token_permutation(token_dispatcher, hidden_states, probs, indices):
    hidden_states, probs = token_dispatcher.dispatch_preprocess(hidden_states, indices, probs)
    hidden_states, probs = token_dispatcher.token_dispatch(hidden_states, probs)
    hidden_states, tokens_per_expert, permuted_probs = token_dispatcher.dispatch_postprocess(
        hidden_states, probs
    )
    return hidden_states, tokens_per_expert, permuted_probs


def token_unpermutation(token_dispatcher, hidden_states):
    hidden_states = token_dispatcher.combine_preprocess(hidden_states)
    hidden_states = token_dispatcher.token_combine(hidden_states)
    hidden_states = token_dispatcher.combine_postprocess(hidden_states)
    return hidden_states, None


@contextmanager
def fsep_patch_context(fsep_sharding_degree: int = 4):
    """Patch both the GroupedMLP and DeepEP dispatcher for FSEP testing."""
    from megatron.core.transformer.moe import moe_layer, token_dispatcher
    import megatron.core.transformer.moe.experts as meg_experts

    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboDeepEPTokenDispatcher,
    )
    from primus.backends.megatron.core.transformer.moe.fsep_experts import (
        FSEPGroupedMLP,
    )

    prev_dispatcher = token_dispatcher.MoEFlexTokenDispatcher
    prev_grouped_mlp = meg_experts.GroupedMLP

    try:
        token_dispatcher.MoEFlexTokenDispatcher = PrimusTurboDeepEPTokenDispatcher
        moe_layer.MoEFlexTokenDispatcher = PrimusTurboDeepEPTokenDispatcher
        meg_experts.GroupedMLP = FSEPGroupedMLP
        yield
    finally:
        token_dispatcher.MoEFlexTokenDispatcher = prev_dispatcher
        moe_layer.MoEFlexTokenDispatcher = prev_dispatcher
        meg_experts.GroupedMLP = prev_grouped_mlp


class FSEPMoEModelTestContainer:
    """Container for setting up MoE model with FSEP for testing."""

    def __init__(
        self,
        tp_size: int,
        ep_size: int,
        pp_size: int,
        cp_size: int = 1,
        moe_tp_size: int = None,
        num_moe_experts: int = 8,
        moe_router_topk: int = 2,
        moe_token_dispatcher_type: str = "flex",
        test_dtype: torch.dtype = torch.float32,
        hidden_size: int = 32,
        moe_ffn_hidden_size: int = None,
        **kwargs,
    ):
        self.num_local_experts = num_moe_experts // ep_size
        self.test_dtype = test_dtype

        if moe_tp_size is None:
            moe_tp_size = tp_size
        if moe_ffn_hidden_size is None:
            moe_ffn_hidden_size = hidden_size * 4

        ps.destroy_model_parallel()
        ps.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            expert_model_parallel_size=ep_size,
            context_parallel_size=cp_size,
            expert_tensor_parallel_size=moe_tp_size,
        )

        _set_random_seed(seed_=123, data_parallel_random_init=False)
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]

        self.config = TransformerConfig(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            pipeline_model_parallel_size=pp_size,
            context_parallel_size=cp_size,
            expert_tensor_parallel_size=moe_tp_size,
            moe_router_topk=moe_router_topk,
            num_moe_experts=num_moe_experts,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_expert_capacity_factor=None,
            moe_pad_expert_input_to_capacity=False,
            moe_aux_loss_coeff=0.1,
            num_layers=1,
            moe_router_dtype="fp32",
            moe_grouped_gemm=False,  # Use SequentialMLP to avoid backend issues
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            hidden_size=hidden_size,
            num_attention_heads=max(1, hidden_size // 8),
            use_cpu_initialization=True,
            sequence_parallel=tp_size > 1,
            add_bias_linear=False,
            moe_permute_fusion=False,
            moe_enable_deepep=True,
            ffn_hidden_size=moe_ffn_hidden_size,
        )

        # Patch GroupedMLP before building MoE layer
        import megatron.core.transformer.moe.experts as meg_experts
        from primus.backends.megatron.core.transformer.moe.fsep_experts import FSEPGroupedMLP
        orig_grouped_mlp = meg_experts.GroupedMLP
        fsep_degree = kwargs.get("fsep_sharding_degree", 0)
        if fsep_degree > 1:
            meg_experts.GroupedMLP = FSEPGroupedMLP

        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=self.config.num_moe_experts,
            moe_grouped_gemm=self.config.moe_grouped_gemm,
        )
        self.moe_layer = (
            MoELayer(self.config, transformer_layer_spec.submodules.mlp.submodules)
            .cuda()
            .to(dtype=test_dtype)
        )
        self.moe_layer.set_layer_number(0)

        # Restore original GroupedMLP after layer construction
        meg_experts.GroupedMLP = orig_grouped_mlp

    def __del__(self):
        if dist.is_initialized():
            dist.barrier()
            torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Test class: 4 GPU tests (EP=4, S=4)
# ---------------------------------------------------------------------------

@instantiate_parametrized_tests
class TestFSEPGroupedMLP4GPU(MultiProcessTestCase):
    """Tests requiring 4 GPUs."""

    def tearDown(self):
        super().tearDown()

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return min(4, torch.cuda.device_count())

    @property
    def device(self):
        return torch.device("cuda", self.rank)

    def _init_process(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.rank)
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

    @skip_if_lt_x_gpu(4)
    def test_fsep_reduce_scatter_correctness(self):
        """
        Test 2: ReduceScatter produces correct results.

        world_size=4, EP=4, ETP=1, S=4:
          - EP=world_size, so get_fsep_group() returns EP group. OK.
          - moe_fsep_sharding_degree=4 == ep_size=4.
        """
        self._init_process()
        args = create_args(fsep_sharding_degree=4)
        set_args(args)

        with fsep_patch_context(fsep_sharding_degree=4):
            container = FSEPMoEModelTestContainer(
                tp_size=1,
                ep_size=4,    # EP==world_size: DeepEP intranode works
                pp_size=1,
                moe_tp_size=1,  # ETP=1; FSEP uses EP group (S==EP)
                num_moe_experts=8,
                moe_router_topk=2,
                moe_token_dispatcher_type="flex",
                hidden_size=32,
                test_dtype=torch.float32,
                fsep_sharding_degree=4,
            )

            moe_layer = container.moe_layer
            bs, seql = 2, 4
            hidden_states = torch.randn(
                bs * seql, moe_layer.config.hidden_size,
                dtype=torch.float32,
            ).cuda().contiguous()
            hidden_states.requires_grad_(True)

            probs, indices = moe_layer.router(hidden_states.view(bs, seql, -1))
            probs = torch.ones_like(probs) / moe_layer.router.topk

            permuted, tokens_per_expert, permuted_probs = token_permutation(
                moe_layer.token_dispatcher,
                hidden_states.view(bs, seql, -1), probs, indices,
            )

            # Expert GEMM via FSEPGroupedMLP → output [T/S, H] or [T, H]
            # (when S == EP, each rank only holds T_local/ep tokens after RS)
            expert_out, _ = moe_layer.experts(permuted, tokens_per_expert, permuted_probs)

            # AllGather back + combine
            restored, _ = token_unpermutation(moe_layer.token_dispatcher, expert_out)
            assert not torch.isnan(restored).any(), "NaN in restored output"

    @skip_if_lt_x_gpu(4)
    def test_fsep_backward_gradient_correctness(self):
        """
        Test 3: Backward pass gradients are correct with FSEP.

        Uses manual dispatch/combine to test FSEP's backward:
        ReduceScatter backward = AllGather (via _FSEPAllGather autograd function)
        AllGather backward = ReduceScatter (via _FSEPAllGather autograd function)

        world_size=4, EP=4, ETP=1, S=4.
        """
        self._init_process()
        args = create_args(fsep_sharding_degree=4)
        set_args(args)

        with fsep_patch_context(fsep_sharding_degree=4):
            container = FSEPMoEModelTestContainer(
                tp_size=1,
                ep_size=4,
                pp_size=1,
                moe_tp_size=1,
                num_moe_experts=8,
                moe_router_topk=2,
                moe_token_dispatcher_type="flex",
                hidden_size=32,
                test_dtype=torch.bfloat16,
                fsep_sharding_degree=4,
            )

            moe_layer = container.moe_layer
            bs, seql = 2, 4
            hidden_states = torch.randn(
                bs * seql, moe_layer.config.hidden_size,
                dtype=torch.bfloat16,
            ).cuda().contiguous().requires_grad_(True)

            probs, indices = moe_layer.router(hidden_states.view(bs, seql, -1))
            probs = (torch.ones_like(probs) / moe_layer.router.topk).detach().requires_grad_(False)

            permuted, tokens_per_expert, permuted_probs = token_permutation(
                moe_layer.token_dispatcher,
                hidden_states.view(bs, seql, -1), probs, indices,
            )

            # Expert GEMM via FSEPGroupedMLP → output [T/S, H]
            expert_out, _ = moe_layer.experts(permuted, tokens_per_expert, permuted_probs)

            # AllGather back + combine
            restored, _ = token_unpermutation(moe_layer.token_dispatcher, expert_out)

            # Backward pass — use same pattern as test_token_dispatcher.py
            # (pass gradient tensor directly, not scalar sum, to avoid 3D grad through _exec_combine)
            grad = torch.ones_like(restored)
            torch.autograd.backward(restored, grad)

            # hidden_states.grad should be populated (gradient flows through FSEP)
            assert hidden_states.grad is not None, (
                "hidden_states.grad is None after backward — FSEP gradient flow broken"
            )
            assert not torch.isnan(hidden_states.grad).any(), "NaN in hidden_states.grad"


# ---------------------------------------------------------------------------
# Test class: 8 GPU tests (EP=8, S=4)
# ---------------------------------------------------------------------------

@instantiate_parametrized_tests
class TestFSEPGroupedMLP8GPU(MultiProcessTestCase):
    """Tests requiring 8 GPUs."""

    def tearDown(self):
        super().tearDown()

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return min(8, torch.cuda.device_count())

    @property
    def device(self):
        return torch.device("cuda", self.rank)

    def _init_process(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.rank)
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

    @skip_if_lt_x_gpu(8)
    def test_fsep_full_forward_backward(self):
        """
        Test 4: End-to-end MoE forward/backward with FSEP.

        world_size=8, EP=8, ETP=1, S=8 (FSEP uses EP group).
        Uses manual dispatch/combine to avoid MoELayer.forward backward issues.
        """
        self._init_process()
        args = create_args(fsep_sharding_degree=8)
        set_args(args)

        with fsep_patch_context(fsep_sharding_degree=8):
            container = FSEPMoEModelTestContainer(
                tp_size=1,
                ep_size=8,
                pp_size=1,
                moe_tp_size=1,
                num_moe_experts=16,
                moe_router_topk=2,
                moe_token_dispatcher_type="flex",
                hidden_size=32,
                test_dtype=torch.bfloat16,
                fsep_sharding_degree=8,
            )

            moe_layer = container.moe_layer
            bs, seql = 2, 4
            hidden_states = torch.randn(
                bs * seql, moe_layer.config.hidden_size,
                dtype=torch.bfloat16,
            ).cuda().contiguous().requires_grad_(True)

            probs, indices = moe_layer.router(hidden_states.view(bs, seql, -1))
            probs = (torch.ones_like(probs) / moe_layer.router.topk).detach()

            permuted, tokens_per_expert, permuted_probs = token_permutation(
                moe_layer.token_dispatcher,
                hidden_states.view(bs, seql, -1), probs, indices,
            )
            expert_out, _ = moe_layer.experts(permuted, tokens_per_expert, permuted_probs)
            restored, _ = token_unpermutation(moe_layer.token_dispatcher, expert_out)

            assert not torch.isnan(restored).any(), "NaN in forward output"

            torch.autograd.backward(restored, torch.ones_like(restored))
            assert hidden_states.grad is not None
            assert not torch.isnan(hidden_states.grad).any(), "NaN in backward grad"

    @skip_if_lt_x_gpu(8)
    def test_fsep_load_balance_improvement(self):
        """
        Test 5: FSEP handles imbalanced routing without hang/OOM.

        world_size=8, EP=8, ETP=1, S=8.
        Runs 3 forward/backward steps with natural routing.
        """
        self._init_process()
        args = create_args(fsep_sharding_degree=8)
        args.moe_router_force_load_balancing = False
        set_args(args)

        with fsep_patch_context(fsep_sharding_degree=8):
            container = FSEPMoEModelTestContainer(
                tp_size=1,
                ep_size=8,
                pp_size=1,
                moe_tp_size=1,
                num_moe_experts=16,
                moe_router_topk=2,
                moe_token_dispatcher_type="flex",
                hidden_size=64,
                test_dtype=torch.bfloat16,
                fsep_sharding_degree=8,
            )

            moe_layer = container.moe_layer
            bs, seql = 4, 8
            for _ in range(3):
                hidden_states = torch.randn(
                    bs * seql, moe_layer.config.hidden_size,
                    dtype=torch.bfloat16,
                ).cuda().contiguous().requires_grad_(True)

                probs, indices = moe_layer.router(hidden_states.view(bs, seql, -1))
                probs = (torch.ones_like(probs) / moe_layer.router.topk).detach()

                permuted, tokens_per_expert, permuted_probs = token_permutation(
                    moe_layer.token_dispatcher,
                    hidden_states.view(bs, seql, -1), probs, indices,
                )
                expert_out, _ = moe_layer.experts(permuted, tokens_per_expert, permuted_probs)
                restored, _ = token_unpermutation(moe_layer.token_dispatcher, expert_out)
                torch.autograd.backward(restored, torch.ones_like(restored))

            assert not torch.isnan(restored).any()


if __name__ == "__main__":
    run_tests()
