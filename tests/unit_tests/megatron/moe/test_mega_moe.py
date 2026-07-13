###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""EP8 parity test: fused ``MegaMoE`` vs Megatron ``MoELayer``.
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.process_groups_config import ProcessGroupCollection
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
    parametrize,
    run_tests,
)

from primus.backends.megatron.core.extensions.mega_moe import PrimusTurboMegaMoELayer

def _create_args():
    args = SimpleNamespace()
    args.sequence_parallel = False
    args.context_parallel_size = 1
    args.micro_batch_size = 1
    args.moe_router_force_load_balancing = False
    args.moe_use_legacy_grouped_gemm = True
    args.use_turbo_grouped_gemm = False
    args.enable_primus_turbo = False
    args.moe_use_fused_router_with_aux_score = False
    args.router_logit_softcapping = None
    return args


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().reshape(-1).float()
    b = b.detach().reshape(-1).float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def _build_config(ep_size, num_moe_experts, moe_router_topk):
    """DeepSeek-V3 MoE config: sigmoid group-limited routing + shared expert."""
    return TransformerConfig(
        num_layers=1,
        hidden_size=7168,
        num_attention_heads=8,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
        pipeline_model_parallel_size=1,
        num_moe_experts=num_moe_experts,
        moe_ffn_hidden_size=2048,
        moe_router_topk=moe_router_topk,
        # DeepSeek-V3 group-limited routing
        moe_router_num_groups=8,
        moe_router_group_topk=4,
        moe_router_topk_scaling_factor=2.5,
        moe_router_score_function="sigmoid",
        moe_router_pre_softmax=False,
        # noaux_tc bias-based balancing; bias stays 0 in this test so routing is
        # pure sigmoid group-topk (mega's fused router ignores expert_bias).
        moe_router_enable_expert_bias=True,
        # aux loss off: keeps every weight's grad a clean fwd/bwd parity check
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.0,
        # DeepSeek-V3 shared expert (1 expert, no gate)
        moe_shared_expert_intermediate_size=2048,
        moe_shared_expert_gate=False,
        moe_token_dispatcher_type="alltoall",
        # fp64 required: fp32 router hits a broken TE general_gemm(workspace=) path
        moe_router_dtype="fp64",
        moe_grouped_gemm=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,  # MegaMoE hardcodes SwiGLU/SiLU
        add_bias_linear=False,
        bias_activation_fusion=False,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        bf16=True,
    )


def _build_layers(config, num_moe_experts, num_layers):
    """Build ``num_layers`` matched (Megatron MoELayer, MegaMoE) pairs.

    Each layer gets its own seed -> distinct-but-matched weights; ref weights
    are copied into the mega layer so per-layer outputs are directly comparable.
    """
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    moe_layers, mega_layers = [], []
    for layer_idx in range(num_layers):
        _set_random_seed(seed_=123 + layer_idx, data_parallel_random_init=False)

        spec = get_gpt_layer_local_spec(num_experts=num_moe_experts, moe_grouped_gemm=False)
        moe_layer = MoELayer(config, spec.submodules.mlp.submodules).cuda().to(torch.bfloat16)
        moe_layer.set_layer_number(layer_idx)

        mega_layer = PrimusTurboMegaMoELayer(
            config, layer_number=layer_idx, pg_collection=pg_collection
        ).cuda()

        mega = mega_layer.mega_moe
        experts = moe_layer.experts.local_experts
        assert len(experts) == mega.w1.shape[0]
        with torch.no_grad():
            # match routing: copy gate weight, drop base router bias (mega has none)
            mega.gate_weight.copy_(moe_layer.router.weight.to(mega.gate_weight.dtype))
            if moe_layer.router.bias is not None:
                moe_layer.router.bias.zero_()
            # zero noaux_tc expert bias -> baseline selection == pure sigmoid group-topk
            if getattr(moe_layer.router, "expert_bias", None) is not None:
                moe_layer.router.expert_bias.zero_()
            for i, expert in enumerate(experts):
                assert expert.linear_fc1.weight.shape == mega.w1[i].shape
                assert expert.linear_fc2.weight.shape == mega.w2[i].shape
                mega.w1[i].copy_(expert.linear_fc1.weight.to(torch.bfloat16))
                mega.w2[i].copy_(expert.linear_fc2.weight.to(torch.bfloat16))
            # shared expert: copy fc1/fc2 (no shared gate in this config)
            if mega.shared_w1 is not None:
                se = moe_layer.shared_experts
                assert se.linear_fc1.weight.shape == mega.shared_w1.shape
                assert se.linear_fc2.weight.shape == mega.shared_w2.shape
                mega.shared_w1.copy_(se.linear_fc1.weight.to(torch.bfloat16))
                mega.shared_w2.copy_(se.linear_fc2.weight.to(torch.bfloat16))
        moe_layers.append(moe_layer)
        mega_layers.append(mega_layer)
    return moe_layers, mega_layers


@instantiate_parametrized_tests
class TestMegaMoEAccuracy(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        super().tearDown()

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(backend="nccl", world_size=self.world_size, rank=self.rank, store=store)
        set_args(_create_args())
        # standalone harness has no Primus logger; attach a plain one so the
        # adapter's log_rank_0 (enable_expert_bias warning) doesn't crash
        import logging

        from primus.core.utils import logger as _primus_logger

        if _primus_logger._logger is None:
            _primus_logger._logger = logging.getLogger("mega_moe_test")

    def _setup_ep(self):
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=self.world_size,
        )

    def _rank0(self, msg):
        if self.rank == 0:
            print(msg, flush=True)

    @skip_if_lt_x_gpu(8)
    @parametrize("moe_router_topk", [8])
    @parametrize("num_ga", [4])
    def test_forward_backward(self, moe_router_topk, num_ga):
        # DeepSeek-V3 EP8 shapes: 256 experts, top-8, hidden 7168.
        # Single MoE layer, num_ga gradient-accumulation microbatches: forward +
        # backward num_ga times without zeroing grads (mirrors real GA training).
        # One layer -> no cross-layer backward accumulation, so grads stay clean.
        num_moe_experts, hidden_size = 256, 7168
        self._init_process()
        self._setup_ep()
        try:
            config = _build_config(self.world_size, num_moe_experts, moe_router_topk)
            moe_layers, mega_layers = _build_layers(config, num_moe_experts, 1)
            moe_layer, mega_layer = moe_layers[0], mega_layers[0]
            moe_layer.train(True)
            mega_layer.train(True)
            mega = mega_layer.mega_moe
            experts = moe_layer.experts.local_experts
            se = moe_layer.shared_experts
            seq, batch = 8192, 1

            fwd_cos, dx_cos = [], []
            # GA loop: accumulate grads over num_ga microbatches, compare each fwd/dx
            for step in range(num_ga):
                torch.manual_seed(1000 + self.rank + step * 97)
                x = torch.randn((seq, batch, hidden_size), dtype=torch.bfloat16, device=self.device)
                g = torch.randn((seq, batch, hidden_size), dtype=torch.bfloat16, device=self.device)

                x_ref = x.clone().requires_grad_(True)
                ref, _ = moe_layer(x_ref)
                ref.backward(g)
                x_meg = x.clone().requires_grad_(True)
                out, _ = mega_layer(x_meg)
                out.backward(g)

                fwd_cos.append(_cosine(out, ref))
                dx_cos.append(_cosine(x_meg.grad, x_ref.grad))
                self._rank0(
                    f"[ga {step}] fwd cos={fwd_cos[-1]:.6f} dx cos={dx_cos[-1]:.6f} "
                    f"max_abs_diff={(out.float()-ref.float()).abs().max().item():.3e}"
                )

            GRAD_FLOOR = 0.95
            failures = []

            def _check(tag, cos):
                if cos <= GRAD_FLOOR:
                    failures.append(f"{tag} cosine {cos:.6f} < {GRAD_FLOOR}")

            # forward + dx parity: every microbatch
            for step in range(num_ga):
                _check(f"ga{step} fwd", fwd_cos[step])
                _check(f"ga{step} dx", dx_cos[step])

            # accumulated per-weight grad parity: every trainable weight
            dgate = _cosine(mega.gate_weight.grad, moe_layer.router.weight.grad)
            _check("router", dgate)
            dsw1 = _cosine(mega.shared_w1.grad, se.linear_fc1.weight.grad)
            dsw2 = _cosine(mega.shared_w2.grad, se.linear_fc2.weight.grad)
            _check("shared dW1", dsw1)
            _check("shared dW2", dsw2)
            # routed experts: full w1/w2 grad tensors (norm-weighted over all local
            # experts); per-expert min is thin-token bf16 noise, informational only.
            ref_dw1 = torch.stack([e.linear_fc1.weight.grad for e in experts])
            ref_dw2 = torch.stack([e.linear_fc2.weight.grad for e in experts])
            dW1 = _cosine(mega.w1.grad, ref_dw1)
            dW2 = _cosine(mega.w2.grad, ref_dw2)
            _check("experts dW1", dW1)
            _check("experts dW2", dW2)
            dw1_min = min(_cosine(mega.w1.grad[i], e.linear_fc1.weight.grad) for i, e in enumerate(experts))
            dw2_min = min(_cosine(mega.w2.grad[i], e.linear_fc2.weight.grad) for i, e in enumerate(experts))
            self._rank0(
                f"[grad accum over {num_ga} ga] dgate={dgate:.6f} dSharedW1={dsw1:.6f} "
                f"dSharedW2={dsw2:.6f} experts dW1={dW1:.6f} dW2={dW2:.6f} "
                f"(per-expert min {dw1_min:.4f}/{dw2_min:.4f})"
            )
            self.assertEqual(failures, [], f"parity below {GRAD_FLOOR}: {failures}")
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
