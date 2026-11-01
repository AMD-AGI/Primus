###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The fused residual RMSNorm's MoE mode around a ``PrimusTurboMegaMoELayer``, bit for bit against the
unfused layer, on one gfx1250 GPU (EP1).

The mode gives the router, the shared expert and the routed experts each an alias of the pre-MLP norm's
output, so their input gradients reach the norm's backward apart, and the kernel adds them in the order
autograd would have: (shared + routed) + router. The layer returns the routed and the shared outputs
unsummed for the next norm to add. A wrong order changes the last bit of about a third of the input
gradient, so comparing every gradient bitwise checks the order on the real layer. The last test runs two
layers through the patch's fused forward, the second one taking the first one's carry.
"""

import contextlib
import inspect
import logging
import socket
from types import SimpleNamespace

import pytest
import torch

from primus.core.utils.gpu_arch import is_gfx1250

if not is_gfx1250():
    pytest.skip("needs a gfx1250 GPU (the MegaMoE kernels)", allow_module_level=True)

from primus_turbo.pytorch.ops.normalization import (  # noqa: E402
    rmsnorm,
    rmsnorm_residual,
)

if "num_outputs" not in inspect.signature(rmsnorm_residual).parameters:
    pytest.skip("Primus-Turbo without the per-consumer rmsnorm_residual", allow_module_level=True)

import torch.distributed as dist  # noqa: E402
import triton  # noqa: E402
from megatron.core import parallel_state, tensor_parallel  # noqa: E402
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add  # noqa: E402
from megatron.core.models.gpt.gpt_layer_specs import (  # noqa: E402
    get_gpt_layer_local_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402
from megatron.training.global_vars import set_args  # noqa: E402
from primus_turbo.triton.normalization import rmsnorm_kernel  # noqa: E402

from primus.backends.megatron.core.extensions import (  # noqa: E402
    fused_residual_rmsnorm,
)
from primus.backends.megatron.core.extensions.mega_moe import (  # noqa: E402
    PrimusTurboMegaMoELayer,
)
from primus.backends.megatron.core.extensions.primus_turbo import (  # noqa: E402
    PrimusTurboRMSNorm,
)

HIDDEN, EXPERTS, EPS = 7168, 32, 1e-6


def _args():
    return SimpleNamespace(
        turbo_mega_moe_precision="bf16",
        turbo_mega_moe_wgrad_accum_fusion=None,
        disable_primus_topk_router=False,
        sequence_parallel=False,
        context_parallel_size=1,
        micro_batch_size=1,
        moe_router_force_load_balancing=False,
        moe_use_legacy_grouped_gemm=True,
        use_turbo_grouped_gemm=False,
        enable_primus_turbo=False,
        moe_use_fused_router_with_aux_score=False,
        router_logit_softcapping=None,
    )


def _config():
    """DeepSeek-V3's MoE layer as in test_mega_moe.py, with 32 experts on the one rank."""
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        num_moe_experts=EXPERTS,
        moe_ffn_hidden_size=2048,
        moe_router_topk=8,
        moe_router_num_groups=8,
        moe_router_group_topk=4,
        moe_router_topk_scaling_factor=2.5,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.0,
        moe_shared_expert_intermediate_size=2048,
        moe_token_dispatcher_type="alltoall",
        moe_router_dtype="fp64",
        moe_grouped_gemm=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        bias_activation_fusion=False,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        bf16=True,
    )


@pytest.fixture(scope="module")
def layer():
    if not dist.is_initialized():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=1, rank=0)
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(1, 1, expert_model_parallel_size=1)
    tensor_parallel.model_parallel_cuda_manual_seed(123)
    set_args(_args())
    from primus.core.utils import logger as primus_logger

    if primus_logger._logger is None:
        primus_logger._logger = logging.getLogger("mega_moe_fused_residual_norm_test")

    spec = get_gpt_layer_local_spec(num_experts=EXPERTS, moe_grouped_gemm=False)
    mlp = spec.submodules.mlp
    submodules = mlp.keywords["submodules"] if hasattr(mlp, "keywords") else mlp.submodules
    torch.manual_seed(0)
    module = PrimusTurboMegaMoELayer(
        _config(),
        submodules,
        layer_number=0,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        name="decoder.layers.0.mlp",
    ).cuda()
    with torch.no_grad():
        if getattr(module.router, "expert_bias", None) is not None:
            module.router.expert_bias.zero_()
    yield module
    parallel_state.destroy_model_parallel()


@contextlib.contextmanager
def _pinned_rmsnorm_bwd():
    """Both grid-stride RMSNorm backward autotuners on one config: the same row reduction tree."""
    tuners = (
        rmsnorm_kernel.rmsnorm_bwd_kernel_grid_stride,
        rmsnorm_kernel.rmsnorm_bwd_residual_kernel_grid_stride,
    )
    saved = [(t.configs, dict(t.cache)) for t in tuners]
    try:
        for t in tuners:
            t.configs = [triton.Config({}, num_warps=8, num_stages=1)]
            t.cache.clear()
        yield
    finally:
        for t, (configs, cache) in zip(tuners, saved):
            t.configs = configs
            t.cache.clear()
            t.cache.update(cache)


def _run(layer, attn_out, residual, gamma, grad_out, fused, swap_router_and_shared=False):
    a, r, g = (t.detach().clone().requires_grad_() for t in (attn_out, residual, gamma))
    layer.zero_grad(set_to_none=True)
    if fused:
        y_shared, y_routed, y_router, h = rmsnorm_residual(a, r, g, EPS, num_outputs=3, exact=True)
        if swap_router_and_shared:
            y_shared, y_router = y_router, y_shared
        (routed, shared), bias = layer(
            y_routed, router_input=y_router, shared_expert_input=y_shared, split_shared_output=True
        )
        assert bias is None
        out = (routed + shared) + h
    else:
        h = a + r
        mlp_out, _ = layer(rmsnorm(h, g, EPS))
        out = h + mlp_out
    out.backward(grad_out)
    grads = {"attn_out": a.grad, "residual": r.grad, "gamma": g.grad}
    grads.update({name: p.grad.clone() for name, p in layer.named_parameters() if p.grad is not None})
    return out.detach(), grads


def _differing(a, b):
    if a.dtype == torch.bfloat16:
        return int((a.view(torch.int16) != b.view(torch.int16)).sum())
    return int((a != b).sum())


def _assert_matches_unfused(runs, stable):
    """``runs``: two unfused runs, then the fused one. The fused run is bitwise the first wherever the two
    unfused runs agree, which must include ``stable``. Run to run, the routed experts' weight gradients
    change in their last bits even unfused (the Mega wgrad without accumulation fusion): those are
    compared norm-wise, as where two layers' contributions cancel, last-bit noise dwarfs an element."""
    (out_ref, ref), (out_again, again), (out, got) = runs
    unstable = {n for n in ref if _differing(ref[n], again[n])}
    assert not _differing(out_ref, out_again) and not unstable & stable
    assert _differing(out, out_ref) == 0, "output"
    assert set(got) == set(ref)
    bad = {n: _differing(got[n], ref[n]) for n in ref if n not in unstable and _differing(got[n], ref[n])}
    assert not bad, f"gradients differ (elements): {bad}"
    for n in unstable:
        err = float((got[n].float() - ref[n].float()).norm() / ref[n].float().norm())
        assert err < 1e-3, f"{n}: relative error {err:.1e}"


def test_moe_mode_bitwise(layer):
    torch.manual_seed(1)
    shape = (1024, 2, HIDDEN)  # [s, b, h]
    attn_out, residual, grad_out = (torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(3))
    gamma = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda")
    with _pinned_rmsnorm_bwd():
        runs = [_run(layer, attn_out, residual, gamma, grad_out, fused=f) for f in (False, False, True)]
    _assert_matches_unfused(runs, stable={"attn_out", "residual", "gamma"})


def test_a_wrong_consumer_order_shows(layer):
    """The check above has teeth: the aliases in another order change the input gradient's last bits."""
    torch.manual_seed(2)
    shape = (1024, 2, HIDDEN)
    attn_out, residual, grad_out = (torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(3))
    gamma = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda")
    with _pinned_rmsnorm_bwd():
        _, ref = _run(layer, attn_out, residual, gamma, grad_out, fused=False)
        _, swapped = _run(layer, attn_out, residual, gamma, grad_out, fused=True, swap_router_and_shared=True)
    assert _differing(swapped["attn_out"], ref["attn_out"]) > ref["attn_out"].numel() // 100


class _Attention(torch.nn.Linear):
    """A stand-in for the attention: a linear map, returned as ``(output, bias)``."""

    def forward(self, x, **kwargs):
        return super().forward(x), None


class _Layer(torch.nn.Module):
    """What the patch's fused forward reads off a ``TransformerLayer``, around the MegaMoE layer."""

    def __init__(self, moe, attention):
        super().__init__()
        self.input_layernorm, self.pre_mlp_layernorm = (
            PrimusTurboRMSNorm(HIDDEN, eps=EPS, params_dtype=torch.bfloat16) for _ in range(2)
        )
        for norm in (self.input_layernorm, self.pre_mlp_layernorm):
            torch.nn.init.normal_(norm.weight)
        self.self_attention, self.mlp = attention, moe
        self.config = SimpleNamespace(bias_dropout_fusion=False)
        self.hidden_dropout = 0.0
        self.mlp_bda = get_bias_dropout_add
        self.bias_dropout_add_exec_handler = contextlib.nullcontext


def _run_layers(layers, x, grad_out, fused):
    x = x.detach().clone().requires_grad_()
    modules = torch.nn.ModuleList(layers)
    modules.zero_grad(set_to_none=True)
    out = x
    for layer in layers:
        if fused:
            out, _ = fused_residual_rmsnorm._do_fused_forward(layer, out)
        else:
            h = out + layer.self_attention(layer.input_layernorm(out))[0]
            out = h + layer.mlp(layer.pre_mlp_layernorm(h))[0]
    out.backward(grad_out)
    grads = {"x": x.grad}
    grads.update({name: p.grad.clone() for name, p in modules.named_parameters() if p.grad is not None})
    return out.detach(), grads


def test_patched_layers_bitwise(layer, monkeypatch):
    """Two layers through the patch's fused forward: the MoE split in both, the first one's carry
    ``(routed, shared, residual)`` added by the second one's input norm, and the explicit bias-dropout-add
    after the last one (no final norm), against the same layers unfused."""
    monkeypatch.setenv("PRIMUS_FUSED_RESIDUAL_NORM_MOE", "1")
    torch.manual_seed(3)
    attention = _Attention(HIDDEN, HIDDEN, bias=False, device="cuda", dtype=torch.bfloat16)
    layers = [_Layer(layer, attention) for _ in range(2)]
    fused_residual_rmsnorm._wire_v2_layer_links(SimpleNamespace(layers=layers, final_layernorm=None))
    x, grad_out = (torch.randn((1024, 2, HIDDEN), dtype=torch.bfloat16, device="cuda") for _ in range(2))
    with _pinned_rmsnorm_bwd():
        runs = [_run_layers(layers, x, grad_out, fused=f) for f in (False, False, True)]
    norms = {f"{i}.{n}.weight" for i in range(2) for n in ("input_layernorm", "pre_mlp_layernorm")}
    _assert_matches_unfused(runs, stable={"x"} | norms)
