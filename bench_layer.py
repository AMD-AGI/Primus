"""Layer-level microbench: ROCMoELayer vs stock Megatron MoELayer (EP=8).

Run under torchrun (8 ranks, single node):
  torchrun --standalone --nproc-per-node=8 bench_layer.py [--which both|rocmoe|baseline]

Builds one MoE layer at deepseek_v3 scale (H=7168, moe_ffn=2048, E=256, topk=8,
shared expert) and times forward and forward+backward in isolation, so we can
compare the two engines without the full training pipeline.
"""
import argparse
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.training.global_vars import set_args
from megatron.core import parallel_state as ps


def apply_te_workspace_shim():
    """Replicate Primus' te general_gemm workspace patch for the standalone baseline."""
    import inspect
    try:
        from megatron.core.extensions import transformer_engine as te_ext
    except ImportError:
        return
    gg = getattr(te_ext, "general_gemm", None)
    if gg is None:
        return
    try:
        params = inspect.signature(gg).parameters
    except (TypeError, ValueError):
        return
    if "workspace" not in params:
        te_ext._get_workspace = None


def init_dist_ep8():
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=8,
    )

SEQ = int(os.environ.get("BENCH_SEQ", "4096"))
MBS = int(os.environ.get("BENCH_MBS", "1"))
SHARED = os.environ.get("BENCH_SHARED", "1") == "1"
H = 7168
F = 2048
E = 256
K = 8


def build_config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=H,
        ffn_hidden_size=18432,
        num_attention_heads=56,
        num_moe_experts=E,
        moe_router_topk=K,
        moe_ffn_hidden_size=F,
        moe_shared_expert_intermediate_size=(F if SHARED else None),
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.0,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=8,
        sequence_parallel=False,
    )


TPROF = os.environ.get("BENCH_TORCH_PROFILE", "0") == "1"


def _torch_profile(layer, name, fwd, gy, iters=10):
    """Kineto trace over fwd-only and fwd+bwd, print per-kernel CUDA breakdown."""
    from torch.profiler import profile, ProfilerActivity, record_function

    for _ in range(3):
        y = fwd(); y.backward(gy)
        layer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dist.barrier()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            with record_function("FWD"):
                y = fwd()
            with record_function("BWD"):
                y.backward(gy)
            layer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    if dist.get_rank() == 0:
        ka = prof.key_averages()
        # total device time over all GPU kernels
        tot = sum(e.self_device_time_total for e in ka)
        print(f"\n===== [{name}] torch.profiler kernel breakdown "
              f"(iters={iters}, total GPU={tot/iters/1e3:.2f} ms/iter) =====", flush=True)
        rows = sorted(ka, key=lambda e: e.self_device_time_total, reverse=True)
        shown = 0
        for e in rows:
            dt = e.self_device_time_total
            if dt <= 0:
                continue
            print(f"  {dt/iters/1e3:8.3f} ms  {e.key[:70]}", flush=True)
            shown += 1
            if shown >= 25:
                break
        path = f"/tmp/trace_{name}.json"
        prof.export_chrome_trace(path)
        print(f"  [trace] {path}", flush=True)


def time_layer(layer, name, iters=20, warmup=5):
    dev = torch.cuda.current_device()
    x = torch.randn(SEQ, MBS, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
    gy = torch.randn(SEQ, MBS, H, device=dev, dtype=torch.bfloat16)

    def fwd():
        out = layer(x)
        return out[0] if isinstance(out, tuple) else out

    # warmup
    for _ in range(warmup):
        y = fwd()
        y.backward(gy)
        x.grad = None
        layer.zero_grad(set_to_none=True)
    dist.barrier()
    torch.cuda.synchronize()

    if TPROF:
        _torch_profile(layer, name, fwd, gy)

    # forward-only
    f_ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(iters)]
    with torch.no_grad():
        for i in range(iters):
            f_ev[i][0].record()
            fwd()
            f_ev[i][1].record()
    torch.cuda.synchronize()
    fwd_ms = sum(a.elapsed_time(b) for a, b in f_ev) / iters

    # forward + backward
    fb_ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(iters)]
    for i in range(iters):
        fb_ev[i][0].record()
        y = fwd()
        y.backward(gy)
        fb_ev[i][1].record()
        x.grad = None
        layer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    fb_ms = sum(a.elapsed_time(b) for a, b in fb_ev) / iters

    if dist.get_rank() == 0:
        print(f"[{name}] fwd={fwd_ms:.2f} ms  fwd+bwd={fb_ms:.2f} ms  bwd={fb_ms - fwd_ms:.2f} ms",
              flush=True)
    return fwd_ms, fb_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["both", "rocmoe", "baseline"])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    init_dist_ep8()
    apply_te_workspace_shim()
    # PrimusTopKRouter.routing reads these off the global args; set them so the
    # standalone bench (no Primus patch/config) falls back to stock routing.
    set_args(SimpleNamespace(seq_length=SEQ, micro_batch_size=MBS,
                             enable_primus_turbo=False,
                             moe_use_fused_router_with_aux_score=False,
                             router_logit_softcapping=None))

    # Primus' logger isn't initialized in this bare harness; stub log_rank_0
    # (used in ROCMoELayer.__init__) to a plain rank-0 print before import.
    import primus.modules.module_utils as _mu
    _mu.log_rank_0 = lambda msg, *a, **k: print(msg) if dist.get_rank() == 0 else None
    torch.manual_seed(1234)

    cfg = build_config()
    submods = get_gpt_layer_with_transformer_engine_submodules(
        num_experts=E, moe_grouped_gemm=True
    ).mlp.submodules

    if dist.get_rank() == 0:
        print(f"=== MoE layer bench: H={H} F={F} E={E} K={K} EP=8 "
              f"seq={SEQ} mbs={MBS} b_per_rank={SEQ * MBS} shared={SHARED} ===", flush=True)

    if args.which in ("baseline", "both"):
        from megatron.core.transformer.moe.moe_layer import MoELayer
        base = MoELayer(cfg, submods).cuda().to(torch.bfloat16)
        time_layer(base, "baseline", iters=args.iters)
        del base
        torch.cuda.empty_cache()

    if args.which in ("rocmoe", "both"):
        from primus.backends.megatron.core.transformer.moe.rocmoe.moe_layer import ROCMoELayer
        roc = ROCMoELayer(cfg, submods).cuda()
        time_layer(roc, "rocmoe", iters=args.iters)
        del roc
        torch.cuda.empty_cache()

    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
