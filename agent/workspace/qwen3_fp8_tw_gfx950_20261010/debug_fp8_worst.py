"""2-GPU debug: worst-tokens + pad 16 FP8 permute mismatch (mirrors test_fp8_permute)."""

import os

import torch
import torch.distributed as dist

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor

rank = int(os.environ["RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
T, H, E, K = 4096, 4096, 256, 8


class Stub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out, record):
        ctx.record = record
        record["input"] = x
        return out.clone()

    @staticmethod
    def backward(ctx, grad):
        ctx.record["grad"] = grad
        return torch.zeros(grad.shape, dtype=torch.bfloat16, device=grad.device), None, None


def case(pad, worst, rep):
    d = turbo.modules.DeepEPTokenDispatcher(
        E, K, dist.group.WORLD, pad_multiple=pad, deepep_num_worst_tokens=worst,
        deepep_use_cuda_num_tokens_per_expert=True,
    )
    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    h0 = torch.randn((T, H), dtype=torch.bfloat16, device="cuda", generator=gen)
    probs = torch.rand((T, E), dtype=torch.float32, device="cuda", generator=gen)
    gout = torch.randn((T, H), dtype=torch.bfloat16, device="cuda", generator=gen)

    def run(q):
        rec = {}
        h = h0.clone().requires_grad_(True)
        x, tp = d._pre_dispatch(h, probs)
        x, dp = d._exec_dispatch(x, tp)
        rec["recv"] = x.detach().clone()
        rec["idx"] = d.dispatched_indices.clone()
        p, tpe, _ = d._post_dispatch(x, dp, quantize_dtype=float8_e4m3 if q else None)
        rec["rmap"] = d.row_id_map.clone()
        out = torch.ones(p.shape, dtype=torch.bfloat16, device="cuda")
        y = Stub.apply(p, out, rec)
        y = d._pre_combine(y, grad_quantize_dtype=float8_e5m2 if q else None)
        d._post_combine(d._exec_combine(y)).backward(gout)
        return rec, tpe

    ref, tpe = run(False)
    got, tpe2 = run(True)
    want = QuantizedTensor.quantize(
        ref["input"].detach(), float8_e4m3, ScalingGranularity.TENSORWISE, axis=-1, group_lens=tpe, pad_align_last=128
    )
    q = got["input"]
    diff = q.qdata.view(torch.uint8) != want.qdata.view(torch.uint8)
    rows = diff.any(1).nonzero().flatten()
    recv_same = torch.equal(ref["recv"], got["recv"])
    idx = ref["idx"]
    nloc = idx.shape[1] and d.num_local_experts
    real = torch.stack([(idx == e).sum() for e in range(nloc)]).to(tpe.device)
    offs = torch.cat([tpe.new_zeros(1), tpe.cumsum(0)])
    pad_mask = torch.zeros(q.shape[0], dtype=torch.bool, device="cuda")
    for e in range(nloc):
        pad_mask[int(offs[e] + real[e]) : int(offs[e + 1])] = True
    ref_rows_nonzero = (ref["input"].detach() != 0).any(1)
    got_rows_nonzero = (q.qdata.view(torch.uint8) != 0).any(1)
    print(
        f"rank{rank} pad={pad} worst={worst} rep={rep}: pad_rows={int(pad_mask.sum())} "
        f"ref_nonzero_pad_rows={int((ref_rows_nonzero & pad_mask).sum())} "
        f"fp8_nonzero_pad_rows={int((got_rows_nonzero & pad_mask).sum())} "
        f"diff_rows_in_pad={int(pad_mask[rows].sum()) if rows.numel() else 0}/{rows.numel()} "
        f"idx_tail_valid={int((idx[ref['rmap'].shape[0] - pad:] >= 0).sum()) if worst else -1}",
        flush=True,
    )
    print(
        f"rank{rank} pad={pad} worst={worst} rep={rep}: tpe_eq={torch.equal(tpe, tpe2)} recv_eq={recv_same} "
        f"scale got={q.scale_inv.item():.6e} want={want.scale_inv.item():.6e} ndiff_rows={rows.numel()} "
        f"first={rows[:6].tolist()} nperm={q.shape[0]}",
        flush=True,
    )


for rep in range(3):
    for pad in (0, 16):
        for worst in (0, T * 8):
            case(pad, worst, rep)
dist.destroy_process_group()
