"""Does moe_permute write every permuted_probs row for top-k indices with -1 slots + top-k probs?"""

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.ops import moe_permute

num_tokens, E, K, H = 333, 16, 8, 256
g = torch.Generator(device="cuda").manual_seed(31)
scores = torch.rand((num_tokens, 4 * E), generator=g, device="cuda")
ge = scores.topk(K, dim=-1).indices
local = ge < E
topk = torch.where(local, ge, torch.full_like(ge, -1)).int()
import os  # noqa: E402

if os.environ.get("PREFIX") == "1":
    order = torch.argsort((~local.any(dim=1)).int(), stable=True)  # routed tokens first, as DeepEP
    topk, ge, local = topk[order], ge[order], local[order]
rows = int(local.sum())
x = torch.randn((num_tokens, H), dtype=torch.bfloat16, device="cuda")
probs = torch.rand((num_tokens, K), dtype=torch.float32, device="cuda")

# Reference: expert-major, token order inside an expert.
ref_tok, ref_prob = [], []
for e in range(E):
    t, k = (topk == e).nonzero(as_tuple=True)
    ref_tok.append(t)
    ref_prob.append(probs[t, k])
ref_tok, ref_prob = torch.cat(ref_tok), torch.cat(ref_prob)

for backend in (BackendType.TURBO, BackendType.TRITON):
    for fill, idx_dtype in ((7.0, torch.int32), (7.0, torch.int64), (-3.0, torch.int64)):
        torch.cuda.empty_cache()
        junk = torch.full((rows * H,), fill, device="cuda")  # noqa: F841 (poison the allocator)
        del junk
        out, _, tpe, _, _, pp = moe_permute(
            x, topk_indices=topk.to(idx_dtype), num_local_experts=E, num_topk=K, num_permuted_tokens=rows,
            probs=probs, probs_layout="topk", backend=backend,
        )
        print(idx_dtype, end=" ")
        bad_tok = (out != x[ref_tok]).any(dim=1).nonzero().flatten()
        bad_prob = (pp != ref_prob).nonzero().flatten()
        print(
            f"{backend.name:6s} fill={fill} rows={rows} tokens bad={bad_tok.numel()} probs bad={bad_prob.numel()} "
            f"first bad prob rows={bad_prob[:6].tolist()} vals={pp[bad_prob[:3]].tolist()}"
        )
        if backend is BackendType.TURBO and idx_dtype is torch.int64 and fill > 0:
            ref_cnt = torch.stack([(topk == e).sum() for e in range(E)])
            print("  ref tokens_per_expert", ref_cnt.tolist())
            print("  got tokens_per_expert", tpe.tolist())
            seg_end = ref_cnt.cumsum(0).tolist()
            print("  segment ends", seg_end)
            print("  bad rows", bad_prob.tolist()[:40])
            multi = (local.sum(dim=1) > 1).sum().item()
            print("  tokens with >1 local expert", multi, "with 0 local", (local.sum(dim=1) == 0).sum().item())
