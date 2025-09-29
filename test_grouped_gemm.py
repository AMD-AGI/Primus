import time

import primus_turbo.pytorch as turbo
import torch

device = "cuda:0"
dtype = torch.bfloat16

# ssi config
# ep=8, hidden_size = 4096, seq=4096
# num_experts=64/128/256/512/1024
# moe_ffn = 4096/2048/1024/512/256,
# num_experts_per_rank=8/16/32/64/128

ep = 8
hidden_size = 4096
seq = 4096
target_weight = 1024 * 256
num_experts = [64, 128, 256, 512, 1024]
# multiply by 2 for linear1 in moe-mlp (swiglu)
moe_ffns = [2 * (target_weight // n) for n in num_experts]

for i in range(len(num_experts)):
    num_experts_per_rank = num_experts[i] // ep
    moe_ffn = moe_ffns[i]

    group_lens = torch.tensor(
        [seq // num_experts_per_rank] * num_experts_per_rank, dtype=torch.long, device=device
    )
    M = seq
    K = hidden_size
    G = num_experts_per_rank
    N = moe_ffn
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(G, K, N, device=device, dtype=dtype)

    print(f"Testing num_experts_per_rank = {num_experts_per_rank}")
    print(f"    >{num_experts[i]=}", flush=True)
    print(f"    >{num_experts_per_rank=}", flush=True)
    print(f"    >{moe_ffn=}", flush=True)
    print(f"    >{a.shape=}", flush=True)
    print(f"    >{b.shape=}", flush=True)

    # warmup
    c = turbo.ops.grouped_gemm(a, b, group_lens, trans_b=False)
    torch.cuda.synchronize()

    iter = 100
    t_s = time.time()
    for j in range(iter):
        c = turbo.ops.grouped_gemm(a, b, group_lens, trans_b=False)
    torch.cuda.synchronize()
    t_e = time.time()
    t_elapsed = (t_e - t_s) / iter
    tflops = M * K * N * 2 / 1e12 / t_elapsed
    print(f"    >label: gg_{G}x({M//num_experts_per_rank}x{K}, {K}x{N})")
    print(f"    >Latency: {t_elapsed * 1000:.3f} ms")
    print(f"    >Throughtput: {tflops:.3f} tflops")
