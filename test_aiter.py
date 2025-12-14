import torch
import aiter
from aiter import dtypes

dtype = torch.bfloat16
device = "cuda:0"


def compute_snr(x: torch.Tensor, y: torch.Tensor):
    x, y = x.float(), y.float()
    signal_power = torch.norm(x).pow(2)
    noise_power = torch.norm(x - y).pow(2)
    return 10 * torch.log10(signal_power / (noise_power + 1e-12)).detach().item()

 
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def attn_ref(q, k, v, dout, causal=True):
    # repeat k and v to match q's num_heads
    num_heads_q = q.size(2)
    num_heads_kv = k.size(2)
    num_groups = num_heads_q // num_heads_kv
    
    # Keep original tensors for gradient
    k_orig, v_orig = k, v
    # k, v: [b, seq, num_heads_kv, head_dim] -> [b, seq, num_heads_q, head_dim]
    k = repeat_kv(k, num_groups)
    v = repeat_kv(v, num_groups)

    torch.cuda.synchronize()
    seq_q, head_dim_q = q.size(1), q.size(3)
    seq_k, head_dim_k = k.size(1), k.size(3)
    seq_v, head_dim_v = v.size(1), v.size(3)
    assert head_dim_q == head_dim_k
    assert seq_k == seq_v

    softmax_scale = q.shape[-1] ** (-0.5)

    q_t = q.transpose(1, 2)  # [b, nh, seq, hd_qk]
    k_t = k.transpose(1, 2)  # [b, nh, seq, hd_qk]
    v_t = v.transpose(1, 2)  # [b, nh, seq, hd_v]

    # [b, nh, seq, seq]  = [b, nh, seq, hd_qk] * [b, nh, hd_qk, seq]
    p = torch.matmul(q_t, k_t.transpose(2, 3)) * softmax_scale
    if causal:
        mask = torch.tril(torch.ones((seq_q, seq_q), device=q.device))
        p[:, :, mask == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)

    # [b, nh, seq, hd_v] = [b, nh, seq, seq] * [b, nh, seq, hd_v]
    out = torch.matmul(p, v_t)
    out = out.transpose(1, 2)

    # Use autograd.grad to get gradients (works with non-leaf tensors)
    dq, dk, dv = torch.autograd.grad(out, (q, k_orig, v_orig), dout)
    torch.cuda.synchronize()
    return out, dq, dk, dv

 
# @torch.compile(fullgraph=True)
def test_aiter_compile(q, k, v, do):
    out, _ = aiter.flash_attn_func(
        q,
        k,
        v,
        # dropout_p=0.0,
        causal=True,
        # window_size=(-1, -1),
        # bias=None,
        # alibi_slopes=None,
        deterministic=False,
        return_lse=True,
        # return_attn_probs=False,
    )
    return out
 
def test_case(
    batch_size,
    seq_len,
    num_heads_q,
    num_heads_kv,
    head_dim,
):
    tflop_fwd = 4 * batch_size * seq_len * seq_len * num_heads_q * head_dim / 1e12
    tflop_fwd = tflop_fwd * 0.5
    tflop_bwd = tflop_fwd * 2.5

    bytes_per_element = 2 # bfloat16
    memory_fwd = bytes_per_element * batch_size * seq_len * head_dim * (2 * num_heads_q + 2 * num_heads_kv)
    memory_fwd_gb = memory_fwd / 1e9
    memory_bwd = bytes_per_element * batch_size * seq_len * head_dim * (4 * num_heads_q + 4 * num_heads_kv)
    memory_bwd_gb = memory_bwd / 1e9

    q = torch.randn(
        (batch_size, seq_len, num_heads_q, head_dim),
        device=device,
        requires_grad=True,
    ).to(dtype)
    k = torch.randn(
        (batch_size, seq_len, num_heads_kv, head_dim),
        device=device,
        requires_grad=True,
    ).to(dtype)
    v = torch.randn(
        (batch_size, seq_len, num_heads_kv, head_dim),
        device=device,
        requires_grad=True,
    ).to(dtype)
    o = torch.randn(
        (batch_size, seq_len, num_heads_q, head_dim),
        device=device,
        requires_grad=True,
    ).to(dtype)
    do = torch.randn(
        (batch_size, seq_len, num_heads_q, head_dim),
        device=device,
        requires_grad=True,
    ).to(dtype)
 
    torch.cuda.synchronize()

    # SNR Check
    q_ref = q.clone()
    k_ref = k.clone()
    v_ref = v.clone()
    do_ref = do.clone()
    o_ref, dq_ref, dk_ref, dv_ref = attn_ref(q_ref, k_ref, v_ref, do_ref, causal=True)

    print("Test compile")
    out = test_aiter_compile(q, k, v, do)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)
    print("Test compile end")

    snr_o = compute_snr(out, o_ref)
    snr_dq = compute_snr(dq, dq_ref)
    snr_dk = compute_snr(dk, dk_ref)
    snr_dv = compute_snr(dv, dv_ref)
    print(f"SNR O: {snr_o:.2f} dB")
    print(f"SNR dQ: {snr_dq:.2f} dB")
    print(f"SNR dK: {snr_dk:.2f} dB")
    print(f"SNR dV: {snr_dv:.2f} dB")
 
    # Warm-UP
    ITER = 100
    for _ in range(ITER):
        (
            o,
            _,
        ) = aiter.flash_attn_func(
            q,
            k,
            v,
            causal=True,
            return_lse=True,
            deterministic=False,
        )
        dq, dk, dv = torch.autograd.grad(o, (q, k, v), do)
 
    # event
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # FWD
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(ITER):
        (
            o,
            _,
        ) = aiter.flash_attn_func(
            q,
            k,
            v,
            causal=True,
            return_lse=True,
            deterministic=False,
        )
    end_event.record()
    torch.cuda.synchronize()
    avg_time_fwd = start_event.elapsed_time(end_event) / 1000 / ITER
 
    # FWD + BWD
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(ITER):
        # q.grad = None
        # k.grad = None
        # v.grad = None
        (
            o,
            _,
        ) = aiter.flash_attn_func(
            q,
            k,
            v,
            causal=True,
            return_lse=True,
            deterministic=False,
        )
        dq, dk, dv = torch.autograd.grad(o, (q, k, v), do)
    end_event.record()
    torch.cuda.synchronize()
    avg_time = start_event.elapsed_time(end_event) / 1000 / ITER
 
    avg_time_bwd = avg_time - avg_time_fwd
 
    tflops_fwd = tflop_fwd / avg_time_fwd
    tflops_bwd = tflop_bwd / avg_time_bwd
 
    bandwidth_fwd = memory_fwd_gb / avg_time_fwd
    bandwidth_bwd = memory_bwd_gb / avg_time_bwd

    print(
        "B={}, Seq={}, HeadQ={}, HeadKV={}, Dim={} \n"
        "FWD: Time={:.6f}s, TFLOPS={:.2f}, Bandwidth={:.2f} GB/s\n"
        "BWD: Time={:.6f}s, TFLOPS={:.2f}, Bandwidth={:.2f} GB/s".format(
            batch_size,
            seq_len,
            num_heads_q,
            num_heads_kv,
            head_dim,
            avg_time_fwd,
            tflops_fwd,
            bandwidth_fwd,
            avg_time_bwd,
            tflops_bwd,
            bandwidth_bwd,
        )
    )
   
if __name__ == "__main__":
    test_case(1, 4096, 32, 8, 128)