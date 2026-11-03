###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""L2 op spec (``OPS``, the l2-op-parity format) for Inkling's fused kernels against their eager
references: the relative-bias flash attention (``inkling_attention_impl: fused``) and the short
convolution through ``causal_conv1d_fn`` with ``seq_idx`` (``inkling_sconv_impl``).

Pytest runs every op on a GPU through ``test_inkling_l2_ops.py`` (forward and backward against
the reference). The same ``OPS`` list is also the input format of an external L2 op-parity runner
(``l2_ops.py ops --spec <this file>``), which is not part of this repository."""

import torch
import torch.nn.functional as F

from primus.backends.megatron.core.transformer.inkling.inkling_layers import (
    causal_conv1d_kernel,
    eager_causal_conv1d,
)

B = 2
BLOCK = 64
_DT = {"fp32": torch.float32, "bf16": torch.bfloat16}


def attention_reference(q, k, v, rel, scale, window, mask=None):
    """Dense eager reference: scale * q k^T + rel[q, q - k] (0 <= q - k < E), causal / window."""
    b, h, s, d = q.shape
    rep = h // k.shape[1]
    k = k.repeat_interleave(rep, 1)
    v = v.repeat_interleave(rep, 1)
    E = rel.shape[-1]
    scores = (q @ k.transpose(-1, -2)) * scale
    pos = torch.arange(s, device=q.device)
    dist = pos[:, None] - pos[None, :]
    idx = dist.clamp(0, E - 1).expand(b, h, s, s)
    bias = torch.gather(rel, -1, idx) * ((dist >= 0) & (dist < E))
    masked = dist < 0
    if window:
        masked = masked | (dist >= window)
    if mask is not None:
        masked = masked | mask
    scores = (scores + bias).masked_fill(masked, float("-inf"))
    p = torch.softmax(scores.float(), -1).nan_to_num(0.0).to(q.dtype)
    return p @ v


def _attn_inputs(H, KV, E, window, with_mask=False):
    def inputs(shape, dtype, device, g):
        S, D = shape["S"], 128
        q = F.normalize(torch.randn(B, H, S, D, generator=g, device=device), dim=-1) * D**0.5
        k = F.normalize(torch.randn(B, KV, S, D, generator=g, device=device), dim=-1) * D**0.5
        v = torch.randn(B, KV, S, D, generator=g, device=device)
        rel = torch.randn(B, H, S, E, generator=g, device=device)
        x = {
            "q": q.to(_DT[dtype]),
            "k": k.to(_DT[dtype]),
            "v": v.to(_DT[dtype]),
            "rel": rel.to(_DT[dtype]),
            "scale": 1.0 / D,
            "window": window,
            "mask": None,
        }
        if with_mask:  # key padding: the last 3 keys of sample 1 masked
            m = torch.zeros(B, 1, 1, S, dtype=torch.bool, device=device)
            m[1, ..., max(1, S - 3) :] = True
            x["mask"] = m
        return x

    return inputs


def attention_kernel(x):
    from primus.backends.megatron.core.transformer.inkling.inkling_relpos_attention import (
        relpos_flash_attention,
    )

    return relpos_flash_attention(x["q"], x["k"], x["v"], x["rel"], x["scale"], x["window"], x["mask"])


def attention_ref(x):
    return attention_reference(x["q"], x["k"], x["v"], x["rel"], x["scale"], x["window"], x["mask"])


def _conv_inputs(C, docs):
    def inputs(shape, dtype, device, g):
        T = shape["T"]
        x = torch.randn(T, B, C, generator=g, device=device).float()  # [s, b, c] as InklingShortConv
        w = torch.randn(C, 4, generator=g, device=device) * 0.5
        seq = None
        if docs:  # document boundaries at random positions (packed sequences)
            cuts = (torch.rand(B, T, generator=g, device=device) < 0.05).int()
            cuts[:, 0] = 0
            seq = cuts.cumsum(-1).to(torch.int32)
        return {"x": x, "weight": w, "seq_idx": seq}

    return inputs


def conv_kernel(x):
    xt = x["x"].contiguous().permute(1, 2, 0)  # channel-last [b, c, s] view, as InklingShortConv
    seq = None if x["seq_idx"] is None else x["seq_idx"].contiguous()
    return causal_conv1d_kernel()(xt, x["weight"], None, seq_idx=seq, activation=None)


def conv_ref(x):
    return eager_causal_conv1d(x["x"].permute(1, 2, 0), x["weight"], x["seq_idx"])


OPS = []
for name, H, KV, E, window in (("local", 8, 4, 512, 512), ("global", 8, 2, 1024, None)):
    for with_mask in (False, True):
        OPS.append(
            {
                "name": f"relpos_flash_attention_{name}" + ("_mask" if with_mask else ""),
                "kernel": attention_kernel,
                "reference": attention_ref,
                "inputs": _attn_inputs(H, KV, E, window, with_mask),
                "ragged": {"S": BLOCK},
                "shapes": [{"S": 1100}],
                "grad": ["q", "k", "v", "rel"],
                "dtypes": ["fp32", "bf16"],
                "tol": {"fp32": 1e-4, "bf16": 2e-2},
            }
        )
if causal_conv1d_kernel() is not None:
    for C in (768, 512, 256):
        for docs in (False, True):
            OPS.append(
                {
                    "name": f"sconv_causal_conv1d_C{C}" + ("_seq_idx" if docs else ""),
                    "kernel": conv_kernel,
                    "reference": conv_ref,
                    "inputs": _conv_inputs(C, docs),
                    "fixed": {"T": 2 * BLOCK},
                    "shapes": [{"T": t} for t in (2 * BLOCK + 1, 2 * BLOCK - 1, BLOCK - 1, 3, 1500)],
                    "grad": ["x", "weight"],
                    "dtypes": ["fp32"],
                    "tol": {"fp32": 1e-4},
                }
            )
