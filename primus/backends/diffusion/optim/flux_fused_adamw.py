"""AdamW with the gradient-clip scale folded into one multi-tensor Triton launch.

torch's fused AdamW can divide by a clip scale (`grad_scale`), but then it also stores the
unscaled gradient back (GradScaler semantics), so the pass saved from clip_grads_with_norm_
comes back as an extra write. This step reads (param, grad, exp_avg, exp_avg_sq), writes
(param, exp_avg, exp_avg_sq) and nothing else, over every fp32 shard in one launch.

The per-element math mirrors ATen's _adam_math (ADAMW mode) op for op, with bias corrections
from the device step count taken in double and rounded to fp32 as ATen does. State layout
(`step`, `exp_avg`, `exp_avg_sq`) is torch's, so checkpoints are interchangeable with
torch.optim.AdamW. Anything this kernel does not cover falls back to the torch fused step.
"""

import math

import numpy as np
import torch
import triton
import triton.language as tl
from torch.optim import AdamW

BLOCK = 1024


@triton.jit
def _adamw_mt_kernel(
    blk_tensor,
    blk_first,
    numels,
    p_tab,
    g_tab,
    m_tab,
    v_tab,
    scale_ptr,
    lr,
    lrwd,
    beta1,
    beta2,
    eps,
    step_size,
    bc2_sqrt,
    HAS_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    t = tl.load(blk_tensor + pid)
    start = (pid - tl.load(blk_first + t)).to(tl.int64) * BLOCK
    n = tl.load(numels + t)
    p_ptr = tl.multiple_of(tl.load(p_tab + t).to(tl.pointer_type(tl.float32)), 16)
    g_ptr = tl.multiple_of(tl.load(g_tab + t).to(tl.pointer_type(tl.float32)), 16)
    m_ptr = tl.multiple_of(tl.load(m_tab + t).to(tl.pointer_type(tl.float32)), 16)
    v_ptr = tl.multiple_of(tl.load(v_tab + t).to(tl.pointer_type(tl.float32)), 16)
    offs = start + tl.arange(0, BLOCK)
    full = start + BLOCK <= n
    mask = offs < n
    if full:
        p = tl.load(p_ptr + offs)
        g = tl.load(g_ptr + offs)
        m = tl.load(m_ptr + offs)
        v = tl.load(v_ptr + offs)
    else:
        p = tl.load(p_ptr + offs, mask=mask)
        g = tl.load(g_ptr + offs, mask=mask)
        m = tl.load(m_ptr + offs, mask=mask)
        v = tl.load(v_ptr + offs, mask=mask)
    if HAS_SCALE:
        g = tl.math.div_rn(g, tl.load(scale_ptr))
    p = p - lrwd * p
    m = tl.fma(beta1, m, tl.fma(-beta1, g, g))
    g2 = g * g
    v = tl.fma(beta2, v, tl.fma(-beta2, g2, g2))
    denom = tl.math.div_rn(tl.math.sqrt_rn(v), bc2_sqrt) + eps
    p = p - tl.math.div_rn(step_size * m, denom)
    if full:
        tl.store(p_ptr + offs, p)
        tl.store(m_ptr + offs, m)
        tl.store(v_ptr + offs, v)
    else:
        tl.store(p_ptr + offs, p, mask=mask)
        tl.store(m_ptr + offs, m, mask=mask)
        tl.store(v_ptr + offs, v, mask=mask)


def _local(t):
    t = getattr(t, "_local_tensor", t)
    # The FSDP2 all-gather extensions wrap the fp32 shard in a storage-less subclass whose
    # data_ptr() is 0; the kernel must address the wrapped storage.
    while type(t) is not torch.Tensor and torch.is_tensor(getattr(t, "_tensor", None)):
        t = t._tensor
    return t


def _usable(t):
    return (
        t.dtype == torch.float32
        and t.is_cuda
        and t.is_contiguous()
        and t.data_ptr() != 0
        and t.data_ptr() % 16 == 0
    )


_said = set()


def _say(msg):
    """Say once, on rank 0, which optimizer path actually ran.

    Every reason this class declines is a silent fallback to torch's fused step, which produces a
    correct run at stock speed -- indistinguishable in any log from the fused kernel running and
    buying nothing. That cost a real measurement: a 4-node cell reported 77.98 block-2 for a recipe
    including FLUX_FUSED_ADAMW=1, and the fused step had in fact rejected every shard, because
    FLUX_FP8_ALL_GATHER=1 wraps the fp32 shard in a storage-less subclass whose data_ptr() is 0.
    The number was read as "the fused AdamW is worth nothing" when the kernel had never run.
    """
    if msg in _said:
        return
    _said.add(msg)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"FusedClipAdamW: {msg}", flush=True)


def _f32(x):
    return float(np.float32(x))


class FusedClipAdamW(AdamW):
    """AdamW(fused=True) whose step also applies `clip_scale` (grads are divided by it)."""

    def __init__(self, params, **kwargs):
        kwargs["fused"] = True
        super().__init__(params, **kwargs)
        self.clip_scale = None
        self._plans = None
        self.register_load_state_dict_post_hook(lambda opt: setattr(opt, "_plans", None))

    def _build_plan(self, group):
        if (
            group["amsgrad"]
            or group["maximize"]
            or group["capturable"]
            or group["differentiable"]
            or torch.is_tensor(group["lr"])
        ):
            return None
        params, grads, m, v, steps = [], [], [], [], []
        self._init_group(group, params, grads, m, v, [], steps)
        if len(params) != len(group["params"]):
            return None
        locs = [[_local(t) for t in ts] for ts in (params, m, v)]
        keep = []
        for i, (p, mi, vi) in enumerate(zip(*locs)):
            if p.numel() == mi.numel() == vi.numel() == 0:
                continue
            # An empty tensor's data_ptr() can be 0, so every shard must be non-empty and match.
            if not p.numel() == mi.numel() == vi.numel() or not all(_usable(t) for t in (p, mi, vi)):
                return None
            keep.append(i)
        if not keep:
            return None
        dev = locs[0][keep[0]].device
        if any(t.device != dev for ts in locs for t in ts if t.numel()):
            return None
        numels = [locs[0][i].numel() for i in keep]
        nblk = [math.ceil(n / BLOCK) for n in numels]
        first = np.concatenate([[0], np.cumsum(nblk)[:-1]])
        blk_tensor = np.repeat(np.arange(len(keep), dtype=np.int32), nblk)
        step_vals = torch.stack(steps).cpu()
        if not bool((step_vals == step_vals[0]).all()):
            return None

        def tab(ts):
            return torch.tensor([ts[i].data_ptr() for i in keep], dtype=torch.int64, device=dev)

        return {
            "params": params,
            "keep": keep,
            "steps": steps,
            "host_step": int(step_vals[0]),
            "nblocks": int(sum(nblk)),
            "blk_tensor": torch.from_numpy(blk_tensor).to(dev),
            "blk_first": torch.tensor(first, dtype=torch.int32, device=dev),
            "numels": torch.tensor(numels, dtype=torch.int64, device=dev),
            "numel_list": numels,
            "p_tab": tab(locs[0]),
            "m_tab": tab(locs[1]),
            "v_tab": tab(locs[2]),
            "g_ptrs": None,
            "g_tab": None,
        }

    def _grad_table(self, plan):
        ptrs = []
        dev = plan["p_tab"].device
        for i, n in zip(plan["keep"], plan["numel_list"]):
            g = plan["params"][i].grad
            if g is None:
                return None
            g = _local(g)
            if g.numel() != n or g.device != dev or not _usable(g):
                return None
            ptrs.append(g.data_ptr())
        if ptrs != plan["g_ptrs"]:
            # The pinned block is recycled only after this copy completes (caching host allocator).
            host = torch.tensor(ptrs, dtype=torch.int64).pin_memory()
            plan["g_tab"] = host.to(plan["p_tab"].device, non_blocking=True)
            plan["g_ptrs"] = ptrs
        return plan["g_tab"]

    def _fallback(self, closure):
        self.grad_scale = self.clip_scale
        try:
            return super().step(closure)
        finally:
            self.grad_scale = None
            self._plans = None

    @torch.no_grad()
    def step(self, closure=None):
        scale, self.clip_scale = self.clip_scale, None
        if closure is not None:
            self.clip_scale = scale
            return self._fallback(closure)
        if self._plans is None:
            self._plans = [self._build_plan(g) for g in self.param_groups]
        if any(p is None for p in self._plans):
            _say("DECLINED -- falling back to the torch fused step, so this run measures stock AdamW")
            self.clip_scale = scale
            return self._fallback(None)
        tabs = [self._grad_table(p) for p in self._plans]
        if any(t is None for t in tabs):
            self.clip_scale = scale
            return self._fallback(None)
        if scale is not None:
            scale = scale.to(device=self._plans[0]["p_tab"].device, dtype=torch.float32)
        _say(
            "fused multi-tensor step ACTIVE over " + str(sum(len(p["keep"]) for p in self._plans)) + " shards"
        )
        for group, plan, g_tab in zip(self.param_groups, self._plans, tabs):
            plan["host_step"] += 1
            torch._foreach_add_(plan["steps"], 1)
            beta1, beta2 = group["betas"]
            step = float(plan["host_step"])
            bc1 = _f32(1 - beta1**step)
            bc2_sqrt = _f32(math.sqrt(1 - beta2**step))
            lr = _f32(group["lr"])
            step_size = float(np.float32(lr) / np.float32(bc1))
            lrwd = float(np.float32(lr) * np.float32(_f32(group["weight_decay"])))
            _adamw_mt_kernel[(plan["nblocks"],)](
                plan["blk_tensor"],
                plan["blk_first"],
                plan["numels"],
                plan["p_tab"],
                g_tab,
                plan["m_tab"],
                plan["v_tab"],
                scale if scale is not None else plan["p_tab"],
                lr,
                lrwd,
                _f32(beta1),
                _f32(beta2),
                _f32(group["eps"]),
                step_size,
                bc2_sqrt,
                HAS_SCALE=scale is not None,
                BLOCK=BLOCK,
                num_warps=8,
                enable_fp_fusion=False,
            )
        return None
