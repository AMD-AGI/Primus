###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class AttnProfileResult:
    fwd_tflops: float
    fwd_time_s: float
    bwd_tflops: float
    bwd_time_s: float


class FlashAttnProfiler:
    def __init__(
        self,
        *,
        batch_size: int,
        seq_len: int,
        num_head_q: int,
        num_head_kv: int,
        head_dim_qk: int,
        head_dim_v: int,
        causal: bool,
        dtype,
        device,
        num_runs: int = 100,
    ):
        if torch is None:
            raise RuntimeError("torch is required for FlashAttnProfiler")

        self.causal = causal
        self.num_runs = int(num_runs)

        self.q = torch.randn(
            (batch_size, seq_len, num_head_q, head_dim_qk),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        self.k = torch.randn(
            (batch_size, seq_len, num_head_kv, head_dim_v),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        self.v = torch.randn(
            (batch_size, seq_len, num_head_kv, head_dim_v),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        self.o = torch.randn(
            (batch_size, seq_len, num_head_q, head_dim_v),
            dtype=dtype,
            device=device,
        )
        self.o_grad = torch.randn_like(self.o)

        tflop_fwd = 2 * batch_size * seq_len * seq_len * num_head_q * (head_dim_qk + head_dim_v) / 1e12
        if causal:
            tflop_fwd *= 0.5
        self.tflop_fwd = float(tflop_fwd)
        self.tflop_bwd = float(self.tflop_fwd * 2.5)

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def profile(self) -> AttnProfileResult:
        if torch is None:
            raise RuntimeError("torch is required for FlashAttnProfiler")

        from flash_attn import flash_attn_func  # type: ignore

        # warm up
        for _ in range(10):
            self.q.grad = None
            self.k.grad = None
            self.v.grad = None
            self.o = flash_attn_func(self.q, self.k, self.v, causal=self.causal)
            self.o.backward(self.o_grad)
        torch.cuda.synchronize()

        # FWD
        self.start_event.record()
        for _ in range(self.num_runs):
            self.q.grad = None
            self.k.grad = None
            self.v.grad = None
            self.o = flash_attn_func(self.q, self.k, self.v, causal=self.causal)
        self.end_event.record()
        torch.cuda.synchronize()
        fwd_time_s = self.start_event.elapsed_time(self.end_event) / self.num_runs / 1000
        fwd_tflops = self.tflop_fwd / max(1e-12, fwd_time_s)

        # FWD + BWD
        self.start_event.record()
        for _ in range(self.num_runs):
            self.q.grad = None
            self.k.grad = None
            self.v.grad = None
            self.o = flash_attn_func(self.q, self.k, self.v, causal=self.causal)
            self.o.backward(self.o_grad)
        self.end_event.record()
        torch.cuda.synchronize()
        fwd_bwd_time_s = self.start_event.elapsed_time(self.end_event) / self.num_runs / 1000

        bwd_time_s = max(1e-12, fwd_bwd_time_s - fwd_time_s)
        bwd_tflops = self.tflop_bwd / bwd_time_s

        return AttnProfileResult(
            fwd_tflops=float(fwd_tflops),
            fwd_time_s=float(fwd_time_s),
            bwd_tflops=float(bwd_tflops),
            bwd_time_s=float(bwd_time_s),
        )


def flash_attention_profile(
    *,
    batch_size: int,
    seq_len: int,
    num_head_q: int,
    num_head_kv: int,
    head_dim_qk: int,
    head_dim_v: int,
    causal: bool,
    dtype,
    device,
) -> AttnProfileResult:
    profiler = FlashAttnProfiler(
        batch_size=batch_size,
        seq_len=seq_len,
        num_head_q=num_head_q,
        num_head_kv=num_head_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        causal=causal,
        dtype=dtype,
        device=device,
    )
    try:
        return profiler.profile()
    except Exception:
        return AttnProfileResult(0.0, 0.0, 0.0, 0.0)


class CKProfiler:
    def __init__(
        self,
        *,
        batch_size: int,
        seq_len: int,
        num_head_q: int,
        num_head_kv: int,
        head_dim_qk: int,
        head_dim_v: int,
        causal: bool,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_head_q = num_head_q
        self.num_head_kv = num_head_kv
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.causal = causal

        tflop_fwd = 2 * batch_size * seq_len * seq_len * num_head_q * (head_dim_qk + head_dim_v) / 1e12
        if causal:
            tflop_fwd *= 0.5
        self.tflop_fwd = float(tflop_fwd)
        self.tflop_bwd = float(self.tflop_fwd * 2.5)

        self.fwd_script = [
            "tile_example_fmha_fwd",
            "-mode=0",
            f"-b={self.batch_size}",
            f"-h={self.num_head_q}",
            f"-h_k={self.num_head_kv}",
            f"-s={self.seq_len}",
            f"-s_k={self.seq_len}",
            f"-d={self.head_dim_qk}",
            f"-d_v={self.head_dim_v}",
            "-prec=bf16",
            "-mask={}".format(2 if self.causal else 0),
            "-warmup=100",
            "-repeat=100",
            "-v=0",
            "-lse=1",
        ]

        self.bwd_script = self.fwd_script.copy()
        self.bwd_script[0] = "tile_example_fmha_bwd"

    def _run_ms(self, cmd) -> float:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        match = re.search(r"([\d\.]+)\s*ms", res.stdout)
        if not match:
            return 0.0
        return float(match.group(1)) / 1000.0

    def profile(self) -> AttnProfileResult:
        fwd_time_s = self._run_ms(self.fwd_script)
        bwd_time_s = self._run_ms(self.bwd_script)
        fwd_tflops = self.tflop_fwd / max(1e-12, fwd_time_s) if fwd_time_s > 0 else 0.0
        bwd_tflops = self.tflop_bwd / max(1e-12, bwd_time_s) if bwd_time_s > 0 else 0.0
        return AttnProfileResult(
            fwd_tflops=float(fwd_tflops),
            fwd_time_s=float(fwd_time_s),
            bwd_tflops=float(bwd_tflops),
            bwd_time_s=float(bwd_time_s),
        )


def ck_attention_profile(
    *,
    batch_size: int,
    seq_len: int,
    num_head_q: int,
    num_head_kv: int,
    head_dim_qk: int,
    head_dim_v: int,
    causal: bool,
) -> AttnProfileResult:
    profiler = CKProfiler(
        batch_size=batch_size,
        seq_len=seq_len,
        num_head_q=num_head_q,
        num_head_kv=num_head_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        causal=causal,
    )
    try:
        return profiler.profile()
    except Exception:
        return AttnProfileResult(0.0, 0.0, 0.0, 0.0)


def resolve_backend(name: str):
    n = (name or "").strip().lower()
    if n == "flash":
        return ("flash",)
    if n == "ck":
        return ("ck",)
    if n == "all":
        return ("flash", "ck")
    raise ValueError(f"Unknown backend: {name}")


def profile_one(
    *,
    backend: str,
    batch_size: int,
    seq_len: int,
    num_head_q: int,
    num_head_kv: int,
    head_dim_qk: int,
    head_dim_v: int,
    causal: bool,
    dtype,
    device,
) -> AttnProfileResult:
    if backend == "flash":
        return flash_attention_profile(
            batch_size=batch_size,
            seq_len=seq_len,
            num_head_q=num_head_q,
            num_head_kv=num_head_kv,
            head_dim_qk=head_dim_qk,
            head_dim_v=head_dim_v,
            causal=causal,
            dtype=dtype,
            device=device,
        )
    if backend == "ck":
        return ck_attention_profile(
            batch_size=batch_size,
            seq_len=seq_len,
            num_head_q=num_head_q,
            num_head_kv=num_head_kv,
            head_dim_qk=head_dim_qk,
            head_dim_v=head_dim_v,
            causal=causal,
        )
    raise ValueError(f"Unknown backend: {backend}")
