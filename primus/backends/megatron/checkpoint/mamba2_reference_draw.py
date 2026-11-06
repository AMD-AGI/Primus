###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Rebuild the random part of a reference all-Mamba-2 student's initialization.

HyLo's from-teacher init (``init_with_kqvo``) copies the teacher's attention into x, B, C, and
``out_proj`` of each Mamba-2 mixer. The rest of the mixer keeps its random constructor values:
the z and dt rows of ``in_proj``, the depthwise convolution, ``A_log``, and ``dt_bias``. Those
tensors barely move in Enhanced-ILD, so a run that starts from another draw ends far from the
reference in every tensor that adapts to them.

The reference drew them with ``torch.manual_seed(seed)`` on a GPU, layer by layer:
``in_proj`` (``nn.Linear`` default, bf16), the convolution weight and bias (``nn.Conv1d``
defaults, bf16), ``dt`` from ``torch.rand`` (bf16, log-uniform in ``[dt_min, dt_max]``),
``A`` from ``uniform_(1, 16)`` (float32), then ``out_proj``. PyTorch's GPU generator maps its
stream to elements by the compute-unit count of the GPU (see ``philox_uniform``), so the draw
can only be rebuilt for the count of the GPU the reference was initialized on.

``scan`` finds that count by matching rebuilt tensors against the reference's trained ones;
``write`` puts the rebuilt draw into a copy of an upcycled checkpoint. Both read the widths
from the reference's ``model.safetensors``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from primus.backends.megatron.checkpoint.philox_uniform import (
    FLAVORS,
    grid_threads,
    offset_increment,
    uniform_like_torch,
)

_CONV_KEY = re.compile(r"^(.*)layers\.(\d+)\.mamba\.conv1d\.weight$")


@dataclass(frozen=True)
class Mamba2Widths:
    """Widths of one grouped Mamba-2 mixer (MambaInLlama ``d_xb``), read off its tensors."""

    d_model: int
    d_inner: int
    d_xb: int
    nheads: int
    d_conv: int
    num_layers: int

    @property
    def conv_dim(self) -> int:
        return self.d_inner + 2 * self.d_xb

    @property
    def in_rows(self) -> int:
        return 2 * self.d_inner + 2 * self.d_xb + self.nheads

    @property
    def dt_row(self) -> int:
        """First ``in_proj`` row of dt; rows ``[0, d_inner)`` are z."""
        return 2 * self.d_inner + 2 * self.d_xb

    def draw_sizes(self) -> List[int]:
        """Elements of each per-layer draw, in the reference's order."""
        return [
            self.in_rows * self.d_model,
            self.conv_dim * self.d_conv,
            self.conv_dim,
            self.nheads,
            self.nheads,
            self.d_model * self.d_inner,
        ]


@dataclass(frozen=True)
class InitRanges:
    """Constructor ranges of the reference's Mamba-2 mixer (HyLo's defaults)."""

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_floor: float = 1e-4
    a_low: float = 1.0
    a_high: float = 16.0


class ReferenceMamba2:
    """Mamba-2 tensors of a reference ``model.safetensors``, read on demand."""

    def __init__(self, path: Path):
        from safetensors import safe_open

        self.path = Path(path)
        layers = {}
        with safe_open(str(self.path), framework="pt") as f:
            for key in f.keys():
                match = _CONV_KEY.match(key)
                if match:
                    layers[int(match.group(2))] = match.group(1)
        if not layers:
            raise ValueError(f"{self.path} has no layers.N.mamba.conv1d.weight tensors")
        if sorted(layers) != list(range(len(layers))):
            raise ValueError(f"{self.path} Mamba-2 layers are not contiguous: {sorted(layers)}")
        self.prefix = layers[0]
        self.widths = self._read_widths(len(layers))

    def tensor(self, layer: int, name: str) -> torch.Tensor:
        from safetensors import safe_open

        with safe_open(str(self.path), framework="pt") as f:
            return f.get_tensor(f"{self.prefix}layers.{layer}.mamba.{name}")

    def _read_widths(self, num_layers: int) -> Mamba2Widths:
        in_rows, d_model = self.tensor(0, "in_proj.weight").shape
        conv_dim, _, d_conv = self.tensor(0, "conv1d.weight").shape
        nheads = self.tensor(0, "A_log").shape[0]
        d_inner = in_rows - conv_dim - nheads
        d_xb, odd = divmod(conv_dim - d_inner, 2)
        if odd or d_xb <= 0 or d_inner <= 0:
            raise ValueError(
                f"in_proj rows {in_rows}, conv channels {conv_dim}, and {nheads} heads do not fit "
                "[z, x, B, C, dt] with x and B of one width"
            )
        out_rows, out_cols = self.tensor(0, "out_proj.weight").shape
        if (out_rows, out_cols) != (d_model, d_inner):
            raise ValueError(f"out_proj is {(out_rows, out_cols)}, expected {(d_model, d_inner)}")
        return Mamba2Widths(int(d_model), int(d_inner), int(d_xb), int(nheads), int(d_conv), num_layers)


def layer_offsets(widths: Mamba2Widths, compute_units: int, threads_per_cu: int = 2048) -> List[List[int]]:
    """Generator offset at the start of each draw, per layer, from offset 0 at the first layer."""
    offset, out = 0, []
    for _ in range(widths.num_layers):
        row = []
        for numel in widths.draw_sizes():
            row.append(offset)
            offset += offset_increment(numel, compute_units, threads_per_cu)
        out.append(row)
    return out


def rebuild_layer(
    widths: Mamba2Widths,
    offsets: Sequence[int],
    compute_units: int,
    seed: int = 42,
    flavor: str = "rocrand",
    threads_per_cu: int = 2048,
    ranges: InitRanges = InitRanges(),
) -> Dict[str, torch.Tensor]:
    """The random tensors of one layer: z and dt rows of ``in_proj``, conv, ``dt_bias``, ``A_log``."""
    o_in, o_conv_w, o_conv_b, o_dt, o_a, _ = offsets
    sizes = widths.draw_sizes()
    common = dict(compute_units=compute_units, seed=seed, flavor=flavor, threads_per_cu=threads_per_cu)

    bound = 1.0 / math.sqrt(widths.d_model)
    z_index = np.arange(widths.d_inner * widths.d_model)
    dt_index = np.arange(widths.dt_row * widths.d_model, widths.in_rows * widths.d_model)
    rows = uniform_like_torch(
        np.concatenate([z_index, dt_index]),
        sizes[0],
        o_in,
        low=-bound,
        high=bound,
        dtype=torch.bfloat16,
        **common,
    ).view(-1, widths.d_model)

    conv_bound = 1.0 / math.sqrt(widths.d_conv)
    conv_w = uniform_like_torch(
        np.arange(sizes[1]),
        sizes[1],
        o_conv_w,
        low=-conv_bound,
        high=conv_bound,
        dtype=torch.bfloat16,
        **common,
    ).view(widths.conv_dim, 1, widths.d_conv)
    conv_b = uniform_like_torch(
        np.arange(sizes[2]),
        sizes[2],
        o_conv_b,
        low=-conv_bound,
        high=conv_bound,
        dtype=torch.bfloat16,
        **common,
    )

    unit = uniform_like_torch(
        np.arange(sizes[3]), sizes[3], o_dt, low=0.0, high=1.0, dtype=torch.bfloat16, **common
    )
    log_span = math.log(ranges.dt_max) - math.log(ranges.dt_min)
    dt = torch.clamp(torch.exp(unit * log_span + math.log(ranges.dt_min)), min=ranges.dt_floor)
    a = uniform_like_torch(
        np.arange(sizes[4]),
        sizes[4],
        o_a,
        low=ranges.a_low,
        high=ranges.a_high,
        dtype=torch.float32,
        **common,
    )
    return {
        "in_proj.weight[z]": rows[: widths.d_inner],
        "in_proj.weight[dt]": rows[widths.d_inner :],
        "conv1d.weight": conv_w,
        "conv1d.bias": conv_b,
        "dt_bias": dt + torch.log(-torch.expm1(-dt)),
        "A_log": torch.log(a).to(torch.bfloat16),
    }


def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


@dataclass
class ScanResult:
    compute_units: int
    conv_median: float
    conv_min: float
    z_rows: Optional[float] = None


def scan(
    reference: ReferenceMamba2,
    candidates: Iterable[int],
    seed: int = 42,
    flavor: str = "rocrand",
    threads_per_cu: int = 2048,
    z_samples: int = 65536,
    ties: float = 1e-6,
) -> List[ScanResult]:
    """Score every candidate compute-unit count against the reference, best first.

    Stage one rebuilds every layer's convolution weight; those tensors are small, so only the
    offsets they start from depend on the count. Counts whose offsets agree tie there, so stage
    two scores the tied counts on sampled z rows of the first layer's ``in_proj``, whose element
    order depends on the exact count.
    """
    widths = reference.widths
    sizes = widths.draw_sizes()
    conv_ref = [reference.tensor(i, "conv1d.weight").float() for i in range(widths.num_layers)]
    conv_index = np.arange(sizes[1])
    conv_bound = 1.0 / math.sqrt(widths.d_conv)
    scored: Dict[tuple, float] = {}
    results = []
    for count in candidates:
        offsets = layer_offsets(widths, count, threads_per_cu)
        threads = grid_threads(sizes[1], count, threads_per_cu)
        scores = []
        for layer in range(widths.num_layers):
            key = (layer, offsets[layer][1], threads)
            if key not in scored:
                rebuilt = uniform_like_torch(
                    conv_index,
                    sizes[1],
                    offsets[layer][1],
                    count,
                    -conv_bound,
                    conv_bound,
                    torch.bfloat16,
                    seed,
                    flavor,
                    threads_per_cu,
                )
                scored[key] = correlation(rebuilt.float(), conv_ref[layer])
            scores.append(scored[key])
        results.append(ScanResult(int(count), float(np.median(scores)), float(min(scores))))
    if not results:
        raise ValueError("no candidate compute-unit counts")
    results.sort(key=lambda r: r.conv_median, reverse=True)
    if z_samples <= 0:
        return results

    best = results[0].conv_median
    tied = [r for r in results if best - r.conv_median <= ties]
    total = widths.d_inner * widths.d_model
    index = np.sort(np.random.default_rng(0).choice(total, size=min(z_samples, total), replace=False))
    z_ref = reference.tensor(0, "in_proj.weight").float().flatten()[torch.from_numpy(index)]
    bound = 1.0 / math.sqrt(widths.d_model)
    for r in tied:
        rebuilt = uniform_like_torch(
            index, sizes[0], 0, r.compute_units, -bound, bound, torch.bfloat16, seed, flavor, threads_per_cu
        )
        r.z_rows = correlation(rebuilt.float(), z_ref)
    tied.sort(key=lambda r: r.z_rows, reverse=True)
    tied_ids = {id(r) for r in tied}
    return tied + [r for r in results if id(r) not in tied_ids]


def write(
    template: Path,
    reference: ReferenceMamba2,
    compute_units: int,
    output_dir: Path,
    seed: int = 42,
    flavor: str = "rocrand",
    threads_per_cu: int = 2048,
) -> dict:
    """Copy ``template`` with each mixer's random tensors replaced by the rebuilt draw."""
    from primus.backends.megatron.checkpoint.hybrid_upcycle import (
        build_upcycle_checkpoint,
        load_checkpoint,
    )

    widths = reference.widths
    state, envelope = load_checkpoint(Path(template))
    mixers = sorted(
        int(m.group(1))
        for key in state
        if (m := re.match(r"^decoder\.layers\.(\d+)\.mixer\.in_proj\.weight$", key))
    )
    if len(mixers) != widths.num_layers:
        raise ValueError(f"template has {len(mixers)} Mamba-2 mixers, the reference {widths.num_layers}")

    offsets = layer_offsets(widths, compute_units, threads_per_cu)
    report = {"compute_units": compute_units, "seed": seed, "flavor": flavor, "layers": []}
    for layer, index in enumerate(mixers):
        drawn = rebuild_layer(widths, offsets[layer], compute_units, seed, flavor, threads_per_cu)
        prefix = f"decoder.layers.{index}.mixer."
        in_proj = state[prefix + "in_proj.weight"]
        if tuple(in_proj.shape) != (widths.in_rows, widths.d_model):
            raise ValueError(
                f"{prefix}in_proj.weight is {tuple(in_proj.shape)}, the reference's is "
                f"{(widths.in_rows, widths.d_model)}"
            )
        in_proj = in_proj.clone()
        in_proj[: widths.d_inner] = drawn["in_proj.weight[z]"].to(in_proj.dtype)
        in_proj[widths.dt_row :] = drawn["in_proj.weight[dt]"].to(in_proj.dtype)
        state[prefix + "in_proj.weight"] = in_proj
        for name in ("conv1d.weight", "conv1d.bias", "dt_bias", "A_log"):
            current = state[prefix + name]
            if tuple(current.shape) != tuple(drawn[name].shape):
                raise ValueError(
                    f"{prefix}{name} is {tuple(current.shape)}, rebuilt {tuple(drawn[name].shape)}"
                )
            state[prefix + name] = drawn[name].to(current.dtype).clone()

        ref_in = reference.tensor(layer, "in_proj.weight")
        report["layers"].append(
            {
                "conv1d.weight": correlation(
                    drawn["conv1d.weight"], reference.tensor(layer, "conv1d.weight")
                ),
                "conv1d.bias": correlation(drawn["conv1d.bias"], reference.tensor(layer, "conv1d.bias")),
                "in_proj.weight[z]": correlation(drawn["in_proj.weight[z]"], ref_in[: widths.d_inner]),
                "in_proj.weight[dt]": correlation(drawn["in_proj.weight[dt]"], ref_in[widths.dt_row :]),
                "A_log": correlation(drawn["A_log"], reference.tensor(layer, "A_log")),
                "dt_bias": correlation(drawn["dt_bias"], reference.tensor(layer, "dt_bias")),
            }
        )

    checkpoint = build_upcycle_checkpoint(envelope, state)
    output_dir = Path(output_dir)
    iter_dir = output_dir / "iter_0000000" / "mp_rank_00"
    iter_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, iter_dir / "model_optim_rng.pt")
    (output_dir / "latest_checkpointed_iteration.txt").write_text("0\n")
    report["widths"] = asdict(widths)
    (output_dir / "reference_draw_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _candidates(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        if "-" in part:
            low, high = (int(v) for v in part.split("-", 1))
            out.extend(range(low, high + 1))
        elif part.strip():
            out.append(int(part))
    return out


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reference", required=True, help="the reference model.safetensors")
    parser.add_argument("--seed", type=int, default=42, help="seed the reference was built with")
    parser.add_argument(
        "--flavor", choices=FLAVORS, default="rocrand", help="random library's float conversion"
    )
    parser.add_argument("--threads-per-cu", type=int, default=2048, help="resident threads per compute unit")


def scan_main(argv: Optional[Sequence[str]] = None) -> List[ScanResult]:
    parser = argparse.ArgumentParser(
        description="Find the compute-unit count that rebuilds the reference draw."
    )
    _common_args(parser)
    parser.add_argument("--candidates", default="1-512", help="counts to try, e.g. '1-512' or '60,64-80'")
    parser.add_argument("--top", type=int, default=5, help="rows to print")
    args = parser.parse_args(argv)

    reference = ReferenceMamba2(Path(args.reference))
    print(f"reference widths: {asdict(reference.widths)}")
    results = scan(reference, _candidates(args.candidates), args.seed, args.flavor, args.threads_per_cu)
    print(f"{'compute units':>13s} {'conv median':>11s} {'conv min':>9s} {'z rows':>7s}")
    for r in results[: args.top]:
        z = f"{r.z_rows:.4f}" if r.z_rows is not None else "-"
        print(f"{r.compute_units:13d} {r.conv_median:11.4f} {r.conv_min:9.4f} {z:>7s}")
    best = results[0]
    print(f"best: --compute-units {best.compute_units}")
    return results


def write_main(argv: Optional[Sequence[str]] = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Write the rebuilt reference draw into an upcycled checkpoint."
    )
    _common_args(parser)
    parser.add_argument("--template", required=True, help="upcycled all-Mamba-2 checkpoint to copy")
    parser.add_argument("--compute-units", type=int, required=True, help="count reported by the scan")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    reference = ReferenceMamba2(Path(args.reference))
    report = write(
        Path(args.template),
        reference,
        args.compute_units,
        Path(args.output_dir),
        args.seed,
        args.flavor,
        args.threads_per_cu,
    )
    names = list(report["layers"][0])
    print("layer " + " ".join(f"{n:>18s}" for n in names) + "   (rebuilt vs reference, correlation)")
    for i, row in enumerate(report["layers"]):
        print(f"L{i:02d}   " + " ".join(f"{row[n]:18.4f}" for n in names))
    print(f"wrote {Path(args.output_dir) / 'iter_0000000'}")
    return report
