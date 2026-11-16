###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compiled-step benchmark projection for MaxText/JAX.

A MaxText step is one XLA program.  With ``scan_layers`` the decoder stack is a
``while`` loop whose body is a single compiled layer, and that body already
contains the layer's intra-node collectives (FSDP all-gather / reduce-scatter,
expert all-to-all, TP all-reduce), its rematerialization and whatever overlap
the latency-hiding scheduler found.  Step time is therefore affine in the
number of loop iterations::

    T(L) = T_fixed + L * T_layer

so timing the *real* training step on one node at two reduced depths measures
both terms directly, on the target's own intra-node mesh.  ``T_fixed`` holds
the embedding, output head, loss and the optimizer update; ``T_layer`` is the
loop body.  Peak memory is affine in depth for the same reason (stacked
parameters, optimizer state and saved residuals all scale with ``L``), so the
same two runs give the memory slope.

What a single node cannot measure is traffic over the scale-out network.  The
DCN mesh axes say exactly which collectives cross it, and the projection adds
them analytically:

* ``dcn_data``: the data-parallel gradient reduction.  Every GPU holds the
  gradient shard its ICI axes give it and all-reduces it with the GPUs at the
  same local rank on the other nodes (one NIC per GPU).
* ``dcn_fsdp``: the FSDP weight all-gather in the forward and backward of every
  layer and the gradient reduce-scatter, over the same rail.
* ``dcn_expert``: the expert all-to-all of every MoE layer.

How much of that traffic hides behind compute is set by the ``overlap`` model
(``--dcn-overlap``): ``none`` charges all of it, ``layer`` lets each layer's
share hide behind that layer's measured compute and charges only the excess.
``allreduce`` (the default) charges the data-parallel gradient all-reduces in
full, because MaxText issues them at the end of the scanned layer body after the
compute they could hide behind, and treats the other collectives like ``layer``.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_STEP_RE = re.compile(
    r"completed step: (\d+), seconds: ([\d.]+), TFLOP/s/device: ([\d.]+), Tokens/s/device: ([\d.]+)"
)
_MEM_TOTAL_RE = re.compile(r"Total memory size: ([\d.]+) GB, Output size: ([\d.]+) GB, Temp size: ([\d.]+) GB, Argument size: ([\d.]+) GB")
_MEMSTATS_RE = re.compile(r"Using \(GB\) ([\d.]+) / ([\d.]+)")
_PARAMS_RE = re.compile(r"number parameters: ([\d.]+) billion", re.IGNORECASE)

_GiB = 1024**3


def _log(msg: str) -> None:
    print(f"[Primus:Projection:CompiledStep] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Workload description
# ---------------------------------------------------------------------------


@dataclass
class MeshSpec:
    ici: Dict[str, int]
    dcn: Dict[str, int]

    def ici_product(self, axes: Sequence[str]) -> int:
        return math.prod(self.ici.get(a, 1) for a in axes)

    def dcn_product(self, axes: Sequence[str]) -> int:
        return math.prod(self.dcn.get(a, 1) for a in axes)


@dataclass
class StepSample:
    layers: int
    step_s: float
    tokens_per_s_per_gpu: float
    tflops_per_gpu: float
    peak_gib: Optional[float]
    compiled_total_gib: Optional[float]
    n_steps: int
    log: str
    live_gib: Optional[float] = None  # live bytes after the last step: the persistent train state


@dataclass
class Projection:
    target_nodes: int
    layers: int
    step_s: float
    t_fixed_s: float
    t_layer_s: float
    dcn_comm_s: float
    dcn_exposed_s: float
    tokens_per_s_per_gpu: float
    peak_gib: Optional[float]
    detail: Dict[str, float] = field(default_factory=dict)


_MESH_AXES = (
    "data",
    "fsdp",
    "fsdp_transpose",
    "sequence",
    "context",
    "context_autoregressive",
    "tensor",
    "tensor_transpose",
    "tensor_sequence",
    "expert",
    "autoregressive",
    "pipeline",
)


def _resolve_group(values: Dict[str, int], devices: int) -> Dict[str, int]:
    unspecified = [a for a, v in values.items() if v < 0]
    if len(unspecified) > 1:
        raise ValueError(f"more than one mesh axis is -1: {unspecified}")
    if unspecified:
        known = math.prod(v for v in values.values() if v > 0)
        values[unspecified[0]] = max(1, devices // known)
    return values


def resolve_mesh(params: Dict[str, object], nodes: int, gpus_per_node: int) -> MeshSpec:
    """MaxText's ICI/DCN mesh for *nodes* nodes, with -1 axes filled."""
    ici = {}
    dcn = {}
    for axis in _MESH_AXES:
        ici[axis] = int(params.get(f"ici_{axis}_parallelism", -1 if axis == "fsdp" else 1))
        dcn[axis] = int(params.get(f"dcn_{axis}_parallelism", -1 if axis == "data" else 1))
    return MeshSpec(ici=_resolve_group(ici, gpus_per_node), dcn=_resolve_group(dcn, nodes))


# ---------------------------------------------------------------------------
# Parameter accounting (per layer, sharded the way MaxText shards it)
# ---------------------------------------------------------------------------


def layer_param_counts(spec) -> Tuple[float, float, float]:
    """(attention+norm params, dense-MLP params, routed-expert params) of one layer."""
    h = spec.hidden_size
    if spec.multi_latent_attention:
        q = (h * spec.q_lora_rank + spec.q_lora_rank * spec.num_attention_heads * (spec.qk_head_dim + spec.qk_pos_emb_head_dim)) if spec.q_lora_rank else h * spec.num_attention_heads * (spec.qk_head_dim + spec.qk_pos_emb_head_dim)
        kv = h * (spec.kv_lora_rank + spec.qk_pos_emb_head_dim) + spec.kv_lora_rank * spec.num_attention_heads * (spec.qk_head_dim + spec.v_head_dim)
        o = spec.num_attention_heads * spec.v_head_dim * h
        attn = q + kv + o
    else:
        attn = h * spec.kv_channels * (spec.num_attention_heads + 2 * spec.num_query_groups) + spec.num_attention_heads * spec.kv_channels * h
    attn += 2 * h
    mats = 3 if spec.swiglu else 2
    if spec.num_experts:
        experts = spec.num_experts * mats * h * spec.moe_ffn_hidden_size
        shared = spec.num_shared_experts * mats * h * spec.moe_ffn_hidden_size + h * spec.num_experts
        return attn, shared, experts
    return attn, mats * h * spec.ffn_hidden_size, 0.0


def model_param_counts(spec) -> Dict[str, float]:
    attn, mlp, experts = layer_param_counts(spec)
    dense_layers = spec.num_dense_layers if spec.num_experts else spec.num_layers
    moe_layers = spec.num_layers - dense_layers if spec.num_experts else 0
    mats = 3 if spec.swiglu else 2
    dense_mlp = mats * spec.hidden_size * spec.ffn_hidden_size
    embed = spec.vocab_size * spec.hidden_size * (1 if spec.tie_embeddings else 2)
    return {
        "per_moe_layer_nonexpert": attn + mlp,
        "per_moe_layer_expert": experts,
        "per_dense_layer": attn + dense_mlp,
        "dense_layers": dense_layers,
        "moe_layers": moe_layers,
        "embedding": embed,
    }


# ---------------------------------------------------------------------------
# Scale-out collectives
# ---------------------------------------------------------------------------


@dataclass
class Fabric:
    nic_gbps: float = 400.0  # per NIC line rate, Gb/s
    nics_per_node: int = 8
    efficiency: float = 0.70  # achieved / line rate for large ring collectives
    latency_us: float = 15.0  # per ring step
    # Measured per-rail bus bandwidth (GB/s) by collective, from the DCN microbenchmark.
    # When set they replace line rate x efficiency for that collective.
    busbw_gbs: Dict[str, float] = field(default_factory=dict)
    # Measured bus bandwidth vs. per-GPU message size: op -> [(bytes, GB/s), ...] sorted by size.
    busbw_curve: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    # Measured exposed time of a DCN all-reduce run concurrently with GEMMs: [(bytes, seconds), ...].
    # Captures compute/communication interference, which bandwidth alone does not.
    overlap_curve: List[Tuple[float, float]] = field(default_factory=list)
    # Each RCCL channel runs one workgroup that holds a compute unit while the collective is in flight.
    cus_per_gpu: int = 304
    # Measured exposed fraction of a DCN collective run next to independent GEMMs (overlap microbenchmark,
    # latency-hiding scheduler on): op -> [(per-GPU input bytes, exposed fraction), ...] sorted by size.
    exposed_frac: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

    def exposed(self, op: str, msg_bytes: float) -> float:
        pts = self.exposed_frac.get(op)
        if not pts:
            pts = sorted(p for v in self.exposed_frac.values() for p in v)
        if not pts or msg_bytes <= 0:
            return 0.0
        if msg_bytes <= pts[0][0]:
            x = pts[0][1]
        elif msg_bytes >= pts[-1][0]:
            x = pts[-1][1]
        else:
            x = next(x0 + math.log(msg_bytes / s0) / math.log(s1 / s0) * (x1 - x0)
                     for (s0, x0), (s1, x1) in zip(pts, pts[1:]) if s0 <= msg_bytes <= s1)
        return min(1.0, max(0.0, x))

    def exposed_allreduce_s(self, msg_bytes: float) -> float:
        c = self.overlap_curve
        if not c or msg_bytes <= 0:
            return 0.0
        if msg_bytes <= c[0][0]:
            return max(0.0, c[0][1]) * msg_bytes / c[0][0]
        if msg_bytes >= c[-1][0]:
            return max(0.0, c[-1][1]) * msg_bytes / c[-1][0]
        for (s0, e0), (s1, e1) in zip(c, c[1:]):
            if s0 <= msg_bytes <= s1:
                f = math.log(msg_bytes / s0) / math.log(s1 / s0)
                return max(0.0, e0 + f * (e1 - e0))
        return 0.0

    def calibrated(self, op: str) -> bool:
        return op in self.busbw_curve or op in self.busbw_gbs

    def bw(self, op: str, nic_share: float = 1.0, msg_bytes: Optional[float] = None) -> float:
        curve = self.busbw_curve.get(op)
        if curve and msg_bytes:
            if msg_bytes <= curve[0][0]:
                gbs = curve[0][1] * msg_bytes / curve[0][0]  # latency-bound below the smallest size
            elif msg_bytes >= curve[-1][0]:
                gbs = curve[-1][1]
            else:
                for (s0, b0), (s1, b1) in zip(curve, curve[1:]):
                    if s0 <= msg_bytes <= s1:
                        f = math.log(msg_bytes / s0) / math.log(s1 / s0)
                        gbs = b0 + f * (b1 - b0)
                        break
            return max(gbs, 1e-3) * 1e9 * nic_share
        if op in self.busbw_gbs:
            return self.busbw_gbs[op] * 1e9 * nic_share
        return self.nic_gbps / 8 * 1e9 * self.efficiency * nic_share

    def latency_s(self, op: str, steps: int) -> float:
        # A measured curve already includes per-step latency.
        return 0.0 if op in self.busbw_curve else steps * self.latency_us * 1e-6


def load_overlap_calibration(path: str, fabric: Fabric) -> Fabric:
    """Fill fabric.exposed_frac from an overlap microbenchmark log (``[ovl-bench] {...}`` lines):
    per collective, the mean exposed fraction over the measured sizes and compute ratios."""
    vals: Dict[str, Dict[float, List[float]]] = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if "[ovl-bench] " not in line:
                continue
            try:
                d = json.loads(line.split("[ovl-bench] ", 1)[1].strip())
            except json.JSONDecodeError:
                continue
            if d.get("op") in ("allgather", "reducescatter") and d.get("comm_s", 0) < d.get("gemm_s", 0):
                vals.setdefault(d["op"], {}).setdefault(float(d["mb_per_gpu"]) * 2**20, []).append(float(d["exposed_frac"]))
    # One value per collective: the mean over the whole sweep, each point clipped to [0, 1]. Per-size values
    # do not transfer, because XLA does not issue training's gathers at the sizes the layer model assumes.
    fabric.exposed_frac = {}
    for op, by in vals.items():
        xs = [min(1.0, max(0.0, x)) for v in by.values() for x in v]
        fabric.exposed_frac[op] = [(1.0, sum(xs) / len(xs))]
    return fabric


def load_fabric_calibration(path: str, fabric: Fabric, nodes: Optional[int] = None) -> Fabric:
    """Fill *fabric* from a DCN microbenchmark log or JSON (``[dcn-bench] {...}`` lines).

    Uses the bus bandwidth of the largest message measured for each collective.
    """
    best: Dict[str, Tuple[float, float]] = {}
    curve: Dict[str, Dict[float, float]] = {}
    ov: Dict[float, float] = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if "[dcn-bench] " in line:
                line = line.split("[dcn-bench] ", 1)[1]
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:  # a line interleaved with other ranks' output
                continue
            if d.get("op") == "overlap" and (not nodes or d.get("nodes") == nodes):
                ov[float(d["mb_per_gpu"]) * 2**20] = float(d["exposed_s"])
                continue
            if d.get("op") == "fabric_health" and d.get("out_of_sequence", 0) > 0:
                _log(f"WARNING: {d['out_of_sequence']} out-of-sequence RoCE packets during the "
                     f"{d.get('nodes')}-node calibration; the scale-out traffic is on a lossy traffic "
                     "class and multi-node runs will be slower than projected (set NCCL_IB_TC to the "
                     "fabric's lossless class)")
                continue
            if "busbw_gbs" not in d or (nodes and d.get("nodes") != nodes):
                continue
            op = {"allreduce_dcn": "allreduce", "allgather_dcn": "allgather",
                  "reducescatter_dcn": "reducescatter", "permute_dcn": "permute"}.get(d["op"])
            if not op:
                continue
            size = float(d.get("mb_per_gpu", d.get("mb_per_gpu_in", 0)))
            if op not in best or size > best[op][0]:
                best[op] = (size, float(d["busbw_gbs"]))
            curve.setdefault(op, {})[size * 2**20] = float(d["busbw_gbs"])
    fabric.busbw_gbs.update({op: v for op, (_, v) in best.items()})
    fabric.busbw_curve.update({op: sorted(c.items()) for op, c in curve.items()})
    if ov:
        fabric.overlap_curve = sorted(ov.items())
    return fabric


def ring_allreduce_s(bytes_per_gpu: float, peers: int, fabric: Fabric, nic_share: float = 1.0) -> float:
    if peers <= 1 or bytes_per_gpu <= 0:
        return 0.0
    bw = fabric.bw("allreduce", nic_share, bytes_per_gpu)
    return 2 * (peers - 1) / peers * bytes_per_gpu / bw + fabric.latency_s("allreduce", 2 * (peers - 1))


# RCCL's defaults on this fabric (NCCL_DEBUG=INFO, RCCL 2.31): a ring with one GPU per node gets
# 32 channels on 2 nodes, 8 on 4 and 4 otherwise; a ring that also spans GPUs inside a node gets 64.
RCCL_NET_CHANNELS = {2: 32, 4: 8}
RCCL_NET_CHANNELS_OTHER = 4
RCCL_MIXED_CHANNELS = 64
RCCL_MAX_CHANNELS = 64


def net_channels(nodes: int = 2, ranks_per_node: int = 1, env: Optional[Dict[str, str]] = None) -> int:
    """Channels RCCL gives a cross-node collective: its default, clamped by NCCL_MIN/MAX_NCHANNELS if set."""
    env = os.environ if env is None else env
    ch = RCCL_MIXED_CHANNELS if ranks_per_node > 1 else RCCL_NET_CHANNELS.get(nodes, RCCL_NET_CHANNELS_OTHER)
    if env.get("NCCL_MAX_NCHANNELS"):
        ch = min(ch, int(env["NCCL_MAX_NCHANNELS"]))
    if env.get("NCCL_MIN_NCHANNELS"):
        ch = max(ch, int(env["NCCL_MIN_NCHANNELS"]))
    return max(1, min(ch, RCCL_MAX_CHANNELS))


def gpu_compute_units(default: int = 304) -> int:
    """Compute units of GPU 0 from rocminfo; *default* (MI300X/MI325X) when it is unavailable."""
    import subprocess
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return default
    cus = [int(x) for x in re.findall(r"Compute Unit:\s+(\d+)", out)]
    gpu = [c for c in cus if c > 64]  # CPU agents list their cores here too
    return gpu[0] if gpu else default


def ring_gather_s(bytes_full: float, peers: int, fabric: Fabric, nic_share: float = 1.0, op: str = "allgather") -> float:
    """All-gather or reduce-scatter of a tensor whose gathered size is *bytes_full*."""
    if peers <= 1 or bytes_full <= 0:
        return 0.0
    # Message size as the microbenchmark defines it: the input buffer per GPU.
    msg = bytes_full / peers if op == "allgather" else bytes_full
    bw = fabric.bw(op, nic_share, msg)
    return (peers - 1) / peers * bytes_full / bw + fabric.latency_s(op, peers - 1)


@dataclass
class PipelinePlan:
    stages: int
    repeats: int
    microbatches: int
    layers_per_stage: int
    delay: bool  # pipeline_delay_activation_forwarding

    @property
    def iterations(self) -> int:
        return self.repeats * self.microbatches + (2 if self.delay else 1) * (self.stages - 1)


def pipeline_plan(params: Dict[str, object], mesh: MeshSpec, num_layers: int) -> Optional[PipelinePlan]:
    """MaxText's circular pipeline over the pipeline mesh axis, with its defaults filled."""
    p = mesh.ici.get("pipeline", 1) * mesh.dcn.get("pipeline", 1)
    if p <= 1:
        return None
    delay = str(params.get("pipeline_delay_activation_forwarding", "false")).lower() == "true"
    m = int(params.get("num_pipeline_microbatches", -1))
    if m <= 0:
        m = 2 * p if delay else p
    lps = int(params.get("num_layers_per_pipeline_stage", 1))
    r = int(params.get("num_pipeline_repeats", -1))
    if r <= 0:
        r = num_layers // (p * lps)
    if p * lps * r != num_layers:
        raise ValueError(f"{num_layers} layers do not split into {p} stages x {lps} layers x {r} repeats")
    return PipelinePlan(stages=p, repeats=r, microbatches=m, layers_per_stage=lps, delay=delay)


def pipeline_microbatch(params: Dict[str, object], plan: PipelinePlan) -> float:
    """Rows each GPU processes per pipeline iteration: the microbatch spread over a stage's GPUs."""
    return float(params.get("per_device_batch_size", 1)) * plan.stages / plan.microbatches


def simulate_pipeline_s(plan: PipelinePlan, fwd_s: float, bwd_s: float) -> float:
    """Step time of the pipelined layers from the shared pipeline-schedule simulator.

    MaxText runs the stages in SPMD lockstep, so every stage executes every iteration (bubble
    iterations included) and a stage's chunk time is the same on every rank.
    """
    try:
        import importlib.util

        from primus.core.pipeline_parallel.scheduler.algorithms.circular import ScheduleCircular

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulator.py")
        spec = importlib.util.spec_from_file_location("_pp_simulator", path)
        sim = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sim)
    except Exception as e:  # noqa: BLE001
        _log(f"pipeline simulator unavailable ({e}); using the closed form")
        return plan.iterations * (fwd_s + bwd_s)
    sched = ScheduleCircular(plan.stages, plan.repeats, plan.microbatches)
    chunk = {"fwd": fwd_s * 1e3, "bwd": bwd_s * 1e3, "wgrad": 0.0}
    runner = sim.SchedulerSimulationRunner({"chunk_time_ms": [[chunk] * plan.repeats for _ in range(plan.stages)]})
    res = runner.simulate_scheduler_table(sched.generate_schedule_table(),
                                          {"pp_size": plan.stages, "vpp_size": plan.repeats,
                                           "micro_batches": plan.microbatches})
    step = max(r["total"] for r in res) / 1e3
    if plan.delay:
        # Delayed forwarding adds stages-1 iterations to each loop; the simulator has no such option.
        step += (plan.stages - 1) * (fwd_s + bwd_s)
    return step


_PIPELINE_KEYS = ("dcn_pipeline_parallelism", "ici_pipeline_parallelism", "num_pipeline_microbatches",
                  "num_pipeline_repeats", "num_layers_per_pipeline_stage", "pipeline_parallel_layers",
                  "pipeline_delay_activation_forwarding", "pipeline_fsdp_ag_once", "pipeline_fsdp_ag_per_repeat")


def bench_overrides(overrides: List[str], params: Dict[str, object], num_layers: int) -> List[str]:
    """Overrides for the single-node bench: no pipeline axis, and the per-GPU batch of one pipeline
    iteration when the target is pipelined (the bench measures the layer a stage runs)."""
    mesh = MeshSpec(ici={"pipeline": int(params.get("ici_pipeline_parallelism", 1))},
                    dcn={"pipeline": int(params.get("dcn_pipeline_parallelism", 1))})
    plan = pipeline_plan(params, mesh, num_layers)
    if not plan:
        return list(overrides)
    out, i = [], 0
    while i < len(overrides):
        tok = overrides[i]
        key = tok[2:].split("=", 1)[0] if tok.startswith("--") else None
        n = 1 if (key is None or "=" in tok or i + 1 >= len(overrides)) else 2
        if key not in _PIPELINE_KEYS and key != "per_device_batch_size":
            out += overrides[i:i + n]
        i += n
    return out + ["--dcn_pipeline_parallelism", "1", "--ici_pipeline_parallelism", "1",
                  "--per_device_batch_size", f"{pipeline_microbatch(params, plan):g}"]


def dcn_collectives(spec, params: Dict[str, object], bench_mesh: MeshSpec, target_mesh: MeshSpec,
                    fabric: Fabric, gpus_per_node: int) -> Dict[str, float]:
    """Seconds of scale-out traffic per step, split into per-layer and per-step parts."""
    counts = model_param_counts(spec)
    weight_bytes = 2 if str(params.get("weight_dtype", "float32")).startswith("bfloat16") else 4
    # XLA reduces gradients across nodes in the weight dtype (the in-loop all-reduce of a scanned
    # layer is bf16 when weights are bf16); grad_dtype only sets the width of the stored gradients.
    grad_bytes = weight_bytes
    grad_mem_bytes = 2 if str(params.get("grad_dtype", "float32")).startswith("bfloat16") else 4
    ici_param_shard = bench_mesh.ici_product(("fsdp", "fsdp_transpose", "tensor", "tensor_transpose", "tensor_sequence"))
    ici_expert_shard = ici_param_shard * bench_mesh.ici_product(("expert",))
    # A GPU's NIC is shared by the GPUs it serves; with one NIC per GPU it is not.
    nic_share = min(1.0, fabric.nics_per_node / gpus_per_node)

    dense_l = counts["dense_layers"]
    moe_l = counts["moe_layers"]
    per_layer_bytes_dense = counts["per_dense_layer"] * grad_bytes / ici_param_shard
    per_layer_bytes_moe = (counts["per_moe_layer_nonexpert"] / ici_param_shard + counts["per_moe_layer_expert"] / ici_expert_shard) * grad_bytes
    embed_bytes = counts["embedding"] * grad_bytes / ici_param_shard

    out = {"grad_bytes_per_gpu": per_layer_bytes_dense * dense_l + per_layer_bytes_moe * moe_l + embed_bytes}
    out["grad_mem_bytes_per_gpu"] = out["grad_bytes_per_gpu"] / grad_bytes * grad_mem_bytes
    dp = target_mesh.dcn_product(("data",))
    fs = target_mesh.dcn_product(("fsdp", "fsdp_transpose"))
    ep = target_mesh.dcn_product(("expert",))

    # Data parallel across nodes: all-reduce of the (DCN-FSDP-sharded) gradient.
    ar_layer_dense = ring_allreduce_s(per_layer_bytes_dense / fs, dp, fabric, nic_share)
    ar_layer_moe = ring_allreduce_s(per_layer_bytes_moe / fs, dp, fabric, nic_share)
    ar_embed = ring_allreduce_s(embed_bytes / fs, dp, fabric, nic_share)

    # FSDP across nodes: weights gathered in forward and again in backward, grads reduce-scattered.
    w_scale = weight_bytes / grad_bytes
    ag_dense = ring_gather_s(per_layer_bytes_dense * w_scale, fs, fabric, nic_share)
    ag_moe = ring_gather_s(per_layer_bytes_moe * w_scale, fs, fabric, nic_share)
    rs_dense = ring_gather_s(per_layer_bytes_dense, fs, fabric, nic_share, op="reducescatter")
    rs_moe = ring_gather_s(per_layer_bytes_moe, fs, fabric, nic_share, op="reducescatter")
    fsdp_passes = 2
    if str(params.get("remat_policy", "full")) == "full":
        fsdp_passes = 3  # the rematerialized forward gathers the weights again

    out["ar_msg_dense_layer"] = per_layer_bytes_dense / fs if dp > 1 else 0.0
    out["ar_msg_moe_layer"] = per_layer_bytes_moe / fs if dp > 1 else 0.0
    out["ar_msg_embed"] = embed_bytes / fs if dp > 1 else 0.0
    out["per_dense_layer_s"] = ar_layer_dense + fsdp_passes * ag_dense + rs_dense
    out["per_moe_layer_s"] = ar_layer_moe + fsdp_passes * ag_moe + rs_moe
    out["ag_msg_dense_layer"] = per_layer_bytes_dense * w_scale / fs if fs > 1 else 0.0
    out["ag_msg_moe_layer"] = per_layer_bytes_moe * w_scale / fs if fs > 1 else 0.0
    out["rs_msg_dense_layer"] = per_layer_bytes_dense if fs > 1 else 0.0
    out["rs_msg_moe_layer"] = per_layer_bytes_moe if fs > 1 else 0.0
    out["ag_dense_layer_s"], out["ag_moe_layer_s"] = fsdp_passes * ag_dense, fsdp_passes * ag_moe
    out["rs_dense_layer_s"], out["rs_moe_layer_s"] = rs_dense, rs_moe
    out["fwd_dense_layer_s"], out["fwd_moe_layer_s"] = ag_dense, ag_moe
    out["bwd_dense_layer_s"] = (fsdp_passes - 1) * ag_dense + rs_dense
    out["bwd_moe_layer_s"] = (fsdp_passes - 1) * ag_moe + rs_moe
    out["fwd_share"] = 0.25 if fsdp_passes == 3 else 1 / 3
    out["per_step_s"] = ar_embed + ring_gather_s(embed_bytes * w_scale, fs, fabric, nic_share) * 2 + ring_gather_s(embed_bytes, fs, fabric, nic_share, op="reducescatter")

    # Expert parallel across nodes: dispatch + combine, forward and backward.
    if ep > 1 and spec.num_experts:
        pdbs = float(params.get("per_device_batch_size", 1))
        seq = int(params.get("max_target_length", 2048))
        tok_bytes = pdbs * seq * spec.moe_router_topk * spec.hidden_size * 2
        a2a = (ep - 1) / ep * tok_bytes / fabric.bw("alltoall", nic_share)
        out["per_moe_layer_s"] += 4 * a2a
        out["fwd_moe_layer_s"] += 2 * a2a
        out["bwd_moe_layer_s"] += 2 * a2a
    # Parameter-proportional bytes live at the peak: weights + Adam mu/nu (weight dtype) + gradients.
    state_b = 3 * weight_bytes + grad_mem_bytes
    out["state_bytes_per_dense_layer"] = per_layer_bytes_dense / grad_bytes * state_b
    out["state_bytes_per_moe_layer"] = per_layer_bytes_moe / grad_bytes * state_b
    out["state_bytes_embed"] = embed_bytes / grad_bytes * state_b
    out["dense_layers"] = dense_l
    out["moe_layers"] = moe_l
    out["total_s"] = out["per_dense_layer_s"] * dense_l + out["per_moe_layer_s"] * moe_l + out["per_step_s"]
    return out


# ---------------------------------------------------------------------------
# Running the bench
# ---------------------------------------------------------------------------


def parse_train_log(text: str, skip: int) -> Dict[str, object]:
    steps = {}
    for m in _STEP_RE.finditer(text):
        steps[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    post = [steps[s] for s in sorted(steps) if s >= skip]
    out: Dict[str, object] = {"n_steps": len(steps)}
    if post:
        secs = sorted(p[0] for p in post)
        out["step_s"] = secs[len(secs) // 2]
        out["tflops"] = sorted(p[1] for p in post)[len(post) // 2]
        out["tps"] = sorted(p[2] for p in post)[len(post) // 2]
    m = _MEM_TOTAL_RE.findall(text)
    if m:
        out["compiled_total_gib"] = float(m[-1][0])
    used = [float(u) for u, _ in _MEMSTATS_RE.findall(text)]
    if used:
        out["memstats_gib"] = max(used)
    peak = re.findall(r"\[peak-mem\] max_peak_bytes=(\d+)", text)
    if peak:
        out["peak_gib"] = int(peak[-1]) / _GiB
    # Persistent train state: MaxText's memstats right after the params/optimizer state are created.
    i = text.find("After params initialized")
    if i >= 0:
        used = [float(u) for u, _ in _MEMSTATS_RE.findall(text[i:i + 20000])[:64]]
        if used:
            out["live_gib"] = max(used)
    return out


def run_bench_step(config: str, overrides: List[str], layers_key: str, layers: int, steps: int, log_path: str,
                   extra_env: Optional[Dict[str, str]] = None, timeout: int = 3600) -> StepSample:
    """Train *steps* steps of the experiment at *layers* decoder layers on this node."""
    primus_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    cmd = [
        os.path.join(primus_root, "runner", "primus-cli"), "direct", "--",
        "train", "pretrain", "--config", config,
        *overrides,
        # Model-config keys only take effect as initialize() kwargs (override_model.*),
        # and MaxText rejects those unless override_model_config is set.
        "--override_model_config", "true",
        f"--override_model.{layers_key}", str(layers),
        "--steps", str(steps),
        "--enable_checkpointing", "false",
    ]
    env = dict(os.environ)
    env.update(extra_env or {})
    # The bench is one node regardless of where the projection targets.
    env["NNODES"] = "1"
    env["NODE_RANK"] = "0"
    env.setdefault("PRIMUS_PEAK_MEM_HOOK", "1")
    _log(f"bench L={layers}: {' '.join(shlex.quote(c) for c in cmd)}")
    t0 = time.time()
    with open(log_path, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=primus_root, timeout=timeout)
    with open(log_path, errors="replace") as fh:
        text = fh.read()
    parsed = parse_train_log(text, skip=min(3, max(1, steps // 3)))
    if rc != 0 or "step_s" not in parsed:
        raise RuntimeError(f"bench at L={layers} failed (rc={rc}); see {log_path}")
    _log(f"bench L={layers}: step {parsed['step_s']*1e3:.1f} ms, wall {time.time()-t0:.0f}s, "
         f"peak {parsed.get('peak_gib') or parsed.get('memstats_gib')}")
    return StepSample(
        layers=layers,
        step_s=float(parsed["step_s"]),
        tokens_per_s_per_gpu=float(parsed["tps"]),
        tflops_per_gpu=float(parsed["tflops"]),
        peak_gib=parsed.get("peak_gib") or parsed.get("memstats_gib"),
        compiled_total_gib=parsed.get("compiled_total_gib"),
        n_steps=int(parsed["n_steps"]),
        log=log_path,
        live_gib=parsed.get("live_gib"),
    )


def choose_depths(spec, requested: Optional[Sequence[int]]) -> Tuple[int, int]:
    if requested:
        a, b = sorted(int(x) for x in requested)[:2]
        return a, b
    base = spec.num_dense_layers if spec.num_experts else 0
    total = spec.num_layers
    lo = base + 2
    hi = min(total, base + 6)
    if hi <= lo:
        hi = min(total, lo + 1)
    return lo, hi


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def project(spec, params: Dict[str, object], samples: Sequence[StepSample], target_nodes: Sequence[int],
            gpus_per_node: int, fabric: Fabric, overlap: str = "layer",
            anchor: Optional[Tuple[int, float]] = None, bench_pdbs: Optional[float] = None) -> List[Projection]:
    """anchor=(nodes, measured step s) replaces the DCN model by the measured excess of one
    multi-node run over the single-node projection, scaled with the ring factor 2(N-1)/N.
    bench_pdbs: per-device batch the samples were measured at (default: the target's)."""
    a, b = sorted(samples, key=lambda s: s.layers)[:2]
    t_layer = (b.step_s - a.step_s) / (b.layers - a.layers)
    t_fixed = a.step_s - t_layer * a.layers
    mem_slope = None
    if a.peak_gib and b.peak_gib:
        mem_slope = (b.peak_gib - a.peak_gib) / (b.layers - a.layers)
    L = spec.num_layers
    t_node = t_fixed + t_layer * L

    pdbs = float(params.get("per_device_batch_size", 1))
    seq = int(params.get("max_target_length", 2048))
    bench_mesh = resolve_mesh(params, 1, gpus_per_node)
    out = []
    for n in target_nodes:
        tmesh = resolve_mesh(params, n, gpus_per_node)
        fsdp_axes = ("fsdp", "fsdp_transpose")
        fsdp_total = tmesh.ici_product(fsdp_axes) * tmesh.dcn_product(fsdp_axes)
        if spec.hidden_size % fsdp_total:
            _log(f"target {n} nodes: invalid layout, hidden size {spec.hidden_size} is not divisible by "
                 f"the FSDP degree {fsdp_total}; skipped")
            continue
        comm = dcn_collectives(spec, params, bench_mesh, tmesh, fabric, gpus_per_node)
        if overlap == "none":
            exposed = comm["total_s"]
        elif overlap == "allreduce":
            # Data-parallel gradient all-reduces are issued at the end of the layer body and stay exposed;
            # FSDP/expert collectives are scheduled against the layer's compute and only the excess shows.
            dp = tmesh.dcn_product(("data",))
            share = min(1.0, fabric.nics_per_node / gpus_per_node)
            exposed = comm["per_step_s"]
            for kind, layers in (("dense", comm["dense_layers"]), ("moe", comm["moe_layers"])):
                ar = ring_allreduce_s(comm[f"ar_msg_{kind}_layer"], dp, fabric, share) if dp > 1 else 0.0
                rest = comm[f"per_{kind}_layer_s"] - ar
                exposed += (ar + max(0.0, rest - t_layer)) * layers
        elif overlap == "phase":
            # As "allreduce", but forward traffic hides only behind the forward share of the layer
            # and backward traffic only behind the backward share.
            dp = tmesh.dcn_product(("data",))
            share = min(1.0, fabric.nics_per_node / gpus_per_node)
            f = comm["fwd_share"]
            exposed = comm["per_step_s"]
            for kind, layers in (("dense", comm["dense_layers"]), ("moe", comm["moe_layers"])):
                ar = ring_allreduce_s(comm[f"ar_msg_{kind}_layer"], dp, fabric, share) if dp > 1 else 0.0
                exposed += (ar + max(0.0, comm[f"fwd_{kind}_layer_s"] - f * t_layer)
                            + max(0.0, comm[f"bwd_{kind}_layer_s"] - (1 - f) * t_layer)) * layers
        elif overlap == "calibrated" and fabric.overlap_curve:
            # Data-parallel all-reduces: the exposed time measured with concurrent GEMMs.
            # FSDP/expert collectives (if any) still use the per-layer hiding model.
            ex = fabric.exposed_allreduce_s
            ar_dense = comm["ar_msg_dense_layer"]
            ar_moe = comm["ar_msg_moe_layer"]
            dp = tmesh.dcn_product(("data",))
            share = min(1.0, fabric.nics_per_node / gpus_per_node)
            rest_dense = comm["per_dense_layer_s"] - ring_allreduce_s(ar_dense, dp, fabric, share)
            rest_moe = comm["per_moe_layer_s"] - ring_allreduce_s(ar_moe, dp, fabric, share)
            exposed = ((ex(ar_dense) + max(0.0, rest_dense - t_layer)) * comm["dense_layers"]
                       + (ex(ar_moe) + max(0.0, rest_moe - t_layer)) * comm["moe_layers"]
                       + comm["per_step_s"])  # embedding/head traffic sits outside the loop: fully exposed
        else:
            # Each layer's scale-out traffic hides behind that layer's own compute;
            # only the excess shows. The per-step part (embedding, head) and the
            # last layer's gradient reduction have nothing left to hide behind.
            per_layer_moe = comm["per_moe_layer_s"]
            per_layer_dense = comm["per_dense_layer_s"]
            exposed_layers = (max(0.0, per_layer_dense - t_layer) * comm["dense_layers"]
                              + max(0.0, per_layer_moe - t_layer) * comm["moe_layers"])
            tail = max(per_layer_moe if comm["moe_layers"] else 0.0, per_layer_dense if comm["dense_layers"] else 0.0)
            exposed = exposed_layers + comm["per_step_s"] + tail
        if overlap == "measured" and fabric.exposed_frac:
            # As "allreduce", but of the FSDP/expert traffic that fits behind the layer, the fraction the
            # overlap microbenchmark finds exposed at this node count still shows.
            dp = tmesh.dcn_product(("data",))
            share = min(1.0, fabric.nics_per_node / gpus_per_node)
            exposed = comm["per_step_s"]
            for kind, layers in (("dense", comm["dense_layers"]), ("moe", comm["moe_layers"])):
                if not layers:
                    continue
                ar = ring_allreduce_s(comm[f"ar_msg_{kind}_layer"], dp, fabric, share) if dp > 1 else 0.0
                rest = comm[f"per_{kind}_layer_s"] - ar
                if rest <= 0:
                    exposed += ar * layers
                    continue
                hidden = min(rest, t_layer)
                ag, rs = comm[f"ag_{kind}_layer_s"], comm[f"rs_{kind}_layer_s"]
                x_ag = fabric.exposed("allgather", comm[f"ag_msg_{kind}_layer"])
                x_rs = fabric.exposed("reducescatter", comm[f"rs_msg_{kind}_layer"])
                other = max(0.0, rest - ag - rs)  # expert all-to-all: no curve of its own
                x = (ag * x_ag + rs * x_rs + other * max(x_ag, x_rs)) / rest
                exposed += (ar + max(0.0, rest - t_layer) + x * hidden) * layers
        if overlap == "contention":
            # Hidden traffic still holds net_channels of the GPU's compute units, so the compute
            # it overlaps runs that much slower.
            dp = tmesh.dcn_product(("data",))
            share = min(1.0, fabric.nics_per_node / gpus_per_node)
            exposed = comm["per_step_s"]
            for kind, layers in (("dense", comm["dense_layers"]), ("moe", comm["moe_layers"])):
                ar = ring_allreduce_s(comm[f"ar_msg_{kind}_layer"], dp, fabric, share) if dp > 1 else 0.0
                rest = comm[f"per_{kind}_layer_s"] - ar
                exposed += (ar + max(0.0, rest - t_layer)) * layers
            axes = fsdp_axes if tmesh.dcn_product(fsdp_axes) > 1 else ("data",)
            ch = net_channels(n, tmesh.ici_product(axes))
            exposed += (comm["total_s"] - exposed) * ch / fabric.cus_per_gpu
        if anchor and n > 1:
            na, step_a = anchor
            exposed = (step_a - t_node) * (n - 1) / n * na / (na - 1)
        step = t_node + exposed
        plan = pipeline_plan(params, tmesh, L)
        pp_detail = {}
        if plan:
            # Each GPU runs every pipeline iteration on a microbatch of b_mb rows through the
            # layers of its stage; the stage shift sends that microbatch's activations to the
            # next stage after the forward and its gradient back after the backward.
            b_mb = pipeline_microbatch(params, plan)
            scale = b_mb / (bench_pdbs or pdbs)
            t_lay_mb = t_layer * scale
            f = comm["fwd_share"]
            act_bytes = b_mb * seq * spec.hidden_size * 2
            p2p = 0.0
            if tmesh.dcn.get("pipeline", 1) > 1:
                share = min(1.0, fabric.nics_per_node / gpus_per_node)
                op = "permute" if fabric.calibrated("permute") else "allgather"
                p2p = act_bytes / fabric.bw(op, share, act_bytes) + fabric.latency_s(op, 1)
            t_f, t_b = plan.layers_per_stage * t_lay_mb * f, plan.layers_per_stage * t_lay_mb * (1 - f)
            if plan.delay:
                ex_f, ex_b = max(0.0, p2p - t_f), max(0.0, p2p - t_b)
            else:
                ex_f = ex_b = p2p
            t_pipe = simulate_pipeline_s(plan, t_f + ex_f, t_b + ex_b)
            # Embedding and head run outside the pipeline on the full per-device batch; a stage
            # holds 1/stages of the layers, so its other scale-out traffic shrinks with it.
            other = comm["per_step_s"] + (exposed - comm["per_step_s"]) / plan.stages
            step = t_fixed * pdbs / (bench_pdbs or pdbs) + t_pipe + other
            exposed = step - t_fixed - L * t_layer
            pp_detail = {"pp_stages": plan.stages, "pp_repeats": plan.repeats, "pp_microbatches": plan.microbatches,
                         "pp_layers_per_stage": plan.layers_per_stage, "pp_iterations": plan.iterations,
                         "pp_microbatch_rows": b_mb, "pp_p2p_s": p2p, "pp_pipe_s": t_pipe,
                         "pp_bubble_s": t_pipe - plan.repeats * plan.microbatches * (t_f + t_b + ex_f + ex_b),
                         "pp_p2p_exposed_s": plan.iterations * (ex_f + ex_b)}
        peak = None
        if mem_slope is not None and not plan:
            peak = a.peak_gib + mem_slope * (L - a.layers)
            # DCN FSDP shards the persistent state further; saved activations and
            # transients are unchanged. DCN data parallelism changes neither.
            fs = tmesh.dcn_product(("fsdp", "fsdp_transpose"))
            if fs > 1:
                if a.live_gib and b.live_gib:
                    # Measured persistent state (weights + optimizer state, live right after init)
                    # plus the gradients, which are live at the peak and sharded the same way.
                    state = a.live_gib + (b.live_gib - a.live_gib) / (b.layers - a.layers) * (L - a.layers)
                    state += comm["grad_mem_bytes_per_gpu"] / _GiB
                else:
                    state = (comm["state_bytes_per_dense_layer"] * comm["dense_layers"]
                             + comm["state_bytes_per_moe_layer"] * comm["moe_layers"]
                             + comm["state_bytes_embed"]) / _GiB
                peak -= state * (1 - 1 / fs)
        out.append(Projection(
            target_nodes=n, layers=L, step_s=step, t_fixed_s=t_fixed, t_layer_s=t_layer,
            dcn_comm_s=comm["total_s"], dcn_exposed_s=exposed,
            tokens_per_s_per_gpu=pdbs * seq / step, peak_gib=peak,
            detail={**{k: v for k, v in comm.items() if isinstance(v, (int, float))}, **pp_detail},
        ))
    return out


def _layers_key(params: Dict[str, object]) -> str:
    return "base_num_decoder_layers"


def launch(args, overrides: List[str]) -> List[Projection]:
    """Entry point: bench on this node at two depths, then project to the targets."""
    from primus.core.projection.frameworks.jax import resolve_model_spec
    from primus.core.utils import yaml_utils
    from types import SimpleNamespace

    exp = yaml_utils.parse_yaml_to_namespace(args.config)
    module = exp.modules.pre_trainer
    params = dict(vars(module.overrides)) if hasattr(module, "overrides") else {}
    import primus as _primus_pkg

    model_yaml = os.path.join(os.path.dirname(_primus_pkg.__file__), "configs", "models", "maxtext",
                              getattr(module, "model", "") or "")
    # Precedence as in training: experiment overrides > model yaml > module defaults (e.g. weight_dtype).
    module_yaml = os.path.join(os.path.dirname(_primus_pkg.__file__), "configs", "modules", "maxtext",
                               getattr(module, "config", "") or "")
    for path in (model_yaml, module_yaml):
        if os.path.isfile(path):
            for k, v in vars(yaml_utils.parse_yaml_to_namespace(path)).items():
                if k != "extends" and not isinstance(v, SimpleNamespace):
                    params.setdefault(k, v)
    # CLI overrides win over the YAML, as they do in training.
    cli = {}
    i = 0
    while i < len(overrides):
        tok = overrides[i]
        if tok.startswith("--") and "=" in tok:
            k, v = tok[2:].split("=", 1)
            cli[k] = v
            i += 1
        elif tok.startswith("--") and i + 1 < len(overrides):
            cli[tok[2:]] = overrides[i + 1]
            i += 2
        else:
            i += 1
    params.update(cli)
    model_name = str(params.get("model_name") or "")
    spec = resolve_model_spec(SimpleNamespace(**params))

    gpus = int(os.getenv("GPUS_PER_NODE", "8"))
    depths = choose_depths(spec, getattr(args, "bench_depths", None))
    steps = int(getattr(args, "bench_steps", None) or 8)
    fabric = Fabric(
        nic_gbps=float(os.getenv("PRIMUS_NIC_GBPS", "400")),
        nics_per_node=int(os.getenv("PRIMUS_NICS_PER_NODE", "8")),
        efficiency=float(os.getenv("PRIMUS_NIC_EFF", "0.70")),
        cus_per_gpu=int(os.getenv("PRIMUS_GPU_CUS") or gpu_compute_units()),
    )
    calib = getattr(args, "fabric_calibration", None) or os.getenv("PRIMUS_FABRIC_CALIBRATION")
    if calib:
        load_fabric_calibration(calib, fabric)
        _log(f"fabric calibration from {calib}: {fabric.busbw_gbs}")
    ovl = os.getenv("PRIMUS_OVERLAP_CALIBRATION")
    if ovl and os.path.isfile(ovl):
        load_overlap_calibration(ovl, fabric)
        _log(f"overlap calibration from {ovl}: {fabric.exposed_frac}")
    outdir = getattr(args, "bench_dir", None) or os.path.join("output", "compiled_step_bench")
    os.makedirs(outdir, exist_ok=True)

    load = getattr(args, "load_benchmark", None)
    bench_pdbs = None
    if load:
        with open(load) as fh:
            art = json.load(fh)
        samples = [StepSample(**s) for s in art["samples"]]
        if art.get("bench_pdbs") or art.get("params", {}).get("per_device_batch_size"):
            bench_pdbs = float(art.get("bench_pdbs") or art["params"]["per_device_batch_size"])
        for smp in samples:
            if not smp.live_gib and smp.log and os.path.isfile(smp.log):
                with open(smp.log, errors="replace") as fh:
                    smp.live_gib = parse_train_log(fh.read(), skip=3).get("live_gib")
    else:
        samples = []
        bench_ov = bench_overrides(list(overrides), params, spec.num_layers)
        if bench_ov != list(overrides):
            bench_pdbs = float(bench_ov[bench_ov.index("--per_device_batch_size") + 1])
            _log(f"pipelined target: bench without the pipeline axis at per_device_batch_size={bench_pdbs:g}")
        for L in depths:
            log_path = os.path.join(outdir, f"bench_L{L}.log")
            samples.append(run_bench_step(args.config, bench_ov, _layers_key(params), L, steps, log_path))
    targets = [int(t) for t in (getattr(args, "target_nodes_list", None) or [getattr(args, "target_nodes", None) or 1])]
    anchor = None
    if getattr(args, "dcn_anchor", None):
        na, st = str(args.dcn_anchor).split(":")
        anchor = (int(na), float(st))
    projections = project(spec, params, samples, targets, gpus, fabric, overlap=getattr(args, "dcn_overlap", "allreduce"),
                          anchor=anchor, bench_pdbs=bench_pdbs)

    _log("=" * 90)
    _log(f"model={model_name} layers={spec.num_layers} pdbs={params.get('per_device_batch_size')} "
         f"seq={params.get('max_target_length')} remat={params.get('remat_policy')}")
    for s in samples:
        _log(f"  bench L={s.layers:3d}: step {s.step_s*1e3:9.1f} ms  peak {s.peak_gib}")
    for p in projections:
        _log(f"  target {p.target_nodes} node(s): step {p.step_s*1e3:9.1f} ms "
             f"(fixed {p.t_fixed_s*1e3:.1f} + {p.layers} x {p.t_layer_s*1e3:.2f} ms/layer "
             f"+ DCN exposed {p.dcn_exposed_s*1e3:.1f} of {p.dcn_comm_s*1e3:.1f} ms) "
             f"-> {p.tokens_per_s_per_gpu:,.1f} tokens/s/GPU, peak {p.peak_gib}")
        print(f"[compiled-step-projection] {json.dumps(asdict(p))}", flush=True)

    save = getattr(args, "save_benchmark", None) or getattr(args, "save_profiling", None)
    if save:
        with open(save, "w") as fh:
            json.dump({"samples": [asdict(s) for s in samples], "params": {k: str(v) for k, v in params.items()},
                       "bench_pdbs": bench_pdbs,
                       "projections": [asdict(p) for p in projections]}, fh, indent=1)
    return projections
