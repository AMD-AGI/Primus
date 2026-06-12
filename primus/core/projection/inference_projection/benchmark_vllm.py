"""vLLM inference benchmark backend for the Primus inference projector.

Unlike the Megatron benchmark worker (which times *training* transformer layers
forward-only), this measures the **real inference engine**: it loads the model
in vLLM with dummy weights at the target precision / tensor-parallel size and
times the full-model prefill and steady-state decode *step* latency using
vLLM's optimized kernels (paged attention, fused MoE, CUDA graphs, FP8/MXFP4).

The result JSON uses the whole-model schema consumed by
``InferencePerformanceProjector.set_benchmark_calibration``::

    {"backend": "vllm",
     "measured": {"model": {"prefill_ms": <float>, "decode_ms": <float>}},
     "meta": {"batch", "input_len", "decode_steps", "tp", "quantization", "model"}}

``prefill_ms`` is the prompt-processing step for ``batch`` sequences of
``input_len`` tokens; ``decode_ms`` is one steady-state decode step for the same
batch (measured by subtracting a 1-token run from a K-token run so prefill and
fixed overheads cancel).

This module is intentionally dependency-light (only ``vllm`` + stdlib) so it can
run inside a vLLM container that does not have Primus installed::

    python3 benchmark_vllm.py --model openai/gpt-oss-120b --tp 1 \
        --input-len 1024 --batch 16 --decode-steps 32 --save out.json
"""

from __future__ import annotations

import argparse
import json
import os
import time


_ZIPF_MARKER = "PRIMUS_ZIPF_ROUTING"

_ZIPF_PATCH = '''

# === {marker} (appended by Primus benchmark; idempotent) ===
import os as _primus_os
import torch as _primus_torch


class _PrimusZipfRouting(RoutingStrategy):
    """Zipfian token->expert routing for a realistic MoE benchmark load.

    Expert popularity follows p(rank) ~ 1/rank**s over a fixed random ranking,
    so a few hot experts receive most tokens and, at small decode batch, only a
    subset of experts are triggered. This reflects production routing far better
    than uniform-random selection on dummy weights, where (a) every expert tends
    to get ~equal load and (b) the grouped/sorted MoE GEMM problem sizes are
    unrepresentative. ``s`` (PRIMUS_ZIPF_S) controls the skew (s=0 -> uniform).
    """

    def __init__(self, s: float = 1.0, seed: int = 1234):
        self.s = float(s)
        self.seed = int(seed)
        self._cache = {{}}

    def _probs(self, num_experts, device):
        key = (num_experts, str(device))
        p = self._cache.get(key)
        if p is None:
            g = _primus_torch.Generator().manual_seed(self.seed)
            perm = _primus_torch.randperm(num_experts, generator=g)
            ranks = _primus_torch.empty(num_experts, dtype=_primus_torch.double)
            ranks[perm] = _primus_torch.arange(1, num_experts + 1, dtype=_primus_torch.double)
            w = 1.0 / ranks.pow(self.s)
            p = (w / w.sum()).to(device=device, dtype=_primus_torch.float32)
            self._cache[key] = p
        return p

    def route_tokens(self, hidden_states, router_logits, top_k, indices_type=None):
        num_tokens = hidden_states.shape[0]
        num_experts = router_logits.shape[-1]
        if indices_type is None:
            indices_type = _primus_torch.long
        probs = self._probs(num_experts, hidden_states.device)
        p = probs.unsqueeze(0).expand(num_tokens, -1).contiguous()
        topk_ids = _primus_torch.multinomial(p, top_k, replacement=False).to(indices_type)
        topk_weights = _primus_torch.full(
            (num_tokens, top_k), 1.0 / top_k,
            device=hidden_states.device, dtype=_primus_torch.float32,
        )
        return topk_weights, topk_ids


RoutingSimulator.register_strategy(
    "zipf", _PrimusZipfRouting(s=float(_primus_os.environ.get("PRIMUS_ZIPF_S", "1.0")))
)
# === END {marker} ===
'''.format(marker=_ZIPF_MARKER)


def _install_zipf_routing(zipf_s: float) -> bool:
    """Register a Zipfian routing strategy in vLLM's routing simulator.

    vLLM selects experts via a class-level strategy registry keyed by the
    ``VLLM_MOE_ROUTING_SIMULATION_STRATEGY`` env var (built-ins: uniform/normal).
    Because the engine spawns worker subprocesses that re-import modules, we
    append the ``zipf`` strategy to the simulator module *on disk* (idempotent)
    so every process registers it on import. Returns True on success.
    """
    import importlib.util

    os.environ["PRIMUS_ZIPF_S"] = str(zipf_s)
    spec = importlib.util.find_spec(
        "vllm.model_executor.layers.fused_moe.router.routing_simulator_router"
    )
    if spec is None or not spec.origin:
        return False
    path = spec.origin
    with open(path, "r") as f:
        src = f.read()
    if _ZIPF_MARKER not in src:
        with open(path, "a") as f:
            f.write(_ZIPF_PATCH)
    os.environ["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = "zipf"
    return True


def _enable_aiter() -> None:
    """Turn on AMD AITER kernels for production-representative inference perf.

    ``VLLM_ROCM_USE_AITER`` is the *master* switch on ROCm: when it is False
    (the vLLM default), every per-component AITER flag is gated off and vLLM
    falls back to the generic Triton fused-MoE / MLA kernels, which on MI3xx
    have no tuned configs (see the "Using default MoE config. Performance might
    be sub-optimal!" warning) and disable full-graph decode capture for MLA.
    That alone accounted for the bulk of the gap vs InferenceX (Triton MXFP4
    MoE also falls off a perf cliff at batch>=32). Set it before importing vllm
    so the spawned engine workers inherit it.
    """
    os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")


def _measure(llm, prompts, out_len: int, reps: int) -> float:
    """Best (min) wall time over ``reps`` runs of generating ``out_len`` tokens."""
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=out_len, ignore_eos=True, temperature=0.0)
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        best = min(best, time.perf_counter() - t)
    return best


def _measure_batch(llm, input_len: int, batch: int, decode_steps: int) -> dict:
    """Measure whole-model prefill + steady-state decode step latency at ``batch``.

    Subtracting a 1-token run from a K-token run cancels prefill and fixed
    overheads, isolating one decode step for the full batch.
    """
    token_ids = [(i % 100) + 1 for i in range(input_len)]
    prompts = [{"prompt_token_ids": list(token_ids)} for _ in range(batch)]

    _measure(llm, prompts, out_len=4, reps=1)  # warmup + CUDA-graph capture
    lat1 = _measure(llm, prompts, out_len=1, reps=3)
    latK = _measure(llm, prompts, out_len=decode_steps, reps=2)

    decode_ms = max(1e-6, (latK - lat1) / max(1, decode_steps - 1) * 1000.0)
    prefill_ms = max(1e-3, lat1 * 1000.0 - decode_ms)
    return {"batch": batch, "prefill_ms": prefill_ms, "decode_ms": decode_ms}


def run_vllm_benchmark(args) -> dict:
    if not args.no_aiter:
        _enable_aiter()

    routing = str(getattr(args, "routing_dist", "zipf") or "none").lower()
    routing_applied = "none"
    if routing == "zipf":
        if _install_zipf_routing(getattr(args, "zipf_s", 1.0)):
            routing_applied = f"zipf(s={getattr(args, 'zipf_s', 1.0)})"
            print(f"[Primus:Inference:vLLM-Benchmark] MoE routing = {routing_applied}")
    elif routing in ("uniform", "uniform_random"):
        os.environ["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = "uniform_random"
        routing_applied = "uniform_random"
    elif routing in ("normal", "normal_routing"):
        os.environ["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = "normal_routing"
        routing_applied = "normal_routing"

    from vllm import LLM

    batches = (
        [int(b) for b in str(args.batches).split(",") if b.strip()]
        if args.batches
        else [args.batch]
    )
    max_batch = max(batches)
    max_len = args.max_model_len or (args.input_len + args.decode_steps + 16)
    kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        load_format="dummy",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=max_len,
        max_num_seqs=max(256, max_batch),
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
    )
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.kv_cache_dtype:
        kwargs["kv_cache_dtype"] = args.kv_cache_dtype

    llm = LLM(**kwargs)  # loaded once; reused across the batch sweep

    sweep = []
    for b in batches:
        entry = _measure_batch(llm, args.input_len, b, args.decode_steps)
        sweep.append(entry)
        print(f"[Primus:Inference:vLLM-Benchmark] batch={b} "
              f"prefill={entry['prefill_ms']:.2f}ms decode_step={entry['decode_ms']:.2f}ms")

    ref = next((e for e in sweep if e["batch"] == args.batch), sweep[0])
    result = {
        "backend": "vllm",
        # ``measured.model`` is the single-batch anchor (projector compat);
        # ``sweep`` carries the per-concurrency measured curve (preferred).
        "measured": {"model": {"prefill_ms": ref["prefill_ms"], "decode_ms": ref["decode_ms"]}},
        "sweep": sweep,
        "meta": {
            "batch": ref["batch"],
            "input_len": args.input_len,
            "decode_steps": args.decode_steps,
            "tp": args.tp,
            "quantization": args.quantization,
            "kv_cache_dtype": args.kv_cache_dtype,
            "enforce_eager": args.enforce_eager,
            "use_aiter": os.environ.get("VLLM_ROCM_USE_AITER", "0") == "1",
            "moe_routing": routing_applied,
            "model": args.model,
        },
    }
    return result


def main():
    ap = argparse.ArgumentParser(description="vLLM inference benchmark backend")
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--quantization", default=None, help="e.g. fp8, mxfp4 (None=from config)")
    ap.add_argument("--kv-cache-dtype", default=None, help="e.g. fp8 (None=auto)")
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=1024, help="recorded in meta only")
    ap.add_argument("--decode-steps", type=int, default=32, help="K for decode-step timing")
    ap.add_argument("--batch", type=int, default=16, help="ref batch (anchor) when no --batches")
    ap.add_argument("--batches", default=None, help="comma list to sweep, e.g. 4,8,16,32,64")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--no-aiter", action="store_true",
                    help="disable AMD AITER kernels (default: enabled on ROCm)")
    ap.add_argument("--routing-dist", default="zipf",
                    choices=["zipf", "uniform", "normal", "none"],
                    help="MoE token->expert distribution for the benchmark "
                         "(default: zipf; 'none' uses the model's own router)")
    ap.add_argument("--zipf-s", type=float, default=1.0,
                    help="Zipfian skew exponent (0=uniform, larger=more skewed)")
    ap.add_argument("--save", required=True)
    args = ap.parse_args()

    result = run_vllm_benchmark(args)
    with open(args.save, "w") as f:
        json.dump(result, f)
    print("[Primus:Inference:vLLM-Benchmark] " + json.dumps(result))
    print(f"[Primus:Inference:vLLM-Benchmark] wrote {args.save}")


if __name__ == "__main__":
    main()
