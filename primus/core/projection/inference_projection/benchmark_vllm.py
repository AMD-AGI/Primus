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
import math
import os
import time


# --- MoE expert-load imbalance <-> Zipf exponent --------------------------------
# Production MoE routing is usually characterised by a *measurable* load-imbalance
# factor  I = max_e(tokens_e) / mean_e(tokens_e)  (1.0 = perfectly balanced; an
# all-to-one degenerate router approaches I = num_experts). We let the user pass
# that workload-meaningful number (``--moe-imbalance``) instead of a raw Zipf
# exponent ``s``: for a Zipf popularity law p(rank) ~ 1/rank**s over N experts,
# the steady-state imbalance is  I(s) = N / H_N(s)  where H_N(s) = sum 1/r**s is
# the generalised harmonic number. I(s) is monot[on]ic in s (I(0)=1, I(inf)=N),
# so we invert it by bisection. ``random`` benchmark/InferenceX-style data lands
# near a modest I (the trained router on random tokens is only mildly skewed),
# while real domain-clustered traffic pushes I higher — hence the field.


def _harmonic(num_experts: int, s: float) -> float:
    return sum(1.0 / (r ** s) for r in range(1, int(num_experts) + 1))


def _imbalance_for_s(num_experts: int, s: float) -> float:
    """Steady-state max/mean expert-load for a Zipf(s) popularity law."""
    if num_experts <= 1:
        return 1.0
    return num_experts / _harmonic(num_experts, s)


def _s_for_imbalance(num_experts: int, imbalance: float, tol: float = 1e-4) -> float:
    """Invert I(s) = N / H_N(s) for the Zipf exponent giving target imbalance."""
    if num_experts <= 1 or imbalance <= 1.0:
        return 0.0
    target = min(float(imbalance), float(num_experts) - 1e-6)
    lo, hi = 0.0, 6.0
    while _imbalance_for_s(num_experts, hi) < target and hi < 64.0:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _imbalance_for_s(num_experts, mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def _num_experts(model: str, trust_remote_code: bool) -> int:
    """Best-effort number of routed experts from the HF config (0 if unknown)."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    except Exception:
        return 0
    for attr in ("num_local_experts", "n_routed_experts", "num_experts",
                 "moe_num_experts", "num_experts_per_tok"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 1 and attr != "num_experts_per_tok":
            return v
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        for attr in ("num_local_experts", "n_routed_experts", "num_experts"):
            v = getattr(text_cfg, attr, None)
            if isinstance(v, int) and v > 1:
                return v
    return 0


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


def _measure_batch(llm, input_len: int, batch: int, decode_steps: int,
                   random_tokens: bool = False, vocab: int = 30000, seed: int = 0) -> dict:
    """Measure whole-model prefill + steady-state decode step latency at ``batch``.

    Subtracting a 1-token run from a K-token run cancels prefill and fixed
    overheads, isolating one decode step for the full batch.

    With ``random_tokens`` each sequence gets independent uniform-random token
    ids. This matters for **real-weight** runs: the trained router maps token
    content -> experts, so identical/degenerate prompts would route to a single
    expert (pathological), whereas random tokens reproduce InferenceX's
    random-data regime and let the genuine router generate the realized
    token->expert distribution per concurrency (no imposed distribution needed).
    """
    if random_tokens:
        import random as _r

        rng = _r.Random(seed)
        prompts = [
            {"prompt_token_ids": [rng.randint(1, max(2, vocab - 1)) for _ in range(input_len)]}
            for _ in range(batch)
        ]
    else:
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
    # Resolve the Zipf exponent from a measurable imbalance factor if requested.
    zipf_s = float(getattr(args, "zipf_s", 1.0))
    imbalance_target = getattr(args, "moe_imbalance", None)
    n_experts = 0
    imbalance_realized = None
    if imbalance_target is not None and float(imbalance_target) > 1.0:
        n_experts = _num_experts(args.model, args.trust_remote_code)
        if n_experts > 1:
            zipf_s = _s_for_imbalance(n_experts, float(imbalance_target))
            imbalance_realized = _imbalance_for_s(n_experts, zipf_s)
            routing = "zipf"
            print(f"[Primus:Inference:vLLM-Benchmark] MoE imbalance "
                  f"I={float(imbalance_target):.2f} -> zipf s={zipf_s:.3f} "
                  f"(N={n_experts} experts, realized I={imbalance_realized:.2f})")
        else:
            print("[Primus:Inference:vLLM-Benchmark] WARNING: could not read "
                  "expert count; falling back to --zipf-s")
    if routing == "zipf":
        if _install_zipf_routing(zipf_s):
            routing_applied = f"zipf(s={zipf_s})"
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
        load_format=args.load_format,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=max_len,
        max_num_seqs=max(256, max_batch),
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
    )
    # The benchmark drives the engine purely with ``prompt_token_ids`` (never
    # text), so the tokenizer is unnecessary. Skipping its init avoids a hard
    # dependency on sentencepiece/tiktoken for models whose fast tokenizer can't
    # be built in the container (e.g. DeepSeek-V4). Auto-on for dummy weights.
    if args.skip_tokenizer_init or args.load_format == "dummy":
        kwargs["skip_tokenizer_init"] = True
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.kv_cache_dtype:
        kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    # Sub-scale ("reduced-layer") benchmarking: override the model's layer count
    # so a huge model fits on fewer GPUs. The per-layer time is recovered by
    # benchmarking two layer counts and differencing; the full model is then
    # projected as fixed_overhead + num_layers x per_layer_time. Mirrors the
    # training sub-node bench that times a few layers and scales up.
    if args.num_hidden_layers:
        kwargs["hf_overrides"] = {"num_hidden_layers": int(args.num_hidden_layers)}

    llm = LLM(**kwargs)  # loaded once; reused across the batch sweep

    real_weights = args.load_format != "dummy"
    # Real weights => the trained router needs varied input to route realistically.
    random_tokens = args.random_tokens or real_weights
    sweep = []
    for b in batches:
        entry = _measure_batch(llm, args.input_len, b, args.decode_steps,
                               random_tokens=random_tokens, vocab=args.vocab)
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
            "num_hidden_layers": args.num_hidden_layers,
            "quantization": args.quantization,
            "kv_cache_dtype": args.kv_cache_dtype,
            "enforce_eager": args.enforce_eager,
            "use_aiter": os.environ.get("VLLM_ROCM_USE_AITER", "0") == "1",
            "load_format": args.load_format,
            "real_weights": real_weights,
            "random_tokens": random_tokens,
            "moe_routing": routing_applied,
            "zipf_s": zipf_s if routing == "zipf" else None,
            "moe_imbalance_target": float(imbalance_target) if imbalance_target else None,
            "moe_imbalance_realized": imbalance_realized,
            "num_experts": n_experts or None,
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
    ap.add_argument("--num-hidden-layers", type=int, default=None,
                    help="override the model's transformer layer count (sub-scale "
                         "benchmarking so a large model fits on fewer GPUs; recorded "
                         "in meta as num_hidden_layers)")
    ap.add_argument("--load-format", default="dummy",
                    help="vLLM load_format: 'dummy' (random weights, needs an "
                         "imposed routing dist) or 'auto'/'safetensors' (REAL "
                         "weights -> the trained router sets the distribution; "
                         "use with --routing-dist none for a constant-free run)")
    ap.add_argument("--random-tokens", action="store_true",
                    help="use independent random token ids per sequence "
                         "(auto-on for real weights; matches InferenceX random data)")
    ap.add_argument("--vocab", type=int, default=30000,
                    help="upper bound for random token ids")
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--skip-tokenizer-init", action="store_true",
                    help="skip loading the tokenizer (benchmark uses token ids "
                         "directly; auto-on for --load-format dummy)")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--no-aiter", action="store_true",
                    help="disable AMD AITER kernels (default: enabled on ROCm)")
    ap.add_argument("--routing-dist", default="zipf",
                    choices=["zipf", "uniform", "normal", "none"],
                    help="MoE token->expert distribution for the benchmark "
                         "(default: zipf; 'none' uses the model's own router)")
    ap.add_argument("--zipf-s", type=float, default=1.0,
                    help="Zipfian skew exponent (0=uniform, larger=more skewed)")
    ap.add_argument("--moe-imbalance", type=float, default=None,
                    help="Target MoE expert-load imbalance I=max/mean tokens-per-"
                         "expert (1.0=balanced). Overrides --zipf-s by solving for "
                         "the Zipf exponent at the model's expert count. Use a "
                         "measured/expected production value (random data ~ low I, "
                         "domain-clustered traffic ~ higher I).")
    ap.add_argument("--save", required=True)
    args = ap.parse_args()

    result = run_vllm_benchmark(args)
    with open(args.save, "w") as f:
        json.dump(result, f)
    print("[Primus:Inference:vLLM-Benchmark] " + json.dumps(result))
    print(f"[Primus:Inference:vLLM-Benchmark] wrote {args.save}")


if __name__ == "__main__":
    main()
