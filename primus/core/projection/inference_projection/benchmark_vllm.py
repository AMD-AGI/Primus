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

Reduce -> benchmark -> restore (matches the training layer benchmark)
--------------------------------------------------------------------
Like ``benchmark.py`` (which reduces the model, times it, and lets
``performance.py`` restore the full model by layer count + parallelism), the vLLM
backend supports the same policy along two axes:

* **Depth** — pass ``--bench-layers`` with two or more REDUCED layer counts. The
  engine is built at each count, step latency is fit vs layer count
  (``t(L) = overhead + L * per_layer``), and the FULL model (``--full-layers`` or
  the HF config) is RESTORED by evaluating the fit at the true depth.
* **Parallelism** — pass ``--benchmark-gpus`` smaller than the target ``TP*PP``.
  Parallelism is reduced in the same ``pp -> ep -> tp`` order as the Megatron
  sub-node benchmark to fit the GPUs on hand (e.g. ``--tp 8 --benchmark-gpus 1``
  runs TP=1 on 1 GPU); the artifact records both the target (``tp``/``ep``/``pp``)
  and benchmark (``benchmark_tp``/``benchmark_ep``/``benchmark_pp``) parallelism,
  and ``performance.py`` RESTORES the whole-model latency to the target the same
  way it restores the per-layer Megatron bench (strip bench comm, scale shardable
  compute by ``bench_tp/target_tp``, add target EP/TP comm + PP delta). In vLLM
  (no data parallelism) EP == TP and the request batch is shared across the group,
  so per-rank expert compute already scales ~1/EP and is captured by the TP
  compute-scaling term — ``num_experts`` is left unchanged; use ``--bench-layers``
  for MoE memory fit.

Both axes compose. Without ``--bench-layers`` the model is measured at full depth
(or a single sub-scale run via legacy ``--num-hidden-layers``); without
``--benchmark-gpus`` it runs at the full target parallelism (no restore needed).

This module is intentionally dependency-light (only ``vllm`` + stdlib) so it can
run inside a vLLM container that does not have Primus installed::

    # full model (no reduction)
    python3 benchmark_vllm.py --model openai/gpt-oss-120b --tp 1 \
        --input-len 1024 --batch 16 --decode-steps 32 --save out.json

    # reduce -> benchmark -> restore a large model from sub-scale runs
    # (depth: 4,8 layers -> 61; parallelism: TP=1 on 1 GPU -> TP=8/EP=8)
    python3 benchmark_vllm.py --model deepseek-ai/DeepSeek-V3 --tp 8 \
        --load-format dummy --enable-expert-parallel --benchmark-gpus 1 \
        --bench-layers 4,8 --full-layers 61 \
        --input-len 1024 --batches 4,16,64 --save out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time

# Shared regime/config keying (stdlib-only module) so the result cache uses the
# exact same scheme as the anchor store. benchmark_vllm.py runs as a standalone
# script inside a bare vLLM container, so import defensively: add this file's
# directory to the path and fall back to inline hashing if unavailable.
try:  # pragma: no cover - import shim for in-container script execution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from search.regime import (  # type: ignore
        config_key as _regime_config_key,
        recipe_from_bench_args as _regime_recipe_from_bench_args,
        regime_signature as _regime_signature,
    )
    _HAVE_REGIME = True
except Exception:  # noqa: BLE001 - any import failure => inline fallback
    _HAVE_REGIME = False


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


def _reduce_parallelism(target_tp, target_pp, target_ep, benchmark_gpus):
    """Reduce parallelism to fit ``benchmark_gpus`` GPUs, in the same
    ``pp -> ep -> tp`` order as the Megatron sub-node benchmark
    (``_calculate_single_node_config``).

    In vLLM (without data parallelism) the world size is ``tp * pp`` and expert
    parallelism is NOT an independent GPU axis: ``EP == TP`` and the request batch
    is *shared* across the TP/EP group (KV duplicated per GPU), so each rank's
    expert-GEMM work already scales as ~1/EP. That reduction is captured by the
    TP compute-scaling term (``bench_tp/target_tp``) on restore — we do NOT touch
    ``num_experts`` (cutting it would change the model's compute, not preserve it,
    unlike Megatron where per-rank token load is fixed by the local micro-batch).
    Memory fit for large MoE models is handled by depth reduction
    (``--bench-layers``) instead. So after collapsing PP, the only GPU-count
    reduction is TP, and EP follows it.

    Returns ``(bench_tp, bench_pp, bench_ep)``.
    """
    tp = max(1, int(target_tp))
    pp = max(1, int(target_pp))
    ep = max(1, int(target_ep))
    if benchmark_gpus is None or benchmark_gpus >= tp * pp:
        return tp, pp, ep

    # Step 1: PP -> 1 (cheapest to add back — a single P2P delta on restore).
    bench_pp = 1
    bench_tp = tp

    # Steps 2/3: EP then TP. EP is tied to TP in vLLM, so reducing TP reduces EP.
    if bench_tp * bench_pp > benchmark_gpus:
        max_tp = max(1, benchmark_gpus // bench_pp)
        bench_tp = 1
        for cand in (max_tp, max_tp // 2, 1):
            if 1 <= cand <= max_tp and tp % cand == 0:
                bench_tp = cand
                break
    bench_ep = bench_tp if target_ep > 1 else 1
    return bench_tp, bench_pp, bench_ep


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


def _full_num_layers(model: str, trust_remote_code: bool) -> int:
    """Best-effort full transformer layer count from the HF config (0 if unknown)."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    except Exception:
        return 0
    for src in (cfg, getattr(cfg, "text_config", None)):
        if src is None:
            continue
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            v = getattr(src, attr, None)
            if isinstance(v, int) and v > 0:
                return v
    return 0


def _parse_bench_layers(spec) -> list:
    """Parse the ``--bench-layers`` comma list into a sorted list of unique ints."""
    if not spec:
        return []
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if tok:
            try:
                v = int(tok)
                if v > 0:
                    out.append(v)
            except ValueError:
                pass
    return sorted(set(out))


def _linfit(points: list):
    """Least-squares fit ``y = a + b*x`` over ``points`` = [(x, y), ...].

    Returns ``(b, a)`` (slope=per-layer, intercept=fixed overhead). With a single
    point, assumes zero overhead (``b = y/x``, ``a = 0``).
    """
    n = len(points)
    if n == 1:
        x, y = points[0]
        return (y / x if x else 0.0), 0.0
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        x, y = points[0]
        return (y / x if x else 0.0), 0.0
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return b, a


def _free_llm(llm) -> None:
    """Release a vLLM engine and its GPU memory before building the next one."""
    import gc

    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


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

    batches = (
        [int(b) for b in str(args.batches).split(",") if b.strip()]
        if args.batches
        else [args.batch]
    )
    max_batch = max(batches)
    # Seeds are swept inside a SINGLE engine build (no re-init per seed): each
    # seed re-rolls the random token content (real weights / --random-tokens) and
    # provides an independent timing sample, giving a cheap noise estimate.
    seeds = (
        [int(s) for s in str(args.seeds).split(",") if s.strip()]
        if getattr(args, "seeds", None)
        else [int(getattr(args, "seed", 0) or 0)]
    )
    max_len = args.max_model_len or (args.input_len + args.decode_steps + 16)
    real_weights = args.load_format != "dummy"
    # Real weights => the trained router needs varied input to route realistically.
    random_tokens = args.random_tokens or real_weights

    # --- REDUCE PARALLELISM (pp -> ep -> tp) to fit --benchmark-gpus ----------
    # Target parallelism is what we PROJECT to; the benchmark may run at a smaller
    # parallelism that fits the GPUs on hand (e.g. TP=1 on 1 GPU). performance.py
    # then RESTORES the whole-model latency to the target TP/EP/PP the same way
    # the Megatron per-layer path does. In vLLM (no DP) EP == TP and the request
    # batch is shared across the group, so per-rank expert compute scales ~1/EP
    # and is handled by the TP compute-scaling term on restore; num_experts is
    # left untouched (use --bench-layers for MoE memory fit).
    target_tp = max(1, int(args.tp))
    target_pp = max(1, int(getattr(args, "pp", 1) or 1))
    target_ep = target_tp if args.enable_expert_parallel else 1
    benchmark_gpus = getattr(args, "benchmark_gpus", None)
    bench_tp, bench_pp, bench_ep = _reduce_parallelism(
        target_tp, target_pp, target_ep, benchmark_gpus
    )
    reduced_parallelism = (bench_tp, bench_pp, bench_ep) != (target_tp, target_pp, target_ep)
    if reduced_parallelism:
        print(f"[Primus:Inference:vLLM-Benchmark] REDUCE PARALLELISM (pp->ep->tp) to fit "
              f"{benchmark_gpus} GPU(s): TP {target_tp}->{bench_tp}, EP {target_ep}->{bench_ep}, "
              f"PP {target_pp}->{bench_pp} (performance.py restores to target)")

    def _build_llm(num_layers_override):
        """Build a vLLM engine at the (possibly reduced) benchmark parallelism,
        optionally overriding the transformer layer count."""
        from vllm import LLM

        kwargs = dict(
            model=args.model,
            tensor_parallel_size=bench_tp,
            load_format=args.load_format,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=max_len,
            max_num_seqs=max(256, max_batch),
            trust_remote_code=args.trust_remote_code,
            enforce_eager=args.enforce_eager,
        )
        if bench_pp > 1:
            kwargs["pipeline_parallel_size"] = bench_pp
        # Expert parallelism: shard MoE experts across the TP ranks (EP = TP)
        # instead of tensor-slicing each expert. Required to expose the
        # imbalance-sensitive effects — the MoE step is then gated by the BUSIEST
        # rank's expert load plus the all-to-all dispatch/combine volume, which a
        # single-rank (EP=1) grouped GEMM hides (FLOPs conserved under skew).
        if bench_ep > 1:
            kwargs["enable_expert_parallel"] = True
        # The benchmark drives the engine purely with ``prompt_token_ids`` (never
        # text), so the tokenizer is unnecessary. Skipping its init avoids a hard
        # dependency on sentencepiece/tiktoken for models whose fast tokenizer
        # can't be built in the container. Auto-on for dummy weights.
        if args.skip_tokenizer_init or args.load_format == "dummy":
            kwargs["skip_tokenizer_init"] = True
        if args.quantization:
            kwargs["quantization"] = args.quantization
        if args.kv_cache_dtype:
            kwargs["kv_cache_dtype"] = args.kv_cache_dtype
        # HF override: reduced layer count only (depth restore). num_experts is
        # deliberately NOT reduced — see the reduce-parallelism note above.
        if num_layers_override:
            kwargs["hf_overrides"] = {"num_hidden_layers": int(num_layers_override)}
        return LLM(**kwargs)

    def _sweep(llm):
        """Sweep every (batch, seed) in ONE engine build and aggregate per batch.

        Multiple seeds are measured back-to-back on the *same* warm engine (no
        re-init between seeds), so a whole seed set costs one engine build. The
        returned per-batch entry carries the mean plus the min and std across
        seeds; ``raw`` keeps every individual (batch, seed) sample.
        """
        raw = []
        for b in batches:
            for sd in seeds:
                entry = _measure_batch(llm, args.input_len, b, args.decode_steps,
                                       random_tokens=random_tokens, vocab=args.vocab,
                                       seed=sd)
                entry["seed"] = sd
                raw.append(entry)
                if len(seeds) > 1:
                    print(f"[Primus:Inference:vLLM-Benchmark]   batch={b} seed={sd} "
                          f"prefill={entry['prefill_ms']:.2f}ms "
                          f"decode_step={entry['decode_ms']:.2f}ms")
        sweep = []
        for b in batches:
            pts = [e for e in raw if e["batch"] == b]
            dvals = [e["decode_ms"] for e in pts]
            pvals = [e["prefill_ms"] for e in pts]
            entry = {
                "batch": b,
                "prefill_ms": statistics.mean(pvals),
                "decode_ms": statistics.mean(dvals),
                "decode_ms_min": min(dvals),
                "decode_ms_std": statistics.pstdev(dvals) if len(dvals) > 1 else 0.0,
                "prefill_ms_std": statistics.pstdev(pvals) if len(pvals) > 1 else 0.0,
                "n_seeds": len(dvals),
            }
            sweep.append(entry)
            spread = (f" (+/-{entry['decode_ms_std']:.2f} over {len(dvals)} seeds)"
                      if len(dvals) > 1 else "")
            print(f"[Primus:Inference:vLLM-Benchmark] batch={b} "
                  f"prefill={entry['prefill_ms']:.2f}ms "
                  f"decode_step={entry['decode_ms']:.2f}ms{spread}")
        return sweep, raw

    # --- REDUCE -> BENCHMARK -> RESTORE -------------------------------------
    # Mirrors the training layer benchmark (benchmark.py + performance.py): the
    # engine is REDUCED to a few layer counts that fit on the available GPUs,
    # BENCHMARKED at each, then the full model is RESTORED by fitting the measured
    # step latency vs layer count (t(L) = overhead + L * per_layer) and evaluating
    # at the true layer count. Because HF layer reduction keeps the first D
    # (fixed) dense layers, the linear fit + restore is exact under a two-type
    # (dense/MoE) homogeneity assumption as long as every benchmarked count keeps
    # those D dense layers. This makes the emitted whole-model latency a genuine
    # full-model projection, which performance.py consumes directly.
    bench_counts = _parse_bench_layers(args.bench_layers)
    restore_meta = None
    sweep_raw = None
    if len(bench_counts) >= 2:
        full_layers = int(args.full_layers or _full_num_layers(args.model, args.trust_remote_code) or 0)
        if full_layers <= 0:
            raise SystemExit(
                "[Primus:Inference:vLLM-Benchmark] --bench-layers restore needs the full "
                "layer count; could not read it from the HF config. Pass --full-layers."
            )
        print(f"[Primus:Inference:vLLM-Benchmark] REDUCE->BENCHMARK->RESTORE: "
              f"bench layer counts {bench_counts} -> restore to {full_layers} layers")
        bench_sweeps = {}
        for li, L in enumerate(bench_counts):
            print(f"[Primus:Inference:vLLM-Benchmark] --- benchmarking {L} layers ---")
            llm = _build_llm(L)
            bench_sweeps[L] = {e["batch"]: e for e in _sweep(llm)[0]}
            # Free every engine except keep-none: each count is a distinct model.
            _free_llm(llm)

        sweep = []
        per_layer = []
        for b in batches:
            pre_pts = [(L, bench_sweeps[L][b]["prefill_ms"]) for L in bench_counts if b in bench_sweeps[L]]
            dec_pts = [(L, bench_sweeps[L][b]["decode_ms"]) for L in bench_counts if b in bench_sweeps[L]]
            bp, ap = _linfit(pre_pts)
            bd, ad = _linfit(dec_pts)
            full_prefill = max(1e-3, ap + bp * full_layers)
            full_decode = max(1e-6, ad + bd * full_layers)
            sweep.append({"batch": b, "prefill_ms": full_prefill, "decode_ms": full_decode})
            per_layer.append({
                "batch": b,
                "per_layer_prefill_ms": bp, "overhead_prefill_ms": ap,
                "per_layer_decode_ms": bd, "overhead_decode_ms": ad,
            })
            print(f"[Primus:Inference:vLLM-Benchmark] RESTORED batch={b} "
                  f"prefill={full_prefill:.2f}ms decode_step={full_decode:.2f}ms "
                  f"(per-layer decode={bd:.3f}ms, overhead={ad:.2f}ms)")
        restore_meta = {
            "bench_layers": bench_counts,
            "full_layers": full_layers,
            "per_layer": per_layer,
            "bench_sweeps": {str(L): list(s.values()) for L, s in bench_sweeps.items()},
        }
        eff_layers = full_layers
    else:
        # No restore: measure the model as configured (full, or a single reduced
        # count via legacy --num-hidden-layers). One engine, reused across batches.
        llm = _build_llm(args.num_hidden_layers)
        sweep, sweep_raw = _sweep(llm)
        eff_layers = args.num_hidden_layers

    ref = next((e for e in sweep if e["batch"] == args.batch), sweep[0])
    result = {
        "backend": "vllm",
        # ``measured.model`` is the single-batch anchor (projector compat);
        # ``sweep`` carries the per-concurrency curve (preferred). Both are the
        # RESTORED full-model latencies when reduce/restore is used.
        "measured": {"model": {"prefill_ms": ref["prefill_ms"], "decode_ms": ref["decode_ms"]}},
        "sweep": sweep,
        # Per-(batch, seed) raw samples when a seed set was swept in one engine.
        "sweep_raw": sweep_raw,
        "meta": {
            "batch": ref["batch"],
            "input_len": args.input_len,
            "decode_steps": args.decode_steps,
            "seeds": seeds,
            # Target parallelism (what performance.py restores TO)...
            "tp": target_tp,
            "ep": target_ep,
            "pp": target_pp,
            # ...and the parallelism the benchmark actually RAN at (restore FROM).
            "benchmark_tp": bench_tp,
            "benchmark_ep": bench_ep,
            "benchmark_pp": bench_pp,
            "benchmark_gpus": benchmark_gpus,
            "num_hidden_layers": eff_layers,
            "restored": restore_meta is not None,
            "restore": restore_meta,
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
    # Stamp the regime signature so the anchor store can index without
    # re-deriving it (falls back silently if the shared module is unavailable).
    if _HAVE_REGIME:
        try:
            env = {"VLLM_ROCM_USE_AITER": os.environ.get("VLLM_ROCM_USE_AITER", "0")}
            result["meta"]["regime_signature"] = _regime_signature(
                _regime_recipe_from_bench_args(args, env)
            )
        except Exception:  # noqa: BLE001
            pass
    return result


# --- result cache ---------------------------------------------------------
# Every benchmark pays a large fixed cost (container/import, engine init,
# torch.compile, CUDA-graph capture) that dwarfs the actual measurement. Since
# a run is fully determined by its config, we key results by the config-
# affecting args (+ the AITER env) and reuse them: a cache HIT skips the vLLM
# import and engine init entirely.
_CACHE_IGNORE_ARGS = {"save", "cache_dir", "no_cache", "force"}

# Args that change the measured number but not the regime/transport axes; keyed
# as ``extra`` so the cache stays exact while regime/transport come from the
# shared signature scheme.
_CACHE_EXTRA_ARGS = (
    "decode_steps", "batches", "bench_layers", "full_layers", "benchmark_gpus",
    "random_tokens", "vocab", "gpu_mem_util", "routing_dist", "zipf_s",
    "moe_imbalance", "load_format", "skip_tokenizer_init", "no_aiter",
    "max_model_len", "output_len", "seed", "seeds",
)


def _cache_key(args) -> str:
    env = {"VLLM_ROCM_USE_AITER": os.environ.get("VLLM_ROCM_USE_AITER", "0")}
    if _HAVE_REGIME:
        recipe = _regime_recipe_from_bench_args(args, env)
        extra = {k: getattr(args, k, None) for k in _CACHE_EXTRA_ARGS}
        return _regime_config_key(recipe, extra)
    # Inline fallback: hash every config-affecting arg + the AITER env.
    payload = {k: v for k, v in sorted(vars(args).items()) if k not in _CACHE_IGNORE_ARGS}
    payload["_env_VLLM_ROCM_USE_AITER"] = env["VLLM_ROCM_USE_AITER"]
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"vllm_bench_{key}.json")


def main():
    ap = argparse.ArgumentParser(description="vLLM inference benchmark backend")
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--tp", type=int, default=1,
                    help="TARGET tensor parallel size to project to (the benchmark "
                         "may run at a smaller TP via --benchmark-gpus and restore).")
    ap.add_argument("--pp", type=int, default=1,
                    help="TARGET pipeline parallel size to project to.")
    ap.add_argument("--benchmark-gpus", type=int, default=None,
                    help="GPUs available for the actual vLLM run. When smaller than "
                         "TP*PP, parallelism is reduced (pp->ep->tp) to fit and "
                         "performance.py restores the whole-model latency to the "
                         "target TP/EP/PP. E.g. --tp 8 --benchmark-gpus 1 runs TP=1 "
                         "on 1 GPU and projects to TP=8.")
    ap.add_argument("--quantization", default=None, help="e.g. fp8, mxfp4 (None=from config)")
    ap.add_argument("--kv-cache-dtype", default=None, help="e.g. fp8 (None=auto)")
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=1024, help="recorded in meta only")
    ap.add_argument("--decode-steps", type=int, default=32, help="K for decode-step timing")
    ap.add_argument("--batch", type=int, default=16, help="ref batch (anchor) when no --batches")
    ap.add_argument("--batches", default=None, help="comma list to sweep, e.g. 4,8,16,32,64")
    ap.add_argument("--seed", type=int, default=0,
                    help="single RNG seed for random token content (default 0)")
    ap.add_argument("--seeds", default=None,
                    help="comma list of seeds to sweep in ONE engine build, e.g. "
                         "'0,1,2'. Each seed re-rolls random token content and adds "
                         "an independent timing sample; the emitted sweep carries the "
                         "per-batch mean + std across seeds (no engine re-init).")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--bench-layers", default=None,
                    help="comma list of REDUCED layer counts to benchmark and "
                         "RESTORE from, e.g. '4,8'. Enables the reduce->benchmark->"
                         "restore policy: the engine is built at each count, the "
                         "step latency is fit vs layer count, and the full model "
                         "(--full-layers, or the HF config) is reconstructed. The "
                         "emitted sweep is the restored full-model latency.")
    ap.add_argument("--full-layers", type=int, default=None,
                    help="full transformer layer count to restore to when using "
                         "--bench-layers (default: read num_hidden_layers from the "
                         "HF config)")
    ap.add_argument("--num-hidden-layers", type=int, default=None,
                    help="[legacy, no restore] override the model's transformer "
                         "layer count for a single sub-scale run. Prefer "
                         "--bench-layers for a full-model projection.")
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
    ap.add_argument("--enable-expert-parallel", action="store_true",
                    help="shard MoE experts across the TP ranks (EP=TP) instead of "
                         "tensor-slicing each expert; exposes imbalance-sensitive "
                         "busiest-rank + all-to-all effects")
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
    ap.add_argument("--cache-dir", default=os.environ.get("PRIMUS_BENCH_CACHE"),
                    help="Directory of cached results keyed by run config. On a "
                         "cache HIT the vLLM engine is never built (skips ~all the "
                         "wall time). Defaults to $PRIMUS_BENCH_CACHE; caching is "
                         "off when neither is set.")
    ap.add_argument("--no-cache", action="store_true",
                    help="Ignore any cached result and do not write one.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run and OVERWRITE the cached result for this config.")
    args = ap.parse_args()

    cache_dir = args.cache_dir
    key = _cache_key(args) if cache_dir else None
    cpath = _cache_path(cache_dir, key) if cache_dir else None

    if cache_dir and not args.no_cache and not args.force and os.path.exists(cpath):
        with open(cpath) as f:
            result = json.load(f)
        with open(args.save, "w") as f:
            json.dump(result, f)
        print(f"[Primus:Inference:vLLM-Benchmark] CACHE HIT {cpath} "
              f"(config key {key}); skipped vLLM run")
        print("[Primus:Inference:vLLM-Benchmark] " + json.dumps(result))
        print(f"[Primus:Inference:vLLM-Benchmark] wrote {args.save}")
        return

    result = run_vllm_benchmark(args)
    with open(args.save, "w") as f:
        json.dump(result, f)
    if cache_dir and not args.no_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cpath, "w") as f:
            json.dump(result, f)
        print(f"[Primus:Inference:vLLM-Benchmark] cached result -> {cpath} "
              f"(config key {key})")
    print("[Primus:Inference:vLLM-Benchmark] " + json.dumps(result))
    print(f"[Primus:Inference:vLLM-Benchmark] wrote {args.save}")


if __name__ == "__main__":
    main()
