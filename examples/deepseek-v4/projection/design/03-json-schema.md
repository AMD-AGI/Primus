# 03 — Breakdown JSON schema

One JSON per model variant (`flash` / `pro`), written to
`site/data/<model>.json`, consumed by the website. All times in **microseconds
(us)**; all FLOP counts are per single layer / per single invocation at the
captured `seq` and `micro_batch_size`.

## Top level

```jsonc
{
  "schema_version": 1,
  "model": "pro",                       // "flash" | "pro"
  "generated_at": "2026-06-18T11:00:00Z",
  "provenance": {
    "commit": "dac0a60c",
    "host": "smci355-ccs-aus-n06-25",
    "container": "dev_primus_wenx",
    "traces": { "cr0": "<path>", "cr4": "<path>", "cr128": "<path>" }
  },

  "capture": {                          // how the trace was taken (the unit)
    "gpu": "MI355X",
    "seq_length": 4096,
    "micro_batch_size": 1,
    "tokens_per_microbatch": 4096,      // seq_length * micro_batch_size
    "ep": 8,
    "ga_for_capture": 2,
    "optimizer": "adam",
    "distributed_optimizer": true,
    "recompute": "off"
  },

  "model_config": {                     // shown on the site's config panel
    "num_layers": 61,
    "hidden_size": 7168,
    "num_attention_heads": 128,
    "kv_channels": 512,
    "num_experts": 384,
    "moe_router_topk": 6,
    "moe_ffn_hidden_size": 3072,
    "index_topk": 1024,
    "vocab_size": 129280,
    "compress_ratios": [128,128,4, /* ... */ ,0],
    "cr_layer_counts": { "0": 1, "4": 29, "128": 31 }   // derived from compress_ratios
  },

  "hardware": {                         // for MI355->MI455 scaling (A20-A23)
    "MI355X": { "peak_tflops_bf16": 2300, "hbm_bandwidth_gbps": 8000 },
    "MI455X": { "peak_tflops_bf16": 4600, "hbm_bandwidth_gbps": 16000 }
  },

  "layers": {                           // per-cr breakdown
    "0":   { "attention": <Breakdown>, "moe": <Breakdown> },
    "4":   { "attention": <Breakdown>, "moe": <Breakdown> },
    "128": { "attention": <Breakdown>, "moe": <Breakdown> }
  },

  "non_layer": {                        // taken once (A15)
    "embedding": <Breakdown>,
    "output":    <Breakdown>,           // final norm + lm_head/logits
    "loss":      <Breakdown>
  },

  "optimizer": {                        // per-iteration term (A3)
    "type": "adam",
    "measured_params": 123456789,       // params updated in the 1-layer trace, this rank
    "time_us": 850.0,                   // measured optimizer-step time for those params
    "bytes_per_param": 18,              // bf16 master-param-remainder adam state bytes
    "class": "memory_bound"
  },

  "comm": {                             // per-microbatch EP cost (A7-A9), per layer
    "ep_dispatch_us": 0.0,
    "ep_combine_us": 0.0
  }
}
```

## `<Breakdown>` object

A phase-split list of modules. Each module is one row group; the site renders
forward left-to-right and backward right-to-left.

```jsonc
{
  "forward":  [ <Module>, ... ],
  "backward": [ <Module>, ... ]
}
```

## `<Module>` object

```jsonc
{
  "module": "attn.core",          // logical module name (from kernel/module map)
  "time_us": 740.0,               // clean min-grouped time for this module/phase
  "class": "compute_bound",       // "compute_bound" | "memory_bound"
  "flop_class": "attn",           // "gemm" | "grouped_gemm" | "attn" | null
  "flops": 1.84e12,               // total FLOPs for this module/phase, or null
  "tflops": 740.0,                // achieved TFLOP/s = flops / time_s / 1e12 (null if memory_bound)
  "kernels": [                    // optional: contributing kernels (debug / drill-down)
    { "name": "_v4_attention_fwd_kernel", "time_us": 500.0, "launches": 1 }
  ]
}
```

## Notes

- `time_us` is always the **clean** (min-grouped, overlap-free) time per A12.
- `tflops` is present only when `flop_class != null` (A14).
- The site computes everything else (full model, PP/EP/DP, MI455) from this JSON;
  the JSON itself is hardware-MI355X, single-layer, single-microbatch ground
  truth + static config.
- `cr_layer_counts` is derived from `compress_ratios` by the parser so the site
  doesn't re-parse the schedule.
- For Flash, `model` = `"flash"`, `cr_layer_counts` e.g. `{"0":3,"4":20,"128":20}`.
