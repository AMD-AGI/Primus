# DeepSeek-V4 Performance Projection

A trace-driven performance projection toolkit + static website for DeepSeek-V4
(Flash / Pro) training on AMD Instinct GPUs (MI355X measured, MI455X projected).

## Idea in one paragraph

We profile a **single transformer layer** of a given compression-ratio (`cr`)
type on **MI355X**, extract a clean forward / backward **time + TFLOPs breakdown**
per module (attention sub-modules, MoE sub-modules), plus the model's non-layer
parts (embedding, output/logits, loss) and the optimizer step. We emit one JSON
per model variant. A static website then loads that JSON and, given a target GPU
and a distributed strategy (PP / VPP / EP / DP / CP), reconstructs the full model
(real layer count + `cr` schedule), models the PP bubble, EP dispatch/combine,
recompute, and the optimizer, and derives **iteration time, TFLOP/s/GPU and
tokens/s/GPU** step by step. Page 1 is the MI355X projection; page 2 scales the
breakdown to MI455X by theoretical ratios.

## Pipeline

```
run profiling script (per cr)  ->  chrome trace JSON (rank 0)
        |                                  |
        |                                  v
        |                       tools/parse_trace.py  (+ kernel/module map)
        v                                  |
  one trace per cr type {0,4,128}          v
                                breakdown JSON (site/data/<model>.json)
                                           |
                                           v
                              static site (site/index.html)
                                  - model config view
                                  - per-cr fwd/bwd breakdown tables
                                  - GPU + parallelism controls
                                  - step-by-step iter-time / TFLOPs / tok/s derivation
                                  - MI355X page + MI455X scaled page
```

## Directory layout

```
deepseek-v4/projection/
  README.md                      # this file
  design/                        # methodology, assumptions, schema, math (the spec)
    01-overview.md
    02-assumptions.md
    03-json-schema.md
    04-projection-math.md
  script/                        # profiling launchers (one trace per cr)
    deepseek_v4_layer_trace-projection.sh
  tools/                         # trace -> breakdown JSON
    parse_trace.py
    kernel_module_map.py
  site/                          # static website (no build step)
    index.html
    assets/app.js
    assets/style.css
    data/<model>.json            # breakdown JSON consumed by the site
```

## Quick start

1. Profile each `cr` on MI355X (run inside the training container):

   ```bash
   # one trace per cr type; 1 layer, seq 4096, adam + dist-opt, GA=2, no recompute
   CR=0   ./deepseek-v4/projection/script/deepseek_v4_layer_trace-projection.sh
   CR=4   ./deepseek-v4/projection/script/deepseek_v4_layer_trace-projection.sh
   CR=128 ./deepseek-v4/projection/script/deepseek_v4_layer_trace-projection.sh
   ```

2. Build the breakdown JSON from the three traces:

   ```bash
   python3 deepseek-v4/projection/tools/parse_trace.py \
     --model pro \
     --trace cr0=<trace_cr0.json> \
     --trace cr4=<trace_cr4.json> \
     --trace cr128=<trace_cr128.json> \
     --out deepseek-v4/projection/site/data/pro.json
   ```

3. Open the site locally:

   ```bash
   python3 -m http.server -d deepseek-v4/projection/site 8000
   # http://localhost:8000/?model=pro
   ```

See `design/` for the full methodology and the exact assumptions baked into the
projection math.
