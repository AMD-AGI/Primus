# Power projection

The power projection answers three questions on top of a performance projection:

1. **How much power does this configuration draw?** Per GPU, per node, per cluster and at the facility.
2. **What does the customer's power limit cost?** When a GPU hits its board power cap, it lowers its clock. The projection solves for that clock and reports the throughput the configuration actually keeps.
3. **How big a cluster fits a power budget?** For a fixed facility or IT budget, it compares per-GPU caps: a lower cap is slower per GPU but fits more GPUs.

The tuning agent uses the same numbers. Under a power cap, it ranks configurations by the power-capped throughput instead of the uncapped one, because the fastest uncapped configuration is often the one the cap throttles hardest.

The built-in model uses only published datasheet numbers. Measured or fitted profiles are loaded from outside the repository (see [Calibrating to telemetry](#calibrating-to-telemetry)).

## Table of contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Calibrating to telemetry](#calibrating-to-telemetry)
- [Infrastructure and capacity planning](#infrastructure-and-capacity-planning)
- [Tuning agent under a power cap](#tuning-agent-under-a-power-cap)
- [Reference](#reference)
- [Assumptions and limitations](#assumptions-and-limitations)

---

## Quick start

```bash
# Power at the default cap (the GPU's published total board power)
primus-cli projection performance --config exp.yaml --profiling-mode simulate \
    --gpu-arch mi355x --target-nodes 4 --power-report

# Under a customer's 1000 W per-GPU limit
primus-cli projection performance --config exp.yaml --profiling-mode simulate \
    --gpu-arch mi355x --target-nodes 4 --power-cap-w 1000

# With infrastructure and a capacity plan under a 5 MW facility budget
primus-cli projection performance --config exp.yaml --profiling-mode simulate \
    --gpu-arch mi355x --target-nodes 4 \
    --power-config examples/hardware_configs/power_config_example.yaml
```

Example output (simulate mode, Llama 3 8B, 2 nodes, 1000 W cap):

```text
Power Projection
   GPU: mi355x (TBP 1400 W, peak clock 2400 MHz, profile: public-spec)
   Workload footprint (simulation): matrix util 0.68, HBM util 0.08, clock-bound share 1.00, non-matrix share 0.15
   Phases (per GPU, per iteration):
     matrix         409.96 ms ->     430.11 ms  clock  95.3%   1000.0 W [capped]
     vector          72.35 ms ->      72.35 ms  clock 100.0%    728.0 W
     comm             8.19 ms ->       8.19 ms  clock 100.0%    495.6 W
     optimizer        9.61 ms ->       9.61 ms  clock 100.0%    781.2 W
   Uncapped Avg Power/GPU: 1027.4 W (peak phase 1096.6 W)
   Power Cap/GPU: 1000 W (instantaneous)
   Avg Power/GPU: 950.2 W (peak phase 1000.0 W)
   Effective Clock: 2307 MHz
   Power-Capped Step Time: 520.258 ms (slowdown 1.040x)
   Power-Capped Throughput: 15,746 tokens/s per GPU
   Energy/Token (GPU): 0.0603 J
   Tokens/J/GPU: 16.571
   Node Power: 9.94 kW | Cluster IT Power: 21.4 kW (2 nodes, 1 switches) | Facility Power: 27.8 kW (PUE 1.30)
   Power Feasible: yes
```

These watt numbers come from the uncalibrated public-spec profile. Treat them as relative (which configuration draws more, and which one the cap throttles) until the profile is fitted to telemetry.

---

## How it works

### Phases of an iteration

The performance projection already knows where an iteration's time goes. The power projection splits that time into phases with different power behavior:

| Phase | Time from the performance projection | Activity |
|-------|--------------------------------------|----------|
| `matrix` | Compute span minus bubbles, GEMM / attention share | Matrix utilization and HBM traffic from the workload footprint |
| `vector` | Compute span minus bubbles, non-matrix share | Memory-bound elementwise work |
| `comm` | Exposed communication (critical path) | Links busy, compute mostly idle |
| `optimizer` | Optimizer step | Memory-bound |
| `bubble` | Pipeline bubble (when the pipeline is simulated) | Idle |

Communication hidden under compute keeps the links active during the `matrix` and `vector` phases.

### Workload footprint

In simulate mode, the projection records every GEMM and attention kernel it prices and derives three ratios:

- **Matrix utilization**: achieved FLOP/s divided by peak FLOP/s.
- **HBM utilization**: bytes moved over time, divided by peak bandwidth.
- **Clock-bound share**: the roofline share of kernel time that is compute-bound, which is the part that slows down when the clock drops.

In benchmark mode (no recorder), documented defaults typical of dense transformer training are used; override them under `power.footprint` in `--power-config`.

### GPU power model

For a phase at relative clock `s = f / f_peak`, board power is modeled as:

```text
P(s) = P_idle + s^alpha * (C_busy * busy + C_matrix * matrix) + (C_hbm * hbm + C_link * link) / stretch
```

- `P_idle` is the power with nothing running.
- The clocked term scales as `s^alpha`. This single exponent stands in for both the frequency and the voltage dependence of dynamic power.
- Memory and link power do not scale with the compute clock. When a lower clock stretches a phase, the same bytes move over a longer time, so their average power drops.

The built-in profiles use only published numbers: total board power (TBP), peak engine clock and peak HBM bandwidth. The coefficients are anchored so that a GPU running a large, well-tuned GEMM stream (busy 1.0, matrix 0.75, HBM 0.5, link 0.5) at its peak clock draws exactly TBP. Idle power (15% of TBP), `alpha` (2.5) and the split of the dynamic headroom are generic defaults, not measurements. [Calibrate](#calibrating-to-telemetry) them for real numbers.

### Power cap

The firmware lowers the clock until board power fits under the cap. A phase's duration stretches as:

```text
t(s) = t0 * (clock_bound_share * s0 / s + (1 - clock_bound_share))
```

Here `s0` is the clock the performance projection's times were computed at: `--gpu-clock-mhz`, or the benchmark GPU's measured effective clock, else the peak.

Two control assumptions are available through `--power-cap-mode`:

- `instantaneous` (the default) caps every phase separately. Power limits act on millisecond windows, so a GEMM-heavy phase throttles even if the iteration average is under the cap.
- `average` uses one clock for the whole iteration, chosen so that the time-averaged power fits.

The cap defaults to the GPU's published TBP. Override it with `--power-cap-w`, `power.cap_w` or `PRIMUS_POWER_CAP_W`. If even the minimum clock cannot meet the cap, the report says `Power Feasible: no`.

---

## Calibrating to telemetry

### 1. Record telemetry

On a node running the workload (or a sweep of workloads at different power caps):

```bash
# Built-in sampler (amdsmi Python bindings): runs the workload and records its load window
primus-cli projection power-sample --device 0 1 2 3 4 5 6 7 \
    --command "bash run_pretrain.sh" --out run.jsonl     # also writes run.window.json

# Or any amd-smi / rocm-smi CSV log with power, clock and activity columns
amd-smi metric --power --clock --usage --csv --watch 1 > run.csv
```

The parser recognizes common amd-smi and rocm-smi column names (socket power, GFX clock, GFX activity, UMC activity, and so on), skips limit, cap and temperature columns, and converts percentages to fractions. Timestamps may be epoch seconds, milliseconds, microseconds, nanoseconds or ISO-8601. When several logs are given (one per host), GPU ids are prefixed with the file name so that each host's GPU 0 stays separate.

### Load window and validity

Telemetry usually runs longer than the load: it includes warmup, idle gaps and teardown. Within the load window, each GPU's power is interpolated at the window edges and integrated with the trapezoid rule, giving per-GPU average power and GPU energy. The measurement is valid only when every GPU's trace covers the whole window. Otherwise the values are withheld and the reasons are listed:

| Reason | Meaning |
|--------|---------|
| `invalid_window` | The window's end is not after its start. |
| `no_samples` / `missing_timestamps` | There is nothing to integrate. |
| `window_not_covered` | A GPU has no sample at or before the start, or at or after the end. |
| `sample_gap_exceeds_max` | Consecutive samples of a GPU are more than 3 s apart (`--max-gap-s`). |
| `device_count_mismatch` | The number of GPUs in the log differs from the number sampled. |

`power-sample --command` records the window around the command, with `--pre-roll-s` of sampling on either side so both edges have a sample. It writes the window and the verdict to `<out>.window.json` and prints a line like this illustrative one:

```text
  Power Valid: yes | 8 GPU(s), 612.4 s | Avg Power/GPU: 1186.2 W | GPU Energy: 5811431.0 J
```

Energy and power cover GPU board power only. They exclude host, network, cooling and facility power.

### 2. Fit a profile

```bash
primus-cli projection power-fit --telemetry run.jsonl --window-file run.window.json \
    --trim-s 60 5 --gpu-arch mi355x --out mi355x_power.yaml
```

Choose the load window with one of:

- `--window-file` (from `power-sample`, or any JSON with `start_unix`/`end_unix` or `benchmark_start_time_unix`/`benchmark_end_time_unix`)
- `--window START END` (Unix seconds)
- `--auto-window` (from the first to the last sample with GFX activity of at least 50%)

`--trim-s HEAD TAIL` drops warmup and teardown from the window. A window that fails validation stops the fit unless you pass `--allow-invalid-window`. Without a window, every sample is fitted and the command prints a warning. To pin idle power, fit a separate log that includes an idle period.

The fit solves non-negative least squares for idle power and the activity coefficients. When the clock varies by more than 5% across the samples, it also fits `alpha` by grid search. Logging at several power caps gives clock variation for free. Without a matrix-utilization column, pass the MFU the training log reported with `--matrix-utilization`, or the busy and matrix terms are fitted as one.

### 3. Use the profile

Fitted and other private profiles do not belong in this repository. Point the projection at them instead:

```bash
export PRIMUS_POWER_PROFILE_DIR=/secure/power_profiles   # directories of *.yaml, os.pathsep-separated
# or
primus-cli projection performance ... --power-config mi355x_power.yaml
```

The resolution order is: public spec, then profiles from `PRIMUS_POWER_PROFILE_DIR` and `power.profile_files`, then inline `power.profile` overrides. Profiles may also describe GPUs without a public spec, as long as they give `tbp_w` and `peak_clock_mhz`:

```yaml
gpu_power_profiles:
  my-gpu:
    tbp_w: 1000
    peak_clock_mhz: 2000
    idle_w: 150
    alpha: 2.2
    coefficients: {busy_w: 160, matrix_w: 480, hbm_w: 170, link_w: 40}
    aliases: [gfx9xx]
```

### Custom power models

A different model can be plugged in without editing Primus. Either call `register_power_model(name, factory)`, or expose a factory in the `primus.power_models` entry-point group and select it with `power.model: <name>`. The factory takes `(profile, config)` and returns an object with `solve(phases, cap_w, cap_mode, base_clock_fraction)` and `phase_power_w(phase, s, s0)`.

---

## Infrastructure and capacity planning

### Node and cluster power

The `infrastructure:` block of `--power-config` adds what sits around the GPUs:

- **Per node:** host CPUs, other host power (DRAM, storage, fans, BMC), scale-out and front-end NICs, an optional per-node share of scale-up switches, and PSU losses.
- **Per cluster:** scale-out switches for a two-tier non-blocking leaf/spine fabric, sized from the NIC count and the switch radix.
- **Facility:** IT power times PUE, plus racks needed when `rack_power_kw` is given.

The defaults are generic placeholders in the range of public datasheets for current server CPUs, NICs and switches. Replace them with the real bill of materials.

### Capacity plan

With `capacity.facility_power_kw` (or `it_power_kw`), the projection evaluates each per-GPU cap in `capacity.caps_w`. For each cap it reports the largest node count that fits the budget (nodes are provisioned at the cap), the cluster throughput, the running power, and tokens per kWh:

```text
   Capacity plan (fixed power budget):
       cap W   tok/s/GPU   avg W  nodes   GPUs   cluster tok/s  run MW      tok/kWh
        1400      16,381    1027    268   2144      35,119,796   3.890   32,501,382
        1200      16,381    1027    304   2432      39,837,381   4.411   32,509,530
        1000      15,746     950    351   2808      44,214,898   4.794   33,199,542
         800      14,220     786    416   3328      47,325,524   4.927   34,577,916
   Best cap under budget: 800 W/GPU -> 47,325,524 tokens/s
```

In this example (Llama 3 8B, 5 MW facility, uncalibrated public-spec profile), capping at 800 W loses 13% per GPU but fits 55% more nodes, for 35% more cluster throughput.

Per-GPU throughput is assumed independent of cluster size (weak scaling at fixed parallelism). Re-run the projection at the chosen node count to confirm.

---

## Tuning agent under a power cap

Add the cap to the target-cluster YAML:

```yaml
target_cluster:
  gpu_arch: mi355x
  power_cap_w: 1000
  power_cap_mode: instantaneous          # optional
  power_config: examples/hardware_configs/power_config_example.yaml   # optional

optimization:
  objective: capped_tokens_per_s_per_gpu # the default when a power cap is set
  power_policy: throttle                 # or "reject"
```

Every performance evaluation then runs with `--power-report` and the cap. Each trial records:

- `capped_tokens_per_s_per_gpu`
- `avg_power_w`, plus `uncapped_avg_power_w` and `uncapped_peak_power_w`
- `power_slowdown`
- `energy_per_token_j` and `tokens_per_joule_per_gpu`
- `facility_power_kw` and `power_feasible`

The LLM sees the cap, the policy and these fields in the cluster blob, the trial history and each result.

There are two policies:

- `throttle` (the default) keeps every configuration and ranks it by the throughput left after throttling. This is what the GPUs will actually do.
- `reject` also drops configurations whose hottest phase would exceed `power_cap_w` at full clock. Use it when the customer requires the job to run unthrottled.

Under either policy, a configuration that cannot meet the cap even at the minimum clock is illegal.

Objectives where lower is better are minimized: `energy_per_token_j`, `avg_power_w`, `iteration_ms`, `capped_iteration_ms` and `memory_per_gpu_gb`. See [`target_cluster_mi355x_4nodes_power_capped.yaml`](https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/guides/tuning_agent/target_cluster_mi355x_4nodes_power_capped.yaml).

---

## Reference

### Options

| Option | Description |
|--------|-------------|
| `--power-report` | Add the power projection to `projection performance` / `both`. Implied by `--power-config` and `--power-cap-w`. |
| `--power-config` | Power YAML (`power`, `infrastructure`, `capacity`), or a fitted profile file. |
| `--power-cap-w` | Per-GPU board power cap in W. Default: published TBP. Also `PRIMUS_POWER_CAP_W`. |
| `--power-cap-mode` | `instantaneous` (default) or `average`. |
| `projection power-fit` | Fit a profile to telemetry: `--telemetry`, `--gpu-arch`, `--out`, `--matrix-utilization`, `--tbp-w`, `--peak-clock-mhz`, `--fit-alpha {auto,yes,no}`, plus the load window: `--window` / `--window-file` / `--auto-window`, `--trim-s`, `--max-gap-s`, `--allow-invalid-window`. |
| `projection power-sample` | Record telemetry from local GPUs: `--device` (one or more), `--command` or `--duration-s`, `--interval-ms`, `--pre-roll-s`, `--out` (plus `<out>.window.json`). |
| `PRIMUS_POWER_PROFILE_DIR` | Directories of private profile YAMLs. |

### Result dictionary

When enabled, the multinode projection result gains a `power` entry with the profile, footprint, time breakdown, per-phase solution, capped throughput, energy per token, the cluster roll-up and the capacity plan.

---

## Assumptions and limitations

1. **Uncalibrated defaults.** Without a fitted profile, absolute watts are only as good as the anchoring assumption (reference load = TBP). Rankings between configurations are more robust than absolute values.
2. **One exponent for clock and voltage.** A single `alpha` cannot capture every operating region of a real part. Fit it on telemetry that covers the clock range you care about.
3. **No thermal or transient effects.** Temperature-dependent leakage, power-excursion handling and clock ramp times are not modeled.
4. **Uniform GPUs.** Every GPU in the job is assumed to run the same phases. Pipeline stages with different loads, and part-to-part variation, are averaged out.
5. **Footprint in benchmark mode.** Without the simulation recorder, workload activity comes from defaults unless `power.footprint` overrides it.

## Related documentation

- [Projection](./projection.md)
- [Tuning agent](./tuning-agent.md)
