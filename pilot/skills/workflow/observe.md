# Observe

**Status**: Stub

Snapshot data definition (§8.3): `tps`, `step_time_ms`, `comm_ratio`, `bubble_ratio`, `overlap_ratio`, `mem_peak_gb`, `gpu_util_avg`. Source: WandB / Prometheus / rocprof timeline parser.

## TODO

- [ ] Metric definition table (units, sources, sampling window)
- [ ] Trace-level metric extraction protocol (delegate to `profiling/trace.md`)
- [ ] Snapshot validity rules (min steps, warmup exclusion)
