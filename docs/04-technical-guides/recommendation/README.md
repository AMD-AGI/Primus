# Recommendation models

The `recommendation` backend (`primus/backends/recommendation/`) trains
recommendation models. A model preset's `model_family` picks the trainer.

| Model family | Model | Recipe |
|---|---|---|
| `dlrm_v4` | HSTU ranker on yambda-5b, MLPerf Training `dlrmv4` | [examples/mlperf/dlrm_v4](../../../examples/mlperf/dlrm_v4/README.md) |

## DLRMv4

- [Benchmark](dlrm_v4/README.md): problem, dataset, preprocessing, model, quality target, data order
- [ROCm 10.2 port](dlrm_v4/rocm-port.md): software stack, results on MI355X, every change and its measured effect, open items
- [FP8](dlrm_v4/fp8.md): the FP8 HSTU kernels and their numerics
- [Multi-node](dlrm_v4/multi-node.md): RDMA fabric and RCCL settings
- [Performance notes](dlrm_v4/perf-notes.md), [MI355X performance notes](dlrm_v4/perf-notes-mi355.md): the optimization log

The trainer came into Primus from the MLPerf reference fork `training_dlrmv4`
(commit `7879b4b`). The port, FP8, multi-node and performance pages were written
there: where they name `scripts/cluster_a/...`, `scripts/cluster_b/...` or
`scripts/launch_slurm.sh`, they mean launchers of that repository, which Primus
replaces with `primus-cli` and `examples/mlperf/dlrm_v4`.

---

[← Technical guides](../README.md)
