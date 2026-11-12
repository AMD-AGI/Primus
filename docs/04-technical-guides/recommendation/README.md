# Recommendation models

The `recommendation` backend (`framework: recommendation`) trains recommendation
models. The model preset's `model_family` selects the trainer.

| Model family | Model | Recipe | Guide |
|---|---|---|---|
| `dlrm_v4` | HSTU ranker on Yambda-5b, MLPerf Training `dlrmv4` (MI355X) | [examples/mlperf/dlrm_v4](../../../examples/mlperf/dlrm_v4/README.md) | [DLRMv4](dlrm_v4/README.md) |

## DLRMv4

- [DLRMv4 guide](dlrm_v4/README.md): requirements, running, configuration,
  precision, results, multi-node, updating the image, troubleshooting
- [Benchmark reference](dlrm_v4/benchmark.md): dataset, data order, model,
  quality target and eval schedule

---

[← Technical guides](../README.md)
