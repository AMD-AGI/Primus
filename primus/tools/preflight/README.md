
# 🧪 Preflight Overview

**Preflight** is a diagnostic tool designed for large-scale cluster environments. Before starting distributed training, it benchmarks the **compute performance of all GPUs**, as well as **intra-node and inter-node communication bandwidth and latency**. Its primary goal is to help users identify **underperforming nodes** or **network bottlenecks** in the cluster, ensuring reliable and efficient training runs.

## Run preflight

Torch / single node:

```bash
primus-cli preflight \
  --dump-path output/preflight \
  --report-file-name preflight_report
```

Slurm (multi-node example):

```bash
NUM_NODES=8 srun -N ${NUM_NODES} --ntasks-per-node=1 --cpus-per-task=256 \
  primus-cli preflight --dump-path output/preflight --report-file-name preflight_report
```


## 📂 Output Directory

After running **Preflight**, all test results and reports are generated under the `output/preflight` directory.

The final reports are:

- `preflight_report.md` – a Markdown version of the test report
- `preflight_report.pdf` – a PDF version of the same report

These reports summarize GPU performance, intra-node and inter-node communication results, and help identify potential issues within the cluster.

---

## 📁 Directory Structure

```bash
output/preflight
├── inter_node_comm
├── intra_node_comm
├── preflight_report.md
├── preflight_report.pdf
├── square_gemm_tflops
└── ...
```

> *Note: The exact contents may vary depending on the tests enabled during runtime.*
