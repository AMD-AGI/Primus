# MoE Training Best Practices on AMD GPUs

This document summarizes the best practices for training Mixture-of-Experts (MoE) models on AMD Instinct™ MI300-series GPUs and the ROCm ecosystem. It covers large-scale sparse model distributed training strategies, key performance bottlenecks, practical optimization techniques, and hands-on engineering tips specifically for AMD platforms. Whether you’re new to MoE distributed architectures or working to optimize trillion-parameter models for scalability and performance, this guide will help you identify typical bottlenecks and implement solutions to maximize efficiency on AMD GPUs.

## 1. MoE Model Overview

Mixture of Experts (MoE) is a model architecture designed to efficiently scale neural networks by routing inputs through a subset of specialized sub-models, or "experts." Each expert is a part of a larger ensemble and is trained to handle specific types of data or tasks. The architecture includes a gating mechanism that dynamically routes data to the most relevant experts based on the input, allowing only a few paths to be activated per input. This enables the model to maintain a large capacity while using fewer computational resources, as only a fraction of the model is used during inference. MoE models have shown success in massively increasing model capacity without proportionally increasing computation cost, proving effective in areas like natural language processing.

## 2. Representative Models

The mainstream MoE models today are DeepSeek-style architectures, spanning from 16B up to 671B parameters. Recently, even larger MoE models have started to appear in the open-source community. To keep pace, we also include 1T and 2T proxy models for evaluating how increasing model scale impacts memory footprint, performance, scalability, and corresponding optimization strategies. The table below summarizes the key model configurations for commonly used open-source DeepSeek models as well as large proxy models.

| Model | Total Params | Active Params | Notes |
| --- | --- | --- | --- |
| DeepSeek V2 Lite | 16B | 2.4B | `deepseek_v2_lite.yaml` |
| DeepSeek V2 | 236B | 21B | `deepseek_v2.yaml` |
| DeepSeek V3 | 671B | 37B | `deepseek_v3.yaml` |
| DeepSeek Proxy 1T | 1T | 44B | `deepseek_proxy_1T.yaml` |
| DeepSeek Proxy 2T | 2T | 80B | `deepseek_proxy_2T.yaml` |


## 3. Profiling and Analysis Workflow

For performance analysis and bottleneck identification during MoE model training, we recommend the following workflow:

- **Step 1: Torch Profiler Trace Generation**
  Use the integrated Torch Profiler (`--use_pytorch_profiler`) during training to generate comprehensive traces capturing operator timings, GPU utilization, and kernel launches. Specify the profiling window using `profile_step_start`/`profile_step_end` to focus on particular training steps.

- **Step 2: Detailed Bottleneck Analysis with TraceLen**
  Analyze the profile traces with TraceLen, a specialized tool for fine-grained identification of performance bottlenecks such as communication stalls, unbalanced compute, or inefficient operator fusion. TraceLen can reveal both compile-time and runtime inefficiencies at different model scales.

- **Step 3: Pipeline Parallelism Visualization**
  Primus provides a built-in PP (pipeline-parallel) visualization tool that helps diagnose and visualize pipeline-stage utilization across ranks. This tool is valuable for discovering pipeline bubbles or imbalances that limit throughput.

## 4. Memory Breakdown (Placeholder)

> *TODO: Insert breakdown charts showing activation, optimizer state, and expert parameter footprints for each model class.*

## 5. Memory-Driven Distributed Strategy Differences (Table Placeholder)

| Model | Recommended Parallel Strategy (TP/PP/EP/VPP) | Pipeline Layout / Notes |
| --- | --- | --- |
|  |  |  |
|  |  |  |
|  |  |  |

## 6. Baseline Bottleneck Highlights

- **CPU-sync-driven launch overhead**: Frequent H2D copy and related syncs slow expert activation; mitigate with NUMA binding and multi-stream launches.
- **All-to-all comm hotspots**: For EP ≥ 8, a2a dominates. Sync-Free Router, DeepEP, or a2a+p2p hybrids flatten the peak.
- **Grouped GEMM efficiency**: Large router top-k or imbalanced experts shrink GEMM batches and waste compute units; enabling Turbo Grouped MLP fuses batches to recover throughput.

## 7. Performance Optimizations

This section maps to the `MoE_Features` in `run_deepseek_v3_pretrain_mi325x.sh`. Each feature includes its purpose, the CLI knobs involved, and a placeholder for future speedup data.

### Feature 0 – Baseline

- Description: Plain Megatron implementation without Turbo or overlap toggles.
- Optimization principle: *TBD*
- Args: None.
- Speedup: *TBD*

### Feature 1 – Turbo Attention + Grouped GEMM

- Description: Enables Primus Turbo kernels for EMA-style attention and Grouped MLP efficiency.
- Optimization principle: *TBD*
- Args:
  - `--enable_primus_turbo True`
  - `--use_turbo_attention True`
  - `--use_turbo_grouped_mlp True`
- Speedup: *TBD*
- Details placeholder: *TODO: add kernel-level notes*

### Feature 2 – Sync-Free MoE (Stage 2)

- Description: Removes synchronization for Router, DeepEP, and GroupMLP stages to shrink a2a stalls.
- Optimization principle: *TBD*
- Args:
  - `--enable_primus_turbo True`
  - `--turbo_sync_free_moe_stage 2`
- Speedup: *TBD*
- Details placeholder: *TODO: document differences among stages 0–3*

### Feature 3 – DeepEP Acceleration

- Description: Activates DeepEP kernels with configurable CU usage and comm streams.
- Optimization principle: *TBD*
- Args:
  - `--enable_primus_turbo True`
  - `--use_turbo_deepep True`
  - `--turbo_deepep_num_cu 32`
  - `--turbo_deepep_use_comm_stream False`
  - `--moe_router_dtype fp32`
- Speedup: *TBD*
- Details placeholder: *TODO: list recommended CU counts for EP 8/16/64*

### Feature 4 – 1F1B MoE Overlap

- Description: Uses 1F1B scheduling to overlap expert communication with backward compute.
- Optimization principle: *TBD*
- Args:
  - `--overlap_moe_expert_parallel_comm True`
  - `--patch_moe_overlap True`
  - `--delay_wgrad_compute False`
  - `--moe_shared_expert_overlap False`
- Speedup: *TBD*
- Details placeholder: *TODO: explain dependency on pipeline partitioning*

### Feature 5 – Zero-Bubble Pipeline

- Description: Applies Zero-Bubble techniques to reduce pipeline bubbles, often with virtual pipeline stages.
  - `primus/backends/megatron/core/pipeline_parallel/zerobubble/README.md`
  - `primus/configs/modules/megatron/zero_bubble.yaml`
- Optimization principle: *TBD*
- Required Args:
```
overlap_grad_reduce: false
overlap_param_gather: false
no_persist_layer_norm: true
create_attention_mask_in_dataloader: false
gradient_accumulation_fusion: true
```
- PP Strategy Args:

| pp strategy / flag | num_virtual_stages_per_pipeline_rank | patch_zero_bubble | zero_bubble_v_schedule | zero_bubble_v_schedule_mem_setup |
|---|---|---|---|---|
| turbo-1f1b | 1 |  false | - | - |
| turbo-1f1b-interleaved | >=2 |  false | - | - |
| zero bubble 1p | 1 | true | false | - |
| zbv | 2 | true | true | zb |
| v-half | 2 | true | true | half |
| v-min | 2 | true | true | min |

- Speedup/Memory: *TBD*
- Details placeholder: *TODO: relate to VPP configuration*

### Feature 6 – Arbitrary Pipeline Partition

- Description: Forces an explicit 8-way pipeline layout to balance uneven layers.
- Optimization principle:
  - Pipeline parallelism is quite useful in large language model training, but also faces many challenges such as uneven memory and compute distribution, bubble overhead, and debugging complexity. To better fulfill pipeline parallelism’s potential in production-grade training, we’ve been investigating the implementation details and making corresponding improvements. Below are two aspects of progress based on Megatron-LM[1] integrated into Primus[2].
  - Developed a visualization tool for pipeline schedule, enabling intuitive visual analysis of the schedule to help easily find the performance bottleneck and optimization possibilities.
  - Enabled arbitrary pipeline split feature, offering more fine-grained control of memory and compute among different ranks to both save the overall memory and improve throughput.
- Args:
  - `--pipeline_model_parallel_size 8`
  - `# --pipeline_model_parallel_layout 'Et*3|(tt|)*29,m|L'` (to be tuned)
- Speedup: *TBD*
- Details placeholder: *TODO: supply layout design guidance*

### Feature 7 – CPU NUMA Binding Helper

- Description: Binds processes to NUMA domains on multi-socket systems, reducing HSA kernarg traffic.
- Optimization principle: *TBD*
- Args / Env:
  - `export ENABLE_NUMA_BINDING=1`
  - `# export HSA_KERNARG_POOL_SIZE=12582912` (enable as needed)
- Speedup: *TBD*
- Details placeholder: *TODO: log MI325 vs. MI355 comparisons*


### TODO
AINIC, cp, hw_queue, manual gc (stability), fused crossentropy


## 8. Code and Reproduction

### 8.1 Key Files

- `examples/moe_package/run_deepseek_v3_pretrain_mi325x.sh`: Main training script that aggregates all toggles.
- `examples/megatron/configs/MI300X/deepseek_v3-pretrain.yaml`: Experiment entry point referencing a model YAML.
- `primus/configs/models/megatron/*.yaml`: Definitive source for model size, depth, and expert settings.

### 8.2 Script Usage

1. Adjust `MoE_Features`, parallel sizes, profiling, and MTP settings as needed.
2. Prepare a ROCm/MI300X environment (Docker image referenced near the top of the script).
3. Run `bash examples/moe_package/run_deepseek_v3_pretrain_mi325x.sh`. Logs and exported configs land under `./output/<team>/<user>/<exp_name>/`.

## 9. References

- DeepSeek V2 Lite / V2 / V3 Hugging Face pages for public configs and parameter counts.
- AMD ROCm™ docs and Instinct™ MI300 performance tuning guides.
- Primus repository Megatron configs and READMEs (continuously updated best practices).
