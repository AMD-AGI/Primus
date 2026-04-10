# DeepSeek V3 Training Configuration (English)

## 1) Launch Script

The launch entry for this setup is:

- `start_training_dsv3.sh`

Typical usage:

```bash
bash start_training_dsv3.sh
```

---

## 2) Training Config File

The script selects the pretrain config through:

- `EXP=examples/megatron/configs/MI355X/deepseek_v3-${PRETRAIN_TYPE}-pretrain.yaml`

Common files:

- `examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml`
- `examples/megatron/configs/MI355X/deepseek_v3-FP8-pretrain.yaml`

What this file controls:

- Experiment metadata (`work_group`, `user_name`, `exp_name`, `workspace`)
- Model file binding via:
  - `model: ${PRIMUS_MODEL:deepseek_v3}.yaml`

---

## 3) Model Config File

Model config path:

- `primus/configs/models/megatron/deepseek_v3.yaml`

Base file:

- `primus/configs/models/megatron/deepseek_v3_base.yaml`

Key model properties in `deepseek_v3.yaml`:

- `num_layers: 61`
- `hidden_size: 7168`
- `ffn_hidden_size: 18432`
- `num_attention_heads: 128`
- MLA settings (`q_lora_rank`, `kv_lora_rank`, head dims)
- MoE settings (`num_experts: 256`, `moe_router_topk: 8`, routing group params)

---

## 5) DSV3 Training Parameters (from `start_training_dsv3.sh`)

### Environment Variables

| Parameter | Value | Note |
|---|---|---|
| `HF_TOKEN` | `${HF_TOKEN:-your_hf_token}` | HuggingFace auth token for model/tokenizer access. |
| `WANDB_API_KEY` | `${WANDB_API_KEY:-your_wandb_api_key}` | Weights & Biases auth token (even if logging is disabled later). |
| `NNODES` | `${NNODES:-32}` | Total node count for distributed run. |
| `USING_AINIC` | `1` | Enables AINIC-related network path behavior. |
| `NCCL_IB_HCA` | `ionic_0:1,...,ionic_9:1` | Selects RDMA HCAs for NCCL. |
| `GLOO_SOCKET_IFNAME` | `ens9np0` | Gloo communication interface. |
| `NCCL_SOCKET_IFNAME` | `ens9np0` | NCCL socket interface. |
| `ENABLE_NUMA_BINDING` | `1` | Enables NUMA binding for memory locality. |
| `HSA_KERNARG_POOL_SIZE` | `12582912` | HSA kernel arg pool tuning for large runs. |

### Training Launch Arguments

| Parameter | Value | Note |
|---|---|---|
| `--num_layers` | `61` | Effective transformer layer count. |
| `--micro_batch_size` | `2` | Default micro batch size (`MBS`). |
| `--global_batch_size` | `128 * NNODES` (default `4096`) | Global batch size formula; default uses `NNODES=32`. |
| `--use_turbo_deepep` | `True` | Default from `TURBO_DEEPEEP`. |
| `--turbo_deepep_num_cu` | `80` | Number of compute units for turbo deepep path. |
| `--use_turbo_rms_norm` | `True` | Default from `TURBO_RMS_NORM`. |
| `--moe_use_legacy_grouped_gemm` | `True` | Default from `LEGACY_GG`. |
| `--enable_experimental` | `True` | Default from `APPLY_ROPE_FUSION`. |
| `--apply_rope_fusion` | `True` | Default from `APPLY_ROPE_FUSION`. |
| `--pipeline_model_parallel_size` | `16` | Default from `PRIMUS_PP`. |
| `--expert_model_parallel_size` | `8` | Default from `PRIMUS_EP`. |
| `--pipeline_model_parallel_layout` | `Et*1\|t*1\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*2\|t*1,L` | Required layout for `PP16_VPP2` tuning profile. |
| `--recompute_layer_ids` | `RECOMP_IDS` best values for different Node Number:<br>`32N_PP16_VPP2`: `0,1,2,4,6,8,10,12,14,16,34,36,38,40,50`<br>`64N_PP16_VPP2`: `0,1,2,4,6,8,10,12,14,16,34,36`<br>`128N_PP16_VPP2`: `0,1,2,4,6,8,10,12,14` | Recommended explicit recompute layers by node count. |
| `--cross_entropy_fusion_impl` | `te` | Fused TE backend selection. |
| `--cross_entropy_loss_fusion` | `True` | Enables TE fusion for performance. |
| `--manual_gc` | `True` | Enables manual garbage collection control. |
| `--manual_gc_interval` | `1` | GC frequency when manual GC is enabled. |
| `--use_precision_aware_optimizer` | `True` | Enables precision-aware optimizer behavior. |
| `--main_grads_dtype` | `bf16` | Gradient dtype. |
| `--exp_avg_dtype` | `bf16` | Adam exp_avg dtype. |
| `--exp_avg_sq_dtype` | `bf16` | Adam exp_avg_sq dtype. |
