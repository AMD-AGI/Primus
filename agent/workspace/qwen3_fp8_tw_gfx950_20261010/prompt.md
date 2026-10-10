# 任务：为 Qwen3-30B-A3B FP8 tensorwise 训练优化 Primus-Turbo kernel（MI355X / gfx950）

## 背景

Primus Megatron 后端，Qwen3-30B-A3B FP8 预训练，单节点 8×MI355X。当前最好成绩：8894 ms/iter，670.6 TFLOP/s/GPU，29,474 tokens/s/GPU。

- config：`examples/megatron/configs/MI355X/qwen3_30B_A3B-FP8-pretrain.yaml`（Primus 分支 `perf/qwen3-30b-a3b-tuning`）
- 量化与 GEMM：`fp8: hybrid`，`fp8_recipe: tensorwise`，开启 `use_turbo_gemm` 和 `use_turbo_grouped_gemm`（FlyDSL grouped GEMM）
- MoE：`use_turbo_deepep`（`turbo_deepep_num_cu: 80`），`turbo_sync_free_moe_stage: 1`
- attention：TE CK v3（`NVTE_CK_USES_FWD_V3=1`，`NVTE_CK_USES_BWD_V3=1`，`NVTE_CK_IS_V3_ATOMIC_FP32=0`）
- Turbo 版本：e2d9f1d7

模型和并行形状：

- 结构：48 层，hidden 2048，32 个 Q head、4 个 KV head（GQA），head_dim 128，开启 qk_layernorm
- MoE：128 个 expert，top-8，moe_ffn_hidden 768（SwiGLU，所以 fc1 输出宽 1536）
- 并行与 batch：TP1 / PP1 / EP8，每个 rank 16 个本地 expert；seq 4096，MBS 8，GBS 512（每步 8 个 microbatch）
- 每个 microbatch、每个 rank：32768 个 token，路由后约 262k 行（even routing 下每个 expert 约 16k 行）
- attention 输入布局：sbhd，causal

## profile 结论

数据来自 rank 0 一个训练步（8 个 microbatch × 48 层），计算流的 GPU 忙碌时间占 97%。所以瓶颈完全在 kernel 本身，CPU 和 launch 不是问题。

| 类别 | 时间 | 占比 | 主要 kernel |
| --- | --- | --- | --- |
| attention | 1578 ms | 18% | `aiter::fmha_bwd_hd128_bf16_causal_a16` 1014 ms；fwd 390 ms；`ck_fused_attn::dk_dv_reduce` 81 ms |
| FP8 量化 | 1310 ms | 15% | `tensorwise_amax_partial_kernel` 252 ms（3264 次）；`quantize_tensorwise_pad_row_kernel` e5m2 220 ms + e4m3 206 ms；`transpose_2d_kernel<uint8>` 83 ms（2304 次） |
| grouped GEMM | 1267 ms | 14.5% | `grouped_nt_persistent` 452 ms；`grouped_nn_persistent` 445 ms；`grouped_tn_wgrad_4wave` 361 ms |
| DeepEP | 1248 ms | 14% | `intranode::dispatch` 591 ms；`intranode::combine` 351 ms；`layout::get_dispatch_layout` 132 ms |
| dense GEMM（hipBLASLt） | 987 ms | 11% | QKV、proj 和 lm head 的 FP8/BF16 GEMM |
| norm / rope | 529 ms | 6% | TE `fused_rope_fwd` + `fused_rope_bwd` 204 ms；rmsnorm |
| permute | 404 ms | 4.6% | `unpermute_kernel` 189 ms；`permute_kernel` 157 ms |

另外有约 1.1 s 的 elementwise 开销（SwiGLU 拆成 silu、mul、cat，再加 probs 乘法），Primus 侧打开 `use_turbo_fused_act_with_probs` 就能解决，不在本任务范围内。

## 需要优化的 kernel（按预期收益排序）

### 1. FP8 tensorwise 量化链路（目标：省 0.6 s 以上 / step）

现在每个 tensor 要先跑一遍 amax，再跑一遍 quantize（带 pad），有的还要额外跑一个 uint8 transpose 给 wgrad 用。至少 3 次读写 HBM。

- 把 amax 融合进上游 producer 的 epilogue（unpermute、rmsnorm、activation、DeepEP combine 输出），或者让 quantize 一次 kernel 同时输出 rowwise 和 colwise（transposed）的 FP8，把 `transpose_2d` 消掉。
- 先算出 `tensorwise_amax_partial` 和 `quantize_tensorwise_pad_row` 在上述形状下的实际带宽（bytes / time），对照 MI355X 的 HBM 峰值，说明现在离峰值还差多少。
- 评估 grouped GEMM 能不能直接吃 unpadded 的行（配合 Primus 的 `turbo_grouped_gemm_without_padding`），把 pad 的拷贝去掉。

### 2. FlyDSL grouped GEMM（FP8 tensorwise）

按 `fwd + dgrad + wgrad` 估算，算力利用率大约 2.2 PFLOP/s，约为 FP8 峰值的 45%。

- 涉及的形状（每个 rank 16 个 expert，每个 expert 约 16k 行）：
  - fc1：K=2048，N=1536
  - fc2：K=768，N=2048
  - 以及两者对应的 dgrad（nn）和 wgrad（tn）
- fc2 的 K=768 偏小；wgrad 的 tn 4wave kernel 尤其值得看。
- 针对这些形状调 tile / split-K / persistent 调度，并给出每个形状 before/after 的 TFLOP/s。
- 同时测 even routing 和 uniform routing 两种 expert 负载分布。

### 3. DeepEP intranode dispatch / combine

- dispatch 目前传的是 bf16 hidden（2048），评估 FP8 dispatch：先按 tensorwise e4m3 量化再传，接收端直接把 FP8 和 scale 交给 fc1。这样通信量减半，还顺带省掉 fc1 前的那次 quantize。
- `get_dispatch_layout` 有 132 ms，看能不能和 router 的 topk 融合，或者在 fwd/bwd 之间复用。
- 测 `num_cu` 在 64、80、96 时的 kernel 时间。

### 4. attention backward（CK 路径，head_dim 128，sbhd，causal，GQA 32/4）

- bwd/fwd 时间比约 2.6。分析 `dk_dv_reduce`（GQA 归约）能不能融进主 kernel。
- 对比 FlyDSL attention：之前端到端测过，开 FlyDSL 比 CK 慢约 2%。说明 FlyDSL 在这个形状下的 fwd 和 bwd 分别差在哪里。
- 注意：现有 `suachong/attn-bwd-q-cache-*` 分支只支持 head_dim 64，这里用不上。

### 5. qk_layernorm + RoPE 融合

Qwen3 有 q/k RMSNorm 后接 RoPE，现在是 TE 的 rmsnorm 加 `fused_rope` 分开跑。参考 `suachong/gptoss-fused-qk-rmsnorm-rope` 分支，给 head_dim 128 提供融合的 fwd 和 bwd。

### 6. permute / unpermute

- `unpermute_kernel<bf16, float probs>` 189 ms，`permute_kernel` 157 ms。
- 看带宽利用率；如果 unpermute 能和 probs 加权、FP8 量化融合，就和第 1 项合并处理。

## 要求

- 每一项先写 microbench，复现上面的形状和当前耗时，再动 kernel。每项都要给出 before/after 的 kernel 时间，以及带宽或 TFLOP/s 利用率。
- 正确性：
  - 新增或扩展 `tests/pytorch/ops` 下的单测，覆盖 fwd 和 bwd、even 和 uniform 负载、非对齐 token 数。
  - FP8 结果和现有实现对比，误差不能变大。
  - attention 和 GEMM 不能引入新的非确定性。
- 不能对现有路径造成回退：默认行为要么不变，要么新路径在全部现有测试下都通过。新功能可以先做成 opt-in 开关，并在 PR 里写清楚开关名。
- 不要改 MXFP4 / MXFP8 / MegaMoE 的现有行为。
- 每项优化单独一个分支、单独一个 PR，commit message 用英文，写明形状、收益和测试结果。
- 完成后告诉我需要在 Primus 侧打开什么开关或改什么接口，我来做端到端验证（基准：8894 ms/iter，even routing，MBS 8，20 iter）。

## 复现

在 Primus 仓库下运行：

```bash
./primus-cli direct -- train pretrain \
  --config examples/megatron/configs/MI355X/qwen3_30B_A3B-FP8-pretrain.yaml \
  --train_iters 14 --profile True --use_pytorch_profiler True \
  --profile_step_start 12 --profile_step_end 13 \
  --torch_profiler_with_stack False --torch_profiler_record_shapes True
```

trace 输出在 `output/amd/root/qwen3_30B_A3B-pretrain/tensorboard/*.pt.trace.json.gz`。开启 `record_shapes` 可以拿到每个 kernel 的实际输入形状。
