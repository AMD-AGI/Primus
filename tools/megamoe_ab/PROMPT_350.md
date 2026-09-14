# MI355X 350 节点 — MegaMoE 2×2 对比 Prompt

复制下面「---」之后到文件末尾的整段，粘贴到 350 机器 Cursor Agent 即可。

---

在 MI355X 350 节点上继续 DeepSeek-V3 MegaMoE 2×2 性能对比实验。

## 环境

- **本机直接跑**，不用 SSH 到其他节点
- 容器：**xiaoming-dev**（本地 docker，`LOCAL=1`）
- 代码：`/perf_apps/xiaoming/Primus` 分支 `feat/megamoe-2x2-comparison`
- MegaMoE：`/perf_apps/xiaoming/MegaMoE`（bind mount，`PRIMUS_MEGAMOE_SRC` 已在 run.sh 设好）

## 先做环境检查

1. `cd /perf_apps/xiaoming/Primus && git fetch origin && git checkout feat/megamoe-2x2-comparison && git pull`
2. `docker ps | grep xiaoming-dev` — 容器在跑
3. bind mount 正常：`/perf_apps/xiaoming/Primus` 和 `/perf_apps/xiaoming/MegaMoE` 在容器内可见
4. wgrad fix：`primus/backends/megatron/core/extensions/primus_turbo.py` 里 non-fused wgrad 是 unconditional add（不是 `elif not weight.grad_added_to_main_grad`）
5. FP8 yaml：`examples/megatron/configs/MI355X/deepseek_v3-FP8-pretrain.yaml` 含 `fp8_recipe: mxfp8`、`use_turbo_grouped_gemm: true`、`moe_use_legacy_grouped_gemm: false`

## 背景

四个 arm（1 node × 8 GPU，4 layer，mock data，50 iter，GBS=512，EP=8）：

- bf16 · MegaMoE vs bf16 · baseline
- mxfp8 · MegaMoE vs mxfp8 · baseline

脚本说明：`tools/megamoe_ab/README.md`

## 运行顺序

所有命令在 **350 本机** `/perf_apps/xiaoming/Primus` 下执行，固定 `LOCAL=1 CONTAINER=xiaoming-dev`。

### Step 1 — smoke test（约 5 分钟）

```bash
cd /perf_apps/xiaoming/Primus
LOCAL=1 CONTAINER=xiaoming-dev \
  TRAIN_ITERS=3 ONLY="mxfp8 False" ./run_ab_matrix.sh
```

mxfp8 baseline 能跑完、loss 不是 ~8.5（wgrad bug 未修时会飙到 8.x）。

### Step 2 — 完整性能 2×2（DeepEP on，约 40 分钟）

```bash
LOCAL=1 CONTAINER=xiaoming-dev ./run_ab_matrix.sh
```

arm 顺序：mxfp8 baseline → bf16 mega → mxfp8 mega → bf16 baseline。不要并行（run.sh 会 pkill python）。

### Step 3 — 汇总

```bash
OUT=ab_2x2/<最新时间戳目录>
python3 tools/parse_ab_matrix.py "$OUT" | tee "$OUT/summary.md"
python3 tools/plot_ab_loss.py "$OUT"
```

把 summary.md 和 loss_curves.png 结果贴回来。

## 成功标准

- 四个 log 都存在且无 crash
- 吞吐（iter > 20）：mxfp8 / bf16 各自 mega vs baseline speedup 约 1.08–1.10×
- mxfp8 baseline loss@50 约 5.x（不是 8.x）
- launch.txt 记录 local=1、容器镜像、primus/megamoe git SHA

## 可选：精度对比（可复现 loss）

```bash
COMMON_EXTRA="--turbo_sync_free_moe_stage 0 --use_turbo_deepep False" \
  LOCAL=1 CONTAINER=xiaoming-dev ./run_ab_matrix.sh
```

DeepEP 开着时 loss gap 不可信，性能对比只看 ms/iter。

## 约束

- 只改脚本/配置如有必要
- log 放 `ab_2x2/`，不要 commit
- 出问题查 `$OUT/*.outer.log` 和 arm log 末尾
