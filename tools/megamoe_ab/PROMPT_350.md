# MI355X 350 节点 — MegaMoE 2×2 对比 Prompt

复制下面 ``` 之间的整段，粘贴到 350 机器 Cursor Agent 即可。

---

在 MI355X 节点上继续 DeepSeek-V3 MegaMoE 2×2 性能对比实验。

## 背景

四个 arm 对比（1 node × 8 GPU，4 layer，mock data，50 iter，GBS=512，EP=8）：

- bf16 · MegaMoE vs bf16 · baseline
- mxfp8 · MegaMoE vs mxfp8 · baseline

脚本已在 Primus 分支 `feat/megamoe-2x2-comparison`，说明见 `tools/megamoe_ab/README.md`。

## 先做环境检查

1. `cd /perf_apps/xiaoming/Primus && git fetch origin && git checkout feat/megamoe-2x2-comparison`
2. 确认 bind mount：
   - `/perf_apps/xiaoming/Primus`
   - `/perf_apps/xiaoming/MegaMoE`（`PRIMUS_MEGAMOE_SRC` 已在 run.sh 里设）
3. 确认容器 `xiaoming-dev` 在跑，镜像和 turbo/MegaMoE git SHA 正常
4. 确认 wgrad fix 已生效：检查 `primus/backends/megatron/core/extensions/primus_turbo.py` 里 non-fused wgrad 路径是 unconditional add（不是 `elif not weight.grad_added_to_main_grad`）。没有的话 cherry-pick `fix/turbo-wgrad-accum-every-microbatch`
5. 确认 FP8 yaml 已是 mxfp8：`examples/megatron/configs/MI355X/deepseek_v3-FP8-pretrain.yaml` 含 `fp8_recipe: mxfp8`、`use_turbo_grouped_gemm: true`、`moe_use_legacy_grouped_gemm: false`

## 运行顺序

### Step 1 — smoke test（约 5 分钟）

```bash
cd /perf_apps/xiaoming/Primus
NODE=$(hostname -s) CONTAINER=xiaoming-dev \
  TRAIN_ITERS=3 ONLY="mxfp8 False" ./run_ab_matrix.sh
```

检查 mxfp8 baseline 能跑完、loss 不是 ~8.5（那是 wgrad bug）。

### Step 2 — 完整性能 2×2（默认 DeepEP on，约 40 分钟）

```bash
NODE=$(hostname -s) CONTAINER=xiaoming-dev ./run_ab_matrix.sh
```

四个 arm 顺序：mxfp8 baseline → bf16 mega → mxfp8 mega → bf16 baseline。
不要并行跑（run.sh 会 pkill python）。

### Step 3 — 汇总

```bash
OUT=ab_2x2/<最新时间戳目录>
python3 tools/parse_ab_matrix.py "$OUT" | tee "$OUT/summary.md"
python3 tools/plot_ab_loss.py "$OUT"
```

把 summary.md 和 loss_curves.png 结果贴回来。

## 成功标准

- 四个 log 都存在且无 crash
- 吞吐（iter > 20）：mxfp8 和 bf16 各自 mega vs baseline 的 speedup 约 1.08–1.10×（参考 n04 结果）
- mxfp8 baseline loss@50 约 5.x（不是 8.x）
- launch.txt 里记录了 node、image、primus/megamoe git SHA

## 可选：精度对比（若需要可复现 loss）

```bash
COMMON_EXTRA="--turbo_sync_free_moe_stage 0 --use_turbo_deepep False" \
  NODE=$(hostname -s) CONTAINER=xiaoming-dev ./run_ab_matrix.sh
```

DeepEP 开着时 loss gap 不可信，只看 ms/iter。

## 约束

- 只改脚本/配置如有必要，不要改 unrelated 代码
- 实验 log 放 `ab_2x2/`，不要 commit log 文件
- 遇到问题先查 outer.log 和 arm log 末尾 stack trace

## 备注

- 若从别的机器 SSH 到 350 跑，`NODE=$(hostname -s)` 改成实际 hostname（如 `smci355-ccs-aus-n04-XX`），需在能 `ssh NODE` + `docker exec` 的机器上执行 `run_ab_matrix.sh`
- 若就在 350 本机且容器也在本机，可改用 local-only 脚本（见 README）
