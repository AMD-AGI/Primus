# GPU Model-Specific Environment Configurations

本目录包含针对不同 AMD GPU 型号的优化配置文件。

## 📋 目录结构

```
envs/
├── README.md          # 本文档
├── MI300X.sh          # AMD MI300X 配置
├── MI300A.sh          # AMD MI300A 配置
├── MI250X.sh          # AMD MI250X 配置
└── default.sh         # 默认配置（未知型号）
```

## 🔧 工作原理

### 1. 自动检测流程

```bash
# 步骤 1: 使用 rocm-smi 检测 GPU 型号
rocm-smi --showproductname

# 步骤 2: 提取型号信息（如 MI300X, MI250X）
GPU_MODEL=$(detect_gpu_model.sh)

# 步骤 3: 加载对应的配置文件
source helpers/envs/${GPU_MODEL}.sh
```

### 2. 配置加载优先级

1. **精确匹配**：如果存在 `envs/${GPU_MODEL}.sh`，直接加载
2. **前缀匹配**：
   - `MI300*` → `MI300X.sh`
   - `MI250*` → `MI250X.sh`
3. **降级到默认**：如果没有匹配项，使用 `default.sh`

### 3. 手动指定 GPU 型号

```bash
# 方式 1: 环境变量
export GPU_MODEL=MI300X
./runner/primus-cli-entrypoint.sh -- train pretrain

# 方式 2: 内联
GPU_MODEL=MI250X ./runner/primus-cli-entrypoint.sh -- benchmark gemm
```

## 🎯 支持的 GPU 型号

### AMD MI300X
**文件**: `MI300X.sh`

**特点**:
- 192GB HBM3 内存
- CDNA 3 架构
- 支持 FP8 训练

**关键配置**:
```bash
HSA_ENABLE_SDMA=1                    # 启用 SDMA
HSA_NO_SCRATCH_RECLAIM=0             # 禁用 scratch 回收
HSA_XNACK=0                          # 禁用 XNACK
GPU_MAX_HEAP_SIZE=100                # 100% GPU 内存
NVTE_USE_CAST_TRANSPOSE_TRITON=1     # 使用 Triton cast transpose
```

### AMD MI300A
**文件**: `MI300A.sh`

**特点**:
- 128GB 统一内存 (HBM + DDR)
- APU 架构（集成 CPU + GPU）
- CDNA 3 架构

**关键配置**:
```bash
HSA_ENABLE_INTERRUPT=1               # APU 中断模式
HSA_XNACK=1                          # 启用 XNACK (统一内存)
GPU_MAX_HEAP_SIZE=100
```

### AMD MI250X
**文件**: `MI250X.sh`

**特点**:
- 128GB HBM2e (64GB x 2 GCD)
- CDNA 2 架构
- 双 GCD 设计

**关键配置**:
```bash
HSA_NO_SCRATCH_RECLAIM=1             # 启用 scratch 回收
RCCL_MSCCLPP_THRESHOLD=512MB         # 较小的阈值
NVTE_USE_CAST_TRANSPOSE_TRITON=0     # 禁用 Triton
NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=1  # 使用优化版本
PATCH_TE_FLASH_ATTN=1                # 启用 Flash Attention 补丁
GPU_MAX_HEAP_SIZE=90                 # 保守的内存设置
```

### 默认配置
**文件**: `default.sh`

用于未知或不支持的 GPU 型号，使用保守的设置。

## ➕ 添加新的 GPU 型号

### 步骤 1: 创建配置文件

```bash
# 例如：添加 MI355 支持
cd runner/helpers/envs
cp MI300X.sh MI355.sh
```

### 步骤 2: 自定义配置

编辑 `MI355.sh`，根据 GPU 特性调整参数：

```bash
#!/bin/bash
###############################################################################
# AMD MI355 GPU-specific optimizations
###############################################################################

LOG_INFO_RANK0 "Loading MI355-specific optimizations..."

# MI355 特定配置
export HSA_ENABLE_SDMA=1
export HSA_NO_SCRATCH_RECLAIM=0

# 根据 MI355 的特性调整
export GPU_MAX_HEAP_SIZE=100
export RCCL_MSCCLPP_THRESHOLD=$((2*1024*1024*1024))  # 2GB

# Transformer Engine 设置
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_CK_USES_BWD_V3=1  # 如果 MI355 支持 v3

# 其他 MI355 特定优化...

log_exported_vars "MI355-specific optimizations" \
    HSA_ENABLE_SDMA GPU_MAX_HEAP_SIZE NVTE_USE_CAST_TRANSPOSE_TRITON
```

### 步骤 3: 测试配置

```bash
# 手动指定型号测试
GPU_MODEL=MI355 ./runner/primus-cli-entrypoint.sh -- preflight --check-all

# 检查日志输出
# 应该看到: "Detected GPU model: MI355"
#         "Loading GPU configuration: .../envs/MI355.sh"
```

### 步骤 4: 更新检测脚本（可选）

如果 `rocm-smi` 输出格式不同，可能需要更新 `detect_gpu_model.sh`：

```bash
# 编辑 helpers/detect_gpu_model.sh
# 添加针对新型号的检测逻辑
```

## 🔍 调试和验证

### 查看检测到的 GPU 型号

```bash
bash runner/helpers/detect_gpu_model.sh
# 输出: MI300X
```

### 查看加载的配置

```bash
# 启用详细日志
export NODE_RANK=0
source runner/helpers/primus-env.sh

# 输出会显示:
# [NODE-0(hostname)] Detected GPU model: MI300X
# [NODE-0(hostname)] Loading GPU configuration: .../envs/MI300X.sh
# [NODE-0(hostname)] ========== MI300X-specific optimizations ==========
# ...
```

### 验证环境变量

```bash
# 加载环境后检查变量
source runner/helpers/primus-env.sh
env | grep -E "HSA_|RCCL_|NVTE_|GPU_"
```

## 📊 配置对比

| 配置项 | MI300X | MI300A | MI250X | 说明 |
|--------|--------|--------|--------|------|
| `HSA_XNACK` | 0 | 1 | 0 | MI300A 需要启用（统一内存） |
| `HSA_NO_SCRATCH_RECLAIM` | 0 | 0 | 1 | MI250X 需要启用 |
| `GPU_MAX_HEAP_SIZE` | 100 | 100 | 90 | MI250X 更保守 |
| `NVTE_USE_CAST_TRANSPOSE_TRITON` | 1 | 1 | 0 | MI250X 使用优化版本 |
| `PATCH_TE_FLASH_ATTN` | 0 | 0 | 1 | MI250X 需要补丁 |
| `RCCL_MSCCLPP_THRESHOLD` | 1GB | 1GB | 512MB | MI250X 较小 |

## 💡 最佳实践

### 1. 为每个 GPU 型号测试配置

```bash
# 运行环境检查
primus preflight --check-all

# 运行小规模训练测试
primus train pretrain --config test_config.yaml

# 运行基准测试
primus benchmark gemm
```

### 2. 记录性能基线

为每个 GPU 型号记录：
- 训练吞吐量
- 内存使用
- 通信效率
- 数值稳定性

### 3. 逐步调优

1. 从默认配置开始
2. 根据性能分析结果调整参数
3. 验证数值正确性
4. 记录最优配置

### 4. 版本控制

```bash
# 在配置文件中记录版本信息
# MI300X.sh

# Configuration version: v1.2
# Last updated: 2025-11-05
# Tested with: ROCm 6.0, PyTorch 2.1
# Performance: 1500 tokens/s on Llama-70B
```

## 🐛 常见问题

### Q1: 检测不到 GPU 型号
```bash
# 手动运行检测脚本查看错误
bash runner/helpers/detect_gpu_model.sh

# 手动指定型号
export GPU_MODEL=MI300X
```

### Q2: 配置文件不生效
```bash
# 检查文件路径
ls -la runner/helpers/envs/

# 检查文件权限
chmod +x runner/helpers/envs/*.sh

# 查看加载日志
export NODE_RANK=0
source runner/helpers/primus-env.sh 2>&1 | grep "Loading GPU configuration"
```

### Q3: 性能不如预期
```bash
# 启用调试模式
export NCCL_DEBUG=INFO
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2

# 运行性能分析
primus benchmark gemm -v
```

## 📚 参考资源

- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [RCCL Tuning Guide](https://github.com/ROCmSoftwarePlatform/rccl)
- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)

## 🔄 更新日志

- **2025-11-05**: 初始版本，支持 MI300X, MI300A, MI250X
- 待添加: MI355, MI200 系列其他型号

---

**维护者**: Primus Team
**最后更新**: 2025-11-05
