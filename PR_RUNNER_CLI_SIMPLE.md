# Runner CLI - Initial Integration

## 概述

首次将重构后的 Primus Runner CLI 合入 main 分支。该 CLI 提供了统一的命令行接口，支持容器、Slurm 和直接执行三种运行模式。

## 主要功能

### 三种运行模式
- **Container**: 容器化执行（支持 Docker/Podman）
- **Slurm**: 集群调度执行（支持 srun/sbatch）
- **Direct**: 直接执行（支持单进程/多进程）

### 核心特性
- ✅ 统一的配置管理（支持 YAML 配置文件）
- ✅ 灵活的参数传递（CLI 参数覆盖配置文件）
- ✅ Dry-run 模式（预览执行命令）
- ✅ Debug 模式（详细日志输出）
- ✅ 完善的验证和错误提示

## 使用示例

### 容器模式
```bash
primus-cli container --image rocm/primus:latest -- train
```

### Slurm 模式
```bash
primus-cli slurm -N 4 -p gpu -- container -- train
```

### 直接模式
```bash
primus-cli direct -- train
```

### 使用配置文件
```bash
primus-cli --config .primus.yaml container -- train
```

## 文件结构

```
runner/
├── primus-cli                    # 主入口脚本
├── primus-cli-container.sh       # 容器模式
├── primus-cli-slurm.sh          # Slurm 模式
├── primus-cli-slurm-entry.sh    # Slurm 入口脚本
├── primus-cli-direct.sh         # 直接模式
├── .primus.yaml                 # 默认配置示例
├── lib/
│   ├── common.sh                # 通用函数库
│   ├── config.sh                # 配置管理
│   └── validation.sh            # 参数验证
└── helpers/
    ├── detect_gpu_model.sh      # GPU 检测
    └── envs/                    # 环境配置

tests/runner/
├── test_primus_cli.sh           # CLI 主逻辑测试
├── test_primus_cli_container.sh # 容器模式测试
├── test_primus_cli_slurm.sh     # Slurm 模式测试
├── test_primus_cli_direct.sh    # 直接模式测试
└── lib/
    ├── test_common.sh           # 通用函数测试
    ├── test_config.sh           # 配置管理测试
    └── test_validation.sh       # 验证函数测试
```

## 测试情况

- ✅ **8/10** 测试套件通过
- ✅ **317** 个单元测试通过
- ⚠️ 2 个测试套件因环境限制失败（需要 Docker/Podman）

## 变更统计

- **新增**: 5 个文件（direct 模式、测试、配置）
- **修改**: 13 个文件（scripts、库、测试）
- **删除**: 1 个文件（弃用的 entrypoint）
- **代码行数**: +3,357 / -818

## 注意事项

### Slurm 参数格式
只支持 `--key value` 格式，不支持 `--key=value`

✅ 正确：`primus-cli slurm --nodes 4`
❌ 错误：`primus-cli slurm --nodes=4`

### 配置文件
布尔值使用 `true`/`false`，不使用 `0`/`1`

```yaml
# 推荐
debug: true
dry_run: false

# 不推荐
debug: 1
dry_run: 0
```

## 后续计划

- [ ] 添加更多使用文档和示例
- [ ] 完善错误提示信息
- [ ] 支持更多自定义选项
- [ ] 优化性能和用户体验

---

**这是 Runner CLI 的首次合入，标志着 Primus 命令行工具的正式启用。**
