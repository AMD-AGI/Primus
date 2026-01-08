# MaxText 新架构实现

## 已完成的工作

参考 TorchTitan 的新架构，为 MaxText 创建了以下组件：

### 1. Backend Adapter
**文件**: `primus/backends/maxtext/maxtext_adapter.py`
- 继承 `BackendAdapter`
- 实现 `prepare_backend()`: 准备环境
- 实现 `convert_config()`: 转换配置
- 实现 `load_trainer_class()`: 加载 trainer 类
- 实现 `detect_backend_version()`: 检测版本

### 2. Argument Builder
**文件**: `primus/backends/maxtext/argument_builder.py`
- `MaxTextConfigBuilder` 类
- 将 Primus config 转换为 MaxText 配置格式

### 3. Base Trainer
**文件**: `primus/backends/maxtext/maxtext_base_trainer.py`
- 继承 `BaseTrainer`
- 提供 MaxText 通用的初始化逻辑
- 版本检测
- 统一的日志输出

### 4. Pretrain Trainer
**文件**: `primus/backends/maxtext/maxtext_pretrain_trainer.py`
- 继承 `MaxTextBaseTrainer`
- 实现 `setup()`: 准备阶段
- 实现 `init()`: 初始化 MaxText 训练组件
- 实现 `run_train()`: 执行训练
- 实现 `prepare_model_overrides()`: 处理模型参数覆盖

### 5. Patches 系统
**目录**: `primus/backends/maxtext/patches/`
- `patches/__init__.py`: 导入所有 patches
- `patches/logger_patches.py`: Logger 重定向 patch

### 6. Backend 注册
**文件**: `primus/backends/maxtext/__init__.py`
- 注册 `MaxTextAdapter`
- 注册 `MaxTextPretrainTrainer`

## 架构对比

### 旧架构 (modules/trainer/maxtext/)
```
primus/modules/trainer/maxtext/
  └── pre_trainer.py  (单一大文件，包含所有逻辑)
```

### 新架构 (backends/maxtext/)
```
primus/backends/maxtext/
  ├── __init__.py                    # 注册 backend
  ├── maxtext_adapter.py             # Backend adapter
  ├── argument_builder.py            # 配置构建器
  ├── maxtext_base_trainer.py        # 基础 trainer
  ├── maxtext_pretrain_trainer.py    # 预训练 trainer
  └── patches/                       # Patches 目录
      ├── __init__.py
      └── logger_patches.py
```

## 下一步工作

### 必需任务

1. **创建剩余的 Patches**
   - [ ] `patches/wandb_patches.py` - WANDB 配置
   - [ ] `patches/checkpoint_patches.py` - Checkpoint 相关
   - [ ] `patches/max_utils_patches.py` - Max utils 相关
   - [ ] `patches/input_pipeline_patches.py` - 数据管道相关
   - [ ] `patches/layer_patches.py` - Layer 相关 (quantization, attention, moe)

2. **完善 Argument Builder**
   - 当前是简单的 SimpleNamespace 转换
   - 需要根据 MaxText 的实际配置格式调整
   - 可能需要注入分布式环境变量

3. **迁移剩余的 Patching 逻辑**
   从 `primus/modules/trainer/maxtext/pre_trainer.py` 迁移：
   - `patch_max_utils()` → `patches/max_utils_patches.py`
   - `patch_checkpoint()` → `patches/checkpoint_patches.py`
   - `patch_input_pipeline()` → `patches/input_pipeline_patches.py`
   - `patch_wandb()` → `patches/wandb_patches.py`
   - `patch_layers()` → `patches/layer_patches.py`

4. **更新 BaseTrainer 的 init**
   - 在 `MaxTextBaseTrainer.__init__` 中调用 `run_patches`
   - 确保 patches 在正确的时机执行

5. **测试和验证**
   - 确保新架构下 MaxText 训练正常运行
   - 验证所有 patches 正确应用
   - 对比新旧架构的训练结果

### 可选优化

- [ ] 添加更多的配置验证
- [ ] 改进错误处理和日志输出
- [ ] 添加单元测试

## 关键区别

### 旧架构
- 所有逻辑集中在 `pre_trainer.py`
- Patches 在 `__init__` 中直接调用
- 配置处理混杂在训练代码中

### 新架构
- 关注点分离：adapter / trainer / patches
- Patches 通过装饰器注册，统一管理
- 配置转换独立到 `argument_builder`
- 遵循 TorchTitan / Megatron 的统一模式

## 使用方式

新架构下，MaxText 的使用方式不变：
```bash
primus-cli direct -- train pretrain --config config.yaml
```

Backend 会自动：
1. 通过 `MaxTextAdapter` 加载
2. 应用所有注册的 patches
3. 实例化 `MaxTextPretrainTrainer`
4. 执行训练流程

## 参考

- TorchTitan 新架构: `primus/backends/torchtitan/`
- Megatron 新架构: `primus/backends/megatron/`
- Patch 系统文档: `primus/core/patches/`
