# MLflow Patch Usage Guide

## Overview

The MLflow patch (`mlflow_patches.py`) adds MLflow logging support to Megatron-LM's training loop without modifying the Megatron source code.

## How It Works

### 1. **Patch Registration**

The patch is automatically registered when the module is imported:

```python
@register_patch(
    "megatron.mlflow.training_log",
    backend="megatron",
    phase="before_train",
    description="Add MLflow logging to Megatron training_log function",
)
def patch_training_log_for_mlflow(ctx: PatchContext):
    # Patch implementation
```

### 2. **Function Wrapping**

The patch wraps Megatron's `training_log` function:

```python
# Store original function
original_training_log = megatron_training.training_log

def patched_training_log(...):
    # Call original function
    result = original_training_log(...)

    # Add MLflow logging
    if mlflow_writer:
        mlflow_writer.log_metric("learning_rate", learning_rate, step=iteration)
        # ... more metrics ...

    return result

# Replace with patched version
megatron_training.training_log = patched_training_log
```

### 3. **Automatic Application**

The patch is automatically applied when `MegatronPretrainTrainer` runs:

```python
# In megatron_pretrain_trainer.py
def run(self):
    # Apply patches at before_train phase
    apply_megatron_patches(
        backend_version=self._detect_version(),
        model_name=getattr(self.backend_args, "model_type", "GPT"),
        phase="before_train",
        extra={"args": self.backend_args},
    )

    # Start training (patches are now active)
    pretrain(...)
```

## Logged Metrics

The patch logs the following metrics to MLflow:

### Core Metrics
- `samples_vs_steps`: Training samples vs iteration steps
- `learning_rate`: Current learning rate
- `batch_size`: Effective batch size
- `grad_norm`: Gradient norm
- `params_norm`: Parameters norm

### Loss Metrics
- All metrics from `loss_dict` (e.g., `lm_loss`, `total_loss`)

### Optional Metrics
- `decoupled_learning_rate`: If using decoupled learning rate
- `loss_scale`: If using loss scaling
- `num_zeros_in_grad`: Number of zeros in gradients
- `mem_*`: Memory statistics (if `log_memory_to_tensorboard` is enabled)

## Configuration

### Enable MLflow Logging

To enable MLflow logging, you need to:

1. **Set up MLflow tracking URI** (in your training config or environment):
   ```yaml
   # primus_config.yaml
   logger:
     mlflow:
       tracking_uri: "http://mlflow-server:5000"
       experiment_name: "megatron_training"
   ```

2. **Initialize MLflow writer** (in Primus logger setup):
   ```python
   # This is handled by Primus's logger initialization
   from primus.core.utils.logger import init_mlflow_writer

   mlflow_writer = init_mlflow_writer(config)
   ```

3. **Pass MLflow writer to trainer** (optional, for explicit control):
   ```python
   # In train_launcher.py or adapter
   apply_megatron_patches(
       phase="before_train",
       extra={
           "args": backend_args,
           "mlflow_writer": mlflow_writer,  # Optional
       },
   )
   ```

### Disable MLflow Logging

MLflow logging is automatically skipped if:
- No MLflow writer is available
- MLflow is not configured
- The patch encounters an error

To explicitly disable the patch:
```bash
export PRIMUS_PATCHES="none"  # Disable all patches
# or
export PRIMUS_PATCHES="megatron.env.cuda_device_max_connections"  # Only enable specific patches
```

## Example Output

When the patch is active, you'll see:

```
[Patch] Successfully patched training_log for MLflow support
[DEBUG] [PatchSystem] ✓ Applied patch: megatron.mlflow.training_log
```

During training, metrics are logged to MLflow at the same interval as TensorBoard:

```
Iteration: 100
  learning_rate: 0.0001
  batch_size: 512
  lm_loss: 3.45
  grad_norm: 1.23
  → Logged to MLflow
```

## Error Handling

The patch is designed to be non-intrusive:

- **If MLflow is not available**: Logging is silently skipped
- **If logging fails**: Error is printed but training continues
- **If patch fails to apply**: Warning is logged but training continues

```python
try:
    mlflow_writer.log_metric(...)
except Exception as e:
    print(f"[Patch] MLflow logging failed: {e}")
    # Training continues normally
```

## Benefits

### ✅ Non-Invasive
- No modifications to Megatron source code
- Can be enabled/disabled via environment variables
- Doesn't affect training if MLflow is not available

### ✅ Comprehensive
- Logs all important training metrics
- Matches TensorBoard logging interval
- Includes memory stats if enabled

### ✅ Maintainable
- Centralized in one patch file
- Easy to add/remove metrics
- Clear separation from training logic

### ✅ Flexible
- Works with any MLflow backend
- Can be customized via context
- Supports multiple experiments

## Advanced Usage

### Custom Metrics

To add custom metrics, modify the patch:

```python
# In mlflow_patches.py
if mlflow_writer:
    # Add your custom metric
    mlflow_writer.log_metric(
        "custom_metric",
        calculate_custom_metric(model),
        step=iteration
    )
```

### Conditional Logging

To log only specific metrics:

```python
# Check config from context
config = ctx.extra.get("config", {})
if config.get("log_detailed_metrics", False):
    # Log additional metrics
    mlflow_writer.log_metric("detailed_metric", value, step=iteration)
```

### Multiple MLflow Experiments

To log to multiple experiments:

```python
# Pass multiple writers in context
extra = {
    "mlflow_writer": primary_writer,
    "mlflow_writer_secondary": secondary_writer,
}
```

## Troubleshooting

### Patch Not Applied

Check the logs:
```bash
# Enable DEBUG logging
export LOG_LEVEL=DEBUG
python train.py ...

# Look for:
[DEBUG] [PatchSystem] Applying megatron.mlflow.training_log: ...
[INFO] [PatchSystem] ✓ Applied patch: megatron.mlflow.training_log
```

### Metrics Not Appearing in MLflow

1. Verify MLflow is configured:
   ```python
   import mlflow
   print(mlflow.get_tracking_uri())
   ```

2. Check MLflow writer is created:
   ```python
   from primus.core.utils.logger import get_mlflow_writer
   writer = get_mlflow_writer()
   print(f"MLflow writer: {writer}")
   ```

3. Verify logging interval:
   ```yaml
   # In Megatron config
   tensorboard_log_interval: 10  # Metrics logged every 10 iterations
   ```

### Performance Impact

The patch has minimal performance impact:
- Logging only happens at `tensorboard_log_interval`
- Metrics are logged asynchronously by MLflow
- Failed logging doesn't block training

## See Also

- [Patch System Documentation](../../../core/patches/patch_system.py)
- [Megatron Training Documentation](https://github.com/NVIDIA/Megatron-LM)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
