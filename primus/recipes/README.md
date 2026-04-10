# Custom Megatron-Bridge Recipes

This directory contains custom recipe configurations for Megatron-Bridge training that are specific to Primus.

## Why Custom Recipes?

Custom recipes allow you to:
1. **Modify training configurations** without changing Megatron-Bridge source code
2. **Add custom target modules** for LoRA (e.g., adding `linear_fc1` and `linear_fc2`)
3. **Customize hyperparameters** and training settings
4. **Version control your recipes** alongside your Primus code

## How to Use Custom Recipes

### 1. YAML Configuration

In your model YAML file (e.g., `primus/configs/models/megatron_bridge/llama2_70b.yaml`), specify the custom recipe:

```yaml
# Use custom recipe from Primus
recipe: primus.recipes.llama2_custom
flavor: llama2_70b_pretrain_config

# Other configuration...
hf_path: meta-llama/Llama-2-70b-hf
train_data_path: [/data/]
valid_data_path: [/data/]
```

### 2. Recipe Format

The `recipe` field can be specified in two formats:

#### Standard Megatron-Bridge Recipe (original behavior)
```yaml
recipe: llama.llama2  # Expands to: megatron.bridge.recipes.llama.llama2
flavor: llama2_70b_pretrain_config
```

#### Custom Recipe (new capability)
```yaml
recipe: primus.recipes.llama2_custom  # Custom module path
flavor: llama2_70b_pretrain_config
```

### 3. Recipe Loading Logic

The system automatically detects whether you're using a standard or custom recipe:

1. **Custom recipe detection**: Recipes starting with `primus.recipes.` or `primus.` are treated as custom
2. **Standard recipe**: Other recipes are prefixed with `megatron.bridge.recipes.`
3. **Fallback**: If standard import fails, tries importing as a custom module

## Available Custom Recipes

### `llama2_custom.py`

Custom Llama2 recipe based on Megatron-Bridge's `llama2.py` with modifications:

**Key differences from standard recipe:**
- Added `linear_fc1` and `linear_fc2` to LoRA target modules
- Can be easily modified for Primus-specific needs

**Available flavors:**
- `llama2_70b_pretrain_config`: Configuration for Llama-2 70B pre-training/fine-tuning

## Creating New Custom Recipes

To create a new custom recipe:

1. **Create a new Python file** in this directory (e.g., `my_custom_recipe.py`)
2. **Define flavor functions** that return `ConfigContainer` objects
3. **Update your YAML** to use the new recipe:
   ```yaml
   recipe: primus.recipes.my_custom_recipe
   flavor: my_custom_flavor
   ```

### Example Recipe Structure

```python
from megatron.bridge.training.config import ConfigContainer

def my_custom_flavor(**kwargs) -> ConfigContainer:
    """
    Custom recipe flavor.
    
    Args:
        **kwargs: Configuration parameters from YAML and overrides
    
    Returns:
        ConfigContainer: Megatron-Bridge configuration
    """
    # Your custom configuration logic here
    return cfg
```

## Benefits

1. **No Megatron-Bridge modifications**: Keep upstream code pristine
2. **Easy version control**: Custom recipes live in Primus repository
3. **Flexible experimentation**: Quickly test different configurations
4. **Team collaboration**: Share custom recipes across team members
5. **Backward compatible**: Standard Megatron-Bridge recipes still work

## See Also

- Standard Megatron-Bridge recipes: `third_party/Megatron-Bridge/src/megatron/bridge/recipes/`
- Configuration utilities: `primus/backends/megatron_bridge/config_utils.py`
- Example usage: `examples/megatron_bridge/configs/MI355X/llama2_70b_lora_posttrain.yaml`
