<img width="1024" height="468" alt="image" src="https://github.com/user-attachments/assets/f1b2bf61-d612-4e62-bac4-ac115928632a" />



An interactive bash script for automated benchmarking of LLMs on AMD GPUs (MI300X/MI355X) using Megatron or TorchTitan backends supported through Primus.

---

## 🚀 Quick Start

### Step 1: Pull and Launch the Container

```bash
docker pull YOUR_IMAGE
docker run -it \
  --device /dev/dri \
  --device /dev/kfd \
  --network host \
  --ipc host \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v $HOME/.ssh:/root/.ssh \
  --name IMAGE_NAME \
  YOUR_IMAGE
```

### Step 2: Navigate to Primus Directory

```bash
cd /workspace/Primus
```

### Step 3: Run the Benchmarking Tool

```bash
bash run_primus_auto_benchmarking_tool.sh
```

---

## 📋 Features

- ✅ **Interactive Menu System** - User-friendly CLI with color-coded outputs
- ✅ **Multi-Backend Support** - Compatible with Megatron and TorchTitan
- ✅ **Batch Processing** - Run multiple model configurations sequentially
- ✅ **Configuration Editing** - Edit YAML configs before execution
- ✅ **Parameter Overrides** - Override specific parameters without editing files
- ✅ **Auto Device Detection** - Automatically detects AMD MI300X/MI355X GPUs
- ✅ **Comprehensive Logging** - Timestamped logs for each benchmark run
- ✅ **Environment Management** - Custom environment variable support

---

## 📖 Complete Walkthrough

### 1️⃣ Backend Selection

When you launch the tool, you'll first choose the backend framework:

```
★ Choose Backend:
  ● 1) megatron
  ● 2) torchtitan

➜ Enter number or name:
```

**Options:**
- Enter `1` or `megatron` for Megatron backend
- Enter `2` or `torchtitan` for TorchTitan backend

---

### 2️⃣ Model Configuration Selection

The tool scans for available YAML configuration files in the selected backend directory:

```
★ Available Model Configs: (megatron)
  ● 1) llama3_8b.yaml
  ● 2) llama3_70b.yaml
  ● 3) gpt3_175b.yaml

➜ Select config number(s) (comma-separated, range, or 'all'):
(Examples: 1,3,5 or 4-8 or all)
```

**Selection Options:**
- **Single:** `1` - Select one config
- **Multiple:** `1,3,5` - Select specific configs (comma-separated)
- **Range:** `4-8` - Select a range of configs
- **All:** `all` - Select all available configs

---

### 3️⃣ View Configuration Parameters

Option to preview parameters in your selected configurations:

```
★ View Configuration Parameters?
➜ (y/n):
```

If you choose `y`, the tool displays the contents of each selected YAML file (excluding comments and empty lines).

---

### 4️⃣ Edit Configuration Files

**For Multiple Configs:**
```
★ Edit any configuration files before running?
➜ (y/n):
```

If `y`, you can select which configs to edit:
```
Selected models:
  ● 1) llama3_8b.yaml
  ● 2) llama3_70b.yaml

● Enter model numbers to edit (comma-separated, or 'all'):
➜
```

**For Single Config:**
```
★ Edit configuration file before running?
➜ (y/n):
```

The tool opens the config in your default editor (tries `nano`, `vim`, `vi`, `code`, or `$EDITOR`). Edit, save, and close to continue. Edited configs are saved to the `logs/` directory.

---

### 5️⃣ Override Parameters

Override specific parameters without editing the entire file:

```
★ Override any parameters?
  (Format: key=value, e.g., batch_size=32)
➜ (y/n):
```

If `y`, enter overrides one per line:
```
➜ Override (or press Enter to finish): batch_size=32
✓ Will override: batch_size = 32
➜ Override (or press Enter to finish): learning_rate=0.001
✓ Will override: learning_rate = 0.001
➜ Override (or press Enter to finish): [Press Enter]

✓ 2 parameter(s) will be overridden
```

---

### 6️⃣ Device Detection

The tool automatically detects your AMD GPU:

```
★ Detecting Device...
  ● Device found: MI300X
✓ GPU Device: MI300X
```

**Auto-detection methods:**
1. Queries `rocminfo` for "AMD Instinct" devices
2. Falls back to architecture detection (gfx942 → MI300X, gfx950 → MI355X)

**Manual Selection (if auto-detection fails):**
```
✗ Could not detect device automatically
★ Please select Device manually:
  ● 1) MI300X
  ● 2) MI355X

➜ Enter number or name:
```

---

### 7️⃣ Device-Specific Environment Variables

Add custom environment variables for your device:

```
★ Add device-specific environment variables for MI300X?
  (e.g., HSA_OVERRIDE_GFX_VERSION=11.0.0)
➜ (y/n):
```

If `y`, enter variables one per line:
```
➜ Variable (or press Enter to finish): HSA_OVERRIDE_GFX_VERSION=11.0.0
✓ Will set: HSA_OVERRIDE_GFX_VERSION=11.0.0
➜ Variable (or press Enter to finish): [Press Enter]

✓ 1 environment variable(s) will be set
```

---

### 8️⃣ Environment Setup

The tool configures the environment:

```
★ Setting up environment...
✓ Set HSA_NO_SCRATCH_RECLAIM=1
✓ Set HSA_OVERRIDE_GFX_VERSION=11.0.0
➜ Enter HuggingFace Token: [hidden input]
✓ HuggingFace token set
```

**Automatic settings:**
- `HSA_NO_SCRATCH_RECLAIM=1` (always set)
- Any custom environment variables you added
- `HF_TOKEN` for HuggingFace authentication

---

### 9️⃣ Benchmark Execution

The tool runs benchmarks for all selected configurations:

```
★ Starting Benchmark 1/2...
   ● Model: llama3_8b
   ● Backend: megatron
   ● Device: MI300X
   ● Config: logs/llama3_8b_megatron_MI300X_2025-12-11_10-30-45_override.yaml
   ● Log: logs/primus_llama3_8b_megatron_MI300X_2025-12-11_10-30-45.log

✓ EXP set to: logs/llama3_8b_megatron_MI300X_2025-12-11_10-30-45_override.yaml

[Benchmark output streams here...]

==========================================
 Benchmark 1/2 Completed!
 Log saved at:
   logs/primus_llama3_8b_megatron_MI300X_2025-12-11_10-30-45.log
 Override config saved at:
   logs/llama3_8b_megatron_MI300X_2025-12-11_10-30-45_override.yaml
==========================================

Preparing next benchmark...

[Continues with next benchmark...]
```

**For each benchmark:**
- Applies edited/overridden configurations
- Exports `EXP` environment variable pointing to the config
- Executes `./examples/run_pretrain.sh`
- Streams output to both terminal and log file
- Saves timestamped logs to `logs/` directory

---

### 🔟 Completion

After all benchmarks complete:

```
=========================================
  All 2 Benchmark(s) Completed!
=========================================
```

---

## 📁 Output Files

All output files are saved in the `logs/` directory with timestamps:

### Log Files
```
logs/primus_{MODEL}_{BACKEND}_{DEVICE}_{TIMESTAMP}.log
```
Example: `logs/primus_llama3_8b_megatron_MI300X_2025-12-11_10-30-45.log`

### Edited/Override Config Files
```
logs/{MODEL}_{BACKEND}_{DEVICE}_{TIMESTAMP}_edited.yaml
logs/{MODEL}_{BACKEND}_{DEVICE}_{TIMESTAMP}_override.yaml
```

---

## 💡 Tips & Best Practices

1. **Batch Processing:** Use `all` or ranges (e.g., `1-5`) to benchmark multiple models efficiently
2. **Parameter Overrides:** Use overrides for quick experiments without modifying config files
3. **Log Management:** Review logs in the `logs/` directory for detailed benchmark results
4. **Environment Variables:** Add device-specific tuning variables for optimal performance
5. **Config Editing:** Edit configs to test different hyperparameters before running

---
