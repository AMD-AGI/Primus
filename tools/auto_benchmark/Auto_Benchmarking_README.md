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
cd /workspace/Primus/tools/auto_benchmark/
```

### Step 3: Run the Benchmarking Tool

```bash
bash run_primus_auto_benchmarking_tool.sh
```

---

## 📋 Features

- ✅ **Interactive Menu System** - User-friendly CLI with color-coded outputs and ASCII banner
- ✅ **Multi-Backend Support** - Compatible with Megatron and TorchTitan
- ✅ **Batch Processing** - Run multiple model configurations sequentially
- ✅ **Configuration Viewing** - Preview YAML configs before execution
- ✅ **Configuration Editing** - Edit YAML configs individually or in batch before execution
- ✅ **Parameter Overrides** - Override specific parameters without editing files
- ✅ **Auto Device Detection** - Automatically detects AMD MI300X/MI355X GPUs with fallback
- ✅ **Comprehensive Logging** - Timestamped logs for each benchmark run
- ✅ **Environment Management** - Custom device-specific environment variable support
- ✅ **Results Summary** - Automatic metrics extraction and formatted summary table
- ✅ **Smart Config Management** - Prevents duplicates and properly handles edited/override configs

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

The tool scans for available YAML configuration files in the selected backend directory, excluding previously edited or override configs:

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

**Note:** The tool automatically filters out duplicate configs and previously edited/override versions.

---

### 3️⃣ View Configuration Parameters

Option to preview parameters in your selected configurations:

```
★ View Configuration Parameters?
➜ (y/n):
```

If you choose `y`, the tool displays the contents of each selected YAML file (excluding comments and empty lines):

```
Parameters in llama3_8b.yaml:
-----------------------------------
batch_size: 16
learning_rate: 0.0001
max_steps: 1000
-----------------------------------
```

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

The tool creates a temporary working copy and opens it in your default editor (tries `nano`, `vim`, `vi`, `code`, or `$EDITOR`). 

**Improvements:**
- Edits are saved to the backend config directory as `{MODEL}_edited.yaml`
- Temporary files are automatically cleaned up after execution
- Multiple editors are supported with intelligent fallback

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

**Override Behavior:**
- Creates `{MODEL}_override.yaml` in the backend config directory
- Applies overrides using sed for precise YAML modification
- Can be combined with edited configs

---

### 6️⃣ Device Detection

The tool automatically detects your AMD GPU with intelligent fallback:

```
★ Detecting Device...
  ● Device found: MI300X
✓ GPU Device: MI300X
```

**Auto-detection methods (in order):**
1. Queries `rocminfo` for "AMD Instinct" devices (direct model name)
2. Falls back to architecture detection (gfx942 → MI300X, gfx950 → MI355X)
3. Manual selection prompt if both methods fail

**Manual Selection (if auto-detection fails):**
```
✗ Could not detect device automatically
★ Please select Device manually:
  ● 1) MI300X
  ● 2) MI355X

➜ Enter number or name:
```

**Supported Inputs:** Numbers (1, 2) or names (MI300X, MI355X) in any case

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
➜ Variable (or press Enter to finish): ROCR_VISIBLE_DEVICES=0,1,2,3
✓ Will set: ROCR_VISIBLE_DEVICES=0,1,2,3
➜ Variable (or press Enter to finish): [Press Enter]

✓ 2 environment variable(s) will be set
```

**Format:** `VAR_NAME=value` (allows empty values)

---

### 8️⃣ Environment Setup

The tool configures the environment:

```
★ Setting up environment...
✓ Set HSA_NO_SCRATCH_RECLAIM=1
✓ Set HSA_OVERRIDE_GFX_VERSION=11.0.0
✓ Set ROCR_VISIBLE_DEVICES=0,1,2,3
➜ Enter HuggingFace Token: [hidden input]
✓ HuggingFace token set
```

**Automatic settings:**
- `HSA_NO_SCRATCH_RECLAIM=1` (always set)
- Any custom environment variables you added
- `HF_TOKEN` for HuggingFace authentication (hidden input)

---

### 9️⃣ Benchmark Execution

The tool runs benchmarks for all selected configurations:

```
★ Starting Benchmark 1/2...
   ● Model: llama3_8b
   ● Backend: megatron
   ● Device: MI300X
   ● Config: /workspace/Primus/examples/megatron/configs/llama3_8b_override.yaml
   ● Log: /workspace/Primus/logs/primus_llama3_8b_megatron_MI300X_2025-12-12_10-30-45.log

✓ EXP set to: /workspace/Primus/examples/megatron/configs/llama3_8b_override.yaml

[Benchmark output streams here...]

ℹ Extracting metrics from log...
✓ Metrics extracted successfully
   ● TPS: 1234.56
   ● TFLOPS: 456.78
   ● Memory: 85.2%
   ● Elapsed Time: 123.45 ms

ℹ Cleaning up temporary config: llama3_8b_override.yaml
✓ Temporary config removed

==========================================
 Benchmark 1/2 Completed!
 Log saved at:
   /workspace/Primus/logs/primus_llama3_8b_megatron_MI300X_2025-12-12_10-30-45.log
 Override config saved at:
   /workspace/Primus/examples/megatron/configs/llama3_8b_override.yaml
==========================================

Preparing next benchmark...

[Continues with next benchmark...]
```

**For each benchmark:**
- Uses edited config if available, otherwise uses original
- Applies parameter overrides to create `{MODEL}_override.yaml`
- Changes to Primus root directory before execution
- Exports `EXP` environment variable pointing to the config
- Executes `./examples/run_pretrain.sh`
- Streams output to both terminal and log file
- Extracts backend-specific metrics automatically
- Cleans up temporary config files
- Saves timestamped logs to `/workspace/Primus/logs/` directory

---

### 🔟 Results Summary

After all benchmarks complete, a formatted summary table is displayed:

```
=========================================
  All 2 Benchmark(s) Completed!
=========================================

★ Benchmark Results Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model                     Backend      TPS             TFLOPS          MFU             Memory (%)      Time (ms)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
llama3_8b                 megatron     1234.56         456.78          -               85.2            123.45
llama3_70b                torchtitan   567.89          234.56          78.9            92.1            -
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Note: MFU (Model FLOPs Utilization) for torchtitan | Time (ms) for megatron

ℹ Log files saved in: /workspace/Primus/logs
```

**Metrics Extracted:**

**For Megatron:**
- TPS (Tokens Per Second)
- TFLOPS (Tera FLOPs)
- Memory Percentage
- Elapsed Time (ms)

**For TorchTitan:**
- TPS (Tokens Per Second)
- TFLOPS (Tera FLOPs)
- MFU (Model FLOPs Utilization)
- Memory Percentage

**Note:** Backend-specific metrics show "-" for non-applicable values.

---

## 📁 Output Files

All output files are saved in `/workspace/Primus/logs/` directory with timestamps:

### Log Files
```
/workspace/Primus/logs/primus_{MODEL}_{BACKEND}_{DEVICE}_{TIMESTAMP}.log
```
Example: `/workspace/Primus/logs/primus_llama3_8b_megatron_MI300X_2025-12-12_10-30-45.log`

### Edited Config Files
```
/workspace/Primus/examples/{BACKEND}/configs/{MODEL}_edited.yaml
```
Example: `/workspace/Primus/examples/megatron/configs/llama3_8b_edited.yaml`

**Note:** Saved to backend config directory for proper path resolution

### Override Config Files
```
/workspace/Primus/examples/{BACKEND}/configs/{MODEL}_override.yaml
```
Example: `/workspace/Primus/examples/megatron/configs/llama3_8b_override.yaml`

**Cleanup:** Temporary override configs are automatically removed after benchmark completion.

---

## 💡 Tips & Best Practices

1. **Batch Processing:** Use `all` or ranges (e.g., `1-5`) to benchmark multiple models efficiently
2. **Parameter Overrides:** Use overrides for quick experiments without modifying config files permanently
3. **Log Management:** Review logs in `/workspace/Primus/logs/` for detailed benchmark results and metrics
4. **Environment Variables:** Add device-specific tuning variables (e.g., `HSA_OVERRIDE_GFX_VERSION`) for optimal performance
5. **Config Editing:** Edit configs to test different hyperparameters; edited versions are saved separately
6. **View Before Running:** Always preview configs before execution to verify parameters
7. **Metrics Summary:** Check the summary table at the end for quick performance comparison across runs
8. **Config Cleanup:** Temporary override configs are auto-cleaned, but edited configs persist for reuse
9. **Backend-Specific Metrics:** Note that Megatron provides elapsed time while TorchTitan provides MFU
10. **Sequential Runs:** The tool includes a 2-second delay between benchmarks for system stability

---

## 🛠️ Technical Details

### Directory Structure
```
/workspace/Primus/
├── examples/
│   ├── megatron/
│   │   └── configs/         # Megatron YAML configs
│   ├── torchtitan/
│   │   └── configs/         # TorchTitan YAML configs
│   └── run_pretrain.sh      # Benchmark execution script
├── logs/                    # Timestamped logs and metrics
└── tools/
    └── auto_benchmark/
        └── run_primus_auto_benchmarking_tool.sh
```

### Environment Variables Set
- `HSA_NO_SCRATCH_RECLAIM=1` (always)
- `HF_TOKEN` (user-provided)
- Custom device-specific variables (optional)
- `EXP` (config path for each benchmark)

### Supported Editors (Priority Order)
1. `nano`
2. `vim`
3. `vi`
4. `code` (VS Code with --wait flag)
5. `$EDITOR` environment variable

### Device Detection Logic
```bash
1. Check rocminfo for "AMD Instinct" → Extract model name
2. If empty or invalid → Check rocminfo for architecture (gfx942/gfx950)
3. If still empty → Prompt for manual selection
```

## 📝 Example Session

```bash
# Full example workflow
cd /workspace/Primus/tools/auto_benchmark/
bash run_primus_auto_benchmarking_tool.sh

# Select backend: megatron
# Select configs: 1,3 (llama3_8b and gpt3_175b)
# View parameters: y
# Edit configs: n
# Override parameters: y
#   - batch_size=64
#   - learning_rate=0.0005
# Add env vars: y
#   - HSA_OVERRIDE_GFX_VERSION=11.0.0
# Enter HF token: [your_token]

# Benchmarks run sequentially
# Results summary displayed
# Review logs in /workspace/Primus/logs/
```

---

## 🆘 Support

For issues or questions:
1. Check log files in `/workspace/Primus/logs/`
2. Verify ROCm installation: `rocminfo`
3. Ensure configs exist in backend directories
4. Review this README for proper usage

---

**Happy Benchmarking! 🚀**
