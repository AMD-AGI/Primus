# 🎯 Primus Auto Benchmarking Tool

<img width="1024" height="468" alt="Primus Banner" src="https://github.com/user-attachments/assets/f1b2bf61-d612-4e62-bac4-ac115928632a" />

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

### Step 2: Navigate to Auto Benchmark Directory

```bash
cd /workspace/Primus/tools/auto_benchmark/
```

### Step 3: Run the Benchmarking Tool

```bash
bash run_primus_autobenchmark.sh
```

---

## 📋 Features

- ✅ **Interactive Menu System** - User-friendly CLI with color-coded outputs and ASCII banner
- ✅ **Multi-Backend Support** - Compatible with Megatron and TorchTitan with device-specific configs
- ✅ **Batch Processing** - Run multiple model configurations sequentially with flexible selection
- ✅ **Configuration Viewing** - Preview YAML configs before execution
- ✅ **Configuration Editing** - Edit YAML configs individually or in batch before execution
- ✅ **Parameter Overrides** - Override specific parameters without editing files permanently
- ✅ **Auto Device Detection** - Automatically detects AMD MI300X/MI355X GPUs with intelligent fallback
- ✅ **Device-Specific Paths** - Automatically uses device-specific config directories (MI300X/MI355X)
- ✅ **Comprehensive Logging** - Timestamped logs saved in organized backend-specific directories
- ✅ **Environment Management** - Custom device-specific environment variable support
- ✅ **Automatic Metrics Generation** - Backend-specific metrics tables generated after completion
- ✅ **Smart Config Management** - Handles edited/override configs properly with automatic cleanup

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

The backend selection determines:
- Which config directory to use
- Which metrics script to run after completion
- Where logs are saved

---

### 2️⃣ Device Detection

The tool automatically detects your AMD GPU with intelligent fallback:

```
★ Detecting Device...
  ● Device found: MI300X
✓ GPU Device: MI300X

✓ Config directory set to: /workspace/Primus/examples/megatron/configs/MI300X
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

**Device-Specific Paths:**
After detection, the config directory is automatically set to:
- Megatron MI300X: `/workspace/Primus/examples/megatron/configs/MI300X`
- Megatron MI355X: `/workspace/Primus/examples/megatron/configs/MI355X`
- TorchTitan MI300X: `/workspace/Primus/examples/torchtitan/configs/MI300X`
- TorchTitan MI355X: `/workspace/Primus/examples/torchtitan/configs/MI355X`

---

### 3️⃣ Model Configuration Selection

The tool scans for available YAML configuration files in the device-specific backend directory:

```
★ Available Model Configs: (megatron / MI300X)
  ● 1) llama3_8b.yaml
  ● 2) llama3_70b.yaml
  ● 3) qwen2.5_7B-FP8-pretrain.yaml

➜ Select config number(s) (comma-separated, range, or 'all'):
(Examples: 1,3,5 or 4-8 or all)
```

**Selection Options:**
- **Single:** `1` - Select one config
- **Multiple:** `1,3,5` - Select specific configs (comma-separated)
- **Range:** `4-8` - Select a range of configs
- **All:** `all` - Select all available configs

**Note:** The tool automatically filters out duplicate configs to prevent redundant processing.

---

### 4️⃣ View Configuration Parameters

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
seq_length: 2048
-----------------------------------
```

---

### 5️⃣ Edit Configuration Files

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

The tool creates a temporary working copy in `/tmp` and opens it in your default editor (tries `nano`, `vim`, `vi`, `code`, or `$EDITOR`).

**Important:**
- Edits are applied to the original config location temporarily during benchmark execution
- Original configs are backed up and restored after each benchmark completes
- Edited configs are preserved in logs directory for reproducibility

---

### 6️⃣ Override Parameters

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
- Creates `{MODEL}_{BACKEND}_{DEVICE}_{TIMESTAMP}_override.yaml` in logs directory
- Applies overrides using sed for precise YAML modification
- Can be combined with edited configs
- Original config is temporarily replaced during execution then restored

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
- `HSA_NO_SCRATCH_RECLAIM=1` (always set for AMD GPUs)
- Any custom environment variables you added
- `HF_TOKEN` for HuggingFace authentication (hidden input for security)

---

### 9️⃣ Benchmark Execution

The tool runs benchmarks for all selected configurations sequentially:

```
ℹ Total configurations to run: 2
ℹ Configuration list:
   ● 1. llama3_8b.yaml
   ● 2. qwen2.5_7B-FP8-pretrain.yaml

╔════════════════════════════════════════════════════════════╗
║  LOOP ITERATION: 1/2
║  CONFIG FILE: llama3_8b.yaml
╚════════════════════════════════════════════════════════════╝

★ Starting Benchmark 1/2...
   ● Model: llama3_8b
   ● Backend: megatron
   ● Device: MI300X
   ● Config: /workspace/Primus/tools/auto_benchmark/results/logs_megatron/llama3_8b_megatron_MI300X_2025-12-17_10-30-45_override.yaml
   ● Log: /workspace/Primus/tools/auto_benchmark/results/logs_megatron/llama3_8b_megatron_MI300X_2025-12-17_10-30-45.log

✓ Copied edited/overridden config to: /workspace/Primus/examples/megatron/configs/MI300X/llama3_8b.yaml
✓ EXP set to: /workspace/Primus/examples/megatron/configs/MI300X/llama3_8b.yaml

  ● Changing to Primus root directory: /workspace/Primus

[Benchmark output streams here...]

✓ Restored original config file

==========================================
 ✓ Benchmark 1/2 Completed Successfully!
 Log saved at:
   /workspace/Primus/tools/auto_benchmark/results/logs_megatron/llama3_8b_megatron_MI300X_2025-12-17_10-30-45.log
 Override config saved at:
   /workspace/Primus/tools/auto_benchmark/results/logs_megatron/llama3_8b_megatron_MI300X_2025-12-17_10-30-45_override.yaml
==========================================

Preparing next benchmark...

ℹ Next: Config 2/2

[Continues with next benchmark...]
```

**For each benchmark:**
1. Displays iteration header with model info
2. Uses edited config if available, otherwise uses original
3. Applies parameter overrides to create timestamped override config in logs
4. Temporarily replaces original config with edited/overridden version
5. Changes to Primus root directory (`/workspace/Primus`) for proper path resolution
6. Exports `EXP` environment variable pointing to the device-specific config path
7. Executes `runner/primus-cli direct -- train pretrain --config <exp>`
8. Streams output to both terminal and timestamped log file
9. Restores original config file after completion
10. Shows completion status with log file location
11. Adds 2-second delay between benchmarks for system stability

**Error Handling:**
- Script continues even if a benchmark fails
- Exit codes are captured and displayed
- Warnings shown for non-zero exit codes

---

### 🔟 Metrics Generation

After all benchmarks complete, backend-specific metrics are automatically generated:

```
=========================================
  All 2 Benchmark(s) Completed!
=========================================

★ Generating Metrics Table...

✓ Running: python metrics_megatron.py

[Metrics table displayed here...]

✓ Metrics table generated successfully
```

**Automatic Metrics Scripts:**
- **Megatron:** Runs `metrics_megatron.py`
- **TorchTitan:** Runs `metrics_torchtitan.py`

The metrics scripts parse the log files and generate formatted summary tables with performance statistics.

---

## 📁 Output Files

All output files are organized in the `results/` directory within the auto_benchmark tool:

### Directory Structure
```
/workspace/Primus/tools/auto_benchmark/
├── run_primus_autobenchmark.sh
├── metrics_megatron.py
├── metrics_torchtitan.py
└── results/
    ├── logs_megatron/
    │   ├── llama3_8b_megatron_MI300X_2025-12-17_10-30-45.log
    │   ├── llama3_8b_megatron_MI300X_2025-12-17_10-30-45_override.yaml
    │   └── llama3_70b_megatron_MI300X_2025-12-17_11-15-20.log
    └── logs_torchtitan/
        ├── qwen2.5_7B_torchtitan_MI355X_2025-12-17_14-20-30.log
        └── qwen2.5_7B_torchtitan_MI355X_2025-12-17_14-20-30_edited.yaml
```

### Log Files
**Format:** `{CONFIG_FILENAME}_{BACKEND}_{DEVICE}_{TIMESTAMP}.log`

**Examples:**
- `/workspace/Primus/tools/auto_benchmark/results/logs_megatron/llama3_8b_megatron_MI300X_2025-12-17_10-30-45.log`
- `/workspace/Primus/tools/auto_benchmark/results/logs_torchtitan/qwen2.5_7B_torchtitan_MI355X_2025-12-17_14-20-30.log`

**Benefits:**
- Full config filename preserved (not just model name)
- Backend-specific directories for easy organization
- Complete benchmark output with metrics
- Timestamped for version tracking

### Override Config Files
**Format:** `{CONFIG_FILENAME}_{BACKEND}_{DEVICE}_{TIMESTAMP}_override.yaml`

**Example:**
```
/workspace/Primus/tools/auto_benchmark/results/logs_megatron/llama3_8b_megatron_MI300X_2025-12-17_10-30-45_override.yaml
```

**Contents:** Copy of original config with parameter overrides applied

### Edited Config Files
**Format:** `{CONFIG_FILENAME}_{BACKEND}_{DEVICE}_{TIMESTAMP}_edited.yaml`

**Example:**
```
/workspace/Primus/tools/auto_benchmark/results/logs_torchtitan/qwen2.5_7B_torchtitan_MI355X_2025-12-17_14-20-30_edited.yaml
```

**Contents:** Copy of the manually edited config used for the benchmark

---

## 💡 Tips & Best Practices

1. **Batch Processing:** Use `all` or ranges (e.g., `1-5`) to benchmark multiple models efficiently
2. **Device-Specific Configs:** Ensure configs exist in the correct device subdirectory (MI300X/MI355X)
3. **Parameter Overrides:** Use overrides for quick experiments without modifying config files permanently
4. **Log Organization:** Logs are automatically organized by backend in `results/logs_{backend}/`
5. **Environment Variables:** Add device-specific tuning variables (e.g., `HSA_OVERRIDE_GFX_VERSION`) for optimal performance
6. **Config Editing:** Edited configs are applied during execution but originals are preserved
7. **View Before Running:** Always preview configs before execution to verify parameters
8. **Metrics Analysis:** Backend-specific metrics scripts automatically parse logs after completion
9. **Sequential Runs:** 2-second delay between benchmarks ensures system stability
10. **Error Resilience:** Script continues running even if individual benchmarks fail
11. **Path Resolution:** Script automatically changes to Primus root directory for proper execution
12. **Config Backup:** Original configs are automatically backed up and restored after each benchmark

---

## 🛠️ Technical Details

### Directory Structure
```
/workspace/Primus/
├── examples/
│   ├── megatron/
│   │   ├── configs/
│   │   │   ├── MI300X/           # MI300X-specific Megatron configs
│   │   │   └── MI355X/           # MI355X-specific Megatron configs
│   │   └── prepare.py
│   ├── torchtitan/
│   │   ├── configs/
│   │   │   ├── MI300X/           # MI300X-specific TorchTitan configs
│   │   │   └── MI355X/           # MI355X-specific TorchTitan configs
│   │   └── prepare.py
├── runner/
│   └── primus-cli                 # Main benchmark launcher (direct mode)
└── tools/
    └── auto_benchmark/
        ├── run_primus_autobenchmark.sh
        ├── metrics_megatron.py
        ├── metrics_torchtitan.py
        └── results/
            ├── logs_megatron/
            └── logs_torchtitan/
```

### Environment Variables Set
- `HSA_NO_SCRATCH_RECLAIM=1` (always set for AMD GPUs)
- `HF_TOKEN` (user-provided for HuggingFace access)
- Custom device-specific variables (optional, user-defined)
- `EXP` (config path for each benchmark, format: `{BACKEND_BASE_DIR}/{DEVICE}/{config}.yaml`)

### Execution Flow
1. Backend selection → Sets `BACKEND_BASE_DIR`
2. Device detection → Sets `DEVICE`
3. Config directory construction → `CONFIG_DIR = {BACKEND_BASE_DIR}/{DEVICE}`
4. Config selection → Scans device-specific directory
5. Config editing/overrides → Creates temporary working copies
6. Environment setup → Exports required variables
7. **For each benchmark:**
   - Backup original config
   - Copy edited/override config to original location
   - Change to `/workspace/Primus` directory
   - Export `EXP` variable
   - Execute `runner/primus-cli direct -- train pretrain --config <exp>`
   - Restore original config
   - Save logs to `results/logs_{backend}/`
8. Generate metrics table using backend-specific script

### Supported Editors (Priority Order)
1. `nano` - Simple terminal editor
2. `vim` - Advanced terminal editor
3. `vi` - Classic Unix editor
4. `code` - VS Code with `--wait` flag
5. `$EDITOR` - Environment variable fallback

### Device Detection Logic
```bash
1. Query rocminfo for "AMD Instinct" → Extract model name (MI300X/MI355X)
2. If empty or invalid → Query rocminfo for architecture (gfx942 → MI300X, gfx950 → MI355X)
3. If still empty → Prompt for manual selection (1=MI300X, 2=MI355X)
```

### Config Management Strategy
- **Original configs:** Never permanently modified
- **Edited configs:** Created in `/tmp`, temporarily replace originals during execution
- **Override configs:** Created in logs directory with all changes applied
- **Backup mechanism:** `.backup_$$` files created before replacement, restored after execution
- **Log preservation:** All config versions saved to logs directory for reproducibility

---

## 📝 Example Session

```bash
# Full example workflow
cd /workspace/Primus/tools/auto_benchmark/
bash run_primus_autobenchmark.sh

# Interactive prompts:
# Backend: megatron (enter '1')
# Device: Auto-detected as MI300X
# Configs: 1,2 (select llama3_8b and qwen2.5_7B)
# View parameters: y (preview configs)
# Edit configs: n (skip editing)
# Override parameters: y
#   batch_size=64
#   learning_rate=0.0005
# Add env vars: y
#   HSA_OVERRIDE_GFX_VERSION=11.0.0
# HF token: hf_xxxxxxxxxxxxx

# Output:
# ✓ 2 benchmarks run sequentially
# ✓ Logs saved to results/logs_megatron/
# ✓ Override configs saved with logs
# ✓ Metrics table automatically generated
# ✓ Original configs remain unchanged
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue:** `find: 'examples/megatron/configs/MI300X': No such file or directory`
- **Cause:** Config directory doesn't exist for the detected device
- **Solution:** Ensure configs exist in `/workspace/Primus/examples/{backend}/configs/{device}/`

**Issue:** `Backend prepare script not found`
- **Cause:** Script not executing from Primus root directory
- **Solution:** Script now automatically changes to `/workspace/Primus` before execution

**Issue:** Edited config not being used
- **Cause:** Config not properly copied to original location
- **Solution:** Script now backs up and replaces original configs during execution

**Issue:** Log files not found
- **Cause:** Logs saved in wrong directory
- **Solution:** Check `results/logs_{backend}/` in auto_benchmark directory

---

## 🆘 Support

For issues or questions:
1. Check log files in `/workspace/Primus/tools/auto_benchmark/results/logs_{backend}/`
2. Verify ROCm installation: `rocminfo`
3. Ensure device-specific configs exist in proper directories
4. Verify script is run from `/workspace/Primus/tools/auto_benchmark/`
5. Check that `runner/primus-cli` exists in `/workspace/Primus/runner/`
6. Review this README for proper usage

---

## 📊 Backend-Specific Metrics

### Megatron Metrics
- **TPS** (Tokens Per Second) - Throughput metric
- **TFLOPS** (Tera FLOPs) - Compute performance
- **Memory (%)** - GPU memory utilization
- **Time (ms)** - Elapsed execution time

### TorchTitan Metrics
- **TPS** (Tokens Per Second) - Throughput metric
- **TFLOPS** (Tera FLOPs) - Compute performance
- **MFU** (Model FLOPs Utilization) - Efficiency metric
- **Memory (%)** - GPU memory utilization

**Note:** Metrics are extracted automatically by running `metrics_{backend}.py` after all benchmarks complete.

---

**Happy Benchmarking! 🚀**
