# Llama 2 70B LoRA (Megatron-Bridge) — run guide

This document matches `setup_llama2_70b_lora_training.sh` at the Primus repo root and `examples/megatron_bridge/configs/MI355X/llama2_70b_lora_posttrain.yaml`. Use it to verify local paths before training, or to run the full automated setup.

## Prerequisites

- Primus repository with submodules (`git submodule update --init --recursive` is run by the setup script).
- Docker (and GPU drivers appropriate for the image).
- **`HF_TOKEN`**: a Hugging Face token with access to **`meta-llama/Llama-2-70b-hf`** (the model config is in `primus/configs/models/megatron_bridge/llama2_70b_lora.yaml` as `hf_path`). Export it before running the script.
- **`WANDB_API_KEY`**: optional; training still runs if unset (you may see a warning).

## Paths: data, cache, and base checkpoint

All of the following default to the **same host directory**, which is bind-mounted into the container at the **identical absolute path** (see `tools/docker/start_container.sh`: `DATA_PATH` is mounted as `DATA_PATH:DATA_PATH`). There is no separate `/data` alias unless you change mounts yourself.

| Purpose | Default host path | Env override |
|--------|-------------------|--------------|
| Data root (packed `.npy`, metadata, HF hub cache, optional Megatron ckpt tree) | `<PRIMUS_ROOT>/data/mlperf_llama2` | `HOST_MLPERF_DATA` |

`<PRIMUS_ROOT>` is the directory that contains `setup_llama2_70b_lora_training.sh` (the Primus repo root).

### If you already have artifacts locally

Before (or instead of) a full download, check that these exist under **`HOST_MLPERF_DATA`** (default: `data/mlperf_llama2` relative to repo root):

1. **Packed MLPerf dataset** (must align with `download_dataset.py` / `convert_dataset.py` output):
   - `train.npy`
   - `validation.npy`
   - `packed_metadata.jsonl`  
   If all three are present, the setup script **skips** dataset download and metadata generation.

2. **Hugging Face cache** (tokenizer / hub files for the recipe):  
   - Default cache when using the script: **`<HOST_MLPERF_DATA>/.cache/huggingface`** (`HF_HOME` is set there so it survives container removal).

3. **LoRA base weights (Megatron checkpoint)** — required for PEFT:  
   - Directory: **`<HOST_MLPERF_DATA>/megatron_checkpoints/`** (override basename with `MEGATRON_CKPT_SUBDIR` if needed).  
   - The script searches for a directory named **`iter_*`** under that tree and, if found, passes **`modules.post_trainer.overrides.pretrained_checkpoint=...`** to training.  
   Example layout:  
   `.../megatron_checkpoints/<your_run_name>/iter_0000000/`

If there is **no** `iter_*` checkpoint, training will warn that `pretrained_checkpoint` is missing; set it in the YAML or place weights under the tree above.

### YAML vs script

In `llama2_70b_lora_posttrain.yaml`, packed paths default to:

- `${PACKED_DATA_DIR:/data}/train.npy` (and the same for val + metadata).

The setup script exports **`PACKED_DATA_DIR=<HOST_MLPERF_DATA>`** inside the training container so training reads the same tree as the script—not the YAML’s `/data` fallback. If you run `primus-cli` by hand, set **`PACKED_DATA_DIR`** to your data root or override the three `packed_*` paths on the CLI.

The **model id** for Hugging Face (`meta-llama/Llama-2-70b-hf`) comes from the model YAML; ensure your token has **gated model access** for that repo.

## One-command setup (recommended)

From the **Primus repo root** (where `setup_llama2_70b_lora_training.sh` lives):

```bash
export HF_TOKEN="hf_..."   # required
export WANDB_API_KEY="..." # optional

bash ./setup_llama2_70b_lora_training.sh
```

The script will:

1. Update Git submodules.  
2. Start or reuse the Docker container (`CONTAINER_NAME`, default `llama2_lora_26_2_primus_upstream`; image `DOCKER_IMAGE`, default `rocm/primus:v26.2`) via `tools/docker/start_container.sh`, mounting **`HOST_MLPERF_DATA`** and the Primus tree.  
3. Download / convert the MLPerf packed dataset and build metadata **only if** the three files above are missing.  
4. Apply Megatron-Bridge patches (LoRA / NeMo-style timing patch is required to apply cleanly; others may be best-effort).  
5. Launch **`primus-cli direct train posttrain`** with `examples/megatron_bridge/configs/MI355X/llama2_70b_lora_posttrain.yaml` and optional `pretrained_checkpoint` override when `iter_*` is found.

### Common overrides

| Variable | Meaning |
|----------|---------|
| `HOST_MLPERF_DATA` | Root for data, HF cache, and `megatron_checkpoints` |
| `DOCKER_IMAGE` | Primus ROCm image tag |
| `CONTAINER_NAME` | Docker container name |
| `CONTAINER_PRIMUS_ROOT` | Path to Primus inside the container (default `/workspace/Primus`; must match `start_container.sh` mount) |

## Megatron-Bridge: keep `third_party` clean, ship patches only

Upstream Megatron-Bridge lives under **`third_party/Megatron-Bridge`**. **Do not commit** hand-edited copies of those files in the Primus repo. All Llama 2 70B–related Megatron-Bridge changes are carried as unified diffs under **`primus/recipes/patches/`**:

| Patch | Purpose |
|-------|---------|
| `megatron_nemo_lora_only.patch` | LoRA fused controls, NeMo-style step timing, `train_utils` logging / `print_rank_0` |
| `megatron_bridge_validation_consumed_samples.patch` | Validation sample accounting and capped val dataloader |
| `megatron_bridge_deterministic_eval.patch` | Deterministic evaluation iterator reset |

`setup_llama2_70b_lora_training.sh` runs **`git reset --hard HEAD`** inside that submodule (in the container), then applies these patches so the tree matches the pinned submodule commit plus the patch layer.

To **drop any local edits** you made directly under `third_party/Megatron-Bridge` and match Git again:

```bash
cd third_party/Megatron-Bridge
git reset --hard HEAD
git clean -fd   # optional: removes untracked files; use only if you intend to delete them
```

To confirm patches still apply on a clean checkout (from Primus repo root):

```bash
cd third_party/Megatron-Bridge
git reset --hard HEAD
git apply --check ../../primus/recipes/patches/megatron_nemo_lora_only.patch
git apply --check ../../primus/recipes/patches/megatron_bridge_validation_consumed_samples.patch
git apply --check ../../primus/recipes/patches/megatron_bridge_deterministic_eval.patch
```

## Pushing your changes to GitHub (Primus repo)

1. **Submodule**  
   Ensure Megatron-Bridge has **no unintended** modified tracked files before you commit the parent repo:

   ```bash
   git -C third_party/Megatron-Bridge status -sb
   ```

   You want a clean line for that submodule (or a deliberate new commit if you bumped the submodule pin).

2. **Stage only what you mean to publish** (patches, setup script, docs, YAML, etc.):

   ```bash
   cd /path/to/Primus
   git add primus/recipes/patches/*.patch
   git add setup_llama2_70b_lora_training.sh
   git add primus/recipes/readme_run_llama2_70b.md primus/recipes/README.md
   # plus any other files you changed on purpose
   git status
   ```

3. **Commit** on your branch with a clear message (what patches do, any setup or YAML changes).

4. **Push** to your remote (replace branch name as needed):

   ```bash
   git push origin dev/mlperf/llama2_70b
   ```

   If the branch does not exist on the remote yet:

   ```bash
   git push -u origin dev/mlperf/llama2_70b
   ```

5. **If you updated the Megatron-Bridge submodule commit** (e.g. after `git submodule update --remote` and recording a new SHA), also run **`git add third_party/Megatron-Bridge`** and explain the bump in the commit message. Otherwise leave the submodule pointer as recorded by Primus and rely on patches only.

6. **Open a PR** on GitHub from your pushed branch into the target branch of the upstream Primus repository, if that is your workflow.

## Related docs

- `primus/recipes/README.md` — recipes overview.  
- `primus/recipes/README_llama2_custom_enabled_knobs.md` — Llama2 custom recipe knobs.  
- `primus/recipes/README_llama2_te_lora_optimization.md` — TE / LoRA tuning notes.
