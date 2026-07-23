# Reproducing ODC on AMD ROCm (Primus + Primus-Turbo)

This is the companion, from-zero reproduction guide for the ROCm blog on using ODC to accelerate AMD SFT training: it walks through single-node 1.5B and dual-node 14B runs — including the `nccl_pad` fair baseline — so you can build the ODC rocSHMEM operators, run the arms, and measure the speedup yourself.

This guide distills the experiments into a follow-along, copy-pasteable tutorial, so that **a reader who has never touched this codebase** can start from zero and fully run the `odc_nopad` experiments for **single-node 1.5B** and **dual-node 14B** (including the `nccl_pad` fair baseline) and compute the speedup.

**Overall flow at a glance (6 steps; strongly recommended to get single-node working first, then move to dual-node):**

1. **Start the container** (see "Get the code, start the container"): clone the PR #864 branch + Primus-Turbo, `docker run` to start the container.
2. **Get the GDA operators in place** (see "Build Primus-Turbo with ODC operators"): **build Primus-Turbo from source** (one host + one GDA build, linking rocSHMEM at build time), then use `.image_bak` to disable the stock turbo shipped in the image that has no ODC operators, consume it via `PRIMUS_TURBO_PATH`, and verify `has_gda=True`.
3. **Prepare offline data** (see "Prerequisites"): stage the HF model/data, set `HF_HOME` + `HF_HUB_OFFLINE=1`.
4. **Turn on ODC via config** (see "Turning on ODC = changing config items"): the `odc_nopad` arm = `enable_odc:true`+`enable_odc_lb_mini:true`+`odc_p2p_backend:rocshmem`; the `nccl_pad` aligned baseline = `enable_odc:false`+`enable_odc_lb_mini:true`.
5. **Start training**: for single-node see "Single-node 1.5B `odc_nopad`" (8 GPUs, no RoCE/GDA needed, **newcomers do this first**); once it works, move to dual-node per "Dual-node 14B `odc_nopad`" (16 GPUs, rocSHMEM GDA).
6. **Validate + compute the speedup** (see "Judging success, reading results, computing the speedup"): confirm from the logs that the three fixes are in effect, loss is decreasing, 0 nan, and `speedup = nccl_pad's ms/iter ÷ odc_nopad's ms/iter`.

When stuck, first check the "Pitfalls checklist" section below.

> **Important change — ODC is now driven by "config items," no longer by environment variables.** PR [#864](https://github.com/AMD-AGI/Primus/pull/864) (branch `feat/odc-consume-turbo`) changed all production feature switches into Primus yaml config items (`enable_odc`, `odc_phase`, `enable_odc_lb_mini`, `odc_p2p_backend`, `odc_rocshmem_gda`, `odc_gda_*` …), which are no longer read from `os.environ` at runtime. So all commands in this guide are based on "change config item / CLI override"; the legacy `ODC_*` environment variables have been deprecated.

To make ODC **run, run correctly, and run fast** on ROCm, **three fixes are indispensable** (division of labor: **Primus** owns the algorithm layer + integration, pure Python under `primus/core/odc/`; **Primus-Turbo** owns the rocSHMEM comm operators):

- **① Primus-Turbo comm operators (`odc_rocshmem_host` / `odc_rocshmem_gda`).** The body of ODC's P2P communication. The operator source has been merged into [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) official `main` (PR #409, including the `--lto-partitions=1` required by GDA), but is gated by `#ifndef DISABLE_ROCSHMEM` — a plain `pip install @main` **will not build the operators** (`has_gda=False`). You must **build from source, pointing `ROCSHMEM_HOME` at a prebuilt rocSHMEM at build time** (one for single-node host / one for dual-node GDA); see "Build Primus-Turbo with ODC operators".
- **② hook into the correct FSDP2 class (PR [#808](https://github.com/AMD-AGI/Primus/pull/808) / [#864](https://github.com/AMD-AGI/Primus/pull/864)).** ODC must hook the new `PrimusTorchFullyShardedDataParallel`, otherwise `reduction_service` is `None` and **iter2 crashes** with `'NoneType' ... clear_accumulations`. PR #864 fixes this, taking effect only when `enable_odc: true` + `use_torch_fsdp2: true`.
- **③ #856 device_id gating (speed preservation).** The ODC arm (`enable_odc: true`) automatically skips the eager-RCCL `device_id` injection of [#856](https://github.com/AMD-AGI/Primus/pull/856) (patch `condition = use_torch_fsdp2 and not enable_odc`); without it, single-node gets its ODC XGMI copy stream serialized by eager-RCCL, ~6% slower.

All three above are **gated by yaml config items** (no longer by `ODC_*` environment variables), with zero impact on `nccl_pad` / native FSDP2.

## Prerequisites

### Hardware

- Single-node 1.5B: 1 AMD MI300X/MI355X (`gfx942`) ×8, intra-node XGMI interconnect suffices.
- Dual-node 14B: 2 machines ×8 = 16 GPUs, requiring RoCE/RDMA (multiple mlx5 NICs) between nodes. **Be sure to pick adjacent nodes under the same leaf switch, mutually reachable over RoCE** — cross-topology node pairs often hang at the first cross-node communication (we measured multiple non-adjacent node pairs stalling at GDA warmup, going a long time without producing an `iteration`).

### Software

- ROCm 7.2.0 (`gfx942`), Docker, (optional for dual-node) SLURM.
- Container image: `tasimage/primus-odc:v26.2` (ROCm 7.2.0 based; `primus/core/odc/README.md` recommends the same one).
- **Primus-Turbo GDA dependency**: a Primus-Turbo build with the ODC rocSHMEM operators compiled in (including `odc_rocshmem_host` / `odc_rocshmem_gda`). Dual-node GDA must use the one built with **`GDA_MLX5` on and device-LTO single-partition** (`-Xoffload-linker --lto-partitions=1`, built into `setup.py` by PR #409), otherwise cross-node `getmem` reads 0 → **dual-node `grad_norm=0`**.
- HF offline cache: stage the model/data locally in advance, and set `HF_HOME` + `HF_HUB_OFFLINE=1` for fully offline training. Single-node 1.5B uses `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, dual-node 14B uses `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`, and the SFT data for both is `zai-org/LongAlign-10k`.

**Pre-download (run once on a machine with network access, after which training can be fully offline):**

```bash
export HF_HOME=$HOME/primus_packed/hf_home           # cache lands here
pip install -U "huggingface_hub[cli]"
# models (single-node 1.5B / dual-node 14B)
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
# SFT dataset
huggingface-cli download zai-org/LongAlign-10k --repo-type dataset
# at training time, also set: export HF_HOME=$HOME/primus_packed/hf_home HF_HUB_OFFLINE=1
```

(In newer versions `huggingface-cli` can also be written as `hf download ...`; the cache directory follows `HF_HOME`, and the training side hits it as long as it points at the same `HF_HOME`.)

## Get the code, start the container

```bash
# 1) get the PR #864 branch (ODC algorithm layer + integration, config-item driven)
git clone -b feat/odc-consume-turbo https://github.com/AMD-AGI/Primus.git
# the comm operators come from Primus-Turbo: its source is on official main (PR #409), but a plain pip install won't expose the operators
# (the operators are compiled out by -DDISABLE_ROCSHMEM; you must link rocSHMEM at build time). See "Build Primus-Turbo with ODC operators": build host + GDA
# from source, consumed via PRIMUS_TURBO_PATH + .image_bak

# 2) start one container per physical node (mounting the code and HF cache)
docker run -d --name odc_dev \
  --privileged --network host --ipc host --shm-size 64G \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --group-add video --cap-add SYS_PTRACE --cap-add CAP_SYS_ADMIN \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v "$HOME":"$HOME" \
  tasimage/primus-odc:v26.2 sleep infinity
```

> The mount/device/`--ipc host`/`--security-opt seccomp=unconfined` combination above matches the `odc_dev200` container we used to validate the run (`/dev/kfd`+`/dev/dri` for GPUs, `/dev/infiniband` for dual-node RoCE). **`-v` must be written as `src:dst`**; writing a bare path lets docker create an anonymous empty volume that shadows it.

## Build Primus-Turbo with ODC operators (build from source + `.image_bak` consumption)

> **Why you can't just `pip install @main` (the most important point in this section).** Although the operator source is on official `main` (PR #409), it is gated by `#ifndef DISABLE_ROCSHMEM`: `setup.py` only compiles the operators in when `ROCSHMEM_HOME` (+`MPI_HOME`) points at a **prebuilt rocSHMEM static library**, otherwise it adds `-DDISABLE_ROCSHMEM` and compiles the whole section out. The image has no rocSHMEM and pip does not set `ROCSHMEM_HOME`, so a plain `pip install @main` yields a hollow package with `has_host=False has_gda=False` (reproduced on three nodes). Switching branches with `pip … @feat/odc-rocshmem-dist` doesn't work either — that branch exists only on a personal fork, not in the official repo.

**The correct approach, and the one used for all results in the blog: build Primus-Turbo from source, pointing `ROCSHMEM_HOME` at prebuilt rocSHMEM at build time, building one for single-node / one for dual-node.** rocSHMEM is a **non-pip external dependency**, whose `librocshmem.a` must be prebuilt separately for the host/XGMI-IPC and GDA/MLX5 paths (denoted `rocshmem_single` / `rocshmem_gda`). Run once **inside each node's container**:

```bash
# inside the container (using the image's own ROCm torch, hence --no-build-isolation, to avoid pip pulling a non-ROCm torch)
export MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi     # openmpi shipped in the image
export GPU_ARCHS="gfx942"                             # fill per card: gfx942 / gfx950

git clone https://github.com/AMD-AGI/Primus-Turbo.git Primus-Turbo-single   # host/single-node build
git clone https://github.com/AMD-AGI/Primus-Turbo.git Primus-Turbo          # GDA/dual-node build

# host build (single-node XGMI-IPC): link the host rocSHMEM
cd Primus-Turbo-single && ROCSHMEM_HOME=<path>/rocshmem_single \
  pip3 install --no-build-isolation -e ".[pytorch]" -v && cd ..
# GDA build (multi-node GPU-Direct): link the GDA rocSHMEM (--lto-partitions=1 already built into setup.py)
cd Primus-Turbo && ROCSHMEM_HOME=<path>/rocshmem_gda \
  pip3 install --no-build-isolation -e ".[pytorch]" -v && cd ..
```

After installation, the site-packages of the image `tasimage/primus-odc:v26.2` may still hold a **stock `primus_turbo` without ODC operators**, which will **import first and shadow** the dev tree you want to use. Use `.image_bak` to rename and disable it, letting the one on `PRIMUS_TURBO_PATH` win, then verify the operators are present:

```bash
# de-shadow: rename the stock primus_turbo (do this once in each node's container, not skippable)
SP=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
[ -e "$SP/primus_turbo" ] && [ ! -L "$SP/primus_turbo" ] && mv "$SP/primus_turbo" "$SP/primus_turbo.image_bak"
for d in "$SP"/primus_turbo-*.dist-info; do [ -e "$d" ] && mv "$d" "$d.image_bak"; done

# verify the ODC operators are present (host=single-node XGMI-IPC, gda=multi-node GPU-Direct) — both dev trees should be True True
PYTHONPATH=<path>/Primus-Turbo-single python -c "import primus_turbo.pytorch._C as C; \
  print('single:', hasattr(C,'odc_rocshmem_host'), hasattr(C,'odc_rocshmem_gda'))"
PYTHONPATH=<path>/Primus-Turbo        python -c "import primus_turbo.pytorch._C as C; \
  print('gda   :', hasattr(C,'odc_rocshmem_host'), hasattr(C,'odc_rocshmem_gda'))"
# expected: both lines are: True True
```

- **Use a separate dev tree for single-node / dual-node** (`Primus-Turbo-single` / `Primus-Turbo`): the two link rocSHMEM with different symmetric-heap / transport backends (host/IPC vs GDA/MLX5), so **they cannot share the same package**; mixing them throws a symmetric-heap error (see the "Pitfalls checklist" below). At runtime, the single-node section below uses `PRIMUS_TURBO_PATH=<...>/Primus-Turbo-single`, and the dual-node section uses `<...>/Primus-Turbo`.
- `PRIMUS_TURBO_PATH` **cannot be omitted**: `run_odc.sh` prepends it to `PYTHONPATH` so that `import primus_turbo` hits the dev tree with the ODC operators; relying on site-packages alone (the stock version or the pip @main version) yields `has_gda=False`.
- The `.image_bak` de-shadowing **cannot be omitted either**: without renaming, the stock version in site-packages imports first and shadows the dev tree.

## Turning on ODC = changing config items (the core of this refactor)

**ODC's switches are now all in yaml config, no longer `export ODC_ENABLE=1`.** The relevant config items (defaults in `primus/configs/modules/megatron/trainer_base.yaml` and `sft_trainer.yaml`):

| Config | Effect | Old env replaced | Default |
|---|---|---|---|
| `enable_odc` | ODC **master switch** (true routes gradients to P2P; false = pure FSDP2+RCCL, all ODC patches no-op) | `ODC_ENABLE` | `false` |
| `odc_phase` | integration depth, `2` = full gradient routing (production value) | `ODC_PHASE` | `2` |
| `enable_odc_lb_mini` | LB-Mini variable-length KK balancing of **data** (**decoupled** from `enable_odc`, can take effect independently): `enable_odc: true` → **decoupled** (ranks may have unequal microbatch counts = nopad, needs ODC P2P); `enable_odc: false` → **aligned** (`all_reduce(MAX)` forces equal microbatch counts across ranks, standard FSDP2+RCCL lockstep = the RCCL-safe "same-data" baseline) | `ODC_LB_MINI` / `LB_MINI_FORCE_DATA` | `false` |
| `lb_mini_cost_model` / `lb_mini_max_token_len` | LB-Mini cost model / per-microbatch token cap | `LB_MINI_*` | `linear` / `0` |
| `odc_p2p_backend` | P2P backend: `rocshmem` (primary in this post, validated) ｜ `mori` | `ODC_P2P_BACKEND` | `mori` |
| `odc_rocshmem_gda` | rocSHMEM GPU-Direct (GDA) device path, **must be on for dual-node** | `ODC_ROCSHMEM_GDA` | `false` |
| `odc_gda_defer_reduce` / `odc_gda_warmup_mode` / `odc_gda_stride_bytes` / `odc_gda_pipe` | GDA settle deferral / warmup / stride / pipeline depth | `ODC_GDA_*` | `auto` / `strided` / `65536` / `1` |
| `odc_grad_spike_threshold` | step-skip threshold for occasional gradient spikes from async reduce | — | `1000.0` |

Key points:

- **The `odc_nopad` arm** = `enable_odc: true` + `enable_odc_lb_mini: true` + `odc_p2p_backend: rocshmem` (add `odc_rocshmem_gda: true` for dual-node). The two example configs `deepseek1.5B-odc-lbmini.yaml`, `qwen14B-odc-dn.yaml` **already include all of these** (`enable_odc: true` / `odc_phase: 2` / `enable_odc_lb_mini: true` / `odc_p2p_backend: rocshmem`, and the dual-node version also carries `odc_rocshmem_gda: true`) — **ready to use out of the box, no need to manually add the backend**. (Note that in `trainer_base.yaml` the **global default of `odc_p2p_backend` is still `mori`**, so if you write your own config remember to set `rocshmem`.)
- **The `nccl_pad` aligned baseline arm** = `enable_odc: false` + **`enable_odc_lb_mini: true`** (standard FSDP2 + RCCL). ⚠️ **`enable_odc_lb_mini: true` cannot be omitted (key to a fair comparison)**: when `enable_odc: false`, it switches to **aligned mode** (`all_reduce(MAX)` forces equal microbatch counts across ranks so collectives can be safely called per-microbatch), letting the baseline consume **exactly the same LB-Mini variable-length data as `odc_nopad`** (the compliant replacement for the old env `LB_MINI_FORCE_DATA`).
- **The `mori` backend has a known bug on dual-node; never use it for dual-node**; all ODC results in this post use `rocshmem`.

Unified entry point `run_odc.sh` (same script for single-node / dual-node, used as a fallback path):

```text
run_odc.sh <mori|rocshmem> <pad|nopad> <exp_yaml_relpath> <exp_name> [KEY=VAL ...]
```

The 1st positional arg is only responsible for laying down the rocSHMEM runtime infra env (heap / bootstrap ifname / a fresh `TRITON_CACHE_DIR` per run); **the backend is actually selected by the yaml's `odc_p2p_backend`**; the 2nd positional arg `pad|nopad` is now a no-op (alignment/decoupling is now auto-derived from `enable_odc`); the remaining `KEY=VAL` only pass infrastructure env (ODC feature switches all go in the yaml). The script lays down `PYTHONPATH` (including the `odc_early` shim and `PRIMUS_TURBO_PATH`) and then calls `run_pretrain.sh`; it **does not pass through CLI overrides**, so change batch/switches in the yaml.

## Single-node 1.5B `odc_nopad` (8 GPUs, host / XGMI-IPC)

**① Config**: the example config `examples/megatron/configs/MI355X/deepseek1.5B-odc-lbmini.yaml` is **ready out of the box** — it includes `enable_odc: true` / `odc_phase: 2` / `enable_odc_lb_mini: true` / **`odc_p2p_backend: rocshmem`** / `lb_mini_cost_model: fit` / `global_batch_size: 16`, with no changes needed (`odc_p2p_backend: rocshmem` is built in; the global default in `trainer_base.yaml` is `mori`).

For the aligned baseline arm, copy one and **only turn off `enable_odc`, keeping `enable_odc_lb_mini: true`** (so the baseline consumes the same variable-length data as odc_nopad, just with microbatch counts aligned):

```bash
cd Primus
cp examples/megatron/configs/MI355X/deepseek1.5B-odc-lbmini.yaml \
   examples/megatron/configs/MI355X/deepseek1.5B-nccl.yaml
# in the overrides: of deepseek1.5B-nccl.yaml:
#   enable_odc: false          # turn off ODC communication
#   enable_odc_lb_mini: true   # keep it! => aligned mode, feed the same LB-Mini variable-length data (fair comparison)
# (do not set odc_p2p_backend; the backend item is meaningless when enable_odc: false)
```

**② Start training** (inside the container, single-node 8 GPUs):

```bash
export PRIMUS_TURBO_PATH=/path/to/Primus-Turbo-single      # single-node IPC Turbo
export NNODES=1 GPUS_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29700
export HF_HOME=$HOME/primus_packed/hf_home HF_HUB_OFFLINE=1 PRIMUS_SKIP_PIP=1
cd Primus
RUN=primus/core/odc/rocshmem_runtime/scripts/run_odc.sh

# odc_nopad (rocSHMEM host/XGMI-IPC) — the 1st positional arg rocshmem lays down the rocSHMEM infra env,
# the backend is actually selected by odc_p2p_backend: rocshmem in the yaml; the 2nd positional arg nopad is now a no-op
bash "$RUN" rocshmem nopad examples/megatron/configs/MI355X/deepseek1.5B-odc-lbmini.yaml odc_nopad

# nccl_pad aligned baseline (yaml: enable_odc: false + enable_odc_lb_mini: true
#  → standard FSDP2 + RCCL, fed the same LB-Mini variable-length data as odc_nopad, with microbatch counts aligned)
bash "$RUN" rocshmem nopad examples/megatron/configs/MI355X/deepseek1.5B-nccl.yaml nccl_pad
```

Single-node goes over intra-node XGMI + HIP-IPC, and reduce-scatter-accumulate is a device-side owner-pull (on-chip fp32 summation, no host watcher subprocess). To sweep different batches, change `global_batch_size` in the yaml (`run_odc.sh` does not pass through CLI overrides).

## Dual-node 14B `odc_nopad` (16 GPUs, rocSHMEM GDA)

Dual-node **starts one rank group per node** (NODE_RANK 0/1, 8 GPUs each → 16 ranks total), and rocSHMEM uses **uid-over-socket** bootstrap (after PR #864 it no longer uses MPI, launching purely via torchrun). The **preferred, and end-to-end validated in this post** dual-node path is `primus-cli slurm srun`: one line fans out to the two nodes, each `docker run --rm` starts a fresh container and then drives `torchrun`, **with no persistent container and no `rank_node.sh`** (if the cluster hostname maps to loopback and this can't be used, see the `srun --overlap` fallback at the end). Because the feature switches are now config items, the `--env` list is reduced to **only genuine infrastructure variables**: import paths, the NCCL control plane, the rocSHMEM runtime (heap/ifname/HCA/GID), the HF offline cache, and the SFT-skip trio for bypassing primus-cli's built-in hook.

**① Config**: `qwen14B-odc-dn.yaml` is **ready out of the box** — it includes `enable_odc: true` / `odc_phase: 2` / `enable_odc_lb_mini: true` / **`odc_p2p_backend: rocshmem`** / **`odc_rocshmem_gda: true`**, with no changes needed (the `--odc_p2p_backend`/`--odc_rocshmem_gda` overrides in the command below are just redundant insurance). GDA's `warmup=strided` / `stride=65536` / `defer_reduce=auto` / `pipe=1` are all config defaults; `defer_reduce: auto` automatically defers the per-microbatch reduce-scatter to a single rendezvous at the end of the minibatch when `n_pes>local_world_size` (16>8), avoiding cross-node barrier deadlock for nopad variable-length microbatches.

```bash
# orchestration node. JOB = the Slurm job id already holding these two nodes (keepalive)
JOB=<slurm_jobid>; PORT=29604
ROOT=<PRIMUS_ROOT>                                       # the Primus worktree with #864
TURBO=/path/to/Primus-Turbo                              # dual-node GDA Turbo (lto=1)
PYDEPS=/path/to/pydeps                                   # flydsl, etc.
SP=/opt/venv/lib/python3.12/site-packages
HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9   # adjust per cluster NIC
TS=$(date +%Y%m%d_%H%M%S)

# primus-cli's built-in pretrain hook lacks the SFT skip; pre-place a .done flag so it skips prep (see pitfall ③)
touch /home/botahu/primus_packed/pcli_notok_bookcorpus.done

cd "$ROOT"
./primus-cli slurm srun --jobid="$JOB" --overlap -N2 --ntasks-per-node=1 \
  -p amd-rccl --nodelist=<nodeA>,<nodeB> \
  -- --image tasimage/primus-odc:v26.2 --clean \
     --privileged --network host --ipc host --shm-size 64G \
     --device /dev/kfd --device /dev/dri --device /dev/infiniband \
     --group-add video --cap-add SYS_PTRACE --cap-add CAP_SYS_ADMIN \
     --security-opt seccomp=unconfined \
     --volume /home/botahu:/home/botahu \
     `# --- import paths / turbo / linker (primus-cli doesn't run setup_pythonpath, must fill in) ---` \
     --env PYTHONPATH="$ROOT:$PYDEPS:$TURBO:$ROOT/primus/core/odc/odc_early:$ROOT/primus/core:$SP" \
     --env PRIMUS_TURBO_PATH="$PYDEPS:$TURBO" \
     --env LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/openmpi/lib \
     `# --- NCCL control plane (override the single-node lo / IB-off defaults) ---` \
     --env NCCL_SOCKET_IFNAME=eth0 --env GLOO_SOCKET_IFNAME=eth0 \
     --env NCCL_IB_DISABLE=0 --env NCCL_IB_GID_INDEX=3 \
     `# --- rocSHMEM runtime infra (heap in raw bytes / bootstrap / provider / NIC / GID) ---` \
     --env ROCSHMEM_HEAP_SIZE=8589934592 --env ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME=eth0 \
     --env ROCSHMEM_GDA_PROVIDER=mlx5 --env ROCSHMEM_HCA_LIST="$HCA" \
     --env ROCSHMEM_IB_GID_INDEX=3 --env ROCSHMEM_ROCE_GID_INDEX=3 --env ROCSHMEM_GID_INDEX=3 \
     `# --- create a fresh triton cache per run (avoid reusing mismatched device kernels → NaN) ---` \
     --env TRITON_CACHE_DIR=/tmp/tcache_rocshmem_pcli_$TS \
     `# --- HF offline cache + node-local cache ---` \
     --env HF_HOME=/home/botahu/primus_packed/hf_home \
     --env HF_HUB_CACHE=/home/botahu/primus_packed/hf_models \
     --env HF_DATASETS_CACHE=/home/botahu/primus_packed/hf_home/datasets \
     --env HF_HUB_OFFLINE=1 --env DATA_PATH=/workspace \
     --env PRIMUS_PACK_CACHE_DIR=/home/botahu/primus_packed \
     --env PRIMUS_PACK_LOCK_DIR=/tmp/primus_lock \
     --env PRIMUS_CACHE_ROOT=/tmp/primus_cache_pcli_$TS --env PRIMUS_SKIP_PIP=1 \
     `# --- bypass the built-in pretrain hook's SFT prep (see pitfall ③) ---` \
     --env TOKENIZED_DATA_PATH=/home/botahu/primus_packed/pcli_notok_bookcorpus \
     --env HF_TOKEN=hf_offline_dummy \
  -- -- train pretrain --config examples/megatron/configs/MI355X/qwen14B-odc-dn.yaml \
     --backend_path "$ROOT/third_party/Megatron-LM" \
     `# --- ODC feature switches: config items, passed directly as CLI overrides (no longer via --env) ---` \
     --odc_p2p_backend rocshmem --odc_rocshmem_gda true --enable_fused_linear_ce true \
     --train_iters 50 --global_batch_size 128 --micro_batch_size 1 \
     --use_torch_fsdp2 True --manual_gc True \
     --profile false --use_pytorch_profiler false \
     --disable_wandb True --disable_tensorboard True --disable_last_saving True
```

The three `--` in `primus-cli slurm srun` separate, in order: **slurm/srun args** ｜ **docker/container args (including all `--env`)** ｜ **`train pretrain` training args (including ODC config overrides)**. `slurm-entry` will automatically inject `MASTER_ADDR/MASTER_PORT/NNODES/NODE_RANK/GPUS_PER_NODE` into each node. `ROCSHMEM_HCA_LIST` is the RoCE NIC allowlist (this cluster has 10 mlx5 cards; `mlx5_1/_6` are eth management ports and must be excluded).

**For the `nccl_pad` aligned baseline**: replace the ODC overrides in the last segment with `--enable_odc false --enable_odc_lb_mini true` (drop `--odc_p2p_backend`/`--odc_rocshmem_gda`, everything else unchanged), i.e., standard FSDP2 + RCCL. ⚠️ **`--enable_odc_lb_mini true` cannot be omitted**: when `enable_odc: false`, it automatically switches to **aligned mode**, feeding the baseline **exactly the same LB-Mini variable-length data as `odc_nopad`** (each rank's microbatch count aligned via `all_reduce(MAX)`, RCCL-safe).

### Three iron rules of correctness (not skippable)

- **RoCE-adjacent nodes**: be sure to pick adjacent nodes under the same leaf switch, mutually reachable over RoCE; non-adjacent node pairs often stall at the first cross-node GDA warmup (a long time with no `iteration`).
- **device-LTO single-partition**: the GDA Turbo must be built with `-Xoffload-linker --lto-partitions=1` (built in by PR #409), otherwise cross-node `getmem` reads 0 → **dual-node `grad_norm=0`**.
- **dedicated dual-node GDA Turbo**: use the `$TURBO` relinked to `rocshmem_gda`, kept separate from the single-node IPC build — the two have different symmetric-heap/connection backends, and mixing them clashes on the symmetric heap.

### 3 pitfalls of the direct `primus-cli` path

1. **`--volume` must be written as `src:dst`** (`--volume /home/botahu:/home/botahu`): writing a bare path lets docker create an anonymous empty volume that shadows it → the container can't see the code/cache.
2. **`PYTHONPATH` must be injected in full**: the direct primus-cli path **skips `setup_pythonpath`**, so you must fill in repo + pydeps + **GDA Turbo** (`$TURBO`, placing it ahead of site-packages lets the GDA operators win, `has_gda=True`, without needing `.image_bak` in the ephemeral container) + `odc/odc_early` + `primus/core` + site-packages, otherwise `import odc`/`import primus_turbo` fails.
3. **the built-in pretrain hook lacks the SFT skip**: the built-in runner hook (`train/pretrain/megatron/prepare.py`) has no `stage=='sft'` skip branch and will demand `HF_TOKEN` + download bookcorpus. Fix: pre-`touch` a `.done` flag and point `TOKENIZED_DATA_PATH` at it, give `HF_TOKEN` a placeholder value, so the hook skips prep (the `--train_data_path` it emits is harmless to the SFT provider).

> **A `slurm-entry` MASTER_ADDR caveat**: `slurm-entry` forces `MASTER_ADDR=`short-hostname. In the cluster validated for this post, the `127.0.1.1` line in `/etc/hosts` is commented out and the short hostname resolves to a routable IP, so it works directly; if the cluster maps the hostname to loopback (`127.0.1.1 <hostname>`), rendezvous binds to the loopback and cross-node connections fail, in which case use the `srun --overlap` fallback below.

### Fallback: `srun --overlap` + `docker exec` + `run_odc.sh`

If the cluster hostname maps to loopback, or you want to reuse the environment preset in a persistent container, you can fall back to **`srun --overlap` + `docker exec` + `run_odc.sh`** (delivering one rank per node into the persistent container), running on each of the two nodes:

```bash
export NNODES=2 GPUS_PER_NODE=8 NODE_RANK=<0|1> MASTER_ADDR=<nodeA-routable-IP> MASTER_PORT=<port>
bash run_odc.sh rocshmem nopad examples/megatron/configs/MI355X/qwen14B-odc-dn.yaml odc_nopad \
  NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3 \
  ROCSHMEM_GDA_PROVIDER=mlx5 ROCSHMEM_HCA_LIST=$HCA \
  ROCSHMEM_IB_GID_INDEX=3 ROCSHMEM_ROCE_GID_INDEX=3 ROCSHMEM_GID_INDEX=3
```

Here `KEY=VAL` only passes infrastructure env, while ODC feature switches still come from the yaml (`qwen14B-odc-dn.yaml` already has them all); the baseline is run the same way with a copy that sets `enable_odc: false` + keeps `enable_odc_lb_mini: true`, via `... <baseline_yaml> nccl_pad`. It is more robust to hostname/loopback (the 0→1 first bring-up used this path), with a wrapping example in `full_exp/matrix_dual_harness/`.

## Judging success, reading results, computing the speedup

**Step one — confirm ODC really is running on rocSHMEM GDA.** In the training log, check these real markers (all below are taken from the actual log of this post's dual-node primus-cli validation run):

1. **Config items in effect (enable_odc + backend)**:

   ```text
   [Primus:Runtime] Applying CLI overrides: {'odc_p2p_backend': 'rocshmem', 'odc_rocshmem_gda': True, ...}
   ```

2. **rocSHMEM GDA backend is up (fix ①)** — the runtime evidence that `has_gda=True` is truly in effect:

   ```text
   init_shmem (rocSHMEM GDA, uid bootstrap): my_pe=15 n_pes=16
   [GDA] reduce-scatter warm-up mode=strided stride_bytes=65536
   ```

   You should also see a batch of `Rank N create tensor gda_acc_* / gda_in_* / gda_scr_*` GDA symmetric-heap tensors. If the backend runs as `mori` (no such `rocSHMEM GDA` / `gda_*` lines), it means `odc_p2p_backend` was not set correctly.

3. **Hooked into the correct FSDP2 class (fix ②)** + **nopad decoupled data**:

   ```text
   [PRE-FORWARD canary] wrap-chain: PrimusTorchFullyShardedDataParallel.module=FSDPFloat16Module.module=GPTModel
   [ODC.lb_mini] built LB-Mini train iterator: ... same_micro_num=False (LB-Mini decoupled (ODC))
   [ODC.torch_fsdp2] hooked optimizer.step (pre_optimizer_step injected)
   ```

   The wrapper must be **`PrimusTorchFullyShardedDataParallel`** (not the old `TorchFullyShardedDataParallel`; hooking the old class crashes in iter2 with `NoneType ... clear_accumulations`); `same_micro_num=False` means nopad/decoupled.
   > Note: the code also has `log_rank_0` markers like `[ODC.torch_fsdp2] runtime config populated ...` and `[Patch] ⊘ Skipped ... device_id` (fix ③, `condition = use_torch_fsdp2 and not enable_odc` being false → skip eager-RCCL); they print at patch registration / early `before_train` and can corroborate but may not necessarily land in the torchrun capture stream, so rely on the lines above. The control group `nccl_pad`, needing an RCCL comm, prints `[Patch] ✓ Applied ... device_id`.

4. **Training advances stably**: `lm loss` **decreases** with iter, **0 nan / 0 crash**, steady-state `ms/iter` is stable, and **`grad norm` is nonzero throughout** — `grad_norm=0` means cross-node GDA `getmem` reads 0 (device-LTO multi-partition), indicating the GDA Turbo was not built correctly.

Quick grab:

```bash
grep -E "rocSHMEM GDA|gda_acc_|reduce-scatter warm-up|LB-Mini decoupled|PrimusTorchFullyShardedDataParallel" "$LOG"
grep -E "lm loss|elapsed time per iteration|grad norm|nan iterations" "$LOG" | tail -40
```

**Step two — read the key metrics, compute the speedup.** Each step's log prints `iteration N/M | ... | elapsed time per iteration (ms): X | ... | lm loss: ...`, plus `compute per GPU (TFLOP/s/GPU)` and `tokens/s/GPU`. Take the median `ms/iter` of the stable segment (skip the first few warmup steps), then:

> speedup = `nccl_pad's ms/iter` ÷ `odc_nopad's ms/iter` (>1 means faster than RCCL)

**Success criteria**: 0 nan throughout, loss decreasing step by step, running the full target step count.

**Final validation results** (config-driven: ODC via `enable_odc`, `nccl_pad` fed the same variable-length data via **aligned** `enable_odc_lb_mini`; strictly same-source apples-to-apples — same image v26.2, same nodes, differing only in the arm):

| Scenario | odc_nopad (config) | aligned nccl_pad | Speedup | loss |
|---|---|---|---|---|
| Dual-node 14B, gbs128 | ≈ **70.5k** ms/iter (primus-cli measured median) | ≈ **80.2k** ms/iter | ≈ **1.15×** | 12.07→9.34, grad_norm nonzero throughout, 0 nan |
| Single-node 1.5B, gbs16 | ≈ **2.1k** ms/iter | ≈ **2.5k** ms/iter | ≈ **1.19–1.21×** | converges to ~10.0, 0 nan |

> The dual-node 14B row is end-to-end validated via this post's **`primus-cli slurm` path**: 0 nan throughout training, `grad_norm` nonzero, steady-state median ~70.5k ms/iter, consistent with the earlier `srun --overlap` fallback path (~71k) — the two paths behave equivalently.

## Pitfalls checklist (strongly recommended reading first)

- **`has_host=False has_gda=False` (fix ①)**: you're using stock turbo or the hollow package from `pip install @main` (operators compiled out by `-DDISABLE_ROCSHMEM`). The correct fix is in "Build Primus-Turbo with ODC operators": build from source, point `ROCSHMEM_HOME` at prebuilt rocSHMEM at build time, then confirm `has_gda==True` via `PRIMUS_TURBO_PATH` (or `.image_bak` de-shadowing in site-packages).
- **iter2 crashes with `'NoneType' ... clear_accumulations` (fix ②)**: you hooked the old FSDP2 class. Confirm Primus is on the `feat/odc-consume-turbo` branch, and that `PrimusTorchFullyShardedDataParallel` appears in the wrap-chain.
- **runs but single-node is ~6% slower (fix ③ not in effect)**: confirm `enable_odc: true` is truly in place and the device_id patch is `⊘ Skipped` (the gate is the `enable_odc` config item, not `ODC_ENABLE`).
- **backend runs as mori**: the global default in `trainer_base.yaml` is `mori`, so a self-written config that forgets to set `rocshmem` will run mori (**dual-node mori has a known bug**); the absence of `rocSHMEM GDA`/`gda_*` lines in the log identifies it.
- **NaN on the very first step**: reused a Triton cache from a heterogeneous toolchain. Use a **fresh timestamped** `TRITON_CACHE_DIR` per run.
- **dual-node `grad_norm=0`**: device-LTO multi-partition; the dual-node GDA Turbo must use `-Xoffload-linker --lto-partitions=1` (built in by PR #409).
- **rocSHMEM heap is a raw byte count**: `ROCSHMEM_HEAP_SIZE` does not accept K/M/G suffixes (8 GiB = `8589934592`).
- **data packing is somewhat slow**: at large gbs (dual-node gbs128), LB-Mini packing may take about 10 minutes before the first `iteration`, which is normal, not a hang.
