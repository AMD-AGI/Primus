# Configuration reference

YAML, env, and CLI for [SpecForge on Primus](README.md). Copy a recipe under
[`configs/`](configs/) and change the fields below. Draft architecture, data
format, optimizer, export, and SGLang serving are SpecForge's — see the
[AMD ROCm tutorial](https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/AMD/amd_rocm.md)
(§4 for online disaggregated).

Set a value in the experiment YAML, or override it on the CLI with the same
path (`specforge_online.server_count=8`). `${VAR:default}` is expanded when
the experiment file loads. Export the variable, or put the literal in YAML.

```yaml
modules:
  pre_trainer:
    framework: specforge                 # required
    config: offline.yaml                 # offline.yaml | online.yaml
    model: qwen3.5-4b-dflash.yaml        # Primus model preset
    overrides:
      specforge_mode: train              # train | capture  (default train)
      specforge_train_mode: online       # online | offline; required for train
      specforge_config: ...              # SpecForge Hydra YAML; required for train
      specforge_root: ...                # SpecForge checkout (overlay: /workspace/SpecForge)
      output_dir: ...                    # checkpoints / logs
      specforge_overrides: {}            # SpecForge Hydra key=value (train)
      specforge_capture: {}              # capture only
      specforge_online: {}               # online train only
```

Do not set `specforge_role`. Primus assigns producer/consumer from node rank.
Do not pass `--role` on `primus-cli`.

## Capture

[`configs/qwen3.5-4b-dflash-offline-capture.yaml`](configs/qwen3.5-4b-dflash-offline-capture.yaml)

```bash
export CAPTURE_DATA_PATH=/data/sharegpt.jsonl
export OUTPUT_DIR=/data/runs/capture
export NPROC_PER_NODE=8
./runner/primus-cli direct -- train pretrain \
  --config examples/specforge/configs/qwen3.5-4b-dflash-offline-capture.yaml
```

```yaml
work_group: ${PRIMUS_TEAM:amd}
user_name: ${PRIMUS_USER:root}
exp_name: ${PRIMUS_EXP_NAME:qwen3.5-4b-dflash-offline-capture}
workspace: ${PRIMUS_WORKSPACE:./output}

modules:
  pre_trainer:
    framework: specforge
    config: offline.yaml
    model: qwen3.5-4b-dflash.yaml
    overrides:
      specforge_mode: capture
      specforge_root: ${SPECFORGE_ROOT:/workspace/SpecForge}
      output_dir: ${OUTPUT_DIR}

      specforge_capture:
        target_model_path: ${TARGET_MODEL:Qwen/Qwen3.5-4B}
        data_path: ${CAPTURE_DATA_PATH}          # ShareGPT-style JSONL
        output_path: ${OUTPUT_DIR}/hidden_states_raw
        nproc_per_node: ${NPROC_PER_NODE:1}      # GPUs for capture
        filter_output_path: ${OUTPUT_DIR}/hidden_states
        filter_block_size: ${BLOCK_SIZE:16}
        filter_min_kept: ${FILTER_MIN_KEPT:1}    # fail if fewer shards survive
        # Remaining keys are SpecForge prepare_hidden_states flags:
        strategy: dflash
        draft_model_config: configs/qwen3.5-4b-dflash.json
        trust_remote_code: true
        cache_dir: ${OUTPUT_DIR}/sf-cache
        chat_template: qwen3.5
        max_length: 2048
        tp_size: 1
        batch_size: ${CAPTURE_BATCH_SIZE:8}
        sglang_attention_backend: aiter
        sglang_disable_radix_cache: true         # required on this overlay
        sglang_mem_fraction_static: 0.8
        sglang_context_length: 2560
```

Train later with `data.hidden_states_path` pointing at `filter_output_path`
(the filtered shards), not `output_path`.

## Offline train

[`configs/qwen3.5-4b-dflash-offline.yaml`](configs/qwen3.5-4b-dflash-offline.yaml)

```bash
export HIDDEN_STATES_PATH=/data/runs/capture/hidden_states
export OUTPUT_DIR=/data/runs/train
export NPROC_PER_NODE=8
export MAX_STEPS=20
./runner/primus-cli direct -- train pretrain \
  --config examples/specforge/configs/qwen3.5-4b-dflash-offline.yaml
```

```yaml
work_group: ${PRIMUS_TEAM:amd}
user_name: ${PRIMUS_USER:root}
exp_name: ${PRIMUS_EXP_NAME:qwen3.5-4b-dflash-offline}
workspace: ${PRIMUS_WORKSPACE:./output}

modules:
  pre_trainer:
    framework: specforge
    config: offline.yaml
    model: qwen3.5-4b-dflash.yaml
    overrides:
      specforge_mode: train
      specforge_train_mode: offline
      specforge_config: ${SPECFORGE_CONFIG:/workspace/SpecForge/examples/configs/offline/colocated/qwen3.5-4b-dflash-offline-amd.yaml}
      specforge_root: ${SPECFORGE_ROOT:/workspace/SpecForge}
      output_dir: ${OUTPUT_DIR}

      specforge_overrides:                       # SpecForge Hydra; see SpecForge docs
        training.max_steps: ${MAX_STEPS:20}
        training.num_epochs: 1
        training.save_interval: ${MAX_STEPS:20}
        training.log_interval: 5
        model.use_liger_kernel: false            # not in this overlay
        data.hidden_states_path: ${HIDDEN_STATES_PATH}
        deployment.trainer.nproc_per_node: ${NPROC_PER_NODE:1}
```

## Online train

[`configs/qwen3.5-4b-dflash-online-2node.yaml`](configs/qwen3.5-4b-dflash-online-2node.yaml)

`-N` is capture nodes + trainer nodes. `--gres=gpu:N` is per node: SGLang GPUs
on capture nodes (`server_count × server_tp`), trainer processes on trainer
nodes (`trainer_nproc`). A single GPU id `0` expands to `0..N-1`. Rank 0
starts Mooncake and the SpecForge producer; other capture ranks run SGLang
only; trainer ranks train.

Leave `127.0.0.1` in the SpecForge YAML. Primus fills real SGLang and
Mooncake addresses after allocate. Put `run_root` on shared storage and
`consumer_state_dir` on node-local disk. Both must be empty for a new
`RUN_ID`.

```bash
export RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
export RUN_ROOT=/path/to/shared/run/$RUN_ID
export OUTPUT_DIR=$RUN_ROOT/output
export CONSUMER_STATE_DIR=/path/to/local/nvme/specforge/$RUN_ID/consumer-state
export TRAIN_DATA_PATH=/path/to/sharegpt_train.jsonl
export MAX_STEPS=20
./runner/primus-cli slurm srun -N 2 --gres=gpu:1 \
  -- container --image primus-specforge:v0.5.14-rocm700-mi35x \
  --volume /shared:/shared \
  -- train pretrain \
  --config examples/specforge/configs/qwen3.5-4b-dflash-online-2node.yaml
```

```yaml
work_group: ${PRIMUS_TEAM:amd}
user_name: ${PRIMUS_USER:root}
exp_name: ${PRIMUS_EXP_NAME:qwen3.5-4b-dflash-online-2node}
workspace: ${PRIMUS_WORKSPACE:./output}

modules:
  pre_trainer:
    framework: specforge
    config: online.yaml
    model: qwen3.5-4b-dflash.yaml
    overrides:
      specforge_mode: train
      specforge_train_mode: online
      specforge_config: ${SPECFORGE_CONFIG:/workspace/SpecForge/examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-amd.yaml}
      specforge_root: ${SPECFORGE_ROOT:/workspace/SpecForge}
      output_dir: ${OUTPUT_DIR}

      specforge_online:
        run_id: ${RUN_ID}                          # required; letters, digits, . _ -
        run_root: ${RUN_ROOT}                      # required; shared
        consumer_state_dir: ${CONSUMER_STATE_DIR}  # required; local NVMe
        capture_nnodes: ${CAPTURE_NNODES:1}
        trainer_nnodes: ${TRAINER_NNODES:1}
        server_count: ${SERVER_COUNT:1}            # SGLang servers per capture node
        server_tp: ${SERVER_TP:1}
        server_gpus: ${SERVER_GPUS:0}              # CSV, or 0 → 0..count*tp-1
        trainer_gpus: ${TRAINER_GPUS:0}            # CSV, or 0 → 0..nproc-1
        trainer_nproc: ${NPROC_PER_NODE:1}
        target_model_path: ${TARGET_MODEL:Qwen/Qwen3.5-4B}
        mooncake_protocol: ${MOONCAKE_PROTOCOL:tcp}
        mooncake_lease_ttl_ms: ${MOONCAKE_LEASE_TTL_MS:500}
        # Optional:
        # capture_layer_ids: 1,8,15,22,29
        # server_port: 30000
        # server_mem_fraction: 0.85
        # sglang_extra_args: --attention-backend aiter --disable-radix-cache
        # mooncake_rpc_port: 35551
        # mooncake_http_port: 35880
        # mooncake_metrics_port: 35903
        # bind_interface: ens3                     # if auto IP is wrong
        # start_timeout_s: 1800
        # peer_timeout_s: 1800

      specforge_overrides:                         # SpecForge Hydra; see SpecForge docs
        training.max_steps: ${MAX_STEPS:20}
        training.num_epochs: 1
        training.save_interval: ${MAX_STEPS:20}
        training.log_interval: 5
        model.use_liger_kernel: false
        data.train_data_path: ${TRAIN_DATA_PATH}
        # runtime.in_flight_high_watermark: 64
        # runtime.in_flight_low_watermark: 32
```

On Ethernet without IB, also pass `NCCL_IB_DISABLE=1` and
`NCCL_SOCKET_IFNAME=<iface>`. To pin IPs instead of auto-detect:

```bash
# capture rank 0 only
export PRIMUS_SPECFORGE_HEAD_IP=10.0.0.1
# every node, or use bind_interface / PRIMUS_SPECFORGE_BIND_IFACE
export PRIMUS_SPECFORGE_LOCAL_IP=10.0.0.2
```

Keep `SGLANG_USE_AITER=1` and `SGLANG_DISABLE_RADIX_CACHE=1` (the overlay
default). Hugging Face cache/token: `HF_HOME`, `HF_TOKEN`.

## CLI

Inside the overlay:

```bash
./runner/primus-cli direct -- train pretrain --config <experiment.yaml> \
  [dotted.overrides...]
```

On Slurm, from the login node, image already loaded:

```bash
./runner/primus-cli slurm srun -N <capture+trainer> --gres=gpu:<gpus-per-node> \
  -- container --image <overlay> \
  --volume <host>:<container> \
  --env KEY=VALUE \
  -- train pretrain --config <experiment.yaml> \
  [dotted.overrides...]
```

Dotted keys match the YAML (`specforge_overrides.training.max_steps=1000`,
`specforge_online.mooncake_lease_ttl_ms=8000`). `--exp` is an alias for
`--config`. `--volume` and `--env` are repeatable. `--backend_path` points at
a SpecForge checkout if it is not `/workspace/SpecForge`.

Slurm `-p`, `-t`, `-o`, `--exclude` and launcher `--debug` / `--dry-run` are
in the [CLI reference](../../docs/02-user-guide/cli-reference.md). Do not wrap
this in `managed_local`. `specforge export` and data-prep scripts stay on the
SpecForge CLI.
