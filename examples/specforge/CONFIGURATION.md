# Configuration reference

YAML, env, and CLI for [SpecForge on Primus](README.md). Copy a recipe under
[`configs/`](configs/) and change the fields below. Draft architecture, data
format, optimizer, export, and SGLang serving are SpecForge's — see the
[AMD ROCm tutorial](https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/AMD/amd_rocm.md)
(§4 for online disaggregated). SpecForge Hydra keys (`specforge_overrides`)
are documented in SpecForge
[`examples/configs/README.md`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/README.md).

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
      specforge_root: ...                # SpecForge checkout (this image: /workspace/SpecForge)
      output_dir: ...                    # checkpoints / logs
      specforge_overrides: {}            # SpecForge Hydra key=value (train)
      specforge_capture: {}              # capture only
      specforge_online: {}               # online train only
```

Do not set `specforge_role`. Primus assigns producer/consumer from node rank.
Do not pass `--role` on `primus-cli`.

## Capture

Example config and usage:
[`configs/qwen3.5-4b-dflash-offline-capture.yaml`](configs/qwen3.5-4b-dflash-offline-capture.yaml)

```bash
export CAPTURE_DATA_PATH=/data/sharegpt.jsonl
export OUTPUT_DIR=/data/runs/capture
export NPROC_PER_NODE=8
./runner/primus-cli direct -- train pretrain \
  --config examples/specforge/configs/qwen3.5-4b-dflash-offline-capture.yaml
```

Full config reference. Extra `specforge_capture` keys are flags for SpecForge
[`scripts/prepare_hidden_states.py`](https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_hidden_states.py).

```yaml
work_group: ${PRIMUS_TEAM:amd}            # Primus experiment metadata
user_name: ${PRIMUS_USER:root}            # Primus experiment metadata
exp_name: ${PRIMUS_EXP_NAME:qwen3.5-4b-dflash-offline-capture}  # run name
workspace: ${PRIMUS_WORKSPACE:./output}  # Primus workspace

modules:
  pre_trainer:
    framework: specforge                 # required backend
    config: offline.yaml                 # Primus module preset
    model: qwen3.5-4b-dflash.yaml        # Primus model preset
    overrides:
      specforge_mode: capture            # train | capture
      specforge_root: ${SPECFORGE_ROOT:/workspace/SpecForge}  # SpecForge checkout
      output_dir: ${OUTPUT_DIR}          # checkpoints / logs

      specforge_capture:
        target_model_path: ${TARGET_MODEL:Qwen/Qwen3.5-4B}  # HF id or local path
        data_path: ${CAPTURE_DATA_PATH}  # ShareGPT-style JSONL
        output_path: ${OUTPUT_DIR}/hidden_states_raw  # raw shards
        nproc_per_node: ${NPROC_PER_NODE:1}  # GPUs for capture
        filter_output_path: ${OUTPUT_DIR}/hidden_states  # filtered shards (train on these)
        filter_block_size: ${BLOCK_SIZE:16}  # DFlash block size
        filter_min_kept: ${FILTER_MIN_KEPT:1}  # fail if fewer shards survive
        strategy: dflash                 # SpecForge capture strategy
        draft_model_config: configs/qwen3.5-4b-dflash.json  # relative to specforge_root
        trust_remote_code: true          # allow custom HF code
        cache_dir: ${OUTPUT_DIR}/sf-cache  # download / KV cache
        chat_template: qwen3.5           # chat template name
        max_length: 2048                 # prompt + completion tokens
        tp_size: 1                       # SGLang tensor parallel
        batch_size: ${CAPTURE_BATCH_SIZE:8}  # capture batch
        sglang_mem_fraction_static: 0.8  # SGLang GPU memory fraction
        sglang_context_length: 2560      # SGLang context length
```

Train later with `data.hidden_states_path` pointing at `filter_output_path`
(the filtered shards), not `output_path`.

## Offline train

Example config and usage:
[`configs/qwen3.5-4b-dflash-offline.yaml`](configs/qwen3.5-4b-dflash-offline.yaml)

```bash
export HIDDEN_STATES_PATH=/data/runs/capture/hidden_states
export OUTPUT_DIR=/data/runs/train
export NPROC_PER_NODE=8
export MAX_STEPS=20
./runner/primus-cli direct -- train pretrain \
  --config examples/specforge/configs/qwen3.5-4b-dflash-offline.yaml
```

Full config reference. `specforge_overrides` keys are SpecForge Hydra — see
[`examples/configs/README.md`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/README.md).

```yaml
work_group: ${PRIMUS_TEAM:amd}            # Primus experiment metadata
user_name: ${PRIMUS_USER:root}            # Primus experiment metadata
exp_name: ${PRIMUS_EXP_NAME:qwen3.5-4b-dflash-offline}  # run name
workspace: ${PRIMUS_WORKSPACE:./output}  # Primus workspace

modules:
  pre_trainer:
    framework: specforge                 # required backend
    config: offline.yaml                 # Primus module preset
    model: qwen3.5-4b-dflash.yaml        # Primus model preset
    overrides:
      specforge_mode: train              # train | capture
      specforge_train_mode: offline      # online | offline
      specforge_config: ${SPECFORGE_CONFIG:/workspace/SpecForge/examples/configs/offline/colocated/qwen3.5-4b-dflash-offline-amd.yaml}  # SpecForge Hydra YAML
      specforge_root: ${SPECFORGE_ROOT:/workspace/SpecForge}  # SpecForge checkout
      output_dir: ${OUTPUT_DIR}          # checkpoints / logs

      specforge_overrides:
        training.max_steps: ${MAX_STEPS:20}  # optimizer steps
        training.num_epochs: 1           # epoch cap (max_steps usually wins first)
        training.save_interval: ${MAX_STEPS:20}  # checkpoint every N steps
        training.log_interval: 5         # log every N steps
        model.use_liger_kernel: false    # not shipped in this image
        data.hidden_states_path: ${HIDDEN_STATES_PATH}  # filtered capture output
        deployment.trainer.nproc_per_node: ${NPROC_PER_NODE:1}  # trainer GPUs
```

## Online train

Example config and usage:
[`configs/qwen3.5-4b-dflash-online-2node.yaml`](configs/qwen3.5-4b-dflash-online-2node.yaml)

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

`-N` is capture nodes + trainer nodes. `--gres=gpu:N` is per node: SGLang GPUs
on capture nodes (`server_count × server_tp`), trainer processes on trainer
nodes (`trainer_nproc`). A single GPU id `0` expands to `0..N-1`. Rank 0
starts Mooncake and the SpecForge producer; other capture ranks run SGLang
only; trainer ranks train.

The SpecForge Hydra file (`specforge_config`) lists SGLang and Mooncake as
`http://127.0.0.1:…`. Leave those loopback URLs in that file. After the job
allocates, Primus rewrites them to the capture nodes. Put `run_root` on shared
storage and `consumer_state_dir` on node-local disk. Both must be empty for a
new `RUN_ID`.

Full config reference. `specforge_overrides` keys are SpecForge Hydra — see
[`examples/configs/README.md`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/README.md).
Online layout is also in the [AMD ROCm tutorial](https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/AMD/amd_rocm.md)
§4.

```yaml
work_group: ${PRIMUS_TEAM:amd}            # Primus experiment metadata
user_name: ${PRIMUS_USER:root}            # Primus experiment metadata
exp_name: ${PRIMUS_EXP_NAME:qwen3.5-4b-dflash-online-2node}  # run name
workspace: ${PRIMUS_WORKSPACE:./output}  # Primus workspace

modules:
  pre_trainer:
    framework: specforge                 # required backend
    config: online.yaml                  # Primus module preset
    model: qwen3.5-4b-dflash.yaml        # Primus model preset
    overrides:
      specforge_mode: train              # train | capture
      specforge_train_mode: online       # online | offline
      specforge_config: ${SPECFORGE_CONFIG:/workspace/SpecForge/examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-amd.yaml}  # SpecForge Hydra YAML
      specforge_root: ${SPECFORGE_ROOT:/workspace/SpecForge}  # SpecForge checkout
      output_dir: ${OUTPUT_DIR}          # checkpoints / logs

      specforge_online:
        run_id: ${RUN_ID}                # required; letters, digits, . _ -
        run_root: ${RUN_ROOT}            # required; shared control dir
        consumer_state_dir: ${CONSUMER_STATE_DIR}  # required; local NVMe
        capture_nnodes: ${CAPTURE_NNODES:1}  # capture ranks
        trainer_nnodes: ${TRAINER_NNODES:1}  # trainer ranks; -N = capture + trainer
        server_count: ${SERVER_COUNT:1}  # SGLang servers per capture node
        server_tp: ${SERVER_TP:1}        # tensor parallel per SGLang server
        server_gpus: ${SERVER_GPUS:0}    # SGLang GPU ids; 0 means 0,1,…,server_count*server_tp-1
        trainer_gpus: ${TRAINER_GPUS:0}  # trainer GPU ids; 0 means 0,1,…,trainer_nproc-1
        trainer_nproc: ${NPROC_PER_NODE:1}  # trainer processes per trainer node
        target_model_path: ${TARGET_MODEL:Qwen/Qwen3.5-4B}  # HF id or local path
        mooncake_protocol: ${MOONCAKE_PROTOCOL:tcp}  # Mooncake transport
        mooncake_lease_ttl_ms: ${MOONCAKE_LEASE_TTL_MS:500}  # KV lease TTL
        # capture_layer_ids: 1,8,15,22,29  # else from SpecForge draft json
        # server_port: 30000             # first SGLang HTTP port
        # server_mem_fraction: 0.85      # SGLang GPU memory fraction
        # mooncake_rpc_port: 35551       # Mooncake RPC
        # mooncake_http_port: 35880      # Mooncake HTTP metadata
        # mooncake_metrics_port: 35903   # Mooncake metrics
        # bind_interface: ens3           # NIC if auto IP is wrong
        # start_timeout_s: 1800          # sidecar start wait
        # peer_timeout_s: 1800           # wait for the other ranks

      specforge_overrides:
        training.max_steps: ${MAX_STEPS:20}  # optimizer steps
        training.num_epochs: 1           # epoch cap (max_steps usually wins first)
        training.save_interval: ${MAX_STEPS:20}  # checkpoint every N steps
        training.log_interval: 5         # log every N steps
        model.use_liger_kernel: false    # not shipped in this image
        data.train_data_path: ${TRAIN_DATA_PATH}  # ShareGPT-style JSONL
        # runtime.in_flight_high_watermark: 64  # SpecForge online queue
        # runtime.in_flight_low_watermark: 32
```

On Ethernet without IB, pass `NCCL_IB_DISABLE=1` and
`NCCL_SOCKET_IFNAME=<iface>`.

Skip the next block unless auto-detected IPs are wrong (Mooncake/SGLang not
reachable across nodes). Then set the cluster NIC, or pin IPs:

```bash
# NIC name, or:
export PRIMUS_SPECFORGE_BIND_IFACE=ens3
# capture rank 0:
export PRIMUS_SPECFORGE_HEAD_IP=10.0.0.1
# every node:
export PRIMUS_SPECFORGE_LOCAL_IP=10.0.0.2
```

## CLI

Inside the docker image:

```bash
./runner/primus-cli direct -- train pretrain --config <experiment.yaml> \
  [dotted.overrides...]
```

On Slurm, from the login node, image already loaded:

```bash
./runner/primus-cli slurm srun -N <capture+trainer> --gres=gpu:<gpus-per-node> \
  -- container --image <tag> \
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
