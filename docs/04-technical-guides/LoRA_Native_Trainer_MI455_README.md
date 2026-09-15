# Native SFT LoRA on MI455X (gfx1250)

1-GPU recipes for B0 / MI455X. Configs live in
`examples/megatron/configs/MI455X/`. All of them go through the **native
Megatron SFT trainer** (`train posttrain` + `sft_trainer.yaml`), not
Megatron-Bridge. See also [Native SFT and LoRA](native-sft-lora.md).

`primus-cli` loads `runner/helpers/envs/primus-env.sh`, which sources
`runner/helpers/envs/MI455X.sh` when the GPU is gfx1250 or when
`PRIMUS_GPU_MODEL=MI455X`. Do **not** `source runner/helpers/envs/MI455X.sh`
by itself — `LOG_INFO_RANK0` is defined in `base_env.sh` and you will get
`command not found`.

Image: `YOUR_DOCKER_IMAGE`.

| config (`examples/megatron/configs/MI455X/`) | What it is | Weights |
|---|---|---|
| `llama3.2_1B-BF16-lora-sft.yaml` | Full Llama-3.2-1B LoRA SFT | Converts HF weights (`meta-llama/Llama-3.2-1B`, gated) |
| `llama2_70B-BF16-lora-sft-1gpu-proxy.yaml` | Llama-2-70B shape, **4 of 80 layers** | Random init, no 70B download |
| `qwen2.5_72B-BF16-lora-sft-1gpu-proxy.yaml` | Qwen2.5-72B shape, **4 of 80 layers** | Random init, no 72B download |
| `qwen3_235B_A22B-BF16-lora-sft-1gpu-proxy.yaml` | Qwen3-235B-A22B MoE shape, **4 of 94 layers** | Random init, no 235B download |

Primus has **Qwen3-235B-A22B**, not a 253B variant. The MoE smoke uses that
in-tree model. LoRA on that recipe is attention-only (`linear_qkv` /
`linear_proj`): grouped-GEMM expert linears are
`PrimusTurboColumnParallelGroupedLinear`, which the LoRA adapter cannot wrap.

Proxy numbers are enablement signals, not model TFLOP/s. Loss staying near
`ln(vocab)` on the random-init recipes is expected.

---

## 0. Host: pull your image

```bash
export DOCKER_IMAGE=YOUR_DOCKER_IMAGE
docker pull "$DOCKER_IMAGE"
```

Confirm the checkout:

```bash
cd /path/to/Primus          # this repo
git submodule update --init --recursive
ls third_party/Megatron-LM
```

---

## 1. Device verification (host)

```bash
rocm-smi
rocminfo | grep -E 'gfx[0-9]+|Marketing Name' | head
ls -l /dev/kfd /dev/dri
```

Expect `gfx1250`. The marketing name may be generic (`AMD Radeon Graphics`);
that is why `primus-env.sh` falls back from ISA to `MI455X`.

Optional host GPU smoke (needs a ROCm Python on the host; skip if you only
have the container):

```bash
timeout 60 python -c 'import torch; x=torch.ones(1,device="cuda"); torch.cuda.synchronize(); print("OK", torch.cuda.get_device_name(0))'
```

---

## 2. Two ways to run

| | When | Entry |
|---|---|---|
| **Outside** (host) | You stay on the host; `primus-cli` starts the container | `./primus-cli container ...` |
| **Inside** | You already have a shell in the image | `./primus-cli direct ...` |

`PRIMUS_TURBO_ATTN_BACKEND=triton` is required: this image has no aiter.

Common env for every 1-GPU recipe:

```bash
export DOCKER_IMAGE=YOUR_DOCKER_IMAGE
export PRIMUS_GPU_MODEL=MI455X
export HIP_VISIBLE_DEVICES=0
export NNODES=1
export GPUS_PER_NODE=1
export PRIMUS_TURBO_ATTN_BACKEND=triton
```

---

## 3. Outside the container (`primus-cli container`)

From the Primus repo root on the host. The CLI mounts this tree and runs
`primus-cli-direct.sh` inside the image, so `MI455X.sh` is applied.

**Llama-3.2-1B (real HF weights, needs `HF_TOKEN`):**

```bash
cd /path/to/Primus
export HF_TOKEN=...          # gated meta-llama/Llama-3.2-1B

./primus-cli container --image YOUR_DOCKER_IMAGE \
  --env HIP_VISIBLE_DEVICES=0 --env NNODES=1 --env GPUS_PER_NODE=1 \
  --env PRIMUS_GPU_MODEL=MI455X \
  --env PRIMUS_TURBO_ATTN_BACKEND=triton \
  --env HF_TOKEN="${HF_TOKEN}" \
  -- train posttrain \
  --config examples/megatron/configs/MI455X/llama3.2_1B-BF16-lora-sft.yaml
```

**Random-init proxies (no 70B / 72B / 235B download).** Tokenizer files still
come from a small ungated Hub repo. Create the dummy load dir on the host if
the container bind-mounts `/tmp`; otherwise it is created inside.

```bash
cd /path/to/Primus
mkdir -p /tmp/primus-proxy-no-ckpt

# Llama-2-70B, 4-layer proxy
./primus-cli container --image YOUR_DOCKER_IMAGE \
  --env HIP_VISIBLE_DEVICES=0 --env NNODES=1 --env GPUS_PER_NODE=1 \
  --env PRIMUS_GPU_MODEL=MI455X \
  --env PRIMUS_TURBO_ATTN_BACKEND=triton \
  -- train posttrain \
  --config examples/megatron/configs/MI455X/llama2_70B-BF16-lora-sft-1gpu-proxy.yaml

# Qwen2.5-72B, 4-layer proxy
./primus-cli container --image YOUR_DOCKER_IMAGE \
  --env HIP_VISIBLE_DEVICES=0 --env NNODES=1 --env GPUS_PER_NODE=1 \
  --env PRIMUS_GPU_MODEL=MI455X \
  --env PRIMUS_TURBO_ATTN_BACKEND=triton \
  -- train posttrain \
  --config examples/megatron/configs/MI455X/qwen2.5_72B-BF16-lora-sft-1gpu-proxy.yaml

# Qwen3-235B-A22B MoE, 4-layer proxy
./primus-cli container --image YOUR_DOCKER_IMAGE \
  --env HIP_VISIBLE_DEVICES=0 --env NNODES=1 --env GPUS_PER_NODE=1 \
  --env PRIMUS_GPU_MODEL=MI455X \
  --env PRIMUS_TURBO_ATTN_BACKEND=triton \
  -- train posttrain \
  --config examples/megatron/configs/MI455X/qwen3_235B_A22B-BF16-lora-sft-1gpu-proxy.yaml
```

---

## 4. Inside the container (`docker run` + `primus-cli direct`)

### 4.1 Start an interactive shell

From the Primus repo root on the host (`$PWD` should be the repo, or mount
it explicitly):

```bash
cd /path/to/Primus

docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host --network=host --privileged \
  -v "$PWD":"$PWD" -w "$PWD" \
  -e HIP_VISIBLE_DEVICES=0 \
  -e NNODES=1 \
  -e GPUS_PER_NODE=1 \
  -e PRIMUS_GPU_MODEL=MI455X \
  -e PRIMUS_TURBO_ATTN_BACKEND=triton \
  -e HSA_ENABLE_SDMA=1 \
  YOUR_DOCKER_IMAGE \
  bash
```

Turbo in this image lives at `/workspace/Primus-Turbo`. Your Primus tree is
the bind-mount (`$PWD`).

### 4.2 Packages (inside the container)

Install these once per container. `tensorboard` is imported by the Megatron
logging path even when `disable_tensorboard: true`. `psutil` is used by
resource / memory logging.

```bash
pip install tensorboard psutil
```

Optional check:

```bash
python -c "import tensorboard, psutil; print('tensorboard', tensorboard.__version__); print('psutil', psutil.__version__)"
```

If you also hit missing hook deps on a stripped image:

```bash
pip install nltk datasets transformers tqdm sentencepiece loguru plotext
```

### 4.3 Device verification (inside the container)

```bash
rocm-smi
rocminfo | grep -m1 -oE 'gfx[0-9]+'

# Must return. If it hangs in InterruptSignal::WaitRelaxed, the GPU is wedged
# (reboot / GPU reset) — do not trust a training failure after that.
timeout 60 python -c 'import torch; x=torch.ones(1,device="cuda"); torch.cuda.synchronize(); print("OK", torch.cuda.get_device_name(0))'
```

Confirm the env file will load. Use `primus-env.sh`, not `MI455X.sh` alone:

```bash
export PRIMUS_GPU_MODEL=MI455X NNODES=1 GPUS_PER_NODE=1
source runner/helpers/envs/primus-env.sh
```

You should see:

```
Detected GPU model: MI455X
Loading MI455X (gfx1250) specific settings...
```

### 4.4 Run a config (inside the container)

```bash
export PRIMUS_GPU_MODEL=MI455X
export HIP_VISIBLE_DEVICES=0
export NNODES=1
export GPUS_PER_NODE=1
export PRIMUS_TURBO_ATTN_BACKEND=triton
mkdir -p /tmp/primus-proxy-no-ckpt

# Llama-3.2-1B (needs HF_TOKEN in the environment)
./primus-cli direct -- train posttrain \
  --config examples/megatron/configs/MI455X/llama3.2_1B-BF16-lora-sft.yaml

# Llama-2-70B 4-layer proxy (random init)
./primus-cli direct -- train posttrain \
  --config examples/megatron/configs/MI455X/llama2_70B-BF16-lora-sft-1gpu-proxy.yaml

# Qwen2.5-72B 4-layer proxy (random init)
./primus-cli direct -- train posttrain \
  --config examples/megatron/configs/MI455X/qwen2.5_72B-BF16-lora-sft-1gpu-proxy.yaml

# Qwen3-235B-A22B MoE 4-layer proxy (random init)
./primus-cli direct -- train posttrain \
  --config examples/megatron/configs/MI455X/qwen3_235B_A22B-BF16-lora-sft-1gpu-proxy.yaml
```

---

## 5. What a pass looks like

Every run:

```
Detected GPU model: MI455X
Loading MI455X (gfx1250) specific settings...
iteration        N/       N | ...
  number of skipped iterations:   0 | number of nan iterations:   0
[after training is done]
```

Proxy (random init) extras:

```
[INFO] Pretrained checkpoint already configured: /tmp/primus-proxy-no-ckpt, skipping conversion
WARNING: could not find the metadata file ... will not load any checkpoints and will start from random
```

Llama-3.2-1B extras (real weights):

```
Native (llama) conversion: dtype=bf16 TP=1 PP=1 EP=1
COMPLETE and EXACT (0 ...)
[PEFT pre-wrap] ... missing=0, unexpected=0
```

`ms/iter` on the first two steps includes compile; quote the later steady value.

---

## 6. Notes

- **1 GPU:** `HIP_VISIBLE_DEVICES=0`, `GPUS_PER_NODE=1`. `MI455X.sh` then sets
  `NCCL_P2P_DISABLE=1` and `AMD_SERIALIZE_COPY=0`.
- **Do not** set `AMD_SERIALIZE_COPY=3`.
- Proxies still download a **tokenizer** (Llama-2-7B tokenizer, Qwen2.5-0.5B,
  or Qwen3-0.6B) and **Alpaca**. They do not download the 70B/72B/235B weights.
- Qwen3 MoE: `use_turbo_deepep: false` (DeepEP CPU recv timeout on this box).
  LoRA targets are attention only; do not add `linear_fc1` / `linear_fc2` or
  LoRA hits `PrimusTurboColumnParallelGroupedLinear` and raises
  `NotImplementedError`.
- If Gloo dies with `Unable to find address for: <ip>`, the image is missing
  `iproute2` or you sourced `MI455X.sh` without `primus-env.sh`.
