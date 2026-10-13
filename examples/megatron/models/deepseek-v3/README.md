# DeepSeek-V3 proxy on MI455X (gfx1250)

The 4-GPU DeepSeek-V3 proxy of the internal GFX1250 runbook v3 (§3), plus a 1-GPU variant, run with
the Primus `direct` launcher. Runs happen in a container that maps only the selected GPUs, and
Primus-Turbo can come from a local checkout.

| File | Runs on | Purpose |
| --- | --- | --- |
| `start_container_mi455x.sh` | host | Starts a container that sees only the selected GPUs (not privileged). |
| `prepare_container_mi455x.sh` | container, as root, once | Patches Triton's gfx1250 expert scheduling off. With `TURBO_DIR`, also builds that Primus-Turbo checkout and pip-installs the flydsl it pins (network, or a wheel via `FLYDSL_SPEC`). |
| `run_deepseek_v3_proxy_mi455x.sh` | container | The launcher. Preset `ep4_4gpu` (default) or `ep1_1gpu`. |

## Quick start

From the Primus checkout on the host:

```bash
GPUS=0,1,2,3 NAME=primus-dsv3-4gpu MOUNTS=/path/to/Primus-Turbo \
    bash examples/megatron/models/deepseek-v3/start_container_mi455x.sh
docker exec -e TURBO_DIR=/path/to/Primus-Turbo primus-dsv3-4gpu \
    bash examples/megatron/models/deepseek-v3/prepare_container_mi455x.sh
docker exec -u "$(id -u):$(id -g)" -e TURBO_DIR=/path/to/Primus-Turbo primus-dsv3-4gpu \
    bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh
```

- **One GPU.** Use, for example, `GPUS=3`, and run the launcher with `-e PRESET=ep1_1gpu`.
- **Options.**
  - Arguments to the launcher are appended to the Primus command line and override the preset, e.g.
    `--train_iters 20` or `--use_turbo_attention True`.
  - `-e PROFILE=1` profiles iterations 6–7 on rank 0.
  - `-e DRY_RUN=1` prints the command without running it.
  - Without `TURBO_DIR`, the image's `primus_turbo` is used.
- **Output.** The launcher prints the path of `output/amd/<user>/<exp>/train.log`, the console log (rank
  0, plus the per-iteration lines, which the last rank prints). Every rank also logs to
  `logs/pre_trainer/rank-<N>/` in the same directory. Under `docker exec -u <uid>`, `<user>` is the
  image's name for that uid or, if it has none, `uid<uid>` (Primus reads a numeric user name as an int).

The launcher refuses to start in these cases:

- Triton still enables expert scheduling.
- `primus_turbo` does not load from `TURBO_DIR`.
- flydsl is not the version that checkout pins.
- `HIP_VISIBLE_DEVICES` does not list exactly the preset's number of GPUs or, when it is unset, the
  container cannot open exactly that many.

If the run succeeds but a kernel in `TRITON_CACHE_DIR` was compiled with expert scheduling, the launcher
exits with code 3.

`TURBO_DIR` reaches the training processes through `primus-cli direct --env PYTHONPATH=...`.
`runner/helpers/envs/base_env.sh` puts site-packages in front of an exported `PYTHONPATH`, so an
exported path would silently lose to the image's `primus_turbo`.

## Reference (2026-10-13)

Image `amdprimus/amdprimus:tas-20261011`, Primus `dev/mi455x_v1` (ab873556), Primus-Turbo
`dev/mi455x_v1` (a1800137), 4 × MI455X:

| Run | ms/iter | TFLOP/s/GPU | lm loss at iteration 10 |
| --- | --- | --- | --- |
| Runbook v3 §3: image Turbo 1d47bb72, privileged `primus-cli container`, Primus commit not recorded | 2500 | 567.8 | 5.580 |
| `ep4_4gpu`, Turbo a1800137 or the image's 1d47bb72 (same numbers) | 2543 | 558 | 5.57984 |

The 1.7% gap is not from Turbo; the host, the Primus commit and the container mode were not compared.

- **Correctness.** 10/10 iterations without a NaN/Inf abort. All six `ep4_4gpu` runs, with either Turbo, logged the same
  grad norms: 3.790, 3.758, 3.795, 4.995, 9.034, 10.183, 7.921, 7.203, 6.827, 6.680.
- **`ep1_1gpu`.** 671 ms/iter on one MI455X of another node. Clean runs logged the grad norms 3.193,
  3.232, 3.314, 3.263, 3.210, 3.713, 3.362, 3.779, 3.329, 3.289 and lm loss 9.4595 at iteration 10.
- **Profile of `ep4_4gpu`, rank 0.**
  - TE's unfused attention takes 47% of the GPU compute time.
  - The MoE all-to-all, about 180 ms per iteration, is not overlapped with compute.

## Known issues

- **Attention backends.** On this image, only two backends run MLA (q/k 192, v 128):
  - TE `UnfusedDotProductAttention`, the default;
  - Primus-Turbo's Triton backend (`--use_turbo_attention True`).

  aiter/CK, FlyDSL, and TE fused/flash attention do not.
- **Silent gradient corruption.** 9 of 29 ten-iteration `ep1_1gpu` runs, all on the same MI455X, left
  the reference grad norms at a random iteration (a wrong value, NaN or Inf). This happened with both
  attention backends and both Turbo builds. Megatron aborts on NaN/Inf gradients but not on a wrong
  finite value, and with bf16 its `nan iterations` counter stays 0, so compare grad norms with a
  reference run. The cause is not known. None of the six `ep4_4gpu` runs (another node) deviated, too
  few to rule it out there.
