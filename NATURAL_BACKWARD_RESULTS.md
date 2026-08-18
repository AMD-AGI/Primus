# FLUX MBS32 natural backward optimization

## Scope

This worktree starts from `dev/zirui/flux-compile` at `f472b21f` and the unified v26.3 image. The separate contraction-kernel campaign is not modified here.

Proxy configuration: one MI355X node, eight GPUs, MBS32, GA4, effective GBS1024, checkpoint ratio 0, per-block `max-autotune-no-cudagraphs`, FSDP2 BF16 reduction and no reshard.

## Results

Matched baseline on nodes 037/251 was `2.178–2.180 s/optimizer-step`.

### Natural-layout wgrad screen

The public FlyDSL TN kernel was screened against `_scaled_mm` for all repeated-block wgrad shapes. Kernel-only FlyDSL was competitive or faster, but full training promotion was shape-selective:

- MBS32 single-up alone was neutral/noisy.
- Double-block square projection alone was neutral.
- Double MLP-up/down plus existing single MLP wgrad paths reduced E2E to about `2.150 s`.
- Expanding all repeated-block wgrads was not better than the MLP-only set.

### Forward-input FP8 reuse

TorchAO normally quantizes the same activation once for forward and again for wgrad. For tensorwise configs with matching forward/wgrad cast policy, saving the forward FP8 input and reusing it in wgrad reduced FP8 cast/scale exposed time from about `97.5 ms` to `53.9 ms` per optimizer step.

Input reuse alone measured `2.160 s`; natural MLP wgrad alone measured about `2.151 s`. Combined, the changes interact favorably:

| Node/run | Step time |
|---|---:|
| 037 mounted candidate | `2.119 s` |
| 236 mounted candidate | `2.111 s` |
| 236 built image `primus-flux-natural-bwd:v0.1` | `2.115 s` |

The built candidate remained finite through step 520 (`loss=0.6609`, `gnorm=0.4762`); step 500 was also finite (`loss=0.6206`, `gnorm=0.3724`). Steady time was `2.1132 s`. A stateful run produced validation loss `0.685480` at step 512 and saved DTCP checkpoint 500/final. Resuming checkpoint 500 completed finite step 530 (`loss=0.6481`, `gnorm=0.3546`) and validation `0.681209`. Peak memory increased from `151.81 GB` to `159.57 GB`. The reuse is runtime-gated and enabled only by the MBS32 selective-FlyDSL launch policy, leaving MBS64 defaults unchanged. An MBS64 smoke run with the gate off completed 30 finite steps.

Relative to the matched unified baseline, this is about `1.03x`; relative to the original ratio-0.25 baseline near `2.42 s`, total speedup is about `1.144x`.

### Rejected screens

- Natural NN dgrad was 20–40% slower than `_scaled_mm` on every tested exact shape and was not integrated.
- Reusing the dgrad grad-output cast for wgrad produced no measurable E2E gain.
- Broad natural wgrad dispatch increased memory and was no faster than the MLP-only exact set.

## Current candidate

Files changed:

- `docker/flux-fp8/Dockerfile.v26.3-combined`
- `local_runs/run_flux_mlperf.sh`

The remaining path to 1.20x depends on the separate selective-contraction kernel campaign and/or a larger complete backward-region fusion. This candidate still requires 500-step validation, checkpoint/resume, MBS64 regression, and final multi-node qualification before promotion.
