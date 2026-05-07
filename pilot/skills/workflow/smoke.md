# SMOKE — tiny-scale liveness gate

**Status**: v1 — single-node implemented; multi-node deferred until
`preflight._dispatch_run_slurm` lands.

SMOKE is the cheapest possible verification that a Primus configuration can
**start, advance, and exit cleanly** at tiny scale before any real BASELINE
or PROJECTION work is committed. It runs ~10–50 iterations on the same
hardware that will eventually run the full job, with `train_iters` shrunk
and (where applicable) `global_batch_size` / `seq_length` reduced.

A failed SMOKE **must** block all downstream stages. Re-entry is always
PROJECTION (re-decide the parallelism / runtime knobs), never a silent retry.

## Inputs

| Input | Source | Why |
|---|---|---|
| `cluster.yaml`         | universal Pilot contract (see SETUP.md)   | declares mode (single/slurm) + slurm job id when applicable |
| Primus exp.yaml `--plan`     | the candidate plan from PROJECTION  | shape of the run (model, parallelism, dtype) |
| `train_iters` override | tool-side, default 20–50            | bound the smoke wallclock |

## Tools used

| Stage step | Tool                                     | Purpose |
|------------|------------------------------------------|---------|
| 1. launch  | `pilot.tools.submit run`                 | spawn the training subprocess; produces `state/runs/<run_id>/handle.yaml` |
| 2. observe | `pilot.tools.observe watch`              | poll snapshots until terminal (or hung) |
| 3. cancel  | `pilot.tools.submit cancel` (on hang/OOM) | stop a misbehaving run; only if observe recommends it |

Both tools consume the same `cluster.yaml`; mode handling and srun fan-out
is the responsibility of `submit`, not SMOKE.

## Tiny-scale knobs

For a typical Megatron pretrain (e.g. `deepseek_v2_lite-FP8-pretrain.yaml`),
the SMOKE override set is:

```bash
--override train_iters=20            # ~1–2 minutes of real training time
--override log_avg_skip_iterations=2  # let the first 2 iters be JIT-compile noise
# leave global_batch_size / micro_batch_size / seq_length AS-IS so SMOKE
# exercises the same memory profile as BASELINE.
```

Larger models or aggressive parallelism choices may also need:

```bash
--override save_interval=999999       # disable checkpoint I/O during SMOKE
--override eval_interval=999999       # skip eval
```

These are **only** to make SMOKE cheap; never to dodge an honest failure.

## End-to-end recipe

```bash
# 1) Submit detached. Returns a SubagentResult with run_id; subprocess survives liveness.
pilot.tools.submit run \
  --cluster-config cluster.yaml \
  --plan examples/megatron/configs/MI355X/deepseek_v2_lite-FP8-pretrain.yaml \
  --override train_iters=20 \
  --run-id smoke_$(date -u +%Y%m%dT%H%M%S)

# 2) Watch until terminal. Default hang threshold = 120s of iter silence.
pilot.tools.observe watch \
  --run-id <run_id> \
  --interval-s 5 \
  --hang-threshold-s 180   # bump if first iter is slow due to JIT compile

# 3) Final snapshot is the JSON emitted at end of `watch`. Read its
#    `recommendation.next` and `symptoms.*` to decide.
```

## Pass / fail criteria

The final RunSnapshot (schema: `pilot/schemas/run_snapshot.schema.json`) is
the single source of truth.

| Outcome | Required snapshot fields | Next stage |
|---|---|---|
| **PASS**          | `status=completed`, `metrics.loss_finite=true`, all `symptoms.*=false`, `progress.pct≈100` | proceed to BASELINE |
| **PASS-WITH-WARN**| `status=completed`, but `progress.iters_per_min` < 50% of analytical projection | continue to BASELINE; flag in REPORT |
| **FAIL — OOM**    | `symptoms.oom_detected=true`                | re-PROJECTION with smaller per-GPU memory plan (TP↑ / micro_batch↓ / activation recomp) |
| **FAIL — hang**   | `status=hung` OR `progress.silent_for_s` > threshold while `process.alive=true` | cancel via `submit cancel`; DIAGNOSE NCCL/RCCL fabric (preflight inter_node baseline + log evidence) |
| **FAIL — collective** | `symptoms.nccl_error=true`              | DIAGNOSE: re-run preflight inter_node scope; check NIC/QoS, then re-PROJECTION if topology bound |
| **FAIL — CUDA**   | `symptoms.cuda_error=true`                  | DIAGNOSE: device-level error (illegal access, ECC); often points at a single bad node — re-run preflight per-host |
| **FAIL — NaN**    | `metrics.loss_finite=false`                 | re-PROJECTION with stricter dtype / lower LR / larger eps |
| **FAIL — exit**   | `status=failed`, `process.exit_code != 0`   | inspect `train.log` referenced in snapshot; usually a Python `Traceback` (see `symptoms.evidence`) |

The snapshot's `recommendation.next` field encodes most of these mappings as
a hint for the orchestrator — but the final call always belongs to the agent.

## Hang handling

`observe` flags `hang_suspected=true` when the process is alive AND no
new Megatron `iteration N/M` line has been emitted for `hang_threshold_s`
seconds. Defaults are:

* **120 s** for steady-state SMOKE (after JIT compile).
* Use `--hang-threshold-s 300` (or higher) when the first iter is expected
  to be slow on cold caches; otherwise the first JIT-compile pass will be
  flagged as a false hang.

When SMOKE detects a hang, the recipe is mandatory:

1. Take one final `observe snapshot --save` to capture log lines + symptom
   evidence.
2. `pilot.tools.submit cancel --run-id <id>` (sends SIGTERM to the process
   group; SIGKILL after 10 s).
3. Open DIAGNOSE; **never** retry the same plan blind.

## Anti-patterns (do not do)

* Lowering `train_iters` below 5 — JIT-compile / dataset prep dominates and
  no real signal remains.
* Disabling FP8 / lowering precision *just* to make SMOKE pass — that
  invalidates any subsequent BASELINE.
* Ignoring `symptoms.evidence` because `recommendation.next == continue` —
  evidence can be a leading indicator before the wall-time hang trips.
* Re-running SMOKE without re-doing PROJECTION after an OOM — the same plan
  will OOM again.

## Multi-node (deferred)

Multi-node SMOKE follows the same recipe but needs:

* `cluster.yaml` mode: `slurm` with a valid `slurm.job_id`.
* `submit run` will internally `srun --nodes=N --ntasks-per-node=1 ...
  bash -c 'export NODE_RANK=$SLURM_NODEID; exec bash run_pretrain.sh'`.
* Hang threshold should be raised (NCCL fabric bring-up adds ~30 s on
  some clusters).

The SLURM dispatch path is implemented in `submit.py` (verified at the
argv-construction level) but has not been live-tested end-to-end. SMOKE
multi-node is gated on `preflight._dispatch_run_slurm` shipping first
(it produces the inter-node RCCL baseline that SMOKE relies on for sane
hang thresholds).
