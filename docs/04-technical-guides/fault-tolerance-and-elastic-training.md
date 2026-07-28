# Fault tolerance and elastic training

Large-scale jobs run for days across thousands of GPUs, where hardware faults, NIC flaps, and node loss are routine. This guide covers the mechanisms Primus exposes to survive and recover from failures: graceful exit + checkpoint-based resume, Megatron's fault-tolerance package and in-process restart, and TorchTitan's [torchft](https://github.com/pytorch/torchft)-based elastic training. Parameters are grounded in `primus/configs/modules/megatron/trainer_base.yaml`, `primus_megatron_module.yaml`, and `primus/configs/modules/torchtitan/pre_trainer.yaml`.

The foundation of all recovery is checkpointing—read [Checkpoint management](./checkpoint-management.md) first.

---

## 1. The recovery model

There are three layers, from simplest to most advanced:

1. **Checkpoint + restart**—periodically save state; on failure, relaunch the job and resume from the last checkpoint. Works on every backend; relies on the scheduler (Slurm `--requeue`, Kubernetes restart policy) to relaunch.
2. **Graceful exit**—detect a signal or time/iteration budget, save a final checkpoint, and exit cleanly so the restart resumes with no lost work.
3. **In-job fault tolerance / elastic**—detect a failed rank and restart in-process (Megatron) or continue with a reduced/replaced replica group (TorchTitan + torchft) without tearing down the whole job.

---

## 2. Graceful exit and auto-resume (Megatron)

Controls in `trainer_base.yaml` let a run stop cleanly at a boundary so the next launch resumes seamlessly:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `exit_signal_handler` | `false` | Install a signal handler that saves a checkpoint and exits gracefully on SIGTERM (e.g. Slurm preemption). |
| `exit_duration_in_mins` | `null` | Exit (after saving) once the job has run this many minutes—useful to fit scheduler time limits. |
| `exit_interval` | `null` | Exit after this many iterations. |
| `adlr_autoresume` | `false` | Enable ADLR auto-resume integration. |
| `adlr_autoresume_interval` | `1000` | Iterations between auto-resume checks. |

Primus-level continuation (`primus_megatron_module.yaml`):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `auto_continue_train` | `false` | Automatically continue from the latest checkpoint in the experiment's save directory on relaunch. |
| `disable_last_saving` | `false` | Disable the final end-of-run checkpoint save (leave `false` so resume points exist). |

**Pattern:** enable `exit_signal_handler` + `exit_duration_in_mins` (or rely on preemption signals), set a reasonable checkpoint `save_interval`, and turn on `auto_continue_train` so requeued jobs pick up where they left off.

---

## 3. Megatron fault-tolerance package and in-process restart

Megatron integrates an optional fault-tolerance package and in-process restart (`trainer_base.yaml`):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `enable_ft_package` | `false` | Enable the Megatron fault-tolerance package (rank monitoring / heartbeat). |
| `calc_ft_timeouts` | `false` | Auto-calculate fault-tolerance timeouts from observed step times. |
| `run_workload_inspector_server` | `false` | Run the workload inspector server for health/diagnostics. |
| `inprocess_restart` | `false` | Restart failed ranks **in process** to avoid a full job teardown. |

In-process restart reduces recovery time by re-initializing the process group and reloading state without re-scheduling the whole allocation. Combine with frequent checkpoints so the restarted ranks have a recent resume point.

---

## 4. Numerical safety nets (Megatron)

Detecting corruption early prevents wasted compute and divergence (`trainer_base.yaml`):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `check_for_nan_in_loss_and_grad` | `true` | Abort/handle on NaN/Inf in loss or gradients. |
| `check_for_spiky_loss` | `false` | Detect anomalous loss spikes. |
| `check_for_large_grads` | `false` | Detect abnormally large gradients. |
| `decrease_batch_size_if_needed` | `false` | Reduce batch size when needed instead of failing. |

These don't recover from hardware faults but stop a corrupted run before it pollutes downstream checkpoints.

---

## 5. Elastic training with torchft (TorchTitan)

TorchTitan supports semi-synchronous, replica-based fault tolerance via [torchft](https://github.com/pytorch/torchft). Configured under `fault_tolerance:` in `primus/configs/modules/torchtitan/pre_trainer.yaml`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `enable` | `false` | Enable torchft fault tolerance. |
| `process_group` | `gloo` | Process group backend for fault-tolerance coordination. |
| `process_group_timeout_ms` | `10000` | Coordination timeout (ms). |
| `replica_id` | `0` | This replica's ID. |
| `group_size` | `0` | Number of replica groups (`0` = auto). |
| `min_replica_size` | `1` | Minimum replicas required to keep training. |
| `semi_sync_method` | `null` | Semi-synchronous algorithm (e.g. DiLoCo-style), `null` = standard. |

With replica groups, the loss of one replica can be tolerated as long as `min_replica_size` is still satisfied—training continues while the failed replica recovers/rejoins, rather than crashing the whole job.

**Install** the optional dependencies before enabling (`requirements-torchft.txt`):

```bash
pip install -r requirements-torchft.txt   # torchft-nightly + OpenTelemetry exporters
```

TorchTitan also exposes communication timeouts under `comm:` (`init_timeout_seconds: 300`, `train_timeout_seconds: 100`) that govern how long collectives wait before declaring a fault.

---

## 6. Scheduler integration

In-job mechanisms still need the scheduler to relaunch on full-job failure:

- **Slurm**—submit with `--requeue` so preempted/failed jobs are re-queued; pair with `exit_signal_handler` to checkpoint on SIGTERM. See [Deployment](../05-operations/deployment.md).
- **Kubernetes**—use a restart policy / operator that recreates pods; mount checkpoint storage on a shared/persistent volume.
- **Shared checkpoint storage**—all ranks must read the same checkpoint directory after relaunch (NFS, Lustre, or object storage). See [Checkpoint management](./checkpoint-management.md).

---

## 7. Recommended setup

1. **Always checkpoint**—set a `save_interval` matched to your mean-time-between-failures; use async/distributed checkpointing to keep overhead low.
2. **Exit cleanly**—`exit_signal_handler: true` (+ `exit_duration_in_mins` for time-boxed allocations).
3. **Resume automatically**—`auto_continue_train: true` (Megatron) and `--requeue` (Slurm).
4. **Reduce recovery time at scale**—`enable_ft_package` + `inprocess_restart` (Megatron) or torchft replica groups (TorchTitan).
5. **Guard numerics**—keep `check_for_nan_in_loss_and_grad` on; consider spiky/large-grad checks for unstable configs.

---

## Related documentation

- [Checkpoint management](./checkpoint-management.md)—save/load formats, async and distributed checkpointing.
- [Deployment](../05-operations/deployment.md)—Slurm/Kubernetes restart and requeue.
- [Multi-node networking](./multi-node-networking.md)—NIC faults and collective timeouts.
- [Megatron parameters](../03-configuration-reference/megatron-parameters.md) and [TorchTitan parameters](../03-configuration-reference/torchtitan-parameters.md).
