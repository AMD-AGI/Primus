---
name: slurm-xiaoming-dev-container
description: Work with xiaoming's SLURM jobs through the xiaoming-dev Podman container. Use when the user wants to inspect xiaoming's squeue jobs, attach to a running SLURM allocation or node, enter the xiaoming-dev container, or wait for existing container processes before running new commands.
---

# SLURM xiaoming-dev Container Workflow

Use this workflow when operating on SLURM jobs owned by user `xiaoming` and running commands inside the `xiaoming-dev` Podman container.

## Workflow

### Step 1: Inspect xiaoming's SLURM Jobs

List current jobs for `xiaoming`:

```bash
squeue -u xiaoming -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"
```

If the user did not specify a job, pick the relevant running job from the output. Prefer jobs in state `R` for attaching. If multiple jobs are plausible, ask the user which job to use.

To inspect one job in detail:

```bash
scontrol show job <job_id>
```

Extract the allocated node list from `NodeList=` or from the `squeue` reason/node column, then expand it if needed:

```bash
scontrol show hostnames <nodelist>
```

### Step 2: Attach to the Job or Node

If a running job ID is available, prefer attaching through SLURM:

```bash
srun --jobid=<job_id> --pty bash
```

If direct `srun --jobid` attach is unavailable, SSH to one allocated node:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no <node>
```

Do not cancel, requeue, or modify existing SLURM jobs unless the user explicitly asks.

### Step 3: Ensure the xiaoming-dev Container Exists

Check whether the container is running:

```bash
podman ps --filter name=^xiaoming-dev$ --format "{{.Names}} {{.Status}}"
```

If it is not running, check whether it exists but is stopped:

```bash
podman ps -a --filter name=^xiaoming-dev$ --format "{{.Names}} {{.Status}}"
```

If the container exists but is stopped, start it:

```bash
podman start xiaoming-dev
```

If it does not exist, create it using the repository's container script if present:

```bash
bash start_container.sh
```

Only fall back to an inline `podman run` command if the script is unavailable. Preserve the container name `xiaoming-dev`.

### Step 4: Check for Existing Work Inside the Container

Before running a new workload, inspect active processes inside `xiaoming-dev`:

```bash
podman exec xiaoming-dev bash -lc 'ps -eo pid,ppid,stat,etime,cmd --sort=start_time'
```

Look for user workloads such as Python, training, build, test, `torchrun`, `primus`, `megatron`, or long-running shell commands. Ignore normal container keepalive commands such as `sleep infinity` and the `ps` command itself.

If another user workload is running, do not start a new workload immediately. Tell the user what is running, then wait and re-check periodically:

```bash
while podman exec xiaoming-dev bash -lc "pgrep -af 'python|torchrun|primus|megatron|pytest|cmake|ninja|make' >/dev/null"; do
  podman exec xiaoming-dev bash -lc "pgrep -af 'python|torchrun|primus|megatron|pytest|cmake|ninja|make'"
  sleep 60
done
```

If the user explicitly confirms the existing process is safe to interrupt, then follow their instruction. Otherwise, wait for it to finish.

### Step 5: Run Commands Inside xiaoming-dev

Run interactive shells with:

```bash
podman exec -it xiaoming-dev bash
```

Run non-interactive commands with:

```bash
podman exec xiaoming-dev bash -lc '<command>'
```

When running commands from the Primus repository, first move to the workspace path mounted inside the container. Common paths are:

```bash
cd "$HOME/workspace/Primus"
```

or:

```bash
cd "$HOME/Primus"
```

Verify the path exists inside the container before assuming it:

```bash
podman exec xiaoming-dev bash -lc 'pwd; ls "$HOME/workspace/Primus" "$HOME/Primus" 2>/dev/null'
```

## Important Notes

- Always inspect `squeue -u xiaoming` before attaching or choosing a node.
- Prefer attaching to an existing running allocation over starting a new SLURM job.
- Use the `xiaoming-dev` Podman container for work on the allocated node.
- If other meaningful processes are running inside the container, wait for them to finish before starting new work.
- Do not stop, remove, or recreate `xiaoming-dev` if it may contain active work, unless the user explicitly approves.
