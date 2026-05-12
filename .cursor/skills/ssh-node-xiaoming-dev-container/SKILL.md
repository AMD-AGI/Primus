---
name: ssh-node-xiaoming-dev-container
description: SSH to a specified cluster node and run work through the xiaoming-dev Podman container. Use when the user provides a node hostname, wants to connect to that node over SSH, enter or use the xiaoming-dev container, and wait for existing container processes before starting new commands.
---

# SSH Node xiaoming-dev Container Workflow

Use this workflow when the user gives a specific node and wants to run commands there through the `xiaoming-dev` Podman container.

## Workflow

### Step 1: Resolve the Target Node

Use the node name provided by the user. If the user has not provided a node, ask for the exact hostname before proceeding.

Optionally verify the node is reachable:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no <node> 'hostname'
```

If SSH fails because authentication is unavailable, tell the user and ask how they want to proceed. Do not modify SSH keys without explicit confirmation.

### Step 2: SSH to the Node

For an interactive session:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no <node>
```

For non-interactive commands, run them through SSH:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no <node> '<command>'
```

### Step 3: Check xiaoming-dev on the Node

Check whether the `xiaoming-dev` Podman container is running on the target node:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no <node> \
  'podman ps --filter name=^xiaoming-dev$ --format "{{.Names}} {{.Status}}"'
```

If it is not running, check whether it exists but is stopped:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no <node> \
  'podman ps -a --filter name=^xiaoming-dev$ --format "{{.Names}} {{.Status}}"'
```

If the container exists but is stopped, start it:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no <node> \
  'podman start xiaoming-dev'
```

If it does not exist, create it from the Primus workspace on the node if available:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no <node> \
  'cd "$HOME/workspace/Primus" && bash start_container.sh'
```

If `$HOME/workspace/Primus` is not present on that node, check `$HOME/Primus`:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no <node> \
  'cd "$HOME/Primus" && bash start_container.sh'
```

Do not stop, remove, or recreate `xiaoming-dev` if it may contain active work unless the user explicitly approves.

### Step 4: Wait for Existing Container Work

Before running a new workload, inspect meaningful processes inside the container:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no <node> \
  "podman exec xiaoming-dev bash -lc 'ps -eo pid,ppid,stat,etime,cmd --sort=start_time'"
```

Treat Python, training, build, test, `torchrun`, `primus`, `megatron`, `pytest`, `cmake`, `ninja`, and `make` commands as active work. Ignore `sleep infinity`, `bash` shells with no active child workload, and the process inspection command itself.

If active work is running, tell the user what was found, then wait and re-check periodically:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no <node> \
  'while podman exec xiaoming-dev bash -lc "pgrep -af '\''python|torchrun|primus|megatron|pytest|cmake|ninja|make'\'' >/dev/null"; do
     podman exec xiaoming-dev bash -lc "pgrep -af '\''python|torchrun|primus|megatron|pytest|cmake|ninja|make'\''"
     sleep 60
   done'
```

Only proceed once no matching active workload remains. If the user explicitly says the existing process can be interrupted, follow their instruction.

### Step 5: Run Commands Inside xiaoming-dev

For an interactive shell inside the container:

```bash
ssh -t -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no <node> \
  'podman exec -it xiaoming-dev bash'
```

For a non-interactive command:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no <node> \
  "podman exec xiaoming-dev bash -lc '<command>'"
```

When running Primus commands, first verify the repository path inside the container:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no <node> \
  "podman exec xiaoming-dev bash -lc 'ls \"$HOME/workspace/Primus\" \"$HOME/Primus\" 2>/dev/null'"
```

Then run from the existing path:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no <node> \
  "podman exec xiaoming-dev bash -lc 'cd \"$HOME/workspace/Primus\" && <command>'"
```

## Important Notes

- Always use the user-provided node as the target and confirm if the node is missing or ambiguous.
- Prefer `ssh -o BatchMode=yes -o ConnectTimeout=10` to avoid hanging on unreachable nodes.
- Use `podman exec` into `xiaoming-dev` for actual work on the node.
- If another meaningful workload is already running inside `xiaoming-dev`, wait for it to finish before starting new work.
- Never stop, remove, or recreate the container while active work may be running unless the user explicitly approves.
