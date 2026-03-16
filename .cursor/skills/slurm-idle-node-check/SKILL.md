---
name: slurm-idle-node-check
description: Check available idle nodes in a SLURM cluster. Use when the user wants to find usable idle nodes, verify node health, check docker status on SLURM nodes, or troubleshoot cluster node availability.
---

# SLURM Idle Node Health Check

Diagnose idle nodes in a SLURM cluster: verify SSH access, check Docker service, verify workspace directory accessibility, and report usable vs problematic nodes.

## Workflow

### Step 1: Obtain Idle Node List

If the user provides a nodelist, use it directly. Otherwise:

1. Run `sinfo -h -o "%P %T %N"` to get all nodes with their **exact state**.
2. Filter to keep **only rows where the state is exactly `idle`** — exclude `drained`, `drain`, `idle*`, `down`, `mixed`, etc.
   - Recommended: `sinfo -h -o "%P %T %N" | awk '$2 == "idle"'`
   - Do **NOT** use `sinfo -t idle` alone — it also matches `drained` nodes whose base state contains `idle`.
3. If multiple partitions have idle nodes, use **AskQuestion** to let the user pick one partition.
4. Expand the nodelist with `scontrol show hostnames <nodelist>` to get individual hostnames.

### Step 2: Ensure SSH Access

1. Pick one node from the list and test: `ssh -o BatchMode=yes -o ConnectTimeout=5 <node> echo ok`
2. If it fails (password required), inform the user and propose:
   - Read `~/.ssh/id_rsa.pub`
   - **Append** (not overwrite) the public key to `~/.ssh/authorized_keys`: `cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys`
3. **Wait for user confirmation** before executing the append.
4. After appending, verify SSH works again.

### Step 3: Run Health Checks (Parallel)

SSH into each idle node and run the checks below. Use **parallel subagents** or batch shell commands to speed up.

For each node, run a single SSH command that performs all checks:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no <node> bash -c '
  # Check 1: Docker service running
  docker info > /dev/null 2>&1
  DOCKER_OK=$?

  # Check 2: Can remove existing containers
  # Try to list and remove stopped containers (dry-safe: only removes stopped ones)
  docker ps -aq --filter status=exited | head -1 | xargs -r docker rm > /dev/null 2>&1
  DOCKER_RM_OK=$?

  # Check 3: Workspace directory accessible
  WORKSPACE_DIR="<workspace_path>"
  if [ -d "$WORKSPACE_DIR" ] && [ -r "$WORKSPACE_DIR" ]; then
    WORKSPACE_OK=0
  else
    WORKSPACE_OK=1
  fi

  ERRORS=""
  if [ $DOCKER_OK -ne 0 ]; then
    ERRORS="${ERRORS}Docker service not available; "
  fi
  if [ $DOCKER_RM_OK -ne 0 ]; then
    ERRORS="${ERRORS}Cannot remove containers; "
  fi
  if [ $WORKSPACE_OK -ne 0 ]; then
    ERRORS="${ERRORS}Workspace directory not accessible ($WORKSPACE_DIR); "
  fi

  if [ -z "$ERRORS" ]; then
    echo "PASS|"
  else
    echo "FAIL|${ERRORS}"
  fi
'
```

If SSH itself fails, mark the node as `FAIL|SSH connection failed`.

#### Current Checks

| # | Check | Command | Failure Meaning |
|---|-------|---------|-----------------|
| 1 | Docker service is running | `docker info` | Docker daemon not started or user has no permission |
| 2 | Can remove existing containers | `docker ps -aq --filter status=exited \| xargs -r docker rm` | Cannot clean up containers |
| 3 | Workspace directory accessible | `[ -d "$WORKSPACE_DIR" ] && [ -r "$WORKSPACE_DIR" ]` | Shared filesystem not mounted or path unreachable on this node |

`<workspace_path>` is the **absolute path of the current repository root** (i.e. the workspace directory where the agent is operating). Determine it at runtime via `pwd` or from the workspace context, then substitute into the script.

> **To add more checks later**: append new check logic inside the `bash -c '...'` block above and update the table.

### Step 4: Display Results

Print **two tables** — one for healthy nodes, one for problematic nodes.

Table format (markdown):

```
| Node | Status | Issue |
|------|--------|-------|
| gpu01 | PASS | - |
```

- Column 1: Node name
- Column 2: PASS or FAIL
- Column 3: Issue description (or `-` if PASS)

Show the healthy-node table first, then the problematic-node table.

### Step 5: Summary

**Always write the summary in English**, regardless of the conversation language.

Provide a summary block:

```
## Summary

- Total idle nodes: <N>
- Healthy: <H>
- Problematic: <P>

### Healthy NODELIST (srun-ready)
<compressed nodelist, e.g. gpu[01-04,06]>

### Problematic NODELIST
<compressed nodelist, e.g. gpu[05,07-08]>
```

To generate compressed nodelists, use: `scontrol show hostlistctrl <node1,node2,...>`
(or `echo "node1,node2,..." | tr ',' '\n' | scontrol show hostlistctrl`)

If `scontrol show hostlistctrl` is not available, fall back to: `scontrol show hostnames` for verification and manually compress contiguous ranges.

## Important Notes

- **Never overwrite** `~/.ssh/authorized_keys` — always **append**.
- **Always ask for user confirmation** before modifying SSH keys.
- Run node checks **in parallel** to save time on large clusters.
- Use `-o StrictHostKeyChecking=no` to avoid interactive SSH prompts.
- Use `-o ConnectTimeout=10` to avoid hanging on unreachable nodes.
