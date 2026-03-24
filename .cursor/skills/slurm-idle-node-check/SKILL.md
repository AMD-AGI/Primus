---
name: slurm-idle-node-check
description: Check available idle nodes in a SLURM cluster. Use when the user wants to find usable idle nodes, verify node health, check docker status on SLURM nodes, check NIC QoS/DCQCN configuration, check RDMA link status, validate GID table, or troubleshoot cluster node availability.
---

# SLURM Idle Node Health Check

Diagnose idle nodes in a SLURM cluster: verify SSH access, check Docker service, verify workspace directory accessibility, validate NIC QoS/DCQCN configuration consistency, check RDMA link status, validate GID table, and report usable vs problematic nodes.

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
  docker ps -aq --filter status=exited | head -1 | xargs -r docker rm > /dev/null 2>&1
  DOCKER_RM_OK=$?

  # Check 3: Workspace directory accessible
  WORKSPACE_DIR="<workspace_path>"
  if [ -d "$WORKSPACE_DIR" ] && [ -r "$WORKSPACE_DIR" ]; then
    WORKSPACE_OK=0
  else
    WORKSPACE_OK=1
  fi

  # Check 4: nicctl QoS configuration
  QOS_OUTPUT=$(sudo nicctl show qos 2>&1)
  QOS_RC=$?
  QOS_HASH=""
  QOS_REQUIRED_OK=1
  if [ $QOS_RC -eq 0 ]; then
    QOS_HASH=$(echo "$QOS_OUTPUT" | md5sum | awk "{print \$1}")
    if echo "$QOS_OUTPUT" | grep -q "Classification type[[:space:]]*:[[:space:]]*DSCP" && \
       echo "$QOS_OUTPUT" | grep -q "DSCP[[:space:]]*:[[:space:]]*10[[:space:]]*==>[[:space:]]*priority[[:space:]]*:[[:space:]]*0" && \
       echo "$QOS_OUTPUT" | grep -q "PFC no-drop priorities[[:space:]]*:[[:space:]]*0"; then
      QOS_REQUIRED_OK=0
    fi
  fi

  # Check 5: nicctl DCQCN configuration
  DCQCN_OUTPUT=$(sudo nicctl show dcqcn 2>&1)
  DCQCN_RC=$?
  DCQCN_HASH=""
  if [ $DCQCN_RC -eq 0 ]; then
    DCQCN_HASH=$(echo "$DCQCN_OUTPUT" | md5sum | awk "{print \$1}")
  fi

  # Check 6: RDMA link status — all links must be state ACTIVE, physical_state LINK_UP
  RDMA_OUTPUT=$(rdma link 2>&1)
  RDMA_RC=$?
  RDMA_OK=0
  RDMA_BAD_DETAIL=""
  if [ $RDMA_RC -eq 0 ]; then
    BAD_LINKS=$(echo "$RDMA_OUTPUT" | grep "^link " | grep -v "state ACTIVE physical_state LINK_UP")
    if [ -n "$BAD_LINKS" ]; then
      RDMA_OK=1
      RDMA_BAD_DETAIL=$(echo "$BAD_LINKS" | awk "{print \$2, \$4, \$6}" | head -5 | tr "\n" ", ")
    fi
  else
    RDMA_OK=1
  fi

  # Check 7: show_gid — each device must have at least index 0 and 1;
  #   ionic-prefixed devices must have exactly index 0 and 1 (no other indices)
  GID_OUTPUT=$(show_gid 2>&1)
  GID_RC=$?
  GID_OK=0
  GID_ERRORS=""
  if [ $GID_RC -eq 0 ]; then
    DEVICES=$(echo "$GID_OUTPUT" | awk "NR>2 && NF>=6 {print \$1}" | sort -u)
    for DEV in $DEVICES; do
      INDICES=$(echo "$GID_OUTPUT" | awk -v dev="$DEV" "NF>=6 && \$1 == dev {print \$3}" | sort -n -u)
      HAS_0=$(echo "$INDICES" | grep -c "^0$")
      HAS_1=$(echo "$INDICES" | grep -c "^1$")
      if [ "$HAS_0" -eq 0 ] || [ "$HAS_1" -eq 0 ]; then
        GID_ERRORS="${GID_ERRORS}${DEV} missing index 0 or 1; "
        GID_OK=1
      fi
      if echo "$DEV" | grep -q "^ionic"; then
        INDEX_COUNT=$(echo "$INDICES" | wc -l)
        if [ "$INDEX_COUNT" -ne 2 ]; then
          GID_ERRORS="${GID_ERRORS}${DEV} has unexpected indices (expected only 0,1, got: $(echo $INDICES | tr "\n" ",")); "
          GID_OK=1
        fi
      fi
    done
  else
    GID_OK=1
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
  if [ $QOS_RC -ne 0 ]; then
    ERRORS="${ERRORS}nicctl show qos failed; "
  elif [ $QOS_REQUIRED_OK -ne 0 ]; then
    ERRORS="${ERRORS}QoS config missing required settings (need: Classification type=DSCP, DSCP 10==>priority 0, PFC no-drop priorities=0); "
  fi
  if [ $DCQCN_RC -ne 0 ]; then
    ERRORS="${ERRORS}nicctl show dcqcn failed; "
  fi
  if [ $RDMA_RC -ne 0 ]; then
    ERRORS="${ERRORS}rdma link command failed; "
  elif [ $RDMA_OK -ne 0 ]; then
    ERRORS="${ERRORS}RDMA links not all ACTIVE/LINK_UP (${RDMA_BAD_DETAIL}); "
  fi
  if [ $GID_RC -ne 0 ]; then
    ERRORS="${ERRORS}show_gid command failed; "
  elif [ $GID_OK -ne 0 ]; then
    ERRORS="${ERRORS}GID table invalid (${GID_ERRORS}); "
  fi

  if [ -z "$ERRORS" ]; then
    echo "PASS||${QOS_HASH}|${DCQCN_HASH}"
  else
    echo "FAIL|${ERRORS}|${QOS_HASH}|${DCQCN_HASH}"
  fi
'
```

If SSH itself fails, mark the node as `FAIL|SSH connection failed|||`.

**Output format**: `STATUS|errors|qos_hash|dcqcn_hash` — fields separated by `|`. The `qos_hash` and `dcqcn_hash` are md5 hashes of the full `nicctl show qos` and `nicctl show dcqcn` output respectively, used for cross-node consistency comparison in Step 3b.

#### Current Checks

| # | Check | Command | Failure Meaning |
|---|-------|---------|-----------------|
| 1 | Docker service is running | `docker info` | Docker daemon not started or user has no permission |
| 2 | Can remove existing containers | `docker ps -aq --filter status=exited \| xargs -r docker rm` | Cannot clean up containers |
| 3 | Workspace directory accessible | `[ -d "$WORKSPACE_DIR" ] && [ -r "$WORKSPACE_DIR" ]` | Shared filesystem not mounted or path unreachable on this node |
| 4 | NIC QoS config valid | `sudo nicctl show qos` | QoS not configured or missing required settings (DSCP classification, DSCP 10→priority 0, PFC no-drop priorities 0) |
| 5 | NIC DCQCN config present | `sudo nicctl show dcqcn` | DCQCN not configured on this node |
| 6 | RDMA links all active | `rdma link` | One or more RDMA links not in state ACTIVE / physical_state LINK_UP |
| 7 | GID table valid | `show_gid` | Device missing required GID indices (all devices need index 0 & 1; ionic devices must have exactly 0 & 1, no extra indices) |

`<workspace_path>` is the **absolute path of the current repository root** (i.e. the workspace directory where the agent is operating). Determine it at runtime via `pwd` or from the workspace context, then substitute into the script.

> **To add more checks later**: append new check logic inside the `bash -c '...'` block above and update the table.

### Step 3b: Cross-Node NIC Configuration Consistency Check

After collecting results from all nodes, perform a **cross-node consistency comparison** for the NIC configuration checks (QoS and DCQCN).

1. Parse the `qos_hash` and `dcqcn_hash` fields from each node's output.
2. For **QoS**: group nodes by their `qos_hash`. If more than one distinct hash exists (ignoring empty hashes from failed nodes), the QoS configuration is **inconsistent**. The group with the most nodes is treated as the "majority" (expected) configuration; nodes in other groups are marked as inconsistent.
3. For **DCQCN**: apply the same majority-based grouping logic on `dcqcn_hash`.
4. For any node flagged as inconsistent, append to its error string: `QoS config inconsistent with majority;` or `DCQCN config inconsistent with majority;` and change its status to FAIL.
5. If you need to show the user what differs, SSH into one representative node from each group and re-run the relevant `sudo nicctl show` command to display the actual output side by side.

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

If any NIC configuration inconsistency was detected in Step 3b, add a separate **NIC Configuration Consistency** section after the tables:
- State whether QoS and DCQCN configs are consistent across all nodes.
- If inconsistent, list which nodes belong to which configuration group (by hash), and show the differing output for each group so the user can compare.

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

### Step 6: Save Report to File

After displaying results to the user, save the full report as a Markdown file under the **repo root** at:

```
output/skills/slurm-idle-node-check-YYYYMMDD.HHMM.md
```

- `YYYYMMDD.HHMM` is the **current date and time** when the check finishes (e.g. `20260318.1107`).
- Create the `output/skills/` directory if it does not exist (`mkdir -p`).
- The file should contain the **complete report in English**, structured as follows:

```markdown
# SLURM Idle Node Health Check Report

- **Date**: YYYY-MM-DD HH:MM
- **Partition**: <partition name>
- **Total idle nodes**: <N>
- **Healthy**: <H>
- **Problematic**: <P>

## Healthy Nodes

| Node | Status | Issue |
|------|--------|-------|
| ... | PASS | - |

### Healthy NODELIST (srun-ready)

<compressed nodelist>

## Problematic Nodes

| Node | Status | Issue |
|------|--------|-------|
| ... | FAIL | <issue description> |

### Problematic NODELIST

<compressed nodelist>

## NIC Configuration Consistency

- **QoS**: consistent / inconsistent (details if inconsistent)
- **DCQCN**: consistent / inconsistent (details if inconsistent)

<If inconsistent, include group details and differing output>
```

- Use the **Write** tool (or shell `cat > file`) to create the file.
- After saving, print the file path so the user knows where to find it.

## Important Notes

- **Never overwrite** `~/.ssh/authorized_keys` — always **append**.
- **Always ask for user confirmation** before modifying SSH keys.
- Run node checks **in parallel** to save time on large clusters.
- Use `-o StrictHostKeyChecking=no` to avoid interactive SSH prompts.
- Use `-o ConnectTimeout=10` to avoid hanging on unreachable nodes.
