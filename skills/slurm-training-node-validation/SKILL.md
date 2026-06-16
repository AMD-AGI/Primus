---
name: slurm-training-node-validation
description: Validate SLURM cluster nodes by running actual training jobs in groups. Use when the user wants to test which idle nodes can successfully run training, verify node health through real workloads, or identify broken nodes in a SLURM cluster.
---

# SLURM Training Node Validation

Launch short training jobs on groups of idle nodes to determine which nodes are healthy enough to run real workloads.

## Prerequisites

Use the **slurm-idle-node-check** skill first (or manually) to get the list of idle, healthy nodes. If the user has not provided a nodelist, obtain one using the slurm-idle-node-check skill's Step 1 (filter for exactly `idle` state).

## Workflow

### Step 0: Pre-flight Checks

Before launching any training, resolve these runtime parameters:

#### HF_TOKEN

Check if `HF_TOKEN` is set in the environment:
```bash
echo "${HF_TOKEN:-NOT_SET}"
```
If it is `NOT_SET` or empty, **ask the user to provide it** using AskQuestion or a conversational prompt. Do NOT proceed without a valid `HF_TOKEN`.

#### Partition

Determine the SLURM partition to use:

1. Check if `amd-aig` exists:
   ```bash
   sinfo -h -o "%P" | tr -d '*' | sort -u
   ```
2. If `amd-aig` is in the list, use it.
3. If `amd-aig` is NOT in the list:
   - If there is **exactly one** partition available, use it automatically.
   - If there are **multiple** partitions, use **AskQuestion** to let the user pick one.

Store the resolved partition in `$PARTITION` for use in the launch command.

#### Runner Script Path

The runner script path is **relative to the workspace root**, not hardcoded:
```bash
RUNNER_SCRIPT="$(pwd)/runner/primus-cli-slurm.sh"
```
Verify it exists before proceeding:
```bash
if [ ! -f "$RUNNER_SCRIPT" ]; then
  echo "ERROR: runner/primus-cli-slurm.sh not found at $RUNNER_SCRIPT"
  exit 1
fi
```

### Step 1: Get Idle Nodes and Form Groups

1. If the user provides a nodelist, use it. Otherwise, run:
   ```bash
   sinfo -h -o "%P %T %N" | awk '$2 == "idle" {print $3}'
   ```
   Then expand with `scontrol show hostnames <nodelist>` to get individual hostnames.

2. If multiple partitions have idle nodes, use **AskQuestion** to let the user pick a partition.

3. Split nodes into **groups of 4**. If the total is not a multiple of 4, the remaining nodes (fewer than 4) form an incomplete group — **skip** that group and inform the user which nodes were excluded.

4. Display the grouping plan to the user before launching. Example:
   ```
   Group 1: node01,node02,node03,node04
   Group 2: node05,node06,node07,node08
   Group 3: node09,node10,node11,node12
   Skipped (incomplete group): node13
   ```

### Step 2: Launch Training Jobs

For each group, launch a training job **in the background** with a dedicated log file.

#### Directory Structure

Use `./output/` relative to the workspace root. The timestamp is generated **once** at the start and reused for all groups:

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASEDIR="./output/slurm-training-node-validation-${TIMESTAMP}"
```

For each group `$i`, first compress the 4 hostnames into SLURM bracket notation, then create the group directory and marker file:

```bash
COMPRESSED_NODELIST=$(scontrol show hostlist "node1,node2,node3,node4")
# e.g. "uswslocpm2m-106-[1628,1691,1697,1707]"

GROUPDIR="${BASEDIR}/group_${i}"
mkdir -p "$GROUPDIR"

# Create nodelist marker file — filename and content both use the SLURM-compressed nodelist
echo "${COMPRESSED_NODELIST}" > "${GROUPDIR}/group_${i}_nodelist_${COMPRESSED_NODELIST}"
```

**The `$COMPRESSED_NODELIST` must always be in SLURM prefix-compressed format** (e.g. `uswslocpm2m-106-[1628,1691,1697,1707]`), NOT individual hostnames joined by commas. Always use `scontrol show hostlist` to produce it.

The resulting directory layout looks like:

```
./output/slurm-training-node-validation-20260316_201333/
├── group_1/
│   ├── log_group_1.txt                                          # training log
│   ├── group_1_nodelist_uswslocpm2m-106-[1628,1691,1697,1707]  # content: the compressed nodelist
│   └── success  OR  failed                                      # result file (created after job finishes)
├── group_2/
│   ├── log_group_2.txt
│   ├── group_2_nodelist_uswslocpm2m-106-[1711,1713,1719,1723]
│   └── success  OR  failed
└── group_3/
    ├── ...
```

#### Launch Command

For each group (index `$i`, nodelist `$NODELIST`), run:

```bash
(
  export SLURM_NODELIST='<nodelist_for_this_group>'
  export USING_AINIC=1
  export NCCL_IB_HCA="ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1"
  export GLOO_SOCKET_IFNAME=ens9np0
  export NCCL_SOCKET_IFNAME=ens9np0
  export CLEAN_DOCKER_CONTAINER=1
  export GPU_MAX_HW_QUEUES=4
  export HF_TOKEN="${HF_TOKEN}"

  bash "${RUNNER_SCRIPT}" \
    -N 4 --time=48:00:00 --partition="${PARTITION}" \
    -- --image docker.io/tasimage/primus:pr-563-ainic --clean \
    -- train pretrain \
    --config examples/megatron/configs/MI355X/deepseek_v2_lite-BF16-pretrain.yaml \
    --num_layers 10 --moe_layer_freq 1 --train_iters 5 \
    --micro_batch_size 2 --global_batch_size 64 \
    --cross_entropy_fusion_impl te --cross_entropy_loss_fusion True \
    --recompute_num_layers 0 --recompute_granularity full \
    --recompute_method block --disable_last_saving True \
    --profile False --use_pytorch_profiler False \
    --profile_step_end 7 --profile_step_start 6
) > "${GROUPDIR}/log_group_${i}.txt" 2>&1 &
```

**Important details:**
- Replace `<nodelist_for_this_group>` with the compressed nodelist for this group. Use `scontrol show hostlist node01,node02,node03,node04` to compress it, or just use the comma-separated list.
- `${HF_TOKEN}` — resolved in Step 0 (from environment or user input).
- `${RUNNER_SCRIPT}` — resolved in Step 0 as `$(pwd)/runner/primus-cli-slurm.sh`.
- `${PARTITION}` — resolved in Step 0 (auto-detected or user-selected).
- Each group runs in a **subshell** so environment variables don't leak between groups.
- Capture the PID of each background job: `PIDS[$i]=$!`
- Log file: `log_group_1.txt`, `log_group_2.txt`, etc. (1-indexed)
- The nodelist marker file name embeds the **compressed** nodelist (e.g. `group_1_nodelist_uswslocpm2m-106-[1628,1691,1697,1707]`). The file content is the compressed nodelist string itself.

After launching all groups, print:
```
Launched <N> training groups. Output directory: $BASEDIR
Waiting for all jobs to complete...
```

### Step 3: Wait and Monitor

1. Use `wait` to wait for all background PIDs to finish:
   ```bash
   for i in "${!PIDS[@]}"; do
     wait ${PIDS[$i]} 2>/dev/null
     EXIT_CODES[$i]=$?
   done
   ```

2. Alternatively, poll logs periodically (every 60 seconds) to provide progress updates to the user. Use the Shell tool with appropriate `block_until_ms` to monitor.

3. **Early failure detection**: While polling, if a log contains `Exited with exit code 1` but does NOT contain `iteration        3`, the job has already failed — no need to wait further. Mark it as failed immediately.

4. **Timeout**: If a job hasn't finished after **5 minutes**:
   - Kill the background process: `kill ${PIDS[$i]} 2>/dev/null`
   - Find and cancel any SLURM jobs launched by this group. Search for jobs on the group's nodes:
     ```bash
     # Find SLURM job IDs associated with this group's nodes
     squeue -u "$USER" -h -o "%i %N" | while read jobid nodes; do
       # Check if any of this group's nodes appear in the job's nodelist
       if echo "$nodes" | grep -qF "<any_node_from_this_group>"; then
         scancel "$jobid"
       fi
     done
     ```
     Alternatively, if the training script outputs a SLURM job ID in the log, extract and cancel it:
     ```bash
     JOB_ID=$(grep -oP 'Submitted batch job \K\d+' "${GROUPDIR}/log_group_${i}.txt" 2>/dev/null || true)
     if [ -n "$JOB_ID" ]; then
       scancel "$JOB_ID"
     fi
     ```
   - Mark the group as failed with reason "Timeout after 5 minutes".

### Step 4: Check Results and Write Status Files

For each group, determine the result by checking the log file:

```bash
HAS_ITER3=$(grep -c "iteration        3" "${GROUPDIR}/log_group_${i}.txt" 2>/dev/null || echo 0)
HAS_EXIT1=$(grep -c "Exited with exit code 1" "${GROUPDIR}/log_group_${i}.txt" 2>/dev/null || echo 0)
```

Decision logic:
1. If `HAS_ITER3 > 0` → **PASS** (regardless of exit code).
2. If `HAS_ITER3 == 0` and `HAS_EXIT1 > 0` → **FAIL** (process exited with error before reaching iteration 3).
3. If `HAS_ITER3 == 0` and `HAS_EXIT1 == 0` and job timed out → **FAIL** (timeout).

**Note**: The pattern `iteration        3` uses **8 spaces** between "iteration" and "3". This is the exact pattern printed by Megatron-LM during training. Use `grep` with the exact string.

#### On success (pattern found):

Create an empty `success` file in the group directory:

```bash
touch "${GROUPDIR}/success"
```

#### On failure (pattern not found):

Extract error context and write a `failed` file with error details:

```bash
# Collect last 50 lines for error context
ERROR_CONTEXT=$(tail -50 "${GROUPDIR}/log_group_${i}.txt")

# Look for known error patterns
ERRORS=""
for pattern in "RuntimeError" "NCCL error" "OutOfMemoryError" "OOM" "Timeout" "ConnectionError" "CalledProcessError" "CUDA error" "Exited with exit code" "Error encountered progressing operation=Connect" "couldn't chdir to"; do
  match=$(grep -m1 "$pattern" "${GROUPDIR}/log_group_${i}.txt" 2>/dev/null || true)
  if [ -n "$match" ]; then
    ERRORS="${ERRORS}${match}\n"
  fi
done

# Write failed file with error info and possible cause
cat > "${GROUPDIR}/failed" <<FAILEOF
=== Training Validation Failed ===
Group: ${i}
Nodelist: ${COMPRESSED_NODELIST}
Log: ${GROUPDIR}/log_group_${i}.txt

--- Detected Errors ---
${ERRORS:-No specific error pattern detected.}

--- Possible Cause ---
$(if echo "$ERRORS" | grep -q "NCCL"; then
    echo "NCCL communication failure — possible network issue or faulty NIC on one of the nodes."
  elif echo "$ERRORS" | grep -q "OOM\|OutOfMemory"; then
    echo "GPU out of memory — possible hardware issue or memory leak on one of the nodes."
  elif echo "$ERRORS" | grep -q "Timeout"; then
    echo "Operation timed out — possible node hang or network partition."
  elif echo "$ERRORS" | grep -q "CUDA error"; then
    echo "CUDA error — possible GPU hardware fault on one of the nodes."
  elif echo "$ERRORS" | grep -q "couldn't chdir to"; then
    echo "Working directory not accessible — network filesystem likely not mounted on one or more nodes."
  else
    echo "Training did not reach iteration 3. Check the log file for details."
  fi)

--- Last 50 Lines of Log ---
${ERROR_CONTEXT}
FAILEOF
```

If the job was killed due to timeout, write:

```bash
cat > "${GROUPDIR}/failed" <<FAILEOF
=== Training Validation Failed ===
Group: ${i}
Nodelist: ${COMPRESSED_NODELIST}

--- Detected Errors ---
Timeout after 5 minutes.

--- Possible Cause ---
Job exceeded the 5-minute timeout. Nodes may be hung or experiencing severe performance issues.
FAILEOF
```

**Important**: If a group has a non-zero exit code but the success pattern IS found in the log, still create `success` (not `failed`), because training may exit non-zero after completing.

### Step 5: Display Results

**Always write the summary in English**, regardless of the conversation language.

#### Results Table

```markdown
## Training Validation Results

| Group | Nodelist | Status | Error |
|-------|----------|--------|-------|
| 1 | node[01-04] | PASS | - |
| 2 | node[05-08] | FAIL | NCCL timeout on node06 |
| 3 | node[09-12] | PASS | - |
```

#### Summary

```markdown
## Summary

- Total groups tested: <N>
- Passed: <P>
- Failed: <F>
- Output directory: <BASEDIR>

### Passed Nodelist (training-ready)
<SLURM-format compressed nodelist, e.g. uswslocpm2m-106-[1628,1691,1697,1707,1711,1713,1719,1723]>

### Failed Nodelist
<SLURM-format compressed nodelist, e.g. uswslocpm2m-106-[1730,1735,1741,1749]>
```

#### Generating SLURM-format Compressed Nodelists

Collect all individual hostnames for passed (or failed) groups, then compress into SLURM nodelist format:

```bash
scontrol show hostlist <comma-separated-hostnames>
```

Example: if passed nodes are `uswslocpm2m-106-1628,uswslocpm2m-106-1691,uswslocpm2m-106-1697,uswslocpm2m-106-1707,uswslocpm2m-106-1711,uswslocpm2m-106-1713`, then:

```bash
$ scontrol show hostlist uswslocpm2m-106-1628,uswslocpm2m-106-1691,uswslocpm2m-106-1697,uswslocpm2m-106-1707,uswslocpm2m-106-1711,uswslocpm2m-106-1713
uswslocpm2m-106-[1628,1691,1697,1707,1711,1713]
```

Nodes with the **same prefix** are automatically merged into bracket notation by `scontrol show hostlist`. The output is the final SLURM-format nodelist to display in the summary.

If `scontrol show hostlist` is unavailable, manually group nodes by common prefix and compress into bracket notation (e.g. `prefix-[id1,id2,id3]`).

#### Write Summary to File

In addition to displaying the summary to the user, write the same content to a `summary` file under the output directory:

```bash
cat > "${BASEDIR}/summary" <<EOF
## Training Validation Results

| Group | Nodelist | Status | Error |
|-------|----------|--------|-------|
| 1 | <nodelist> | PASS | - |
| 2 | <nodelist> | FAIL | <error> |
...

## Summary

- Total groups tested: <N>
- Passed: <P>
- Failed: <F>
- Output directory: ${BASEDIR}

### Passed Nodelist (training-ready)
<SLURM-format compressed nodelist>

### Failed Nodelist
<SLURM-format compressed nodelist>
EOF
```

The final directory layout includes:

```
./output/slurm-training-node-validation-20260316_201333/
├── summary                # overall results and nodelists
├── group_1/
│   ├── log_group_1.txt
│   ├── group_1_nodelist_...
│   └── success or failed
├── group_2/
│   └── ...
```

## Important Notes

- **Always run jobs in background** so all groups execute concurrently.
- **Isolate environment variables** per group using subshells `( ... )`.
- **Kill timed-out jobs** — do not let them hang indefinitely.
- Use `-o StrictHostKeyChecking=no` for any SSH operations.
- The **exact success pattern** is `iteration        3` (8 spaces). Do not modify the spacing.
- Output is stored under `./output/slurm-training-node-validation-<TIMESTAMP>/`. Each group has its own subdirectory with log, status file, and nodelist marker.
- The **nodelist marker file** is an empty file whose name encodes the group's compressed nodelist (e.g. `group_1_nodelist_uswslocpm2m-106-[1628,1691,1697,1707]`). This makes it easy to identify which nodes belong to each group by listing the directory.
- The **status file** is either `success` (empty) or `failed` (contains error details and possible cause).
- If a group has a non-zero exit code but the success pattern is found in the log, still create `success` (training may exit non-zero after completing).
