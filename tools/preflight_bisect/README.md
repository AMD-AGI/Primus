# preflight_bisect

Recursive Slurm-nodelist bisection for finding the node(s) causing an NCCL hang
or failure during Primus preflight.

Each trial runs `preflight --perf-test` on a subset of nodes via
[`runner/primus-cli-direct-preflight.sh`](../../runner/primus-cli-direct-preflight.sh).
Subsets that pass are pruned; subsets that fail or time out are split in half
and re-tested until a singleton (the suspect node) is isolated.

## Prerequisites

1. Working non-container preflight setup (Python venv on shared FS, `VENV_PATH`
   exported). See [`docs/run-preflight-without-container.md`](../../docs/run-preflight-without-container.md).
2. Run from the Slurm **login / head node** (needs `scontrol` and `srun`).
3. `scontrol show hostnames "<nodelist>"` must resolve your node expression.

## Minimal example (32-node cluster)

```bash
# From the Slurm login node:
cd /path/to/Primus
export VENV_PATH=~/envs/preflight/.venv/bin/activate

python tools/preflight_bisect/bisect.py \
    --nodelist "node[01-32]" \
    --partition gpus \
    --trial-timeout-sec 900 \
    --output-dir ./bisect-out \
    --preflight-env NCCL_CROSS_NIC=1 \
    --preflight-env NCCL_PXN_DISABLE=0
```

Outputs under `./bisect-out/`:
- `trial-000.log`, `trial-001.log`, ... - full stdout/stderr of each `srun`.
- `summary.txt` - one line per trial plus `SUSPECT_NODES: ...`.

Example `summary.txt`:

```
2026-04-20T18:22:04+00:00 bisect nodes=32
[000] N=32 FAIL  nodes=node01..node32
[001] N=16 PASS  nodes=node01..node16
[002] N=16 FAIL  nodes=node17..node32
[003] N=8  PASS  nodes=node17..node24
[004] N=8  FAIL  nodes=node25..node32
[005] N=4  PASS  nodes=node25..node28
[006] N=4  FAIL  nodes=node29..node32
[007] N=2  PASS  nodes=node29..node30
[008] N=2  FAIL  nodes=node31..node32
[009] N=1  FAIL  nodes=node31
[010] N=1  PASS  nodes=node32
SUSPECT_NODES: node31
```

## AINIC (Pensando Pollara) cluster example

```bash
python tools/preflight_bisect/bisect.py \
    --nodelist "node[01-32]" --partition gpus \
    --trial-timeout-sec 900 --output-dir ./bisect-out \
    --preflight-env USING_AINIC=1 \
    --preflight-env NCCL_IB_GID_INDEX=1 \
    --preflight-env NCCL_PXN_DISABLE=0
```

## Common flags

| Flag | Default | Notes |
|---|---|---|
| `--nodelist` | required | Slurm expression, expanded via `scontrol show hostnames` |
| `-p`, `--partition` | (unset) | `srun -p` |
| `--trial-timeout-sec` | 900 | Wall-clock per trial; on timeout the trial is marked `HANG` |
| `--slurm-time` | `00:45:00` | `srun -t` |
| `--cpus-per-task` | 128 | `srun -c` (per the non-container doc) |
| `--gpus-per-node` | 8 | `srun --gpus-per-node` |
| `--preflight-env KEY=VALUE` | (repeatable) | Passed through as `runner/primus-cli-direct-preflight.sh --env KEY=VALUE` |
| `--output-dir` | `./bisect-out` | Trial logs + `summary.txt` |
| `--runner` | `runner/primus-cli-direct-preflight.sh` | Override if you ship a custom runner |
| `--scancel-user-on-hang` | off | DANGEROUS: runs `scancel --user $USER` on timeout |

## How classification works

- `srun` exits 0 -> `PASS`
- `srun` exits non-zero within timeout -> `FAIL` (subset gets bisected further)
- `srun` exceeds `--trial-timeout-sec` -> `HANG` (process group SIGKILL'd; subset bisected further)

`FAIL` and `HANG` are treated identically for routing: both mean "this subset is bad".

## Caveats

- **Scale-only hangs**: if the bug only reproduces at full N, every half-size
  subset will `PASS` and the tool reports `SUSPECT_NODES: (none)`. That itself
  is useful signal: the issue is fabric/scale, not a single node.
- **Multiple bad nodes**: both halves can fail; the final suspects list is the
  union of singleton failures.
- **Timeout tuning**: start at roughly 2-3x the expected healthy full-N runtime.
  Too short -> false HANGs -> over-reporting suspects.
- **Killing hung steps**: the script `SIGKILL`s the `srun` process group on
  timeout, which is usually enough for Slurm to release the step's nodes. If a
  trial's nodes remain `ALLOCATED` after a kill, pass `--scancel-user-on-hang`,
  but only if you have no unrelated Slurm jobs running (it cancels everything
  for `$USER`).

## Testing

For a POC like this, **manual end-to-end on the target cluster is the fastest
useful test**:

1. Pick a small healthy nodelist (e.g. 2 nodes). It should exit 0 with one
   `PASS` trial and `SUSPECT_NODES: (none)`.
2. Then run on a larger nodelist known to have (or reproduce) the hang to
   confirm convergence.

A pytest suite isn't worth writing for the MVP because the interesting
behavior is all in the `srun` / `scontrol` / Slurm interaction, which can't
be faithfully mocked. If you want a quick offline sanity check of just the
bisection logic, you can point `--runner` at a small shim script that returns
exit codes based on the nodelist (e.g. exit 1 iff `node31` is in the list).
That would verify only the recursion and summary-writing, which have almost
no logic. Skip until you're promoting this out of POC status.
