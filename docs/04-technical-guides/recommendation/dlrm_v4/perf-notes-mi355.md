# Performance Optimizations — MI355X

MI355X (`gfx950`, 8 GPU/node) on top of the MI350X tree: cluster B and, from ship17, a second MI355X cluster, "cluster C" (hosts `c*`). The Hosts column of the latest evaluations tells their rows apart.
Companion to `perf-notes.md`.

- **Base:** the MI350X tree at the commit in the mapping below, same recipe as MI350X:
  - LR 1e-6 / 2e-6 / 4e-6;
  - local batch 1024;
  - same eval cadence and skip.
- **Comparison:** time to target vs the MI350X rows in `perf-notes.md` for the same base, at equal samples to target.
- **Run conditions:**
  - fresh container per run;
  - cold page cache;
  - MLPerf 6.1.0 compliance checker SUCCESS on every log listed.

## Commit mapping (MI355X ← MI350X base)

| Build | MI355X commit (`dlrmv4_opt-mi355`) | MI350X commit (`dlrmv4_opt`) |
|---|---|---|
| ship17 | `377e04d` | `9dd4b02` (tip `f16173d`) |
| ship16 | `b62c4af` | `544e691` (tip `78f5c82`) |
| ship15.1 | `9dee86f` | `d905151` (tip `b23945a`) |
| ship15 | `4e5b23d` | `4401cf2` (tip `34d35d6`) |
| ship14 | `b87bc0d` | `c68395d` (tip `35007e0`) |
| ship13 | `f6e2a57` | `18d12b4` (tip `d93920d`) |

Each MI350X ship is taken by rebasing the MI355X commits onto it (ship14 – ship17: no conflicts); a tree-preserving merge then keeps the previous build's line in the branch history, so every MI355X commit above is on `dlrmv4_opt-mi355`. The MI355X changes are the same across builds: launchers, toolkit, docs, a default-off guard, and from ship14 the GDR-flush fix (row 7). Row 8 (memfd mapping, needed on the cluster C kernel) is on the branch after `377e04d`: the cluster C ship17 rows ran `e2a61f98`, which is `377e04d`'s trainer plus row 8. `rcp/mi355/f6e2a57` is the ship13 RCP campaign.

## Latest evaluations

Each row is measured on the MI355X commit built on the MI350X base it names (see the mapping above), and compared with the MI350X result for the same base.

| MI350X base | MI355X commit | Nodes | Hosts | MI350X | MI355X + fixes | Speedup | Job | Status |
|---|---|---|---|---|---|---|---|---|
| ship17 `9dd4b02` | `377e04d` | 1 | b044 | 1092.2 s | **936.9 s** | **1.166×** | 219784 | Done (3rd attempt): equal samples (66,519,040), AUC 0.75299, compliance SUCCESS; −12.8% vs MI355X ship16 |
| ship17 `9dd4b02` | `e2a61f98` (`377e04d` + row 8) | 1 | c23 | 1092.2 s | **927.9 s** | **1.177×** | 8907 | Done (1st attempt with row 8; without it, job 8894 failed at the first m4 window): equal samples (66,519,040), AUC 0.75338, compliance SUCCESS; −1.0% vs cluster B ship17 1n |
| ship17 `9dd4b02` | `377e04d` | 2 | — | 722.3 s | _pending_ | _pending_ | 221127 | Queued (burst; hyperloom / primus copies 221131 / 221132, first past setup runs); 3 attempts failed preflight on 022 (`ionic_5` down), now excluded |
| ship17 `9dd4b02` | `e2a61f98` (`377e04d` + row 8) | 2 | c35, c03 | 722.3 s | **620.0 s** | **1.165×** | 8908 | Done (1st attempt with row 8; without it, job 8897 failed the same way): equal samples (84,869,120), AUC 0.75694, compliance SUCCESS |
| ship17 `9dd4b02` | `377e04d` | 4 | — | 495.2 s | _pending_ | _pending_ | 221128 | Queued (burst; hyperloom / primus copies 221133 / 221134) |
| ship17 `9dd4b02` | `e2a61f98` (`377e04d` + row 8) | 4 | c35, c03, c12, c10 | 495.2 s | **427.9 s** | **1.157×** | 8919 | Done (1st attempt to reach `run_start`; 8909 stopped at staging on g41, a root-owned lock file left by an older build): equal samples (110,100,480), AUC 0.75998, compliance SUCCESS |
| ship16 `544e691` | `b62c4af` | 1 | b266 | 1260.3 s | **1074.4 s** | **1.173×** | 215032 | Done (6th attempt): equal samples (66,519,040), AUC 0.75263, compliance SUCCESS |
| ship16 `544e691` | `b62c4af` | 2 | b187, b188 | 823.2 s | **702.0 s** | **1.173×** | 214956 | Done (7th attempt): equal samples (84,869,120), AUC 0.75645, compliance SUCCESS; −1.6% vs MI355X ship15.1 |
| ship16 `544e691` | `b62c4af` | 4 | — | 547.0 s | not run | — | — | Not requested |
| ship15.1 `d905151` | `9dee86f` | 1 | b061 | 1316.8 s | _pending_ (~1089 s projected) | _pending_ (~1.21× projected) | 211518 | Trains normally (AUC at 1st eval 0.6347, = ship14); runs cancelled at the 20-min burst limit before target. Projection below |
| ship15.1 `d905151` | `9dee86f` | 2 | b217, b172 | 854.2 s | **713.4 s** | **1.197×** | 211538 | Done (7th attempt): equal samples (84,869,120), AUC 0.75656, compliance SUCCESS; −1.8% vs MI355X ship14 |
| ship15.1 `d905151` | `9dee86f` | 4 | — | 562.6 s | not measured | — | 213716 | Stopped when moving to ship16 (12 attempts: launch holds / dispatch failures; 1 run cancelled 4 min after `run_start`) |
| ship15 `4401cf2` | `4e5b23d` | 1/2/4 | — | 1279.9 / 827.2 / 550.8 s | not run | — | 209979 | M4 N table is an MI350X-local file; superseded by ship15.1 (M4 removed) |
| ship14 `c68395d` | `b87bc0d` | 1 | b188 | 1397.1 s | **1106.9 s** | **1.262×** | 209246 | Done |
| ship14 `c68395d` | `b87bc0d` | 2 | b186, b080 | 865.7 s | **726.3 s** | **1.192×** | 208875 | Done |
| ship14 `c68395d` | `b87bc0d` | 4 | b174, b150, b217, b022 | 577.6 s | **528.4 s** | **1.093×** | 209193 | Done |

- **MI350X** = the cold, same-seed rows of that ship in `perf-notes.md`.
- **"+ fixes"** = rows 2–4, plus row 7 (GDR flush) at 2n/4n; row 1 is the default since ship14; no tag check (row 5).
- **cluster C rows:** the second MI355X cluster (hosts `c*`, Ubuntu 6.8 kernel, 6.8.0-60 where checked; its own scheduler, so its job IDs are not cluster B's). Same ship17 trainer plus row 8, same recipe, seed and samples to target, and the same "+ fixes" except: row 3 through this cluster's runner (per-file `posix_fadvise` DONTNEED flush verified at 0.00% resident before INIT, no global `drop_caches`; timed warm read after `run_start`); row 4's fabric env with this fabric's RoCE traffic classes (TC 104 / FIFO TC 192), HCAs `rdma0`–`rdma7` and no overlay libs; plus `HSA_USERPTR_FOR_PAGED_MEM=0`. Pre-run check, bus bandwidth at 256 MB: all_to_all 89 GB/s and all_reduce 371 GB/s on the 2n hosts, 57 / 288 GB/s on the 4n hosts.
- **cluster B rows:** hosts `b*` in rank order, from each run's driver logs; "—" where no run reached `run_start`. Kernel 6.8.0-107 maps the sealed memfd `MAP_SHARED` (ship17 1n ran m4 on `377e04d`), so cluster B runs without row 8.
- **ship14:** equal samples to target (66,519,040 / 84,869,120 / 110,100,480), final AUC 0.75310 / 0.75666 / 0.75995, compliance SUCCESS, no 4n corruption; vs MI355X ship13 −7.1% / −8.3% / −5.4%.
- **ship15.1 1n projection:** first eval (fixed sample count) at 886.1 s vs 900.8 s on MI355X ship14 (−1.6%), applied to the ship14 time to target (1106.9 s). Same-build check on ship14: the first-eval ratio vs MI350X (1073.9 s, median of 30 RCP runs) is 1.19×, the time-to-target ratio 1.26×, so the first-eval ratio is a conservative lower bound.
- **Burst QoS:** jobs are cancelled at the partition's `PreemptExemptTime` (1200 s) on the busy cluster; runs are resubmitted and placed on nodes holding the image and staged data. 1n needs ~23 min including setup, so it completes only when the node is not reclaimed (ship14: 3rd attempt).

## Reference traces

Step-52 profiler traces of the cluster C ship17 build (the `e2a61f98` trainer, captured at `a59be9d`), stitched across all ranks like the MI350X ship17 traces that `perf-notes.md` links (`traces/mi350x/<n>_n/trace_step52.json.gz`). They open in Perfetto or `chrome://tracing`.

| Nodes | Trace | Job | Hosts | Ranks | Steps | Events | Size | GPU step (median per rank-step) | MI350X ship17 trace | Speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **[trace_step52.json.gz](../traces/mi355x/1_n/trace_step52.json.gz)** | 9019 | c23 | 8 | 52–56 | 430,636 | 8.9 MB | **97.34 ms** | 112.34 ms | **1.154×** |
| 2 | **[trace_step52.json.gz](../traces/mi355x/2_n/trace_step52.json.gz)** | 9024 | c23, c41 | 16 | 52–56 | 782,169 | 16.1 MB | **100.23 ms** | 117.15 ms | **1.169×** |
| 4 | **[trace_step52.json.gz](../traces/mi355x/4_n/trace_step52.json.gz)** | 9039 | c23, c25, c41, c42 | 32 | 52–56 | 1,622,016 | 33.3 MB | **109.95 ms** | 125.84 ms | **1.144×** |

- **Capture:** the MI350X trace-arm recipe: seed 79275716, batch 1024 per GPU, page cache flushed on every node, no eval (`SKIP_EVAL_EPOCH_PCT=0`, `EVAL_EVERY_DATA_PCT=0`), `START_TS=0`, `NUM_TRAIN_TS=38` (44 at 4 nodes), gin `Profiler.trace_steps = [41, 51]`; the window exported as step 52 is stitched with `tools/recommendation/dlrm_v4/stitch_traces.py --step 52 --gzip`. Run settings as in the cluster C ship17 rows: `DL_FAST_BATCH=1`, `NUM_WORKERS=8`, `HSA_USERPTR_FOR_PAGED_MEM=0`; 2n/4n add `IONIC_LOCKFREE=all`, `HSA_NO_SCRATCH_RECLAIM=1` and the RCCL settings (GDR flush on, `RCCL_MSCCL_ENABLE=0`).
- **GPU step:** `scripts/cluster_b/trace_compare.py` on the two published traces: per rank, the GPU span of each `ProfilerStep` on the main compute stream, median over the 40 / 80 / 160 rank-steps; GPU idle inside a step is 0.1–0.2 ms on both at every scale. It is not the paired median of the MI350X rows (steps 43–45 and 53–55), which reads the MI350X 1n trace as 113.5 ms.
- **By category** (ms per rank-step, MI350X → MI355X): attention 34.93 → 27.39 (1n), 35.08 → 28.51 (2n) and 37.11 → 29.55 (4n), led by `bwd_dkdv_kernel_0` (1.31× / 1.28× / 1.30×). At 1n, communication rises 7.19 → 13.61 ms: the int64 MSCCL all-reduce kernel (`mscclKernel_Sum_int64_t_LL`) shows 8.49 against 2.24 ms. That is kernel time, which includes spinning while ranks arrive and overlaps compute; the step span still drops 1.154×. MSCCL is off at 2n/4n, where communication is 1.16× / 1.14× faster than on MI350X.
- **Node balance (4n):** g42 was excluded from the later RCP lanes for slow eval passes, and g25 is in the ship17 exclusion list after an earlier host-memory exhaustion (both clean when the trace ran), so the trace was checked for a straggler node. Median compute per node (non-collective kernel union per rank-step) is 102.07–103.59 ms, a 1.5% spread against 1.3% in the MI350X 4n trace; g42 is at 103.06 ms. g25 is the slowest, and its ranks spend about 2.5 ms less in collectives than the other nodes'. GPU idle is 0.11–0.15 ms and the step span 109.9–110.0 ms on every node.
- The step-52 speedups match the RCP campaign's speedup at equal samples (`rcp/mi355/a59be9d`: 1.168× / 1.162× / 1.148×). Per-run outputs, including each `compare_vs_mi350x.txt` and the 4n `node_balance.txt`, are in `/scratch/traces355/<job>_<n>n/`; runner `run_trace.sbatch` and `node_balance.py` next to them.

## Previous evaluations

| MI350X base | MI355X commit | Nodes | MI350X | MI355X (no fixes) | MI355X + fixes | Speedup | Job |
|---|---|---|---|---|---|---|---|
| ship13 `18d12b4` | `f6e2a57` | 1 | 1426.5 s | 1470.9 s (0.970×) | **1190.9 s** (rows 1–4, repeat of 1191.0) | **1.198×** | 207533, 207410 |
| ship13 `18d12b4` | `f6e2a57` | 2 | 969.3 s | 993.4 s (0.976×) | **792.0 s** | **1.224×** | 207540 |
| ship13 `18d12b4` | `f6e2a57` | 4 | 660.7 s | crash | **558.5 s** (mean of 3: 558.9 / 570.0 / 546.6) | **1.183×** | 207575, 207630, 207632 |

- **Same samples to target in every run** (66,519,040 / 84,869,120 / 110,100,480) and equal final accuracy.
- **"+ fixes"** = rows 1–4, plus row 5 at 4n.

## Fixes on top of the MI350X commit

| # | Fix | Setting | Type | Root cause | Effect (MI355X) | MLPerf compliant | Evidence |
|---|---|---|---|---|---|---|---|
| 1 | Batched train loader | `DL_FAST_BATCH=1` (upstream builder, bit-identical; already default for eval; **default on in ship14**, item 61) | env | Per-sample builder supplies one batch per ~130 ms/GPU at 6 workers vs MI355X step ~120–140 ms; one late rank (e.g. 426 ms in `DataLoader.__next__`) stalls all ranks in `splits poll`; pre-first-eval steps ~22 ms slower | 1n 1470.9 → 1198.7 s (−18.5%) | Yes — bit-identical batches, same samples/order; checker SUCCESS | 207408 vs 206210  |
| 2 | 8 train loader workers | `NUM_WORKERS=8` | env | With #1 on, supply margin over the GPU step is still thin; 8 workers widen it. Single chained pool (ship12 item 37) primes once, so no window-seam burst | 1n 1198.7 → 1191.0 s; later seams 3.1 → 2.8 s | Yes — system setting, no data/numerics change; checker SUCCESS | 207410  |
| 3 | Cold-cache handling (C1) | Launcher `warm`: verified flush before INIT (per-file `fdatasync` + `fadvise` DONTNEED + privileged `drop_caches`, no global `sync`; `mincore` < 1%), sequential warm read after `run_start` (timed) | launcher | Cold random mmap reads on this storage cap input supply below MI355X demand | Removes early-run input stragglers | Yes — data starts on disk after verified flush; warm read is timed. Disclose in submission | 196xxx–201xxx  |
| 4 | ionic RoCE fabric env | PXN on, TC 96/184, DMABUF, overlay libs | launcher | PXN off → per-NIC incast → `status=12` RETRY_EXC; without DMABUF/overlay RCCL falls back to TCP | Multi-node all-to-all over RDMA | Yes — transport settings only | 2n/4n runs  |
| 5 | 4n collective-tag check | `TORCHREC_VALIDATE_COLLECTIVES=1` | env | Without it, default 4n runs hit `_get_recat` int32 overflow (garbage `batch_size_per_rank` from an inter-node peer), MI355X only. Root cause: GDR flush disabled (row 7). The tag row does not prevent it: 6 of 16 later 4n runs failed the tag check | 4n 3/3 clean, 558.5 s (1.183×); also turns any mismatched exchange into a loud error | Yes — torchrec debug check, no numerics change; checker SUCCESS | 207575, 207630, 207632 |
| 6 | Launch hygiene | Heartbeat before staging; no `--privileged`/`CAP_SYS_ADMIN`; container-local `.pyc` | launcher | Cluster docker policy denials; peers declared a still-staging node dead | Runs launch and stay up | Yes — no cross-run state (fresh container, no reused caches) | 205872+  |
| 7 | GDR flush on (the 4n corruption) | `NCCL_GDR_FLUSH_DISABLE=0` (RCCL default; launcher `conv_multinode.sbatch`, and `fabric_guard.py` in the trainer) | launcher + trainer | The ionic recipe disabled the GDR flush. That recipe is for the ANP net plugin; with RCCL's built-in IB transport a receiver can read a buffer before the NIC's writes are visible. Caught in an eval pass: one inter-node 144 B chunk of the splits all_to_all held fp16 values (±3.2e-4), the previous sequence-embedding all_to_all's bytes; all 32 ranks had issued the same collectives (no desync) | Same 4 nodes, seed and code: flush off failed at about eval 40; flush on 262 late evals clean (2 × 45 min: 130 + 132). run_start to first eval 425.4–425.9 s vs 424.5–426.6 s off (no cost) | Yes — transport setting, no numerics change | runner plans/dlrmv4 09; 208061 (crashrepro_hl1, crashrepro_flush) |
| 8 | memfd window mapping (cluster C kernel) | m4 `window_array()` maps each window buffer `MAP_PRIVATE` instead of `MAP_SHARED` (`e2a61f98`) | trainer | `m4_otf` hands each window's N to the helper as a memfd sealed with `F_SEAL_WRITE` (plus shrink, grow, seal). Linux has allowed a read-only `MAP_SHARED` mapping of a write-sealed memfd since 6.7. Ubuntu 6.8.0-58 took the stable patch "mm: resolve faulty mmap_region() error path behaviour", which refuses it with EPERM, and 6.8.0-60's backport of "mm: reinstate ability to map write-sealed memfd mappings read-only" does not restore it: refused on 6.8.0-60 (cluster C hosts) and 6.8.0-71 (cluster C login node: EPERM with `F_SEAL_WRITE`, allowed with `F_SEAL_FUTURE_WRITE`; `MAP_PRIVATE` allowed). A later 6.8 update allows it again: cluster B's 6.8.0-107 maps it, and so does the MI350X cluster's kernel. Affected: 6.8.0-58 to at least 6.8.0-71 | cluster C: without it ship17 fails at the first m4 window after `run_start` (`PermissionError` in `_rr_costs` → `window_array`) and reaches no eval. With it: 1n 927.9 s (1.177×), 2n 620.0 s (1.165×) | Yes — every window fd is checked for the write seal on receipt, so private and shared mappings read the same bytes: no change to data, m4 decisions or cost; checker SUCCESS | 8894 / 8897 (fail) vs 8907 / 8908; `m4_otf_test` on 6.8.0-71: 6 failures before, none after (5 tests need torch / gin, not run) |

## Tried, not adopted

| Change | Result |
|---|---|
| `NUM_WORKERS=16` (two-pool loader) | 1n slower (2081 s); CPU oversubscribed |
| FP8 out-proj hipBLASLt algo / BF16 pins / full Triton autotune re-sweep | < 1% steady step, early and late windows |
| Fabric knobs (GDR flush, optional recv completion, no LL128, no relaxed ordering) | No effect on 4n corruption; 180k-iteration a2a check clean |
| `DLRM_HSTU_FP8_SYNCFREE_SPLITWAIT_POLL=0` | 4n still crashes |
| Splits a2a `record_stream` guard (`DLRM_HSTU_FP8_SYNCFREE_SPLIT_RECORD_STREAM=1`) | No effect: wait runs on the allocating stream and syncs to host; 1 clean / 1 crash |
| `DLRM_HSTU_FP8_SYNCFREE_DISTINIT_MAIN=0` | 1 clean / 1 crash |
| `DLRM_HSTU_FP8_SYNCFREE=0` (4n) | 3/3 clean but 626.0 s mean (1.055×): ~7% slower than row 5 |
| `DLRM_HSTU_FP8_SYNCFREE_LPK=0` (4n) | Clean, 659.2 s: eval time doubles |

## Notes

| Topic | Note |
|---|---|
| Steady vs e2e | Pre-eval single-rank loader stalls are invisible in 200-step (no-eval) throughput and in 5-step averages; check per-rank `## next_batch ##` |
| Late windows | ~19% more GPU work per step than step 52 (longer histories); step-52 trace overstates the run-level GPU speedup |
| Node variance | Bad / slow / squatted nodes are tracked in `scripts/cluster_b/node_registry.tsv` |
| 4n corruption (root cause: GDR flush off, row 7) | `_get_recat` int32 overflow, garbage `batch_size_per_rank` from an inter-node peer; 5 crashes before row 5. With row 5, 6 of 16 4n runs failed its tag check, all inside an eval pass. "MI355X only" was also "cluster B fabric only": the 69 clean MI350X 4n runs used Ruby's bnxt_re NICs and another recipe. The standalone a2a checks never reproduced it. The earlier "GDR flush: no effect" came from those checks, and "optional recv completion off" used `NET_OPTIONAL_RECV_COMPLETION`, a name RCCL ignores (it reads `NCCL_NET_OPTIONAL_RECV_COMPLETION`) |
| 4n corruption (2026-10-11 analysis, ship13) | All 7 crashes are in the eval pipeline's splits wait, at evals 1/2/5; the garbage values decode as fp16 embedding-like data (±1e-4). Per run, not per eval boundary: shipped config 7/12 runs crash; an eval-every-70-steps run crossed 48 boundaries clean. LPK off 2/2 clean (one run cut at eval 7). ship14 items 66/67 change multi-node input-dist and stream ordering, so the ship14 4n run re-tests it Root cause since found (row 7): the GDR flush was off. The collective that failed in the repro carried no LPK rows (`lpk_rows=0`), so LPK was not the cause |
| ship14 overlap | Item 36 (default-on cold read-ahead walk) runs alongside the launcher `warm` read (row 3); both are after `run_start` |
