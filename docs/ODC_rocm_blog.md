# 把 ODC（On-Demand Communication）搬到 AMD ROCm：用 rocSHMEM/MORI 在 MI300X + Primus 上复现按需通信

> 面向做分布式训练的工程师。本文记录我们把 [sail-sg/odc](https://github.com/sail-sg/odc)（ICLR 2026 论文《On-Demand Communication for FSDP》，[OpenReview PDF](https://openreview.net/pdf?id=iIEEgI6WsF)）从 NVIDIA/NVSHMEM 移植到 AMD ROCm（MI300X, ROCm 7.2）、并在 [Primus](https://github.com/AMD-AGI/Primus) 框架里跑通单机 + 双机的全过程。所有加速比、trace 现象都来自真实实验日志，不达标的地方（小 batch 更慢、双机某些档不如 RCCL）也如实写出来。

---

## 目录

1. [为什么 FSDP 会慢：逐层同步屏障与负载不均衡气泡](#1)
2. [ODC 的核心思想：把集合换成按需 p2p](#2)
3. [在 trace 上看见它：六张图由浅及深](#3)
4. [ROCm 适配要点：rocSHMEM/MORI、XGMI/RDMA 与踩过的坑](#4)
5. [数据说话：单机 1.5B 与双机 14B 的加速比](#5)
6. [结论与展望](#6)

---

<a id="1"></a>

## 1. 为什么 FSDP 会慢：逐层同步屏障与负载不均衡气泡

标准的 FSDP2（PyTorch `fully_shard`）把每一层的参数按 DP 维度切片存放。为了算一层，它必须：

- **前向**：对本层参数做一次 `all_gather`，把完整权重拼回来，算完再 reshard 丢掉；
- **反向**：算出完整梯度后做一次 `reduce_scatter`，把梯度规约并切回本 rank 的分片。

这套集合通信有两个隐藏成本：

1. **逐层同步屏障**。集合通信（all-gather / reduce-scatter）要求 DP 组里**所有 rank 在同一层、同一时刻、以相同的调用次数**一起进入。任何一个 rank 慢了，其它 rank 都得在集合原语里等它。于是每一层都变成一道"通信对齐屏障"。

2. **负载不均衡气泡**。在变长序列的 SFT / 长上下文训练里，不同 rank 拿到的 token 数天然不同（有的样本 6 万 token，有的几百）。可集合通信强制"所有 rank 步调一致"，**轻 rank 只能空等重 rank**——这就是论文 Fig.1/Fig.2 描述的 bubble：GPU 明明有活可以往前算，却被卡在通信屏障上干等。

论文的关键观察是：这种气泡是 **FSDP 通信模式本身**造成的，不是网络带宽不够。要根治，就得把"逐层、逐 microbatch、全体对齐"的集合通信，换成一种**不强制步调一致**的通信方式。

> 值得强调：ODC 论文里的对照基线**不是**一个弱基线。它的对手是**已经开了负载均衡的集合通信版本（Collective + LB-Micro）**——也就是说，先用打包/padding 把各 rank 的负载对齐、再跑标准 RCCL/NCCL collective。ODC 要赢的是这个"武装到牙齿"的集合基线。这一点后文的 `NCCL_pad` 公平基线会严格对齐。

---

<a id="2"></a>

## 2. ODC 的核心思想：把集合换成按需 p2p

ODC 是给 FSDP 打的一个通信替换 patch。它做了三件事（对应官方仓库 README 与论文 Fig.5）：

### 2.1 集合 → 单边 p2p 的两个原语

- **gather（取参数）**：反向/前向需要某层完整权重时，**按需**从持有各分片的 peer 那里"拉"（单边 `getmem`）。谁需要、谁去拉，不需要全组一起调用。
- **scatter-accumulate（推梯度）**：算出梯度后，把每个分片**单边推**（`putmem`）到"拥有该分片的 rank"（相当于一个参数服务器），由对方**异步累加**到它的梯度累加器上。推完就走，不等对方。

这两个原语都是**单边**的：发起方不需要对端"同时也在调同一个集合"。这就打破了集合通信的"次数必须一致"约束。

### 2.2 同步频率：per-iteration → per-minibatch

标准 FSDP 每个 microbatch、每一层都要规约一次梯度。ODC 把**跨 rank 的同步从"每一层/每个 microbatch"降到"每个 minibatch 一次"**：一个 minibatch 内的所有 microbatch 梯度先在本地/参数服务器上累加，直到 minibatch 结束才做一次"结算（settle）"，确保所有梯度都落地，然后优化器再读。

在我们的移植里，这条时间线由 `odc/fsdp/fsdp2.py` 明确编排：

- `pre_minibatch_start()`：清空累加器 + 一次 `dist.barrier()`（让上一步的优化器更新对所有 rank 可见）；
- 反向里每层调用 `ReductionService.scatter_accumulate(...)`（推梯度 + 触发累加）；
- `pre_optimizer_step()` 里调用 `scatter.sync(group)` 做**唯一一次** minibatch 级结算，然后 `update_gradients()` 让优化器读到最终梯度。

### 2.3 与反向重叠 + 末尾一次 settle

因为推梯度是"发射后不管"，理论上它可以和后续的反向计算重叠；只在 minibatch 末尾做一次统一的等待/结算。**这是 ODC 省气泡的关键**——它把"逐层的通信等待"折叠成"minibatch 末的一次等待"。（我们在 ROCm 上这一步的 overlap 还没完全做好，第 3、6 节会诚实展开。）

**原版实现的通信底座**：节点内用 CUDA IPC（把 peer 的显存映射进本进程直接读写），节点间用 NVSHMEM（GPU-initiated 单边 RDMA）。移植到 AMD，就是要把这两层换成 ROCm 上的等价物——这正是第 4 节的主题。

---

<a id="3"></a>

## 3. 在 trace 上看见它：六张图由浅及深

光讲原理不够，我们用 PyTorch Profiler 抓了单机/双机、NCCL/ODC、pad/nopad 的 trace。下面六张图（用户自行插图，此处给语义化图注）由浅及深地展示"ODC 到底改变了什么"。

### 图 1 —— 单机 1.5B，NCCL 基线：逐层同步屏障

> **图注**：单机 8 卡、DeepSeek-R1-Distill-Qwen-1.5B、标准 FSDP2 + RCCL。时间轴上密密麻麻地排着 `nccl:reduce_scatter_base` / `nccl:all_gather_base`，**每一层一道**。这些集合核就是"通信对齐屏障"——所有 rank 必须在此处对齐才能继续。

这是"病症"的基线：通信被切成许多小集合，穿插在每一层的前反向之间，形成规律的锯齿状同步点。

### 图 2 —— 开 ODC 后：逐层集合屏障消失，同步搬到 minibatch 末

> **图注**：同一模型、同一配置，只把通信后端换成 ODC。**逐层的集合核不见了**；反向过程中是连续的 p2p 推送/本地累加，跨 rank 的对齐点被搬到了 minibatch 末尾（一次 `scatter_accumulate_sync`）。原来"每层一道"的通信对齐屏障被整体放松。

这就是 2.2 节所说的"per-iteration → per-minibatch"在 trace 上的直接体现。

### 图 3 —— ODC 反向里的 settle 没能和计算 overlap（当前待优化点）

> **图注**：放大 ODC 的反向。每次都能看到 `FSDP::post_backward_reduce`（即 scatter-accumulate 的结算 / `WAIT_ACC` 等待）**串在关键路径的尾部**，与反向计算的 overlap 很低（实测 ~5%）。也就是说，虽然我们把同步降到了 per-minibatch，但每层的结算等待仍然"露"在计算之外。

这是**诚实要承认的一个短板**：ODC 的理论优势之一是"推梯度与反向重叠"，但我们在 ROCm 上目前 settle 基本没重叠进去（原因见 4.2 的可见性 settle 开销 + 结算的流水未打通）。这也引出了后面 `ODC_SETTLE_DEFER` 的尝试（第 5、6 节）。

### 图 4 —— 对照：原生 NCCL 的反向通信能和计算 overlap

> **图注**：作为对照，看原生 FSDP2 + NCCL 的反向。`reduce_scatter` 的通信核被 FSDP2 的 prefetch 机制**藏在了计算后面**（下一层的通信在当前层计算时就已发起），有实打实的 comm/compute overlap。

图 3 vs 图 4 放一起看，结论很清楚：**在"通信能否被计算掩盖"这件事上，成熟的 RCCL/NCCL + FSDP2 prefetch 目前领先于我们的 ODC 移植**。这不是 ODC 思想的问题，而是我们移植版本的结算流水还没打磨到位——是明确的 roadmap，而非死结。

### 图 5 vs 图 6 —— odc_nopad vs odc_pad：变长微批 vs 补空桶

这两张图解释了 ODC 真正的杀手锏，也是 `nopad` 为什么只有 ODC 能跑。

设定：`global_batch_size=16`、`dp=8`，于是 `16/8 = 2` 个样本/rank。

> **图 5（odc_nopad，LB-Mini 变长微批数）**：2 个样本被打包（KK, Karmarkar-Karp 均衡）成一个 seq≈65536 的微批。由于各 rank 拿到的样本长度不同，**各 rank 的微批数可以不一样**（有的 rank 1 个微批，有的 2 个）。反向里各 rank 按自己的真实负载往前跑，谁也不等谁——**这只有 ODC 的单边 p2p 能驱动**：集合通信要求各 rank 调用次数一致，微批数不等会直接死锁。

> **图 6（odc_pad，LB-Micro `same_num_in_dp`）**：为了让所有 rank 微批数相同，负载重的 rank（样本更长、超过单微批 token 上限）会拆出更多微批，其它 rank 就得**补齐到相同微批数**——补的是"空桶"（padding 微批）。trace 上能看到这些空桶微批在做无用功。

**odc_nopad 的优势 = 免掉补空桶**。而"变长微批数"这件事本身，**只有 ODC 的 p2p 能做**（`odc/primitives/scatter_accumulate.py` 的 DEFER 注释里写得很清楚：nopad 下各 rank 微批数不同，逐微批调集合会 barrier 次数不匹配 → 跨节点死锁）。这也是为什么在集合基线里，要跑变长数据就必须 pad（`LB_MINI_SAME_MICRO=1`），从而付出 padding 的算力浪费。

---

<a id="4"></a>

## 4. ROCm 适配要点：rocSHMEM/MORI、XGMI/RDMA 与踩过的坑

这一节是移植的核心。原则是：**尽量不改上游算法层的代码，把平台差异收敛到通信原语后端**，并用 `_IS_ROCM = torch.version.hip is not None` 做分支，NVIDIA 路径保持原样，方便未来跟 ODC 上游合并。

### 4.1 通信后端：NVSHMEM → MORI / rocSHMEM

| 层级 | 原版（NVIDIA） | 移植版（ROCm/MI300X） |
|---|---|---|
| 节点内 | CUDA IPC（peer 显存映射直读写） | **XGMI + HIP IPC**：`SymmBufferRegistry` 用对端 peer-view，`peer_buf.copy_` 直接 XGMI 拷贝；`csrc/tensor_ipc` 扩展（PyTorch `hipify` 自动转，**0 行手改**）拿 IPC handle |
| 节点内累加 | watcher 子进程 | **watcher 子进程 + IPC 握手**：每 rank spawn 一个 `ReductionWatcher`，通过 request/response 对称 buffer（int32）做"请求累加/回执"握手，owner 侧 `acc.add_(buf)` 把 peer 推来的分片累加进 fp32 累加器 |
| 节点间 | NVSHMEM（GPU-initiated RDMA） | 两条可选后端：**MORI**（默认，GPU-initiated，OpenSHMEM 风格 host API + Triton IR，`pip install amd_mori` 即用）；**rocSHMEM host-API**，其中跨节点又分 **RO（host 驱动，CPU 发起 `putmem/getmem/int_p`）** 与 **GDA（device-kernel `getmem`，片上 fp32 pull-reduce 累加）** |

Gather 的实现直接体现这套分层（`gather.py`）：节点内 peer 用 XGMI `copy_`，跨节点 peer 用 `getmem`（RO 走 host、GDA 走 device kernel）拉到对称 buffer 再重组。Scatter-accumulate 同理：节点内 `peer_buf.copy_` + watcher 握手，跨节点用单边 `putmem` + 请求信号（RO）或 device pull-reduce（GDA）。

几个务实的选型决策：

- **用 MORI 替代 NVSHMEM**：AMD 官方维护、PyPI 有 wheel、API 与 NVSHMEM 基本 1:1，硬件矩阵明确覆盖 gfx942（MI300X）。
- **单节点 int_p/int_g 走 MORI `ptr_p2p` + `tl.store/load`，不走 `int32_p`**：实测 `int32_p` 在单节点会强制走 RDMA 路径、没配 NIC 时挂死；`ptr_p2p` 拿到 peer 本地映射后直接 `tl.store`，等价 NVSHMEM 的节点内 IPC 写法。
- **`tensor_ipc` C++ 扩展一行不改**：PyTorch 的 `hipify` 自动把 `cudaIpcGetMemHandle → hipIpcGetMemHandle` 等符号映射过去。

### 4.2 移植踩的坑与修法（都是实测出来的）

**坑 1：MI300X 的 L2 陈旧性 → 用 system-scope 原子自旋。**
上游 same-node 结算核用的是 `while r != next_request_id: quiet(); int_g(...)` 这种"在 kernel 里反复 volatile 读 peer 地址"的自旋。在 MI300X 上它**永远读不到 watcher 写的回执**——GPU L2 里的值是陈旧的。修法是把等待换成 `int_wait_until_equals`（`scatter_accumulate.py` 的 `nvshmem_wait_accumulation_same_node_kernel`），它用**绕过 L2 的 system-scope 原子读**；NVIDIA 路径则回退到原来的 `quiet + int_g` 自旋（那边有 acquire 语义，本来就对）。

**坑 2：跨节点写可见性 → HDP flush / strided-touch settle。**
这是双机 GDA 最硬的一个坑。reduce-scatter 暂存的是**刚写入**的梯度，GPU 的写可能还在 HDP（Host Data Path）缓存里，远端 NIC 的 RDMA 读会读到**旧值** → 产生一个巨大的伪梯度尖峰（grad spike）。我们试了几种 settle 策略（`ODC_GDA_WARMUP_MODE`）：

- `full`：在真正的 reduce-scatter 前，**整份 shard** 做一遍 throwaway"读一遍触发可见"。最稳（确定性 0 尖峰、loss=单机），但**warm-up 占了 reduce-scatter 时间的 ~59%**（几乎把 RS 做了两遍）。
- `hdp`：用 HSA 取到本卡的 `HDP_MEM_FLUSH_CNTL` MMIO 寄存器，一次 O(1) 写就整卡冲刷 HDP，省掉整份 warm-up（~11% 更快）。但实测**寄存器写不能确定性消除竞态**（50 轮里 nopad 出现 6 次间歇尖峰），只作 opt-in 提速；`__threadfence_system`（`fence` 模式）更弱，尖峰立刻复发（反例）。
- **`strided`（默认，既快又稳）**：只对每个 64KB 页读 1 个元素 × 覆盖所有 PE——既触发了所有页 / 全 8 张 NIC 的读路径（保留"读触发 settle"的确定性），又把读取体积从整份 shard 降到极小。实测 **nopad/pad 双双 0 尖峰、nopad loss 严格对齐单机、比 full 快 ~9–10%**。步长 `ODC_GDA_STRIDE_BYTES` 默认 65536（4KB 也 0 尖峰但触碰太多≈full 速度）。

**坑 3：TE / MORI 加载顺序 → sitecustomize 早注入。**
在 MI300X + ROCm 7.2 上，**先 import `transformer_engine` 再初始化 MORI** 会让 MORI 的 C++ 运行时（甚至只是分配对称堆）以 `free(): invalid pointer` 崩溃——是动态库 load-order / 全局构造函数冲突。修法：在 `odc_early/sitecustomize.py` 里、由 Python 的 site 机制在**解释器启动最早期**（早于任何 Primus/Megatron/TE import）就 `import mori`，把加载顺序钉死。它由 `ODC_ENABLE` 门控，非 ODC run 完全 no-op。

**坑 4：变长微批 → 用 all_gather_object 的"计数栅栏"代替"次数栅栏"。**
nopad 下各 rank 微批数不同，如果 minibatch 结束时按"调用次数"对齐就会死锁。`ReductionService.sync()` 改成用 `torch.distributed.all_gather_object` 收集各 rank 的 `dispatched_tasks` 总数，让 watcher 等到**全局任务计数**达标（`wait_and_reset_task_count`），而不是等某个固定的调用次数——这样各 rank 微批数不同也能正确 rendezvous。GDA 路径则改成"本地累加 + 每 minibatch 一次带 barrier 的 reduce-scatter（`ODC_GDA_DEFER_REDUCE`）"，barrier 次数 = #groups，跨 rank 天然一致，避免死锁。

**双机移植另外修的 3 个真实工程 bug：**

1. **mpirun 直启分支**：双机用 `mpirun` 拉起 16 rank（8/节点），需要走 `PRIMUS_LAUNCHER=mpi` 分支，把 `OMPI_COMM_WORLD_*` 正确映射成 torch.distributed 的 rank/world/master 环境；直接沿用 torchrun 的假设会拿错 rank。
2. **Emerging-Optimizers 并发装**：多 rank 同时首次触发某优化器扩展的按需编译/安装会互相打架，需要串行化或预装。
3. **NFS 文件锁 ESTALE**：打包缓存的 filelock 若放在 NFS 上，多节点抢锁会 `Stale file handle (ESTALE)`。修法是把锁目录放到**本地 `/tmp`**（`PRIMUS_PACK_LOCK_DIR=/tmp/primus_lock`）、缓存本身仍在共享盘——日志里能看到各 rank 在 `/tmp/primus_lock/sft_pack_*.lock` 上有序抢锁、无 ESTALE。

### 4.3 LB-Mini：变长负载均衡怎么接进 Megatron

LB-Mini 是"变长 token 打包 + Karmarkar-Karp 跨 DP 均衡 + 每 rank 各自的微批数"的数据层，全部以 Primus 层的 monkey-patch 实现（不改第三方 Megatron 源码，见 `odc_lb_mini_patches.py`）：

- **一个开关，两条路径**：`enable_odc_lb_mini=false`（默认）时整个 patch 是 no-op，Megatron 跑它原生的定长、全 rank lockstep 调度，字节级不变。
- `enable_odc_lb_mini=true` **且 `ODC_ENABLE=1`** 时，数据按变长服务、KK 均衡到各 DP rank，**每 rank 跑自己那份（可能不同的）微批数**（patch `forward_backward_no_pipelining` 用 rank-local 的 `num_microbatches`）。注释里点明：只有 ODC 的 p2p 能驱动各 rank 出 lockstep 而不集合死锁，所以强制要求 `ODC_ENABLE=1`。
- **A/B 例外**：`LB_MINI_FORCE_DATA=1` 让变长数据层在 NCCL 下也能跑（但**只在 round-robin/pad 保证各 rank 微批数相等时安全**）——这正是我们做公平基线 `NCCL_pad`（同样的变长数据、只换通信）的钥匙。

---

<a id="5"></a>

## 5. 数据说话：单机 1.5B 与双机 14B 的加速比

> 口径说明：加速比 = `NCCL_pad 的 ms/步 ÷ 本 run 的 ms/步`（>1 表示比 NCCL 快）。基线是**开了打包/pad 的标准 RCCL collective**（`NCCL_pad`），即论文意义上的"武装到牙齿的集合基线"，不是弱基线。数字取自各自实验日志真实值。

### 5.1 单机 1.5B（d53，8 GPU，500 轮，总时间口径）

| gbs | ODC_nopad 加速比 | 趋势解读 |
|---|---|---|
| 8 | ≈ **0.95×**（略慢） | minibatch 太小，p2p 的固定开销摊不掉，反而略慢于 RCCL |
| 16 | ≈ **1.20×（峰值）** | KK 变长均衡收益最大：既省了补空桶、通信又走 XGMI 按需 p2p |
| 32 | ≈ **1.10×** | 仍有优势，但 compute 变大、通信占比下降 |
| 64 | ≈ **1.0×** | 与 RCCL 基本持平 |
| 128 | ≈ **1.0×**（约 0.99×） | 计算已完全主导，ODC 的固定红利被摊薄，与 RCCL 收敛 |

**小结（单机）**：ODC 的优势呈"先升后回落"——**在中等 batch（gbs16–32）见顶，gbs16 的 odc_nopad 达 ~1.20×**；小 batch 因固定开销摊不掉而略慢（0.95×），大 batch 因 compute 主导而与 RCCL 趋同。

拆解单机为什么会快（同节点 apples-to-apples 的 `NCCL_pad → ODC_pad → ODC_nopad` 阶梯，另一组 50 轮实验）：`NCCL_pad` → `ODC_pad` 先快 ~**9.8%**（按需 p2p 走节点内 XGMI 比 RCCL collective 省），再到 `ODC_nopad` 又快 ~**8.2%**（免掉补空桶的负载均衡红利），合计 ~**18%**。即单机的收益 = **XGMI 按需通信 + 变长负载均衡**两块叠加。

### 5.2 双机 14B（d51，16 GPU，100 轮；另有干净仓库 50 轮交叉印证）

| gbs | ODC 加速比 | 趋势解读 |
|---|---|---|
| 16 | ≈ **0.78–0.83×**（明显慢） | 小 gbs，每步的跨节点 p2p 固定开销未被摊薄 |
| 32 | ≈ 0.92×（nopad）/ 0.87×（pad） | 收窄，但仍不及 RCCL |
| 64 | ≈ **1.16×（odc_nopad 反超）** | 大 batch 摊薄了跨节点异步规约开销，ODC_nopad 首次胜出 |
| 128 | ≈ **1.18×** | gbs 越大、跨节点集合越贵，ODC 的摊销收益越大 |

**小结（双机）**：与单机相反，双机 ODC 的加速比随 gbs **单调上升**——小 gbs 明显吃亏（跨节点 RDMA + barrier 是每步固定开销），到 **gbs64 起 odc_nopad 反超 RCCL（1.16×），gbs128 到 ~1.18×**。方向与论文 Fig.10 完全一致：**ODC 的收益随设备数/异质性增长**（跨节点越贵、负载越不均衡，ODC 越划算）。

### 5.3 迁移到干净 upstream Primus 分支：行为一致、无功能回归

我们把整套改动从最初的开发仓库迁到一个干净的 upstream Primus 分支重跑。新仓库**复现了旧仓库的加速比曲线**：

| 场景 | 旧仓库 | 新仓库 | 结论 |
|---|---|---|---|
| 单机 gbs16 nopad | 1.20× | 1.21× | 一致 |
| 双机 gbs16 nopad | 0.78× | 0.80× | 一致 |

lr 对齐 d51 的 qwen14B 续训设置（`2e-6`）。两处曲线几乎重合，说明**移植行为一致、无功能回归**。

### 5.4 trace 级证据：双机慢在哪、pad 为什么补气泡

把四份 trace（单机/双机 × pad/nopad）按 GPU kernel 聚合后：

- **跨节点通信核**：双机 ~1063ms vs 单机 ~365ms（**约 3×**，占 GPU-kernel 时间 27% vs 9%）——就是 `reduce_scatter`(pull) + `all_gather` 的 RDMA。
- **GPU 空闲/同步等待**：双机 nopad 58.7% vs 单机 51.5%（跨节点 barrier 把 GPU 闲在那等）。
- **pad vs nopad（关键假说验证）**：两者的 comm 通信量几乎一样（1063 vs 1021ms），但 **nopad 的真 GPU 空闲 58.7% vs pad 52.2%**（+6.5pp）。即 **nopad 多出来的是"同步等待气泡"而非通信量**——nopad 各 rank token 数不等，到跨节点同步点时快 rank 干等慢 rank，气泡更大；pad 把微批数对齐后气泡减小。单机 nopad/pad 空闲几乎相同（无跨节点同步点），反证该气泡是**跨节点特有**。

### 5.5 诚实标注（不夸大，反而更可信）

- **小 batch ODC 会更慢**：单机 gbs8 ~0.95×、双机 gbs16 ~0.78–0.83×。ODC 不是处处更快，它的红利需要"大 batch / 跨节点 / 有负载不均衡"来兑现。
- **双机某些档不如 RCCL**：在同节点公平基线下，双机 gbs32 附近我们的 GDA 跨节点 p2p 比优化过的 RCCL collective 慢 ~12–20%——这是本 ROCm GDA 实现的跨节点固定开销（warm-up settle / barrier / 逐组 getmem）尚不如 RCCL 成熟所致，也正是第 6 节的待优化项。
- **大批量偶发梯度尖峰**：gbs128 大批量下偶发梯度尖峰，靠重试 / 梯度裁剪压住；个别 `ODC_pad` run 的 loss 会发散（但 perf 数据仍可用，已在表里标注）。
- **`ODC_SETTLE_DEFER` 原型**：我们试过把 settle 移出反向关键路径（用一个 minibatch 末的聚合 join 代替逐层 `wait_stream`）。**验证正确**（不 hang、loss 与基线一致），但**性能无明显收益**——因为在当前实现里 settle 本身占比就小，把它挪走省不下多少（trace 佐证：settle 核随 gbs 增长，但相对 compute 仍是小头）。这条路目前收益有限，留作后续与 overlap 一起做。

---

<a id="6"></a>

## 6. 结论与展望

**结论**：

1. **移植跑通**：ODC 的算法层（gather / scatter-accumulate、per-minibatch 同步、LB-Mini 变长均衡）已在 AMD ROCm / MI300X 上用 **MORI（默认）/ rocSHMEM** 后端跑通，**单机 + 双机**都能正确收敛（loss 与单机基线对齐、0 nan）。NVIDIA 代码路径保留为 fallback，便于跟上游合并。
2. **收益场景清晰**：ODC 在**"大 batch / 跨节点 / 有负载不均衡"**时收益最大——单机 gbs16 峰值 ~1.20×，双机 gbs64 起反超 RCCL、gbs128 达 ~1.18×。这与论文"ODC 从根本上减少 FSDP 的负载不均衡气泡、且加速比随设备数增长"的结论一致。
3. **不夸大**：小 batch 下 ODC 会略慢，双机中等 batch 我们的 GDA p2p 目前不如 RCCL collective——这些都如实标注。

**展望（待优化点，按性价比）**：

- **让 settle 与反向 overlap（首要）**：当前 scatter-accumulate 的结算基本"露"在关键路径上（图 3），而原生 NCCL 能靠 prefetch 把通信藏进计算（图 4）。做一条跨迭代的软流水（独立 stream + event，把上一组的 reduce-scatter 与本组反向重叠），是最对齐论文思想、也最可能追平/反超 RCCL 的方向。
- **削 / 合并跨节点固定开销**：warm-up settle 已从 full 优化到 strided（省 ~9–10%）；进一步可把每步 31 个小集合分桶合并、减少 barrier 次数。
- **弱扩展放大 batch**：跨节点固定开销是每步固定的，增大 global batch 能把它摊薄——这也是双机 gbs64/128 反超的根本原因。

一句话：**ODC 的按需 p2p 让"变长负载均衡训练"在 ROCm 上既能正确跑、又能在大 batch/跨节点场景真正省下气泡**；剩下的主要工程债是把结算的通信/计算 overlap 打磨到 RCCL 的成熟度。

---

### 参考

- 论文：《On-Demand Communication for FSDP》(ICLR 2026) — [OpenReview PDF](https://openreview.net/pdf?id=iIEEgI6WsF)（动机 Fig.1/2、方法 Fig.5、Collective LB-Micro 基线、加速比随设备数增长 Fig.10）
- 官方仓库：[sail-sg/odc](https://github.com/sail-sg/odc)（FSDP patch、gather + scatter-accumulate 原语、CUDA IPC + NVSHMEM 底座）
- 底座：[ROCm/mori](https://github.com/ROCm/mori)（MORI-SHMEM / MORI-IR，替代 NVSHMEM）、[Primus](https://github.com/AMD-AGI/Primus)（训练框架）
- 移植源码：`odc_rocm_dev/odc/primitives/{gather,scatter_accumulate,_rocshmem_backend,nvshmem_triton,utils}.py`、`odc/fsdp/fsdp2.py`、`odc_early/sitecustomize.py`、`primus/backends/megatron/patches/odc_{lb_mini,torch_fsdp2}_patches.py`
