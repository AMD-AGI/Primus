# 使用ODC来加速amd sft训练

> 面向从事分布式训练的工程师。本文记录我们把 [sail-sg/odc](https://github.com/sail-sg/odc)（ICLR 2026 论文《On-Demand Communication for FSDP》，[OpenReview PDF](https://openreview.net/pdf?id=iIEEgI6WsF)）从 NVIDIA/NVSHMEM 移植到 AMD ROCm（MI300X，ROCm 7.2），并在 [Primus](https://github.com/AMD-AGI/Primus) 框架中跑通单机与双机的完整过程。文中所有加速比与 trace 现象均取自真实实验日志；不达标之处（小 batch 更慢、双机部分档位不及 RCCL）也如实呈现，不作粉饰。

---

## 目录

1. [为什么 FSDP 会慢：逐层同步屏障与负载不均衡气泡](#1)
2. [ODC 的核心思想：把集合换成按需 p2p](#2)
3. [在 trace 上看见它：由浅及深](#3)
4. [数据说话：单机 1.5B 与双机 14B 的加速比](#4)
5. [结论与展望](#5)

---

<a id="1"></a>

## 1. 为什么 FSDP 会慢：逐层同步屏障与负载不均衡气泡

标准的 FSDP2（PyTorch `fully_shard`）把每一层的参数按 DP 维度切片存放。为了计算其中一层，它必须：

- **前向**：对本层参数做一次 `all_gather`，把完整权重拼回来，算完立即 reshard 释放；
- **反向**：算出完整梯度后做一次 `reduce_scatter`，把梯度规约并切回本 rank 的分片。

这套集合通信有两笔隐藏成本：

1. **逐层同步屏障**。集合通信（all-gather / reduce-scatter）要求 DP 组里**所有 rank 在同一层、同一时刻、以相同的调用次数**一起进入。只要有一个 rank 变慢，其他 rank 就都得卡在集合原语里等它。于是每一层都成了一道"通信对齐屏障"。**这道屏障是实打实的性能浪费**：每算完一层就得停下来等全组对齐，通信被钉死在关键路径上、无法与计算重叠，于是大量 GPU 周期不是花在"算数据"上，而是白白耗在"等通信、等别人"上——层数越多，这类等待浪费越大。

2. **负载不均衡气泡**。在变长序列的 SFT / 长上下文训练中，不同 rank 拿到的 token 数天然不同（有的样本 6 万 token，有的只有几百）。但集合通信强制"所有 rank 步调一致"，**轻 rank 只能空等重 rank**——这正是论文 Fig.1/Fig.2 所描述的 bubble：GPU 明明有活可以往前算，却被卡在通信屏障上干等。**更麻烦的是，纯集合通信对此几乎无解**：为了让各 rank 步调一致、能安全地逐微批调集合，唯一的办法就是用 padding 把负载轻的 rank"补空桶"填平到和最重的 rank 一样多的微批数（即后文第 3 节要对比的 `pad` 路线）。这等于**用无效算力换取步调对齐**——轻 rank 的算力被白白浪费在空桶的无用前反向上，浪费的量正比于各 rank 负载的不均衡程度。

论文的关键观察是：这两类浪费源自 **FSDP 通信模式本身**，而非网络带宽不足。要根治，就必须把"逐层、逐 microbatch、全体对齐"的集合通信，换成一种**不强制步调一致**的通信方式。

借用官方仓库对 ODC 的一句话定义：*"ODC is a patch to FSDP that adapts Parameter Server (PS) into FSDP by replacing collective all-gather and reduce-scatter with on-demand point-to-point communication."*（ODC 是给 FSDP 打的补丁，用**按需点对点通信**替换集合式的 all-gather / reduce-scatter，把参数服务器（PS）的思路融入 FSDP）。它带来的直接效果是把**同步频率从 per-iteration 降到 per-minibatch**，从根本上压掉 FSDP 的负载不均衡气泡。

下面这张来自[论文](https://openreview.net/pdf?id=iIEEgI6WsF)/[官方仓库](https://github.com/sail-sg/odc)的示意图。

![图 7：论文示意（Figure 1 + Figure 2）——上半是集合通信在 FSDP 里引入的逐层同步屏障（Device 0/1 每层都要对齐，灰色即 Device Idle 气泡）；下半是 ODC 把这些屏障放松到 minibatch 末，只在末尾做一次结算（OS＝Optimizer Step），从而省出右侧标注的“Time Saved”。](odc_blog/fig7_paper_barriers.png)

- **上半（Figure 1）**：在标准 FSDP 中，集合 all-gather/reduce-scatter 让两张卡在**每一层**都要停下来对齐（逐层同步屏障），任一张卡变慢就拖着整体一起等——灰块就是被浪费掉的 GPU 空闲。
- **下半（Figure 2）**：ODC 把逐层对齐**放松到 minibatch 末**，中间各卡按自己的节奏连续做前反向，直到末尾才统一结算一次，于是右侧多出一段 **Time Saved**。这正是我们要在 ROCm 上复现的目标。

> 值得强调：ODC 论文的对照基线**并非**弱基线。它的对手是**已经开启负载均衡的集合通信版本（Collective + LB-Micro）**——即先用打包/padding 把各 rank 的负载对齐，再跑标准 RCCL/NCCL collective。ODC 要战胜的正是这个"武装到牙齿"的集合基线；这一点，后文的 `NCCL_pad` 公平基线会严格对齐。

---

<a id="2"></a>

## 2. ODC 的核心思想：把集合换成按需 p2p

ODC 是给 FSDP 打的一个通信替换 patch。它主要做了三件事（对应官方仓库 README 与论文 Fig.5）：

### 2.1 集合通信原语 → 单边 p2p 的两个原语

- **gather（取参数）**：前向/反向需要某层完整权重时，**按需**从持有各分片的 peer 那里"拉"回来（单边 `getmem`）。谁需要、谁去拉，无需全组一起调用。
- **scatter-accumulate（推梯度）**：算出梯度后，把每个分片**单边推送**（`putmem`）给"拥有该分片的 rank"（相当于一个参数服务器），由对方**异步累加**到它的梯度累加器上。推完即走，不等对方。

这两个原语都是**单边**的：发起方并不要求对端"同时也在调用同一个集合"，从而打破了集合通信"调用次数必须一致"的硬约束。

![图 8：论文 Figure 5——ODC 的两个单边原语。左（gather，to device 0）：device 0 需要完整参数时，把散在各设备的分片（Param0/Param1）按需拉到本地拼齐；右（scatter-accumulate，from device 0）：device 0 算出的各分片梯度（Grad0/Grad1）被单边推到“拥有该分片的 owner”，由对方累加进梯度累加器（Acc）。出处：论文 https://openreview.net/pdf?id=iIEEgI6WsF ，官方仓库 https://github.com/sail-sg/odc 。](odc_blog/fig8_paper_primitives.png)

对照上图：**gather** 是"缺哪片就去哪台拉哪片"，**scatter-accumulate** 是"算完哪片就推给对应 owner 去累加"。两者都由需要方单边发起、发射后无需等待对端进入同一次集合——这正是 ODC 能容忍"各 rank 步调不一致"的根本原因。官方仓库也点明：其通信底座在节点内用 CUDA IPC、节点间用 NVSHMEM 来实现这套 RDMA 单边原语；我们在 ROCm 上则把它替换为 XGMI/HIP-IPC + rocSHMEM/MORI 的等价物。

### 2.2 同步频率：per-iteration → per-minibatch

标准 FSDP 每个 microbatch、每一层都要规约一次梯度。ODC 则把**跨 rank 的同步从"每一层 / 每个 microbatch"降到"每个 minibatch 一次"**：一个 minibatch 内所有 microbatch 的梯度先在本地/参数服务器上累加，直到 minibatch 结束才做一次"结算（settle）"，确保所有梯度都已落地，优化器再读取。

在我们的移植中，这条时间线由 `odc/fsdp/fsdp2.py` 明确编排：

- `pre_minibatch_start()`：清空累加器，并做一次 `dist.barrier()`（让上一步的优化器更新对所有 rank 可见）；
- 反向过程中每层调用 `ReductionService.scatter_accumulate(...)`（推梯度并触发累加）；
- `pre_optimizer_step()` 中调用 `scatter.sync(group)` 做**唯一一次** minibatch 级结算，随后由 `update_gradients()` 让优化器读到最终梯度。

### 2.3 与反向重叠 + 末尾一次 settle

因为推梯度是"发射后不管"，理论上它可以和后续的反向计算重叠，只在 minibatch 末尾做一次统一的等待/结算。**这正是 ODC 省下气泡的关键**——它把"逐层的通信等待"折叠成"minibatch 末的一次等待"。

**原版实现的通信底座**：节点内用 CUDA IPC（把 peer 的显存映射进本进程后直接读写），节点间用 NVSHMEM（GPU-initiated 单边 RDMA）。移植到 AMD，就是要把这两层替换为 ROCm 上的等价物：节点内走 **XGMI + HIP IPC**（peer 显存直接读写），节点间走 **rocSHMEM或MORI（默认，GPU-initiated RDMA）**。

---

<a id="3"></a>

## 3. 在 trace 上看见它：由浅及深

光讲原理还不够。我们用 PyTorch Profiler 抓取了单机/双机、NCCL/ODC、pad/nopad 各组合的真实 trace，下面这组实测截图由浅入深地展示"ODC 到底改变了什么"：先看逐层集合屏障如何消失（图 1、图 2），再看变长微批相对补空桶的形态差异（图 5、图 6）。

### 图 1 —— 单机 1.5B，NCCL 基线：逐层同步屏障

![图1：单机1.5B，NCCL基线——逐层集合同步屏障](odc_blog/fig1_nccl_perlayer_barriers.png)

**图注**：单机 8 卡、DeepSeek-R1-Distill-Qwen-1.5B、标准 FSDP2 + RCCL。多个 rank（`[R0]/[R1]/[R2]`）的 `stream 14` 上密密麻麻地铺满 `nccl:reduce_scatter_base` + `ncclDevKernel_Generic_2`，**每一层一道**；`stream 0` 则是计算（`void ck_tile`）。这些集合核就是"通信对齐屏障"——所有 rank 必须在此处对齐，才能继续往下算。

这就是"病症"的基线：通信被切成许多小集合，穿插在每一层的前反向之间，形成规律的锯齿状同步点。

### 图 2 —— 开 ODC 后：逐层集合屏障消失，同步搬到 minibatch 末

![图2：单机1.5B，开启ODC——逐层集合核消失、同步搬到 minibatch 末](odc_blog/fig2_odc_no_barriers.png)

**图注**：同一模型、同一配置，仅把通信后端换成 ODC。`stream 0` 是连续的众多小微批计算（`void ck_tile`），`stream 11` 是连续的绿色 F（前向），而 `stream 16–23` 上**再也看不到逐层的集合核**——反向过程中只剩连续的 p2p 推送 / 本地累加，跨 rank 的对齐点被整体搬到了 minibatch 末尾（一次 `scatter_accumulate_sync`）。原来"每层一道"的通信对齐屏障被彻底放松，逐层隐式同步不复存在。

图 1 与图 2 的对比，正是 2.2 节"per-iteration → per-minibatch"在 trace 上最直接的体现：屏障从"每层一道"塌缩成"每 minibatch 一次"。

### 图 5 vs 图 6 —— odc_nopad vs odc_pad：变长微批 vs 补空桶

这两张图揭示了 ODC 真正的杀手锏，也说明了为什么 `nopad` 只有 ODC 能跑。**先把两张图的归属讲清楚：图 5 是"没有补空桶"的 odc_nopad，图 6 是"有补空桶"的 odc_pad。**

**先说两张图共同的实验设定，以及它为何关键。** 两张 trace 都取自同一档配置：`global_batch_size=16`、`dp=8`，于是每个 DP rank 平均只分到 `16/8 = 2` 个样本；这些变长样本经 KK（Karmarkar-Karp）均衡跨 rank 分配后，在本 rank 内打包成 seq≈65536（64K token）的微批。**由图中可以看出**，每 rank 只有 2 个样本，"样本长度的天然抖动"几乎无法在 rank 内被平均掉——某个 rank 可能两条都是长文（要拆成 2 个 64K 微批），另一个 rank 可能两条都短（1 个微批就装下）。于是"各 rank 微批数不一致"在这一档被放大到最明显，正好用来对比 pad / nopad 两条路线：**pad 强行把大家对齐成相同微批数（代价是空桶算力），nopad 则允许各跑各的（免掉空桶，但对通信语义提出了更苛刻的要求）。** 这也是 ODC 价值最集中体现的一档（对应第 4 节单机 gbs16 的峰值加速比）。

![图5：odc_nopad——各 rank 按真实变长负载跑、微批数可不同、免补空桶，trace 呈两段密集 burst](odc_blog/fig6_odc_nopad.png)

**图 5（odc_nopad，LB-Mini 变长微批数）——这才是 ODC 想要的形态。** 因为允许各 rank 微批数不同，数据层（LB-Mini）按每 rank 的**真实变长负载**排布：有的 rank 分到的两条样本合起来要 2 个 64K 微批，有的只要 1 个。反映到 trace 上，**快 rank 跑完自己的微批无需空等慢 rank，慢 rank 也不必被拖着走**；中间那段"看似空白"，其实是各 rank 微批数不同、在时间轴上自然错开所致，而非集合屏障造成的干等。**关键在于：这种"各 rank 微批数可以不等"的形态，只有 ODC 的单边 p2p 才驱动得起来。** 集合通信（all-gather / reduce-scatter）要求 DP 组里每个 rank 以**完全相同的次数**进入同一个原语——一旦某个 rank 少调一次（微批数不同时必然发生），跨节点集合就会因 barrier 次数不匹配而直接 hang 死。而 ODC 的 `getmem`/`putmem` 是单边的，谁需要谁发起、发射后不管，天生不受"调用次数必须一致"的约束，这才让 nopad 成为可能（实现细节见 `odc/primitives/scatter_accumulate.py` ）。

![图6：odc_pad——为对齐各 rank 微批数而补“空桶”，trace 呈单个密集 burst](odc_blog/fig5_odc_pad.png)

**图 6（odc_pad，LB-Micro `same_num_in_dp`）——为兼容集合语义而付出的代价。** pad 路线要求**所有 rank 的微批数强制相同**：负载最重的那个 rank 决定了"每 rank 要跑几个微批"，其他较轻的 rank 必须**补齐到同样的微批数**——补上的正是**"空桶"（padding 微批）**，里面塞满无意义的 padding token，纯粹为了让各 rank 步调对齐、以便安全地逐微批调集合。反映到 trace 上，**多了一个空桶做无用功**换来的——那些 padding 微批照样消耗前向/反向的算力和显存带宽，却不贡献任何有效梯度。这也解释了为什么在纯集合基线里，只要想跑变长数据就**必须** pad（`LB_MINI_SAME_MICRO=1` / `NCCL_pad`）：集合通信别无选择，只能用空桶把不齐的负载"填平"。

**两图对读，结论十分直接：** odc_nopad（图 5）与 odc_pad（图 6）的**有效计算量完全相同**，差别在于 pad 多做了一批空桶。nopad 把这批空桶省掉——省下的量正比于"各 rank 负载的不均衡程度"，在 gbs 小（每 rank 样本少、抖动大）、序列变长明显的场景里最为可观。而"变长微批数、免补空桶"这条路，**只有 ODC 的单边 p2p 走得通**：集合基线为了不死锁只能 pad，这就造成只能浪费补充的 padding 算力。这就是 ODC 在"负载不均衡"场景下的结构性红利，也是第 4 节 gbs16 峰值加速比里"变长负载均衡"那部分收益的来源。

---

<a id="4"></a>

## 4. 实验结果数据说话：单机 1.5B 与双机 14B 的加速比

> 口径说明：加速比 = `NCCL_pad 的 ms/步 ÷ 本 run 的 ms/步`（>1 表示比 NCCL 快）。基线是**开启打包/pad 的标准 RCCL collective**（`NCCL_pad`），即论文意义上"武装到牙齿的集合基线"，而非弱基线。所有数字均取自各自实验日志的真实值；三条路线的 loss 收敛曲线与 NCCL 基线对齐、全程 0 nan。

### 4.1 单机 1.5B（8 GPU，device 路径，总时间口径）

模型 DeepSeek-R1-Distill-Qwen-1.5B、单机 8 卡、节点内 XGMI + HIP IPC。下表按 gbs 给出 `ODC_nopad` 相对 `NCCL_pad` 基线的加速比与趋势（三条路线的 loss 收敛均与基线对齐、全程 0 nan）：

| gbs | ODC_nopad 加速比 | 趋势解读 |
|---|---|---|
| 8 | ≈ **0.911×**（略慢） | minibatch 太小，反向中 p2p / 结算的固定开销占比大、又没能 overlap 起来 |
| 16 | ≈ **1.201×（峰值）** | 两块红利叠加，XGMI 按需 p2p 省集合 + 变长均衡免补空桶 |
| 32 | ≈ **1.142×** | 仍有优势，但 compute 变大、通信/均衡红利被摊薄 |
| 64 | ≈ **1.083×** | compute 变大、红利被摊薄，与 RCCL 基本持平 |
| 128 | ≈ **1.051×** | compute 已完全主导，ODC 的固定红利被摊平，与 RCCL 收敛|

单机的加速比曲线是**"gbs8 略慢 → gbs16 冲上峰值 → 之后随 gbs 增大缓慢回落，但始终稳定 >1"**。注意：它并**不会**在大 batch 处收敛到与 RCCL 持平，以下是对其实验结果详细的分析。

- **gbs8 略慢（nopad 0.911×、pad 0.898×）——固定开销摊不掉。** 单机每步的通信总量本就不大（1.5B 模型、节点内 XGMI），但 ODC 的 device 路径每个 minibatch 仍要付一笔**近乎固定的开销**：per-minibatch 的 `barrier` 加上 scatter-accumulate 的 `sync`/结算。当 gbs 只有 8 时，一步里真正的前反向计算量太小，这笔固定开销的**占比被放大**；再加上此时反向的 p2p 推送尚未能与计算 overlap（见第 5 节图 3），固定成本"露"在关键路径上收不回来，于是净体验比高度优化的 RCCL collective 略慢。这并非 ODC 的败笔，而恰恰印证了"红利需要足够的 batch 才能兑现"。

- **gbs16 见峰（nopad 1.201×）——峰值来自消除每层的隐式同步和减少补空桶的算力"。** 把峰值拆成两块乘子，正好落在实测上：
  1. **通信侧（`NCCL_pad → ODC_pad`）：XGMI 按需 p2p 省掉集合。 ODC 的 gather/scatter-accumulate 在节点内走 XGMI + HIP IPC（把 peer 显存直接映射进本进程、copy_ 直读写），是"谁需要谁去拉/推"的单边访问；而且同步频率从 per-layer 降到 per-minibatch，每 minibatch 的同步开销远小于每层一道集合的同步开销，省掉了 RCCL collective 里"全组逐层对齐 + ring/tree 调度"的固定成本。
  2. **负载侧（`ODC_pad → ODC_nopad`）：变长均衡免补空桶。 gbs16 / dp8 → 每 rank 只有 2 个样本，各 rank 负载抖动最大（见第 3 节图 3/图 4），nopad 允许各 rank 微批数不等、免去 pad 的空桶算力浪费，这块红利在小 gbs、强变长时最高。

  两者复合起来——用同节点 apples-to-apples 的 NCCL_pad → ODC_pad → ODC_nopad 阶梯（另一组 50 轮实验、只逐步改一个变量）：NCCL_pad → ODC_pad 先快 ~9.8%（纯减小每层同步的红利：数据、pad 策略完全一样，只把 RCCL 的逐层集合换成 ODC 的 XGMI 按需 p2p，每 minibatch 同步一次省下逐层同步）；ODC_pad → ODC_nopad 再快 ~10.2%（纯负载均衡红利：通信后端一样，只把"补空桶"换成"变长微批"）。两者复合起来 1.098 × 1.102 ≈ 1.21×（约合 ~20%），正好落在实测的 gbs16 峰值上。即单机的收益 = XGMI 按需通信 + 变长负载均衡，两块可加、缺一不可。

- 大 gbs 回落到持平（gbs64→1.0×、gbs128→~0.99×）——compute 主导、红利被摊薄。 gbs 越大，一步里的前反向 GEMM 计算量越大、占满了 GPU；而上面那两块红利是每步近乎固定或按比例但增速慢于 compute 的量。当 compute 成为绝对大头，固定红利被摊薄到噪声级别，ODC 与 RCCL 自然收敛到持平——这也解释了为什么单机是"驼峰"而非"单调"。

### 4.2 双机 14B（16 GPU，GDA/DEFER 路径，总时间口径）

跨节点场景换用 14B 模型、2×8 = 16 卡，节点间走 GDA（GPU-Direct RDMA）后端、nopad 走 DEFER rendezvous。下表按 gbs 给出 `ODC_nopad` 相对 `NCCL_pad` 基线的加速比与趋势：

| gbs | ODC_nopad 加速比 | 趋势解读 |
|---|---|---|
| 16 | ≈ **0.796×**（明显慢） | 缺 GDRW（GPU-initiated RDMA write），只能手动同步读保序，每层反向多一次同步开销 |
| 32 | ≈ **0.892×**（pad 0.844×） | 劣势收窄，但仍不及 RCCL |
| 64 | ≈ **1.120×（首次反超）** | 大 batch 摊薄跨节点固定开销 + 变长气泡收益兑现 |
| 128 | ≈ **1.154×** | gbs 越大、跨节点集合越贵，ODC 的摊销收益越大 |

双机的曲线与单机**恰好相反：随 gbs 单调上升**，而且其拐点与斜率都能用"跨节点固定开销 vs 可摊薄的 batch"这一对矛盾来解释：

- **小 gbs 明显慢（gbs16 ~0.796×、gbs32 nopad 0.892×/pad 0.844×）——跨节点固定开销叠加缺 GPU-initiated 的手动同步。** 跨节点这一跳，ODC 的 GDA 后端每步都要付几笔硬开销：① **跨节点 RDMA 本身**（`reduce_scatter` 的 pull 与 `all_gather`，实测占比约为单机同类核的 3×）；② **每步的同步 / settle**——由于我们的 ROCm GDA 目前**缺少 GDRW（GPU-initiated RDMA write）实现**，无法像 NVSHMEM 那样由 device kernel 直接发起并保证写有序可见，为确保"刚写下的梯度对远端 NIC 的 RDMA 读有序且正确可见"，必须**手动插一次同步读**（以读触发 HDP 冲刷，即第 5 节展望提到的 strided-touch），再叠加 per-minibatch 的 `barrier`——**每层反向都因此多背一笔同步开销**。这些都是**每步固定**的量，gbs16 时一步的有效计算太小、根本摊不掉，于是明显慢于成熟的 RCCL collective。

- **随 gbs 单调上升——固定开销被摊薄，变长红利同时放大。** gbs 从 16→32→64→128，一步里的有效前反向计算线性增长，而上面那笔"跨节点固定开销"基本不随 gbs 增长。于是每步"固定开销 / 有效计算"的比值单调下降，ODC 相对 RCCL 的劣势被持续吃掉；与此同时，跨节点场景下"负载不均衡气泡"比单机更昂贵（快 rank 得跨网络等慢 rank），nopad 免空桶、免同步等待的红利也随之放大——两股力量同向叠加，推着加速比一路走高。值得注意的是，跨节点场景下 `ODC_nopad` 全程都比 `ODC_pad` 更优（变长均衡的价值比单机更突出）。

- **gbs64 起 odc_nopad 反超（1.120×）、gbs128 达 1.154×——大 batch 兑现结构性红利。** 到 gbs64，摊薄后的固定开销已低于"省集合 + 免空桶气泡"的收益，odc_nopad **首次反超 RCCL**（此时 odc_pad 还差一口气，0.950×）；gbs128 时 batch 更大，RCCL 一侧的跨节点集合更贵，pad/nopad 双双反超（1.130× / 1.154×），ODC 的按需 p2p + 变长均衡愈发划算。这条"设备越多、越跨节点、负载越不均衡，ODC 越赚"的规律，与论文 Fig.10 的结论**完全一致**——ODC 的价值本就随规模与异质性增长。

---

<a id="5"></a>

## 5. 结论与展望

**回到第 1 节提出的两笔浪费。** FSDP 的集合通信同时制造了两类性能浪费：一是**逐层同步屏障的等待浪费**——每层都要停下来等全组对齐，GPU 周期空耗在跨 rank 等待上、通信无法与计算重叠；二是**负载不均衡气泡的算力浪费**——为了让集合能安全逐微批调用，只能靠"补空桶"把轻 rank 填平，轻 rank 的算力被 padding 微批白白吃掉。**ODC 正是对症下药**：把逐层屏障放松到 per-minibatch 一次结算，消除前一类等待浪费；用单边 p2p 支持变长微批、免掉空桶，消除后一类算力浪费。本文第 4 节的加速比，正是这两类浪费被消除后兑现的净收益——在 batch 足够大、负载足够不均衡的场景里，它们叠加成实打实的墙钟时间节省。

**结论**：

1. **移植跑通**：ODC 的算法层（gather / scatter-accumulate、per-minibatch 同步、LB-Mini 变长均衡）已在 AMD ROCm / MI300X 上用 **rocSHMEM / MORI** 后端跑通，**单机与双机**均能正确收敛（loss 与基线对齐、全程 0 nan）。NVIDIA 代码路径保留作 fallback，便于跟随上游合并。
2. **收益场景清晰**：ODC 在**"大 batch / 跨节点 / 存在负载不均衡"**时收益最大——单机 gbs16 峰值 ~1.201×、且大 batch 仍稳定领先（gbs128 ~1.051×）；双机自 gbs64 起 nopad 反超 RCCL、gbs128 达 ~1.154×。这与论文"ODC 从根本上减少 FSDP 的负载不均衡气泡，且加速比随设备数增长"的结论一致。


### 5.1 进一步优化的方向：settle 还没和反向 overlap

**ODC 的结算（settle）目前基本"露"在反向的关键路径上，未能与计算重叠。** 这一点在 trace 上看得非常清楚，也恰好与成熟的原生 NCCL 形成刺眼的对照——下面两张图（图 3 vs 图 4）。

![图3：开启ODC 反向——FSDP::post_backward_reduce 堆叠聚集在同一时刻，未与计算铺开重叠](odc_blog/fig4_odc_settle_no_overlap.png)

**图 3（ODC 反向：settle 堆在关键路径上，未 overlap）**：放大 ODC 的反向（`ProfilerStep#8`）。`stream 11` 上是 `FSDP::all_gather`，而 `stream 16–23` 上一整列 `FSDP::post_backward_reduce`（即 scatter-accumulate 的结算 / `WAIT_ACC` 等待，带 `M/n` 标记）**堆叠聚集在同一时刻**，并未铺开去与 `stream 0` 的计算重叠——结算被串在了关键路径上，overlap 很低（实测 ~5%）。也就是说，尽管我们已把跨 rank 同步从 per-iteration 降到 per-minibatch（第 3 节图 2），但**每层结算的等待仍"露"在计算之外**，未被后续反向计算掩盖。

![图4：原生 NCCL 反向——reduce_scatter_base 与 void ck_tile 计算在同一时段并行重叠（红框，ProfilerStep#10）](odc_blog/fig3_nccl_backward_overlap.png)

**图 4（对照：原生 NCCL 反向能把通信藏进计算）**：作为标杆，再看原生 FSDP2 + NCCL 的反向。红框圈出的是同一时段内 `stream 14` 的 `nccl:reduce_scatter_base`（+`ncclDevKernel_Generic_2`）与 `stream 0` 的 `void ck_tile` 计算**并行重叠**（`ProfilerStep#10`）——`reduce_scatter` 的通信核被 FSDP2 的 **prefetch** 机制**藏到了计算后面**：下一层的通信在当前层计算时就已发起，通信与计算实打实地并行。


**展望（待优化点，按性价比）**：

- **让 settle 与反向 overlap（首要）** 原生 NCCL 靠 prefetch 把通信藏进计算，我们要做的则是一条**跨迭代的软流水**——用独立 stream + event，把"上一组 microbatch 的 reduce-scatter / settle"与"本组 microbatch 的反向计算"重叠起来，只在 minibatch 末做一次总的 join。这最贴合 ODC 论文"推梯度与反向重叠"的思想，也最有可能把双机中等 gbs 追平、乃至把峰值再往上顶。
- **削减 / 合并跨节点固定开销**：warm-up settle 已从 full 优化到 strided（省 ~9–10%）；进一步可将每步的多个小集合分桶合并、减少 barrier 次数，直接压掉 4.2 提到的那笔"每步固定开销"。

---

### 参考

- 论文：《On-Demand Communication for FSDP》(ICLR 2026) — [OpenReview PDF](https://openreview.net/pdf?id=iIEEgI6WsF)（动机 Fig.1/2、方法 Fig.5、Collective LB-Micro 基线、加速比随设备数增长 Fig.10）
- 官方仓库：[sail-sg/odc](https://github.com/sail-sg/odc)（FSDP patch、gather + scatter-accumulate 原语、CUDA IPC + NVSHMEM 底座）
- 底座：[ROCm/mori](https://github.com/ROCm/mori)（MORI-SHMEM / MORI-IR，替代 NVSHMEM）、[Primus](https://github.com/AMD-AGI/Primus)（训练框架）
- 移植源码：`odc_rocm_dev/odc/primitives/{gather,scatter_accumulate,_rocshmem_backend,nvshmem_triton,utils}.py`、`odc/fsdp/fsdp2.py`、`odc_early/sitecustomize.py`、`primus/backends/megatron/patches/odc_{lb_mini,torch_fsdp2}_patches.py`
