# ASpec：我们如何加速推测解码的 Draft 训练

> **TL;DR：** [ASpec](https://github.com/inclusionAI/ASpec) 是面向推测解码的高性能 Draft 训练框架，优化 Draft 执行与完整训练流水线，也支持**共享 Target 特征，同时运行多个独立的 Draft 训练实验**。
>
> 在 Qwen3-8B / 4K 的 Hopper 对照中，单卡 Draft 训练输入吞吐达到 TorchSpec / SpecForge 的 **1.96× / 2.07×**，八卡端到端稳态输入吞吐达到 **2.28× / 2.50×**。

![Qwen3-8B / 4K DSpark training throughput: single-GPU Draft-only on the left and eight-GPU online steady state on the right, comparing TorchSpec, SpecForge, AngelSpec and ASpec.](assets/aspec-training-overview.svg)

---

刚接触推测解码时，我们试了几套开源框架，原本只想先把训练跑起来，再研究算法和模型。

但用下来发现，Draft 的计算和显存开销、Target 的供数效率，以及多个实验之间重复的特征生成等，都还有不少优化空间。

我们因此开始做 ASpec，希望给社区提供一套面向生产场景优化的 Draft 训练基础设施：既让每次训练更快，也减少多个实验之间的重复工作。

ASpec 已提供 Qwen、Ling 的多种 Draft 训练路径。本文以投入最多的 DSpark 路径为例，从一次前后向讲起，再看完整训练和多实验共享。

## Draft 训练的开销从哪里来

DSpark 训练时，Target 先对文档做 prefill，提供中间层特征。Draft 从文档中采样一批预测起点，称为 anchors；每个 anchor 展开一个长度为 $B$ 的 block：首个输入是已知的 anchor token，其余位置用 `[MASK]` 初始化。这些 block 可以并行计算。

在标准 full-attention 设置下，每个 block 能读取自己 anchor 之前的 context 特征，以及自身 block 内的全部位置。它看不到其他 anchor 的 Draft block，也看不到 anchor 之后的 context 特征。Target prefill 本身是 causal 的，Draft block 内则是双向 attention；其中的 `[MASK]` 输入不包含未来真实 token。

![DSpark full-attention visibility: each Draft block attends to the context prefix before its anchor and its own entire block. Other Draft blocks and future context are masked.](assets/dspark-attention-pattern.svg)

即使 $B=7$，每个 query 也要读取可能长达数万 token 的 context 前缀。经过多层 attention 和 MLP 后，hidden states 还要通过 LM Head 投影到整个词表，得到 logits，再计算 token prediction 和 distillation loss。预测 block 很短，并不意味着训练开销很小。

## 先把 Draft 训练跑快

先移除无效 query blocks。训练通常只对部分 token 计算 loss，例如对话中的 assistant 回答。受文档边界、loss mask 和张量对齐影响，一部分预留槽位没有有效的训练目标。如果只在最终 loss 上乘零，前面的 attention 和 MLP 仍然会执行。

ASpec 在支持的 Dense full/sliding attention 路径上，将整块无效 query 移除，把有效 block 紧凑排列，也就是 compaction。有效 anchors 和 attention 可见关系保持不变，后续各层需要处理的 query 数量则可以减少。

### DSpark-aware kernel 优化

DSpark 的 context 与 Draft block 有明确的组织方式，attention 的可见范围也由文档边界和 anchor 决定。我们沿着 Q/K 准备、attention 执行和 GQA 反向这条计算链，利用这些结构优化 kernel，减少中间搬运、重复判断和梯度归约的显存开销。

![DSpark kernel optimizations: fused Q/K RMSNorm and indexed RoPE, preplanned visibility with empty-tile skipping, and on-chip dK/dV reduction for shared KV heads in GQA backward.](assets/dspark-kernel-optimizations.svg)

**融合 Q/K 准备，直接产出所需布局。** Q/K 进入 attention 前，需要完成 RMSNorm 和 indexed RoPE，K 还要组织成 context 与 Draft 两部分的布局。逐步执行会先拼接 K，再多次读写中间张量。ASpec 用 CuTeDSL 融合这些操作的前后向，直接读取 context K 和 Draft K，写出 attention 所需的布局，省掉预先拼接 K 的中间结果及对应的数据搬运。投影 GEMM 和 V 的拼接仍独立执行。

**预处理可见性，减少主 kernel 中的判断。** 每条 query 的可见范围可以表示为“自己文档的 context 前缀”和“自己的 Draft block”两个区间。通用 mask 可以逐项判断文档、anchor 和 block 的关系，但这些关系在进入 attention 前就已确定。ASpec 将区间端点交给 cuDNN，预先生成稀疏遍历计划，并把边界 tile 的可见性编码为紧凑的 predicates。

这样，完全不可见的 tile 可以跳过，完全可见的 tile 无需逐项判断，边界 tile 则读取预编码的 predicates。部分可见的 tile 仍执行稠密矩阵运算，但主 kernel 不必反复重建那套文档和 block 关系。同一批 anchors 和文档布局下，attention 策略、张量形状与布局一致的层还可以复用 plan。

**在片上归约 GQA 梯度，减少全局显存往返。** 多个 Q heads 共享同一个 KV head 时，它们对 dK/dV 的贡献需要相加。原生 cuDNN 的这条路径先在全局 FP32 buffer 中累加，再读取、缩放并转换为最终梯度；清零、累加和后处理都需要访问这些 buffer。

ASpec 提供可选的联合 backward，在同一次 KV tile 遍历中收齐四个 Q heads 的贡献，在片上完成 FP32 累加，直接写出 dK/dV。Q/dO 也按原有布局读取，无需额外重排。该路径适用于 context 未跨卡切分的受支持配置，forward 沿用原生 cuDNN，dQ workspace 仍保留。

### LM Head 与 loss：复用 logits，按需重建 softmax

Attention 之后，每个预测位置都要通过 LM Head 投影到整个词表。distillation loss 需要同时处理 Draft 和 Target 两份 logits；即使 Draft 层数很少，这部分计算和中间张量仍随预测位置数与词表大小增长。

其中有一部分投影是在重复计算：不同 anchors 的 block 可能覆盖同一个 Target 位置，冻结的 Target 特征和 LM Head 会给出相同的 logits。按 block 各算一次，就重复做了词表投影；head/loss 启用 activation checkpointing 后，反向重计算又可能再做一次。全部缓存虽能省计算，却要留下大量随词表大小增长的张量。

![LM Head and L1 loss: budgeted Target logits reuse across prediction blocks, tiled loss over the full vocabulary, and backward softmax reconstruction from logits and statistics. The eight FP32 scalars per prediction position cover replay statistics only.](assets/dspark-head-loss.svg)

ASpec 为 logits 设置缓存预算，比较不同复用方式在前向和反向重计算中的总投影量。context 未跨卡切分时，如果整段 Target context 只投影一次更省计算、也放得进预算，就缓存这份结果，供各个 block 按位置读取；预算内也可以保留部分 chunk 的 Target 和 Draft logits，避免 checkpointing 时重复投影。缓存只在同一次前后向内复用，这里的 checkpointing 也限定在 head/loss，不会因此重跑 backbone。

投影之后，L1 distillation loss 还要比较两个完整词表上的 softmax 输出。直接保留这些 FP32 中间结果，相当于在 logits 之外再存一批随词表扩大的张量。ASpec 沿词表分块计算 loss，用 FP32 统计完成全词表归一化；反向需要的 softmax 值，再由 logits 按块重建。

L1 的重算统计缓存每个预测位置仅需 **8 个 FP32 标量**，不随词表大小增长；logits 缓存和临时 workspace 则单独管理。完整词表和 loss 归一化保持不变，但不再需要为反向保留整份 FP32 softmax。

### 训练执行的实际收益

我们使用相同的初始权重，将 Target 预先生成的特征加载到 GPU，在单张 Hopper GPU 上测量 Draft 的前向、反向和参数更新。在下面的 4K 与长上下文对照中，ASpec 的 Draft 输入吞吐约为对照框架的 **1.8–2.5 倍**，短文档和长上下文都有提升。

表中的 tok/s 按输入文档 token 数统计，倍率为 ASpec 输入吞吐除以对应框架输入吞吐。

| 模型 / 输入长度上限 | ASpec 输入吞吐（tok/s） | 相对 TorchSpec | 相对 SpecForge | 相对 AngelSpec |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-8B / 4K | **15,256.44** | **1.96×** | **2.07×** | **2.48×** |
| Qwen3-8B / 128K | **288,380.52** | **1.96×** | **1.88×** | **2.29×** |
| Qwen3.8-27B / 4K | **9,573.42** | **1.90×** | **1.96×** | **2.40×** |
| Qwen3.8-27B / 256K | **229,232.34** | **1.79×** | **1.76×** | — |

<sub>注：AngelSpec 的 27B/256K 单卡配置未通过容量验证，没有对应倍率。</sub>

## 从 Draft 执行到完整训练

### 变长数据与长上下文

训练数据通常是变长的。放大 microbatch 后，如果把每篇文档都补到这一批的最长长度，就会多出 padding。例如三篇 1K、1K、2K 的文档，补齐后占用 6K context token 槽位，其中 2K 是 padding。

ASpec 可以把长度接近的文档组成 microbatch，兼顾 padding 和各卡的工作量；也可以通过 packing，把这三篇文档紧凑地放进一行 4K，同时保留各自的 attention 边界、文档内位置和逐文档 anchor 预算。这主要帮助较大 microbatch 减少 padding 与存储空洞，实际收益取决于 batch 和显存预算。前面的吞吐表统一采用逐文档 MBS1，布局比较另见[实测结果](benchmarks/results.md#aspec-layout-reference)。

单篇文档长到 128K、256K 时，即使没有 padding，Target 特征、context K/V 和训练激活也会随长度增长。ASpec 用 Context Parallelism（CP）分摊这部分压力：context 和 Draft K/V 分布到多张卡，各卡计算自己负责部分的 attention，再借助 softmax 归一化信息合并结果，反向沿相同关系传播。这样可以分摊长上下文的存储与计算，避免每张卡都复制完整 K/V。cuDNN 路径直接接收序列边界与可见区间，减少稠密 mask 和逐层拆装文档的开销，具体见 [attention 与 CP](benchmarks/cudnn-thd-cp.md)。

### 异步流水线

在线训练中，如果 Target 供数跟不上，Draft 算得再快，也会有一部分时间在等待。长上下文下，Target prefill 的开销尤其突出。

ASpec 将 prefill、特征捕获、传输与 Draft 更新异步执行，让这些工作尽量重叠。在同机 NVLink P2P 路径上，Target 特征从 GPU 发送 buffer 直接传到 Draft 侧的 GPU 接收 buffer，无需经过 CPU 内存中转。

启用 direct-prefill 时，捕获的特征直接写入发送 buffer，省去先收集、拼接，再拷入发送区的中间步骤；接收层可直接用 tensor view 读取接收 buffer，减少额外拷贝。

源端存储、接收 buffer 和队列各有容量限制，未消费的数据达到预算时，背压会减缓供数；特征在读取完成后释放，避免持续运行时显存积压。

下面是单机八张 Hopper GPU 的在线稳态结果。计时包含 Target 特征生成、传输、供数等待和 Draft 更新，各框架使用各自合理的 Target/Draft 配比。

| 模型 / 输入长度上限 | ASpec 输入吞吐（tok/s） | 相对 TorchSpec | 相对 SpecForge | 相对 AngelSpec |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-8B / 4K | **77,721.37** | **2.28×** | **2.50×** | **2.68×** |
| Qwen3-8B / 128K | **121,237.15** | **1.83×** | **1.16×** | **1.65×** |
| Qwen3.8-27B / 4K | **33,926.93** | **1.53×** | **1.64×** | **1.82×** |
| Qwen3.8-27B / 256K | **35,805.46** | **1.56×** | **1.09×** | — |

<sub>注：AngelSpec 的 27B/256K 没有完整端到端结果，没有对应倍率。</sub>

4K 场景中，Draft 执行优化和流水线一起带来了明显的端到端提升。长上下文的两组结果则呈现了另一种情况：相对 SpecForge，Qwen3-8B/128K 的 Draft 独立吞吐为 **1.88 倍**，端到端为 **1.16 倍**；Qwen3.8-27B/256K 则分别为 **1.76 倍**和 **1.09 倍**。

**这两组长上下文训练主要受 Target 供数速度限制，Draft 的处理能力尚未充分发挥。** 两组 ASpec 配置都将八卡分为 6 张 Target GPU 和 2 张 Draft GPU，长序列 prefill 仍然跟不上训练侧。

如果有更多 GPU 预算，可以保留所需的 Draft 资源，为 Target 增加供数 GPU。随着实际特征供给能力提升，Draft 的等待时间能够减少，训练侧优势也能更充分地转化为端到端提速。表中展示的是固定八卡预算下的结果；增配后的具体加速比，需要在新的资源配比下测量。

## 一份 Target 特征，多个 Draft 实验

尝试不同的层数、模型结构或训练算法时，多组实验往往使用相同的 Target 和输入。

ASpec 允许多个独立 Draft 共享兼容的 Target 特征。不同实验需要的捕获层可以取并集，再分别提供各自需要的特征。每个 Draft 仍然拥有自己的权重、optimizer、学习率、随机状态和 checkpoint，按自己的进度训练，不必在每次参数更新时等待其他实验。

![ASpec：异步生成与传输 Target 特征，供多个独立 Draft 实验使用；各实验保留自己的训练状态。](assets/architecture.svg)

系统会限制待消费的数据量，避免较慢的实验让特征无限积压。一个实验结束或退出后，其他实验可以继续运行；恢复时，每个实验从各自已提交的训练进度继续。当前多 Draft 共享模式支持单机，配置与恢复方式见[使用指南](user-guide/sharing.md)。

## 接下来

ASpec 也提供 DFlash、EAGLE3/4 和原生 MTP 等训练路径，覆盖 Qwen、Ling 模型，以及 Dense、所支持的 MoE 和多种 attention 结构。不同组合的支持范围见[算法](user-guide/algorithms.md)、[结构](user-guide/structures.md)和 [recipe 指南](user-guide/recipes.md)。

如果想先运行一次短训练，可以从 [Quickstart](user-guide/quickstart.md) 的 Qwen3-8B 或 Ling tiny 开始，完成训练、恢复和导出，再接入自己的数据。

后续性能优化仍然是投入重点：继续优化 Draft 算子和 head/loss，改善长上下文下的 Target 供数与通信，并探索 LM Head FP8 等低精度计算。模型与结构方面，我们也会继续推进 Ling 生态和 CSA2 等方向，具体计划见 [Roadmap](../ROADMAP.md)。

感谢 TorchSpec、AngelSpec、SpecForge 等开源项目提供的实现与经验。ASpec 复用了其中的基础，也希望将我们积累的优化回馈给社区。

如果你也在做 Draft 训练，欢迎试试 ASpec。有问题或想法，可以在 [GitHub](https://github.com/inclusionAI/ASpec) 上和我们交流。

---

## 附录：性能数据说明

框架吞吐来自 **2026 年 10 月 30 日**的 Hopper GPU 对照，使用五层 DSpark Draft、BF16 和 block size 7。各框架按原始文档对齐 batch 与 anchor 预算，每张 Draft 卡每个 microstep 处理一篇文档。4K、128K、256K 表示输入长度上限，文档保留自然长度。

**Draft-only** 在计时前将 Target 特征加载到 GPU，测量前向、反向和参数更新；**端到端稳态**包含 Target 供数、特征传输、等待与 Draft 更新，排除冷编译和初始流水线填充。文中的 AngelSpec 使用针对本次测试配置的兼容性补丁。

这些数字衡量训练输入处理效率。数据集、软件版本、资源配比和完整计时口径见[正式结果](benchmarks/results.md)，复现方式见 [benchmark 指南](benchmarks/README.md)。

训练效果已有独立的[训练质量与接受效果对照](benchmarks/dspark-alignment.md)，包含 held-out CE 和真实 verifier 的平均接受长度。吞吐与质量报告分别列出各自的配置；输入吞吐倍率不等同于推理解码加速，也不直接表示达到相同模型质量所需的训练时间倍率。
