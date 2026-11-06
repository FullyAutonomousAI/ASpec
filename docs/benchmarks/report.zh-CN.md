# ASpec 训练性能概览

本页汇总 2026-10-30 在 H200（Hopper）上完成的 DSpark 最终对照。所有 headline
统一采用 padded-document（BSHD）MBS1，即每个 Draft rank、每个 microstep 处理一篇
原始文档；TorchSpec、SpecForge 和 AngelSpec 同样取原生 MBS1。THD 作为
独立的布局与容量选项说明，不在每行择优混用。

指标是**训练输入 tokens/s**，不是推测解码速度，也不代表等质量训练加速。详细配置、
统计窗口和语义边界见[完整结果](results.md)。

## 端到端结果

单机八卡在线运行 Target 特征生成、传输与 Draft 训练。headline 采用 steady 窗口：
从第 8 个 warmup update 完成后计到最后一次 optimizer 完成，只统计该窗口内实际提交
的 tokens。正常供数、传输、Draft 训练和有限语料尾部仍在窗口内。

| 模型／长度 | TorchSpec MBS1 | SpecForge MBS1 | AngelSpec MBS1 | ASpec BSHD MBS1 | ASpec / TS、SF、AS |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-8B / 4K | 34,080.95 | 31,063.56 | 28,952.97 | **77,721.37** | **2.28x / 2.50x / 2.68x** |
| Qwen3-8B / 128K | 66,103.68 | 104,216.51 | 73,561.63 | **121,237.15** | **1.83x / 1.16x / 1.65x** |
| Qwen3.8-27B / 4K | 22,223.87 | 20,681.76 | 18,593.31 | **33,926.93** | **1.53x / 1.64x / 1.82x** |
| Qwen3.8-27B / 256K | 22,940.76 | 32,743.82 | 无完整结果 | **35,805.46** | **1.56x / 1.09x / --** |

AngelSpec 27B/256K 在已测原生容量下没有完整结果，因此不计算倍率。每个配置运行一次，
不声称统计显著性或通用倍率。

## THD 与 BSHD

MBS 在两种布局中的物理单位不同：

- BSHD MBS1 是每个 Draft rank、每个 microstep 一篇原始文档；
- THD MBS1 是一条 packed token row，其中可以包含多篇边界隔离的完整文档。

同窗口的两种 ASpec 布局记录如下。BSHD MBS1 作为整张公开矩阵的统一口径；THD 适合需要
在一行中容纳多篇文档、减少 padding，或压缩同一组常驻 token 显存的场景，但不承诺
无条件更快或绝对峰值显存更低。

| 模型／长度 | ASpec BSHD MBS1 steady（T+D） | ASpec THD steady（T+D） | headline 布局 |
| --- | ---: | ---: | --- |
| Qwen3-8B / 4K | **77,721.37**（2+6） | 75,606.91（2+6） | BSHD MBS1 |
| Qwen3-8B / 128K | **121,237.15**（6+2） | 127,645.69（7+1） | BSHD MBS1 |
| Qwen3.8-27B / 4K | **33,926.93**（3+5） | 34,157.98（3+5） | BSHD MBS1 |
| Qwen3.8-27B / 256K | **35,805.46**（6+2） | 35,810.69（6+2） | BSHD MBS1 |

8B/128K 的角色配比不同，自然 THD 每次更新的原始文档数也会变化；其中 27B/256K
提交了 optimizer 对齐的 4,094 篇前缀。因此上表不是纯布局消融，下面的四文档常驻
对照才固定了相同文档集合。

单独报告的 BSHD MBS4 吞吐不进入 headline。在 Qwen3-8B/4K 同四篇文档的受控
常驻对照中，BSHD MBS4 的 allocated 峰值为 57.08 GiB，THD MBS1 为 48.12 GiB，
THD 节省 15.7%；两者吞吐分别为 15,809.82 和 15,738.93 tokens/s，因此这是显存
结论而不是加速结论。自然单文档 MBS1 下则可能由 BSHD 获得更低的绝对峰值。

## Draft-only 结果

单卡 CP1 使用共同的驻留 Target 特征与初始 Draft 权重，预热 16 步后连续测量 64 步；
不包含 Target 执行、传输和特征加载。

| 模型／长度 | TorchSpec MBS1 | SpecForge MBS1 | AngelSpec MBS1 | ASpec BSHD MBS1 | ASpec / TS、SF、AS |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-8B / 4K | 7,774.31 | 7,370.28 | 6,155.94 | **15,256.44** | **1.96x / 2.07x / 2.48x** |
| Qwen3-8B / 128K | 147,198.24 | 153,246.20 | 125,672.09 | **288,380.52** | **1.96x / 1.88x / 2.29x** |
| Qwen3.8-27B / 4K | 5,035.68 | 4,873.88 | 3,996.08 | **9,573.42** | **1.90x / 1.96x / 2.40x** |
| Qwen3.8-27B / 256K | 128,328.29 | 130,057.91 | 容量不通过 | **229,232.34** | **1.79x / 1.76x / --** |

Draft-only 是训练侧容量，不可替代端到端吞吐。所有 ASpec 测量步都有有限 loss／gradient
norm，并在五层 Draft 上实际观察到 cuDNN attention。

## 数据与比较边界

- 4K 使用已有 Target-generated 回答，不重新解码或生成；8B 最终整轮提交 60,000 篇、
  43,840,380 input tokens。
- 128K／256K 使用固定顺序的 4,096 篇 LongCite 完整问答，各模型使用自己的 tokenizer；
  不复制、不截断、不人工扩写。
- 每篇原始文档独立拥有 512-anchor 上限；无效位置保持 mask，不重复 anchor 填满预算。
- 采用 BF16、五层 DSpark、block size 7 和各框架原生 optimizer／学习率语义。
- 正常启用 NVLink 和容量充足的核心 buffer；不做 CPU、tile、kernel 或学习率专项追分。
- 128K／256K 表示配置和 workload 的行容量，不要求自然语料必须出现恰好达到上限的 row。

对照固定语料身份、逐文档 anchor 预算、每次更新的原始文档数和完整墙钟统计，但不声称
不同框架实现的独立训练轨迹完全一致。

## 复现入口

[完整结果](results.md) · [测量方法](methodology.md) ·
[复现工具与配置模板](README.md) · [同序列训练与接受长度对比](dspark-alignment.md)
