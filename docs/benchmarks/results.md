# Training benchmark results

This page combines the **2026-10-30 H200 DSpark comparison** for Qwen3-8B
and Qwen3.8-27B with the **2026-11-07 Qwen3.5-9B comparison**. Each study
retains its measured source and runtime identity. Throughput means committed
training input tokens per second; it is not speculative-decoding speed or an
equal-quality training speedup.

The comparison uses an independent 512-anchor cap for every original document.
All headline rows use one padded document per Draft rank and microstep (BSHD
MBS1). TorchSpec, SpecForge and AngelSpec use the same native MBS1
semantics. Packed-token execution (THD) is reported separately as a layout and
capacity option; results are not selected opportunistically between layouts.

See [methodology](methodology.md) for the measurement contract and
[benchmark tools](README.md) for reproduction entry points.

## End-to-end training

Each run uses one eight-GPU node for online Target feature generation, transport
and Draft training. The headline is a continuous **steady window** from completion
of warmup update eight through the final optimizer completion. It counts only
the actual tokens committed in that interval. Target supply, transport, Draft
training and the finite tail remain inside the window; cold compilation and
initial pipeline fill do not.

### Steady-window throughput

| Workload | TorchSpec MBS1 | SpecForge MBS1 | AngelSpec MBS1 | ASpec BSHD MBS1 | ASpec / TS, SF, Angel |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-8B / 4K | 34,080.95 | 31,063.56 | 28,952.97 | **77,721.37** | **2.28x / 2.50x / 2.68x** |
| Qwen3-8B / 128K | 66,103.68 | 104,216.51 | 73,561.63 | **121,237.15** | **1.83x / 1.16x / 1.65x** |
| Qwen3.5-9B / 4K | 27,858.39 | 25,024.26 | 22,478.02 | **66,786.94** | **2.40x / 2.67x / 2.97x** |
| Qwen3.5-9B / 256K | 55,040.72 | 105,469.48 | 87,413.03 | **113,141.77** | **2.06x / 1.07x / 1.29x** |
| Qwen3.8-27B / 4K | 22,223.87 | 20,681.76 | 18,593.31 | **33,926.93** | **1.53x / 1.64x / 1.82x** |
| Qwen3.8-27B / 256K | 22,940.76 | 32,743.82 | No complete result | **35,805.46** | **1.56x / 1.09x / --** |

![End-to-end steady-window input throughput for TorchSpec, SpecForge, AngelSpec and ASpec across six Qwen workloads. All six panels share a 0–130k tokens-per-second scale; ASpec is the highest measured result in each workload.](../assets/benchmark-steady-throughput.svg)

Ratios are calculated from unrounded steady rates. AngelSpec 27B/256K did not
complete under the tested native capacity, so it supplies no denominator.
Native async ordering can change the post-warmup token subset; each rate uses
its own actual committed tokens rather than a corpus-average estimate.

### Complete-window reference

The complete window starts at native training-loop entry and includes cold
compilation, warmup and initial pipeline fill through the last optimizer
completion. It excludes process/model initialization, final save and shutdown.
These rates describe complete-corpus elapsed time and are not used in the
headline ratios above.

| Workload | TorchSpec MBS1 | SpecForge MBS1 | AngelSpec MBS1 | ASpec BSHD MBS1 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-8B / 4K | 32,291.61 | 29,628.43 | 27,575.80 | 68,348.38 |
| Qwen3-8B / 128K | 64,567.14 | 90,114.58 | 70,672.97 | 116,436.66 |
| Qwen3.5-9B / 4K | 26,227.84 | 24,114.84 | 21,445.72 | 62,392.92 |
| Qwen3.5-9B / 256K | 53,881.33 | 91,641.71 | 85,614.53 | 108,731.03 |
| Qwen3.8-27B / 4K | 21,538.26 | 20,048.01 | 18,162.37 | 32,481.90 |
| Qwen3.8-27B / 256K | 22,812.26 | 31,321.04 | No complete result | 35,377.05 |

### ASpec layout reference

ASpec's MBS has a different physical unit in the two layouts:

- **Padded documents (BSHD):** MBS1 means one original document per Draft rank
  and microstep.
- **Packed tokens (THD):** MBS1 means one packed token row per Draft rank and
  microstep. A row can contain several complete documents separated by explicit
  attention and position boundaries.

The same-window ASpec layout observations below cover Qwen3-8B and Qwen3.8-27B.
BSHD MBS1 is the public headline because it preserves the same physical
microbatch semantics as the external baselines across the full matrix. THD
remains useful when several documents must share a row, padding would otherwise
dominate, or the resident token set must fit a tighter memory budget; it is not
an unconditional speed or absolute-memory improvement.

| Workload | ASpec BSHD MBS1 steady (T+D) | ASpec THD steady (T+D) | Headline layout |
| --- | ---: | ---: | --- |
| Qwen3-8B / 4K | **77,721.37** (2+6) | 75,606.91 (2+6) | BSHD MBS1 |
| Qwen3-8B / 128K | **121,237.15** (6+2) | 127,645.69 (7+1) | BSHD MBS1 |
| Qwen3.8-27B / 4K | **33,926.93** (3+5) | 34,157.98 (3+5) | BSHD MBS1 |
| Qwen3.8-27B / 256K | **35,805.46** (6+2) | 35,810.69 (6+2) | BSHD MBS1 |

The 8B/128K role split differs, and natural THD uses dynamic original-document
counts per update; its 27B/256K run commits the 4,094-document
optimizer-aligned prefix. This table is therefore not a pure layout ablation.
The controlled resident comparison below holds the four documents fixed.

In a controlled Qwen3-8B/4K same-four-document resident comparison, BSHD MBS4
peaks at 57.08 GiB allocated while THD MBS1 peaks at 48.12 GiB, a 15.7%
reduction for THD. Their input rates are 15,809.82 and 15,738.93 tokens/s,
respectively, so this is a resident-memory result rather than a speedup claim.
Natural one-document MBS1 batches can instead give BSHD the lower absolute peak.
The headline fixes padded-document MBS1 across frameworks rather than selecting
a different physical microbatch size for each implementation.

### Work committed by ASpec

All headline rows use original documents as the logical batch unit.

| Workload | Target + Draft | Draft CP | Original documents/update | Committed documents / input tokens | Optimizer updates |
| --- | --- | ---: | ---: | --- | ---: |
| Qwen3-8B / 4K | 2+6 | 1 | 480 | 60,000 / 43,840,380 | 125 |
| Qwen3-8B / 128K | 6+2 | 1 | 4 | 4,096 / 132,381,588 | 1,024 |
| Qwen3.5-9B / 4K | 2+6 | 1 | 480 | 59,040 / 43,742,746 | 123 |
| Qwen3.5-9B / 256K | 6+2 | 1 | 4 | 4,096 / 132,300,003 | 1,024 |
| Qwen3.8-27B / 4K | 3+5 | 1 | 480 | 59,520 / 43,941,539 | 124 |
| Qwen3.8-27B / 256K | 6+2 | 2 | 4 | 4,096 / 132,455,651 | 1,024 |

The comparison fixes corpus identity, per-document anchor budget, original
documents per update and timing boundary. It does not claim equal independent
weight trajectories across different framework implementations.

## Draft-only training

Draft-only controls use one H200 GPU, CP1, common resident Target activations and
common initial Draft weights. They run 16 warmup plus 64 measured updates and
exclude Target execution, transport, feature loading and initialization. Every
framework uses padded-document MBS1; accumulation preserves the workload's
logical documents per update.

| Workload | TorchSpec MBS1 | SpecForge MBS1 | AngelSpec MBS1 | ASpec BSHD MBS1 | ASpec / TS, SF, Angel |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-8B / 4K | 7,774.31 | 7,370.28 | 6,155.94 | **15,256.44** | **1.96x / 2.07x / 2.48x** |
| Qwen3-8B / 128K | 147,198.24 | 153,246.20 | 125,672.09 | **288,380.52** | **1.96x / 1.88x / 2.29x** |
| Qwen3.5-9B / 4K | 6,203.70 | 5,922.16 | 4,785.72 | **12,646.83** | **2.04x / 2.14x / 2.64x** |
| Qwen3.5-9B / 256K | 143,166.12 | 152,704.74 | 121,555.04 | **294,661.38** | **2.06x / 1.93x / 2.42x** |
| Qwen3.8-27B / 4K | 5,035.68 | 4,873.88 | 3,996.08 | **9,573.42** | **1.90x / 1.96x / 2.40x** |
| Qwen3.8-27B / 256K | 128,328.29 | 130,057.91 | Capacity failure | **229,232.34** | **1.79x / 1.76x / --** |

The ASpec rows use 64 original documents per optimizer update at 4K and four
at long context. These resident rates measure Draft execution capacity and
must not be substituted for online end-to-end throughput.

All measured ASpec updates have finite loss and gradient norm records and use
cuDNN attention on all five Draft layers. Common-state numerical and restore
checks are separate from these performance claims.

## Data and training semantics

- **4K:** existing Target-generated answers; no decoding or answer regeneration.
  Each model's tokenizer and admission rules yield the complete prefix disclosed
  in the committed-work table above.
- **128K/256K:** 4,096 seed-42 LongCite conversations in frozen order, using each
  model's tokenizer. Conversations are not duplicated, truncated or expanded.
- **Supervision:** every original document has an independent cap of 512 anchors.
  Structurally unavailable anchors remain masked and are never repeated to fill
  the cap.
- **Precision/objective:** BF16 five-layer DSpark, block size 7, native framework
  optimizer and learning-rate semantics. Drafts are independent of the Target's
  Native MTP head, where present.
- **Tails:** finite tails are explicit and no work is duplicated. The 8B/9B/27B
  4K runs drop the final 416/55/384 documents that cannot form a complete
  480-document update; the long-context runs consume all 4,096 documents.

Input-length labels identify the configured row capacity and workload, not a
requirement that a natural corpus contain a row at the exact ceiling.

## Runtime and framework identity

The studies retain each framework's normal software stack and native backend;
it is not a same-version kernel ablation. The runtime columns below list
**PyTorch / SGLang** for each measured model group.

| Framework | Qwen3-8B / Qwen3.8-27B (2026-10-30) | Qwen3.5-9B (2026-11-07) | Source identity |
| --- | --- | --- | --- |
| ASpec | 2.13 / 0.5.21 | 2.13.0+cu130 / 0.5.21 | [8B/27B measured runtime snapshots](source-identities.json) |
| TorchSpec | 2.11 / 0.5.19.dev | 2.11.0+cu130 / 0.5.19.dev8+g49e384ce9 | `a4e5c1085bfc5990ddcae6cf5832bb26380b38d3` |
| SpecForge | 2.13 / 0.5.18 | 2.13.0+cu130 / 0.5.18 | `03f3682ea852b73488d62dab4cd539033cc60e3e` |
| AngelSpec | 2.9.1 / 0.5.18 | 2.9.1+cu128 / 0.5.18 | Based on [`d3412bed231025ea85d35dbf5e0af44f1ac62a5b`](https://github.com/Tencent/AngelSpec/commit/d3412bed231025ea85d35dbf5e0af44f1ac62a5b), with compatibility and native NVLink transport adaptations; [8B/27B measured runtime snapshots](source-identities.json) |

ASpec uses cuDNN with padded-document workload semantics (CP may flatten
internally), FA4 4.0.0b34, CUTLASS DSL 4.7.1 and Quack 0.6.5. Baselines retain
native FlexAttention. SpecForge's 9B runs use
`torch.compile(create_block_mask, dynamic=True)` for dynamic-length compatibility,
verified against native mask metadata and an independent dense forward/backward
reference; its sampler, objective and optimizer remain unchanged.

The 2026-10-30 ASpec and modified AngelSpec measurements used source exports without Git metadata.
The [source identities](source-identities.json) bind the 2026-10-30 rows to the
runtime-module SHA-256 values recorded by their adapters, with separate online
and resident measurements.
Git revisions for those exports remain `null`; the AngelSpec revision above identifies the upstream
base. The fingerprints cover the captured modules, while dependencies and workload settings are
reported separately.

All workloads use H200 allocations, with eight GPUs per online run and one GPU
per resident control. The 9B study uses two nodes independently; their NVML
labels report L20X/8.9 while actual PyTorch execution reports SM90 on both.
Baselines and ASpec 4K use one node; ASpec 256K uses the other. Each online run
remains within a single node.

Normal NVLink and capacity-correct source/ready/destination/host buffers are
enabled where supported. No CPU, tile, kernel or learning-rate sweep is used to
manufacture an individual result. These are single-run observations, without a
repeated-run confidence or statistical-significance claim.

## Related measurements

- [DeepSpec comparison](deepspec.md): resident Draft training and the native
  cache-first workflow, reported under their separate timing boundaries.
- [EAGLE3](eagle3-h200.md): train-core latency, memory and numerical scope.
- [Packed attention with CP](cudnn-thd-cp.md): component-level attention results.
- [Training and acceptance](dspark-alignment.md): TorchSpec/ASpec BSHD/THD
  training alignment and verifier metrics, separate from input throughput.
- [CP quality alignment](dspark-cp-alignment.md): CP2 versus same-layout CP1
  for ASpec BSHD and THD.

## Reproduce a comparison

Start with the [benchmark tools and templates](README.md). Match the dataset,
tokenizer, initial Draft weights, per-document anchor policy, role split, logical
work per update and timing boundary. Record physical MBS in its layout-specific
unit, packed rows, original documents, input tokens, optimizer updates and finite
tails separately. Keep generated data, logs, checkpoints and profiler traces
outside the public source tree.
