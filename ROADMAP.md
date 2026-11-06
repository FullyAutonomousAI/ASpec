# ASpec roadmap

ASpec is a Ling open-source project building a training platform for multiple speculative
decoding algorithms, with model- and algorithm-specific contracts, data paths and kernels where
specialization provides meaningful value.

- `[x]` Ready in the documented `v0.1` release-candidate boundary.
- `[ ]` Planned or under evaluation; not ready and without a promised delivery date.

The authoritative details and limitations are in [supported paths](docs/support.md).

## Training platform

- [x] Asynchronous Target feature generation and Draft training.
- [x] Explicit Plan/Evidence, exact recovery, evaluation and strict export.
- [x] Sequence packing, context parallelism and independent Drafts sharing compatible Target data.
- [x] Primary SGLang + Qwen3/Ling path on single-node multi-GPU systems.
- [x] Bounded static multi-node, single-Draft execution.
- [ ] Multi-node multi-Draft training: share compatible Target features across independent Draft
  experiments, preserving per-Draft progress and checkpoint recovery without requiring a shared filesystem.
- [ ] Elastic membership and broader failure recovery.

## Algorithms and model support

- [x] DSpark and DFlash as the primary optimized algorithms.
- [x] EAGLE3, EAGLE4 and Qwen3.5 native single-predictor MTP.
- [x] Qwen3-family plus Ling-3.0-tiny, Ling-3.0-flash and Ling-3.1-flash Target recipes, with explicit
  tokenizer, feature-layer, normalization and checkpoint contracts.
- [x] Dense, selected MoE, MHA/GQA, materialized MLA, SWA, CSA, QSA and DSA paths.
- [ ] Promote the existing experimental DFlash2 path with complete recipe and quality evidence.
- [ ] Generalized multi-layer/MoE native MTP.
- [ ] JetSpec-style training with native objective, export and evaluation contracts.
- [ ] P-EAGLE, Medusa and other tree/head Draft algorithms.
- [ ] CSA2 training and export support.
- [ ] Native KDA/absorbed MLA and K3-native specialization.
- [ ] Extend Ling coverage across more Draft algorithms and build maintained long-context,
  convergence, acceptance and downstream-serving evidence.

See [Ling recipes and model contracts](docs/user-guide/ling.md).

## Algorithm-specific performance

- [x] ASpec-owned DSpark Q/K normalization and RoPE CuTeDSL kernels.
- [x] FA4 sparse/flattened-token paths and bounded cuDNN flattened-token/FlexAttention paths.
- [x] Packing/CP-aware execution for the documented algorithm and attention combinations.
- [ ] Dedicated forward/backward attention kernels for additional block, tree and sparse masks.
- [ ] Extend native ragged THD to more attention structures and shapes, guided by end-to-end measurements.
- [ ] More fused projection, normalization, RoPE, loss and pointwise paths based on profiler evidence.
- [ ] LM Head quantization (for example, FP8), with numerical and model-quality validation.
- [ ] Architecture-specific kernel schedules with the same declared numerical policy.

## Quality, release and ecosystem

- [x] Source and wheel packaging, distribution checks and third-party attribution.
- [x] Fail-closed configuration and checkpoint/export identity for supported paths.
- [x] Reproducible benchmark methodology that separates throughput, correctness and quality claims.
- [ ] Improve repeated greedy decode reproducibility under default inference settings;
  the documented deterministic Qwen3-8B DSpark evaluation already matches Target-only output.
- [ ] Broaden long-run model-quality and reliability evidence across supported recipes.
- [ ] Add faithful consumers for more inference engines without turning ASpec into a serving scheduler.

New features need appropriate training, recovery, export and numerical validation. Lower-precision
results must disclose the precision policy and quality impact. Weaker supervision or changed masks
do not count as performance optimizations.

See [contributing](CONTRIBUTING.md) and [engineering priorities](docs/framework/ROADMAP.md)
for development and validation guidance.
