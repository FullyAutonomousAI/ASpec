# Draft structures and execution

Choose structure independently from algorithm and Target architecture. The JSON model configs in
`aspec/config/` define geometry, FFN/routing, attention policy, feature layers and asset identity.
The YAML recipes define resources and execution. See [support boundaries](../support.md).

## Dense, MoE and MLA

- `configs/examples/qwen3_8b_moe_dspark.yaml` selects Qwen-style routed experts.
- `configs/examples/ling3_flash_multi_structure_dspark.yaml` compares Dense, Qwen MoE and Ling MoE
  Drafts. Ling MoE uses grouped routing, a shared expert and FP32 router bias/control state;
  its auxiliary training differs from Qwen load balancing. For one Ling MoE Draft, use the
  `ling-flash` recipe with `model.draft_model_config=aspec/config/dspark_ling3_flash_moe_draft_config.json`.
- `configs/examples/qwen3_8b_mla_dspark.yaml` uses materialized MLA with a Qwen GQA Target. This does not
  make the Target MLA or provide a compressed-latent inference kernel.
- `configs/recipes/ling31_flash_mla_moe_dspark.yaml` combines materialized MLA and Ling MoE;
  [the Ling guide](ling.md) specifies its bindings.

For EAGLE3 MLA, use the `eagle3` recipe with
`model.draft_model_config=aspec/config/eagle3_qwen3_8b_mla_draft_config.json`,
`training.attention_backend=fa4`, `training.eagle3_target_cache=lazy` and
`training.eagle3_loss_gradient_checkpointing=true`.

REPLICATE is the normal strategy for a Draft that fits on each GPU. Count frozen embeddings/head,
optimizer/FP32 master, gradients, activations and transfer buffers when checking capacity. Exact
resume/export preserves expert count, top-k, normalization, auxiliary policy and control state;
missing or unverifiable routing state is not silently accepted.

## Packing and context parallelism

ASpec supports two dense attention layouts. The padded-document layout keeps one
original document in each batch element. The packed-token layout flattens one or
more complete documents into a row while preserving their attention and position
boundaries. CP splits one context across cooperating GPUs; it does not multiply
the number of independent samples.

Choose the padded-document layout for a direct MBS1 comparison or when each
microbatch already contains one document. Choose packed tokens when a physical
row should carry several documents, especially when eliminating multi-document
padding improves useful-token density or makes the intended resident batch fit.
THD MBS1 is one packed row, whereas BSHD MBS1 is one document; neither is an
unconditional speed or absolute-memory guarantee. In a controlled Qwen3-8B/4K
four-document resident comparison, packed tokens reduced Draft peak allocation
by 15.7%, while natural one-document MBS1 runs can have a lower absolute peak. The
[final benchmark](../benchmarks/results.md#aspec-layout-reference) keeps these
layout observations separate from its BSHD MBS1 framework headline.

The normal padded-document path uses `training.dflash_packing=false` and
`training.micro_batch_size=1` for MBS1.

To enable packed execution, set:

```bash
aspec --config configs/examples/qwen3_8b_moe_dspark.yaml --dry-run \
  training.attention_backend=fa4 training.dflash_packing=true \
  training.draft_context_parallel_size=4 training.training_num_gpus_per_node=4
```

Allocate Target GPUs as well. Check Plan for actual DP, CP, head layout, accumulation and buffers.
FA4 uses fixed token-major execution for full attention and selected-key variable-length execution
for sparse attention. With packing enabled, cuDNN provides native flattened-token execution for
DSpark/DFlash full and sliding attention at CP1 and above, while retaining context-sharded K/V and
exact LSE merging.
Plan/Evidence distinguish the effective backend and kernel mode. Selected-entry CSA/QSA/DSA CP
remains FA4-only. Legal degrees follow resources, heads, anchors and shard alignment;
representative CP1/2/4 tests are not a whitelist.

The [cuDNN packed-attention record](../benchmarks/cudnn-thd-cp.md) states the bounded correctness and
component-performance evidence. It is not an end-to-end training or model-quality claim.

On SM90, cuDNN THD can fold GQA query-head groups into the token axis while keeping K/V heads
unchanged. The automatic choice uses the actual per-rank attention-call shapes, including gathered
queries: for group ratio `R = Hq/Hkv > 1`, folding requires the selected KV length to cover `R × Q`.
Window limits bound the context contribution to that length. This is a conservative execution
heuristic, independent of model names, configured context presets and CP degree.
`gqa_layout`, `gqa_selection_key_tokens` and `physical_query_tokens` record the choice.
Output and differentiable LSE return to their logical layouts before CP merging. Derived metadata
and plans are reused across layers within one forward; they are not retained across updates.
No additional configuration or checkpoint state is required. See the
[layout comparison](../benchmarks/results.md#aspec-layout-reference).

Each CP consumer needs source descriptors. If dispatch waits for `N` documents, provision at least
the corresponding `N × CP` descriptor capacity and sufficient bytes. The planner rejects impossible
minimum-dispatch/resident-capacity combinations. Ready pools and source/destination rings remain
distinct limits. Opportunity-based grouping may differ across runs, so independent CP trajectories
need not have identical stepwise losses even with the same seed.

Finite data executes complete configured optimizer steps only. Plan records requested, scheduled,
alignment and drop policy; runtime Evidence records effective documents and dropped tail. Input
documents, packed rows, input tokens and sampled supervision are separate counters.

### Independent sample anchor budgets

DSpark/DFlash also support `training.anchor_sampling_scope=sample`. With packing, each original
document independently receives up to `dspark_num_anchors` / `dflash_num_anchors` candidate anchors;
documents no longer share one row-level budget. The default `row` preserves existing recipes.
Without packing the two scopes have the same per-sample budget. DFlash2 inherits this sampling
policy but retains its experimental training boundary.

Sampling preserves within-document candidates and spacing, never repeats anchors to fill a short
document and never crosses a document boundary for labels or attention. Structurally unavailable
slots and CP alignment padding remain masked. Loss normalization still pools valid weighted
supervision across documents, accumulation and ranks; this is not equal-document loss weighting.

For Dense full/SWA models, ASpec may skip whole masked query blocks inside their original CP
owners and evaluate the vocabulary head/loss in bounded chunks. MoE and indexer objectives retain
their complete query population. Projection and loss replay caches are bounded, forward-local and
released with the autograd graph; they do not change anchor selection, CP ownership, original head
chunk boundaries, pooled normalization, precision or learning rate. Unsupported layouts retain the
ordinary path.

Native cuDNN padded batches also reuse Dense full/SWA backbone selection and
restore original slots before the head. Single-chunk CP1 batches can reuse a
frozen Target context projection when it removes repeated prediction-position
projections. The complete `B × context_tokens` allocation must fit a chunk-derived
bound without multiplying that allowance by MBS. This adds no checkpoint replay
or extra Draft projection cache; long/unprofitable contexts and trainable Target
heads/features retain their existing execution. The newer packed projection-budget
filling and L1-statistics replay optimizations remain enabled on their own paths.

The scope is recorded in Plan and checkpoints. Exact resume cannot switch scopes; old checkpoints
mean `row`. Use weights-only initialization for a new policy. Anchor budget does not redefine GBS:
packed GBS still counts physical rows. Report original documents, actual supervision and per-document
anchor budgets separately when comparing a packed run with an unpacked baseline.

## Sliding-window attention

A positive `sliding_window` enables a window; optional `layer_types` chooses full/sliding layers and
must match model depth. In DSpark, the window covers context before the anchor and the Draft block
retains its own visibility; it is not a generic per-query causal decoder window. DFlash2 causal/SWA
has its own query-relative rules.

`configs/examples/ling3_flash_swa_compare_dspark.yaml` compares independent full/SWA Drafts under shared
capture. `configs/examples/ling3_flash_swa_128k_packing_cp4.yaml` illustrates a long packed CP path; it
requires explicitly prepared long data, adequate Target prefill capacity and source-ring bytes.
Short bundled conversations do not exercise 128K context.
For a Ling MoE/SWA Draft, use the `ling-flash` recipe with
`model.draft_model_config=aspec/config/dspark_ling3_flash_moe_swa_draft_config.json`.

## CSA and QSA stages

Both structures use **main dense → indexer warmup → sparse** phases. They are dependent phases,
not three concurrent random initializations. CSA uses ratio-4 overlapping compression and an
independent Lightning Indexer. QSA uses ratio-4 compressed MQA indexing, a 2,048-original-token
selection budget and gathered original K/V with FA4 varlen sparse execution. These architectures,
checkpoints and indexers are not interchangeable.

Start `configs/examples/ling3_flash_csa_dspark.yaml` or `configs/examples/ling3_flash_qsa_dspark.yaml`. After a complete
dense checkpoint, switch model JSON and start a new weights-only phase:

```bash
RAY_ADDRESS=local aspec --config configs/examples/ling3_flash_csa_dspark.yaml \
  model.target_model_path=/path/to/Ling-3.0-flash \
  model.draft_model_config=aspec/config/dspark_ling3_flash_csa_indexer_draft_config.json \
  training.continual_training=true training.continue_source_cursor=true \
  training.load_path=/path/to/dense/checkpoints output_dir=outputs/csa-indexer
```

Then select `dspark_ling3_flash_csa_sparse_draft_config.json` and the indexer checkpoint for the
sparse phase. QSA uses the corresponding `qsa_indexer` and `qsa_sparse` files. Each phase gets a
fresh optimizer/scheduler and output directory. `continue_source_cursor=true` retains the committed
source frontier only; within-stage exact resume uses `continual_training=false` and the same stage.

CSA indexer inputs are detached; QSA retains its own input-gradient policy. Warmup freezes the main model and reports indexer KL rather than
LM/CE/L1/confidence or acceptance. Sparse-stage gradients retain their documented separation.
Packed/CP execution preserves compression/shard alignment; QSA shards must align to ratio 4.
Short-context selection may cover nearly everything while adding indexer overhead. No universal
speedup or Cartesian product of structures is promised.

CSA/QSA and materialized MLA export with `--format aspec`. Strict native loading preserves stage,
compressor/indexer, head, mapping and mixed dtype. See [export](evaluation.md).

## DSA: token-level sparse attention

DSA uses a weighted-ReLU Lightning Indexer to select original context tokens **for each Draft query**.
It neither compresses tokens like CSA/QSA nor shares one selection across a whole Draft block.
The selected original K/V and the complete current bidirectional Draft block feed attention;
context visibility remains strictly before the anchor and inside the same packed document.

The [Qwen3-8B DSA example](../../configs/examples/qwen3_8b_dsa_dspark.yaml) uses the same three dependent
phases: `dense` (indexer frozen), `indexer_warmup` (only indexer trainable), then `sparse` (main model
and indexer trainable; a zero auxiliary coefficient freezes the indexer). The example's 16 steps
and bundled short conversations are a functional
starting point, not a sparse-performance or quality claim. Supply your actual training data.

```bash
# Start the dense phase; replace Target and data with your own paths.
aspec --config configs/examples/qwen3_8b_dsa_dspark.yaml \
  model.target_model_path=/path/to/Qwen3-8B dataset.train_data_path=/path/to/train.parquet

# After a complete dense checkpoint, start a weights-only indexer phase.
aspec --config configs/examples/qwen3_8b_dsa_dspark.yaml \
  model.target_model_path=/path/to/Qwen3-8B dataset.train_data_path=/path/to/train.parquet \
  model.draft_model_config=aspec/config/dspark_qwen3_8b_dsa_indexer_draft_config.json \
  inference.store_last_hidden_states=false \
  training.continual_training=true training.continue_source_cursor=true \
  training.load_path=/path/to/dense/checkpoints output_dir=outputs/qwen3-8b-dsa-indexer
```

Then select `dspark_qwen3_8b_dsa_sparse_draft_config.json` and the complete indexer checkpoint for
the sparse phase, with `inference.store_last_hidden_states=true` for DSpark's original objective.
Within a stage, use exact resume (`continual_training=false`) and a fresh output
directory. Stage, token budget, indexer geometry and gradient policy are checkpoint identity;
`dsa_workspace_bytes` controls chunking and may change across resume without changing the objective.

Following [DeepSeek-V3.2](https://arxiv.org/html/2512.02556v1#S2.SS1), indexer inputs and main-attention
teachers are detached. Warmup matches the normalized head-summed dense attention distribution;
sparse training matches the distribution on the selected context tokens. KL trains only the indexer;
the main model keeps its DSpark/DFlash objective. Frozen main-model state includes MoE router bias.
The GQA Draft owns a low-rank query adapter; this is not a claim to load an official MLA DSA model.
Mixed-precision training does not emulate the official inference implementation's FP8 quantization.

Use FA4 for packed/CP execution; CP geometry follows normal resource constraints, without a
compression-ratio alignment or degree whitelist. Token layouts and RoPE factors are shared across
layers within each forward. Search and selected-set KL use independent bounded chunks; the detached
selected teacher is retained once rather than recomputed during backward. Full-context warmup
scores/teacher and FA4 gathers use bounded chunks and recomputation. CP teacher softmax statistics
are reduced once per bounded chunk across all KV heads, rather than issuing collectives per head;
the implementation retains neither a query-by-full-context backward graph nor a full-context CP
gather. Individual queries
still perform full-context lightweight indexing; DSA is not sublinear token search.
The Indexer uses DeepSeek's non-interleaved rotary layout, distinct from the main MLA rotation.
The workspace setting controls temporary chunks with a one-query minimum, not persistent projections
or total Draft memory; even one query can exceed a very small workspace setting.

DSA supports the DSpark/DFlash GQA/MHA path with native `--format aspec` assets and strict
stage/indexer loading. Whole-model compile, DFlash2, materialized-MLA integration, mixing DSA with
CSA/QSA, and SGLang serving integration are not implemented. Short contexts can be slower; long-context
memory checks and numerical tests do not establish end-to-end speedup or model quality.

The maintained validation boundary includes three-stage real-Target training with a reduced Draft,
exact recovery, packed CP numerical checks and 128K component/asset checks. This is functional and
numerical coverage, not an end-to-end speed, convergence or official-MLA claim. New evidence must
retain its exact source revision, environment and failures outside the source checkout.
