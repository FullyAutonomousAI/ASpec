# Configuration and planning

Training recipes under `configs/recipes/` require absolute `ASPEC_TRAIN_DATA` and
`ASPEC_EVAL_DATA` inputs before loading (Quickstart is the exception). See the
[recipe launcher](recipes.md) and [runtime validation](../contributing/validation.md).

Use YAML plus `section.field=value` overrides, validated at the same boundary:

```bash
aspec --config configs/recipes/quickstart_qwen3_8b.yaml --print-config-only
aspec --config configs/recipes/quickstart_qwen3_8b.yaml --dry-run \
  model.target_model_path=/path/to/Qwen3-8B output_dir=outputs/my-run
```

| Section | Responsibility |
| --- | --- |
| `model` | Target/Draft identity, initial weights, frozen tensor bindings |
| `dataset` | Train/eval data, template, supervision and source order |
| `training` | Objective, backend, GPU roles, CP, batch, optimizer, LR and checkpoints |
| `inference` | Target replicas/TP, capture, concurrency and ready pool |
| `mooncake` | Transport, required NVLink, GPU source/destination and host capacities |
| `logging` / `debug` | Tracking and explicitly requested diagnostics |
| `experiments` | Independent Draft overrides with a compatible shared Target |

Complete fields are in [train_config.py](../../aspec/config/train_config.py). Prefer a
[tested recipe](recipes.md); a field existing does not mean every combination is supported.

`model.trust_remote_code` defaults to `false` and governs every Transformers target-config and
tokenizer load, including automatic Draft-config generation. Enable it only for a reviewed model
repository that requires custom code; ASpec does not silently override an explicit opt-out.

## Paths and batching

- YAML local data paths resolve relative to the YAML file; data overrides resolve from working directory.
- Output/cache and local model/config paths resolve from working directory; Hub identifiers remain IDs.
- Source tools and guide commands assume repository root. Supply all required model/data/cluster inputs from the configuration catalog.

Without packing, GBS is `DP × micro_batch_size × draft_accumulation_steps`. CP cooperates on one
context; do not multiply CP into independent samples. Packed batches count rows, not source documents.
Padded-document batches use their actual longest input length, without a fixed 128-token
bucket. DSpark/DFlash use the input validity mask to keep anchor candidates within each
document's native unpadded range; adding batch padding must not expand that range.
Backend-specific CP partitioning remains separate from physical batch padding.
With `training.streaming_dispatch_with_training=true`, a complete final optimizer batch can also
be consumed incrementally instead of requiring the entire batch to fit in source/ready buffers.
This does not permit partial optimizer updates; the Plan's finite-source tail policy still applies.

For non-packed microbatches with more than one document, the dispatcher groups
similar lengths within each already-selected dispatch before assigning complete
microbatches to DP ranks. Assignment balances accumulated padded-context work
within the current optimizer step and resets at its boundary. Node-local feature
affinity and producer-aware source selection still apply. No full-GBS buffer,
additional ready threshold or new option is required; this does not sort the
entire GBS or epoch. MBS1 assignment and the THD row packer retain their paths.
The final controller status reports `batching_dispatched_input_tokens`,
`batching_dispatched_context_tokens`, `batching_dispatched_microbatches` and
`batching_context_efficiency`. These count dispatched BSHD shapes from metadata,
once per logical DP rank; they exclude eval and are not committed throughput.
Regrouping preserves document budgets and pooled loss normalization. Native
random anchor draws can change with rank/batch assignment; debug deterministic
sampling is not an invariant per-document replay across different groupings.
Batch-dependent router/indexer auxiliary statistics can likewise change with
composition; their native objective and reduction remain in effect.

For DSpark/DFlash, both FA4 and cuDNN consume packed documents without document or row-tail padding.
When packing is enabled, full and sliding attention use native flattened-token execution with either
backend at CP1 and above; cuDNN keeps K/V context-sharded and uses interval metadata for
document/window rules. Selected-entry CSA/QSA/DSA packing and CP remain FA4-only.

Small Drafts default to `REPLICATE`; sharding needs a demonstrated capacity reason. Target TP affects
replica count, not just speed. Ready pool, in-flight requests, source ring, destination ring and host
buffer are separate limits; increasing one does not automatically enlarge the others.
For strict GPU TransferEngine output, native Target capture and the optimized direct-prefill path must retain enough samples
for a complete Draft dispatch and the configured packing supply threshold. Plan rejects an
unreachable threshold; increasing a host/ready pool does not add GPU source slots. Node-local
feature domains apply this capacity check separately per node.

`dataset.eval_cache_idle_timeout_seconds` bounds waiting without eval-cache progress. It defaults
to 300 seconds and must be finite and positive. The large Ling 3.1 recipe explicitly allows 1200
seconds for first-use FP8 kernel compilation. Producer failures still interrupt this wait; a larger
cold-start budget does not suppress backend errors. Backend `init_timeout` separately bounds parallel
actor/model initialization.

## Plan versus Evidence

Plan describes requested/resolved execution. Evidence records actual capabilities, fallback/omission,
committed work, metrics and cleanup. Dry-run does not certify hardware/transport. Supported static
multi-node setups offer `--preflight-only` to check cluster placement and transport before training.
It does not exercise node-loss recovery.

The [topology planner](topology.md) selects within expert-approved
capability bundles; it does not invent algorithms or bypass required capabilities.

### WSD schedule

Set `training.lr_decay_style=WSD`. The shared defaults are `training.lr_wsd_decay_style=cosine` and `training.lr_wsd_decay_ratio=0.2`. DSpark/DFlash also accept the legacy `training.wsd_decay_style` and `training.wsd_decay_ratio` overrides; an omitted legacy value inherits the shared setting. Invalid styles and ratios fail during configuration loading. Short runs retain at least one decay step.

All schedules require positive configured step horizons, `min_lr` in `[0, learning_rate]`, and
`warmup_ratio` in `[0, 1)`. Invalid values fail during configuration loading rather than after actor
or optimizer initialization.

`--dry-run` checks that explicitly local training and evaluation dataset paths exist. Hub dataset IDs remain unresolved during planning; their availability and contents are checked when the data is loaded.

### Draft activation and RoPE

Dense DSpark/DFlash FFNs execute the configured `hidden_act` through Transformers' activation registry; the default remains SiLU. Their standard GQA path supports unscaled, linear and YaRN RoPE through `rope_scaling` or `rope_parameters`, using the pinned Transformers scaling functions. YaRN uses `original_max_position_embeddings` when provided and otherwise the configured maximum context. Unsupported GQA scaling types and partial rotary dimensions fail during configuration loading. MLA and compressed/indexer rotary contracts remain structure-specific.

### FlexAttention kernel selection

The shared `flex_attention` wrapper defaults to the standard Triton kernel
(`BACKEND=TRITON`). This avoids PyTorch 2.13's automatic decoding selection failing
with `NoValidChoicesError` for short GQA training rows. Explicit `BACKEND` choices
and the legacy `FORCE_USE_FLEX_ATTENTION` selector are preserved. GPU regressions
compare outputs and Q/K/V gradients with SDPA, including nested `torch.compile`.
