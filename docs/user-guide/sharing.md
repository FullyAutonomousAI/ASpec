# Share a Target across Drafts

Shared mode captures compatible Target features once and trains independent Draft experiments.
Each experiment owns its weights, optimizer, LR, RNG, evaluation, work frontier, checkpoint and
tracking output. Shared execution is currently single-node.

Use this workflow to compare Draft algorithms, structures or hyperparameters for the same Target.
Algorithm selection chooses what each experiment trains; shared execution lets several such
experiments run concurrently with compatible Target capture generated once. Each keeps its own
training state and progress rather than participating in one Draft's data-parallel optimizer step.

| Example | Purpose | Starting GPU budget |
| --- | --- | --- |
| `configs/recipes/qwen3_8b_4k_multi_draft_nvlink.yaml` | Two independent DSpark experiments | 4 Target + 4 Draft |
| `configs/examples/qwen3_8b_multi_algorithm_feature_views.yaml` | Independent DSpark/EAGLE3/DFlash feature views and packing/CP | 1 Target + 7 Draft |
| `configs/examples/ling3_flash_multi_structure_dspark.yaml` | Dense, Qwen MoE and Ling MoE Drafts | TP2 Target + 3 Draft |
| `configs/examples/ling3_flash_swa_compare_dspark.yaml` | Full context versus SWA | TP2 Target + 2 Draft |

Supply training and held-out evaluation data and inspect all child Plans before starting. The Qwen 4K
shared template uses `Qwen/Qwen3-8B` by default; set `ASPEC_TARGET_MODEL` to use a local copy,
and provide `ASPEC_TRAIN_DATA` and `ASPEC_EVAL_DATA`. The Ling functional example below uses
bundled smoke data:

```bash
aspec --config configs/examples/ling3_flash_multi_structure_dspark.yaml --dry-run \
  model.target_model_path=/path/to/Ling-3.0-flash
RAY_ADDRESS=local aspec --config configs/examples/ling3_flash_multi_structure_dspark.yaml \
  model.target_model_path=/path/to/Ling-3.0-flash
aspec --report outputs/ling3-flash-multi-structure
aspec --report outputs/ling3-flash-multi-structure/dense
```

`experiments` entries override Draft-specific model/training/logging settings. Feature sharing checks
actual Target, normalization and dtype, not architecture names. Capture can take a compatible layer
union while each consumer reads its ordered view. A receiver may need a compact tensor copy;
shared capture is not a blanket zero-copy claim. Native MTP's post-norm feature cannot substitute
for a regular auxiliary layer.

The Qwen algorithm example covers both shared capture and independent feature views. To compare
identical views, select the full-layer EAGLE3 and DFlash model configs in its experiment entries and
set DFlash's target-layer count to five. Adjust per-child packing and CP in those same entries;
each child Plan still validates its own execution requirements.

## Progress, resources and failure

Experiments progress independently without a step barrier. Slow consumers apply bounded
backpressure; completed, cancelled or failed readers have their rights fenced before reclamation.
Safe healthy siblings can continue, but a failed/cancelled experiment makes the session exit nonzero.
Reports distinguish shared source counts and each experiment's committed work. Do not add child
consumption counts and label the sum unique Target production.

Check each child's required capabilities, fallback, lifecycle and evaluation. Per-child throughput
is coupled to shared supply and backpressure, not isolated compute capacity. Root wall time and GPU
budget describe the whole session; different objectives/auxiliary losses do not form a quality ranking.

## Resume and tracking

Copy the original saved `config.yaml` outside the run directory and edit each existing
`experiments[].training.load_path` to that child's complete checkpoint. Keep the full experiment
list, model/data/ordering, LR horizon and execution semantics. Current dotlist overrides do not
support patching an individual list index; replacing the list with partial entries can lose fields.

Set `training.continual_training=false`, keep the full `lr_total_steps`, choose a fresh root output
directory, and set `num_train_steps` to the intended total frontier. Children may resume from different
committed frontiers. One child's progress never licenses skipping another child's work.

TensorBoard and W&B use separate runs/writers per experiment. See
[training and tracking](training.md#tensorboard-and-wb). Online weight synchronization, multi-node
shared execution and arbitrary multi-tenant isolation are outside the current contract.
