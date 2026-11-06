# Ling Target recipes

These recipes use the pinned official SGLang runtime and [training patch](installation.md).
Use the original model tokenizer/template and a reviewed local checkpoint; do not substitute Qwen
tokens or feature-layer offsets. Model/dataset licenses are independent of ASpec.

| Target | YAML | Starting allocation | Draft |
| --- | --- | --- | --- |
| Ling-3.0-tiny | `configs/recipes/ling3_tiny_dspark.yaml` | 1 Target + 1 Draft GPU | Two-layer Dense DSpark |
| Ling-3.0-flash BF16 | `configs/recipes/ling3_flash_dspark.yaml` | TP2 Target + 1 Draft GPU | Five-layer Dense DSpark |
| Ling-3.1-flash FP8 | `configs/recipes/ling31_flash_mla_moe_dspark.yaml` | TP4/EP4 Target + 4 Draft GPUs | Five-layer BF16 materialized MLA + Ling MoE |

These are capacity starting points, not optimal-role or quality claims. Tiny uses capture layers
`[1,6,11,16,21]`; flash uses `[1,11,23,29,35]`; 3.1 flash uses `[7,23,39,55,71]`.
Ling captures residuals after the named layers, without Qwen's extra layer offset. Teacher output
is post-final-norm. Frozen bindings are `model.word_embeddings.weight`, `lm_head.weight` and
`model.norm.weight`. Tiny uses gMASK 156894; flash recipes use PAD 156892.

The 3.1 FP8 Target is about 574 GB on disk. The four-GPU Target allocation has tight memory headroom
on 140-GiB devices: the recipe limits each inference batch to one request, reserves four source slots,
waits for four samples before packed dispatch, and sets an explicit backend token budget. A larger
packing threshold would wait for more live samples than the ring can retain and is rejected by Plan.
Its 0.98 static memory fraction requires exclusive GPUs;
measure actual fit on your hardware.
First-use DeepGEMM compilation can occur during the first prefill, so
`dataset.eval_cache_idle_timeout_seconds: 1200` also covers that cold eval-cache generation. Other
recipes retain the 300-second default, and producer failures still interrupt the wait immediately.
Use a writable node-local compilation cache through `ASPEC_CACHE_HOME`; later runs reuse it.
The recipe launcher also defaults `SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1` for this recipe, using SGLang's
supported smaller precompile shape set while keeping FP8 kernels and on-demand compilation enabled.
An explicit caller value takes precedence. When invoking the YAML directly with `aspec`, set
that environment variable yourself if the same startup policy is desired.

Cold loading of the sharded Flash checkpoints can exceed the generic 300-second initialization
budget. Both Flash recipes and the multi-structure, CSA, QSA and 128K SWA examples set
`inference.sglang.init_timeout: 1200` to allow time for actor initialization.

Run a Dense recipe with your Target and conversations:

```bash
export ASPEC_TARGET_MODEL=/models/Ling-3.0-flash
export ASPEC_TRAIN_DATA=/data/ling-train.jsonl
export ASPEC_EVAL_DATA=/data/ling-eval.jsonl
aspec --config configs/recipes/ling3_flash_dspark.yaml --dry-run \
  model.target_model_path=/path/to/Ling-3.0-flash
RAY_ADDRESS=local aspec --config configs/recipes/ling3_flash_dspark.yaml \
  model.target_model_path=/path/to/Ling-3.0-flash \
  dataset.train_data_path=/path/to/conversations.jsonl output_dir=outputs/ling-flash
```

The `ling3` parser uses the checkpoint Jinja template and retains reasoning/tool messages. Assistant
supervision runs from the assistant-role marker through role end. Qwen-specific direct-prefill/shared
workspace is disabled for these Ling paths. The formal recipes require explicit train/eval inputs; bundled data belongs only to Quickstart.

## Ling-3.1-flash original-answer data

This optional workflow selects the same original conversations by stable source ID, then tokenizes
them with Ling. It does not decode/reuse Qwen tokens or generate new answers. Supply a selection
Parquet containing `source_metadata_json.source_id`, and its original JSONL source:

```bash
python -m tools.prepare_selected_answers \
  --selection /path/to/selection.parquet --source /path/to/original-conversations.jsonl \
  --target /path/to/Ling-3.1-flash --template ling3 --output data/ling31-original-answers
export ASPEC_TRAIN_DATA="$PWD/data/ling31-original-answers/documents.parquet"
export ASPEC_EVAL_DATA=/data/ling31-held-out/documents.parquet
aspec --config configs/recipes/ling31_flash_mla_moe_dspark.yaml --dry-run \
  model.target_model_path=/path/to/Ling-3.1-flash
```

Missing source IDs fail instead of being guessed from row numbers. Oversized and insufficiently
supervised pairs are filtered and counted; manifests record tokenizer, content and filtering identity.
Prepare held-out data independently. This selection workflow is optional; the recipe also accepts
properly prepared compatible data supplied by the user.

## Recovery and assets

Use [exact recovery](training.md) with the same model/data/routing and LR horizon, a complete
checkpoint and a new output directory. Ling MoE router bias/control state is part of recovery.

Dense Ling DSpark uses faithful SGLang-format export with the correct embedding key. Qwen/Ling
MoE export declares the `dflash_moe_v1` loader requirement and the verified `moe_ep_size=1` boundary;
it is not an expert-parallel serving promise. MLA + MoE requires `--format aspec` and strict
native loading. Target hybrid KDA/MLA does not make every Draft structure or Target cache compatible.

Short train/resume/export checks establish functionality. They do not establish convergence,
acceptance or decode acceleration. See [support](../support.md) and [structure choices](structures.md).
