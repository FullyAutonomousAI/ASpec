# Configuration catalog

Start with the [training recipes](../docs/user-guide/recipes.md). In the active ASpec venv,
use `aspec recipes list`, then `aspec recipes plan RECIPE` to inspect a configuration.
These commands work with both a source installation and a release wheel.

| Directory | Purpose |
| --- | --- |
| [recipes/](recipes/README.md) | Training starting points; EAGLE4 has two stage configs |
| [examples/](examples/README.md) | Additional structures, algorithms and topology examples |
| `benchmarks/` | Comparison workloads and upstream baselines, not training defaults |

Training recipes require absolute `ASPEC_TRAIN_DATA` and `ASPEC_EVAL_DATA` paths.
Quickstart alone uses bundled synthetic conversations. `ASPEC_TARGET_MODEL` optionally selects a
local copy of the declared model; Qwen3.8-27B and multi-node training require this input explicitly.
Multi-node training requires the same absolute local model path on every node. It is not a generic model
substitution switch: each recipe binds architecture, capture layers, vocabulary and native context.
EAGLE4 stage two also requires `ASPEC_INITIAL_DRAFT` from stage one's native export.

YAML data paths resolve relative to the YAML file; CLI data overrides resolve from the working directory.
Model/Draft/output/cache paths resolve from the working directory. Hub IDs remain IDs.
Required environment interpolations must exist before config loading, including when using CLI overrides.

Except for Quickstart, recipes default to three dataset passes, held-out evaluation every 100
steps and checkpoints every 100 steps plus the final checkpoint. These are editable training starting
points, not tuned optimal hyperparameters. The validation launcher overrides the
step window and accumulation without editing those files. Exact resume keeps the full LR horizon fixed.

Use `plan` to check configuration and resource requirements before reserving GPUs.
Memory fit and transport are checked in the runtime environment; training quality and decoding
use their own evaluation workflows. See [runtime validation](../docs/contributing/validation.md).
