# Training recipes

The authoritative workflow/resource map is [tools/recipe_catalog.json](../../tools/recipe_catalog.json).
The [recipe guide](../../docs/user-guide/recipes.md) explains inputs, resources and the two-stage workflow.
[Validation commands](../../docs/contributing/validation.md) keep short-run overrides separate from
training defaults. EAGLE4's candidate-stage config requires a trained first-stage Draft.

The optional validation command exports and reloads a native asset. `native_only_export` marks workflows
whose supported export boundary is native-only; `false` does not disable the native roundtrip.

Use each recipe's documented inputs and check capacity on your own hardware.
The [shared Target walkthrough](../../examples/qwen3-8b-shared-target/README.md) uses
the recipe for experiment comparison, separate curves and recovery from different checkpoints.
Supplemental variants belong under `configs/examples/`; benchmark comparisons remain under
`configs/benchmarks/`.
