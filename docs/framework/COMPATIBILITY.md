# Compatibility boundaries

ASpec supports its native configuration and a declared subset of the pinned TorchSpec-style
input. `configs/examples/sglang_qwen3_8b_dspark.yaml` is the compatibility example. This is a boundary adapter,
not a promise to accept every upstream option or a rolling upstream `main` schema.

Canonical workload/execution types are created before the core runtime. Required unsupported
combinations fail closed; fallback and omission must appear in Plan/Evidence. File suffixes alone
must not silently reinterpret an incompatible dataset. See [configuration](../user-guide/configuration.md)
and [data](../user-guide/data.md) for native inputs and typed pretokenized conversion.

## Checkpoints and assets

Exact recovery requires a complete checkpoint with compatible model, algorithm/objective, structure,
feature/vocabulary binding, data/progress and execution semantics. Missing optimizer/RNG/work state
must not silently become weights-only initialization. Explicit weights-only phases use
`model.initial_draft_model_path` or `training.continual_training`; see [recovery](../user-guide/training.md).

Export carries the effective structure, stage, mapping, source identity, hashes and required frozen
Target tensors. The selected downstream format must faithfully represent those semantics; otherwise
choose a supported native format or receive an explicit error. A native loader/proposal component
is not a generic serving runtime. See [export](../user-guide/evaluation.md) and [support](../support.md).

Known legacy metadata is normalized only under the corresponding verified algorithm contract.
For example, old native MTP attention labels may be corrected to their actually executed causal
semantics; changing a real objective or binding remains incompatible.

## Source and runtime

Internal TorchSpec Python classes are not a public compatibility API. Retained implementations can
change behind ASpec contracts; preserve their origin and verify behavior. Upstream revisions
are explicit references, not merge/rebase obligations or automatic support for newer profiles.

The current runtime is the [pinned SGLang lane](../user-guide/installation.md). Other upstream backends
are not supported merely because a legacy YAML names them. Model/algorithm/backend combinations
require their own scope and executable evidence.
