# Documentation

ASpec trains speculative decoding Draft models with asynchronous Target features, explicit
execution plans and recoverable training state. Start with a working run, then choose the algorithm,
structure and execution suited to your data and hardware.

## First run

1. [One-command source setup](user-guide/installation.md), or the optional [container](user-guide/containers.md).
2. [Run Qwen3-8B or Ling tiny on two GPUs](user-guide/quickstart.md).
3. [Choose a recipe](user-guide/recipes.md) and [prepare data](user-guide/data.md).
4. Check [supported paths and known limitations](support.md).

The [project roadmap](../ROADMAP.md) lists implemented features and planned extensions.

## Training workflows

| Task | Guide |
| --- | --- |
| Configure resources, batches and Plan | [Configuration](user-guide/configuration.md) |
| Choose DSpark, DFlash, EAGLE or native MTP | [Algorithms](user-guide/algorithms.md) |
| Choose Dense/MoE/MLA, SWA/CSA/QSA or packing/CP | [Structures](user-guide/structures.md) |
| Use Ling Targets | [Ling recipes](user-guide/ling.md) |
| Train independent Drafts with shared capture | [Shared Target](user-guide/sharing.md) |
| Plan and run a static cluster | [Multi-node](user-guide/multinode.md) |
| Select bounded role/buffer topology | [Topology selection](user-guide/topology.md) |
| Train, track, stop and resume; replay offline features | [Training and recovery](user-guide/training.md) |
| Evaluate, select and export checkpoints | [Evaluation and export](user-guide/evaluation.md) |
| Diagnose installation or runtime failures | [Troubleshooting](user-guide/troubleshooting.md) |
| Inspect native prerequisites or manual installation | [Advanced installation](user-guide/installation-advanced.md) |
| Interpret and reproduce performance | [Benchmarks](user-guide/benchmarks.md) / [tools and protocol](benchmarks/README.md) |
| Compare training quality and actual verifier acceptance | [Fixed-sequence Qwen3-8B study](benchmarks/dspark-alignment.md) |
| Configure paths, caches and interfaces | [Environment portability](user-guide/portability.md) |

The [configuration catalog](../configs/README.md) identifies required inputs and GPU budgets.
Bundled short examples are functional recipes, not model-quality promises.

For the engineering approach and measured results, read
[ASpec: accelerating Draft training for speculative decoding (Chinese)](aspec.zh-CN.md).

## Contributing

[CONTRIBUTING.md](../CONTRIBUTING.md) covers development and verification.
[Architecture and decisions](framework/README.md) describe current contracts.
[Engineering priorities](framework/ROADMAP.md), [runtime validation](contributing/validation.md) and
[release preparation](releasing.md) guide contributions and maintenance.
Project prose is maintained primarily in English, with a [Chinese overview](../README_zh-CN.md).
