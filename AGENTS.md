# Working on ASpec

Read [README.md](README.md), [architecture](docs/framework/ARCHITECTURE.md),
[current decisions](docs/framework/DECISIONS.md), the relevant code/tests and the task's acceptance
criteria. [CONTRIBUTING.md](CONTRIBUTING.md) describes the environment and checks. Historical
experiment outputs and machine-specific records are intentionally outside the public source tree.

## Scope and invariants

The primary performance path is SGLang + DSpark/DFlash + Qwen3 on multi-GPU systems. Python
namespace and CLI use `aspec`; source release does not imply PyPI publication. Other implemented
paths have explicit [support boundaries](docs/support.md). Avoid speculative extensions.

Preserve these invariants:

1. Express workload semantics, available capability, selected Plan and observed Evidence separately.
2. Convert supported legacy input at the boundary; do not carry legacy fields into the core runtime.
3. Fail closed for illegal combinations or unavailable required capabilities. Record fallback/omission.
4. Preserve stable work identity: committed work is neither lost nor repeated across async completion,
   packing, regrouping and recovery. Finite tails use explicit drop-last.
5. Track one current owner per storage/lease and observable terminal states after success/failure/cancel.
6. Plan device model memory, backend token capacity, transfer/storage buffers and host memory separately.
7. Match claims to evidence: throughput, capacity, numerical correctness, lifecycle and quality differ.

## Implementation and attribution

Review the pinned TorchSpec implementation, tests and dependencies before changing overlapping
capabilities. Adapt working code by default. Reimplement only with a concrete reason and a verifiable
comparison. [Reuse boundaries](docs/framework/RETENTION_BOUNDARY.md) and source headers identify origins.

Classify upstream use as copied/adapted/reimplemented/inspired. Copied/adapted content retains applicable
copyright, license and notices, including transitive TorchSpec → AngelSpec ancestry. State sources and
main changes in headers or adjacent documentation and distribute required texts under `licenses/`.
Moving files does not remove attribution obligations.

Use the smallest implementation that protects current invariants. Component names express
responsibilities, not mandatory classes, processes or directory layouts. Keep one runnable vertical
path; avoid parallel unconnected skeletons or abstractions for hypothetical extensions.

## Changes and verification

- Resolve uncertainties from code, current docs and the actual environment before implementation.
- Make reasonable reversible choices autonomously; ask only when missing facts affect correctness,
  compatibility, task scope or irreversible actions. Preserve existing authorization.
- Keep unrelated work intact, including concurrent experiments. Use an isolated checkout when needed;
  verify starting revision/changes before transferring patches. Never overwrite an entire shared tree.
- Match tests to risk. Reproduce behavioral bugs first where practical; documentation changes do not
  need artificial tests. Run relevant checks, then broaden only for unresolved risk or failures.
- Update public guides/configs when public behavior changes. Describe outcome, actual checks and
  unverified scope; preserve failed gates instead of redefining them to pass.
- Performance work follows [the measurement contract](docs/benchmarks/methodology.md). Disclose actual
  hardware/software/workload/config/window identity; hostnames may differ for comparable GPU models.

Use an isolated Linux venv for Python installation, testing and training. Follow the task's selected
host, checkout and GPU allocation; no hostname or mount path in an old report is a connection target.
Do not use system-site-packages or import another checkout through PYTHONPATH. Verify console script,
module and worker origins. Transfer only task-related changes and verify the final bytes match tests.

On shared systems, stop only processes/clusters owned by the current run. Never perform host-wide
Ray/GPU cleanup or change system drivers to make a check pass. Preserve config, Plan/Evidence and
failure logs. Do not publish, push, upload tracking data or contact others unless authorized.
