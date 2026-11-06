# Training benchmark methodology

Measure the complete normal training workflow with disclosed, reproducible inputs and effective
configuration. The public online DSpark matrix compares TorchSpec, SpecForge and AngelSpec with
compatibility patches for the tested configuration. Each comparison,
including the separate DeepSpec workflow, identifies its revisions, configuration and timing boundary.

## Comparable work and normal enablement

Match GPU model/count, model, data/length distribution, objective, precision and statistics. Hosts
may differ; report relevant hardware, driver, CUDA, package/source versions and configuration
differences. Identical configuration text alone does not establish fairness.

Enable normal NVLink and sufficient ready/source/destination/host buffers on both sides. Small
Drafts that fit use REPLICATE unless sharding is justified by capacity. Correct buffer starvation,
invalid loss and inactive requested capabilities before interpreting performance.

Each framework can use its own valid Target/Draft balance and implementation. Choose role splits
from observed bottlenecks, with bounded evidence-based neighboring checks. Do not add an unbounded
CPU/kernel/concurrency/LR sweep to manufacture one result. Disclose native thread defaults.

The public DSpark framework headline uses each baseline's native MBS1 path and
ASpec's padded-document BSHD MBS1 path. Packed-token THD is reported separately
as a layout/capacity choice because its MBS counts packed rows rather than source
documents. It is not selected opportunistically per row to maximize the headline.

## Metrics

The training throughput metric is committed work / wall time over a declared continuous
optimizer-step window. The public online DSpark matrix starts after warmup update eight and ends
at final optimizer completion, retaining the finite tail and excluding cold compilation and initial
pipeline fill. It separately reports complete-loop rates including those startup costs. Other
studies must likewise state their warmup, startup/JIT and tail boundaries; never mix those
boundaries under one speedup label. Formal cross-framework comparisons use
independent continuous wall clocks and actual tokens committed in each window. Native per-step
timer sums and corpus-mean token estimates remain reference diagnostics, excluded from those ratios.

Report original documents, input tokens, effective supervised positions, sampled anchors, physical
rows, padding/fill, finite-source drops and optimizer steps separately. State whether anchors are
budgeted per original document or per physical row; do not compare the two policies as equal work.
Likewise, BSHD MBS counts padded documents while THD MBS counts packed rows and cannot establish a
logical-batch match by itself. Input-token capacity is not equal convergence. Validate actual
admitted/committed counters instead of assuming configured rows equal documents.

Do not substitute median-step extrapolation, burst throughput, prefilled backlog, instantaneous
producer rate or trainer-only compute for the cumulative declared window. End-to-end service rates
are backpressure-coupled; isolated capacity requires a separate complete profile.

## Validity and reporting

For a useful comparison, check and record:

- Actual required capabilities active, with fallback, omission and failures recorded.
- Unique committed work, explicit tail/drop accounting and terminal ownership.
- Finite loss, gradient norms, rates and wall time.
- Resource cleanup and material capacity issues such as OOM or repeated allocator retries.
- Exact command, input/source/config identity, environment, window and repetition protocol.

Publish parity or a limitation when normal, correct enablement has no advantage. Keep diagnostic
logs with the run and disclose issues that affect the comparison; publishing every intermediate
attempt or an artifact-by-artifact audit is not required. Do not use unhealthy or unmatched
denominators. Training throughput, capacity, lifecycle correctness, held-out learning and actual
decode equality/acceptance/timing are separate claims.

The [benchmark guide](../user-guide/benchmarks.md) describes recorded comparisons. Reproduction
tools and configuration inputs are indexed in [benchmark tools](README.md).

## Implementation checks

- Check actual concurrent Target execution, not just actor count: a prompt-buffer watermark below
  the engine window can serialize supply despite multiple active Target actors.
- Validate the longest real sample before a full long-context run. Global kernel pointer offsets
  must widen before stride multiplication; unchanged local tile indices do not solve int32 overflow.
- Count replicated CP source documents, input tokens, physical tokens and global anchor budgets
  once per DP group. Sum only disjoint CP prediction work and actual kernel calls across CP ranks.
- Use CPU-side validated resident-feature metadata for token/document counting. Avoid adding a
  GPU reduction and host synchronization solely for a per-microbatch observer counter.
- Preserve the backend's native Tensor/auxiliary-output return contract. Observation must not
  change training math, kernel selection, learning rate or optimizer behavior.
