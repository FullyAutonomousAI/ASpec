# Training performance

ASpec benchmarks report training input throughput and capacity. Held-out Draft
quality and serving speed have separate evaluation protocols.

## Results

| Comparison | Workload and boundary | Results |
| --- | --- | --- |
| DSpark framework comparison | H200, Qwen3-8B 4K/128K and Qwen3.8-27B 4K/256K; per-document anchors; online loop wall time | [Final framework matrix](../benchmarks/results.md#end-to-end-training) |
| Resident DSpark | One GPU, common real features and initial values; excludes Target/transport | [Draft-only results](../benchmarks/results.md#draft-only-training) |
| DeepSpec | 8B 4K/128K; resident Draft training and native offline cache-first workflow | [DeepSpec comparison](../benchmarks/deepspec.md) |
| EAGLE3 | H200, synthetic 4K/16K, single-GPU train core | [Latency and memory](../benchmarks/eagle3-h200.md) |
| Packed attention/CP | Matched attention forward/backward, not full training | [Component results](../benchmarks/cudnn-thd-cp.md) |
| Training quality and acceptance | Qwen3-8B, fixed document/anchor sequence, ten epochs; BSHD and THD/CP1 | [Validation and real verifier metrics](../benchmarks/dspark-alignment.md) |

The final online headline uses the post-warmup steady window. Every framework
uses padded-document MBS1; ASpec's THD path is reported separately as a layout
and capacity option. ASpec measures **1.53–2.28x** TorchSpec, **1.09–2.50x**
SpecForge and **1.65–2.68x** AngelSpec in completed pairs. Higher BSHD
microbatch sizes are not selected opportunistically. Each result
describes its measured configuration and software, not a universal speedup or
quality-equivalence claim.

The 4K corpus uses approximately 60K existing Target-generated answers. Long
inputs use 4,096 frozen LongCite conversations with native tokenizers. Every
original document has an independent 512-anchor cap. H200 AngelSpec 27B/256K
has no complete result.

## Read the metrics

- Compare the same timing boundary: the final online headline uses completion of
  warmup update eight through final optimizer completion. Complete-loop,
  resident Draft-only and offline preparation-plus-training remain separate.
- Match anchor policy: the final matrix assigns every original document an
  independent 512-anchor cap, including documents sharing a packed row.
- Interpret MBS by layout: the headline's BSHD MBS1 is one original document per
  rank and microstep. THD MBS1 is one packed row and can contain multiple
  documents; use it when packing density or resident memory is the governing
  constraint rather than as an unconditional faster default.
- Report actual consumed documents, input tokens, supervision and finite tails.
  A larger input rate alone does not establish faster convergence.
- Each framework uses its disclosed role split, native software stack, normal
  interconnects and sufficient buffers. Small single-run differences do not
  establish statistical significance or a universal optimum.

The [DeepSpec resident comparison](../benchmarks/deepspec.md) measures **9.24× /
3.79×** ASpec/DeepSpec input throughput at 4K/128K, with different packed versus
single-document supervision counts. Its offline workflow is reported separately;
no offline-phase/online-loop E2E multiplier is inferred.

## Training quality and acceptance

The [fixed-sequence Qwen3-8B DSpark study](../benchmarks/dspark-alignment.md) compares TorchSpec,
ASpec BSHD and ASpec THD/CP1 over ten epochs and 1,180 updates, with the same documents and anchors
assigned to every update/rank, initial weights and LR schedule.

| Selected checkpoint | Validation CE | Measured mean accepted length (tokens/round, including Target bonus token) |
| --- | ---: | ---: |
| TorchSpec | 1.560284 | 3.665717 |
| ASpec BSHD | 1.555042 | 3.691665 |
| ASpec THD / CP1 | 1.515060 | 3.746300 |

BSHD stays close to TorchSpec in validation CE. Measured mean accepted length is 0.71% higher for
BSHD and 2.20% higher for THD on these checkpoints. With official SGLang deterministic inference
and normal radix cache, all three match Target-only greedy output token IDs on 256 held-out
questions across three repeats. This is one training sequence and initialization; it does not
establish the same result across seeds, models or CP sizes, or an equal-quality speedup for the
separate throughput matrix.

## Reproduce

Read [methodology](../benchmarks/methodology.md), then use the
[tools and templates](../benchmarks/README.md) with your model, dataset and
initial weights. For checkpoint quality and actual inference acceptance, use
[evaluation](evaluation.md) and the
[fixed-sequence training/acceptance comparison](../benchmarks/dspark-alignment.md)
of TorchSpec and ASpec BSHD/THD. The separate
[CP quality report](../benchmarks/dspark-cp-alignment.md) compares CP2 with the matching
CP1 reference in each ASpec layout. These quality results are independent of the
input-throughput tables above.
