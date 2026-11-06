# DeepSpec DSpark comparison

Qwen3-8B DSpark training on H200, with context parallelism disabled (CP1). Measurements dated
**2026-10-26** cover two boundaries: one-GPU Draft training with resident Target features, and
DeepSpec's native eight-GPU offline cache-and-train workflow.

ASpec's measured resident Draft input throughput is **9.24× at 4K** and **3.79× at 128K**
that of DeepSpec. DeepSpec also completed one native cache-first pass over each full corpus;
those results are reported separately alongside ASpec's online workflow measurements.

## Resident Draft-only results

| Training-row cap | DeepSpec input tokens/s | ASpec input tokens/s | ASpec / DeepSpec |
| --- | ---: | ---: | ---: |
| 4K | 4,672.19 | 43,178.58 | **9.24×** |
| 128K | 110,664.21 | 419,248.62 | **3.79×** |

Each measurement uses 16 warmup updates followed by one continuous 64-update window.
There is one physical row per update. Initialization, feature loading, H2D and warmup compilation
are excluded; forward, backward, optimizer updates and native in-window reporting are included.
These are single-run observations, not repeated-run confidence intervals.

| Case | Processed documents | Input tokens | Supervised predictions | Window seconds |
| --- | ---: | ---: | ---: | ---: |
| DeepSpec / 4K | 64 | 47,937 | 166,985 | 10.2601 |
| ASpec / 4K | 353 | 256,097 | 229,376 | 5.9311 |
| DeepSpec / 128K | 64 | 2,144,028 | 144,382 | 19.3742 |
| ASpec / 128K | 241 | 7,976,260 | 222,509 | 19.0251 |

The banks contain **64 original 4K documents** and **32 original long documents**, cycled during
the control. They come from the same Target-generated 4K corpus and frozen LongCite corpus used
in the framework study; answers are not regenerated. The longest long-bank document is 130,982
tokens. A 128K cap is not a claim that every source document has that length. This is not a
complete-corpus training pass.

Both frameworks retain **512 anchor slots per physical row**—one sequence presented to the Draft,
which may contain several complete documents. ASpec packs documents into an unpadded row;
DeepSpec processes one document per row. Consequently input and supervision counts differ.
The input-throughput ratio is not an equal-supervision, convergence, quality or
decode-speed claim. Packing and the execution backends contribute together; this is not a
single-kernel ablation.

### Resident conditions and native enablement

- Five-layer GQA DSpark, BF16, block size 7, Markov rank 256 and confidence head enabled.
- Common initial Draft tensor values, frozen embeddings and the same Target LM head. Shared
  intermediate features use logical decoder-layer outputs; final Target features are normalized.
- Native CE/L1/confidence objectives and anchor sampling; learning rate `6e-4`, decay gamma 4,
  weights `0.1/0.9/1.0`, maximum gradient norm 1 and a 1,000-update LR horizon.
- One GPU, accumulation 1. DeepSpec uses `no_shard`;
  ASpec uses `REPLICATE`.
- [DeepSpec `005e03b`](https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2),
  pinned from official `main`:
  PyTorch 2.9.1/CUDA 12.8, Transformers 5.10.2, native FlexAttention and default
  `torch_compile=true`. Native logging interval 10 is retained.
- ASpec `7618a01`: PyTorch 2.13/CUDA 13, cuDNN unpadded packed attention (THD) and fused
  Q/K normalization and rotary-position preparation, verified active in both runs.
- Native CPU/thread policies are retained: ASpec's eight-thread default, with no explicit
  OMP/MKL/OpenBLAS overrides for DeepSpec. There is no CPU, tile, concurrency or learning-rate sweep.

All 80 updates per run have finite loss and gradient-norm observations. The paired runs share the
same complete feature-bank identity. DeepSpec replays the recorded TorchSpec resident-control
document order; ASpec uses its production row packer with a 32-document packing pool.
Shared initial tensor values were checked after BF16 conversion.

### Resident adapter and reproduction

The [DeepSpec adapter](../../tools/performance/deepspec_benchmark.py) calls its native trainer,
sampler, losses, backward, clipping and optimizer. The input adapter renames three backbone weight
keys, supplies the same frozen Target head and presents batch-one GPU views of the shared features.
Those views were checked against native `CacheCollator` values. Resident input bypasses the
offline collator's CPU allocations, avoiding a GPU → CPU → GPU roundtrip in this compute-only test.
It does not modify the DeepSpec checkout's training/model sources.

Use an isolated DeepSpec environment with its pinned requirements. Prepare the common features,
initial weights and matching TorchSpec control using the [benchmark tools](README.md). From the
DeepSpec checkout, run the ASpec adapter without injecting another checkout through `PYTHONPATH`:

```bash
cd /path/to/DeepSpec
CUDA_VISIBLE_DEVICES=0 python -c \
  'import runpy; runpy.run_path("/path/to/ASpec/tools/performance/deepspec_benchmark.py", run_name="__main__")' \
  --checkout "$PWD" \
  --features /path/to/shared-features \
  --initial /path/to/common-draft/model.safetensors \
  --reference-result /path/to/torchspec-resident-result.json \
  --output /path/to/new-deepspec-control \
  --max-length 4096 --warmup 16 --steps 64 --accumulation-steps 1
```

For the long group use its feature bank/reference result and `--max-length 131072`.
The reference check rejects mismatched feature identities, ordering or measurement windows.
Keep output JSON, commands, configurations and failure logs outside the source tree.

## Native cache-first full-corpus results

DeepSpec's native workflow precomputes an offline Target cache, then trains the Draft from disk.
The resident measurements above exclude Target feature computation and cache I/O. Here each case
uses eight GPUs for cache generation, followed by eight replicated Draft GPUs for disk-fed training,
rather than simultaneous Target/Draft roles.

The source inputs are the same frozen corpora as [the framework study](results.md#data-and-training-semantics):
60,416 existing Target-generated 4K documents and 4,096 original LongCite conversations admitted
at 128K. Neither answers nor token IDs are regenerated. The offline adapter preserves every
document without truncation or re-filtering.

Cache token IDs, loss masks, lengths and tensor extents were checked against the frozen inputs.
Native training consumed every document exactly once, without a dropped tail. All observed
microbatch losses and optimizer-step gradient norms were finite.

### Native configuration

DeepSpec uses the same pinned revision, BF16 Draft geometry, frozen Target bindings, native losses
and `torch_compile=true` as the resident control. Local batch is one, with global batches 512 at
4K and 8 at 128K (the minimum for eight independent Draft ranks). Native seeded shuffling uses
seed 42. One complete epoch sets the native LR horizon to **118 / 512 updates**, respectively.
Learning rate is `6e-4`, with native 4% warmup and cosine decay.
Native logging interval 10, four loader workers and
prefetching are retained. Preparation local batches are 16 / 1 for 4K / 128K.

The H200 environment exposes Hopper SM90 and 143,771 MiB per GPU, driver 570.148.08, 192 logical
CPUs and shared-filesystem cache storage. DeepSpec uses PyTorch 2.9.1/CUDA 12.8 and native
HF Target execution with BF16 SDPA. The ASpec resident control uses its CUDA 13 stack with
process-local CUDA compatibility libraries. Filesystem, OS page and compiler caches are not
forcibly flushed.

### Measured phases

| Cap | Documents | Input tokens | Raw cache tensors (TB) | Cache preparation seconds |
| --- | ---: | ---: | ---: | ---: |
| 4K | 60,416 | 44,174,395 | 2.17 | 889.87 |
| 128K | 4,096 | 132,381,588 | 6.51 | 1,924.32 |

Preparation time includes worker/model initialization, native Target execution, writing and
finalization, not the subsequent verification pass or training. It is not a pure Target compute
window. Cache sizes use decimal TB and exclude manifests and indices. The longest source documents
are 4,094 and 130,982 tokens.

| Cap | Updates / global batch | Training phase seconds | Slowest-rank loop seconds | Preparation + training seconds | Combined-phase input tokens/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 4K | 118 / 512 | 2,277.53 | 2,159.77 | 3,167.40 | 13,946.59 |
| 128K | 512 / 8 | 1,653.69 | 1,470.84 | 3,578.00 | 36,998.72 |

Combined-phase time sums the measured preparation and training phases. It includes input hashing,
worker/model initialization, native compilation, checkpoint saving and cleanup. Initial parent-process
imports/checks and the gap between phases, including cache verification, are excluded. Training-loop time
is the slowest rank's elapsed loop, excluding final saving. Peak allocated GPU memory during training
was 42.14 / 61.34 GiB for 4K / 128K.

Native `torch_compile=true` was retained throughout. The slowest first update took **1,122.31 s**
at 4K and **294.15 s** at 128K, including compilation and training work.
After 16 updates, the continuous rank-0 optimizer-completion windows covered 102 / 496 further
updates and measured **42,763.09 / 110,719.24 input tokens/s**, using disjoint token totals across
all eight ranks. These supplementary windows do not replace the combined-phase rates.

### ASpec online workflow

The **2026-10-24** H200 measurements in [the framework results](results.md) use the same frozen
corpora and initial weights. ASpec `0befaab` used five Target plus three Draft GPUs at 4K and
seven plus one at 128K, with packing/cuDNN, 512 anchors and CP1:

| Cap | Committed documents / input tokens | Loop-to-last-update seconds | Input tokens/s |
| --- | --- | ---: | ---: |
| 4K | 60,414 / 44,172,561 | 395.26 | 111,754.46 |
| 128K | 4,095 / 132,284,874 | 1,070.60 | 123,561.58 |

The document-count differences are explicit packed drop-last tails: two documents at 4K and one
at 128K. These online loop windows exclude initialization/final saving/shutdown, while the combined-phase
DeepSpec rates above include those costs. They provide an identified workflow reference, not a
same-boundary process-wall multiplier. Global physical batches also differ: ASpec uses 96 / 4
packed rows at 4K / 128K, versus DeepSpec's 512 / 8 single-document rows. Packing and native data
order lead to different optimizer counts and LR trajectories; this is not an equal-supervision
or convergence comparison.

### Interpretation

The resident controls show higher input-processing throughput for ASpec's packed execution
path, especially for short documents. They do not isolate the contribution of one kernel.
In the native offline workflow, cache generation/storage, disk-fed training and cold
compilation are additional visible costs: raw caches occupy about **2.17 / 6.51 TB** at 4K / 128K.
The 4K cold first update is substantial. Cache-read and data-feed observations also show higher
per-document costs for long inputs, but do not establish isolated Target, disk or Draft capacity.

This is a one-pass workflow comparison. An already-generated cache can be reused across epochs
or compatible experiments; its generation cost is then amortized. Shared-filesystem/cache behavior
also affects the result. A cold cache-first pass is not a universal multiplier for every offline
training schedule, just as resident Draft throughput is not a complete-workflow rate.

### Offline adapter and reproduction

The [offline adapter](../../tools/performance/deepspec_offline_benchmark.py) accepts the frozen
pre-tokenized JSONL without retokenizing, truncating, regenerating answers or filtering documents.
It delegates cache generation and the binary format to upstream. For 128K, the Target uses the
same YaRN factor 4 / original length 32,768 policy as the online study. This does not change
the common Draft's rotary configuration.

Use the isolated DeepSpec runtime and pinned checkout above. Copy only the two standalone adapter
modules into it (no ASpec runtime import or `PYTHONPATH` overlay), then run from that checkout:

```bash
cd /path/to/DeepSpec
mkdir -p tools/performance
cp /path/to/ASpec/tools/performance/deepspec{,_offline}_benchmark.py tools/performance/
python -m tools.performance.deepspec_offline_benchmark audit \
  --source /path/to/frozen-documents.jsonl --output /path/to/new-audit --max-length 131072
python -m tools.performance.deepspec_offline_benchmark prepare \
  --source /path/to/frozen-documents.jsonl --output /path/to/new-preparation \
  --target /path/to/Qwen3-8B --max-length 131072 --local-batch 1
python -m tools.performance.deepspec_offline_benchmark verify \
  --source /path/to/frozen-documents.jsonl --cache /path/to/new-preparation/cache \
  --output /path/to/cache-verification.json --max-length 131072
python -m tools.performance.deepspec_offline_benchmark train \
  --cache /path/to/new-preparation/cache --output /path/to/new-training \
  --target /path/to/Qwen3-8B --initial /path/to/common-draft/model.safetensors \
  --geometry /path/to/shared-features/manifest.json --max-length 131072 --global-batch 8
```

Use all eight visible GPUs. For 4K use `--max-length 4096`, preparation batch 16 and training
global batch 512. Verify the longest-document preparation first using the audit's `longest.jsonl`.
Each phase has its own output and elapsed time. Native CPU cache reads and asynchronous H2D
remain enabled. Preserve the declared preparation, training-loop and checkpoint timing boundaries
when comparing a new run with these results.
