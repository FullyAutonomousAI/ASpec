# Joint GQA backward

This opt-in ASpec route keeps native cuDNN GQA forward and replaces backward
with ASpec-maintained CuTe kernels adapted from cuDNN Frontend 1.31.0. The installed
cuDNN package is not patched. Set `cudnn_gqa_mode: internal` in the draft model
configuration and use `attention_backend: cudnn_flex_attention`. Here, `internal`
means that query heads are grouped inside the kernel; it is a public configuration option.

## Execution contract

- Supported: SM90 and SM100 family, BF16 contiguous Q/K/V, Dqk=Dv=128, Hq/Hkv=4,
  head-independent interval masks, fixed BSHD and ragged THD, CP=1.
- Unsupported explicit requests fail; this mode does not silently fall back.
- `auto` preserves existing ASpec selection. `native` forces native GQA.
  `external` forces the existing physical query-head fold, including its copies.
- Softmax, mask semantics, BF16 attention boundaries, FP32 reductions and
  optimizer precision are unchanged. Floating-point summation order differs
  from native GQA, so gradient comparisons use numerical tolerances rather
  than bitwise equality.
- First-order autograd includes the LSE gradient. Higher-order autograd and
  multi-rank context parallelism are not supported by this route.
- cuDNN Frontend is pinned to 1.31.0 for this route because imported private
  planner/kernel interfaces are version-sensitive. Other modes retain their
  normal dependency behavior.

## How it works

For logical query token `t`, KV head `h`, and group member `r in [0,4)`,
the main backward kernel sees virtual row `4*t+r`, head `h`, while reading
the original tensor element `Q[t,4*h+r,:]`. CuTe's nested shape and strides
express this mapping without materializing a folded Q or dO tensor.

All four Q heads are consequently reduced within the same KV-head traversal.
The upstream MHA backward path writes dK/dV directly instead of using the native
GQA FP32 global dK/dV accumulation workspaces and conversion kernels.
SM90 uses its native 64x128 WGMMA tile and matching 256-thread dQ postprocess;
SM100 keeps its native tile/CTA configuration. The same virtual-row mapping applies
to fixed and ragged layouts; cumulative query lengths are multiplied by four and
ragged Q/dO accesses convert the virtual offset back to original tokens.

The preprocess kernel reads original O/dO/LSE and writes folded scalar statistics;
the dQ postprocess writes directly to the original query-head layout.
There are still statistics and FP32 dQ workspaces, and metadata construction costs
are included in the native-trainer comparison.

The SM100 dS exchange relay publishes only local shared-memory data. Its
shared-specific release fence retains cluster visibility while avoiding the
general all-memory release in the hot loop. MMA ordering, BF16/FP32 arithmetic,
and the acquire wait are unchanged; see the synchronization audit for the
publication and buffer-reuse proof.

Each autograd call owns its scratch tensors and output gradients. Cached plans
contain immutable metadata, and the bounded JIT cache contains compiled functions
rather than activation or gradient buffers. Multiple outstanding forwards must
therefore not share writable backward workspace.

The compact sparse metadata length changes when anchors are resampled. That
leading extent is dynamic in CuTe and is excluded from the JIT specialization
key; payload dimensions, strides, dtype/device and zero/singleton extents remain
distinct. A regression changes metadata cardinality between backward calls and
checks both gradient accuracy and compiled-function reuse.

## Source and maintenance boundary

The architecture-specific backward kernels and shared pre/postprocess retain
upstream pipelines, sparse traversal and MMA arithmetic. Changes are concentrated
in virtual tensor layouts, ragged sample offsets, original-layout statistics/
gradient access, the shared-only dS relay fence, and an explicit final TMA
shared-read drain in the SM100 fixed-layout epilogue. The synchronization audit
records the publication and storage-lifetime guarantees for these changes.
The host adapter supplies version/shape guards, autograd ownership, and a bounded
compile cache. No scheduling or lower-precision approximation is introduced.

These are **adapted**, not independent clean-room kernels. The source release is
the installed `nvidia-cudnn-frontend==1.31.0` wheel; exact source SHA-256 values
are in `upstream_sha256.json`. No Git revision is inferred from the wheel version.

| ASpec destination | Upstream under `cudnn/flex_attention/` |
| --- | --- |
| `backward.py` | `kernels/sm100/bwd/backward.py` |
| `backward_sm90.py` | `kernels/sm90/bwd/backward.py` |
| `preprocess.py` | `kernels/common/backward_preprocess.py` |
| `postprocess.py` | `kernels/common/backward_postprocess.py` |
| `helpers.py` | Layout ideas/code adapted from `kernels/common/pack_gqa.py` and `runtime/dsl_utils.py` |
| `aspec/models/ops/cudnn_gqa_backward.py` | Host sequencing adapted from `dispatch.py` |

The source chain includes FlashAttention/CuTe material credited in each file.
Applicable Apache-2.0 and BSD-3-Clause terms are retained; see the repository
`LICENSE`, `licenses/FlashAttention-BSD-3-Clause.txt`, and
`licenses/FlashAttention-AUTHORS`. The source index is
`docs/framework/REFERENCES.md`. The wheel source identifies its version and hashes;
this is distinct from the older Q/K-normalization kernel's Git revision.

All modified backward code is owned here, including dispatch, pre/postprocess and
virtual layouts. Unmodified forward, sparse planner, pipeline/math helpers and
runtime conversion utilities remain normal imports from cuDNN Frontend. This is
a self-contained **adaptation**, not a cuDNN-free replacement: install the pinned
runtime in `requirements/sglang-v0.5.21.txt`, including
`nvidia-cudnn-frontend==1.31.0` and `nvidia-cutlass-dsl[cu13]==4.7.1`.
No source rewriting, monkey patching or custom cuDNN checkout is required.

## Validation

GPU regressions are opt-in:
`ASPEC_RUN_CUDNN_FLEX_INTEGRATION=1 python -m pytest tests/test_cudnn_internal_gqa_gpu.py`.
They cover fixed/ragged layouts, empty visible rows, LSE gradients, all Q/K/V
gradients, and independent outstanding autograd graphs.

`python -m tools.performance.benchmark_cudnn_gqa --block 7 --output /tmp/gqa.json`
compares native, external-fold and internal-joint attention on the same 128K
packed workload. It records correctness, anchor identity, tile counts and shuffled
timing windows. A separate [five-path performance study](../../../../docs/dspark_attn_kernel_optimization.md#one-workload-five-execution-paths)
records its own workload and timing boundary. Synchronization and workspace
ownership are reviewed in the [synchronization audit](FENCE_AUDIT.md).
