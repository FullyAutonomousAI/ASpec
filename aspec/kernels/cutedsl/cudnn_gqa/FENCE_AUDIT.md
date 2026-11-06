# Synchronization audit

**Baseline fence audit: passed (2026-10-31, `dab7a2c`).** The audit checked the local
dS exchange completion, shared-memory release/cluster-acquire handoff, and
consume-to-reuse edges below against source and generated SASS. The reviewed
kernel and regression-test bytes matched the retained 104-test validation bundle.
Device memcheck and synccheck passed for the selected main kernel; racecheck
remains inconclusive because the tool aborts on both baseline and candidate.
Device checks used `--report-api-errors no` after both baseline and candidate
encountered the same API lookup errors; this suppresses API error reporting,
not device memory instrumentation. These selected-launch checks do not certify
every cuDNN or trainer kernel.

The SM100 fixed-layout dK/dV epilogue also has an explicit final TMA shared-memory
read drain. Its validation is recorded separately below; the baseline's
104-test bundle does not certify this later change.

Scope: the supported BF16, D=128, ratio-4 route, SM90 64×128/two dQ
warpgroups and SM100 128×128/two-CTA configuration, fixed and ragged layouts.
Inactive upstream branches for other dimensions, deterministic mode and other
group ratios are not exposed by the adapter. Re-audit before exposing them.

The source comparison against the hashes in `upstream_sha256.json` preserves
MMA sequencing, pipeline stage counts, sparse iteration order and buffer reuse.
ASpec changes the Q/O/dO/stat/dQ address mapping. A virtual row is a bijection
`(4*t+r,h) -> (t,4*h+r)`, including each ragged sample offset; it introduces no
alias between writable query rows or samples.

The SM100 dS relay uses a shared-memory-specific cluster release before its
notification: `fence.release.sync_restrict::shared::cta.cluster`, followed by
`mbarrier.arrive.relaxed.cluster.shared::cluster` on the leader CTA's barrier.
The leader retains `mbarrier.try_wait.parity.acquire.cluster.shared::cta`.
This is a release/acquire publication, not a relaxed-only notification.

The relay first waits for the peer's bulk dS exchange to complete in its own
shared memory. Its only published payload is that completed local dS storage;
it publishes no GMEM data, descriptor updates, or global workspace ownership.
The shared release covers that payload, including the visibility acquired by the
completion wait. Both relay arrivals are still required before the leader's dQ
MMA. Original generic-to-async proxy fences, transaction completion waits,
phase progression, and buffer-reuse acknowledgments remain unchanged.

Restricting the ordered state space does not restrict the communicating scope:
it remains the CTA cluster. In CUDA 13 / DSL 4.7.1 on the tested SM100 device,
the former general `mbarrier.arrive.release.cluster` lowered to
`MEMBAR.ALL.GPU` and `ERRBAR` in the hot relay loop. The shared-specific release
removes those instructions without changing MMA arithmetic or removing the
required publication. The SASS GPU scope was an ordering cost, not a device-wide
execution barrier. Do not reuse this helper for a relay that publishes GMEM.


## Producer, consumer and reuse edges

| Resource / participants | Completion and visibility | Reuse / final drain |
| --- | --- | --- |
| Host launch inputs, sparse metadata and tensor descriptors → GPU | Normal current-stream CUDA ordering; descriptors and metadata are immutable for the invocation. TVM-FFI obtains that stream at launch. | Autograd retains input tensors and its plan; the cache owns compiled functions, never writable activation buffers. Cross-stream callers must establish dependencies for shared input/mask tensors. |
| Preprocess O/dO/LSE → FP32 dPsum/log2-LSE and zeroed dQ accumulator in GMEM | CTA-owned output tiles; generic writes complete before main through stream ordering. Ragged padding uses the same folded lengths as main/postprocess. | Each backward call allocates its own workspace; no cache reuse across live graphs. |
| SM90/SM100 TMA Q/K/V/dO and scalar statistics → SMEM consumers | Producer acquire waits for empty stage, then sets transaction count. Full mbarrier completion/acquire wait precedes generic reads or MMA. No generic SMEM writes are substituted for TMA completion. | Consumer release follows its last read/MMA completion. Producer tails retain final-stage protection. Phase/index advancement and transaction bytes remain upstream. |
| SM90 registers/SMEM → WGMMA S/dP/dK/dV/dQ | `sm90_utils.gemm` emits WGMMA fence, MMA, commit and the requested wait. Generic P/dS SMEM stores have async proxy fences and the PdS named barrier before WGMMA. | dO is released after the dV group completes; Q after `wait_group(0)` completes dK. Double-buffered P/dS and end-of-iteration waits protect stage wrap. |
| SM90 dQ registers → SMEM → GMEM reduction warp | Empty/full named barriers transfer ownership; generic writes precede `fence_view_async_shared`, then the full arrival. Bulk reduction commit/wait tracks async access. | Empty notification follows the store warp's completion; final bulk waits protect SMEM lifetime. |
| SM90 dK/dV registers → epilogue SMEM → TMA GMEM output | CTA epilogue barriers plus async shared proxy fences precede TMA writes. Each KV tile has one writer after the joint traversal. | Bulk-group read waits precede SMEM overwrite; final drain precedes exit. Native GQA dK/dV global accumulation buffers are not used. |
| SM100 MMA → TMEM S/dP/dQ/dK/dV → generic register consumers | `PipelineUmmaAsync` full arrival lowers to `tcgen05.commit` on the matching CTA group; successful full wait precedes TMEM loads. | `fence_view_async_tmem_load` lowers to `tcgen05.wait::ld` before release/overlap reuse. Named barriers collect participating compute/reduction warps where required. |
| SM100 generic registers → TMEM P/dS → MMA | TMEM stores precede `tcgen05.wait::st`; compute barriers precede pipeline release/commit. | Shared S/P and dP/dS storage is not overwritten until the matching consumer phase completes. |
| SM100 generic dS → local SMEM / peer DSMEM → leader MMA | Generic stores use the async shared proxy fence and compute barrier. Bulk shared-to-cluster copy completes against the peer transaction barrier. Each relay waits for completion, then a shared-specific cluster release and the leader's cluster acquire gather both CTAs. | The dS pipeline's MMA completion protects both local and exchanged buffers. For D=128 its first/intermediate/final notifications retain the upstream deferred S-load-completion protocol; do not add an unmatched producer tail. |
| SM100 dQ TMEM → SMEM → atomic GMEM reduction | TMEM load wait precedes release; generic SMEM stores use proxy fence and reduction barrier before bulk reduction. | Bulk-group read waits and reduction barriers protect stage reuse; final waits drain stores before the warp reports completion. |
| SM100 fixed-layout dK/dV TMEM → SMEM → TMA GMEM output | dKV full wait and TMEM load waits protect accumulator access; generic SMEM stores use the async shared proxy fence and an epilogue barrier before TMA. | Each stage, including the final stage, commits and waits for TMA shared reads before its warpgroup barrier. This protects shared storage through CTA exit. The inactive deterministic-KV branch instead retains its final full-completion wait before semaphore release. |
| SM100 ragged dK/dV TMEM → direct GMEM output | dKV full wait and TMEM load waits protect final accumulator access; the epilogue writes the one owned KV tile directly. | dKV release follows the last read. Empty sparse tiles follow the zero-output epilogue without consuming a nonexistent MMA result. |
| SM100 TMEM allocation/deallocation and barrier storage | Pipeline initialization fences/cluster synchronization precede use. Allocation pointer publication uses the named allocation barrier. | Compute/reduction warps finish their loops and TMEM reads before the allocator warp's final barrier/free. The two-CTA allocator coordinates deallocation. |
| Main GMEM gradients → postprocess → caller | Kernel ordering on the same CUDA stream makes completed dQ reductions visible. Postprocess applies scale/conversion and the inverse layout mapping. Its shared-memory redistribution retains CTA barriers and async-copy waits. | Output tensors belong to the invocation; no later call can reuse their storage while autograd retains them. |

All sparse participants derive their iteration count/order from the same metadata.
Empty-tile branches skip matching producer and consumer work together. Ragged tail
predicates mask data access, not required barrier arrivals. Phase advancement is
unchanged for first use, stage wrap and the final iteration. No new early return
or divergent warp-group barrier is introduced.

## Fixed-layout dK/dV epilogue

The final nondeterministic fixed-layout dK/dV store now commits its bulk group
and waits for shared-memory read completion before the existing warpgroup
barrier. Actual compiled SM100 SASS contains `UTMACMDFLUSH` and
`DEPBAR.LE SB0, 0` after both TMA output stores. This extends the intermediate
stages' storage-lifetime rule through the final stage; it changes no MMA or
rounding operation. Numerical corruption was not reproduced in the old version.

After this change, 34 GQA/gradient GPU cases passed, including the independent
FP64 oracle, changing-payload graph replay and optimizer/checkpoint checks.
Compute Sanitizer memcheck and synccheck each reported zero errors for one
selected fixed-layout SM100 main-kernel launch. These narrow sanitizer runs do
not replace the protocol audit or certify other architectures. No fresh SM90
run or conclusive racecheck result is claimed here.

## Helper evidence and authority

Reviewed the installed **cuDNN Frontend 1.31.0** `common/pipeline.py`, sparse
producer/consumer helpers, `sm90_utils.gemm`, and **CUTLASS DSL 4.7.1**
`MbarrierArray`, `mbarrier_wait/arrive`, async shared fence and TMEM fence helpers.
The latter lower explicitly to async shared proxy fencing and `tcgen05.wait::ld/st`;
`MbarrierArray`'s TCGen05 operation lowers arrival to `tcgen05.commit`, not an
ordinary early generic arrival. SM90 GEMM retains fence/commit/wait separately.

The [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
is the authority: sections on mbarrier release/acquire and cluster example,
WGMMA asynchronous proxy operations, and tcgen05 completion/anti-dependency
ordering. Passing numerical tests supports this audit; it does not replace the
producer/consumer and consume/reuse proof. Recheck these helpers and rerun both
architectures before changing the pinned dependency versions.

The [fence specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence)
explicitly gives the shared-specific release followed by a relaxed cluster
arrival as a publication pattern. The restriction selects local shared-memory
objects; the communicating scope remains the cluster. The relay's prior
completion wait acquires that local payload, and the leader's acquire wait
precedes dQ MMA. No global-memory publication relies on this restricted fence.
