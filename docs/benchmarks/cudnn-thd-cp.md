# cuDNN flattened-token context parallelism

## Supported boundary

DSpark/DFlash-family GQA/MHA and materialized-MLA full or sliding-window attention can use native
cuDNN flattened-token (`THD`) execution with context parallelism. The implementation keeps
ASpec's direct-CP ownership and exact cross-rank softmax merge:

- queries are gathered by stable block ownership;
- context and Draft K/V remain sharded instead of being replicated;
- cuDNN receives flattened Q/K/V plus cumulative sequence lengths;
- packed-document, causal and window rules use interval endpoints without a dense query-by-key
  mask or per-layer document regrouping;
- differentiable LSE participates in the existing exact partial merge and backward.

There is no CP-degree whitelist. Legal degrees still depend on world size, anchor divisibility,
shard alignment and capacity. CSA/QSA/DSA selected-entry execution remains on FA4 and fails closed
when combined with the cuDNN backend.

## Verification record

The H200 measurements used
Torch 2.13.0/CUDA 13, FA4 4.0.0b34 and cuDNN Frontend 1.31.0. Native attention checks covered
packed GQA forward, LSE, Q/K/V backward and representative CP1/2/4 degrees against the unsharded
mathematical reference. Sixteen-step packed GQA and materialized-MLA training completed at CP1/2/4
with equal replicas. Relative to CP1, maximum loss-trajectory relative differences were 0.0992%
(CP2) and 0.0819% (CP4) for GQA, and 0.0696% (CP2) and 0.0869% (CP4) for MLA. DFlash2 causal and
sliding-window paths additionally completed 16-step packed CP2 training.

The component timing below uses the same BF16 packed GQA workload for both backends and measures
forward plus backward after three warmups. It is evidence for this attention boundary, not an
end-to-end training-throughput claim.

| Context / CP | Anchors × block | Runs | FA4 median | cuDNN median | Result |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K / CP1 | 128 × 16 | 10 | 2.210 ms | 1.807 ms | cuDNN 1.22× |
| 8K / CP2 | 128 × 16 | 10 | 2.463 ms | 2.701 ms | FA4 1.10× |
| 128K / CP2 | 512 × 16 | 5 | 18.811 ms | 12.228 ms | cuDNN 1.54× |
| 128K / CP4 | 512 × 16 | 5 | 11.286 ms | 8.019 ms | cuDNN 1.41× |

The short CP2 case is communication/launch dominated and does not support a universal cuDNN
speedup claim. The long-context cases match the intended use: cuDNN improves the attention boundary
while CP collectives and the rest of the training step remain unchanged.

Reproduce one component point from the repository root, changing the backend and launch size for
the comparison:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m tools.performance.bench_thd_cp_attention \
  --backend cudnn_flex_attention --context-length 131072 \
  --num-anchors 512 --warmup 3 --runs 5 --output result.json
```

The table and numerical summary above are the maintained public record. New runs should write JSON
outside the checkout and retain it with the exact source revision, environment and command; the
benchmark tool does not depend on historical artifacts.
