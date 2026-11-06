# EAGLE3 frozen-Target reuse on H200

## Scope

This record measures the single-GPU EAGLE3 Draft train core. It excludes Target generation,
transport, data loading, and distributed orchestration, so the result is not an end-to-end service
throughput claim. Both frameworks execute the same Qwen3-8B-shaped EAGLE3 workload: BF16 model and
Target logits, FP32 hidden-state projection and loss softmax, TTT length 3, batch 1, 32 query heads,
8 KV heads, head dimension 128, hidden size 4096, intermediate size 12288, vocabulary 151936, five
auxiliary hidden states, FlexAttention, and backbone activation checkpointing. TF32 and FP8 are not
used.

The measured model is a randomly initialized, one-layer EAGLE3 Draft with the dimensions above,
not a loaded Qwen3-8B checkpoint. Token IDs, auxiliary hidden states, teacher features and the frozen
teacher head are synthetic and already resident on the GPU. Each run repeats those inputs while
updating the Draft weights. This is not a dataset training or quality experiment.

Steps use fused AdamW with FP32 master weights, learning rate `6e-4`, zero weight decay and
gradient clipping at 0.5. The cosine LR horizon is warmup + measured steps + 8, with no LR warmup.
Timing uses CUDA events after gradient zeroing through optimizer completion; initialization,
warmup and subsequent host-side metric collection are excluded. Cached Target projection is
included inside each measured step, not precomputed outside the timer.

The reference is TorchSpec revision `a4e5c1085bfc5990ddcae6cf5832bb26380b38d3`; the H200
comparisons below use ASpec `a4b27f1`. Each software cohort records its own runtime and
numerical qualification. Performance measurements execute compiled loss functions; eager-loss
unit tests check loss math and do not reproduce this compiled backward path.

## Optimization

The Target is frozen, while its hidden states and full-vocabulary head are invariant across EAGLE3
recurrent depths. The baseline nevertheless repeated the same BF16 vocabulary projection at each
depth and again when checkpoint backward recomputed the loss. ASpec now offers two independent,
fail-closed controls on the full-vocabulary padded path:

- `eagle3_target_cache=logits` computes the frozen BF16 Target logits once per microbatch and reuses
  them at every depth. Each depth still executes the original FP32 softmax and loss.
- `eagle3_loss_gradient_checkpointing=false` retains only the loss graph while keeping Draft
  backbone checkpointing enabled. Backward no longer repeats the full-vocabulary projection or loss
  reductions.

The public Qwen3-8B EAGLE3 recipe enables both. Pruned-vocabulary and FA4/THD paths retain `lazy`
Target materialization and loss checkpointing; invalid combinations fail Plan compilation.


## H200 results

Measured with PyTorch 2.12.1+cu130, CUDA 13.0, Transformers 5.12.1 and Triton
3.7.1 on one H200 (SM90). Native CPU thread settings are retained. This component
benchmark runtime is distinct from the project's pinned SGLang installation lane.
ASpec control uses lazy Target materialization and loss checkpointing; optimized
ASpec enables cached Target logits and disables loss checkpointing. Both retain
backbone checkpointing and native fused clipping. Each row uses two warmup
updates, followed by 15 measured updates at 4K or five at 16K.

| Sequence | Implementation | Median step | Mean step | Peak allocated | Relative to TorchSpec |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4K | TorchSpec | 240.188 ms | 240.061 ms | 22.70 GiB | 1.000x |
| 4K | ASpec control | 239.049 ms | 239.065 ms | 22.70 GiB | 1.005x |
| 4K | ASpec optimized | **169.738 ms** | 169.435 ms | 25.05 GiB | **1.415x** |
| 16K | TorchSpec | 981.335 ms | 981.984 ms | 32.42 GiB | 1.000x |
| 16K | ASpec control | 977.295 ms | 975.964 ms | 32.42 GiB | 1.004x |
| 16K | ASpec optimized | **691.256 ms** | 690.548 ms | 41.41 GiB | **1.420x** |

Relative to the same-revision ASpec control, the optimized path is **1.408x at 4K** and
**1.414x at 16K**, reducing median latency by **28.99% / 29.27%**. Relative to TorchSpec, latency
falls **29.33% / 29.56%**, at an additional peak allocation of **2.35 / 8.99 GiB**.

### Numerical scope

All measured losses, gradient norms and timings are finite. Supervision is identical across the
three implementations: 12,240 / 49,104 predictions per step at 4K / 16K. Maximum per-step
differences for optimized ASpec are:

| Sequence | Reference | Loss absolute diff | Loss relative diff | Gradient-norm absolute diff | Gradient-norm relative diff |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4K | TorchSpec | 1.717e-5 | 0.00005844% | 2.539e-5 | 0.01876% |
| 4K | ASpec control | 0 | 0 | 0 | 0 |
| 16K | TorchSpec | 3.433e-5 | 0.0001171% | 1.474e-5 | 0.01062% |
| 16K | ASpec control | 3.433e-5 | 0.0001171% | 1.474e-5 | 0.01062% |

At 4K, all 15 recorded losses and gradient norms match ASpec control bitwise. TorchSpec uses
separate clipping while ASpec uses native fused clipping. At 16K, optimized ASpec does **not** match either
reference bitwise. The fixed-weight 4K result must not be extrapolated to all lengths or to
independent updated-weights trajectories. These performance rows are single-run train-core
observations, not full-gradient/FP32-update equality, quality or end-to-end throughput claims.

At 16K, compiler-selected RMSNorm reduction order changes BF16 rounding and
some parameter gradients despite identical cached/recomputed Target logits.
At fixed weights, the largest observed parameter-gradient relative L2 difference
is 0.3786%. Loss and gradient-norm agreement alone therefore does not establish
full-gradient equality. PyTorch 2.13 also does not meet strict independent-
trajectory equality for this optimization. Numerical conclusions are scoped to
the measured runtime and shape; no held-out quality claim follows from this
synthetic train-core benchmark.

## Reproduction

Run each framework from its isolated checkout and environment. Keep the GPU, package versions,
seed, and source revisions fixed. The benchmark writes raw per-step records, loss, gradient norm,
timings, precision flags, and memory use to JSON.
Use the same `tools/performance/bench_eagle3_train_core.py` from this repository in both checkouts;
copy that file into the TorchSpec checkout if absent. Do not use another checkout's `PYTHONPATH`.
Use PyTorch `2.12.1+cu130` for the current H200 comparison; results from other PyTorch versions are not interchangeable denominators. Leave each framework's native optimizer path
enabled and do not change the LR schedule or loss precision for numerical agreement.

```bash
python -m tools.performance.bench_eagle3_train_core \
  --framework torchspec --seq-len 4096 --warmup 2 --steps 15 \
  --output /path/to/torchspec-eagle3-s4096.json

python -m tools.performance.bench_eagle3_train_core \
  --framework aspec --seq-len 4096 --warmup 2 --steps 15 \
  --cache-target-logits --disable-loss-checkpoint \
  --output /path/to/aspec-eagle3-s4096.json
```

For Nsight Systems, add `--profile` and capture the `profile_window` NVTX range. Compare every
element of `losses` and `gradient_norms` before interpreting a timing result.
For 16K, use `--seq-len 16384 --warmup 2 --steps 5`. For a ASpec control row, omit both
optimization flags. The reported optimization is on the padded, full-vocabulary path; this test
does not measure packed training or context parallelism.
