# Benchmark tools

Read [results](results.md) for measured configurations and
[methodology](methodology.md) for timing and comparison rules. Use the pinned
runtime and run commands from the repository root. Benchmark adapters for other
frameworks require separate native environments and their disclosed revisions.

| Tool | Purpose |
| --- | --- |
| `tools.performance.torchspec_benchmark` | Resident DSpark Draft training and packed/padded layout comparisons |
| `tools.performance.torchspec_e2e_benchmark` | Online Target-to-Draft workflow measurements |
| `tools.performance.specforge_draft_benchmark` | SpecForge resident Draft training |
| `tools.performance.specforge_e2e_benchmark` | SpecForge continuous-wall-clock workflow measurements |
| `tools.performance.deepspec_benchmark` | DeepSpec resident Draft training |
| `tools.performance.deepspec_offline_benchmark` | Native cache preparation and disk-fed training |
| `tools.performance.bench_eagle3_train_core` | EAGLE3 train-core latency and memory |
| `tools.performance.bench_thd_cp_attention` | Packed attention with context parallelism |

Run `python -m <tool> --help` for arguments. Templates are listed in the
[configuration index](../../configs/README.md). Supply your own model, initial
Draft weights and dataset; normal ASpec training needs no upstream source checkout.

## Online training

```bash
export BENCH_TARGET_MODEL=/path/to/target
export BENCH_INITIAL_DRAFT=/path/to/initial-draft
export BENCH_TRAIN_DATA=/path/to/train.parquet
export BENCH_OUTPUT=/path/to/new-run
export BENCH_CACHE=/path/to/cache
python -m tools.performance.torchspec_e2e_benchmark \
  --framework aspec --config configs/benchmarks/aspec_qwen3_8b.yaml \
  --result /path/to/new-result.json
```

The template is a bounded starting point, not the final matrix configuration.
For the published ASpec headline, set `training.dflash_packing=false` and
`training.micro_batch_size=1`, then match the result table's corpus horizon,
role split, original-document GBS, anchor policy and timing window. Select the
128K template for long inputs. Upstream YAMLs use their own loaders and runtime;
replace their path placeholders and use unique output/control directories.

Enable normal hardware interconnects and sufficient buffers. The pinned Mooncake
single-node managed NVLink launcher selects `MC_FORCE_MNNVL=1`; CUDA IPC uses
`MC_USE_NVLINK_IPC=1`. Confirm the effective transport in worker logs.

## Resident Draft training

Prepare common Target features and initial weights using the adapter's `data`
and `features` commands. For four original documents per forward:

```bash
python -m tools.performance.torchspec_benchmark run --framework aspec \
  --features /path/to/features --initial /path/to/initial.pt \
  --layout-mode non-packing --micro-batch-size 4 --documents-per-step 64 \
  --anchor-sampling-scope sample --document-ordering length-grouped \
  --warmup 16 --steps 64 --result /path/to/bshd-mbs4.json
```

Use `equal-work` with the same arguments and a fresh result path for a packed
row containing the same four documents. Each document has a 512-anchor cap.
Fixed-count row capacity equals the bank's maximum document length times MBS.
Production `natural-packing` instead uses the row packer and may consume a
variable document count per update; production packed MBS counts physical rows.

Each command above uses one GPU. Distributed padded controls use
`torch.distributed.run`, the same global `--documents-per-step` and a shared
initial state. The adapter uses fixed strided DP assignment; aggregate throughput
uses disjoint work and the slowest rank's continuous window. Target computation,
transport and feature loading are excluded; final metric flush and device
completion are included.

## Other comparisons

- [DeepSpec](deepspec.md): resident and native cache-first workflows.
- [EAGLE3](eagle3-h200.md): synthetic train-core measurement and numerical scope.
- [Packed attention/CP](cudnn-thd-cp.md): attention component results.
- [Training and acceptance](dspark-alignment.md): TorchSpec/ASpec BSHD/THD
  training alignment and verifier metrics.
- [CP quality alignment](dspark-cp-alignment.md): CP2 versus same-layout CP1
  for ASpec BSHD and THD.

Keep raw data, logs, checkpoints and profiler traces outside the source tree.
Record the effective configuration, software, input identity and timing window.
For numerical/operator checks, see [runtime validation](../contributing/validation.md).
