# Supplementary configuration examples

Start with the [training recipes](../../docs/user-guide/recipes.md). These examples cover additional
structures, shared feature views, topology and evaluation workflows. Some use short training windows
or disable evaluation/checkpointing; inspect each file before use.

| Example | Purpose |
| --- | --- |
| [Qwen3.5-9B DSpark / 4K](qwen3_5_9b_4k_cudnn_nvlink.yaml) | Single-node DSpark without packing or CP splitting |
| [Qwen3.8-27B DSpark / 4K](qwen3_8_27b_4k_cudnn_nvlink.yaml) | Single-node DSpark without packing or CP splitting |
| [Ling shared structures](ling3_flash_multi_structure_dspark.yaml) | Independent Dense, Qwen MoE and Ling MoE Drafts sharing one Target |
| [Ling 128K SWA / CP4](ling3_flash_swa_128k_packing_cp4.yaml) | Long packed sliding-window training |
| [Ling CSA](ling3_flash_csa_dspark.yaml) | Dense → indexer warmup → compressed sparse attention |
| [Ling QSA](ling3_flash_qsa_dspark.yaml) | Dense → indexer warmup → query-shared sparse attention |
| [Ling full/SWA comparison](ling3_flash_swa_compare_dspark.yaml) | Independent full-context and sliding-window Drafts |
| [Qwen shared algorithm views](qwen3_8b_multi_algorithm_feature_views.yaml) | DSpark, EAGLE3 and DFlash with independent feature layers and CP |
| [Qwen MoE](qwen3_8b_moe_dspark.yaml) | Qwen-style routed experts |
| [Qwen MLA](qwen3_8b_mla_dspark.yaml) | Materialized MLA with a GQA Target |
| [Qwen DSA](qwen3_8b_dsa_dspark.yaml) | Token-level sparse attention and its three training stages |
| [Qwen DFlash2](qwen3_8b_dflash2.yaml) | Experimental DFlash2 objective |
| [Qwen role-separated TCP](qwen3_8b_4k_multinode_tcp.yaml) | Separate Target and Draft nodes with shared storage |
| [Qwen Target-generated quality protocol](qwen3_8b_dspark_target_generated.yaml) | Ten-pass training with held-out Target-generated data |
| [TorchSpec-style input](sglang_qwen3_8b_dspark.yaml) | Fixed compatibility input, not rolling upstream support |
| [Offline materialization](qwen3_8b_offline_materialize.yaml) / [replay](qwen3_8b_offline_replay.yaml) | Generate DSpark features once, then train with zero Target GPUs |

For complete source-checkout workflows, follow the
[offline generation, training and recovery example](../../examples/qwen3-8b-offline/README.md)
or the [DSA three-stage script](../../examples/qwen3-8b-dsa/README.md).

The 9B and 27B DSpark examples each start with one eight-GPU node: four TP1 Target engines and
four Draft GPUs, cuDNN attention, MBS1, CP1 and 4K context. Set `ASPEC_TARGET_MODEL` to the matching local checkpoint
and `ASPEC_TRAIN_DATA` to prepared token IDs/masks. These examples use a 13-step execution window
with evaluation and checkpoint saving disabled. The 9B DSpark example trains an independent Draft;
the `native-mtp` recipe fine-tunes Qwen3.5's native predictor.

Dense DSpark, DFlash and DFlash2 examples use cuDNN by default. CSA/QSA retain FlexAttention
for their unpacked CP1 paths: CSA uses compressed-entry masks, while QSA's sparse stage gathers
selected K/V and invokes FA4 varlen attention internally. Their packed/CP paths require FA4;
cuDNN does not implement these sparse structures. Other FA4 examples cover long packed attention,
DSA or shared feature views that include EAGLE3.

Change sequence length, attention backend or CP through overrides on a suitable starting config
rather than choosing a separate YAML for every combination. Inspect Plan and provide compatible
model context policies, data and memory capacity. Qwen3.5 MTP, Qwen3.8-27B long context and node-local
multi-node training have dedicated recipes.

See [algorithms](../../docs/user-guide/algorithms.md), [structures](../../docs/user-guide/structures.md),
[sharing](../../docs/user-guide/sharing.md), [multi-node](../../docs/user-guide/multinode.md) and
[decode reproducibility](../../docs/support.md#decode-reproducibility) for execution and evidence boundaries.

Use the documented support boundaries, required environment inputs and fresh outputs. Do not assume
one model's pretokenized data works with another tokenizer.
