# Copyright 2026 Ant Group Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations.

"""Compare block compaction/head reuse with the original dense execution.

Launch with torchrun, CP1/2/4. Sixteen native optimizer updates include an
empty CP rank, sparse supervision, partial-depth tails and a step-eight in-memory
model/optimizer/RNG restore. This small synthetic fixture is not a throughput or
full Qwen3 training-quality claim. Optional --features/--initial use the complete
recorded Draft, frozen Target head and real production packing rows instead.
cuDNN is the default; FA4 and sliding-window controls are explicit options.
BF16, FP32 masters, fused AdamW, clipping and
the existing benchmark LR schedule are identical. Common-state and independent
trajectory comparisons are reported separately.
"""

import argparse
import copy
import json
import os
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from aspec.data.utils import DataCollatorWithPadding, DFlashPackingCollator
from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from aspec.models.dspark import DSparkModel
from aspec.training.checkpoint import OptimizerState
from aspec.training.optimizer import BF16Optimizer
from aspec.utils.distributed import init_direct_cp_groups


@torch.no_grad()
def relative(left, right):
    pairs = [(left, right)] if isinstance(left, torch.Tensor) else zip(left, right, strict=True)
    numerator = denominator = 0
    for actual, reference in pairs:
        actual, reference = actual.float(), reference.float()
        numerator = numerator + (actual - reference).square().sum(dtype=torch.float64)
        denominator = denominator + reference.square().sum(dtype=torch.float64)
    return float((numerator / denominator.clamp(min=1e-40)).sqrt())


def run(args):
    if Path(args.result).exists():
        raise FileExistsError(args.result)
    if args.reference_stage == "gqa-layout" and args.attention_backend != "cudnn_flex_attention":
        raise ValueError("gqa-layout compares cuDNN folded/native attention")
    if args.micro_batch_size < 1:
        raise ValueError("micro-batch-size must be positive")
    if args.reference_stage == "bshd-native" and (
        args.layout_mode != "non-packing" or args.attention_backend != "cudnn_flex_attention"
    ):
        raise ValueError("bshd-native isolates the cuDNN non-packing execution changes")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cp = dist.get_world_size()
    if args.reference_stage == "padding" and (cp != 1 or args.layout_mode != "non-packing"):
        raise ValueError("padding isolates CP1 BSHD physical sequence bucketing")
    init_direct_cp_groups(cp)
    torch.manual_seed(71)
    manifest = None
    if args.features is not None:
        if args.initial is None:
            raise ValueError("Real feature validation requires the matching --initial weights")
        manifest = json.loads((args.features / "manifest.json").read_text())
        config = DSparkConfig(**manifest["draft_config"])
    else:
        config = DSparkConfig(hidden_size=512, intermediate_size=1024, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=128,
                         vocab_size=8192, num_target_layers=1, target_hidden_size=512,
                         target_num_hidden_layers=4, target_layer_ids=[1],
                         mask_token_id=8191, markov_rank=32)
    if args.sliding_window is not None:
        config.sliding_window = args.sliding_window
        config.layer_types = ["sliding_attention"] * config.num_hidden_layers

    def model(compact):
        result = DSparkModel(DSparkDraftModel(config), block_size=7, num_anchors=512 if manifest else args.num_anchors,
                             anchor_sampling_scope="sample", attention_backend=args.attention_backend,
                             draft_context_parallel_size=cp, deterministic_anchor_sampling=True,
                             confidence_head_alpha=args.confidence_alpha)
        latest = compact or args.reference_stage in ("gqa-layout", "head-replay", "padding")
        retained = latest or args.reference_stage in ("block-head", "context-head", "spare-layout", "indexed-head")
        result._compact_output_blocks = retained
        result._compact_backbone_blocks = retained
        result._reuse_head_projections = retained
        result._reuse_context_teacher_projection = latest or args.reference_stage in ("context-head", "spare-layout", "indexed-head")
        result._reuse_context_draft_projection = latest or args.reference_stage in ("spare-layout", "indexed-head")
        result._use_l1_kernel_optimizations = latest or args.reference_stage == "indexed-head"
        # Historical stage controls isolate their named change at the original
        # cache bound. Dense/current and indexed-head exercise the latest bound.
        result._head_projection_cache_chunks = (
            2 if latest and args.reference_stage in ("dense", "indexed-head", "gqa-layout", "head-replay", "padding") else 1
        )
        current_replay = (
            args.reference_stage in ("dense", "gqa-layout", "head-replay", "padding")
            and (compact or args.reference_stage in ("gqa-layout", "padding"))
        )
        result._fill_head_projection_cache = current_replay
        result._reuse_l1_statistics = current_replay
        if args.reference_stage == "bshd-native":
            # Previous BSHD already compacted the head and used indexed L1.
            # Isolate the new backbone selection and single-chunk context reuse.
            result._compact_output_blocks = True
            result._compact_backbone_blocks = compact or cp > 1
            result._reuse_head_projections = True
            result._reuse_context_teacher_projection = compact
            result._use_l1_kernel_optimizations = True
            result._head_projection_cache_chunks = 2
        result.draft_model.freeze_embedding()
        return result.cuda().bfloat16()

    actual, reference = model(True), model(False)
    if manifest:
        actual.load_state_dict(torch.load(args.initial, mmap=True, weights_only=True))
    reference.load_state_dict(actual.state_dict())

    def optimizer(current):
        return BF16Optimizer(current.draft_model, lr=6e-4, weight_decay=0,
                             max_grad_norm=1.0, total_steps=1000, warmup_ratio=0.04,
                             min_lr=0.0)

    optimizers = [optimizer(current) for current in (actual, reference)]
    lengths = (96, 137, 44, 89, 65, 111)
    items = []
    for index, length in enumerate(lengths):
        mask = torch.ones(1, length, device="cuda")
        mask[:, :5] = 0
        if index < 2:
            mask.zero_()
        if index in (2, 4):
            mask.zero_()
            mask[:, 8:10 if index == 2 else 12] = 1
        left, right = length * rank // cp, length * (rank + 1) // cp
        # Generate full identical tensors on every rank before owner slicing.
        items.append({"input_ids": torch.randint(0, 8191, (1, length), device="cuda"),
                      "loss_mask": mask,
                      "hidden_states": torch.randn(1, length, 512, device="cuda", dtype=torch.bfloat16)[:, left:right],
                      "last_hidden_states": torch.randn(1, length, 512, device="cuda", dtype=torch.bfloat16)[:, left:right],
                      "_aspec_work": {"work_id": str(index), "execution_id": str(index)}})
    batch = DFlashPackingCollator(sum(lengths), pad_to_max_length=False,
                                 context_parallel_rank=rank, context_parallel_size=cp)(items)
    head = torch.randn(8192, 512, device="cuda", dtype=torch.bfloat16) * 0.03
    inputs = dict(input_ids=batch["input_ids"], hidden_states_list=batch["hidden_states"],
                  loss_mask=batch["loss_mask"], lm_head_weight=head,
                  last_hidden_states=batch["last_hidden_states"], ctx_doc_ids=batch["ctx_doc_ids"],
                  base_position_ids=batch["base_position_ids"],
                  packed_context_positions=batch.get("packed_context_positions"),
                  packed_document_lengths=lengths)
    if manifest:
        from aspec.models.target.target_utils import load_synced_target_lm_head
        from tools.performance.torchspec_benchmark import natural_packing_schedule, sha256

        target_head = load_synced_target_lm_head(manifest["target"], load_norm=False)
        head = target_head.lm_head.weight
        if args.layout_mode == "non-packing":
            n = len(manifest["documents"])
            # Span the length range even for MBS1 in a 16-state check.
            schedule = [tuple((step * 17 + offset) % n for offset in range(args.micro_batch_size))
                        for step in range(16)]
        else:
            schedule = natural_packing_schedule(manifest["documents"], pool_size=32,
                                                max_seq_length=args.max_sequence_length, batches=16)
        resident = []
        for record in manifest["documents"]:
            path = args.features / record["path"]
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Feature identity changed: {path}")
            resident.append(torch.load(path, mmap=True, weights_only=True))
    records = []
    for step in range(16):
        if manifest:
            row = []
            for index in schedule[step]:
                data = resident[index]
                length = data["input_ids"].shape[1]
                left, right = length * rank // cp, length * (rank + 1) // cp
                row.append({key: (value[:, left:right] if key in ("hidden_states", "last_hidden_states")
                                  else value).cuda() for key, value in data.items()})
                row[-1]["_aspec_work"] = {"work_id": str(index), "execution_id": str(index)}
            batch = (DataCollatorWithPadding(direct_cp_enabled=cp > 1)(row)
                     if args.layout_mode == "non-packing" else DFlashPackingCollator(
                         args.max_sequence_length, pad_to_max_length=False,
                         context_parallel_rank=rank, context_parallel_size=cp)(row))
            inputs = dict(input_ids=batch["input_ids"], hidden_states_list=batch["hidden_states"],
                          loss_mask=batch["loss_mask"], lm_head_weight=head,
                          last_hidden_states=batch["last_hidden_states"], ctx_doc_ids=batch.get("ctx_doc_ids"),
                          base_position_ids=batch.get("base_position_ids"),
                          packed_context_positions=batch.get("packed_context_positions"),
                          packed_document_lengths=batch.get("packed_document_lengths"),
                          attention_mask=batch.get("attention_mask") if args.layout_mode == "non-packing" else None)
        elif args.layout_mode == "non-packing":
            row = [items[(step * args.micro_batch_size + offset) % len(items)]
                   for offset in range(args.micro_batch_size)]
            batch = DataCollatorWithPadding(direct_cp_enabled=cp > 1)(row)
            inputs = dict(input_ids=batch["input_ids"], hidden_states_list=batch["hidden_states"],
                          loss_mask=batch["loss_mask"], lm_head_weight=head,
                          last_hidden_states=batch["last_hidden_states"],
                          attention_mask=batch["attention_mask"])
        if args.comparison == "common":
            reference.load_state_dict(actual.state_dict())
            OptimizerState(reference.draft_model, optimizers[1]).load_state_dict(
                copy.deepcopy(OptimizerState(actual.draft_model, optimizers[0]).state_dict()))
            optimizers[1].scheduler.load_state_dict(optimizers[0].scheduler.state_dict())
        outputs, gradients, norms, layouts = [], [], [], []
        learning_rate = optimizers[0].get_learning_rate()
        for current, current_optimizer in zip((actual, reference), optimizers, strict=True):
            current_optimizer.zero_grad(set_to_none=True)
            # Force the layout pair for numerical checks, including small CP
            # fixtures which intentionally do not amortize folding's cost.
            context = patch("aspec.models.ops.cudnn_flex_attention._prefer_thd_physical_gqa",
                            return_value=current is actual) if args.reference_stage == "gqa-layout" else nullcontext()
            with context:
                current_inputs = inputs
                if args.reference_stage == "padding" and current is reference:
                    # Both sides sample the same document-valid anchors. Only
                    # physical input/context padding differs, not the objective.
                    import torch.nn.functional as F

                    pad = -inputs["input_ids"].shape[1] % 128
                    current_inputs = dict(inputs)
                    for key in ("input_ids", "loss_mask", "attention_mask"):
                        current_inputs[key] = F.pad(inputs[key], (0, pad))
                    for key in ("hidden_states_list", "last_hidden_states"):
                        current_inputs[key] = F.pad(inputs[key], (0, 0, 0, pad))
                out = current(**current_inputs, anchor_seed=101 + step)
                totals = torch.stack([out[6][0].detach().float(), out[6][1].detach().float()])
                dist.all_reduce(totals)
                (out[6][0] / totals[1]).backward()
            if args.reference_stage == "gqa-layout":
                observed_layouts = [layer.self_attn.attention_observation.get("gqa_layout")
                                    for layer in current.draft_model.layers]
                expected_layout = "query_head_fold" if current is actual else "native"
                assert observed_layouts and all(value == expected_layout for value in observed_layouts), observed_layouts
                layouts.append(observed_layouts)
            for name, parameter in current.named_parameters():
                if parameter.requires_grad:
                    if name.startswith("draft_model.confidence_head.") and args.confidence_alpha <= 0:
                        assert parameter.grad is None, name
                        continue
                    assert parameter.grad is not None, name
                    dist.all_reduce(parameter.grad)
            gradient = [p.grad for p in current.parameters() if p.grad is not None]
            gradients.append(gradient)
            norms.append(float(torch.stack([value.float().norm() for value in gradient]).norm()))
            outputs.append(totals)
        torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=0)
        losses = [out[0] / out[1] for out in outputs]
        record = {"step": step + 1, "loss": float(losses[0]), "reference_loss": float(losses[1]),
                  "loss_absolute_diff": float((losses[0] - losses[1]).abs()),
                  "loss_relative_diff": relative(losses[0], losses[1]),
                  "gradient_relative_l2": relative(*gradients), "grad_norms": norms,
                  "learning_rate": learning_rate,
                  "denominator": float(outputs[0][1]),
                  "attention_layouts": layouts,
                  "local_output_blocks": actual.loss_head_blocks_observed,
                  "local_backbone_blocks": actual.backbone_blocks_observed,
                  "local_cached_teacher_blocks": actual.loss_head_cached_teacher_blocks_observed,
                  "local_cached_draft_blocks": actual.loss_head_cached_draft_blocks_observed,
                  "local_cached_context_tokens": actual.loss_head_cached_context_tokens_observed,
                  "local_original_blocks": actual.loss_head_original_blocks_observed}
        for current_optimizer in optimizers:
            current_optimizer.step()
        parameters = [[p.detach() for p in m.parameters() if p.requires_grad]
                      for m in (actual, reference)]
        record["parameter_relative_l2"] = relative(*parameters)
        records.append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)
        assert record["loss_relative_diff"] < 0.003, record
        assert record["gradient_relative_l2"] < 0.03, record
        assert record["parameter_relative_l2"] < 0.003, record
        del outputs, gradients, parameters, gradient, out
        if step == 7:
            state = copy.deepcopy(actual.state_dict())
            moments = copy.deepcopy(OptimizerState(actual.draft_model, optimizers[0]).state_dict())
            scheduler = copy.deepcopy(optimizers[0].scheduler.state_dict())
            rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            actual = model(True)
            actual.load_state_dict(state)
            optimizers[0] = optimizer(actual)
            OptimizerState(actual.draft_model, optimizers[0]).load_state_dict(moments)
            optimizers[0].scheduler.load_state_dict(scheduler)
            restored = OptimizerState(actual.draft_model, optimizers[0]).state_dict()
            for name in moments["fp32_params"]:
                torch.testing.assert_close(restored["fp32_params"][name], moments["fp32_params"][name], rtol=0, atol=0)
            for index, fields in moments["optim"]["state"].items():
                for name, value in fields.items():
                    torch.testing.assert_close(restored["optim"]["state"][index][name], value, rtol=0, atol=0)
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state(rng[1])
            del state, moments, scheduler, restored, rng
        if manifest:
            # Release the previous packed row before materializing the next one.
            # Correctness comparison holds two full models, unlike the benchmark.
            inputs.clear()
            row.clear()
            del batch
    if rank == 0:
        Path(args.result).write_text(json.dumps({"cp": cp, "precision": "bf16",
                                                "comparison": args.comparison,
                                                "reference_stage": args.reference_stage,
                                                "layout_mode": args.layout_mode,
                                                "micro_batch_size": args.micro_batch_size,
                                                "optimizer": "native BF16Optimizer with FP32 masters",
                                                "confidence_alpha": args.confidence_alpha,
                                                "attention_backend": args.attention_backend,
                                                "sliding_window": args.sliding_window,
                                                "features": str(args.features) if manifest else None,
                                                "initial_sha256": sha256(args.initial) if manifest else None,
                                                "draft_layers": config.num_hidden_layers,
                                                "hidden_size": config.hidden_size,
                                                "vocab_size": config.vocab_size,
                                                "steps": records, "restored_after_step": 8}, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--comparison", choices=("common", "independent"), default="common")
    parser.add_argument("--reference-stage", choices=("dense", "block-head", "context-head", "spare-layout", "indexed-head", "gqa-layout", "head-replay", "bshd-native", "padding"), default="dense")
    parser.add_argument("--layout-mode", choices=("natural-packing", "non-packing"), default="natural-packing")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--num-anchors", type=int, default=8)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--confidence-alpha", type=float, default=1.0)
    parser.add_argument("--attention-backend", choices=("cudnn_flex_attention", "fa4"),
                        default="cudnn_flex_attention")
    parser.add_argument("--sliding-window", type=int)
    run(parser.parse_args())
