# Copyright 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-GPU EAGLE3 train-core benchmark for paired ASpec/TorchSpec runs.

This deliberately measures one optimizer step rather than the Target producer,
transport, or orchestration.  It uses the Qwen3-8B EAGLE3 model dimensions and
records per-step loss and gradient norms so a faster result cannot silently
change the training trajectory.
"""

import argparse
import importlib
import json
import os
import statistics
import tempfile
import time

import torch
import torch.distributed as dist
from transformers import LlamaConfig


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _summary(values):
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "p10_ms": _percentile(values, 0.10),
        "p90_ms": _percentile(values, 0.90),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("aspec", "torchspec"), required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", default="flex_attention")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--cache-target-logits", action="store_true")
    parser.add_argument("--disable-loss-checkpoint", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.framework != "aspec" and (
        args.cache_target_logits or args.disable_loss_checkpoint
    ):
        parser.error("ASpec-only optimizations cannot be selected for TorchSpec")
    if args.seq_len <= 16 or args.warmup < 1 or args.steps < 1:
        parser.error("seq-len must exceed 16 and warmup/steps must be positive")
    return args


def main():
    args = _arguments()
    prefix = args.framework
    draft_module = importlib.import_module(f"{prefix}.models.draft.llama3_eagle")
    eagle_module = importlib.import_module(f"{prefix}.models.eagle3")
    optimizer_module = importlib.import_module(f"{prefix}.training.optimizer")

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    config = LlamaConfig(
        architectures=["LlamaForCausalLMEagle3"],
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        draft_vocab_size=151936,
        target_hidden_size=4096,
        max_position_embeddings=max(40960, args.seq_len + 16),
        pad_token_id=0,
        rms_norm_eps=1e-6,
        pretraining_tp=1,
        num_aux_hidden_states=5,
        rope_theta=1_000_000,
    )

    with tempfile.TemporaryDirectory(prefix=f"{prefix}-eagle3-bench-") as directory:
        dist.init_process_group(
            "nccl", init_method=f"file://{directory}/store", rank=0, world_size=1
        )
        try:
            draft = draft_module.LlamaForCausalLMEagle3(
                config, attention_backend=args.backend
            ).cuda().bfloat16()
            draft.freeze_embedding()
            model_options = {
                "length": 3,
                "attention_backend": args.backend,
                "gradient_checkpointing": True,
                "loss_type": "forward_kl",
            }
            if prefix == "aspec":
                model_options["loss_gradient_checkpointing"] = not (
                    args.disable_loss_checkpoint
                )
            model = eagle_module.Eagle3Model(draft, **model_options)
            optimizer = optimizer_module.BF16Optimizer(
                draft,
                lr=6e-4,
                weight_decay=0.0,
                total_steps=args.warmup + args.steps + 8,
                warmup_ratio=0.0,
            )

            batch = 1
            sequence = args.seq_len
            torch.manual_seed(args.seed + 1)
            inputs = {
                "input_ids": torch.randint(
                    0, config.vocab_size, (batch, sequence), device="cuda"
                ),
                "attention_mask": torch.ones(
                    batch, sequence, dtype=torch.long, device="cuda"
                ),
                "hidden_states": torch.randn(
                    batch,
                    sequence,
                    config.hidden_size * config.num_aux_hidden_states,
                    dtype=torch.bfloat16,
                    device="cuda",
                ),
                "loss_mask": torch.ones(
                    batch, sequence, dtype=torch.float32, device="cuda"
                ),
            }
            inputs["loss_mask"][:, :16] = 0
            teacher_hidden = torch.randn(
                batch,
                sequence,
                config.hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
            )
            teacher_head = (
                torch.randn(
                    config.vocab_size,
                    config.hidden_size,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                * 0.01
            )
            lazy_target = eagle_module.compute_lazy_target_padded(
                teacher_hidden, teacher_head, 3
            )
            weights = (1.0, 0.8, 0.64)

            def run_step(timed, step=-1):
                use_ranges = args.profile and timed
                if use_ranges:
                    torch.cuda.nvtx.range_push(f"train_step_{step}")
                optimizer.zero_grad()
                events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                events[0].record()
                if use_ranges:
                    torch.cuda.nvtx.range_push("forward_and_loss")
                target = (
                    eagle_module.compute_cached_target_logits_padded(
                        teacher_hidden, teacher_head, 3
                    )
                    if args.cache_target_logits
                    else lazy_target
                )
                outputs = model(**inputs, target=target)
                loss = sum(
                    weight * value
                    for weight, value in zip(weights, outputs[0], strict=True)
                )
                if use_ranges:
                    torch.cuda.nvtx.range_pop()
                events[1].record()
                if use_ranges:
                    torch.cuda.nvtx.range_push("backward")
                loss.backward()
                if use_ranges:
                    torch.cuda.nvtx.range_pop()
                events[2].record()
                if use_ranges:
                    torch.cuda.nvtx.range_push("optimizer")
                gradient_norm = optimizer.step()
                if use_ranges:
                    torch.cuda.nvtx.range_pop()
                events[3].record()
                torch.cuda.synchronize()
                if use_ranges:
                    torch.cuda.nvtx.range_pop()
                if not timed:
                    return None
                counts = [float(value) for value in outputs[3]]
                return {
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                    "counts": counts,
                    "forward_ms": events[0].elapsed_time(events[1]),
                    "backward_ms": events[1].elapsed_time(events[2]),
                    "optimizer_ms": events[2].elapsed_time(events[3]),
                    "total_ms": events[0].elapsed_time(events[3]),
                }

            cold_start = time.perf_counter()
            run_step(False)
            cold_first_step_seconds = time.perf_counter() - cold_start
            for _ in range(args.warmup - 1):
                run_step(False)
            torch.cuda.reset_peak_memory_stats()
            if args.profile:
                torch.cuda.nvtx.range_push("profile_window")
            records = [run_step(True, step) for step in range(args.steps)]
            if args.profile:
                torch.cuda.nvtx.range_pop()

            result = {
                "framework": prefix,
                "boundary": "single_gpu_eagle3_train_core",
                "backend": args.backend,
                "dtype": "bfloat16",
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "batch_size": batch,
                "sequence_length": sequence,
                "ttt_length": 3,
                "gradient_checkpointing": True,
                "loss_gradient_checkpointing": not args.disable_loss_checkpoint,
                "cache_target_logits": args.cache_target_logits,
                "hidden_size": config.hidden_size,
                "intermediate_size": config.intermediate_size,
                "heads": config.num_attention_heads,
                "kv_heads": config.num_key_value_heads,
                "vocab_size": config.vocab_size,
                "aux_hidden_states": config.num_aux_hidden_states,
                "warmup_steps": args.warmup,
                "timed_steps": args.steps,
                "cold_first_step_seconds": cold_first_step_seconds,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "losses": [record["loss"] for record in records],
                "gradient_norms": [record["gradient_norm"] for record in records],
                "supervised_predictions_per_step": sum(records[0]["counts"]),
                "timing": {
                    name: _summary([record[name] for record in records])
                    for name in (
                        "forward_ms",
                        "backward_ms",
                        "optimizer_ms",
                        "total_ms",
                    )
                },
                "raw_records": records,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0),
                "device_memory_gib": (
                    torch.cuda.get_device_properties(0).total_memory / 2**30
                ),
                "pid": os.getpid(),
            }
            with open(args.output, "w", encoding="utf-8") as destination:
                json.dump(result, destination, indent=2)
                destination.write("\n")
            print(json.dumps(result, indent=2))
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
