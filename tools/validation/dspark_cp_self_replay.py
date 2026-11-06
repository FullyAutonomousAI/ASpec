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

"""Native ASpec CP self-control on sixteen saved real optimizer windows.

Reuse the real per-document owner collator, native FSDP2, trainer normalization
and BF16Optimizer. No target regeneration, precision change or benchmark claim.
"""

import argparse
import importlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from tools.validation.dspark_cp1_probe import build_model, native_window, wrap_native_fsdp
from tools.validation.dspark_torchspec_alignment import pipeline
from tools.validation.dspark_update_alignment import cpu_tensor, save_new


def owner_inputs(row, rank, size):
    """Reassemble the exact producer shards through the production collator."""
    import torch

    from aspec.data.utils import DFlashPackingCollator

    docs = row["ctx_doc_ids"][0]
    features = []
    width = row["hidden_states_list"][0].shape[-1]
    for doc in docs.unique(sorted=True).tolist():
        if doc < 0:
            continue
        positions = (docs == doc).nonzero().flatten()
        start, stop = int(positions[0]), int(positions[-1]) + 1
        if positions.numel() != stop - start:
            raise ValueError("Noncontiguous source document")
        length = stop - start
        left, right = start + length * rank // size, start + length * (rank + 1) // size
        features.append(
            {
                "input_ids": row["input_ids"][:, start:stop],
                "loss_mask": row["loss_mask"][:, start:stop],
                "hidden_states": torch.cat(
                    [value[:, left:right] for value in row["hidden_states_list"]], -1
                ),
                "last_hidden_states": row["last_hidden_states"][:, left:right],
                "_aspec_work": {"work_id": f"doc{doc}", "execution_id": f"doc{doc}-e0"},
            }
        )
    collated = DFlashPackingCollator(
        row["input_ids"].shape[1],
        pad_to_max_length=False,
        context_parallel_rank=rank,
        context_parallel_size=size,
    )(features)
    result = {
        key: collated[key]
        for key in (
            "input_ids",
            "loss_mask",
            "ctx_doc_ids",
            "base_position_ids",
            "last_hidden_states",
        )
    }
    result["hidden_states_list"] = list(collated["hidden_states"].split(width, -1))
    if "packed_context_positions" in collated:
        result["packed_context_positions"] = collated["packed_context_positions"]
    result.update(
        injected_anchors=row["injected_anchors"],
        injected_keep_mask=row["injected_keep_mask"],
        anchor_seed=row.get("anchor_seed"),
    )
    for key in ("input_ids", "loss_mask", "ctx_doc_ids", "base_position_ids"):
        if not torch.equal(row[key], result[key]):
            raise ValueError(f"Owner replay changed global {key}")
    return result


def run(args):
    import torch
    import torch.distributed as dist

    from aspec.utils.distributed import init_direct_cp_groups

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    if dist.get_world_size() != args.cp:
        raise ValueError("Self-control requires one DP replica with world size equal to CP")
    init_direct_cp_groups(args.cp)
    root = Path(args.output)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    model = wrap_native_fsdp(
        build_model(args.fixture, "aspec", "fa4", args.cp), args.fixture, "aspec"
    )
    config = OmegaConf.load(Path(args.fixture) / "fast.yaml").training
    optimizer = importlib.import_module("aspec.training.optimizer").BF16Optimizer(
        model.draft_model,
        lr=config.learning_rate,
        min_lr=config.min_lr,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        total_steps=config.lr_total_steps,
        warmup_ratio=config.warmup_ratio,
    )
    captured = Path(args.captured)
    inputs = captured / "inputs"
    paths = sorted(inputs.glob("step-*-micro-*.pt"))
    if len(paths) != 32:
        raise ValueError("All sixteen saved optimizer windows are required")
    head = torch.load(inputs / "lm-head.pt", weights_only=True).to("cuda")
    reference = [
        json.loads(line) for line in (captured / "loss-steps.jsonl").read_text().splitlines()
    ]
    records = []
    audit = None
    if args.audit_steps is not None:
        from tools.validation.dspark_cp_state_audit import NativeUpdateAudit

        audit = NativeUpdateAudit(model, optimizer, root, args.cp, args.audit_steps)
    for step in range(1, 17):
        optimizer.zero_grad()
        packed = [
            owner_inputs(torch.load(path, weights_only=True), rank, args.cp)
            for path in paths
            if path.name.startswith(f"step-{step:04d}-")
        ]
        if len(packed) != 2:
            raise ValueError("Incomplete optimizer window")
        local = native_window(model, packed, head, "aspec", False, optimizer)
        metrics = ("loss", "ce", "l1", "confidence")
        totals = torch.tensor(
            [local["denominator"], *[local[key] * local["denominator"] for key in metrics]],
            device="cuda",
            dtype=torch.float64,
        )
        dist.all_reduce(totals)
        denominator = float(totals[0])
        record = {key: float(totals[index + 1]) / denominator for index, key in enumerate(metrics)}
        record.update(
            step=step,
            cp=args.cp,
            denominator=denominator,
            learning_rate_before_update=optimizer.get_learning_rate(),
        )
        if step == 1:
            gradients = {}
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    full = cpu_tensor(parameter.grad)
                    if rank == 0:
                        gradients[name] = full
            if rank == 0:
                save_new(gradients, root / "first-gradients.pt")
            del gradients
        if audit is not None:
            audit.before(step)
        record["gradient_norm"] = float(optimizer.step())
        if audit is not None:
            audit.after(step, record["gradient_norm"])
        record["vs_native_cp1"] = {
            key: {
                "absolute_diff": abs(record[key] - reference[step - 1][key]),
                "relative_diff": abs(record[key] - reference[step - 1][key])
                / max(abs(reference[step - 1][key]), 1e-12),
            }
            for key in (*metrics, "denominator", "gradient_norm")
        }
        if rank == 0:
            records.append(record)
            with (root / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    if rank == 0:
        summary = {
            "steps": 16,
            "cp": args.cp,
            "diagnostic_only": True,
            "semantics_precision_lr_changed": False,
            "full_coordinate_update_audit": audit is not None,
            "snapshot_steps": args.audit_steps,
            "harness_sha256": pipeline.file_sha(__file__),
            "normalization": "Unmodified native trainer FP32 tensor coefficient and FSDP2",
            "input_sha256": [pipeline.file_sha(path) for path in paths],
            "metrics": {
                key: {
                    "max_absolute_diff": max(
                        row["vs_native_cp1"][key]["absolute_diff"] for row in records
                    ),
                    "max_relative_diff": max(
                        row["vs_native_cp1"][key]["relative_diff"] for row in records
                    ),
                }
                for key in records[0]["vs_native_cp1"]
            },
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--captured", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cp", type=int, required=True)
    parser.add_argument(
        "--audit-steps",
        type=int,
        nargs="*",
        default=None,
        help="Observe all 16 updates; save complete before/after states at these steps",
    )
    args = parser.parse_args()
    if args.cp < 1:
        parser.error("CP must be positive")
    if args.audit_steps is not None and any(step < 2 or step > 16 for step in args.audit_steps):
        parser.error("Full snapshots require existing moments, at steps 2 through 16")
    run(args)


if __name__ == "__main__":
    main()
