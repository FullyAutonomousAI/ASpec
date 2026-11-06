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

"""Separate CP1 semantics, packed grouping and native kernels on real inputs.

All modes use identical initial weights, teacher tensors and active anchors.
Per-document Flex is a diagnostic control, not a new production training mode.
Native packed modes preserve the trainer's numerator/2 then 2/denominator rule.
"""

import argparse
import importlib
import importlib.machinery
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from tools.validation.dspark_torchspec_alignment import (
    UPSTREAM_DSPARK_SHA,
    document_batches,
    loss_record,
    pipeline,
)
from tools.validation.dspark_update_alignment import cpu_tensor, save_new


def to_device(values, device):
    import torch

    return {
        key: [item.to(device) for item in value]
        if isinstance(value, list)
        else value.to(device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in values.items()
    }


def build_model(fixture, namespace, backend, cp=1):
    import torch

    alias = sys.modules.get("cutlass.utils.ampere_helpers")
    if alias is not None and alias.__spec__ is None:
        alias.__spec__ = importlib.machinery.ModuleSpec("cutlass.utils.ampere_helpers", loader=None)
    draft_module = importlib.import_module(namespace + ".models.draft.dspark")
    wrapper_module = importlib.import_module(namespace + ".models.dspark")
    if (
        namespace == "torchspec"
        and pipeline.file_sha(wrapper_module.__file__) != UPSTREAM_DSPARK_SHA
    ):
        raise ValueError("TorchSpec reference source changed")
    manifest = json.loads((Path(fixture) / "manifest.json").read_text())
    if pipeline.file_sha(Path(fixture) / "initial-state.pt") != manifest["initial_state_sha256"]:
        raise ValueError("Initial fixture weights changed")
    draft = draft_module.DSparkDraftModel(draft_module.DSparkConfig(**manifest["draft_config"]))
    draft.freeze_embedding()
    extra = (
        {"attention_backend": backend, "draft_context_parallel_size": cp}
        if namespace == "aspec"
        else {}
    )
    model = wrapper_module.DSparkModel(
        draft, block_size=7, num_anchors=manifest["anchors"], **extra
    )
    model.load_state_dict(
        torch.load(Path(fixture) / "initial-state.pt", weights_only=True), strict=True
    )
    return model.to(device="cuda", dtype=torch.bfloat16).train()


def rows_for_mode(packed, per_document):
    return [row for batch in packed for row in document_batches(batch)] if per_document else packed


def wrap_native_fsdp(model, fixture, namespace):
    """Reuse the repository's actual FSDP precision and layer-sharding policy."""
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from torch.distributed.device_mesh import init_device_mesh

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    size = dist.get_world_size()
    if size != getattr(model, "draft_context_parallel_size", 1):
        raise ValueError("This control uses one DP replica across its CP group")
    config = OmegaConf.load(Path(fixture) / "fast.yaml").training
    mesh = init_device_mesh("cuda", (size,), mesh_dim_names=("dp_with_cp",))
    module = importlib.import_module(namespace + ".training.fsdp")
    return module.apply_fsdp2(
        model, mesh=mesh, args=config, modules_to_shard=list(model.draft_model.layers)
    )


def native_window(model, packed, head, namespace, per_document, optimizer):
    """Call native backward/normalization, including FP32 tensor coefficients."""
    import torch.distributed as dist

    trainer_class = importlib.import_module(namespace + ".training.dflash_trainer").DFlashTrainer
    trainer = SimpleNamespace(optimizer=optimizer, dp_group=dist.group.WORLD)
    device = next(model.parameters()).device
    rows = rows_for_mode(packed, per_document)
    totals, denominator_sum = [], None
    for index, row in enumerate(rows):
        model.set_requires_gradient_sync(index == len(rows) - 1)
        values = to_device(row, device)
        if per_document:
            anchors, keep = values.pop("anchors"), values.pop("keep")
            if namespace == "torchspec":
                model._sample_anchor_positions = lambda seq, mask, dev: (anchors, keep)
            else:
                values.update(injected_anchors=anchors, injected_keep_mask=keep)
        out = model(**values, lm_head_weight=head)
        numerator, denominator = out[6]
        if denominator_sum is None:
            denominator_sum = denominator.detach().clone()
        else:
            denominator_sum.add_(denominator.detach())
        trainer_class._backward(trainer, numerator, len(rows))
        totals.append(
            [
                float(numerator.detach()),
                float(denominator),
                *[float(out[5][key][0]) for key in ("ce_loss", "l1_loss", "confidence_loss")],
            ]
        )
        del out, values
    trainer_class._normalize_accumulated_gradients(trainer, denominator_sum, len(rows))
    return loss_record(totals)


def probe_mode(fixture, captured, output, mode, step):
    import torch

    namespace = "torchspec" if mode.startswith("torch-doc-flex") else "aspec"
    per_document = "doc-flex" in mode
    backend = "fa4" if "fa4" in mode else "flex_attention"
    model = build_model(fixture, namespace, backend)
    native_fsdp = mode.endswith("-fsdp")
    if native_fsdp:
        model = wrap_native_fsdp(model, fixture, namespace)
    device = next(model.parameters()).device
    head = torch.load(Path(captured) / "inputs/lm-head.pt", weights_only=True).to(device)
    paths = sorted((Path(captured) / "inputs").glob(f"step-{step:04d}-micro-*.pt"))
    if len(paths) != 2:
        raise ValueError("Both actual packed microbatches are required")
    packed = [torch.load(path, weights_only=True) for path in paths]
    totals = []
    rows = [] if native_fsdp else rows_for_mode(packed, per_document)
    for row in rows:
        values = to_device(row, device)
        if per_document:
            anchors, keep = values.pop("anchors"), values.pop("keep")
            if namespace == "torchspec":
                model._sample_anchor_positions = lambda seq, mask, dev: (anchors, keep)
            else:
                values.update(injected_anchors=anchors, injected_keep_mask=keep)
        out = model(**values, lm_head_weight=head)
        numerator, denominator = out[6]
        (numerator if per_document else numerator / 2).backward()
        totals.append(
            [
                float(numerator.detach()),
                float(denominator),
                *[float(out[5][key][0]) for key in ("ce_loss", "l1_loss", "confidence_loss")],
            ]
        )
        del out, values
    record = (
        native_window(
            model,
            packed,
            head,
            namespace,
            per_document,
            SimpleNamespace(model_params=[p for p in model.parameters() if p.requires_grad]),
        )
        if native_fsdp
        else loss_record(totals)
    )
    gradients = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if parameter.grad is None:
                    raise ValueError(f"Missing gradient: {name}")
                if native_fsdp:
                    pass  # Already normalized by the unmodified native trainer method.
                elif per_document:
                    parameter.grad.div_(record["denominator"])
                else:
                    parameter.grad.mul_(2 / record["denominator"])
                gradients[name] = cpu_tensor(parameter.grad)
    record.update(
        mode=mode,
        step=step,
        forwards=len(rows_for_mode(packed, per_document)),
        input_sha256=[pipeline.file_sha(path) for path in paths],
        initial_state_sha256=pipeline.file_sha(Path(fixture) / "initial-state.pt"),
        gradient_dtypes=sorted({str(value.dtype) for value in gradients.values()}),
        gradient_numel=sum(value.numel() for value in gradients.values()),
        production_semantics_changed=False,
        diagnostic_execution="Native FSDP2/trainer normalization; no target regeneration or optimizer update"
        if native_fsdp
        else "Legacy plain BF16 control; Python scalar normalization differs from native trainer",
        harness_sha256=pipeline.file_sha(__file__),
    )
    save_new(gradients, Path(output) / f"{mode}-gradients.pt")
    (Path(output) / f"{mode}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if native_fsdp:
        import torch.distributed as dist

        dist.destroy_process_group()


def gradient_difference(reference, actual):
    """Compare every coordinate with bounded temporary memory."""
    if reference.keys() != actual.keys():
        raise ValueError("Gradient parameter sets differ")
    totals = dict(square_diff=0.0, square_ref=0.0, square_actual=0.0, unequal=0, numel=0)
    parameters = {}
    for name, ref in reference.items():
        val = actual[name]
        if ref.shape != val.shape:
            raise ValueError(f"Gradient shapes differ: {name}")
        row = dict(square_diff=0.0, square_ref=0.0, square_actual=0.0, unequal=0, numel=ref.numel())
        for left, right in zip(
            ref.flatten().split(1_000_000), val.flatten().split(1_000_000), strict=True
        ):
            row["unequal"] += int((left != right).sum())
            left, right = left.double(), right.double()
            row["square_diff"] += float((right - left).square().sum())
            row["square_ref"] += float(left.square().sum())
            row["square_actual"] += float(right.square().sum())
        row["relative_l2_diff"] = math.sqrt(row["square_diff"] / max(row["square_ref"], 1e-30))
        parameters[name] = row
        for key in totals:
            totals[key] += row[key]
    totals["relative_l2_diff"] = math.sqrt(totals["square_diff"] / max(totals["square_ref"], 1e-30))
    return {**totals, "parameters": parameters}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--captured", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=(
            "torch-doc-flex",
            "fast-doc-flex",
            "fast-pack-flex",
            "fast-pack-fa4",
            "fast-pack-fa4-repeat",
            "torch-doc-flex-fsdp",
            "fast-doc-flex-fsdp",
            "fast-pack-flex-fsdp",
            "fast-pack-fa4-fsdp",
        ),
        required=True,
    )
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / f"{args.mode}.json").exists():
        raise FileExistsError(root / f"{args.mode}.json")
    probe_mode(args.fixture, args.captured, root, args.mode, args.step)


if __name__ == "__main__":
    main()
