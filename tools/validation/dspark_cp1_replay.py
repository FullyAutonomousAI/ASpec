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

"""Replay all sixteen real CP1 optimizer windows without a target regeneration.

TorchSpec native replay checks the saved-input control itself. A one-time real
ASpec state transplant before step three tests early perturbation causality.
ASpec per-document Flex is a semantic control only, not a product mode.
"""

import argparse
import importlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from tools.validation.dspark_cp1_probe import (
    build_model,
    native_window,
    rows_for_mode,
    to_device,
    wrap_native_fsdp,
)
from tools.validation.dspark_torchspec_alignment import loss_record, pipeline


def transplant_plain(model, optimizer, source):
    import torch
    from torch.distributed.tensor import DTensor, distribute_tensor

    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if set(source) != {name for name, _ in named}:
        raise ValueError("Transplant parameter sets differ")
    if len(named) != len(optimizer.model_params):
        raise ValueError("Transplant model/master order differs")
    with torch.no_grad():
        for (name, parameter), expected, master in zip(
            named, optimizer.model_params, optimizer.fp32_params, strict=True
        ):
            if parameter is not expected:
                raise ValueError("Transplant model/master order differs")
            state = optimizer.optimizer.state[master]
            if state["step"].item() != source[name]["optimizer_step"]:
                raise ValueError("Transplant Adam step phase differs")
            for field, target in {
                "model": parameter,
                "master": master,
                "exp_avg": state["exp_avg"],
                "exp_avg_sq": state["exp_avg_sq"],
            }.items():
                value = source[name][field].to(target.device)
                if value.shape != target.shape or value.dtype != target.dtype:
                    raise ValueError(f"Transplant shape/dtype differs: {name}/{field}")
                if isinstance(target, DTensor):
                    value = distribute_tensor(value, target.device_mesh, target.placements)
                target.copy_(value)
                actual = target.to_local() if isinstance(target, DTensor) else target
                expected_value = value.to_local() if isinstance(value, DTensor) else value
                if not torch.equal(actual, expected_value):
                    raise ValueError(f"Transplant was not exact: {name}/{field}")


def replay(fixture, captured, output, mode):
    import torch

    namespace = "aspec" if mode.startswith("fast-doc-flex-native") else "torchspec"
    model = build_model(fixture, namespace, "flex_attention")
    native_fsdp = mode.endswith("-fsdp")
    if native_fsdp:
        model = wrap_native_fsdp(model, fixture, namespace)
    config = OmegaConf.load(Path(fixture) / "fast.yaml").training
    optimizer_class = importlib.import_module(namespace + ".training.optimizer").BF16Optimizer
    optimizer = optimizer_class(
        model,
        lr=config.learning_rate,
        min_lr=config.min_lr,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        total_steps=config.lr_total_steps,
        warmup_ratio=config.warmup_ratio,
    )
    device = next(model.parameters()).device
    root = Path(captured)
    inputs = root / "inputs"
    paths = sorted(inputs.glob("step-*-micro-*.pt"))
    if len(paths) != 32:
        raise ValueError("Exactly sixteen complete real optimizer windows are required")
    head = torch.load(inputs / "lm-head.pt", weights_only=True).to(device)
    native = [
        json.loads(line)["independent"]
        for line in (root / "torchspec-steps.jsonl").read_text().splitlines()
    ]
    if len(native) != 16:
        raise ValueError("Native TorchSpec reference is incomplete")
    reference_label = "legacy-reference-v1"
    if mode == "torch-fast-after2-fsdp":
        native = [
            json.loads(line)
            for line in (root.parent / "replay-torch-native-fsdp-r2/steps.jsonl")
            .read_text()
            .splitlines()
        ]
        if len(native) != 16:
            raise ValueError("Native FSDP control is incomplete")
        reference_label = "native-fsdp-perdoc-control"
    records = []
    for step in range(1, 17):
        if step == 3 and mode.startswith("torch-fast-after2"):
            state_path = root / "fast-audit/state-after-0002.pt"
            transplant_plain(model, optimizer, torch.load(state_path, mmap=True, weights_only=True))
            (Path(output) / "transplant.json").write_text(
                json.dumps(
                    {
                        "before_step": 3,
                        "all_trainable_master_moments_bitwise_equal": True,
                        "learning_rate_precision_semantics_changed": False,
                        "source_sha256": pipeline.file_sha(state_path),
                    },
                    indent=2,
                )
                + "\n"
            )
        optimizer.zero_grad()
        packed = [
            torch.load(path, weights_only=True)
            for path in paths
            if path.name.startswith(f"step-{step:04d}-")
        ]
        if len(packed) != 2:
            raise ValueError(f"Incomplete packed optimizer window {step}")
        totals = []
        for row in [] if native_fsdp else rows_for_mode(packed, True):
            values = to_device(row, device)
            anchors, keep = values.pop("anchors"), values.pop("keep")
            if namespace == "torchspec":
                model._sample_anchor_positions = lambda seq, mask, dev: (anchors, keep)
            else:
                values.update(injected_anchors=anchors, injected_keep_mask=keep)
            out = model(**values, lm_head_weight=head)
            numerator, denominator = out[6]
            numerator.backward()
            totals.append(
                [
                    float(numerator.detach()),
                    float(denominator),
                    *[float(out[5][key][0]) for key in ("ce_loss", "l1_loss", "confidence_loss")],
                ]
            )
            del out, values
        record = (
            native_window(model, packed, head, namespace, True, optimizer)
            if native_fsdp
            else loss_record(totals)
        )
        for parameter in [] if native_fsdp else model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(record["denominator"])
        record["learning_rate_before_update"] = optimizer.get_learning_rate()
        record["gradient_norm"] = float(optimizer.step())
        record.update(step=step, mode=mode)
        record["vs_native_torch"] = {
            key: {
                "absolute_diff": abs(record[key] - native[step - 1][key]),
                "relative_diff": abs(record[key] - native[step - 1][key])
                / max(abs(native[step - 1][key]), 1e-12),
            }
            for key in ("loss", "ce", "l1", "confidence", "denominator", "gradient_norm")
        }
        records.append(record)
        with (Path(output) / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    summary = {
        "steps": 16,
        "mode": mode,
        "inputs": "Exact real teacher and anchors replay; no regeneration",
        "diagnostic_only": True,
        "reference_label": reference_label,
        "normalization": "Unmodified native trainer and FSDP2"
        if native_fsdp
        else "Legacy plain control: Python scalar normalization differs from trainer",
        "harness_sha256": pipeline.file_sha(__file__),
        "metrics": {
            key: {
                "max_absolute_diff": max(
                    row["vs_native_torch"][key]["absolute_diff"] for row in records
                ),
                "max_relative_diff": max(
                    row["vs_native_torch"][key]["relative_diff"] for row in records
                ),
            }
            for key in records[0]["vs_native_torch"]
        },
    }
    (Path(output) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if native_fsdp:
        import torch.distributed as dist

        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--captured", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "torch-native",
            "torch-fast-after2",
            "fast-doc-flex-native",
            "torch-native-fsdp",
            "fast-doc-flex-native-fsdp",
            "torch-fast-after2-fsdp",
        ),
        required=True,
    )
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=False)
    replay(args.fixture, args.captured, args.output, args.mode)


if __name__ == "__main__":
    main()
