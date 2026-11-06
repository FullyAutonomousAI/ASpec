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

"""Explain the native confidence-head gradient on fixed real teacher features.

Read-only diagnostic: retain native BF16 backward and independently reconstruct
the pooled linear-head gradient from its actual features and BCE residuals.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from tools.validation.dspark_nonpacking_alignment import vector_comparison
from tools.validation.dspark_trajectory_probe import (
    build_model,
    load_state,
    load_target_head,
    prepare_rows,
)


def confidence_weights(row):
    import torch

    anchors, keep = row["injected_anchors"], row["injected_keep_mask"]
    labels = anchors.unsqueeze(-1) + torch.arange(1, 8, device=anchors.device)
    safe = torch.where(keep.unsqueeze(-1), labels.clamp(max=row["input_ids"].shape[1] - 1), 0)
    loss_mask = torch.gather(row["loss_mask"].unsqueeze(1).expand(-1, 512, -1), 2, safe)
    valid = keep.unsqueeze(-1) & (labels < row["input_ids"].shape[1]) & (loss_mask > 0.5)
    return valid.int().cumprod(-1).float() * torch.exp(-torch.arange(7, device=labels.device) / 4)


def capture(model, rows, head):
    import torch
    import torch.nn.functional as functional

    features, residuals, logits, targets, weights = [], [], [], [], []
    native_bce = functional.binary_cross_entropy_with_logits

    def observed_bce(prediction, target, *args, **kwargs):
        logits.append(prediction.detach().clone())
        targets.append(target.detach().clone())
        residuals.append(torch.sigmoid(prediction.detach().float()) - target.detach())
        return native_bce(prediction, target, *args, **kwargs)

    hook = model.draft_model.confidence_head.register_forward_pre_hook(
        lambda module, values: features.append(values[0].detach().clone())
    )
    try:
        with patch.object(functional, "binary_cross_entropy_with_logits", observed_bce):
            for row in rows:
                values = dict(row)
                anchors, keep = values.pop("injected_anchors"), values.pop("injected_keep_mask")
                model._sample_anchor_positions = lambda seq, mask, dev: (anchors, keep)
                output = model(**values, lm_head_weight=head.lm_head.weight)
                (output[6][0] / len(rows)).backward()
                weights.append(confidence_weights(row))
        denominator = sum(weight.sum() for weight in weights)
        actual = model.draft_model.confidence_head.proj.weight.grad.detach().float()
        # This is the native post-accumulation scale, applied in BF16.
        actual = (actual.to(torch.bfloat16) * (len(rows) / denominator)).float()
        return {
            "features": torch.cat([value.reshape(-1, value.shape[-1]) for value in features]),
            "residual": torch.cat([value.reshape(-1) for value in residuals]),
            "logit": torch.cat([value.reshape(-1) for value in logits]),
            "target": torch.cat([value.reshape(-1) for value in targets]),
            "weight": torch.cat([value.reshape(-1) for value in weights]),
            "actual_gradient": actual.reshape(-1),
        }
    finally:
        hook.remove()


def projected_gradient(features, residual, weight):
    return ((weight * residual).unsqueeze(0) @ features.float()).squeeze(0) / weight.sum()


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "fast", "reference", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Use a fresh diagnostic output")
    torch.set_num_threads(8)
    fixture = Path(args.fixture)
    config = json.loads((fixture / "manifest.json").read_text())["draft_config"]
    head = load_target_head(fixture)
    rows, identities = prepare_rows(
        next((fixture / "cache/eval_cache").glob("*/eval_rank_0.pt")), 4, head
    )
    captures = {}
    for name, path in (("fast", args.fast), ("reference", args.reference)):
        model = build_model("torchspec", "flex_attention", config, load_state(path), "confidence")
        captures[name] = capture(model, rows, head)
        del model
        torch.cuda.empty_cache()
    fast, reference = captures["fast"], captures["reference"]
    if not torch.equal(fast["weight"], reference["weight"]):
        raise ValueError("Confidence supervision differs")
    weight = reference["weight"]
    ref_grad = projected_gradient(reference["features"], reference["residual"], weight)
    fast_grad = projected_gradient(fast["features"], fast["residual"], weight)
    result = {
        "documents": identities,
        "backend": "native TorchSpec Flex for both states",
        "no_updates": True,
        "comparisons": {},
        "statistics": {},
    }
    for name, value in captures.items():
        oracle = projected_gradient(value["features"], value["residual"], weight)
        result["statistics"][name] = {
            "weighted_mean_target": float((weight * value["target"]).sum() / weight.sum()),
            "weighted_mean_probability": float(
                (weight * torch.sigmoid(value["logit"])).sum() / weight.sum()
            ),
            "weighted_mean_residual": float((weight * value["residual"]).sum() / weight.sum()),
            "weighted_residual_rms": float(
                ((weight * value["residual"].square()).sum() / weight.sum()).sqrt()
            ),
            "actual_vs_fp32_gradient": vector_comparison([value["actual_gradient"]], [oracle]),
        }
    variants = {
        "both_changed": fast_grad,
        "only_residual_changed": projected_gradient(
            reference["features"], fast["residual"], weight
        ),
        "only_features_changed": projected_gradient(
            fast["features"], reference["residual"], weight
        ),
    }
    for name, value in variants.items():
        result["comparisons"][name] = vector_comparison([value], [ref_grad])
    result["comparisons"]["sigmoid_probability"] = vector_comparison(
        [torch.sigmoid(fast["logit"]) * weight.sqrt()],
        [torch.sigmoid(reference["logit"]) * weight.sqrt()],
    )
    result["comparisons"]["target_acceptance"] = vector_comparison(
        [fast["target"] * weight.sqrt()], [reference["target"] * weight.sqrt()]
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
