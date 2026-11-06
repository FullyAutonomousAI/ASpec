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

"""CP1 full-chain THD vs unchanged TorchSpec per-document BSHD.

Reuse the real target/data/optimizer protocol. The upstream sampler alone is
replaced by exact replay, not its backbone or loss. Common-weight computation
and an independent native BF16Optimizer trajectory are reported separately.
This is validation instrumentation, never a benchmark mode.
"""

import argparse
import importlib
import importlib.machinery
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from tools.validation import dspark_pipeline_alignment as pipeline

UPSTREAM_DSPARK_SHA = "bc5233787317e8e6c63b80b0e83a185ecee1ef7314bbfda1529c0818e5db87e5"


def document_batches(kwargs):
    """Map shared packed anchors to each original document, without resampling."""
    import torch

    docs = kwargs["ctx_doc_ids"]
    if docs.shape[0] != 1:
        raise ValueError("CP1 alignment expects one actual packed row")
    anchors, keep = kwargs["injected_anchors"], kwargs["injected_keep_mask"]
    result = []
    for doc in docs[0].unique(sorted=True).tolist():
        if doc < 0:
            continue
        positions = (docs[0] == doc).nonzero().flatten()
        start, stop = int(positions[0]), int(positions[-1]) + 1
        if len(positions) != stop - start:
            raise ValueError("A document must occupy a contiguous token interval")
        active = keep & (anchors >= start) & (anchors < stop)
        local = torch.where(active, anchors - start, torch.zeros_like(anchors))
        row = {
            "input_ids": kwargs["input_ids"][:, start:stop],
            "loss_mask": kwargs["loss_mask"][:, start:stop],
            "hidden_states_list": [v[:, start:stop] for v in kwargs["hidden_states_list"]],
            "last_hidden_states": kwargs["last_hidden_states"][:, start:stop],
            "anchors": local,
            "keep": active,
        }
        # Own copies: the production fetcher's receive buffers can be reused.
        result.append(
            {
                key: [v.detach().cpu().clone() for v in value]
                if isinstance(value, list)
                else value.detach().cpu().clone()
                for key, value in row.items()
            }
        )
    if sum(int(row["keep"].sum()) for row in result) != int(keep.sum()):
        raise ValueError("Replay lost or duplicated an active anchor")
    return result


def loss_record(totals):
    values = [sum(row[index] for row in totals) for index in range(5)]
    numerator, denominator, ce, l1, confidence = values
    if denominator <= 0 or not all(math.isfinite(v) for v in values):
        raise ValueError("Empty or non-finite reference optimizer window")
    return {
        "loss": numerator / denominator,
        "denominator": denominator,
        "ce": ce / denominator,
        "l1": l1 / denominator,
        "confidence": confidence / denominator,
    }


class TorchReference:
    def __init__(self, trainer, fixture, run_dir):
        import torch
        from omegaconf import OmegaConf

        # Both namespaces install this existing FA4 compatibility alias. Its
        # missing import metadata must not make the second import fail.
        name = "cutlass.utils.ampere_helpers"
        alias = sys.modules.get(name)
        if alias is not None and alias.__spec__ is None:
            alias.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        from torchspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
        from torchspec.models.dspark import DSparkModel
        from torchspec.training.dflash_trainer import DFlashTrainer
        from torchspec.training.fsdp import apply_fsdp2
        from torchspec.training.optimizer import BF16Optimizer

        upstream = importlib.import_module("torchspec.models.dspark")
        if pipeline.file_sha(upstream.__file__) != UPSTREAM_DSPARK_SHA:
            raise ValueError("TorchSpec DSpark source differs from the audited fixed revision")
        manifest = json.loads((Path(fixture) / "manifest.json").read_text())
        config = OmegaConf.load(Path(fixture) / "fast.yaml")
        draft = DSparkDraftModel(DSparkConfig(**manifest["draft_config"]))
        draft.freeze_embedding()
        self.model = DSparkModel(draft, block_size=7, num_anchors=manifest["anchors"])
        self.model.load_state_dict(
            torch.load(Path(fixture) / "initial-state.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        self.model.to(device="cuda", dtype=torch.bfloat16)
        self.model.train()
        self.model = apply_fsdp2(
            self.model,
            mesh=trainer.dp_mesh,
            args=config.training,
            modules_to_shard=list(draft.layers),
        )
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if trainable != manifest["trainable_parameters"]:
            raise ValueError("The reference is not the full default draft")
        self.optimizer = BF16Optimizer(
            self.model,
            lr=config.training.learning_rate,
            min_lr=config.training.min_lr,
            weight_decay=config.training.weight_decay,
            max_grad_norm=config.training.max_grad_norm,
            total_steps=config.training.lr_total_steps,
            warmup_ratio=config.training.warmup_ratio,
        )
        self.trainer, self.run_dir = trainer, Path(run_dir)
        self.native_trainer_class = DFlashTrainer
        self.native_trainer = SimpleNamespace(optimizer=self.optimizer, dp_group=trainer.dp_group)
        self.document_index = 0
        self.denominator_sum = None
        self.documents, self.common_totals = [], []
        self.common_weights_equal = False
        self.lm_head = None
        files = {}
        for module in (
            "models.dspark",
            "models.dflash",
            "models.draft.dspark",
            "models.draft.dflash",
            "training.optimizer",
            "training.dflash_trainer",
            "training.fsdp",
        ):
            source = importlib.import_module("torchspec." + module).__file__
            files[source] = pipeline.file_sha(source)
        (self.run_dir / "torchspec-reference.json").write_text(
            json.dumps(
                {
                    "revision": "6c042a87140a84d13839e341ece2c5c3ada918bc",
                    "trainable_parameters": trainable,
                    "source_files": files,
                    "loss_aggregation": "Sum additive numerators/denominators over original documents",
                    "changed_upstream_behavior": "Exact anchor replay only",
                    "loader_compatibility": "Populate __spec__ on the existing FA4 cutlass alias",
                    "native_precision": "BF16 CE, native FSDP2 accumulation and fused BF16Optimizer",
                    "normalization": "Unmodified TorchSpec trainer backward and FP32 tensor normalization",
                    "protocol_version": 2,
                    "independent_trajectory": True,
                },
                indent=2,
            )
            + "\n"
        )

    def begin(self):
        import torch
        from torch.distributed.tensor import DTensor

        state = {
            key: value.full_tensor() if isinstance(value, DTensor) else value
            for key, value in self.trainer.model.state_dict().items()
        }
        from torchspec.training.fsdp import fsdp2_load_full_state_dict

        # The checkpoint helper replaces values in the supplied mapping with
        # DTensors. Preserve the independent dense expected state for equality.
        fsdp2_load_full_state_dict(self.model, dict(state), self.trainer.dp_mesh, cpu_offload=None)
        own = self.model.state_dict()
        self.common_weights_equal = all(
            torch.equal(
                own[key].full_tensor() if isinstance(own[key], DTensor) else own[key], value
            )
            for key, value in state.items()
        )
        if not self.common_weights_equal:
            raise ValueError("Common-state copy was not bitwise exact")
        self.optimizer.zero_grad()
        self.document_index = 0
        self.denominator_sum = None

    def compute_document(self, row):
        device = next(self.model.parameters()).device
        values = {
            key: [v.to(device) for v in value] if isinstance(value, list) else value.to(device)
            for key, value in row.items()
        }
        anchors, keep = values.pop("anchors"), values.pop("keep")
        self.model._sample_anchor_positions = lambda seq, mask, dev: (anchors, keep)
        self.model.set_requires_gradient_sync(self.document_index == len(self.documents) - 1)
        out = self.model(**values, lm_head_weight=self.lm_head)
        numerator, denominator = out[6]
        self.native_trainer_class._backward(self.native_trainer, numerator, len(self.documents))
        if self.denominator_sum is None:
            self.denominator_sum = denominator.detach().clone()
        else:
            self.denominator_sum.add_(denominator.detach())
        self.document_index += 1
        return [
            float(numerator.detach()),
            float(denominator),
            *[float(out[5][key][0]) for key in ("ce_loss", "l1_loss", "confidence_loss")],
        ]

    def pre_forward(self, module, positional, kwargs):
        # Collect only: nested backward here would run FSDP's global autograd
        # callback while ASpec is between unshard and its own forward.
        self.lm_head = kwargs["lm_head_weight"].detach()
        self.documents.extend(document_batches(kwargs))

    def before_update(self):
        import torch
        from torch.distributed.tensor import DTensor

        # ASpec's backward is complete, but neither optimizer has updated.
        self.begin()
        self.common_totals = [self.compute_document(row) for row in self.documents]
        common = loss_record(self.common_totals)
        self.native_trainer_class._normalize_accumulated_gradients(
            self.native_trainer, self.denominator_sum, len(self.documents)
        )
        fast_parameters = dict(self.trainer.model.named_parameters())
        square_diff = square_ref = square_fast = 0.0
        for name, parameter in self.model.named_parameters():
            reference = parameter.grad
            actual = fast_parameters[name].grad
            if reference is None or actual is None:
                if (reference is None) != (actual is None):
                    raise ValueError(f"Gradient presence differs: {name}")
                continue
            reference = reference.full_tensor() if isinstance(reference, DTensor) else reference
            actual = actual.full_tensor() if isinstance(actual, DTensor) else actual
            ref, val = reference.float(), actual.float()
            square_diff += float((val - ref).square().sum())
            square_ref += float(ref.square().sum())
            square_fast += float(val.square().sum())
        common["gradient_norm"] = math.sqrt(square_ref)
        common["fast_gradient_norm"] = math.sqrt(square_fast)
        common["gradient_relative_l2_diff"] = math.sqrt(square_diff / max(square_ref, 1e-30))
        # Restore the independent reference from its own FP32 masters, not ASpec.
        self.optimizer.zero_grad()
        self.document_index = 0
        self.denominator_sum = None
        with torch.no_grad():
            for parameter, master in zip(
                self.optimizer.model_params, self.optimizer.fp32_params, strict=True
            ):
                parameter.copy_(master)
        independent_totals = [self.compute_document(row) for row in self.documents]
        independent = loss_record(independent_totals)
        self.native_trainer_class._normalize_accumulated_gradients(
            self.native_trainer, self.denominator_sum, len(self.documents)
        )
        lr = self.optimizer.get_learning_rate()
        if abs(lr - self.trainer.optimizer.get_learning_rate()) > 1e-12:
            raise ValueError("Independent optimizers have different LR phases")
        independent["gradient_norm"] = float(self.optimizer.step())
        record = {
            "step": self.trainer.global_step + 1,
            "common_weight": common,
            "independent": independent,
            "common_weights_bitwise_equal": self.common_weights_equal,
            "documents": len(self.documents),
            "learning_rate_before_update": lr,
        }
        with (self.run_dir / "torchspec-steps.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print("TORCHSPEC_ALIGNMENT_STEP " + json.dumps(record), flush=True)
        self.documents.clear()
        self.common_totals.clear()


def compare(run_dir, steps):
    import csv

    root = Path(run_dir)
    fast = [json.loads(s) for s in (root / "loss-steps.jsonl").read_text().splitlines()]
    upstream = [json.loads(s) for s in (root / "torchspec-steps.jsonl").read_text().splitlines()]
    expected = list(range(1, steps + 1))
    if [r["step"] for r in fast] != expected or [r["step"] for r in upstream] != expected:
        raise ValueError("Comparison requires all sixteen or more steps")
    rows = []
    for actual, record in zip(fast, upstream, strict=True):
        if not record["common_weights_bitwise_equal"]:
            raise ValueError("Common-state evidence is missing")
        for mode in ("common_weight", "independent"):
            reference = record[mode]
            for metric in ("loss", "ce", "l1", "confidence", "denominator", "gradient_norm"):
                ref, val = reference[metric], actual[metric]
                if not math.isfinite(ref) or not math.isfinite(val):
                    raise ValueError("Non-finite comparison")
                absolute = abs(val - ref)
                rows.append(
                    {
                        "step": actual["step"],
                        "mode": mode,
                        "metric": metric,
                        "torchspec": ref,
                        "aspec": val,
                        "absolute_diff": absolute,
                        "relative_diff": absolute / max(abs(ref), 1e-12),
                    }
                )
    with (root / "torchspec-comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "optimizer_steps": steps,
        "cp": 1,
        "compared_real_documents": sum(r["documents"] for r in upstream),
        "same_state_full_gradient_max_relative_l2": max(
            r["common_weight"]["gradient_relative_l2_diff"] for r in upstream
        ),
    }
    for mode in ("common_weight", "independent"):
        summary[mode] = {}
        for metric in ("loss", "ce", "l1", "confidence", "denominator", "gradient_norm"):
            group = [r for r in rows if r["mode"] == mode and r["metric"] == metric]
            summary[mode][metric] = {
                "max_absolute_diff": max(r["absolute_diff"] for r in group),
                "max_relative_diff": max(r["relative_diff"] for r in group),
                "mean_relative_diff": sum(r["relative_diff"] for r in group) / steps,
                "first_step": group[0],
                "last_step": group[-1],
            }
    (root / "torchspec-comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 16:
        parser.error("This alignment protocol requires at least sixteen steps")
    if not args.compare_only:
        original = pipeline.install_observers

        def install(trainer, role, run_dir, capture, replay):
            original(trainer, role, run_dir, capture, replay)
            reference = TorchReference(trainer, args.fixture, run_dir)
            trainer.model.register_forward_pre_hook(reference.pre_forward, with_kwargs=True)
            original_step = trainer.optimizer.step

            def step(*positional, **kwargs):
                reference.before_update()
                return original_step(*positional, **kwargs)

            trainer.optimizer.step = step

        pipeline.install_observers = install
        pipeline.run(
            argparse.Namespace(
                role="fast",
                cp=1,
                fixture=args.fixture,
                run_dir=args.run_dir,
                steps=args.steps,
                learning_rate=None,
                capture_trajectory=None,
                replay_trajectory=None,
            )
        )
    compare(args.run_dir, args.steps)


if __name__ == "__main__":
    main()
