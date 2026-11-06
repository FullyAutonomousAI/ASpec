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

"""Read-only state/backend counterfactuals on real cached DSpark teacher features.

No optimizer updates, LR changes or production-code patches. Component-only
backwards are diagnostic decompositions, never alternative training objectives.
Each case replays the exact same documents, anchors and teacher tensors.
"""

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

from tools.validation.dspark_checkpoint_comparison import read_group
from tools.validation.dspark_nonpacking_alignment import (
    additive_terms,
    tensor_sha,
    vector_comparison,
)


def load_state(path):
    from safetensors.torch import load_file

    path = Path(path)
    if (path / "model.safetensors").is_file():
        return {
            "draft_model." + key: value
            for key, value in load_file(path / "model.safetensors").items()
        }
    return read_group(path / "model", "model_state", "model", "model_state.model.")


def load_target_head(fixture):
    # Preserve the paired trainer's hash-aware CUTLASS shim import order.
    importlib.import_module("aspec.models.draft.dspark")
    from omegaconf import OmegaConf

    from aspec.models.target.target_utils import TargetLMHead
    from tools.validation.dspark_teacher_contract import fixture_teacher_contract

    target = OmegaConf.load(Path(fixture) / "fast.yaml").model.target_model_path
    contract = fixture_teacher_contract(fixture)
    head = TargetLMHead.from_pretrained(target, load_norm=contract["last_hidden_states_prenorm"])
    head._aspec_teacher_contract = contract
    return head


def build_model(namespace, backend, config, state, component):
    import torch

    draft_api = importlib.import_module(namespace + ".models.draft.dspark")
    wrapper_api = importlib.import_module(namespace + ".models.dspark")
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        draft = draft_api.DSparkDraftModel(draft_api.DSparkConfig(**config))
        draft.freeze_embedding()
        options = {"attention_backend": backend} if namespace == "aspec" else {}
        model = wrapper_api.DSparkModel(draft, block_size=7, num_anchors=512, **options)
    finally:
        torch.set_default_dtype(original_dtype)
    model.load_state_dict(state, strict=True)
    model = model.to(device="cuda", dtype=torch.bfloat16).train()
    if component != "total":
        model.ce_loss_alpha = 0.1 if component == "ce" else 0.0
        model.l1_loss_alpha = 0.9 if component == "l1" else 0.0
        model.confidence_head_alpha = 1.0 if component == "confidence" else 0.0
    return model


def prepare_rows(cache, count, head):
    import torch
    from torchspec.models.ops.anchors import sample_anchor_positions

    from tools.validation.dspark_teacher_contract import (
        audit_native_teacher_input,
        teacher_last_hidden_states,
    )

    samples = torch.load(cache, map_location="cpu", weights_only=True)["samples"][:count]
    rows, identities = [], []
    for sample in samples:
        ids, mask = sample["input_ids"].cuda(), sample["loss_mask"].cuda()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.cuda.manual_seed(int(tensor_sha(ids)[:15], 16))
            anchors, keep = sample_anchor_positions(ids.shape[1], mask, 512, 7, ids.device)
        row = dict(
            input_ids=ids,
            loss_mask=mask,
            hidden_states_list=list(sample["hidden_states"].cuda().split(4096, dim=-1)),
            last_hidden_states=teacher_last_hidden_states(head, sample["last_hidden_states"].cuda()),
            injected_anchors=anchors,
            injected_keep_mask=keep,
        )
        rows.append(row)
        identities.append(
            {
                "tokens": ids.numel(),
                "input": tensor_sha(ids),
                "anchors": tensor_sha(anchors),
                "keep": tensor_sha(keep),
                "last": tensor_sha(row["last_hidden_states"]),
                "cached_last": tensor_sha(sample["last_hidden_states"]),
                "native_model_last": audit_native_teacher_input(head, sample, row["last_hidden_states"]),
            }
        )
    if len(rows) != count:
        raise ValueError("Not enough real cached documents")
    return rows, identities


def run_case(model, namespace, rows, head):
    import torch

    trainer_api = importlib.import_module(namespace + ".training.dflash_trainer").DFlashTrainer
    parameters = [(key, p) for key, p in model.named_parameters() if p.requires_grad]
    proxy = SimpleNamespace(optimizer=SimpleNamespace(model_params=[p for _, p in parameters]))
    totals, observations = [], []
    hooks = []
    current = {}

    def observe(name):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                current[name] = output.detach().cpu().clone()

        return hook

    for name in ("final_norm", "confidence_head"):
        hooks.append(getattr(model.draft_model, name).register_forward_hook(observe(name)))
    model.zero_grad(set_to_none=True)
    try:
        for row in rows:
            values = dict(row)
            if namespace == "torchspec":
                anchors, keep = values.pop("injected_anchors"), values.pop("injected_keep_mask")
                model._sample_anchor_positions = lambda seq, mask, dev: (anchors, keep)
            output = model(**values, lm_head_weight=head.lm_head.weight)
            trainer_api._backward(proxy, output[6][0], len(rows))
            totals.append(additive_terms(output).cpu())
            observations.append(dict(current))
            current.clear()
        trainer_api._normalize_accumulated_gradients(
            proxy, torch.stack(totals).sum(dim=0)[1].cuda(), len(rows)
        )
        gradients = {
            name: p.grad.detach().cpu().clone() if p.grad is not None else None
            for name, p in parameters
        }
        numerator, denominator, ce, l1, confidence = torch.stack(totals).sum(dim=0).tolist()
        reconstructed = (
            model.ce_loss_alpha * ce
            + model.l1_loss_alpha * l1
            + model.confidence_head_alpha * confidence
        )
        if denominator <= 0 or abs(reconstructed - numerator) > max(1e-3, abs(numerator) * 1e-5):
            raise ValueError("Diagnostic components do not reconstruct the native numerator")
        result = dict(
            loss=numerator / denominator,
            denominator=denominator,
            ce=ce / denominator,
            l1=l1 / denominator,
            confidence=confidence / denominator,
        )
        return result, gradients, observations
    finally:
        for hook in hooks:
            hook.remove()


def compare_maps(actual, reference):
    if actual.keys() != reference.keys():
        raise ValueError("Compared parameter identities differ")
    return {
        "global": vector_comparison(actual.values(), reference.values()),
        "per_parameter": {
            key: vector_comparison([actual[key]], [reference[key]]) for key in actual
        },
    }


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--fast", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--documents", type=int, default=4)
    parser.add_argument("--component", choices=("total", "ce", "l1", "confidence"), default="total")
    parser.add_argument(
        "--confidence-transplants",
        action="store_true",
        help="isolate head weights versus backbone using native TorchSpec only",
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    fixture = Path(args.fixture)
    manifest = json.loads((fixture / "manifest.json").read_text())
    head = load_target_head(fixture)
    rows, identities = prepare_rows(
        next((fixture / "cache/eval_cache").glob("*/eval_rank_0.pt")), args.documents, head
    )
    results, gradients, observed = {}, {}, {}
    states = {"initial": args.initial, "fast32": args.fast, "reference32": args.reference}
    backends = (
        ("torch", "torchspec", "flex_attention"),
        ("fastflex", "aspec", "flex_attention"),
        ("cudnn", "aspec", "cudnn_flex_attention"),
    )
    transplants = None
    if args.confidence_transplants:
        if args.component != "confidence":
            raise ValueError("Head transplants require the confidence-only diagnostic")
        fast_state, reference_state = load_state(args.fast), load_state(args.reference)
        head_keys = {key for key in fast_state if key.startswith("draft_model.confidence_head.")}
        if head_keys != {
            "draft_model.confidence_head.proj.weight",
            "draft_model.confidence_head.proj.bias",
        }:
            raise ValueError("Unexpected confidence head identity")
        transplants = {
            "fast32": fast_state,
            "reference32": reference_state,
            "fast32-reference-head": {
                key: reference_state[key] if key in head_keys else value
                for key, value in fast_state.items()
            },
            "reference32-fast-head": {
                key: fast_state[key] if key in head_keys else value
                for key, value in reference_state.items()
            },
        }
        states, backends = dict.fromkeys(transplants), backends[:1]
    for state_name, path in states.items():
        state = load_state(path) if transplants is None else transplants[state_name]
        for backend_name, namespace, backend in backends:
            name = state_name + "/" + backend_name
            print("TRAJECTORY_START " + name, flush=True)
            model = build_model(namespace, backend, manifest["draft_config"], state, args.component)
            result, gradient, observation = run_case(model, namespace, rows, head)
            results[name], gradients[name], observed[name] = result, gradient, observation
            print("TRAJECTORY_CASE " + json.dumps({"case": name, "loss": result}), flush=True)
            if backend_name == "torch":
                _, repeat_grad, repeat_observed = run_case(model, namespace, rows, head)
                gradients[name + "-repeat"] = repeat_grad
                observed[name + "-repeat"] = repeat_observed
            del model
            torch.cuda.empty_cache()
        del state
    pairs = []
    for state in states:
        if not args.confidence_transplants:
            pairs.extend(
                [(state + "/cudnn", state + "/torch"), (state + "/fastflex", state + "/torch")]
            )
        pairs.append((state + "/torch-repeat", state + "/torch"))
    for backend, _, _ in backends:
        pairs.append(("fast32/" + backend, "reference32/" + backend))
    if args.confidence_transplants:
        pairs.extend(
            [
                ("fast32-reference-head/torch", "reference32/torch"),
                ("reference32-fast-head/torch", "reference32/torch"),
                ("fast32-reference-head/torch", "fast32/torch"),
                ("reference32-fast-head/torch", "fast32/torch"),
            ]
        )
    comparisons = {}
    for actual, reference in pairs:
        key = actual + " vs " + reference
        record = compare_maps(gradients[actual], gradients[reference])
        record["activations"] = {
            name: vector_comparison(
                [row[name] for row in observed[actual]], [row[name] for row in observed[reference]]
            )
            for name in observed[reference][0]
        }
        comparisons[key] = record
        print(
            "TRAJECTORY_COMPARISON "
            + json.dumps(
                {"pair": key, "global": record["global"], "activations": record["activations"]}
            ),
            flush=True,
        )
    payload = {
        "scope": "bounded real-feature read-only counterfactual, no optimizer updates",
        "component": args.component,
        "documents": identities,
        "cases": results,
        "comparisons": comparisons,
    }
    (root / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
