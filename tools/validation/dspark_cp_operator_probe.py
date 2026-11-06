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

"""Check CP's operator Jacobian on an actual full-draft attention activation.

The FP64 math check is an isolated oracle, not a training-precision change.
Native FA4 is compared to that oracle with the identical real upstream gradient.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from tools.validation.dspark_cp1_probe import build_model, native_window, wrap_native_fsdp
from tools.validation.dspark_torchspec_alignment import pipeline
from tools.validation.dspark_update_alignment import cpu_tensor, save_new


def capture(args):
    import torch
    import torch.distributed as dist

    from aspec.models.ops import direct_cp

    torch.cuda.set_device(0)
    dist.init_process_group("nccl")
    if dist.get_world_size() != 1:
        raise ValueError("Capture the whole actual CP1 attention activation")
    model = wrap_native_fsdp(build_model(args.fixture, "aspec", "fa4"), args.fixture, "aspec")
    original = direct_cp.direct_cp_attention_from_kv
    saved = {}

    def observed(query_own, key, value, metadata, *positional, **kwargs):
        context_key, draft_key_own = (
            key[:, :, : metadata.context_length],
            key[:, :, metadata.context_length :],
        )
        context_value, draft_value_own = (
            value[:, :, : metadata.context_length],
            value[:, :, metadata.context_length :],
        )
        first = not saved
        if first:
            saved.update(
                {
                    name: cpu_tensor(value)
                    for name, value in {
                        "q": query_own,
                        "ck": context_key,
                        "cv": context_value,
                        "dk": draft_key_own,
                        "dv": draft_value_own,
                    }.items()
                }
            )
            saved["metadata"] = {
                name: cpu_tensor(getattr(metadata, name))
                for name in (
                    "anchor_positions",
                    "block_keep_mask",
                    "context_position_ids",
                    "context_doc_ids",
                    "anchor_doc_ids",
                )
            }
            saved.update(
                block_size=metadata.block_size,
                sequence_length=metadata.sequence_length,
                scale=kwargs.get("scale", query_own.shape[-1] ** -0.5),
            )
        output = original(
            query_own,
            key,
            value,
            metadata,
            *positional,
            **kwargs,
        )
        if first:
            output[0].register_hook(
                lambda gradient: saved.update(output_gradient=cpu_tensor(gradient))
            )
        return output

    direct_cp.direct_cp_attention_from_kv = observed
    root = Path(args.captured) / "inputs"
    packed = [
        torch.load(root / f"step-0001-micro-{micro:02d}.pt", weights_only=True)
        for micro in range(2)
    ]
    head = torch.load(root / "lm-head.pt", weights_only=True).cuda()
    native_window(
        model,
        packed,
        head,
        "aspec",
        False,
        SimpleNamespace(model_params=[p for p in model.parameters() if p.requires_grad]),
    )
    if "output_gradient" not in saved:
        raise ValueError("Actual first-layer backward was not observed")
    saved["harness_sha256"] = pipeline.file_sha(__file__)
    save_new(saved, args.activation)
    print(
        {key: list(value.shape) for key, value in saved.items() if isinstance(value, torch.Tensor)},
        flush=True,
    )
    dist.destroy_process_group()


def difference(reference, actual):
    reference, actual = reference.double(), actual.double()
    return (actual - reference).square().sum(), reference.square().sum()


def check(args):
    import torch
    import torch.distributed as dist

    from aspec.models.ops.direct_cp import DirectCPMetadata, direct_cp_attention

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    size = dist.get_world_size()
    data = torch.load(args.activation, weights_only=True)
    fields = ("q", "ck", "cv", "dk", "dv")
    global_values = {key: data[key].cuda().double().requires_grad_() for key in fields}
    base = {key: value.cuda() for key, value in data["metadata"].items()}
    full_metadata = DirectCPMetadata(
        **base,
        block_size=data["block_size"],
        sequence_length=data["sequence_length"],
        cp_rank=0,
        cp_size=1,
        group=None,
        backend="math",
    )
    full_output, _ = direct_cp_attention(
        *(global_values[key] for key in fields), full_metadata, scale=data["scale"]
    )
    full_output.backward(data["output_gradient"].cuda().double())
    docs = data["metadata"]["context_doc_ids"][0]
    indices = []
    for doc in docs.unique(sorted=True).tolist():
        if doc >= 0:
            positions = (docs == doc).nonzero().flatten()
            indices.append(
                positions[positions.numel() * rank // size : positions.numel() * (rank + 1) // size]
            )
    context_indices = torch.cat(indices).cuda()
    blocks = data["metadata"]["anchor_positions"].shape[1]
    if blocks % size:
        raise ValueError("Captured anchor slots must be divisible by this CP degree")
    query_indices = torch.arange(
        blocks * rank // size * data["block_size"],
        blocks * (rank + 1) // size * data["block_size"],
        device="cuda",
    )
    owned = {key: (context_indices if key in ("ck", "cv") else query_indices) for key in fields}
    local_metadata = DirectCPMetadata(
        anchor_positions=base["anchor_positions"],
        block_keep_mask=base["block_keep_mask"],
        context_position_ids=base["context_position_ids"].index_select(1, context_indices),
        context_doc_ids=base["context_doc_ids"].index_select(1, context_indices),
        anchor_doc_ids=base["anchor_doc_ids"],
        block_size=data["block_size"],
        sequence_length=data["sequence_length"],
        cp_rank=rank,
        cp_size=size,
        group=dist.group.WORLD,
    )
    results = {}
    for backend, dtype in (("math", torch.float64), ("fa4", torch.bfloat16)):
        local_metadata.backend = backend
        values = {
            key: data[key].cuda().index_select(2, owned[key]).to(dtype).requires_grad_()
            for key in fields
        }
        output, _ = direct_cp_attention(
            *(values[key] for key in fields), local_metadata, scale=data["scale"]
        )
        output.backward(data["output_gradient"].cuda().index_select(2, query_indices).to(dtype))
        pairs = {
            "output": difference(
                full_output.detach().index_select(2, query_indices), output.detach()
            )
        }
        pairs.update(
            {
                key: difference(
                    global_values[key].grad.index_select(2, owned[key]), values[key].grad
                )
                for key in fields
            }
        )
        results[backend] = {}
        for key, (delta, reference) in pairs.items():
            numbers = torch.stack((delta, reference))
            dist.all_reduce(numbers)
            results[backend][key] = {
                "relative_l2": float((numbers[0] / numbers[1].clamp_min(1e-30)).sqrt()),
                "difference_squared": float(numbers[0]),
                "reference_squared": float(numbers[1]),
            }
        del values, output
    result = {
        "cp": size,
        "actual_full_default_draft_first_layer": data.get("layer", 0) == 0,
        "layer": data.get("layer", 0),
        "step": data.get("step", 1),
        "state_sha256": data.get("state_sha256"),
        "activation_sha256": pipeline.file_sha(args.activation),
        "harness_sha256": pipeline.file_sha(__file__),
        "training_precision_changed": False,
        "isolated_oracle_precision": "FP64",
        "results": results,
    }
    if rank == 0:
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    if max(value["relative_l2"] for value in results["math"].values()) > 1e-10:
        raise ValueError("CP FP64 operator/Jacobian differs from unpartitioned attention")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("capture", "check"), required=True)
    parser.add_argument("--fixture")
    parser.add_argument("--captured")
    parser.add_argument("--activation", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    capture(args) if args.role == "capture" else check(args)


if __name__ == "__main__":
    main()
