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

"""Isolate FA4 delta rounding on saved real THD/CP attention activations.

The delta-oracle intervention uses FP64 attention only in this diagnostic. It
does not change training, installed FA4 source, inputs, or upstream gradients.
Its memory/compute cost is deliberately not a proposed production solution.
"""

import argparse
import inspect
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from aspec.models.ops import direct_cp as cp
from tools.validation.dspark_torchspec_alignment import pipeline

FIELDS = ("q", "ck", "cv", "dk", "dv")


def vector_stats(reference, actual, group=None):
    reference, actual = reference.double(), actual.double()
    delta = actual - reference
    totals = torch.stack(
        (
            reference.square().sum(),
            actual.square().sum(),
            delta.square().sum(),
            (reference * actual).sum(),
        )
    )
    if group is not None:
        dist.all_reduce(totals, group=group)
    ref2, actual2, delta2, product = (float(value) for value in totals)
    result = {
        "reference_l2": math.sqrt(ref2),
        "actual_l2": math.sqrt(actual2),
        "absolute_l2": math.sqrt(delta2),
        "relative_l2": math.sqrt(delta2 / max(ref2, 1e-30)),
        "cosine": product / max(math.sqrt(ref2 * actual2), 1e-30),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Non-finite numerical diagnostic")
    return result


def analytic_partial(query, key, value, metadata, scale, dout, dlse, used_output=None):
    """Exact fixed-mask VJP with an independently selectable softmax delta."""
    q, k, v = (tensor.double() for tensor in (query, key, value))
    k, v = cp._expand_kv_heads(q, k, v)
    visible = cp._visibility(metadata, q.device)[:, None]
    scores = (q @ k.transpose(-1, -2)) * scale
    scores = torch.where(visible, scores, cp.MASK_VALUE)
    probabilities = torch.softmax(scores, -1) * visible
    output = probabilities @ v
    g = dout.double()
    delta = (g * (output if used_output is None else used_output.double())).sum(-1)
    if dlse is not None:
        delta = delta - dlse.double()
    ds = probabilities * ((g @ v.transpose(-1, -2)) - delta.unsqueeze(-1))
    dq = (ds @ k) * scale
    dk = (ds.transpose(-1, -2) @ q) * scale
    dv = probabilities.transpose(-1, -2) @ g
    repeats = query.shape[1] // key.shape[1]
    if repeats > 1:
        dk = dk.unflatten(1, (key.shape[1], repeats)).sum(2)
        dv = dv.unflatten(1, (key.shape[1], repeats)).sum(2)
    return output, (dq, dk, dv)


def delta_lse_correction(output, exact_output, dout):
    """Diagnostic dLSE offset making FA4's delta use the exact partial output."""
    return ((output.double() - exact_output.double()) * dout.double()).sum(-1).float()


def check_layer(path, rank, size):
    from flash_attn.cute import interface as fa4

    data = torch.load(path, weights_only=True)
    base = {key: value.cuda() for key, value in data["metadata"].items()}
    full_metadata = cp.DirectCPMetadata(
        **base,
        block_size=data["block_size"],
        sequence_length=data["sequence_length"],
        cp_rank=0,
        cp_size=1,
        group=None,
        backend="math",
    )
    full = {key: data[key].cuda().double().requires_grad_() for key in FIELDS}
    full_output, _ = cp.direct_cp_attention(
        *(full[key] for key in FIELDS), full_metadata, scale=data["scale"]
    )
    full_output.backward(data["output_gradient"].cuda().double())
    documents = base["context_doc_ids"][0]
    parts = []
    for doc in documents.unique(sorted=True).tolist():
        if doc >= 0:
            positions = (documents == doc).nonzero().flatten()
            parts.append(
                positions[positions.numel() * rank // size : positions.numel() * (rank + 1) // size]
            )
    context_indices = torch.cat(parts)
    blocks = base["anchor_positions"].shape[1]
    if blocks % size:
        raise ValueError("Captured anchor slots must be divisible by CP degree")
    query_indices = torch.arange(
        blocks * rank // size * data["block_size"],
        blocks * (rank + 1) // size * data["block_size"],
        device="cuda",
    )
    indices = {key: context_indices if key in ("ck", "cv") else query_indices for key in FIELDS}
    metadata = cp.DirectCPMetadata(
        anchor_positions=base["anchor_positions"],
        block_keep_mask=base["block_keep_mask"],
        context_position_ids=base["context_position_ids"].index_select(1, context_indices),
        context_doc_ids=base["context_doc_ids"].index_select(1, context_indices),
        anchor_doc_ids=base["anchor_doc_ids"],
        block_size=data["block_size"],
        sequence_length=data["sequence_length"],
        cp_rank=rank,
        cp_size=size,
        group=dist.group.WORLD if size > 1 else None,
        backend="fa4",
    )
    group = dist.group.WORLD
    native_partial = cp._fa4_partial
    native_backward = fa4._flash_attn_bwd
    signature = inspect.signature(native_backward)
    results = {}
    baseline_gradients = None
    baseline_output = None
    for mode in ("native", "delta_oracle", "delta_and_merge_oracle"):
        captured = {}

        def observe_partial(query, key, value, local_metadata, scale):
            output, lse = native_partial(query, key, value, local_metadata, scale)
            captured.update(
                q=query.detach(),
                k=key.detach(),
                v=value.detach(),
                output=output.detach(),
                metadata=local_metadata,
                scale=scale,
            )
            if mode == "delta_and_merge_oracle":
                # Diagnostic-only FP32 partial storage. Its backward still goes
                # through native FA4, including the BF16 dO cast at that boundary.
                with torch.no_grad():
                    exact_output, _ = cp._math_partial(
                        query.double(), key.double(), value.double(), local_metadata, scale
                    )
                    valid = cp._visibility(local_metadata, query.device).any(-1)
                    exact_output = torch.where(
                        valid[:, None, :, None], exact_output, torch.zeros_like(exact_output)
                    )
                output = output.float() + (exact_output.float() - output.detach().float())
            return output, lse

        def observe_backward(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            dout = arguments["dout"].transpose(1, 2).detach()
            dlse = arguments["dlse"]
            with torch.no_grad():
                exact_output, exact_gradients = analytic_partial(
                    captured["q"],
                    captured["k"],
                    captured["v"],
                    captured["metadata"],
                    captured["scale"],
                    dout,
                    dlse,
                )
                _, rounded_gradients = analytic_partial(
                    captured["q"],
                    captured["k"],
                    captured["v"],
                    captured["metadata"],
                    captured["scale"],
                    dout,
                    dlse,
                    captured["output"],
                )
                captured.update(
                    exact_gradients=exact_gradients, rounded_gradients=rounded_gradients
                )
                if mode != "native":
                    correction = delta_lse_correction(captured["output"], exact_output, dout)
                    arguments["dlse"] = correction if dlse is None else dlse + correction
            gradients = native_backward(*bound.args, **bound.kwargs)
            captured["kernel_gradients"] = tuple(g.detach().transpose(1, 2) for g in gradients)
            return gradients

        # Native bwd resolves this global attribute for its unchanged JIT cache.
        observe_backward.compile_cache = native_backward.compile_cache
        cp._fa4_partial = observe_partial
        fa4._flash_attn_bwd = observe_backward
        values = {
            key: data[key].cuda().index_select(2, indices[key]).requires_grad_() for key in FIELDS
        }
        try:
            output, _ = cp.direct_cp_attention(
                *(values[key] for key in FIELDS), metadata, scale=data["scale"]
            )
            output.backward(data["output_gradient"].cuda().index_select(2, query_indices))
        finally:
            cp._fa4_partial = native_partial
            fa4._flash_attn_bwd = native_backward
        row = {"output": vector_stats(full_output.index_select(2, query_indices), output, group)}
        for key in FIELDS:
            reference = full[key].grad.index_select(2, indices[key])
            row[key] = vector_stats(reference, values[key].grad, group)
            row[f"{key}_bf16_cast_floor"] = vector_stats(reference, reference.bfloat16(), group)
        if baseline_gradients is None:
            baseline_gradients = {key: values[key].grad.detach().clone() for key in FIELDS}
            baseline_output = output.detach().clone()
        else:
            row["vs_native"] = {
                key: vector_stats(baseline_gradients[key], values[key].grad, group)
                for key in FIELDS
            }
            row["vs_native"]["output"] = vector_stats(baseline_output, output, group)
            unequal = torch.tensor(
                (baseline_output != output).sum(), device=output.device, dtype=torch.int64
            )
            dist.all_reduce(unequal)
            row["forward_unequal_coordinates_vs_native"] = int(unequal)
            if mode == "delta_oracle" and int(unequal) != 0:
                raise ValueError("Delta-only intervention changed the forward output")
        # Each rank owns different KV; Q here is the kernel's unreduced contribution.
        selected_prediction = captured[
            "rounded_gradients" if mode == "native" else "exact_gradients"
        ]
        row["local_kernel_vs_selected_prediction"] = {
            key: vector_stats(prediction, actual, group)
            for key, prediction, actual in zip(
                ("q", "k", "v"), selected_prediction, captured["kernel_gradients"]
            )
        }
        # Reproduce the CP SUM analytically in FP64, isolating delta from BF16 reduction.
        for label, prediction in (
            ("rounded_delta_prediction", captured["rounded_gradients"]),
            ("exact_delta_prediction", captured["exact_gradients"]),
        ):
            predicted_q = prediction[0].clone()
            dist.all_reduce(predicted_q, group=group)
            predicted_q = predicted_q.index_select(2, query_indices)
            predicted_k = prediction[1]
            row[label] = {
                "q": vector_stats(
                    full["q"].grad.index_select(2, query_indices), predicted_q, group
                ),
                "dk": vector_stats(
                    full["dk"].grad.index_select(2, query_indices),
                    predicted_k[:, :, context_indices.numel() :],
                    group,
                ),
            }
            if mode == "native" and label == "rounded_delta_prediction":
                native_error = values["q"].grad.double() - full["q"].grad.index_select(
                    2, query_indices
                )
                predicted_error = predicted_q - full["q"].grad.index_select(2, query_indices)
                row["delta_prediction_error_direction"] = vector_stats(
                    native_error, predicted_error, group
                )
        results[mode] = row
        if rank == 0:
            print(
                json.dumps(
                    {
                        "layer": data.get("layer"),
                        "cp": size,
                        "mode": mode,
                        "q": row["q"],
                        "dk": row["dk"],
                        "exact_delta_prediction": row["exact_delta_prediction"],
                    }
                ),
                flush=True,
            )
        captured.clear()
        del values, output, selected_prediction, prediction
    return {
        "layer": data.get("layer"),
        "step": data.get("step"),
        "cp": size,
        "state_sha256": data.get("state_sha256"),
        "activation_sha256": pipeline.file_sha(path),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl")
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        records = [
            check_layer(Path(args.activations) / f"layer-{layer}.pt", rank, size)
            for layer in args.layers
        ]
        if rank == 0:
            result = {
                "source_sha256": pipeline.file_sha(__file__),
                "cp": size,
                "production_source_changed": False,
                "diagnostic_intervention": "FP64 delta and optional FP32 partial merge oracle",
                "not_a_production_fix_or_performance_claim": True,
                "rows": records,
            }
            destination.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
