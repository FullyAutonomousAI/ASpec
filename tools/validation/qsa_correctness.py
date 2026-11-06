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

"""Compare the QSA selected-token FA4 core with an explicit FP32 reference."""

import json

import torch

from aspec.models.draft.qsa import (
    QSAForwardMetadata,
    _selected_raw_tokens,
    build_qsa_compression_layout,
    qsa_sparse_attention,
)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        (
            (actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-12)
        ).detach()
    )


def main() -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(20261015)
    batch, anchors, block_size = 1, 3, 2
    context_length, query_heads, kv_heads, head_dim = 12, 4, 2, 128
    query_length = anchors * block_size
    metadata = QSAForwardMetadata(
        anchor_positions=torch.tensor([[3, 7, 11]], device=device),
        block_keep_mask=torch.ones(batch, anchors, dtype=torch.bool, device=device),
        block_size=block_size,
    )
    layout = build_qsa_compression_layout(
        metadata, torch.arange(context_length, device=device).view(1, -1), ratio=4
    )
    representative = torch.tensor(
        [[[-1, -1], [0, -1], [0, 1]]], device=device
    )
    total_length = context_length + query_length
    query = torch.randn(
        batch,
        query_length,
        query_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = torch.randn(
        batch,
        total_length,
        kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    value = torch.randn_like(key, requires_grad=True)
    scale = head_dim**-0.5
    actual = qsa_sparse_attention(
        query=query,
        key=key,
        value=value,
        metadata=metadata,
        layout=layout,
        selected=representative,
        scale=scale,
        workspace_bytes=64 * 1024 * 1024,
    )

    rows = torch.arange(0, query_length, block_size, device=device)
    tokens, valid, _ = _selected_raw_tokens(
        metadata=metadata,
        layout=layout,
        selected=representative,
        start=0,
        end=anchors,
        draft_length=query_length,
        query_indices=rows,
    )
    expected_rows = []
    repetitions = query_heads // kv_heads
    for query_id in range(query_length):
        owner = query_id // block_size
        indices = tokens[0, owner][valid[0, owner]]
        head_rows = []
        for head in range(query_heads):
            scores = (
                query[0, query_id, head].float()
                @ key[0, indices, head // repetitions].float().T
            ) * scale
            probabilities = scores.softmax(dim=-1)
            head_rows.append(
                probabilities @ value[0, indices, head // repetitions].float()
            )
        expected_rows.append(torch.stack(head_rows))
    expected = torch.stack(expected_rows).unsqueeze(0)
    upstream = torch.randn_like(expected)
    (actual.float() * upstream).sum().backward()
    actual_gradients = [query.grad.clone(), key.grad.clone(), value.grad.clone()]
    query_reference = query.detach().clone().requires_grad_(True)
    key_reference = key.detach().clone().requires_grad_(True)
    value_reference = value.detach().clone().requires_grad_(True)
    reference_rows = []
    for query_id in range(query_length):
        owner = query_id // block_size
        indices = tokens[0, owner][valid[0, owner]]
        head_rows = []
        for head in range(query_heads):
            scores = (
                query_reference[0, query_id, head].float()
                @ key_reference[0, indices, head // repetitions].float().T
            ) * scale
            head_rows.append(
                scores.softmax(dim=-1)
                @ value_reference[0, indices, head // repetitions].float()
            )
        reference_rows.append(torch.stack(head_rows))
    reference = torch.stack(reference_rows).unsqueeze(0)
    (reference * upstream).sum().backward()
    reference_gradients = [
        query_reference.grad,
        key_reference.grad,
        value_reference.grad,
    ]
    names = ("output", "query_gradient", "key_gradient", "value_gradient")
    pairs = [(actual, reference), *zip(actual_gradients, reference_gradients)]
    metrics = {
        name: {
            "max_abs": float(
                (left.float() - right.float()).abs().max().detach()
            ),
            "relative_l2": _relative_l2(left, right),
        }
        for name, (left, right) in zip(names, pairs, strict=True)
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))
    assert metrics["output"]["relative_l2"] < 0.01
    assert metrics["query_gradient"]["relative_l2"] < 0.02
    assert metrics["key_gradient"]["relative_l2"] < 0.02
    assert metrics["value_gradient"]["relative_l2"] < 0.02


if __name__ == "__main__":
    main()
