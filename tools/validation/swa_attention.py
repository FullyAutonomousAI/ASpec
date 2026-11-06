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

"""GPU correctness audit for anchor-context SWA, not a benchmark or user entry.

torchrun --standalone --nproc-per-node=4 -m tools.validation.swa_attention --output /path/result.json
FA4 forward/backward is checked against an independent global FP32 definition.
BF16 inputs and dO are preserved; CP owners use global, not rank-local positions.
"""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist

from aspec.models.ops.direct_cp import (
    DirectCPMetadata,
    build_context_positions,
    direct_cp_attention,
)


def error(actual, expected):
    difference = actual.float() - expected.float()
    return {"abs_max": difference.abs().max().item(),
            "relative_l2": (difference.norm() / expected.float().norm().clamp_min(1e-12)).item()}


def reference(inputs, anchors, keep, block_size, docs, window):
    query, ck, cv, dk, dv = inputs
    length = ck.shape[2]
    blocks = torch.arange(query.shape[2], device=query.device) // block_size
    qa = anchors[:, blocks]
    positions = torch.arange(length, device=query.device)[None, None]
    context = (positions < qa[:, :, None]) & (positions >= qa[:, :, None] - window)
    if docs is not None:
        qdocs = docs.gather(1, qa)
        context &= docs[:, None, :] == qdocs[:, :, None]
    draft = (blocks[:, None] == blocks[None, :])[None]
    token_keep = keep[:, blocks] & (qa + torch.arange(query.shape[2], device=query.device) % block_size < length)
    visible = torch.cat((context, draft.expand(query.shape[0], -1, -1)), -1)
    visible &= token_keep[:, :, None]
    visible[..., length:] &= token_keep[:, None, :]
    k, v = [torch.cat(pair, 2).repeat_interleave(query.shape[1] // ck.shape[1], 1)
            for pair in ((ck, dk), (cv, dv))]
    scores = query @ k.transpose(-1, -2) / query.shape[-1]**0.5
    scores = scores.masked_fill(~visible[:, None], -1e30)
    lse = scores.logsumexp(-1)
    output = (scores - lse[..., None]).exp() @ v
    output = torch.where(token_keep[:, None, :, None], output, torch.zeros_like(output))
    lse = torch.where(token_keep[:, None], lse, torch.full_like(lse, -1e30))
    return output, lse


def run_case(window, packed, layout):
    rank, size, device = dist.get_rank(), dist.get_world_size(), torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    length, block, count = 539, 7, 80
    docs = torch.cat([torch.full((n,), i, device=device) for i, n in enumerate([257, 263, 19])])[None] if packed else None
    positions = (torch.cat([build_context_positions(n, rank, size, layout=layout) + offset
                            for n, offset in zip([257, 263, 19], [0, 257, 520])]).to(device)[None]
                 if packed else build_context_positions(length, rank, size, layout=layout).to(device)[None])
    anchors = torch.linspace(0, 537, count, device=device).long()[None]
    keep = torch.ones_like(anchors, dtype=torch.bool)
    keep[:, 13] = False
    meta = DirectCPMetadata(
        anchors, keep, positions, block, rank, size, dist.group.WORLD if size > 1 else None,
        context_doc_ids=None if docs is None else docs.gather(1, positions),
        anchor_doc_ids=None if docs is None else docs.gather(1, anchors),
        sequence_length=length, context_window=window, layout=layout,
    )
    torch.manual_seed(79)
    shapes = [(1, 4, count*block, 128), (1, 2, length, 128), (1, 2, length, 128),
              (1, 2, count*block, 128), (1, 2, count*block, 128)]
    full = [torch.randn(s, device=device, dtype=torch.bfloat16) for s in shapes]
    qp = meta.local_block_ids.to(device).repeat_interleave(block)*block + torch.arange(block, device=device).repeat(meta.blocks_per_rank)
    selections = [qp, positions[0], positions[0], qp, qp]
    inputs = [v.index_select(2, p).detach().requires_grad_() for v, p in zip(full, selections)]
    refs = [v.float().detach().requires_grad_() for v in full]
    # Populate the full-attention traversal in the SAME forward owner first;
    # SWA must not reuse its unwindowed full tiles.
    with torch.no_grad():
        direct_cp_attention(*inputs, replace(meta, context_window=None, attention_observation={}))
    actual, lse = direct_cp_attention(*inputs, meta)
    expected, ref_lse = reference(refs, anchors, keep, block, docs, window)
    torch.testing.assert_close(actual.float(), expected.index_select(2, qp), atol=.025, rtol=.025)
    torch.testing.assert_close(lse, ref_lse.index_select(2, qp), atol=.025, rtol=.025)
    dout = torch.randn_like(expected).bfloat16()
    dlse = torch.randn_like(ref_lse)*.1
    ((actual*dout.index_select(2, qp)).sum() + (lse*dlse.index_select(2, qp)).sum()).backward()
    ((expected*dout.float()).sum() + (ref_lse*dlse).sum()).backward()
    row = {"window": window, "packing": packed, "layout": layout, "rank": rank,
           "output": error(actual, expected.index_select(2, qp)),
           "lse": error(lse[lse > -1e20], ref_lse.index_select(2, qp)[lse > -1e20]),
           "gradients": {}}
    for name, value, ref, selection in zip(("q", "context_k", "context_v", "draft_k", "draft_v"), inputs, refs, selections):
        target = ref.grad.index_select(2, selection)
        row["gradients"][name] = error(value.grad, target)
        torch.testing.assert_close(value.grad.float(), target, atol=.05, rtol=.05)
        assert row["gradients"][name]["relative_l2"] < .015, row
    if packed:
        assert meta.attention_observation["context_window"] == window
        assert meta.attention_observation["backward_observed"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    rows = []
    for window, packed, layout in (
        (1, False, "contiguous"), (127, False, "contiguous"), (2048, False, "contiguous"),
        (1, True, "contiguous"), (127, True, "contiguous"), (2048, True, "contiguous"),
        (127, True, "zigzag"),
    ):
        rows.append(run_case(window, packed, layout))
    gathered = [None]*dist.get_world_size()
    dist.all_gather_object(gathered, rows)
    if dist.get_rank() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"cp_size": dist.get_world_size(),
                                          "dtype": "bfloat16", "dO_dtype": "bfloat16",
                                          "reference": "independent_global_fp32",
                                          "cases_by_rank": gathered}, indent=2))
        print(f"PASS SWA BSHD/THD CP{dist.get_world_size()} output/LSE/dQ/dK/dV + mixed policy cache")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
