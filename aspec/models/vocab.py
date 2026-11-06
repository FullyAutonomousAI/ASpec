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

"""Checkpoint-bound EAGLE vocabulary identity; d2t stores offsets, not token IDs."""

import hashlib

import torch


def validate_vocab_mapping(d2t, t2d, draft_vocab_size, target_vocab_size):
    if not isinstance(d2t, torch.Tensor) or not isinstance(t2d, torch.Tensor):
        raise ValueError("Vocabulary mapping requires d2t and t2d tensors")
    if d2t.dtype != torch.int64 or t2d.dtype != torch.bool:
        raise ValueError("Vocabulary mapping requires int64 d2t offsets and boolean t2d membership")
    if tuple(d2t.shape) != (draft_vocab_size,) or tuple(t2d.shape) != (target_vocab_size,):
        raise ValueError("Vocabulary mapping shape does not match the configured head")
    selected = t2d.nonzero().flatten()
    if selected.numel() != draft_vocab_size or not torch.equal(
        selected, d2t.to(selected.device) + torch.arange(draft_vocab_size, device=selected.device),
    ):
        raise ValueError("Vocabulary mapping does not bind head rows to the sorted Target tokens")


def vocab_mapping_manifest(d2t, t2d):
    validate_vocab_mapping(d2t, t2d, d2t.numel(), t2d.numel())
    return {
        "target_vocab_size": t2d.numel(),
        "draft_vocab_size": d2t.numel(),
        "d2t_semantics": "target_token_id_minus_draft_token_id",
        "mapping_sha256": hashlib.sha256(
            d2t.detach().cpu().numpy().tobytes() + t2d.detach().cpu().numpy().tobytes(),
        ).hexdigest(),
    }
