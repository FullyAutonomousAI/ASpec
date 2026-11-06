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

# Adapted from vLLM Speculators 7ec7aba9450b533334e8f7992baf4e93be3c1a57,
# src/speculators/models/mtp/core.py (Apache-2.0).
"""Native teacher-forced multi-step CE. No EAGLE TTT suffix cache."""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def mtp_step_weights(steps, beta):
    # Scale by the largest power to avoid overflow for finite beta > 1.
    raw = [beta ** (k if beta <= 1 else k - steps + 1) for k in range(steps)]
    return [value / sum(raw) for value in raw]


class MTPModel(nn.Module):
    def __init__(self, draft_model, length=3, gradient_checkpointing=False):
        super().__init__()
        self.draft_model = draft_model
        self.length = length
        self.gradient_checkpointing = gradient_checkpointing
        self.vocab_pruning = False

    def forward(self, input_ids, hidden_states, loss_mask, attention_mask, thd_layout=None):
        if thd_layout is not None:
            return self._forward_thd(hidden_states, thd_layout)
        lengths = attention_mask.sum(-1).long().tolist()
        zero = sum(p.reshape(-1)[0] for p in self.parameters() if p.requires_grad) * 0.0
        sums = [zero for _ in range(self.length)]
        counts = [zero.detach() for _ in range(self.length)]
        correct = [zero.detach() for _ in range(self.length)]
        for row, length in enumerate(lengths):
            steps = min(self.length, max(0, length - 2))
            valid_length = length - steps - 1
            if not steps:
                continue
            state = hidden_states[row:row + 1, :valid_length].detach()
            positions = torch.arange(valid_length, device=state.device)[None]
            for depth in range(steps):
                ids = input_ids[row:row + 1, depth + 1:depth + 1 + valid_length]
                if self.gradient_checkpointing and self.training:
                    state = checkpoint(self.draft_model.predict, state, ids, positions, use_reentrant=False)
                else:
                    state = self.draft_model.predict(state, ids, positions)
                # Keep every valid attention position, but supervise only real
                # response labels at t+depth+2. Never shift masks like EAGLE TTT.
                labels = input_ids[row, depth + 2:depth + 2 + valid_length]
                keep = loss_mask[row, depth + 2:depth + 2 + valid_length].bool()
                selected = keep.nonzero().flatten()
                for indices in selected.split(128):
                    targets = labels.index_select(0, indices)
                    features = state[0].index_select(0, indices)
                    def loss_chunk(features, targets):
                        logits = self.draft_model.lm_head(features)
                        return (F.cross_entropy(logits.float(), targets, reduction="sum"),
                                (logits.argmax(-1) == targets).float().sum())
                    if self.training and torch.is_grad_enabled():
                        loss, hits = checkpoint(loss_chunk, features, targets, use_reentrant=False)
                    else:
                        loss, hits = loss_chunk(features, targets)
                    sums[depth] = sums[depth] + loss
                    correct[depth] = correct[depth] + hits.detach()
                counts[depth] = counts[depth] + selected.numel()
        losses = [value / count.clamp_min(1) for value, count in zip(sums, counts, strict=True)]
        return (losses, [value.detach() for value in losses],
                [value / count.clamp_min(1) for value, count in zip(correct, counts, strict=True)],
                counts, [zero.detach() for _ in range(self.length)])

    def _forward_thd(self, hidden_states, layout):
        """Owner-local recurrent states; only attention Q/K/V cross CP ranks."""
        steps = torch.tensor([min(self.length, max(0, n - 2)) for n in layout.lengths],
                             device=hidden_states.device)
        prefixes = torch.tensor(layout.lengths, device=hidden_states.device) - steps - 1
        # All depths use the same valid prefix, as in the official MTP objective.
        # Integer masks and sparse metadata are built once per batch, not per depth.
        for docs, rope in ((layout.docs, layout.rope), (layout.global_docs, layout.global_rope)):
            valid = (docs >= 0) & (rope < prefixes[docs.clamp_min(0)]) & (steps[docs.clamp_min(0)] > 0)
            docs.masked_fill_(~valid, -1)
        state = hidden_states.detach()
        results = ([], [], [], [], [])
        for depth in range(self.length):
            requested, same_doc = layout.shifted(depth + 1)
            ids = layout.lookup(layout.full_ids, requested).unsqueeze(0)
            positions = layout.rope.clamp_min(0).unsqueeze(0)
            if self.gradient_checkpointing and self.training:
                state = checkpoint(self.draft_model.predict, state, ids, positions, layout,
                                   use_reentrant=False)
            else:
                state = self.draft_model.predict(state, ids, positions, layout)
            label_positions = layout.positions + depth + 2
            labels = layout.lookup(layout.full_ids, label_positions)
            keep = ((layout.docs >= 0) & (depth < steps[layout.docs.clamp_min(0)]) & same_doc
                    & layout.lookup(layout.full_loss, label_positions).bool())
            selected = keep.nonzero().flatten()
            numerator = state.sum() * 0.0
            correct = numerator.detach()
            for indices in selected.split(128):
                targets, features = labels[indices], state[0, indices]

                def loss_chunk(features, targets):
                    logits = self.draft_model.lm_head(features)
                    return (F.cross_entropy(logits.float(), targets, reduction="sum"),
                            (logits.argmax(-1) == targets).float().sum())

                if self.training and torch.is_grad_enabled():
                    loss, hits = checkpoint(loss_chunk, features, targets, use_reentrant=False)
                else:
                    loss, hits = loss_chunk(features, targets)
                numerator, correct = numerator + loss, correct + hits.detach()
            stats = torch.stack((numerator.detach().float(), correct.float(),
                                 numerator.new_tensor(selected.numel()).float()))
            if layout.cp_size > 1:
                dist.all_reduce(stats, group=layout.group)
            denominator = stats[2].clamp_min(1)
            results[0].append(numerator / denominator * layout.cp_size)
            results[1].append(stats[0] / denominator)
            results[2].append(stats[1] / denominator)
            results[3].append(stats[2])
            results[4].append(torch.zeros_like(denominator))
        return results
