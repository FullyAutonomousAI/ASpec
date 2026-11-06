# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
#
# Adapted from Transformers v4.57.1, commit 8cb5963cc22174954e7dca2c0a3320b7dc2f4edc:
# src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
# Qwen3MoeSparseMoeBlock and load_balancing_loss_func.
"""Qwen Top-K experts with explicit, per-forward router observations."""

import torch
import torch.nn.functional as F
from torch import nn


class QwenDraftMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        from aspec.models.draft.dflash import DFlashMLP

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            DFlashMLP(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states):
        shape = hidden_states.shape
        hidden = hidden_states.reshape(-1, shape[-1])
        router_logits = self.gate(hidden)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected = routing_weights.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(-1, keepdim=True)
        routing_weights = routing_weights.to(hidden.dtype)
        result = torch.zeros_like(hidden)
        # Empty expert inputs intentionally stay in the graph. Every replica
        # reaches every parameter, even when ranks select different experts.
        # No device-scalar Python branch or persistent activation cache.
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(selected == expert_id)
            contribution = expert(hidden[token_ids]) * routing_weights[token_ids, slots, None]
            result.index_add_(0, token_ids, contribution.to(result.dtype))
        return result.reshape(shape), router_logits


def router_loss_terms(router_logits, objective_weights, denominator, top_k, cp_group=None):
    """Qwen load balancing over valid Draft slots and all MoE layers.

    Statistics are local to one forward, as in the Qwen reference. The trainer
    pools the resulting auxiliary objective using its existing supervised-token
    denominator; this is not global-batch balancing over accumulated forwards.
    """
    logits = torch.cat(router_logits, dim=0)
    mask = (objective_weights.reshape(-1) > 0).float().repeat(len(router_logits))
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
    selected = probabilities.topk(top_k, dim=-1).indices
    expert_mask = F.one_hot(selected, logits.shape[-1]).sum(dim=1).float()
    count = mask.sum()
    safe_count = count.clamp_min(1)
    load_sum = (expert_mask * mask[:, None]).sum(0)
    importance_sum = (probabilities * mask[:, None]).sum(0)
    if cp_group is not None:
        import torch.distributed as dist
        from torch.distributed.nn.functional import all_reduce

        totals = torch.cat([count.reshape(1), load_sum])
        dist.all_reduce(totals, group=cp_group)
        count, load_sum = totals[0], totals[1:]
        importance_sum = all_reduce(importance_sum, group=cp_group)
        safe_count = count.clamp_min(1)
    load = load_sum / safe_count
    importance = importance_sum / safe_count
    auxiliary = logits.shape[-1] * (load * importance).sum()
    numerator = auxiliary * denominator
    components = {
        "router_aux_loss": (numerator.detach(), denominator.detach()),
        "router_max_load": ((load.max() / top_k * denominator).detach(), denominator.detach()),
        "router_used_experts": (((load > 0).sum() * denominator).detach(), denominator.detach()),
    }
    return numerator, components
