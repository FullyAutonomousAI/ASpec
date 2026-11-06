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

# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# Adapted from inclusionAI/Ling-3.0-flash, fcc60a52b34c69c6271d0ac8a8cb0815020b6a43,
# modeling_bailing_moe_v3.py (BailingMoeV3Gate/SparseMoeBlock/MLP).
# Clamp: official SGLang e00930c5489053f26d86b179cee0d087f846acbb, BailingMLP.
# Bias update: Ling-V2 733190f90783d07faabd7060fb64550b65d7114d Megatron patch
# over Megatron-LM core_v0.13.0 c550cf6c41c31cd3ec72e05c25ea0c979f2b6631.
# Ling-V2 portions: MIT, licenses/Ling-V2-MIT.txt.
# Megatron-LM portions: BSD-3-Clause, licenses/NVIDIA-Megatron-LM-BSD-3-Clause.txt.
"""Ling FFN routing and training statistics; no alternate trainer or attention."""

import torch
import torch.nn.functional as F
from torch import nn

from aspec.models.draft.dflash import DFlashMLP


class LingExpert(DFlashMLP):
    def __init__(self, config, width, limit):
        super().__init__(config, intermediate_size=width)
        self.limit = limit

    def forward(self, x):
        gate, up = F.silu(self.gate_proj(x)), self.up_proj(x)
        if self.limit:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        return self.down_proj(gate * up)


class LingGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.register_buffer("expert_bias", torch.zeros(config.num_experts, dtype=torch.float32))
        self.top_k = config.num_experts_per_tok
        self.groups = config.n_group
        self.group_topk = config.topk_group
        self.scale = config.routed_scaling_factor
        self.bias_enabled = config.moe_router_enable_expert_bias

    def _apply(self, fn, recurse=True):
        # Bias is a checkpointed FP32 control state, never a BF16 Adam parameter.
        # Preserve values when the rest of the Draft is cast to BF16.
        bias = self.expert_bias
        super()._apply(fn, recurse=recurse)
        self.expert_bias = (self.expert_bias.float() if bias.is_meta
                            else bias.to(device=self.expert_bias.device, dtype=torch.float32))
        return self

    def forward(self, hidden):
        logits = F.linear(hidden.reshape(-1, hidden.shape[-1]).float(), self.weight.float())
        scores = logits.sigmoid()
        selection_scores = scores + self.expert_bias if self.bias_enabled else scores
        group_scores = selection_scores.view(-1, self.groups, scores.shape[-1] // self.groups)
        group_scores = group_scores.topk(2, dim=-1).values.sum(-1)
        selected_groups = group_scores.topk(self.group_topk, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, selected_groups, True)
        score_mask = group_mask.unsqueeze(-1).expand(-1, -1, scores.shape[-1] // self.groups).reshape_as(scores)
        selected = selection_scores.masked_fill(~score_mask, float("-inf")).topk(self.top_k, dim=-1).indices
        weights = scores.gather(1, selected)
        # Official Ling Top-1 deliberately does NOT renormalize its score to one.
        if self.top_k > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return selected, weights * self.scale, logits


class LingDraftMoE(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = LingGate(config)
        self.experts = nn.ModuleList([
            LingExpert(config, config.moe_intermediate_size, config.expert_swiglu_limit_list[layer_idx])
            for _ in range(config.num_experts)
        ])
        self.shared_experts = (
            LingExpert(config, config.num_shared_experts * config.moe_shared_expert_intermediate_size,
                       config.share_expert_swiglu_limit_list[layer_idx])
            if config.num_shared_experts else None
        )

    def forward(self, hidden_states):
        shape = hidden_states.shape
        hidden = hidden_states.reshape(-1, shape[-1])
        selected, weights, logits = self.gate(hidden)
        # Accumulate in FP32, matching official Ling's expert-output * FP32
        # routing-weight reduction, then cast BEFORE adding the shared expert.
        result = torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(selected == expert_id)
            contribution = expert(hidden[token_ids]).float() * weights[token_ids, slots, None]
            result.index_add_(0, token_ids, contribution)
        result = result.to(hidden.dtype).reshape(shape)
        if self.shared_experts is not None:
            result = result + self.shared_experts(hidden_states)
        return result, (logits, selected)


def ling_router_terms(observations, objective_weights, denominator, z_coefficient, cp_group=None):
    """Layer-wise bias counts and z-loss; no Qwen softmax auxiliary loss.

    Valid supervised Draft slots exclude padding/invalid anchors. Counts remain
    local and detached until the trainer pools ALL micro-batches and DP/CP ranks.
    z-loss sums layer objectives (Megatron convention), normalized per forward
    and then pooled with the existing supervised-token denominator.
    """
    mask = (objective_weights.reshape(-1) > 0).float()
    counts = torch.stack([
        (F.one_hot(selected, logits.shape[-1]).sum(1).float() * mask[:, None]).sum(0)
        for logits, selected in observations
    ])
    z_sums = torch.stack([
        (logits.float().logsumexp(-1).square() * mask).sum() for logits, _ in observations
    ])
    count = mask.sum()
    metric_counts = counts.detach().clone()
    if cp_group is not None:
        import torch.distributed as dist
        from torch.distributed.nn.functional import all_reduce

        z_sums = all_reduce(z_sums, group=cp_group)
        count = count.clone()
        dist.all_reduce(count, group=cp_group)
        dist.all_reduce(metric_counts, group=cp_group)
    z_loss = z_sums.sum() / count.clamp_min(1)
    numerator = z_loss * denominator
    top_k = observations[0][1].shape[-1]
    load = metric_counts / (count.clamp_min(1) * top_k)
    return z_coefficient * numerator, {
        "router_z_loss": (numerator.detach(), denominator.detach()),
        "router_max_load": ((load.max() * denominator).detach(), denominator.detach()),
        "router_used_experts": (((metric_counts > 0).sum(-1).float().mean() * denominator).detach(),
                                denominator.detach()),
        "_ling_router_counts": counts.detach(),
    }


@torch.no_grad()
def update_ling_bias(layers, counts, update_rate, zero_mean=True, group=None, apply_update=True):
    """Once per completed optimizer step, from accumulated valid-slot counts."""
    if group is not None:
        import torch.distributed as dist

        dist.all_reduce(counts, group=group)
    offset_sign = (counts.mean(-1, keepdim=True) - counts).sign()
    if zero_mean:
        offset_sign = offset_sign - offset_sign.mean(-1, keepdim=True)
    # Empty/failed optimizer steps do not advance the routing control state.
    active = (counts.sum(-1, keepdim=True) > 0) & torch.as_tensor(apply_update, device=counts.device)
    delta = torch.where(active, offset_sign * update_rate, 0)
    for layer, change in zip(layers, delta, strict=True):
        if layer.gate.bias_enabled:
            layer.gate.expert_bias.add_(change)
