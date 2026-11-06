# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted config class from TorchSpec a4e5c10 auto.py; ASpec policy validation added.
# ASpec modifications:
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
"""EAGLE4 semantics, independent of attention execution and serving."""
import math
from collections.abc import Mapping

from transformers import LlamaConfig, PretrainedConfig


def eagle4_policy(raw: Mapping) -> dict:
    if raw.get("num_hidden_layers", 1) != 1:
        raise ValueError("EAGLE4 recurrent Draft requires one decoder layer")
    stage = raw.get("eagle4_training_stage", "draft")
    if stage not in {"draft", "candidate_head"}:
        raise ValueError("EAGLE4 stage must be draft or candidate_head")
    if (raw.get("draft_vocab_size") or raw["vocab_size"]) != raw["vocab_size"]:
        raise ValueError("EAGLE4 candidate retrieval requires the full target vocabulary")
    if raw.get("lm_head_rank_divisor"):
        raise ValueError("EAGLE4 needs a dense exact head, not lm_head_rank_divisor")
    head = raw.get("efficient_lm_head_config")
    if not isinstance(head, Mapping) or head.get("type") != "low_rank":
        raise ValueError("EAGLE4 requires efficient_lm_head_config.type=low_rank")
    rank, candidates = head.get("rank"), head.get("num_candidates")
    teacher_k = raw.get("candidate_teacher_top_k", candidates)
    chunk = raw.get("candidate_loss_chunk_size", 128)
    for name, value, limit in (
        ("rank", rank, min(raw["hidden_size"], raw["vocab_size"])),
        ("num_candidates", candidates, raw["vocab_size"]),
        ("candidate_teacher_top_k", teacher_k, raw["vocab_size"]),
        ("candidate_loss_chunk_size", chunk, None),
    ):
        if type(value) is not int or value <= 0 or (limit is not None and value > limit):
            raise ValueError(f"EAGLE4 {name} must be a positive integer within its tensor extent")
    for name in ("fc_norm", "norm_output", "attn_norm", "enable_post_mlp_norm", "enable_moe", "use_rope"):
        if name in raw and not isinstance(raw[name], bool):
            raise ValueError(f"EAGLE4 {name} must be boolean")
    for name in ("enable_2hs", "enable_defa", "bias", "attention_bias", "mlp_bias"):
        if raw.get(name):
            raise ValueError(f"EAGLE4 does not implement {name}")
    if raw.get("rope_is_neox_style", True) is False and raw.get("use_rope", True):
        raise ValueError("EAGLE4 interleaved RoPE requires a matching attention implementation")
    return {
        "stage": stage, "rank": rank, "num_candidates": candidates,
        "teacher_top_k": teacher_k, "loss_chunk_size": chunk,
        "candidate_objective": "frozen_draft_topk_forward_kl",
        "initialization": "randomized_truncated_svd",
    }


def eagle4_training_policy(raw: Mapping) -> dict:
    """Resume identity excludes a workspace bound, not any retrieval semantics."""
    policy = eagle4_policy(raw)
    policy.pop("loss_chunk_size")
    return policy


def eagle4_structure(raw: Mapping) -> dict:
    if not raw.get("enable_moe", False):
        if raw.get("ffn_type", "dense") != "dense" or any(raw.get(k, 0) for k in ("n_routed_experts", "n_shared_experts", "num_experts_per_tok", "moe_intermediate_size")):
            raise ValueError("EAGLE4 MoE settings require enable_moe=true")
        return {"ffn_type": "dense"}
    if raw.get("ffn_type", "eagle4_moe") not in {"dense", "eagle4_moe"}:
        raise ValueError("EAGLE4 uses its explicit upstream router, not another FFN policy")
    architecture = raw.get("moe_architecture", "sparse_swiglu")
    if architecture not in {"sparse_swiglu", "nemotron_latent_relu2"}:
        raise ValueError(f"Unsupported EAGLE4 MoE architecture: {architecture!r}")
    n, k = raw.get("n_routed_experts"), raw.get("num_experts_per_tok")
    group, selected = raw.get("n_group", 1) or 1, raw.get("topk_group", raw.get("n_group", 1)) or 1
    for name, value in {
        "n_routed_experts": n, "num_experts_per_tok": k,
        "moe_intermediate_size": raw.get("moe_intermediate_size"),
        "n_group": group, "topk_group": selected,
    }.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"EAGLE4 {name} must be positive")
    if k > n or n % group or selected > group or k > n // group * selected or (group > 1 and n // group < 2):
        raise ValueError("EAGLE4 MoE group/top-k geometry is invalid")
    shared = raw.get("n_shared_experts", 0) or 0
    if type(shared) is not int or shared < 0:
        raise ValueError("EAGLE4 n_shared_experts must be non-negative")
    if raw.get("scoring_func", "sigmoid") != "sigmoid" or raw.get("topk_method", "noaux_tc") != "noaux_tc":
        raise ValueError("EAGLE4 MoE requires sigmoid/noaux_tc routing")
    if type(raw.get("norm_topk_prob", True)) is not bool:
        raise ValueError("EAGLE4 norm_topk_prob must be boolean")
    if raw.get("hidden_act", "silu") != ("relu2" if architecture == "nemotron_latent_relu2" else "silu"):
        raise ValueError("EAGLE4 MoE activation disagrees with architecture")
    rate = raw.get("expert_bias_update_rate", raw.get("moe_router_bias_update_rate", 0.001))
    scale = raw.get("routed_scaling_factor", 1.0)
    for name, value in (("expert_bias_update_rate", rate), ("routed_scaling_factor", scale)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"EAGLE4 {name} must be finite and non-negative")
    result = {
        "ffn_type": "eagle4_moe", "moe_architecture": architecture,
        "n_routed_experts": n, "num_experts_per_tok": k,
        "moe_intermediate_size": raw["moe_intermediate_size"],
        "n_shared_experts": shared, "n_group": group, "topk_group": selected,
        "norm_topk_prob": raw.get("norm_topk_prob", True), "routed_scaling_factor": float(scale),
        "expert_bias_update_rate": float(rate),
    }
    if architecture == "nemotron_latent_relu2":
        for name in ("moe_latent_size", "moe_shared_expert_intermediate_size"):
            if type(raw.get(name)) is not int or raw[name] <= 0:
                raise ValueError(f"EAGLE4 {name} must be positive")
            result[name] = raw[name]
    for key in ("aux_loss_alpha", "seq_aux_loss_alpha", "load_balance_loss_alpha", "moe_seq_aux_loss_coeff"):
        if raw.get(key, 0):
            raise ValueError("EAGLE4 implements auxiliary-loss-free routing, not nonzero " + key)
    return result


def eagle4_asset_metadata(raw, steps):
    policy = eagle4_policy(raw)
    candidate = policy["stage"] == "candidate_head"
    if type(steps) is not int or steps < 0 or candidate != (steps > 0):
        raise ValueError("EAGLE4 candidate asset requires completed training; draft assets require zero candidate steps")
    return {**policy, "candidate_optimizer_steps": steps, "candidate_head_trained": candidate,
            "scope": "trained_candidate" if candidate else "draft_initialization"}


class Eagle4Config(LlamaConfig):
    """Explicit-head-dimension Llama config; adapted from TorchSpec a4e5c10 auto.py."""

    def validate_architecture(self):
        PretrainedConfig.validate_architecture(self)

    def __init__(self, **kwargs):
        kwargs.setdefault("eagle4_training_stage", "draft")
        kwargs.setdefault("pretraining_tp", 1)
        if kwargs.get("efficient_lm_head_config"):
            kwargs.setdefault("candidate_teacher_top_k", kwargs["efficient_lm_head_config"].get("num_candidates"))
        kwargs.setdefault("candidate_loss_chunk_size", 128)
        kwargs.setdefault("use_rope", not (
            kwargs.get("position_embedding_type") == "none" or kwargs.get("partial_rotary_factor") == 0
        ))
        if kwargs.get("architectures") == ["LlamaForCausalLMEagle4"] and "hidden_size" in kwargs and "vocab_size" in kwargs:
            eagle4_policy(kwargs)
            eagle4_structure(kwargs)
        super().__init__(**kwargs)
        if self.architectures == ["LlamaForCausalLMEagle4"]:
            eagle4_policy(self.to_dict())
            eagle4_structure(self.to_dict())
