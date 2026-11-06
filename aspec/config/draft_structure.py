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

"""Draft FFN semantics shared by config loading and the preflight Plan."""

import math
from collections.abc import Mapping


def resolve_draft_structure(raw: Mapping) -> dict:
    if raw.get("architectures") == ["LlamaForCausalLMEagle4"]:
        from aspec.config.eagle4 import eagle4_structure

        return eagle4_structure(raw)
    ffn_type = raw.get("ffn_type", "dense")
    if ffn_type not in {"dense", "qwen3_moe", "ling_moe"}:
        raise ValueError(f"Unsupported Draft ffn_type={ffn_type!r}")
    if ffn_type != "ling_moe" and any(raw.get(key) for key in (
        "router_z_loss_coef", "moe_router_bias_update_rate", "bias_zero_mean_update",
        "expert_swiglu_limit_list", "share_expert_swiglu_limit_list",
    )):
        raise ValueError("Ling training settings require explicit ffn_type=ling_moe")
    experts = raw.get("num_experts", 0)
    if ffn_type == "dense":
        if experts != 0 or any(raw.get(key) not in (None, 0) for key in (
            "num_experts_per_tok", "moe_intermediate_size", "router_aux_loss_coef",
            "shared_expert_intermediate_size", "moe_shared_expert_intermediate_size",
            "num_shared_experts", "n_shared_experts", "moe_router_enable_expert_bias",
        )):
            raise ValueError("MoE settings require explicit ffn_type=qwen3_moe or ling_moe")
        from transformers.activations import ACT2FN

        if raw.get("hidden_act", "silu") not in ACT2FN:
            raise ValueError(f"Unsupported Dense hidden_act={raw.get('hidden_act')!r}")
        return {"ffn_type": "dense"}
    values = {
        "num_experts": experts,
        "num_experts_per_tok": raw.get("num_experts_per_tok"),
        "moe_intermediate_size": raw.get("moe_intermediate_size"),
    }
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"MoE Draft {key} must be a positive integer")
    if values["num_experts_per_tok"] > experts:
        raise ValueError("MoE Draft num_experts_per_tok exceeds num_experts")
    norm = raw.get("norm_topk_prob", True)
    if not isinstance(norm, bool):
        raise ValueError("MoE Draft norm_topk_prob must be boolean")
    alpha = raw.get("router_aux_loss_coef")
    if alpha is None:
        alpha = 0.0 if ffn_type == "ling_moe" else 0.001
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("MoE Draft router_aux_loss_coef must be finite and non-negative")
    if ffn_type == "ling_moe":
        from aspec.config.ling_structure import resolve_ling_structure

        return resolve_ling_structure(raw, {"ffn_type": ffn_type, **values,
                                           "norm_topk_prob": norm, "router_aux_loss_coef": float(alpha)})
    # These describe different algorithms, not harmless Qwen3 aliases.
    for key in ("shared_expert_intermediate_size", "moe_shared_expert_intermediate_size",
                "num_shared_experts", "n_shared_experts", "n_group", "topk_group"):
        if raw.get(key, 0):
            raise ValueError(f"qwen3_moe does not implement {key}; use a matching router implementation")
    if raw.get("mlp_only_layers") or raw.get("decoder_sparse_step", 1) != 1:
        raise ValueError("qwen3_moe Draft currently uses MoE in every decoder layer")
    if raw.get("hidden_act", "silu") != "silu":
        raise ValueError("qwen3_moe Draft requires SwiGLU hidden_act=silu")
    if raw.get("topk_method", "greedy") != "greedy":
        raise ValueError("qwen3_moe Draft implements ungrouped greedy Top-K routing")
    for key in ("score_function", "scoring_func"):
        if raw.get(key, "softmax") != "softmax":
            raise ValueError(f"qwen3_moe requires {key}=softmax")
    if raw.get("moe_router_enable_expert_bias", False) or raw.get("routed_scaling_factor", 1.0) != 1.0:
        raise ValueError("qwen3_moe does not implement expert bias or routed scaling")
    return {"ffn_type": ffn_type, **values, "norm_topk_prob": norm,
            "router_aux_loss_coef": float(alpha)}

def checkpoint_ffn_policy(checkpoint_dir, topology):
    """Read authoritative routing semantics, including pre-policy checkpoints."""
    recorded = topology.get("draft_ffn_policy")
    if recorded is not None:
        return recorded
    if checkpoint_dir is not None:
        import json
        from pathlib import Path

        saved = Path(checkpoint_dir) / "draft_config.json"
        if saved.is_file():
            return resolve_draft_structure(json.loads(saved.read_text()))
    return None
