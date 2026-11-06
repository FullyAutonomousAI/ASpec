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

"""Ling-specific FFN boundary validation; shared Plan/model semantics."""

import math


def resolve_ling_structure(raw, common):
    def positive_int(key, default=None):
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"ling_moe {key} must be a positive integer")
        return value

    def nonnegative(key, default):
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"ling_moe {key} must be finite and non-negative")
        return float(value)

    experts, top_k = common["num_experts"], common["num_experts_per_tok"]
    groups = positive_int("n_group", 1)
    group_topk = positive_int("topk_group", groups)
    if experts % groups or experts // groups < 2:
        raise ValueError("ling_moe group scoring requires at least two experts per equal-sized group")
    if group_topk > groups or top_k > group_topk * (experts // groups):
        raise ValueError("ling_moe Top-K exceeds the selected groups' expert capacity")
    if not common["norm_topk_prob"]:
        raise ValueError("ling_moe uses official normalized routing (Top-1 retains its sigmoid score)")
    if common["router_aux_loss_coef"] != 0:
        raise ValueError("ling_moe uses bias balancing, not Qwen router_aux_loss_coef")
    for key, expected in (("topk_method", "noaux_tc"), ("score_function", "sigmoid"),
                          ("scoring_func", "sigmoid"), ("router_dtype", "fp32")):
        if raw.get(key, expected) != expected:
            raise ValueError(f"ling_moe requires {key}={expected}")
    if raw.get("mlp_only_layers") or raw.get("decoder_sparse_step", 1) != 1 or raw.get("first_k_dense_replace", 0):
        raise ValueError("ling_moe Draft currently uses MoE in every decoder layer")
    if raw.get("hidden_act", "silu") != "silu" or raw.get("scale_router_input", False) or raw.get("up_proj_norm", False):
        raise ValueError("ling_moe requires unscaled router input and SwiGLU without up-projection normalization")
    for alias in ("shared_expert_intermediate_size", "n_shared_experts"):
        if raw.get(alias):
            raise ValueError(f"ling_moe uses canonical shared expert fields, not {alias}")
    shared = raw.get("num_shared_experts", 1)
    if isinstance(shared, bool) or not isinstance(shared, int) or shared < 0:
        raise ValueError("ling_moe num_shared_experts must be a non-negative integer")
    width = positive_int("moe_shared_expert_intermediate_size") if shared else 0
    if not shared and raw.get("moe_shared_expert_intermediate_size", 0) not in (None, 0):
        raise ValueError("ling_moe shared expert width requires num_shared_experts > 0")
    enabled = raw.get("moe_router_enable_expert_bias", True)
    zero_mean = raw.get("bias_zero_mean_update", True)
    if not isinstance(enabled, bool) or not isinstance(zero_mean, bool):
        raise ValueError("ling_moe bias flags must be boolean")
    rate = nonnegative("moe_router_bias_update_rate", 0.001 if enabled else 0.0)
    if not enabled and rate:
        raise ValueError("ling_moe bias update rate requires enabled expert bias")
    scale = nonnegative("routed_scaling_factor", 2.5)
    if scale == 0:
        raise ValueError("ling_moe routed_scaling_factor must be positive")
    limits = {}
    for key in ("expert_swiglu_limit_list", "share_expert_swiglu_limit_list"):
        values = raw.get(key)
        if values is None:
            values = [0.0] * positive_int("num_hidden_layers")
        if not isinstance(values, (list, tuple)) or len(values) != raw.get("num_hidden_layers"):
            raise ValueError(f"ling_moe {key} must have one limit per Draft layer")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"ling_moe {key} limits must be finite and non-negative (zero disables clamp)")
        limits[key] = [float(v) for v in values]
    return {**common, "num_shared_experts": shared, "moe_shared_expert_intermediate_size": width,
            "n_group": groups, "topk_group": group_topk, "routed_scaling_factor": scale,
            "router_dtype": "fp32", "score_function": "sigmoid", "scoring_func": "sigmoid",
            "topk_method": "noaux_tc", "moe_router_enable_expert_bias": enabled,
            "moe_router_bias_update_rate": rate, "bias_zero_mean_update": zero_mean,
            "router_z_loss_coef": nonnegative("router_z_loss_coef", 0.0000035), **limits}
