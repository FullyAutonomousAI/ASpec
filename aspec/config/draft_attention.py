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

"""DSpark/DFlash attention policy, shared by preflight and model loading."""

import math
from collections.abc import Mapping


def resolve_draft_head_layout(raw: Mapping) -> dict:
    """Resolve the physical Q/KV head layout used by DFlash/DSpark attention."""

    def positive_integer(name: str, default: int) -> int:
        value = raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Draft {name} must be a positive integer")
        return value

    hidden_size = positive_integer("hidden_size", 4096)
    query_heads = positive_integer("num_attention_heads", 32)
    architectures = raw.get("architectures")
    materialized_mla = raw.get("model_type") == "k3_dspark" or architectures == [
        "K3DSparkModel"
    ] or architectures == ["Eagle3DeepseekV2ForCausalLM"]
    if materialized_mla:
        key_value_heads = positive_integer("num_key_value_heads", query_heads)
        if key_value_heads != query_heads:
            raise ValueError(
                "Materialized MLA requires num_key_value_heads == "
                "num_attention_heads after latent KV expansion"
            )
        qk_nope_head_dim = positive_integer("qk_nope_head_dim", 128)
        qk_rope_head_dim = positive_integer("qk_rope_head_dim", 64)
        value_head_dim = positive_integer("v_head_dim", 128)
        kv_lora_rank = positive_integer("kv_lora_rank", 512)
        q_lora_rank = raw.get("q_lora_rank", 1536)
        if q_lora_rank is not None and (
            isinstance(q_lora_rank, bool)
            or not isinstance(q_lora_rank, int)
            or q_lora_rank <= 0
        ):
            raise ValueError("Draft q_lora_rank must be a positive integer or null")
        if qk_rope_head_dim % 2:
            raise ValueError(
                "Draft qk_rope_head_dim must be even for rotary attention"
            )
        return {
            "type": "mla_materialized",
            "query_heads": query_heads,
            "key_value_heads": key_value_heads,
            "query_heads_per_key_value_head": 1,
            "head_dim": qk_nope_head_dim + qk_rope_head_dim,
            "head_dim_source": "qk_nope_head_dim+qk_rope_head_dim",
            "qk_nope_head_dim": qk_nope_head_dim,
            "qk_rope_head_dim": qk_rope_head_dim,
            "value_head_dim": value_head_dim,
            "q_lora_rank": q_lora_rank,
            "kv_lora_rank": kv_lora_rank,
            "latent_kv_materialized_before_attention": True,
        }
    key_value_heads = positive_integer("num_key_value_heads", 8)
    if query_heads % key_value_heads:
        raise ValueError(
            "Draft num_attention_heads must be divisible by num_key_value_heads; "
            f"got {query_heads} and {key_value_heads}"
        )

    configured_head_dim = raw.get("head_dim")
    if configured_head_dim is None:
        if hidden_size % query_heads:
            raise ValueError(
                "Draft hidden_size must be divisible by num_attention_heads when "
                f"head_dim is omitted; got {hidden_size} and {query_heads}"
            )
        head_dim = hidden_size // query_heads
        head_dim_source = "hidden_size/num_attention_heads"
    else:
        if (
            isinstance(configured_head_dim, bool)
            or not isinstance(configured_head_dim, int)
            or configured_head_dim <= 0
        ):
            raise ValueError("Draft head_dim must be a positive integer or null")
        head_dim = configured_head_dim
        head_dim_source = "explicit"
    if head_dim % 2:
        raise ValueError(f"Draft head_dim must be even for rotary attention, got {head_dim}")

    layout = (
        "mha"
        if query_heads == key_value_heads
        else "mqa"
        if key_value_heads == 1
        else "gqa"
    )
    return {
        "type": layout,
        "query_heads": query_heads,
        "key_value_heads": key_value_heads,
        "query_heads_per_key_value_head": query_heads // key_value_heads,
        "head_dim": head_dim,
        "head_dim_source": head_dim_source,
    }


def resolve_draft_attention(raw: Mapping) -> dict:
    layers = raw.get("num_hidden_layers", 1)
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError("Draft num_hidden_layers must be a positive integer")
    window = raw.get("sliding_window")
    if window is not None and (
        isinstance(window, bool) or not isinstance(window, int) or window <= 0
    ):
        raise ValueError("Draft sliding_window must be a positive integer or null")
    schedule = raw.get("layer_types")
    ratios = raw.get("compress_ratios")
    if ratios is not None:
        if not isinstance(ratios, (list, tuple)) or len(ratios) != layers:
            raise ValueError("Draft compress_ratios must contain one entry per decoder layer")
        if any(isinstance(ratio, bool) or not isinstance(ratio, int) or ratio < 0 for ratio in ratios):
            raise ValueError("Draft compress_ratios must contain non-negative integers")
        ratios = list(ratios)
    legacy = raw.get("use_sliding_window")
    if legacy is not None and not isinstance(legacy, bool):
        raise ValueError("Draft use_sliding_window must be boolean")
    if raw.get("max_window_layers") is not None:
        raise ValueError("Use explicit Draft layer_types instead of max_window_layers")
    if schedule is None:
        if ratios is not None and any(ratios):
            if window is None:
                raise ValueError("CSA Draft requires sliding_window for its local attention branch")
            schedule = [
                "compressed_sparse_attention" if ratio else "sliding_attention"
                for ratio in ratios
            ]
        else:
            active = window is not None and legacy is not False
            schedule = ["sliding_attention" if active else "full_attention"] * layers
    if not isinstance(schedule, (list, tuple)) or len(schedule) != layers:
        raise ValueError("Draft layer_types must contain one entry per decoder layer")
    supported = {
        "full_attention",
        "sliding_attention",
        "compressed_sparse_attention",
        "qwen_sparse_attention",
        "deepseek_sparse_attention",
    }
    if any(kind not in supported for kind in schedule):
        raise ValueError(
            "Draft layer_types supports full_attention, sliding_attention and "
            "compressed_sparse_attention, qwen_sparse_attention and deepseek_sparse_attention only"
        )
    window_active = any(
        kind in {"sliding_attention", "compressed_sparse_attention"}
        for kind in schedule
    )
    csa_active = "compressed_sparse_attention" in schedule
    qsa_active = "qwen_sparse_attention" in schedule
    dsa_active = "deepseek_sparse_attention" in schedule
    if sum((csa_active, qsa_active, dsa_active)) > 1:
        raise ValueError("Draft CSA, QSA and DSA cannot be mixed in one layer schedule")
    if window_active and window is None:
        raise ValueError("Draft sliding/CSA attention requires sliding_window")
    if legacy is not None and legacy != window_active:
        raise ValueError("Draft use_sliding_window conflicts with layer_types/sliding_window")
    if ratios is None:
        ratios = [0] * layers
    for layer_id, (kind, ratio) in enumerate(zip(schedule, ratios, strict=True)):
        if kind == "compressed_sparse_attention" and ratio < 2:
            raise ValueError(
                f"Draft CSA layer {layer_id} requires compress_ratio >= 2"
            )
        if kind != "compressed_sparse_attention" and ratio:
            raise ValueError(
                f"Draft compress_ratios[{layer_id}] is only valid for "
                "compressed_sparse_attention"
            )
        if kind == "compressed_sparse_attention" and ratio != 4:
            raise ValueError(
                f"Draft CSA layer {layer_id} currently requires compress_ratio=4; "
                "the deterministic high-ratio compressor has separate semantics"
            )

    csa = None
    csa_fields = {
        "csa_training_stage",
        "index_topk",
        "index_n_heads",
        "index_head_dim",
        "csa_indexer_loss_coef",
        "csa_workspace_bytes",
        "compress_rope_theta",
    }
    if csa_active:
        stage = raw.get("csa_training_stage", "dense")
        if stage not in {"dense", "indexer_warmup", "sparse"}:
            raise ValueError(
                "Draft csa_training_stage must be dense, indexer_warmup or sparse"
            )

        def positive_integer(name: str, default: int) -> int:
            value = raw.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Draft {name} must be a positive integer")
            return value

        loss_coef = raw.get("csa_indexer_loss_coef", 1.0)
        if (
            isinstance(loss_coef, bool)
            or not isinstance(loss_coef, (int, float))
            or not math.isfinite(loss_coef)
            or loss_coef < 0
        ):
            raise ValueError("Draft csa_indexer_loss_coef must be finite and non-negative")
        if stage == "indexer_warmup" and loss_coef == 0:
            raise ValueError(
                "Draft csa_indexer_loss_coef must be positive during indexer_warmup"
            )
        rope_theta = raw.get("compress_rope_theta", raw.get("rope_theta", 10000.0))
        if (
            isinstance(rope_theta, bool)
            or not isinstance(rope_theta, (int, float))
            or not math.isfinite(rope_theta)
            or rope_theta <= 0
        ):
            raise ValueError("Draft compress_rope_theta must be finite and positive")
        csa = {
            "variant": "csa",
            "training_stage": stage,
            "compress_ratios": ratios,
            "index_topk": positive_integer("index_topk", 64),
            "index_n_heads": positive_integer("index_n_heads", 8),
            "index_head_dim": positive_integer("index_head_dim", 64),
            "indexer_loss_coef": float(loss_coef),
            "workspace_bytes": positive_integer(
                "csa_workspace_bytes", 128 * 1024 * 1024
            ),
            "compress_rope_theta": float(rope_theta),
            "indexer_input_detached": True,
            "cross_layer_reuse": False,
        }
    elif any(raw.get(field) is not None for field in csa_fields):
        raise ValueError("CSA settings require a compressed_sparse_attention Draft layer")

    qsa = None
    qsa_fields = {
        "qsa_training_stage",
        "qsa_token_budget",
        "qsa_compress_ratio",
        "qsa_index_n_heads",
        "qsa_index_head_dim",
        "qsa_rope_dim",
        "qsa_rope_theta",
        "qsa_indexer_loss_coef",
        "qsa_workspace_bytes",
    }
    if qsa_active:
        stage = raw.get("qsa_training_stage", "dense")
        if stage not in {"dense", "indexer_warmup", "sparse"}:
            raise ValueError(
                "Draft qsa_training_stage must be dense, indexer_warmup or sparse"
            )

        def qsa_positive_integer(name: str, default: int) -> int:
            value = raw.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Draft {name} must be a positive integer")
            return value

        compress_ratio = qsa_positive_integer("qsa_compress_ratio", 4)
        if compress_ratio < 2:
            raise ValueError("Draft qsa_compress_ratio must be at least two")
        token_budget = qsa_positive_integer("qsa_token_budget", 2048)
        if token_budget % compress_ratio:
            raise ValueError(
                "Draft qsa_token_budget must be divisible by qsa_compress_ratio"
            )
        index_head_dim = qsa_positive_integer("qsa_index_head_dim", 128)
        rope_dim = qsa_positive_integer("qsa_rope_dim", 64)
        if rope_dim > index_head_dim or rope_dim % 2:
            raise ValueError(
                "Draft qsa_rope_dim must be even and no larger than qsa_index_head_dim"
            )
        rope_theta = raw.get("qsa_rope_theta", raw.get("rope_theta", 10000.0))
        if (
            isinstance(rope_theta, bool)
            or not isinstance(rope_theta, (int, float))
            or not math.isfinite(rope_theta)
            or rope_theta <= 0
        ):
            raise ValueError("Draft qsa_rope_theta must be finite and positive")
        loss_coef = raw.get("qsa_indexer_loss_coef", 1.0)
        if (
            isinstance(loss_coef, bool)
            or not isinstance(loss_coef, (int, float))
            or not math.isfinite(loss_coef)
            or loss_coef < 0
        ):
            raise ValueError("Draft qsa_indexer_loss_coef must be finite and non-negative")
        if stage == "indexer_warmup" and loss_coef == 0:
            raise ValueError(
                "Draft qsa_indexer_loss_coef must be positive during indexer_warmup"
            )
        qsa = {
            "variant": "qsa",
            "training_stage": stage,
            "token_budget": token_budget,
            "compress_ratio": compress_ratio,
            "block_topk": token_budget // compress_ratio,
            "index_n_heads": qsa_positive_integer("qsa_index_n_heads", 4),
            "index_head_dim": index_head_dim,
            "rope_dim": rope_dim,
            "rope_theta": float(rope_theta),
            "indexer_loss_coef": float(loss_coef),
            "workspace_bytes": qsa_positive_integer(
                "qsa_workspace_bytes", 128 * 1024 * 1024
            ),
            "indexer_input_detached": False,
            "cross_layer_reuse": False,
            "draft_block_index_reuse": True,
            "sparse_core_backend": "flash_attn_varlen_gather",
        }
    elif any(raw.get(field) is not None for field in qsa_fields):
        raise ValueError("QSA settings require a qwen_sparse_attention Draft layer")
    dsa = None
    dsa_fields = {
        "dsa_training_stage", "dsa_token_budget", "dsa_index_n_heads",
        "dsa_index_head_dim", "dsa_q_lora_rank", "dsa_rope_dim", "dsa_rope_theta",
        "dsa_indexer_loss_coef", "dsa_workspace_bytes",
    }
    if dsa_active:
        def dsa_positive_integer(name: str, default: int) -> int:
            value = raw.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Draft {name} must be a positive integer")
            return value

        stage = raw.get("dsa_training_stage", "dense")
        if stage not in {"dense", "indexer_warmup", "sparse"}:
            raise ValueError("Draft dsa_training_stage must be dense, indexer_warmup or sparse")
        head_dim = dsa_positive_integer("dsa_index_head_dim", 128)
        rope_dim = dsa_positive_integer("dsa_rope_dim", min(64, head_dim))
        if rope_dim % 2 or rope_dim > head_dim:
            raise ValueError("Draft dsa_rope_dim must be even and no larger than dsa_index_head_dim")
        theta = raw.get("dsa_rope_theta", raw.get("rope_theta", 10000.0))
        coefficient = raw.get("dsa_indexer_loss_coef", 1.0)
        for name, value, positive in (("dsa_rope_theta", theta, True),
                                      ("dsa_indexer_loss_coef", coefficient, stage == "indexer_warmup")):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or (positive and value == 0)):
                raise ValueError(f"Draft {name} must be finite and {'positive' if positive else 'non-negative'}")
        dsa = {
            "variant": "dsa", "training_stage": stage,
            "token_budget": dsa_positive_integer("dsa_token_budget", 2048),
            "index_n_heads": dsa_positive_integer("dsa_index_n_heads", 8),
            "index_head_dim": head_dim,
            "q_lora_rank": dsa_positive_integer("dsa_q_lora_rank", 128),
            "rope_dim": rope_dim, "rope_theta": float(theta),
            "indexer_loss_coef": float(coefficient),
            "workspace_bytes": dsa_positive_integer("dsa_workspace_bytes", 128 * 1024 * 1024),
            "indexer_input_detached": True, "cross_layer_reuse": False,
            "draft_block_index_reuse": False,
            "sparse_core_backend": "flash_attn_varlen_gather",
        }
    elif any(raw.get(field) is not None for field in dsa_fields):
        raise ValueError("DSA settings require a deepseek_sparse_attention Draft layer")
    return {
        "sliding_window": window,
        "layer_types": list(schedule),
        "compress_ratios": ratios,
        "csa": csa,
        "qsa": qsa,
        **({"dsa": dsa} if dsa is not None else {}),
        # DSpark's anchor-shared context window, NOT a moving causal query window.
        "context_windows": [
            window
            if kind in {"sliding_attention", "compressed_sparse_attention"}
            else None
            for kind in schedule
        ],
        "window_semantics": (
            "causal_ttt" if raw.get("architectures") in (["LlamaForCausalLMEagle3"], ["LlamaForCausalLMEagle4"], ["Eagle3DeepseekV2ForCausalLM"])
            else "query_relative" if raw.get("architectures") == ["DFlash2DraftModel"]
            else "anchor_context"
        ),
        "draft_block_attention": (
            "causal_native_mtp" if raw.get("architectures") == ["NativeMTPDraftModel"]
            else "causal_ttt_cache" if raw.get("architectures") in (["LlamaForCausalLMEagle3"], ["LlamaForCausalLMEagle4"], ["Eagle3DeepseekV2ForCausalLM"])
            else [
                "causal" if (raw.get("is_causal") if raw.get("is_causal") is not None
                             else kind == "sliding_attention") else "bidirectional"
                for kind in schedule
            ] if raw.get("architectures") == ["DFlash2DraftModel"]
            else "bidirectional"
        ),
    }


def draft_context_shard_alignment(attention: Mapping) -> int:
    """Derive the minimum context-shard alignment from active sparse structures."""

    ratios: list[int] = []
    csa = attention.get("csa")
    if isinstance(csa, Mapping):
        ratios.extend(int(value) for value in csa["compress_ratios"] if value)
    qsa = attention.get("qsa")
    if isinstance(qsa, Mapping):
        ratios.append(int(qsa["compress_ratio"]))
    return math.lcm(*ratios) if ratios else 1


def draft_thd_contract(
    attention: Mapping,
    *,
    backend: str = "fa4",
    frozen_backbone: bool = False,
) -> dict:
    """Report the actual fixed/selected-varlen kernels, including mixed layers."""
    frozen_backbone = frozen_backbone or any(
        (attention.get(kind) or {}).get("training_stage") == "indexer_warmup"
        for kind in ("csa", "qsa", "dsa")
    )
    modes = sorted({
        "varlen_selected" if (
            layer == "compressed_sparse_attention" and (attention.get("csa") or {}).get("training_stage") == "sparse"
            or layer == "qwen_sparse_attention" and (attention.get("qsa") or {}).get("training_stage") == "sparse"
            or layer == "deepseek_sparse_attention" and (attention.get("dsa") or {}).get("training_stage") == "sparse"
        ) else "fixed"
        for layer in attention["layer_types"]
    })
    if backend == "cudnn_flex_attention":
        if "varlen_selected" in modes:
            raise ValueError(
                "cuDNN THD does not implement CSA/QSA/DSA selected-entry attention"
            )
        backward = [
            "not_required_frozen_backbone"
            if frozen_backbone
            else "cudnn_flex_varlen"
        ]
        return {
            "qkv_format": "thd",
            "kernel_backend": backend,
            "kernel_mode": "varlen_interval",
            "backward_strategy": backward[0],
        }
    backward = (["not_required_frozen_backbone"] if frozen_backbone else
                sorted({"fa4_varlen_selected" if mode == "varlen_selected" else "fa4_fixed_sparse" for mode in modes}))
    return {
        "qkv_format": "thd", "kernel_backend": "fa4",
        "kernel_mode": modes[0] if len(modes) == 1 else "mixed",
        "fa4_mode": modes[0] if len(modes) == 1 else "mixed",
        "backward_strategy": backward[0] if len(backward) == 1 else "mixed",
        **({"fa4_modes": modes} if len(modes) > 1 else {}),
        **({"backward_strategies": backward} if len(backward) > 1 else {}),
    }
