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

"""Algorithm identity for the currently registered Draft architectures."""

from collections.abc import Mapping

DRAFT_ALGORITHMS = {
    "NativeMTPDraftModel": "mtp",
    "DFlashDraftModel": "dflash",
    "DFlash2DraftModel": "dflash2",
    "DSparkDraftModel": "dspark",
    "Qwen3DSparkModel": "dspark",
    "K3DSparkModel": "dspark",
    "LlamaForCausalLMEagle3": "eagle3",
    "LlamaForCausalLMEagle4": "eagle4",
    "Eagle3DeepseekV2ForCausalLM": "eagle3",
}


def resolve_draft_algorithm(raw: Mapping) -> str:
    architectures = raw.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError("Draft algorithm requires exactly one registered architecture")
    try:
        return DRAFT_ALGORITHMS[architectures[0]]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unsupported Draft algorithm architecture: {architectures!r}") from exc


def draft_model_semantics(config) -> dict:
    """Checkpoint the effective model, not its filesystem or HF bookkeeping.

    Callers pass the resolved config, so omitted defaults and explicit defaults
    have the same meaning. Attention/FFN policies are recorded separately by
    the checkpoint topology.
    """
    raw = config if isinstance(config, Mapping) else config.to_dict()
    if raw.get("architectures") and raw["architectures"][0] in DRAFT_ALGORITHMS:
        from aspec.models.draft.auto import AutoDraftModelConfig

        raw = AutoDraftModelConfig.from_dict(raw).to_dict()
    raw = dict(raw)
    if raw.get("hidden_size") and raw.get("num_attention_heads"):
        from aspec.config.draft_attention import resolve_draft_head_layout

        raw["head_dim"] = resolve_draft_head_layout(raw)["head_dim"]
    raw["fc_norm"] = bool(raw.get("fc_norm"))
    raw["norm_output"] = bool(raw.get("norm_output"))
    raw["lm_head_rank_divisor"] = raw.get("lm_head_rank_divisor") or None
    fields = (
        "native_config", "architectures", "hidden_size", "target_hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
        "vocab_size", "draft_vocab_size", "hidden_act", "rms_norm_eps",
        "max_position_embeddings", "rope_theta", "rope_scaling", "rope_parameters",
        "target_layer_ids", "num_target_layers", "target_num_hidden_layers",
        "num_aux_hidden_states", "fc_norm", "norm_output", "mask_token_id",
        "lm_head_rank_divisor", "attention_mode", "q_lora_rank", "kv_lora_rank",
        "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "block_size",
        "conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k",
        "efficient_lm_head_config", "candidate_teacher_top_k", "use_rope", "attn_norm",
        "enable_post_mlp_norm",
        "input_embedding_scale", "output_multiplier", "final_logit_softcapping", "is_causal",
    )
    semantics = {key: raw[key] for key in fields if key in raw}
    if raw.get("mtp_conditioning") is not None:
        from aspec.config.mtp_conditioning import resolve_mtp_conditioning

        semantics["mtp_conditioning"] = resolve_mtp_conditioning(raw)
    return semantics
