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

# Adapted from TorchSpec a4e5c1085bfc5990ddcae6cf5832bb26380b38d3
# torchspec/models/draft/llama3_eagle.py. Reuses retained execution; adds explicit
# stage ownership and complete low-rank retrieval state. No Eagle source copied.
"""EAGLE4 Draft backbone and candidate-head assets."""
import torch
from torch import nn

from aspec.config.draft_attention import resolve_draft_attention
from aspec.config.eagle4 import Eagle4Config, eagle4_policy
from aspec.models.draft.base import load_tensor_from_pretrained
from aspec.models.draft.eagle4_moe import Eagle4MoE, split_latent_experts, stack_latent_experts
from aspec.models.draft.keymap import to_export_keys, to_internal_keys
from aspec.models.draft.llama3_eagle import (
    LlamaDecoderLayer,
    LlamaForCausalLMEagle3,
    LlamaMLP,
    LlamaRMSNorm,
)


class Eagle4DecoderLayer(LlamaDecoderLayer):
    @staticmethod
    def build_mlp(config):
        return Eagle4MoE(config) if getattr(config, "enable_moe", False) else LlamaMLP(config)

    def __init__(self, config, attention_backend="sdpa"):
        super().__init__(config, attention_backend)
        if getattr(config, "attn_norm", False):
            self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
            self.pre_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if getattr(config, "enable_post_mlp_norm", False):
            self.post_mlp_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_emb, hidden_states, cache_keys=None, cache_values=None,
                attention_mask=None, position_ids=None, use_cache=False):
        fused = torch.cat((self.input_layernorm(input_emb), self.hidden_norm(hidden_states)), -1)
        residual = hidden_states
        if hasattr(self, "eh_proj"):
            residual = self.eh_proj(fused)
            fused = self.pre_attention_layernorm(residual)
        attended, keys, values = self.self_attn(
            hidden_states=fused, cache_keys=cache_keys, cache_values=cache_values,
            attention_mask=attention_mask, position_ids=position_ids, use_cache=use_cache,
        )
        hidden = residual + attended
        return hidden + self.mlp(self.post_attention_layernorm(hidden)), keys, values


class Eagle4DraftModel(LlamaForCausalLMEagle3):
    config_class = Eagle4Config
    decoder_layer_class = Eagle4DecoderLayer

    def __init__(self, config, attention_backend="sdpa"):
        super().__init__(config, attention_backend)
        self.eagle4_policy = eagle4_policy(config.to_dict())
        self.sliding_window = resolve_draft_attention(config.to_dict())["context_windows"][0]
        self.midlayer.self_attn.eagle4_window = self.sliding_window
        rank = self.eagle4_policy["rank"]
        self.efficient_lm_head_down_proj = nn.Linear(config.hidden_size, rank, bias=False)
        self.efficient_lm_head = nn.Linear(rank, self.target_vocab_size, bias=False)
        self.register_buffer("candidate_head_steps", torch.zeros((), dtype=torch.long))

    def next_depth_hidden(self, hidden_states):
        if self.norm_output:
            return self.norm(hidden_states)
        norm = getattr(self.midlayer, "post_mlp_layernorm", None)
        return hidden_states if norm is None else norm(hidden_states)

    @property
    def candidate_stage(self):
        return self.eagle4_policy["stage"] == "candidate_head"

    @property
    def preserve_loaded_embedding(self):
        return self.candidate_stage

    def configure_training_stage(self):
        for name, parameter in self.named_parameters():
            candidate = name.startswith(("efficient_lm_head.", "efficient_lm_head_down_proj."))
            parameter.requires_grad_(candidate if self.candidate_stage else not candidate)
        self.freeze_embedding()
        # Upstream gives norm_output precedence; the unused fallback norm must
        # not enter DDP/optimizer ownership as a trainable-but-unreachable leaf.
        if self.norm_output and hasattr(self.midlayer, "post_mlp_layernorm"):
            self.midlayer.post_mlp_layernorm.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "eagle4_policy") and self.candidate_stage:
            for name, module in self.named_children():
                if name not in {"efficient_lm_head", "efficient_lm_head_down_proj"}:
                    module.eval()
        return self

    def validate_target_binding(self, target_path, *, embedding_key, lm_head_key=None):
        if self.candidate_stage:
            expected = load_tensor_from_pretrained(target_path, embedding_key)
            actual = self.embed_tokens.weight.detach()
            if actual.shape != expected.shape or not torch.equal(actual.cpu(), expected.to(actual.dtype).cpu()):
                raise ValueError("Candidate training requires the pretrained Draft's same frozen Target embedding")

    @torch.no_grad()
    def initialize_candidate_head(self, *, device=None):
        if int(self.candidate_head_steps) != 0:
            raise ValueError("Refusing to overwrite a trained candidate head with SVD")
        # Randomized truncated SVD bounds the one-time workspace by V*(R+16),
        # rather than a full V*H left-singular matrix. No change on exact resume.
        weight = self.lm_head.weight.detach().to(device=device, dtype=torch.float32)
        rank = self.eagle4_policy["rank"]
        u, s, v = torch.svd_lowrank(weight, q=min(rank + 16, *weight.shape), niter=2)
        self.efficient_lm_head_down_proj.weight.copy_(v[:, :rank].T)
        self.efficient_lm_head.weight.copy_(u[:, :rank] * s[:rank])

    def candidate_scores(self, hidden_states):
        return self.efficient_lm_head(self.efficient_lm_head_down_proj(hidden_states))

    @torch.no_grad()
    def score_candidates(self, hidden_states):
        """Exact Draft logits on a dynamic full-vocabulary shortlist; not serving."""
        count = self.eagle4_policy["num_candidates"]
        ids = self.candidate_scores(hidden_states).topk(count, -1).indices
        logits = torch.matmul(self.lm_head.weight[ids], hidden_states.unsqueeze(-1)).squeeze(-1)
        return ids, logits

    def backbone(self, input_embeds, hidden_states, attention_mask, position_ids,
                 cache_keys=None, cache_values=None, use_cache=True):
        mlp = self.midlayer.mlp
        if isinstance(mlp, Eagle4MoE):
            from aspec.models.ops.eagle_thd import EagleTHDLayout
            if isinstance(attention_mask, EagleTHDLayout):
                depth = 0 if cache_keys is None else len(cache_keys)
                mlp.query_mask = attention_mask.shifted(depth + 1)[1]
            else:
                depth = 0 if cache_keys is None else cache_keys.shape[2]
                valid = (torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool)
                         if attention_mask is None else attention_mask.bool() if attention_mask.ndim == 2 else
                         attention_mask[:, 0].diagonal(dim1=-2, dim2=-1) >= 0)
                lengths = valid.sum(-1)
                mlp.query_mask = torch.arange(hidden_states.shape[1], device=hidden_states.device)[None] + depth + 1 < lengths[:, None]
        try:
            return super().backbone(input_embeds, hidden_states, attention_mask, position_ids,
                                    cache_keys, cache_values, use_cache)
        finally:
            if isinstance(mlp, Eagle4MoE):
                mlp.query_mask = None

    def commit_router_load(self):
        mlp = self.midlayer.mlp
        if self.training and not self.candidate_stage and isinstance(mlp, Eagle4MoE):
            mlp.gate.load.add_(mlp.last_load)

    def prepare_initial_native_state(self, state, saved_config, *, manifest):
        from aspec.config.draft_algorithm import draft_model_semantics
        from aspec.draft_artifact import validate_eagle4_native_state

        validate_eagle4_native_state(saved_config, state, manifest)
        from aspec.config.draft_structure import resolve_draft_structure
        def body(config):
            result = draft_model_semantics(config)
            for key in ("efficient_lm_head_config", "candidate_teacher_top_k"):
                result.pop(key, None)
            return result
        if body(saved_config) != body(self.config) or resolve_draft_structure(saved_config.to_dict()) != resolve_draft_structure(self.config.to_dict()):
            raise ValueError("EAGLE4 weights-only initialization changes the pretrained Draft body")
        expected = self.state_dict()
        keys = ("efficient_lm_head.weight", "efficient_lm_head_down_proj.weight")
        if any(state[k].shape != expected[k].shape for k in keys) or getattr(saved_config, "eagle4_training_stage", "draft") == "draft":
            state = dict(state)
            state.update({k: expected[k] for k in keys})
            state["candidate_head_steps"] = expected["candidate_head_steps"].new_zeros(())
        return state

    def state_dict_from_serving(self, tensors):
        return to_internal_keys(stack_latent_experts(tensors), self.state_dict().keys())

    def state_dict_for_serving(self):
        return split_latent_experts(to_export_keys(self.state_dict()))
