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


"""Strict native assets and a block-proposal component, not a serving runtime."""

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from aspec.config.draft_algorithm import draft_model_semantics, resolve_draft_algorithm
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_native_asset(path, *, expected_config=None, weights_only=False):
    """Check bytes and effective semantics before allocating/loading a model."""
    path = Path(path)
    manifest = json.loads((path / "export_manifest.json").read_text())
    if manifest.get("storage_format") != "aspec_native":
        raise ValueError("Expected an explicit ASpec native artifact, not SGLang weights")
    for name, field in (("config.json", "config_sha256"), ("model.safetensors", "model_sha256")):
        if _sha256(path / name) != manifest.get(field):
            raise ValueError("Native artifact checksum mismatch: " + name)
    raw = json.loads((path / "config.json").read_text())
    config = AutoDraftModelConfig.from_dict(raw)
    if resolve_draft_algorithm(config.to_dict()) != manifest.get("algorithm"):
        raise ValueError("Native artifact algorithm identity mismatch")
    if expected_config is not None and draft_model_semantics(config) != draft_model_semantics(expected_config):
        raise ValueError("Native artifact semantics disagree with the requested Draft")
    from aspec.config.draft_attention import resolve_draft_attention

    attention = resolve_draft_attention(config.to_dict())
    recorded_dsa = ((manifest.get("draft_structure") or {}).get("attention") or {}).get("dsa")
    if recorded_dsa != attention.get("dsa"):
        raise ValueError("Native artifact DSA policy disagrees with its model configuration")
    if expected_config is not None:
        expected_attention = resolve_draft_attention(
            expected_config if isinstance(expected_config, dict) else expected_config.to_dict()
        )
        if attention.get("dsa") is not None or expected_attention.get("dsa") is not None:
            def dsa_semantics(policy, *, phase_transition):
                dsa = policy.get("dsa")
                return (policy["layer_types"], None if dsa is None else {
                    key: value for key, value in dsa.items()
                    if key != "workspace_bytes" and not (
                        phase_transition and key == "training_stage"
                    )
                })
            if dsa_semantics(attention, phase_transition=weights_only) != dsa_semantics(
                expected_attention, phase_transition=weights_only
            ):
                raise ValueError("Native artifact DSA stage/semantics disagree with the requested Draft")
    return config, manifest


def validate_eagle4_native_state(config, state, manifest):
    from aspec.config.eagle4 import eagle4_asset_metadata

    steps = state.get("candidate_head_steps")
    if steps is None or steps.shape != () or steps.dtype != torch.long:
        raise ValueError("EAGLE4 native asset has invalid candidate step state")
    if manifest.get("eagle4") != eagle4_asset_metadata(config.to_dict(), int(steps)):
        raise ValueError("EAGLE4 native asset stage or candidate training state disagrees with manifest")


def load_native_draft(path, *, device="cpu", dtype=None):
    """Load all native weights strictly; no dropped indexer/compressor/head tensors."""
    path = Path(path)
    config, manifest = validate_native_asset(path)
    state = load_file(str(path / "model.safetensors"), device="cpu")
    if len(state) != manifest.get("tensor_count"):
        raise ValueError("Native artifact tensor count mismatch")
    checkpoint_dtype = state["embed_tokens.weight"].dtype
    model = AutoEagle3DraftModel.from_config(config, torch_dtype=dtype or checkpoint_dtype)
    # Establish mixed precision BEFORE load; casting FP32 compressors through
    # BF16 and back would silently destroy the checkpoint's retained precision.
    if hasattr(model, "keep_csa_projection_precision"):
        model.keep_csa_projection_precision()
    if dtype is not None:
        expected = model.state_dict()
        state = {key: value.to(expected[key].dtype) if value.is_floating_point() else value
                 for key, value in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    model = model.to(device=device).eval()
    if manifest.get("algorithm") == "eagle4":
        validate_eagle4_native_state(config, model.state_dict(), manifest)
    return model


class BlockDraftConsumer:
    """One DSpark/DFlash block from verifier features and its frozen output head.

    The caller owns the verifier, its KV/feature cache, verification and acceptance.
    This component makes no SGLang compatibility or end-to-end speed claim.
    """

    def __init__(self, draft_model, *, block_size):
        from aspec.models.dflash import DFlashModel

        self.model = draft_model.eval()
        self.algorithm = resolve_draft_algorithm(draft_model.config.to_dict())
        if self.algorithm not in {"dspark", "dflash"}:
            raise ValueError("BlockDraftConsumer requires a DSpark or DFlash asset")
        if type(block_size) is not int or block_size < 2:
            raise ValueError("block_size must be an integer >= 2")
        saved = getattr(draft_model.config, "block_size", None)
        if saved is not None and saved != block_size:
            raise ValueError("Proposal block size disagrees with the trained asset")
        self.block_size = block_size
        self.backbone = DFlashModel(draft_model, block_size=block_size, num_anchors=1,
                                   attention_backend="fa4")
        # Explicit inference component: keep the actual stage and selection, but
        # never project teacher distributions or compute training-only indexer KL.
        from aspec.models.draft.csa import CSALayer, CSALightningIndexer
        from aspec.models.draft.dsa import DSALayer, DSALightningIndexer
        from aspec.models.draft.qsa import QSAIndexer, QSALayer

        for module in draft_model.modules():
            if isinstance(module, (CSALayer, CSALightningIndexer, QSALayer, QSAIndexer,
                                   DSALayer, DSALightningIndexer)):
                module.compute_auxiliary_loss = False

    @torch.no_grad()
    def propose(self, input_ids, target_features, *, lm_head_weight):
        """IDs/features describe one known prefix, including its last anchor token."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("Block proposal currently consumes one nonempty prefix")
        length = input_ids.shape[1]
        config = self.model.config
        width = config.target_hidden_size * config.num_target_layers
        if target_features.shape != (1, length, width):
            raise ValueError("Target features must match the asset's ordered layers and prefix")
        if lm_head_weight.shape != (config.vocab_size, config.hidden_size):
            raise ValueError("Frozen Target output head shape does not match the Draft")
        if input_ids.device != target_features.device or input_ids.device != lm_head_weight.device:
            raise ValueError("Proposal IDs, features and output head must share a device")
        if length + self.block_size - 1 > config.max_position_embeddings:
            raise ValueError("Proposal exceeds the artifact's supported position range")
        # Pad only cheap token IDs for the training wrapper's anchor bounds.
        # Future context features are neither constructed nor copied.
        anchor = torch.tensor([[length - 1]], device=input_ids.device)
        hidden = self.backbone._draft_backbone(
            F.pad(input_ids, (0, self.block_size)), target_features,
            torch.ones((1, length + self.block_size), device=input_ids.device),
            packed_context_positions=torch.arange(length, device=input_ids.device)[None],
            injected_anchors=anchor, injected_keep_mask=torch.ones_like(anchor, dtype=torch.bool),
        )[0][0]
        if self.algorithm == "dflash":
            return {"token_ids": F.linear(hidden[1:], lm_head_weight).argmax(-1)[None],
                    "algorithm": self.algorithm}
        logits = F.linear(hidden, lm_head_weight)
        previous = input_ids[:, -1]
        tokens, confidence = [], []
        markov = self.model.markov_head
        for position in range(self.block_size):
            row = logits[position:position + 1]
            if markov is not None:
                row = row + markov.compute_step_bias(previous)
            head = self.model.confidence_head
            if head is not None:
                features = hidden[position:position + 1]
                if self.model.confidence_head_with_markov:
                    features = torch.cat((features, markov.get_prev_embeddings(previous).to(features.dtype)), -1)
                confidence.append(head(features))
            previous = row.argmax(-1)
            tokens.append(previous)
        result = {"token_ids": torch.stack(tokens, dim=1), "algorithm": self.algorithm}
        if confidence:
            result["draft_confidence_logits"] = torch.stack(confidence, dim=1)
        return result


class MLAEagleConsumer:
    """Greedy recurrent MLA proposals with an owner-local materialized KV cache.

    Features at t align with next_input_ids at t+1, as in EAGLE3 training.
    The last next_input_id is already sampled/accepted by the verifier.
    """

    def __init__(self, draft_model):
        if draft_model.config.architectures != ["Eagle3DeepseekV2ForCausalLM"]:
            raise ValueError("MLAEagleConsumer requires the retained DeepSeek MLA EAGLE asset")
        self.model = draft_model.eval()

    @torch.no_grad()
    def propose(self, next_input_ids, target_features, *, num_tokens=3):
        model, layer = self.model, self.model.midlayer
        attention = layer.self_attn
        if next_input_ids.ndim != 2 or next_input_ids.shape[0] != 1 or next_input_ids.shape[1] == 0:
            raise ValueError("MLA EAGLE proposal requires one aligned nonempty prefix")
        length = next_input_ids.shape[1]
        if target_features.shape != (1, length, model.fc.in_features):
            raise ValueError("MLA EAGLE features must match the asset's ordered layers")
        if type(num_tokens) is not int or num_tokens < 1 or length + num_tokens - 1 > model.config.max_position_embeddings:
            raise ValueError("Invalid MLA EAGLE proposal length")
        if next_input_ids.device != target_features.device:
            raise ValueError("MLA EAGLE IDs and features must share a device")
        if model.vocab_size != model.target_vocab_size:
            from aspec.models.vocab import validate_vocab_mapping

            validate_vocab_mapping(model.d2t, model.t2d, model.vocab_size, model.target_vocab_size)
        hidden = model.project_hidden_states(target_features)
        embeddings = model.embed_input_ids(next_input_ids)
        keys = values = None
        tokens, logits_out = [], []
        for depth in range(num_tokens):
            combined = torch.cat((layer.input_layernorm(embeddings), layer.hidden_norm(hidden)), -1)
            query, nope, rope, value = attention._project_qkv(combined)
            start = 0 if depth == 0 else length + depth - 1
            positions = torch.arange(start, start + combined.shape[1], device=combined.device)[None]
            query, key = attention._apply_rope_and_assemble(
                query, nope, rope, positions, 0, rotary_length=length + depth,
            )
            if keys is None:
                keys = key.new_empty(1, key.shape[1], length + num_tokens - 1, key.shape[-1])
                values = value.new_empty(1, value.shape[1], length + num_tokens - 1, value.shape[-1])
            end = start + key.shape[2]
            keys[:, :, start:end].copy_(key)
            values[:, :, start:end].copy_(value)
            # Only the last query is needed. No context-by-context attention
            # matrix or repeated full-prefix decoder/MLP execution.
            attended = F.scaled_dot_product_attention(
                query[:, :, -1:], keys[:, :, :end], values[:, :, :end], dropout_p=0.0,
                is_causal=False, scale=attention.softmax_scale,
            ).transpose(1, 2).flatten(2)
            output = hidden[:, -1:] + attention.o_proj(attended)
            output = output + layer.mlp(layer.post_attention_layernorm(output))
            logits = model.compute_logits(output)[:, 0]
            row = logits.argmax(-1)
            if model.vocab_size != model.target_vocab_size:
                row = row + model.d2t[row]
            tokens.append(row)
            logits_out.append(logits)
            hidden = model.norm(output) if model.norm_output else output
            embeddings = model.embed_input_ids(row[:, None])
        return {"token_ids": torch.stack(tokens, 1), "logits": torch.stack(logits_out, 1),
                "algorithm": "eagle3"}
