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

import copy

import pytest
import torch
import torch.nn.functional as F

from aspec.models.draft.dflash import DFlashConfig, DFlashMLP
from aspec.models.draft.dspark import DSparkConfig


def config(cls=DFlashConfig, **overrides):
    return cls(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
               num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
               num_target_layers=2, target_hidden_size=32, target_num_hidden_layers=12,
               target_layer_ids=[1, 9], mask_token_id=63, max_position_embeddings=128,
               rope_theta=10000.0, **overrides)


@pytest.mark.parametrize("cls", [DFlashConfig, DSparkConfig])
@pytest.mark.parametrize("activation,reference", [("silu", F.silu), ("gelu", F.gelu), ("relu", F.relu)])
def test_dense_activation_matches_reference_and_gradients(cls, activation, reference):
    torch.manual_seed(7)
    model = DFlashMLP(config(cls, hidden_act=activation))
    other = copy.deepcopy(model)
    x = torch.randn(2, 3, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    actual = model(x)
    expected = other.down_proj(reference(other.gate_proj(y)) * other.up_proj(y))
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    for p, q in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_unknown_dense_activation_is_rejected_before_allocation():
    with pytest.raises(ValueError, match="hidden_act"):
        config(hidden_act="unknown_activation")

@pytest.mark.parametrize("cls", [DFlashConfig, DSparkConfig])
@pytest.mark.parametrize("field", ["rope_scaling", "rope_parameters"])
@pytest.mark.parametrize("kind", ["linear", "yarn"])
def test_scaled_gqa_rope_matches_transformers_and_survives_dtype_cast(cls, field, kind):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    from aspec.models.draft.dflash import DFlashAttention
    params = {"rope_type": kind, "factor": 4.0, "rope_theta": 10000.0,
              "original_max_position_embeddings": 128}
    cfg = config(cls, **{field: params})
    rotary = DFlashAttention(cfg).rotary_emb.to(torch.bfloat16)
    reference = Qwen3RotaryEmbedding(Qwen3Config(hidden_size=32, num_attention_heads=4, head_dim=8,
                                               max_position_embeddings=128, rope_parameters=params))
    x = torch.ones(1, 1, 257, 8, dtype=torch.bfloat16)
    cos, sin = rotary(x, seq_len=257)
    expected_cos, expected_sin = reference(x, torch.arange(257).unsqueeze(0))
    torch.testing.assert_close(cos[:, 0], expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(sin[:, 0], expected_sin, rtol=0, atol=0)
    ordinary = DFlashAttention(config(cls)).rotary_emb
    base_cos, _ = ordinary(x, seq_len=257)
    assert not torch.equal(cos, base_cos)


def test_gqa_unknown_rope_type_is_rejected_at_config_boundary():
    with pytest.raises(ValueError, match="RoPE type"):
        config(rope_parameters={"rope_type":"dynamic", "factor":4.0})

@pytest.mark.parametrize("cls", [DFlashConfig, DSparkConfig])
@pytest.mark.parametrize("moe", [False, True])
def test_checkpointing_recomputes_layers_and_preserves_gradients(cls, moe, monkeypatch):
    from aspec.models.draft.dflash import DFlashAttention, DFlashDraftModel
    from aspec.models.draft.dspark import DSparkDraftModel
    # Dense attention reference keeps this contract test independent of CUDA kernels.
    def attention(self, draft_hidden, context_hidden, **kwargs):
        return self.o_proj(self.q_proj(draft_hidden)) + context_hidden.mean(1, keepdim=True)
    monkeypatch.setattr(DFlashAttention, "forward", attention)
    kwargs = dict(ffn_type="qwen3_moe", num_experts=4, num_experts_per_tok=2,
                  moe_intermediate_size=16) if moe else {}
    model_cls = DSparkDraftModel if cls is DSparkConfig else DFlashDraftModel
    torch.manual_seed(9)
    ordinary = model_cls(config(cls, **kwargs))
    ordinary.freeze_embedding()
    recomputed = copy.deepcopy(ordinary)
    recomputed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    counts = [0, 0]
    for i, layer in enumerate(recomputed.layers):
        def count(module, inputs, index=i):
            counts[index] += 1
        layer.mlp.register_forward_pre_hook(count)
    results = []
    saved_sizes = []
    for model in (ordinary, recomputed):
        sizes=[]
        def pack(tensor):
            sizes.append(tensor.numel()*tensor.element_size())
            return tensor
        context = torch.randn(1, 5, 32, generator=torch.Generator().manual_seed(8), requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor:tensor):
            output = model(torch.tensor([[1,2,3]]), context, torch.arange(3).unsqueeze(0),
                           torch.arange(5).unsqueeze(0), return_router_logits=moe)
            hidden, logits = output if moe else (output, ())
            loss = hidden.square().sum() + sum(value.square().mean() for value in logits)
        saved_sizes.append(sum(sizes))
        loss.backward()
        results.append((hidden, context.grad))
    assert counts == [2, 2]
    assert saved_sizes[1] < saved_sizes[0]
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected)
    for (name,p),(_,q) in zip(ordinary.named_parameters(),recomputed.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,msg=name)
    recomputed.eval()
    counts[:] = [0,0]
    with torch.no_grad():
        recomputed(torch.tensor([[1,2,3]]), context.detach(), torch.arange(3).unsqueeze(0), torch.arange(5).unsqueeze(0))
    assert counts == [1,1]
