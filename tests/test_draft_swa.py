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

"""Independent anchor-window reference: boundaries, packing, sparse tiles and contracts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from aspec.config.draft_attention import resolve_draft_attention
from aspec.models.dflash import DFlashModel, _create_dflash_mask_mod
from aspec.models.draft.dflash import DFlashConfig
from aspec.models.ops.cudnn_flex_attention import build_dspark_mla_mask
from aspec.models.ops.direct_cp import DirectCPMetadata, _visibility, direct_cp_attention
from aspec.models.ops.fa4_sparse import _direct_cp_tile_masks
from aspec.training.checkpoint import _logical_training_topology, _validate_exact_resume_topology
from tools.convert_to_hf import _serving_config


def _expected(anchors, keep, length, block, window, docs=None):
    # Independent scalar definition, matching the official DSpark ring:
    # W previous context tokens at the anchor + the entire current draft block.
    result = torch.zeros(1, anchors.numel() * block, length + anchors.numel() * block, dtype=torch.bool)
    for q in range(result.shape[1]):
        a, owner = int(anchors[0, q // block]), q // block
        if not keep[0, owner]:
            continue
        for k in range(result.shape[2]):
            if k < length:
                same_doc = docs is None or docs[0, a] == docs[0, k] >= 0
                result[0, q, k] = same_doc and max(0, a - window) <= k < a
            else:
                result[0, q, k] = (k - length) // block == owner
    return result


@pytest.mark.parametrize("window", [1, 3, 8, 100])
@pytest.mark.parametrize("packed", [False, True])
def test_anchor_window_mask_and_cudnn_endpoints_match_official_ring_semantics(window, packed):
    length, block = 19, 5
    anchors = torch.tensor([[0, 3, 7, 12, 18]])
    keep = torch.tensor([[True, True, False, True, True]])
    docs = torch.tensor([[0] * 9 + [1] * 10]) if packed else None
    expected = _expected(anchors, keep, length, block, window, docs)
    mask = _create_dflash_mask_mod(anchors, keep, length, block, context_window=window, ctx_doc_ids=docs)
    endpoint = build_dspark_mla_mask(anchor_positions=anchors, block_keep_mask=keep,
                                    context_length=length, block_size=block,
                                    context_window=window, ctx_doc_ids=docs).mask_func
    b = torch.tensor(0)
    for q in range(block * anchors.numel()):
        endpoints = endpoint[0, :, q].tolist()
        for k in range(expected.shape[-1]):
            visible = k < endpoints[0] or any(endpoints[i] <= k < endpoints[i+1]
                                             for i in range(1, len(endpoints), 2))
            assert visible == bool(expected[0, q, k])
            assert bool(mask(b, b, torch.tensor(q), torch.tensor(k))) == visible
    # Offsets do NOT shrink the context window or truncate the noncausal draft block.
    assert torch.equal(expected[:, 5], expected[:, 9])


@pytest.mark.parametrize("change", [
    {"sliding_window": 0}, {"sliding_window": True}, {"sliding_window": 1.5},
    {"sliding_window": -1}, {"layer_types": ["unknown"]},
    {"num_hidden_layers": 2, "layer_types": ["full_attention"]},
    {"layer_types": ["sliding_attention"]},
    {"sliding_window": 3, "layer_types": ["sliding_attention"], "use_sliding_window": False},
    {"sliding_window": 3, "max_window_layers": 1},
])
def test_invalid_attention_configuration_fails_closed(change):
    with pytest.raises(ValueError):
        DFlashConfig(**change)


def test_default_full_and_mixed_policy_survive_config_roundtrip():
    full = DFlashConfig(num_hidden_layers=3)
    assert resolve_draft_attention(full.to_dict())["context_windows"] == [None] * 3
    config = DFlashConfig(num_hidden_layers=3, sliding_window=7,
                         layer_types=["full_attention", "sliding_attention", "full_attention"])
    restored = DFlashConfig.from_dict(config.to_dict())
    assert resolve_draft_attention(restored.to_dict())["context_windows"] == [None, 7, None]
    wrapper = object.__new__(DFlashModel)
    object.__setattr__(wrapper, "draft_model", SimpleNamespace(config=restored))
    assert [wrapper._block_mask_options_for_layer(i) for i in range(3)] == [
        {}, {"context_window": 7}, {},
    ]


@pytest.mark.parametrize("rank,size", [(0, 1), (0, 2), (1, 2), (0, 4), (3, 4)])
def test_window_uses_global_context_positions_across_cp_owners(rank, size):
    from aspec.models.ops.direct_cp import build_context_positions

    anchors = torch.tensor([[1, 5, 10, 15, 18, 22, 26, 29]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    positions = build_context_positions(32, rank, size)[None]
    meta = DirectCPMetadata(anchors, keep, positions, 2, rank, size,
                            object() if size > 1 else None, context_window=5)
    actual = _visibility(meta, torch.device("cpu"))
    query_blocks = meta.collective_block_ids.repeat_interleave(2)
    for q, b in enumerate(query_blocks.tolist()):
        assert torch.equal(actual[0, q, :positions.numel()],
                           (positions[0] < anchors[0, b]) & (positions[0] >= anchors[0, b] - 5))


@pytest.mark.parametrize("window", [1, 3, 8, 100])
def test_swa_math_output_and_gradients_match_independent_reference(window):
    torch.manual_seed(61)
    anchors, keep = torch.tensor([[2, 6, 10, 14]]), torch.ones(1, 4, dtype=torch.bool)
    docs = torch.tensor([[0] * 8 + [1] * 8])
    meta = DirectCPMetadata(anchors, keep, torch.arange(16)[None], 2, 0, 1, None,
                            context_doc_ids=docs, anchor_doc_ids=docs.gather(1, anchors),
                            context_window=window, backend="math")
    shapes = [(1, 4, 8, 8), (1, 2, 16, 8), (1, 2, 16, 8),
              (1, 2, 8, 8), (1, 2, 8, 8)]
    actual_inputs = [torch.randn(s, dtype=torch.float64, requires_grad=True) for s in shapes]
    ref = [t.detach().clone().requires_grad_() for t in actual_inputs]
    actual, _ = direct_cp_attention(*actual_inputs, meta)
    q, ck, cv, dk, dv = ref
    k, v = [torch.cat(pair, dim=2).repeat_interleave(2, dim=1) for pair in ((ck, dk), (cv, dv))]
    visible = _expected(anchors, keep, 16, 2, window, docs)
    scores = (q @ k.transpose(-1, -2)) / 8**0.5
    expected = scores.masked_fill(~visible[:, None], -1e30).softmax(-1) @ v
    torch.testing.assert_close(actual, expected)
    dout = torch.randn_like(actual)
    (actual * dout).sum().backward()
    (expected * dout).sum().backward()
    for x, y in zip(actual_inputs, ref):
        torch.testing.assert_close(x.grad, y.grad)


@pytest.mark.parametrize("window", [1, 3, 11])
def test_swa_tile_cover_never_omits_visible_or_marks_masked_as_full(window):
    anchors = torch.tensor([[0, 5, 9, 17]])
    keep = torch.tensor([[True, True, False, True]])
    block, length, tm, tn = 3, 24, 4, 5
    qb = torch.arange(12)[None] // block
    qkeep = keep.repeat_interleave(block, 1)
    kkeep = torch.cat((torch.ones(1, length, dtype=torch.bool), qkeep), 1)
    partial, full = _direct_cp_tile_masks(
        anchor_positions=anchors, query_blocks=qb, context_positions=torch.arange(length)[None],
        draft_blocks=qb, query_token_keep=qkeep, key_token_keep=kkeep, tile_m=tm, tile_n=tn,
        context_window=window,
    )
    reference = _expected(anchors, keep, length, block, window)
    for m in range(partial.shape[2]):
        for n in range(partial.shape[3]):
            tile = reference[0, m*tm:(m+1)*tm, n*tn:(n+1)*tn]
            if tile.any():
                assert partial[0, 0, m, n] or full[0, 0, m, n]
            if full[0, 0, m, n]:
                assert tile.all()
    # Policy keys must not alias a full-attention layer's sparse traversal.
    original = DirectCPMetadata(anchors, keep, torch.arange(length)[None], block, 0, 1, None)
    swa = replace(original, context_window=window)
    assert swa._fa4_cache is original._fa4_cache


def test_exact_resume_checks_window_schedule_and_legacy_full_attention():
    def args(window):
        return SimpleNamespace(world_size=1, draft_model_config_obj=DFlashConfig(sliding_window=window))
    metadata = {"world_size": 1, "training_topology": _logical_training_topology(args(4), 1)}
    _validate_exact_resume_topology(args(4), metadata)
    with pytest.raises(RuntimeError, match="topology"):
        _validate_exact_resume_topology(args(5), metadata)
    with pytest.raises(RuntimeError, match="legacy"):
        _validate_exact_resume_topology(args(4), {"world_size": 1})
    _validate_exact_resume_topology(args(None), {"world_size": 1})


def test_swa_serving_export_preserves_window_instead_of_becoming_full_attention():
    exported = _serving_config({"sliding_window": 128})
    assert exported["sliding_window"] == 128


def test_public_full_swa_recipe_has_shared_target_and_explicit_plan(monkeypatch):
    from pathlib import Path

    from aspec.config.train_config import load_config
    from aspec.multi_draft import compile_multi_draft_plan

    root = Path(__file__).parents[1]
    monkeypatch.chdir(root)
    plan = compile_multi_draft_plan(load_config(
        str(root / "configs/examples/ling3_flash_swa_compare_dspark.yaml"), save_snapshot=False,
    ))
    assert set(plan.plans) == {"full", "swa"}
    full, swa = plan.plans["full"], plan.plans["swa"]
    assert full.workload.features == swa.workload.features
    assert full.workload.models["draft"]["attention"]["context_windows"] == [None]*5
    assert swa.workload.models["draft"]["attention"]["context_windows"] == [128]*5
    assert full.to_runtime_args().learning_rate == swa.to_runtime_args().learning_rate


def test_public_128k_swa_packing_cp4_recipe_is_one_validated_combination(monkeypatch):
    from pathlib import Path

    from aspec.config.train_config import load_config
    from aspec.planning import compile_execution_plan

    root = Path(__file__).parents[1]
    monkeypatch.chdir(root)
    plan = compile_execution_plan(load_config(
        str(root / "configs/examples/ling3_flash_swa_128k_packing_cp4.yaml"),
        save_snapshot=False,
    ))
    draft = plan.workload.models["draft"]
    assert draft["structure"]["ffn_type"] == "dense"
    assert draft["attention"]["head_layout"]["type"] == "mha"
    assert draft["attention"]["context_windows"] == [128] * 5
    args = plan.to_runtime_args()
    assert args.attention_backend == "fa4"
    assert args.max_seq_length == 131072
    assert args.draft_context_parallel_size == 4
    assert args.training_num_gpus_per_node == 4
    assert args.dflash_packing and args.packing_min_samples_per_dispatch == 2
    assert not args.target_prefill_memory_optimized
    assert args.mooncake_source_gpu_ring_slots == 8
    assert args.inference_num_gpus == args.inference_num_gpus_per_engine == 4


@pytest.mark.parametrize("ffn_type", ["dense", "qwen3_moe", "ling_moe"])
def test_mixed_schedule_uses_per_layer_masks_without_changing_ffn(ffn_type, monkeypatch):
    from aspec.models.draft.dflash import DFlashDraftModel

    extra = {}
    if ffn_type != "dense":
        extra = {"ffn_type": ffn_type, "num_experts": 4, "num_experts_per_tok": 2,
                 "moe_intermediate_size": 12}
    if ffn_type == "ling_moe":
        extra.update(num_shared_experts=1, moe_shared_expert_intermediate_size=12,
                     n_group=1, topk_group=1)
    config = DFlashConfig(hidden_size=16, intermediate_size=32, vocab_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         num_target_layers=1, target_hidden_size=16,
                         max_position_embeddings=64, sliding_window=3,
                         layer_types=["full_attention", "sliding_attention"], **extra)
    model = DFlashDraftModel(config)
    masks, observed = [object(), object()], []
    for layer in model.layers:
        def attention(*, draft_hidden, block_mask, **_):
            observed.append(block_mask)
            return draft_hidden
        monkeypatch.setattr(layer.self_attn, "forward", attention)
    values = dict(draft_input_ids=None, context_feature=torch.randn(1, 8, 16),
                  draft_position_ids=torch.tensor([[2, 3, 5, 6]]),
                  context_position_ids=torch.arange(8)[None],
                  noise_embedding=torch.randn(1, 4, 16))
    output = model(**values, block_mask=masks)
    output.square().sum().backward()
    assert observed == masks
    assert all(layer.mlp is not None for layer in model.layers)
    with pytest.raises(ValueError, match="effective attention mask"):
        model(**values)


def test_invalid_saved_topology_still_reports_a_resume_error():
    with pytest.raises(RuntimeError, match="invalid training_topology"):
        _validate_exact_resume_topology(SimpleNamespace(world_size=1),
                                        {"world_size": 1, "training_topology": 7})


@pytest.mark.parametrize("schedule", [
    ["full_attention"]*2, ["sliding_attention"]*2, ["full_attention", "sliding_attention"],
])
def test_uniform_fa4_policy_does_not_copy_metadata_per_layer(schedule, monkeypatch):
    from aspec.models.ops import direct_cp
    from aspec.utils import distributed

    monkeypatch.setattr(distributed, "get_direct_cp_group", lambda: None)
    config = DFlashConfig(
        hidden_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        sliding_window=3,
        layer_types=schedule,
    )
    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config, self.num_layers, self.mask_token_id = config, 2, 7
            self.embed_tokens = torch.nn.Embedding(8, 4)

        def extract_context_feature(self, values):
            return values[0]

        def forward(self, *, noise_embedding, direct_cp_metadata, **_):
            self.observed = direct_cp_metadata
            return noise_embedding
    draft = Draft()
    wrapper = DFlashModel(draft, block_size=2, num_anchors=4, attention_backend="fa4")
    monkeypatch.setattr(wrapper, "_sample_anchor_positions",
                        lambda *a, **k: (torch.tensor([[1, 4, 8, 12]]), torch.ones(1, 4, dtype=torch.bool)))
    calls = []
    original = direct_cp.DirectCPMetadata.__post_init__
    def validate(metadata):
        calls.append(metadata)
        original(metadata)
    monkeypatch.setattr(direct_cp.DirectCPMetadata, "__post_init__", validate)
    wrapper._draft_backbone(torch.ones(1, 16, dtype=torch.long), [torch.randn(1, 16, 4)],
                            torch.ones(1, 16))
    if schedule[0] == schedule[1]:
        assert len(calls) == 1 and draft.observed is calls[0]
    else:
        assert len(calls) == 2
        assert draft.observed[0]._fa4_cache is draft.observed[1]._fa4_cache
