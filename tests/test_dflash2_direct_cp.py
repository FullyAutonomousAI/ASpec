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

"""DFlash2 query-relative visibility, distinct from DSpark anchor windows."""

from dataclasses import replace

import pytest
import torch

from aspec.models.dflash import _create_dflash_mask_mod
from aspec.models.ops.direct_cp import DirectCPMetadata, _visibility
from aspec.models.ops.fa4_sparse import _direct_cp_tile_masks


@pytest.mark.parametrize("window", [None, 1, 3, 8])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_visibility_matches_retained_flex_contract(window, causal, packed):
    context_length, block = 24, 4
    anchors = torch.tensor([[4, 8, 16, 20]])
    keep = torch.tensor([[True, False, True, True]])
    docs = torch.tensor([[0] * 12 + [1] * 12]) if packed else None
    metadata = DirectCPMetadata(
        anchor_positions=anchors, block_keep_mask=keep,
        context_position_ids=torch.arange(context_length)[None], block_size=block,
        cp_rank=0, cp_size=1, group=None, sequence_length=context_length,
        query_window=window, draft_is_causal=causal,
        context_doc_ids=docs, anchor_doc_ids=docs.gather(1, anchors) if packed else None,
        backend="math",
    )
    mask = _create_dflash_mask_mod(anchors, keep, context_length, block, is_causal=causal,
                                  sliding_window=window, ctx_doc_ids=docs)
    queries = torch.arange(16)
    keys = torch.arange(40)
    reference = mask(torch.tensor(0), torch.tensor(0), queries[:, None], keys[None, :])
    # Direct execution neutralizes invalid query/key slots, as the FA4 path does.
    key_keep = torch.cat((torch.ones(24, dtype=torch.bool), keep[0].repeat_interleave(block)))
    reference &= key_keep[None]
    observed = _visibility(metadata, "cpu")
    assert torch.equal(observed[0], reference)
    query_blocks = torch.arange(4).repeat_interleave(block)[None].int()
    partial, full = _direct_cp_tile_masks(
        anchor_positions=anchors.int(), query_blocks=query_blocks,
        context_positions=metadata.context_position_ids.int(), draft_blocks=query_blocks,
        query_token_keep=keep.repeat_interleave(block, dim=1),
        key_token_keep=key_keep[None], tile_m=4, tile_n=4,
        anchor_doc_ids=metadata.anchor_doc_ids.int() if packed else None,
        context_doc_ids=docs.int() if packed else None,
        query_window=window, draft_block_size=block,
    )
    tiles = observed.reshape(1, 4, 4, 10, 4).permute(0, 1, 3, 2, 4)
    assert (tiles.any((-1, -2))[:, None] <= (partial | full)).all()
    assert (full <= tiles.all((-1, -2))[:, None]).all()


def test_cache_identity_separates_window_and_causality_without_copying_aux():
    metadata = DirectCPMetadata(torch.tensor([[4, 12]]), torch.ones(1, 2, dtype=torch.bool),
                                torch.arange(16)[None], 4, 0, 1, None)
    variant = replace(metadata, query_window=2, draft_is_causal=True)
    assert metadata._fa4_cache is variant._fa4_cache
    assert not torch.equal(_visibility(metadata, "cpu"), _visibility(variant, "cpu"))


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_invalid_query_window_fails_closed(value):
    with pytest.raises(ValueError, match="query_window"):
        DirectCPMetadata(torch.tensor([[4]]), torch.ones(1, 1, dtype=torch.bool),
                         torch.arange(8)[None], 4, 0, 1, None, query_window=value)


@pytest.mark.parametrize("shape", [(1, 1), (3, 1), (1, 7), (3, 7)])
def test_fixed_aux_singletons_have_unique_unit_stride_without_semantic_padding(shape):
    from aspec.models.ops.fa4_sparse import fixed_mask_aux

    source = torch.arange(shape[0] * shape[1]).reshape(shape)
    actual = fixed_mask_aux(source)
    assert actual.shape == source.shape
    assert actual.dtype == torch.int32
    assert sum(stride == 1 for stride in actual.stride()) == 1
    torch.testing.assert_close(actual, source.int(), atol=0, rtol=0)


@pytest.mark.parametrize("algorithm", ["dspark", "dflash", "dflash2"])
def test_serving_window_and_causality_survive_official_hf_config_loading(algorithm):
    from transformers import Qwen3Config

    from aspec.models.draft.dflash2 import DFlash2Config
    from tools.convert_to_hf import _serving_config

    raw = DFlash2Config(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=128,
        vocab_size=97, mask_token_id=96, num_target_layers=2,
        target_hidden_size=32, target_num_hidden_layers=8, target_layer_ids=[1, 5],
        conv_group_size=4, selector_rank=8, selector_top_k=4,
        layer_types=["full_attention", "sliding_attention"], sliding_window=5,
        is_causal=True,
    ).to_dict()
    if algorithm != "dflash2":
        raw["architectures"] = ["Qwen3DSparkModel" if algorithm == "dspark" else "DFlashDraftModel"]
        raw["model_type"] = "qwen3_dspark" if algorithm == "dspark" else "dflash"
    exported = _serving_config(raw)
    loaded = Qwen3Config(**exported)
    assert loaded.sliding_window == 5
    assert loaded.layer_types == raw["layer_types"]
    assert loaded.is_causal is True
    assert loaded.use_sliding_window is True
    if algorithm == "dflash2":
        reloaded = DFlash2Config.from_dict(loaded.to_dict())
        assert reloaded.sliding_window == 5 and reloaded.is_causal is True
